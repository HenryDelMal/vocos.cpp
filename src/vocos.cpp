#include "vocos.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <fstream>
#include <limits>
#include <map>
#include <numbers>
#include <stdexcept>
#include <string>

namespace vocos {
namespace {
using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
struct tensor { std::vector<uint32_t> shape; std::vector<float> data; };
void require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
uint32_t read_u32(std::istream& in) {
    std::array<unsigned char,4> b{};
    in.read(reinterpret_cast<char*>(b.data()),4);
    require(bool(in), "Truncated model header");
    return uint32_t(b[0]) | (uint32_t(b[1])<<8) | (uint32_t(b[2])<<16) | (uint32_t(b[3])<<24);
}
float gelu(float x) { return 0.5f*x*(1.0f+std::erf(x/std::sqrt(2.0f))); }
}
struct codec::impl {
    model_info meta{};
    std::map<std::string,tensor> weights;
    const tensor& get(const std::string& name, std::initializer_list<uint32_t> shape) const {
        auto it=weights.find(name);
        require(it!=weights.end(), "Missing tensor: "+name);
        require(it->second.shape==std::vector<uint32_t>(shape), "Invalid tensor shape: "+name);
        return it->second;
    }
    explicit impl(const std::filesystem::path& path) {
        std::ifstream in(path,std::ios::binary);
        require(bool(in), "Cannot open model");
        char magic[8]{}; in.read(magic,8);
        require(std::string(magic,8)==std::string("VOCOSN2\0",8), "Invalid model magic");
        require(read_u32(in)==2, "Unsupported model version");
        std::array<uint32_t,11> m{}; for(auto& v:m) v=read_u32(in);
        meta={m[0],m[1],m[2],m[3],m[4],m[5],m[6],m[7],m[8],m[9],m[10]};
        require(meta.sample_rate==24000 && meta.channels==1 && meta.hop_length==320 && meta.n_fft==1280,
                "Model must be Vocos 24 kHz mono (FFT 1280, hop 320)");
        require(meta.latent_dim==128 && meta.hidden_dim==384 && meta.intermediate_dim==1152 &&
                meta.layers==8 && meta.codebooks==16 && meta.entries==1024 && meta.bandwidths==4,
                "Model dimensions do not match pretrained Vocos EnCodec 24 kHz");
        auto count=read_u32(in); require(count<2048,"Too many tensors");
        size_t total=0;
        for(uint32_t i=0;i<count;++i) {
            auto length=read_u32(in); require(length>0 && length<=256,"Invalid tensor name");
            std::string name(length,'\0'); in.read(name.data(),length);
            auto rank=read_u32(in); require(rank>0 && rank<=4,"Invalid tensor rank");
            tensor t; size_t n=1;
            for(uint32_t j=0;j<rank;++j) {
                auto d=read_u32(in); require(d>0 && d<=100000 && n<=100000000/d,"Invalid tensor extent");
                t.shape.push_back(d); n*=d;
            }
            total+=n; require(total<=100000000,"Model exceeds float limit");
            t.data.resize(n);
            for(auto& value:t.data) { value=std::bit_cast<float>(read_u32(in)); require(std::isfinite(value),"Nonfinite weight"); }
            require(weights.emplace(name,std::move(t)).second,"Duplicate tensor");
        }
        require(in.peek()==std::char_traits<char>::eof(),"Trailing model bytes");
        // Validate every tensor before any inference allocation.
        get("rvq",{meta.codebooks,meta.entries,meta.latent_dim});
        get("embed.weight",{meta.hidden_dim,meta.latent_dim,7});
        get("embed.bias",{meta.hidden_dim});
        get("norm.scale",{meta.bandwidths,meta.hidden_dim});
        get("norm.shift",{meta.bandwidths,meta.hidden_dim});
        get("final_norm.weight",{meta.hidden_dim}); get("final_norm.bias",{meta.hidden_dim});
        get("head.weight",{meta.n_fft+2,meta.hidden_dim});
        get("head.bias",{meta.n_fft+2});
        for(uint32_t i=0;i<meta.layers;++i) {
            auto p="blocks."+std::to_string(i);
            get(p+".dwconv.weight",{meta.hidden_dim,1,7});
            get(p+".dwconv.bias",{meta.hidden_dim});
            get(p+".norm.scale",{meta.bandwidths,meta.hidden_dim});
            get(p+".norm.shift",{meta.bandwidths,meta.hidden_dim});
            get(p+".pwconv1.weight",{meta.intermediate_dim,meta.hidden_dim});
            get(p+".pwconv1.bias",{meta.intermediate_dim});
            get(p+".pwconv2.weight",{meta.hidden_dim,meta.intermediate_dim});
            get(p+".pwconv2.bias",{meta.hidden_dim});
            get(p+".gamma",{meta.hidden_dim});
        }
    }
    Matrix linear(const Matrix& x,const std::string& p,uint32_t out) const {
        auto& w=get(p+".weight",{out,uint32_t(x.cols())}); auto& b=get(p+".bias",{out});
        Matrix y=x*Eigen::Map<const Matrix>(w.data.data(),out,x.cols()).transpose();
        for(Eigen::Index r=0;r<y.rows();++r) for(uint32_t c=0;c<out;++c) y(r,c)+=b.data[c];
        return y;
    }
    Matrix conv(const Matrix& x,const std::string& p,uint32_t out,uint32_t kernel,uint32_t stride,bool depthwise=false) const {
        uint32_t inputs=depthwise?1:uint32_t(x.cols());
        auto& w=get(p+".weight",{out,inputs,kernel}); auto& b=get(p+".bias",{out});
        size_t frames=(size_t(x.rows())+stride-1)/stride;
        // SAME padding: odd extra padding goes on the right.
        size_t needed=(frames-1)*stride+kernel;
        size_t pad=needed>size_t(x.rows())?(needed-size_t(x.rows()))/2:0;
        Matrix y(frames,out);
        for(size_t t=0;t<frames;++t) for(uint32_t o=0;o<out;++o) {
            float value=b.data[o];
            for(uint32_t c=0;c<inputs;++c) for(uint32_t k=0;k<kernel;++k) {
                auto source=int64_t(t*stride+k)-int64_t(pad);
                if(source>=0 && source<x.rows()) value+=x(source,depthwise?o:c)*w.data[(o*inputs+c)*kernel+k];
            }
            y(t,o)=value;
        }
        return y;
    }
    Matrix norm(Matrix x,const std::string& p) const {
        auto& w=get(p+".weight",{uint32_t(x.cols())}); auto& b=get(p+".bias",{uint32_t(x.cols())});
        for(Eigen::Index r=0;r<x.rows();++r) {
            float mean=x.row(r).mean(), variance=(x.row(r).array()-mean).square().mean();
            float scale=1/std::sqrt(variance+1e-6f);
            for(Eigen::Index c=0;c<x.cols();++c) x(r,c)=(x(r,c)-mean)*scale*w.data[c]+b.data[c];
        }
        return x;
    }
    Matrix ada_norm(Matrix x,const std::string& p,uint32_t bandwidth) const {
        auto& scale=get(p+".scale",{meta.bandwidths,meta.hidden_dim});
        auto& shift=get(p+".shift",{meta.bandwidths,meta.hidden_dim});
        for(Eigen::Index r=0;r<x.rows();++r) {
            float mean=x.row(r).mean(), variance=(x.row(r).array()-mean).square().mean();
            float inv=1/std::sqrt(variance+1e-6f);
            for(uint32_t c=0;c<meta.hidden_dim;++c)
                x(r,c)=(x(r,c)-mean)*inv*scale.data[bandwidth*meta.hidden_dim+c]+shift.data[bandwidth*meta.hidden_dim+c];
        }
        return x;
    }
    Matrix backbone(const Matrix& features,uint32_t bandwidth) const {
        Matrix x=ada_norm(conv(features,"embed",meta.hidden_dim,7,1),"norm",bandwidth);
        for(uint32_t i=0;i<meta.layers;++i) {
            auto p="blocks."+std::to_string(i);
            Matrix y=linear(ada_norm(conv(x,p+".dwconv",meta.hidden_dim,7,1,true),p+".norm",bandwidth),p+".pwconv1",meta.intermediate_dim);
            y=y.unaryExpr([](float v){return gelu(v);});
            y=linear(y,p+".pwconv2",meta.hidden_dim);
            auto& gamma=get(p+".gamma",{meta.hidden_dim});
            for(Eigen::Index t=0;t<x.rows();++t) for(uint32_t c=0;c<meta.hidden_dim;++c) x(t,c)+=y(t,c)*gamma.data[c];
        }
        return norm(x,"final_norm");
    }
    std::vector<float> synthesize(const Matrix& features,uint32_t bandwidth) const {
        Matrix prediction=linear(backbone(features,bandwidth),"head",meta.n_fft+2);
        size_t frames=features.rows(), n=meta.n_fft, hop=meta.hop_length, bins=n/2+1;
        size_t length=(frames-1)*hop+n, pad=(n-hop)/2;
        std::vector<float> result(frames*hop), window(n), envelope(length,0);
        for(size_t j=0;j<n;++j) window[j]=0.5f-0.5f*std::cos(2*std::numbers::pi_v<float>*float(j)/float(n));
        for(size_t t=0;t<frames;++t) for(size_t j=0;j<n;++j) envelope[t*hop+j]+=window[j]*window[j];
        Eigen::FFT<float> fft;
        std::vector<float> audio(length,0), frame;
        std::vector<std::complex<float>> spectrum(n);
        for(size_t t=0;t<frames;++t) {
            for(size_t k=0;k<bins;++k) {
                float magnitude=std::exp(std::min(prediction(t,k),std::log(100.0f)));
                float phase=prediction(t,bins+k);
                spectrum[k]=std::polar(magnitude,phase);
            }
            spectrum[0]={spectrum[0].real(),0}; spectrum[n/2]={spectrum[n/2].real(),0};
            for(size_t k=bins;k<n;++k) spectrum[k]=std::conj(spectrum[n-k]);
            fft.inv(frame,spectrum);
            for(size_t j=0;j<n;++j) audio[t*hop+j]+=frame[j]*window[j];
        }
        for(size_t j=0;j<frames*hop;++j) {
            require(envelope[j+pad]>1e-11f,"Invalid overlap envelope");
            result[j]=audio[j+pad]/envelope[j+pad];
        }
        return result;
    }
};
codec::codec(const std::filesystem::path& path):state(std::make_unique<impl>(path)){}
codec::~codec()=default;
codec::codec(codec&&) noexcept=default;
codec& codec::operator=(codec&&) noexcept=default;
model_info codec::info() const {return state->meta;}
std::vector<float> codec::decode_features(std::span<const float> values,size_t frames,uint32_t bandwidth) const {
    auto dim=state->meta.latent_dim;
    require(frames>0 && frames<=1000000 && values.size()/dim==frames && values.size()%dim==0 && bandwidth<4,"Invalid features/bandwidth");
    for(float v:values) require(std::isfinite(v),"Nonfinite feature");
    return state->synthesize(Eigen::Map<const Matrix>(values.data(),frames,dim),bandwidth);
}
std::vector<float> codec::decode(const tokens& codes,uint32_t bandwidth) const {
    constexpr uint32_t q_for_bandwidth[]{2,4,8,16};
    auto m=info(); require(bandwidth<4 && codes.codebooks==q_for_bandwidth[bandwidth] && codes.frames>0 && codes.frames<=1000000 &&
                          codes.indices.size()==codes.frames*codes.codebooks,"Invalid token dimensions");
    auto& table=state->get("rvq",{m.codebooks,m.entries,m.latent_dim});
    Matrix features=Matrix::Zero(codes.frames,m.latent_dim);
    for(size_t t=0;t<codes.frames;++t) for(uint32_t q=0;q<codes.codebooks;++q) {
        auto index=codes.indices[t*codes.codebooks+q]; require(index<m.entries,"Invalid token index");
        for(uint32_t c=0;c<m.latent_dim;++c) features(t,c)+=table.data[(q*m.entries+index)*m.latent_dim+c];
    }
    return state->synthesize(features,bandwidth);
}
}
