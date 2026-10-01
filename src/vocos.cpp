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
        require(std::string(magic,8)==std::string("VOCOSN1\0",8), "Invalid model magic");
        require(read_u32(in)==1, "Unsupported model version");
        std::array<uint32_t,11> m{}; for(auto& v:m) v=read_u32(in);
        meta={m[0],m[1],m[2],m[3],m[4],m[5],m[6],m[7],m[8],m[9],m[10]};
        require(meta.sample_rate>0 && meta.channels>=1 && meta.channels<=2, "Invalid audio metadata");
        require(meta.n_fft>=4 && meta.n_fft<=8192 && meta.n_fft%2==0 && meta.hop_length>0 &&
                meta.hop_length<meta.n_fft && (meta.n_fft-meta.hop_length)%2==0, "Invalid FFT/hop");
        require(meta.latent_dim>0 && meta.latent_dim<=2048 && meta.hidden_dim>0 && meta.hidden_dim<=2048 &&
                meta.intermediate_dim>0 && meta.intermediate_dim<=8192 && meta.layers<=64 &&
                meta.codebooks>0 && meta.codebooks<=64 && meta.entries>0 && meta.entries<=65536 &&
                meta.encoder_layers<=8, "Invalid model dimensions");
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
        Matrix dummy=Matrix::Zero(1,meta.latent_dim);
        (void)backbone(dummy);
        get("head.weight",{meta.channels*(meta.n_fft+2),meta.hidden_dim});
        get("head.bias",{meta.channels*(meta.n_fft+2)});
        uint32_t input=meta.channels, product=1;
        for(uint32_t i=0;i<meta.encoder_layers;++i) {
            auto p="encoder."+std::to_string(i);
            auto& spec=get(p+".spec",{3});
            require(spec.data[0]>=1 && spec.data[0]<=2048 && spec.data[1]>=1 && spec.data[1]<=128 &&
                    spec.data[2]>=1 && spec.data[2]<=32,"Invalid encoder spec range");
            uint32_t output=uint32_t(spec.data[0]), kernel=uint32_t(spec.data[1]), stride=uint32_t(spec.data[2]);
            require(output>0 && output<=2048 && kernel>0 && kernel<=128 && stride>0 && stride<=32 &&
                    spec.data[0]==float(output) && spec.data[1]==float(kernel) && spec.data[2]==float(stride),"Invalid encoder spec");
            require(product<=meta.hop_length/stride,"Encoder stride overflow"); product*=stride;
            get(p+".weight",{output,input,kernel}); get(p+".bias",{output}); input=output;
        }
        require(meta.encoder_layers==0 || (product==meta.hop_length && input==meta.latent_dim),"Encoder hop/latent mismatch");
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
    Matrix backbone(const Matrix& features) const {
        Matrix x=norm(conv(features,"embed",meta.hidden_dim,7,1),"norm");
        for(uint32_t i=0;i<meta.layers;++i) {
            auto p="blocks."+std::to_string(i);
            Matrix y=linear(norm(conv(x,p+".dwconv",meta.hidden_dim,7,1,true),p+".norm"),p+".pwconv1",meta.intermediate_dim);
            y=y.unaryExpr([](float v){return gelu(v);});
            y=linear(y,p+".pwconv2",meta.hidden_dim);
            auto& gamma=get(p+".gamma",{meta.hidden_dim});
            for(Eigen::Index t=0;t<x.rows();++t) for(uint32_t c=0;c<meta.hidden_dim;++c) x(t,c)+=y(t,c)*gamma.data[c];
        }
        return norm(x,"final_norm");
    }
    std::vector<float> synthesize(const Matrix& features) const {
        Matrix prediction=linear(backbone(features),"head",meta.channels*(meta.n_fft+2));
        size_t frames=features.rows(), n=meta.n_fft, hop=meta.hop_length, bins=n/2+1;
        size_t length=(frames-1)*hop+n, pad=(n-hop)/2;
        std::vector<float> result(frames*hop*meta.channels), window(n), envelope(length,0);
        for(size_t j=0;j<n;++j) window[j]=0.5f-0.5f*std::cos(2*std::numbers::pi_v<float>*float(j)/float(n));
        for(size_t t=0;t<frames;++t) for(size_t j=0;j<n;++j) envelope[t*hop+j]+=window[j]*window[j];
        Eigen::FFT<float> fft;
        for(uint32_t channel=0;channel<meta.channels;++channel) {
            std::vector<float> audio(length,0), frame;
            std::vector<std::complex<float>> spectrum(n);
            for(size_t t=0;t<frames;++t) {
                size_t base=channel*2*bins;
                for(size_t k=0;k<bins;++k) {
                    float magnitude=std::exp(std::min(prediction(t,base+k),std::log(100.0f)));
                    float phase=prediction(t,base+bins+k);
                    spectrum[k]=std::polar(magnitude,phase);
                }
                spectrum[0]={spectrum[0].real(),0}; spectrum[n/2]={spectrum[n/2].real(),0};
                for(size_t k=bins;k<n;++k) spectrum[k]=std::conj(spectrum[n-k]);
                fft.inv(frame,spectrum);
                for(size_t j=0;j<n;++j) audio[t*hop+j]+=frame[j]*window[j];
            }
            for(size_t j=0;j<frames*hop;++j) {
                require(envelope[j+pad]>1e-11f,"Invalid overlap envelope");
                result[j*meta.channels+channel]=audio[j+pad]/envelope[j+pad];
            }
        }
        return result;
    }
};
codec::codec(const std::filesystem::path& path):state(std::make_unique<impl>(path)){}
codec::~codec()=default;
codec::codec(codec&&) noexcept=default;
codec& codec::operator=(codec&&) noexcept=default;
model_info codec::info() const {return state->meta;}
std::vector<float> codec::decode_features(std::span<const float> values,size_t frames) const {
    auto dim=state->meta.latent_dim;
    require(frames>0 && frames<=1000000 && values.size()/dim==frames && values.size()%dim==0,"Invalid features");
    for(float v:values) require(std::isfinite(v),"Nonfinite feature");
    return state->synthesize(Eigen::Map<const Matrix>(values.data(),frames,dim));
}
std::vector<float> codec::decode(const tokens& codes) const {
    auto m=info(); require(codes.frames>0 && codes.frames<=1000000 && codes.codebooks>0 && codes.codebooks<=m.codebooks &&
                          codes.indices.size()==codes.frames*codes.codebooks,"Invalid token dimensions");
    auto& table=state->get("rvq",{m.codebooks,m.entries,m.latent_dim});
    Matrix features=Matrix::Zero(codes.frames,m.latent_dim);
    for(size_t t=0;t<codes.frames;++t) for(uint32_t q=0;q<codes.codebooks;++q) {
        auto index=codes.indices[t*codes.codebooks+q]; require(index<m.entries,"Invalid token index");
        for(uint32_t c=0;c<m.latent_dim;++c) features(t,c)+=table.data[(q*m.entries+index)*m.latent_dim+c];
    }
    return state->synthesize(features);
}
tokens codec::encode(std::span<const float> pcm,uint32_t count) const {
    auto m=info(); require(m.encoder_layers>0,"Decoder-only model");
    require(!pcm.empty() && pcm.size()%m.channels==0 && pcm.size()/m.channels<=48000000 && count>0 && count<=m.codebooks,"Invalid audio/codebook count");
    for(float v:pcm) require(std::isfinite(v),"Nonfinite PCM");
    size_t samples=pcm.size()/m.channels, padded=((samples+m.hop_length-1)/m.hop_length)*m.hop_length;
    Matrix x=Matrix::Zero(padded,m.channels);
    std::copy(pcm.begin(),pcm.end(),x.data());
    for(uint32_t i=0;i<m.encoder_layers;++i) {
        auto p="encoder."+std::to_string(i); auto& spec=state->get(p+".spec",{3});
        x=state->conv(x,p,uint32_t(spec.data[0]),uint32_t(spec.data[1]),uint32_t(spec.data[2]));
        if(i+1<m.encoder_layers) x=x.unaryExpr([](float v){return gelu(v);});
    }
    tokens codes{size_t(x.rows()),count,{}}; codes.indices.resize(codes.frames*count);
    auto& table=state->get("rvq",{m.codebooks,m.entries,m.latent_dim});
    for(size_t t=0;t<codes.frames;++t) for(uint32_t q=0;q<count;++q) {
        float best=std::numeric_limits<float>::infinity(); uint32_t winner=0;
        for(uint32_t k=0;k<m.entries;++k) {
            float distance=0;
            for(uint32_t c=0;c<m.latent_dim;++c) {float d=x(t,c)-table.data[(q*m.entries+k)*m.latent_dim+c]; distance+=d*d;}
            if(distance<best) {best=distance; winner=k;}
        }
        codes.indices[t*count+q]=uint16_t(winner);
        for(uint32_t c=0;c<m.latent_dim;++c) x(t,c)-=table.data[(q*m.entries+winner)*m.latent_dim+c];
    }
    return codes;
}
}
