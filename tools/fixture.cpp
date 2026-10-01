#include "vocos.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
int main(int argc,char** argv) {
    try {
        if(argc!=4 && argc!=5) throw std::runtime_error("Usage: vocos_fixture MODEL INPUT_F32 OUTPUT_F32 [--features]");
        vocos::codec model(argv[1]);
        std::ifstream input(argv[2],std::ios::binary|std::ios::ate);
        if(!input || input.tellg()<=0 || input.tellg()%4!=0) throw std::runtime_error("Invalid input");
        std::vector<float> pcm(size_t(input.tellg())/4); input.seekg(0);
        input.read(reinterpret_cast<char*>(pcm.data()),pcm.size()*4);
        std::vector<float> output;
        size_t frames=0;
        if(argc==5) {
            if(std::string(argv[4])!="--features") throw std::runtime_error("Unknown mode");
            frames=pcm.size()/model.info().latent_dim;
            output=model.decode_features(pcm,frames);
        } else {
            auto codes=model.encode(pcm,model.info().codebooks);
            frames=codes.frames;
            output=model.decode(codes);
        }
        std::ofstream stream(argv[3],std::ios::binary);
        stream.write(reinterpret_cast<const char*>(output.data()),output.size()*4);
        if(!stream) throw std::runtime_error("Output write failed");
        std::cout << frames << " frames, " << output.size() << " PCM values\n";
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
