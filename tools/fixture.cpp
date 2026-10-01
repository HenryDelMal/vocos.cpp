#include "vocos.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
int main(int argc,char** argv) {
    try {
        if(argc!=6) throw std::runtime_error("Usage: vocos_fixture MODEL INPUT OUTPUT (--features|--codes) BANDWIDTH_ID");
        vocos::codec model(argv[1]);
        const std::string mode=argv[4];
        uint32_t bandwidth=uint32_t(std::stoul(argv[5]));
        if(bandwidth>3) throw std::runtime_error("Bandwidth ID must be 0..3");
        std::vector<float> output;
        if(mode=="--features") {
            std::ifstream input(argv[2],std::ios::binary|std::ios::ate);
            auto end=input.tellg();
            if(!input || end<=0 || end%4!=0) throw std::runtime_error("Invalid feature input");
            std::vector<float> features(size_t(end)/4); input.seekg(0);
            input.read(reinterpret_cast<char*>(features.data()),std::streamsize(features.size()*sizeof(float)));
            size_t frames=features.size()/128;
            if(frames*128!=features.size()) throw std::runtime_error("Invalid feature dimensions");
            output=model.decode_features(features,frames,bandwidth);
            std::cout<<frames<<" Vocos feature frames\n";
        } else if(mode=="--codes") {
            constexpr uint32_t counts[]{2,4,8,16};
            std::ifstream input(argv[2],std::ios::binary|std::ios::ate);
            auto end=input.tellg();
            if(!input || end<=0 || end%2!=0) throw std::runtime_error("Invalid token input");
            std::vector<uint16_t> indices(size_t(end)/2); input.seekg(0);
            input.read(reinterpret_cast<char*>(indices.data()),std::streamsize(indices.size()*sizeof(uint16_t)));
            uint32_t q=counts[bandwidth];
            if(indices.size()%q) throw std::runtime_error("Token count is not divisible by bandwidth codebooks");
            vocos::tokens codes{indices.size()/q,q,std::move(indices)};
            output=model.decode(codes,bandwidth);
            std::cout<<codes.frames<<" frames, "<<q<<" codebooks\n";
        } else throw std::runtime_error("Unknown mode");
        std::ofstream stream(argv[3],std::ios::binary);
        stream.write(reinterpret_cast<const char*>(output.data()),std::streamsize(output.size()*sizeof(float)));
        if(!stream) throw std::runtime_error("Output write failed");
        std::cout<<output.size()<<" mono PCM samples at 24000 Hz\n";
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
