#pragma once
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace vocos {
struct model_info {
    uint32_t sample_rate, channels, hop_length, n_fft, latent_dim;
    uint32_t hidden_dim, intermediate_dim, layers, codebooks, entries, encoder_layers;
};
struct tokens {
    size_t frames{};
    uint32_t codebooks{};
    // Frame-major, then codebook. No entropy coding or container implied.
    std::vector<uint16_t> indices;
};
class codec {
public:
    explicit codec(const std::filesystem::path& weights);
    ~codec();
    codec(codec&&) noexcept;
    codec& operator=(codec&&) noexcept;
    model_info info() const;
    // Interleaved PCM; right zero padding to the next hop.
    tokens encode(std::span<const float> pcm, uint32_t codebooks) const;
    // Unclipped interleaved PCM, exactly frames * hop samples/channel.
    std::vector<float> decode(const tokens& codes) const;
    std::vector<float> decode_features(std::span<const float> frame_major, size_t frames) const;
private:
    struct impl;
    std::unique_ptr<impl> state;
};
}
