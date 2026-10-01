#pragma once
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace vocos {
struct model_info {
    uint32_t sample_rate, channels, hop_length, n_fft, latent_dim;
    uint32_t hidden_dim, intermediate_dim, layers, codebooks, entries, bandwidths;
};
struct tokens {
    size_t frames{};
    uint32_t codebooks{};
    // Frame-major EnCodec indices, then codebook. No packet/container implied.
    std::vector<uint16_t> indices;
};
class codec {
public:
    explicit codec(const std::filesystem::path& weights);
    ~codec();
    codec(codec&&) noexcept;
    codec& operator=(codec&&) noexcept;
    model_info info() const;
    // Decode official Vocos/EnCodec tokens. bandwidth_id maps to 1.5/3/6/12 kbps.
    std::vector<float> decode(const tokens& codes, uint32_t bandwidth_id) const;
    // Decode frame-major, 128-D Vocos features at an explicit bandwidth ID.
    std::vector<float> decode_features(std::span<const float> frame_major, size_t frames,
                                       uint32_t bandwidth_id) const;
private:
    struct impl;
    std::unique_ptr<impl> state;
};
}
