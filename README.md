# Native Vocos-style stereo codec

> **Warning:** This project was built using AI assistance.

Initial C++20/Eigen float32 inference implementation for a jointly trained codec.
No Python, MLX, Torch, or GPU runtime is required by the C++ library. Eigen's
unsupported FFT module supplies the portable inverse transform. This project is
separate from the existing EnCodec repositories and does not modify ECNG.

Implemented: SAME-padded convolutional encoder with GELU, greedy RVQ, codebook
lookup/sum, unconditioned Vocos ConvNeXt backbone, exact-erf GELU, channel LayerNorm
(epsilon 1e-6), learned layer scales, joint stereo magnitude/phase projection,
periodic Hann inverse-STFT with envelope normalization and SAME trimming.

The encoder is a new compact strided-convolution architecture, not Meta SEANet
and not a pretrained Vocos encoder (Vocos has no waveform encoder). There are no
trained stereo weights yet. This is offline segment inference, not a streaming
player; chunk context, container, quantization, and Android performance remain
future work. Released Vocos checkpoints use different tensor names and can use
adaptive bandwidth normalization, which this first graph does not implement.
They cannot be loaded directly. No pretrained-quality claims are implied.

## Build and numerical verification

From the workspace root, reuse the already downloaded Eigen headers:

```sh
cmake -S vocos.cpp -B vocos.cpp/build -DCMAKE_BUILD_TYPE=Release \
  -DVOCOS_EIGEN_DIR="$PWD/encodec.cpp/.cache/cpm/eigen3/04d2"
cmake --build vocos.cpp/build --parallel 2
.venv-mlx-training/bin/python vocos.cpp/tools/check_parity.py vocos.cpp/build/vocos_fixture
```

The test exports deterministic synthetic weights and compares the complete native
encoder/RVQ/stereo decoder with an independent NumPy implementation at 48 kHz,
hop 320 and FFT 1280. Random weights test numerical behavior, not audio quality.

## Runtime model version 1

Little-endian magic `VOCOSN1\0`, uint32 version, followed by eleven uint32 metadata
fields in the order in `tools/export_model.py`, uint32 tensor count, then named
tensors. Each tensor has uint32 UTF-8 name length, name bytes, uint32 rank, uint32
dimensions, then contiguous little-endian float32 data. No training state is stored.
The loader bounds dimensions and rejects malformed/nonfinite tensor data.

Canonical convolution weights are `[output, input-per-group, kernel]`; linear
weights are `[output, input]`; RVQ is `[codebook, entry, latent]`. Activations are
frame-major. MLX convolution weights must be transposed from `[output, kernel,
input-per-group]` during export. Each `encoder.N.spec` stores three integer-valued
float32 values: output channels, kernel, stride. Encoder stride product equals
hop length. The last encoder layer has no activation. Decoder-only exports set
encoder_layers to zero and omit encoder tensors; RVQ remains present.

Encoder input is interleaved PCM, padded right to a hop multiple. Tokens are
frame-major uint16 indices. Decoded PCM is interleaved and unclipped, with length
`token_frames * hop` per channel; callers retain original length and trim padding.
The network currently has no automatic loudness normalization or frame scales.

`tools/export_model.py metadata.json canonical-weights.npz output.bin` exports
weights without overwriting existing files. A direct MLX checkpoint adapter awaits
the matching training model implementation.

Architecture references: [Vocos](https://github.com/gemelo-ai/vocos) (MIT),
particularly `models.py`, `modules.py`, `heads.py`, and `spectral_ops.py`. The
native implementation was written for this project. The public C++ API and
build organization were also informed by
[`pfeatherstone/encodec.cpp`](https://github.com/pfeatherstone/encodec.cpp),
whose repository is MIT licensed. See [LICENSE](LICENSE) for this repository's
MIT license and [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES/README.md) for
third-party attributions and dependency licensing.
