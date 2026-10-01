# Vocos.cpp — Experimental Vocos C++ implementation

> **Warning:** This project was built using AI assistance. This README.md too (okay, this text was human-written), so it may be very imprecise between different iterations. Sometimes LLMs understand weird things.

This repository reimplements the **pretrained Vocos EnCodec 24 kHz decoder** in
C++20. It loads the released `charactr/vocos-encodec-24khz` checkpoint, converts
EnCodec RVQ indices into the feature vectors Vocos expects, then runs Vocos's
ConvNeXt backbone and inverse STFT head. It is a decoder implementation; it
does not contain a waveform encoder or retrain Vocos. Create codes with a
compatible EnCodec 24 kHz encoder, or supply 128-dimensional Vocos features.

Supported checkpoint geometry is mono 24 kHz, 75 frames/second, 1280-point FFT,
320-sample hop, 128 input features, eight 384-wide ConvNeXt blocks, and 1152-wide
intermediate layers. It supports the pretrained bandwidth IDs 0–3 (1.5, 3, 6,
and 12 kbps), corresponding to 2, 4, 8, and 16 EnCodec codebooks. This matches
the released Vocos config's adaptive normalization and learned codebook table.
Other Vocos configurations are rejected.

This is offline segment inference. Streaming, entropy coding, packet/container
format, automatic gain metadata, and Android integration are outside the current
implementation. Vocos quality depends on using the matching EnCodec code indices
and checkpoint; this repository does not include pretrained model weights.

## Build

Requirements: CMake 3.23+, a C++20 compiler, and Eigen 3 headers.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DVOCOS_EIGEN_DIR=/path/to/eigen
cmake --build build --parallel
```

## Convert the pretrained checkpoint

Download `config.yaml` and `pytorch_model.bin` from
[`charactr/vocos-encodec-24khz`](https://huggingface.co/charactr/vocos-encodec-24khz),
then run with Python, PyTorch, Vocos, NumPy, and PyYAML installed:

```sh
python tools/export_vocos.py config.yaml pytorch_model.bin vocos24.vocos
```

The exporter validates the Vocos architecture and codebook table before writing
a portable little-endian inference file. It includes the decoder, adaptive
normalization tables, and the Vocos EnCodec codebook embeddings.

## C++ API

`vocos::codec::decode` accepts frame-major EnCodec token indices and a Vocos
`bandwidth_id` (0–3). `decode_features` accepts frame-major 128D feature vectors
at the same explicit bandwidth ID. Both return unclipped mono float PCM at
24 kHz. The API expects already encoded tokens: this library does not analyze
waveform audio or create RVQ codes.

`tools/vocos_fixture` is a small raw-binary harness for these two decoder paths.
Input tokens are uint16 frame-major indices; feature input and decoded output
are float32. This harness is for integration and parity checks, not a packaged
audio file interface.

## Verify against Vocos

With the converted checkpoint and Python Vocos environment available:

```sh
python tools/check_pretrained.py build/vocos_fixture config.yaml pytorch_model.bin
```

The parity script compares C++ output with the official PyTorch Vocos backbone
and ISTFT head for both code and feature inputs at all four pretrained bandwidth
IDs. It includes one-frame and longer segments plus left/right context chunk
checks matching the decoding approach used by the Radios Emergencia Chile app.
It tests actual pretrained parameters, not synthetic random weights.

## Format

The `.vocos` file uses `VOCOSN2` magic and a versioned, little-endian tensor container. Version 2 stores
the fixed Vocos 24 kHz mono geometry, 16 codebooks of 1024 128D embeddings, all
eight adaptive-normalized ConvNeXt blocks, and the inverse-STFT projection.
`tools/export_model.py` contains the canonical tensor writer; the public
`tools/export_vocos.py` converts the official Vocos PyTorch checkpoint.

## Attribution and licensing

The neural decoder follows Vocos's ConvNeXt backbone, bandwidth-conditioned
normalization, Fourier head, and same-padded inverse STFT. See
[`THIRD_PARTY_LICENSES`](THIRD_PARTY_LICENSES/README.md) for Vocos attribution,
the EnCodec C++ design reference, and Eigen's external dependency license.
This repository is MIT licensed; see [`LICENSE`](LICENSE).
