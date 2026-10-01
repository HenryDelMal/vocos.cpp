"""Independent NumPy reference for the new unconditioned stereo codec graph."""
import argparse
import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from export_model import export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    meta = dict(sample_rate=48000, channels=2, hop_length=320, n_fft=1280,
                latent_dim=4, hidden_dim=6, intermediate_dim=12, layers=2,
                codebooks=2, entries=8, encoder_layers=4)
    weights = {}

    def random(name, shape):
        weights[name] = rng.normal(0, 0.1, shape).astype(np.float32)

    def conv_weights(name, out, inputs, kernel):
        random(name + ".weight", (out, inputs, kernel))
        random(name + ".bias", (out,))

    def norm_weights(name):
        weights[name + ".weight"] = np.ones(6, np.float32)
        weights[name + ".bias"] = np.zeros(6, np.float32)

    random("rvq", (2, 8, 4))
    inputs = 2
    for i, stride in enumerate((8, 5, 4, 2)):
        name = f"encoder.{i}"
        weights[name + ".spec"] = np.array([4, stride * 2, stride], np.float32)
        conv_weights(name, 4, inputs, stride * 2)
        inputs = 4
    conv_weights("embed", 6, 4, 7)
    norm_weights("norm")
    norm_weights("final_norm")
    for i in range(2):
        p = f"blocks.{i}"
        conv_weights(p + ".dwconv", 6, 1, 7)
        norm_weights(p + ".norm")
        random(p + ".pwconv1.weight", (12, 6))
        random(p + ".pwconv1.bias", (12,))
        random(p + ".pwconv2.weight", (6, 12))
        random(p + ".pwconv2.bias", (6,))
        weights[p + ".gamma"] = np.full(6, 0.5, np.float32)
    random("head.weight", (2 * 1282, 6))
    random("head.bias", (2 * 1282,))

    def conv(x, name, stride=1, depthwise=False):
        w, b = weights[name + ".weight"], weights[name + ".bias"]
        kernel = w.shape[-1]
        frames = (len(x) + stride - 1) // stride
        total = max(0, (frames - 1) * stride + kernel - len(x))
        padded = np.pad(x, ((total // 2, total - total // 2), (0, 0)))
        y = np.empty((frames, len(b)))
        for t in range(frames):
            section = padded[t * stride:t * stride + kernel].T
            if depthwise:
                y[t] = np.sum(section * w[:, 0], axis=1) + b
            else:
                y[t] = np.einsum("ock,ck->o", w, section) + b
        return y

    def norm(x, name):
        return ((x - x.mean(-1, keepdims=True)) /
                np.sqrt(x.var(-1, keepdims=True) + 1e-6) * weights[name + ".weight"] +
                weights[name + ".bias"])

    def linear(x, name):
        return x @ weights[name + ".weight"].T + weights[name + ".bias"]

    def gelu(x):
        return 0.5 * x * (1 + np.vectorize(math.erf)(x / np.sqrt(2)))

    # Non-hop-aligned input exercises right padding and all strided layers.
    pcm = rng.normal(0, 0.1, (1287, 2)).astype(np.float32)
    x = np.pad(pcm.astype(np.float64), ((0, 1600 - len(pcm)), (0, 0)))
    for i, stride in enumerate((8, 5, 4, 2)):
        x = conv(x, f"encoder.{i}", stride)
        if i != 3:
            x = gelu(x)
    features = np.zeros_like(x)
    for q in range(2):
        table = weights["rvq"][q]
        indices = ((x[:, None] - table[None]) ** 2).sum(-1).argmin(-1)
        features += table[indices]
        x -= table[indices]
    x = norm(conv(features, "embed"), "norm")
    for i in range(2):
        p = f"blocks.{i}"
        y = norm(conv(x, p + ".dwconv", depthwise=True), p + ".norm")
        y = linear(gelu(linear(y, p + ".pwconv1")), p + ".pwconv2")
        x += y * weights[p + ".gamma"]
    prediction = linear(norm(x, "final_norm"), "head").reshape(5, 2, 2, 641)
    spectrum = np.minimum(np.exp(prediction[:, :, 0]), 100) * np.exp(1j * prediction[:, :, 1])
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(1280) / 1280)
    frames = np.fft.irfft(spectrum, n=1280) * window
    audio = np.zeros((2560, 2))
    envelope = np.zeros(2560)
    for t in range(5):
        audio[t * 320:t * 320 + 1280] += frames[t].T
        envelope[t * 320:t * 320 + 1280] += window ** 2
    expected = audio[480:-480] / envelope[480:-480, None]
    with tempfile.TemporaryDirectory(prefix="vocos-parity-") as directory:
        root = Path(directory)
        export(root / "model.bin", meta, weights)
        pcm.tofile(root / "input.f32")
        subprocess.run([str(args.executable.resolve()), str(root / "model.bin"),
                        str(root / "input.f32"), str(root / "output.f32")], check=True)
        actual = np.fromfile(root / "output.f32", dtype=np.float32).reshape(-1, 2)
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-4)
        print(f"Stereo encoder/RVQ/decoder parity passed; max error {np.max(np.abs(actual-expected)):.3g}")
        decoder_meta = dict(meta, encoder_layers=0)
        export(root / "decoder.bin", decoder_meta,
               {k: v for k, v in weights.items() if not k.startswith("encoder.")})
        features.astype(np.float32).tofile(root / "features.f32")
        subprocess.run([str(args.executable.resolve()), str(root / "decoder.bin"),
                        str(root / "features.f32"), str(root / "decoded.f32"), "--features"], check=True)
        decoded = np.fromfile(root / "decoded.f32", dtype=np.float32).reshape(-1, 2)
        np.testing.assert_allclose(decoded, expected, atol=2e-6, rtol=2e-4)
        print("Decoder-only export and feature input parity passed")
        # A malformed model must fail cleanly, not allocate/infer with bad metadata.
        malformed = bytearray((root / "model.bin").read_bytes())
        malformed[24:28] = (0).to_bytes(4, "little")  # n_fft
        (root / "invalid.bin").write_bytes(malformed)
        result = subprocess.run([str(args.executable.resolve()), str(root / "invalid.bin"),
                                 str(root / "input.f32"), str(root / "unused.f32")], capture_output=True)
        assert result.returncode != 0 and b"Invalid FFT/hop" in result.stderr
        print("Invalid metadata rejection passed")


if __name__ == "__main__":
    main()
