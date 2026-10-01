"""Export canonical NumPy tensors (also usable after converting MLX arrays)."""
import argparse
import json
import struct
from pathlib import Path

import numpy as np

FIELDS = ("sample_rate", "channels", "hop_length", "n_fft", "latent_dim", "hidden_dim",
          "intermediate_dim", "layers", "codebooks", "entries", "encoder_layers")


def export(path, metadata, weights):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    with path.open("xb") as stream:
        stream.write(b"VOCOSN1\0")
        stream.write(struct.pack("<12I", 1, *(int(metadata[k]) for k in FIELDS)))
        stream.write(struct.pack("<I", len(weights)))
        for name, value in sorted(weights.items()):
            value = np.asarray(value, dtype="<f4", order="C")
            if not 1 <= value.ndim <= 4 or not np.isfinite(value).all():
                raise ValueError(f"Invalid tensor: {name}")
            key = name.encode("utf-8")
            stream.write(struct.pack("<I", len(key)))
            stream.write(key)
            stream.write(struct.pack("<I", value.ndim))
            stream.write(struct.pack("<" + "I" * value.ndim, *value.shape))
            stream.write(value.tobytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("tensors", type=Path, help="NPZ with canonical tensor names/layouts")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with np.load(args.tensors, allow_pickle=False) as tensors:
        export(args.output, json.loads(args.metadata.read_text()), dict(tensors))
