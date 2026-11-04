#!/usr/bin/env python3
"""Generate golden dequantization vectors for Q5_0 / Q4_K / Q5_K / Q6_K from the independent gguf-py implementation.

Dev tooling only (NOT part of the runtime): needs `pip install gguf numpy`.
    python tools/gen-quant-vectors.py test/data/quant-vectors.json
"""
import base64
import json
import sys

import numpy as np
from gguf import GGMLQuantizationType as Q
from gguf import quants

rng = np.random.default_rng(1234)
TYPES = {
    "Q5_0": (Q.Q5_0, 32, 22),
    "Q4_K": (Q.Q4_K, 256, 144),
    "Q5_K": (Q.Q5_K, 256, 176),
    "Q6_K": (Q.Q6_K, 256, 210),
}
F16_SCALES = np.array([0.0, 1.0, -1.0, 0.0123, -0.0456, 2.0 ** -24, -(2.0 ** -24), 2.0 ** -14, 3.5, 0.00031, -7.25, 60000.0], dtype=np.float16)


def f16_bytes(v):
    return np.frombuffer(np.array([v], dtype=np.float16).tobytes(), dtype=np.uint8)


def make_blocks(name, n_blocks):
    _, qk, bsz = TYPES[name]
    raw = rng.integers(0, 256, size=(n_blocks, bsz), dtype=np.uint8)
    for b in range(n_blocks):
        pick = lambda: F16_SCALES[rng.integers(0, len(F16_SCALES))]
        if name == "Q5_0":
            raw[b, 0:2] = f16_bytes(pick())
        elif name in ("Q4_K", "Q5_K"):
            raw[b, 0:2] = f16_bytes(pick())
            raw[b, 2:4] = f16_bytes(pick())
        elif name == "Q6_K":
            raw[b, 208:210] = f16_bytes(pick())
    # a few extreme blocks
    if n_blocks >= 4:
        raw[0, 2:] = 0xFF if name == "Q5_0" else raw[0, 2:]
        raw[1, :] = 0
        raw[2, 4:] = 0xFF if name != "Q6_K" else raw[2, 4:]
    return raw


def main(out_path):
    out = {}
    for name, (qt, qk, bsz) in TYPES.items():
        blocks_per_row = 4 if qk == 256 else 8
        rows = 6
        raw = make_blocks(name, rows * blocks_per_row)
        row_bytes = blocks_per_row * bsz
        raw_rows = raw.reshape(rows, row_bytes)
        deq = quants.dequantize(raw_rows, qt).astype(np.float32)
        assert deq.shape == (rows, blocks_per_row * qk), deq.shape
        out[name] = {
            "rows": rows,
            "cols": blocks_per_row * qk,
            "raw": base64.b64encode(raw_rows.tobytes()).decode(),
            "expected": base64.b64encode(deq.astype("<f4").tobytes()).decode(),
        }
    with open(out_path, "w") as f:
        json.dump(out, f)
    print("wrote", out_path, {k: (v["rows"], v["cols"]) for k, v in out.items()})


if __name__ == "__main__":
    main(sys.argv[1])
