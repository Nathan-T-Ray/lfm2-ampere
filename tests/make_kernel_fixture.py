#!/usr/bin/env python3
"""Dump the real F32 weights the kernel tests need.

Only the tensors whose exact layout/semantics are subtle are taken from the
pack: the shortconv tap weights (tap axis is FASTEST and w[0] is the OLDEST),
the MoE router matrix, and the router selection bias.  Everything else in the
test is generated in C++ so the reference can be computed in double precision
without a serialisation format.
"""
import json
import os
import pathlib
import struct
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))

MAN = os.environ["LK_MANIFEST"]
PACK = os.environ["LK_PACK"]


def main():
    man = json.loads(pathlib.Path(MAN).read_text())
    ent = {e["name"]: e for e in man["tensors"]}
    f = open(PACK, "rb")

    def raw(name):
        e = ent[name]
        p = e["planes"]["data"]
        f.seek(p["offset"])
        b = f.read(p["size"])
        assert len(b) == p["size"], name
        return b

    out = pathlib.Path(os.environ.get("LK_REF", ".")) / "kernel_fixture.bin"
    with open(out, "wb") as w:
        w.write(b"LKKF")
        for name, count in (("blk.0.shortconv.conv.weight", 3 * 2048),
                            ("blk.2.ffn_gate_inp.weight", 2048 * 64),
                            ("blk.2.exp_probs_b.bias", 64)):
            b = raw(name)
            a = np.frombuffer(b, dtype=np.float32)
            assert a.size == count, (name, a.size, count)
            w.write(struct.pack("<I", a.size))
            w.write(a.astype(np.float32).tobytes())
            print(f"  {name:34s} {a.size:8d} floats  range [{a.min():+.4f}, {a.max():+.4f}]")
    print(f"wrote {out} ({os.path.getsize(out):,} bytes)")


if __name__ == "__main__":
    main()
