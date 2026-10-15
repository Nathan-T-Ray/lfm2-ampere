#!/usr/bin/env python3
"""Emit a binary fixture of REAL weight blocks + reference dot products.

Cases are drawn from the actual lfm25.pack, so the test exercises the real
planar layouts and the real quantised data rather than synthetic values.

Reference values are computed two ways:
  expected_exact : dot against the exactly-dequantised Q8_1 activation
                   (isolates formula error -- should match to ~1e-6)
  expected_fp32  : dot against the original fp32 activation
                   (includes Q8_1 rounding -- ~1e-3)
"""
import json
import os
import pathlib
import struct
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))
import gguf as G

KIND_Q4_0, KIND_Q6_K = 0, 1


def q8_1_quantize(x, nblocks):
    """ggml-cuda convention, PER 32-ELEMENT BLOCK: ds = (amax/127, RAW sum).

    Both the amax and the sum are reduced over the 32 elements of one block
    (ggml does this warp-cooperatively), NOT over the whole activation vector.
    """
    xb = x.reshape(nblocks, 32)
    amax = np.max(np.abs(xb), axis=1)                       # (nblocks,)
    d = (amax / 127.0).astype(np.float32)
    safe = np.where(d > 0, d, 1.0).astype(np.float32)
    q = np.where(d[:, None] > 0, np.round(xb / safe[:, None]), 0.0)
    qi = np.clip(q, -128, 127).astype(np.int8)
    return qi.reshape(-1), d, np.sum(xb, axis=1).astype(np.float32)


def main():
    man = json.loads(pathlib.Path(os.environ["LK_MANIFEST"]).read_text())
    pack = open(os.environ["LK_PACK"], "rb")
    ent = {e["name"]: e for e in man["tensors"]}
    rng = np.random.default_rng(20261015)
    cases = []

    def plane(e, p):
        o, s = e["planes"][p]["offset"], e["planes"][p]["size"]
        pack.seek(o)
        return pack.read(s)

    # ---- Q4_0 cases: blk.2.attn_q.weight [2048,2048], 64 blocks/row ----
    e = ent["blk.2.attn_q.weight"]
    qs = np.frombuffer(plane(e, "qs"), np.uint8).reshape(-1, 16)
    dd = np.frombuffer(plane(e, "d"), np.uint16).reshape(-1)
    assert e["blocks_per_row"] == 64
    for nblk, row in ((64, 0), (64, 1234), (8, 77), (1, 5)):
        base = row * e["blocks_per_row"]
        sl = slice(base, base + nblk)
        q = qs[sl]
        dh = dd[sl].copy().view(np.float16).astype(np.float32)
        # exact dequantised weights
        nib_lo = (q & 0x0F).astype(np.int16) - 8
        nib_hi = (q >> 4).astype(np.int16) - 8
        w = np.concatenate([nib_lo, nib_hi], axis=1).astype(np.float32).reshape(-1) * np.repeat(dh, 32)
        x = rng.standard_normal(nblk * 32).astype(np.float32)
        qi, d8, rawsum = q8_1_quantize(x, nblk)
        xq = qi.astype(np.float32) * d8.repeat(32)
        # ggml-cuda's exact integer convention, block by block:
        #   value = sum_blocks d4 * (d8 * sum(nib*q8) - 8 * SUM_raw)
        nib = np.concatenate([(q & 0x0F), (q >> 4)], axis=1).astype(np.int32)  # (nblk,32)
        qb = qi.reshape(nblk, 32).astype(np.int32)
        sumi = (nib * qb).sum(axis=1).astype(np.float64)
        ggml = float(np.sum(dh.astype(np.float64) *
                            (d8.astype(np.float64) * sumi - 8.0 * rawsum.astype(np.float64))))
        cases.append(dict(kind=KIND_Q4_0, nblocks=nblk,
                          planes=[q.tobytes(), dd[sl].tobytes()],
                          x=x, expected_exact=ggml,
                          expected_fp32=float(np.dot(w, x))))

    # ---- Q6_K cases: token_embd.weight [2048,128000], 8 superblocks/row ----
    e = ent["token_embd.weight"]
    ql = np.frombuffer(plane(e, "ql"), np.uint8).reshape(-1, 128)
    qh = np.frombuffer(plane(e, "qh"), np.uint8).reshape(-1, 64)
    sc = np.frombuffer(plane(e, "scales"), np.uint8).reshape(-1, 16)
    kd = np.frombuffer(plane(e, "d"), np.uint16).reshape(-1)
    assert e["blocks_per_row"] == 8
    for nblk, row in ((8, 124890), (8, 0), (2, 4321), (1, 7)):
        base = row * e["blocks_per_row"]
        sl = slice(base, base + nblk)
        qlb, qhb, scb, kdb = ql[sl], qh[sl], sc[sl], kd[sl]
        # exact dequant via the verified mapping, reused from tools/gguf.py
        raw = np.concatenate([qlb, qhb, scb, kdb.view(np.uint8).reshape(-1, 2)], axis=1).tobytes()
        w = G.dequantize(raw, 14, nblk * 256).astype(np.float32)
        x = rng.standard_normal(nblk * 256).astype(np.float32)
        qi, d8, rawsum = q8_1_quantize(x, nblk * 8)
        xq = qi.astype(np.float32) * d8.repeat(32)
        cases.append(dict(kind=KIND_Q6_K, nblocks=nblk,
                          planes=[qlb.tobytes(), qhb.tobytes(), scb.tobytes(), kdb.tobytes()],
                          x=x, expected_exact=float(np.dot(w, xq)),
                          expected_fp32=float(np.dot(w, x))))

    out = pathlib.Path(os.environ.get("LK_REF", ".")) / "quant_fixture.bin"
    with open(out, "wb") as f:
        f.write(b"LKQF")
        f.write(struct.pack("<I", len(cases)))
        for c in cases:
            f.write(struct.pack("<IIIf", c["kind"], c["nblocks"], len(c["planes"]), 0.0))
            for p in c["planes"]:
                f.write(struct.pack("<I", len(p)))
                f.write(p)
            f.write(struct.pack("<I", len(c["x"])))
            f.write(c["x"].astype(np.float32).tobytes())
            f.write(struct.pack("<ff", c["expected_exact"], c["expected_fp32"]))
    print(f"wrote {out} ({os.path.getsize(out):,} bytes), {len(cases)} cases")
    for c in cases:
        kn = "Q4_0" if c["kind"] == KIND_Q4_0 else "Q6_K"
        print(f"  {kn:5s} nblocks={c['nblocks']:3d} exact={c['expected_exact']:+.6f} "
              f"fp32={c['expected_fp32']:+.6f}")


if __name__ == "__main__":
    main()
