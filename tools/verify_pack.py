#!/usr/bin/env python3
"""Gate G1: prove lfm25.pack is a bit-exact, lossless repack of the source GGUF.

For every tensor we reassemble the original interleaved block layout from the
planar pack and compare bytes with the source.  Byte equality is the strongest
possible statement: no requantisation, no rounding, no reordering loss.
"""
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gguf as G

BLOCK_BYTES = {2: 18, 14: 210}


def reassemble(pack, ent):
    """Rebuild the source interleaved block bytes from planar planes."""
    n = ent["blocks"]
    tn = ent["type_name"]
    planes = ent["planes"]

    def rd(p):
        off, size = planes[p]["offset"], planes[p]["size"]
        pack.seek(off)
        b = pack.read(size)
        assert len(b) == size, f"short read {ent['name']}:{p}"
        return b

    if tn == "F32":
        return rd("data")
    if tn == "Q4_0":
        qs = np.frombuffer(rd("qs"), dtype=np.uint8).reshape(n, 16)
        d = np.frombuffer(rd("d"), dtype=np.uint8).reshape(n, 2)
        blk = np.empty((n, 18), dtype=np.uint8)
        blk[:, 0:2] = d
        blk[:, 2:18] = qs
        return blk.tobytes()
    if tn == "Q6_K":
        ql = np.frombuffer(rd("ql"), dtype=np.uint8).reshape(n, 128)
        qh = np.frombuffer(rd("qh"), dtype=np.uint8).reshape(n, 64)
        sc = np.frombuffer(rd("scales"), dtype=np.uint8).reshape(n, 16)
        d = np.frombuffer(rd("d"), dtype=np.uint8).reshape(n, 2)
        blk = np.empty((n, 210), dtype=np.uint8)
        blk[:, 0:128] = ql
        blk[:, 128:192] = qh
        blk[:, 192:208] = sc
        blk[:, 208:210] = d
        return blk.tobytes()
    raise NotImplementedError(tn)


def main():
    model = os.environ["LK_MODEL"]
    packp = os.environ["LK_PACK"]
    manp = os.environ["LK_MANIFEST"]

    man = json.loads(pathlib.Path(manp).read_text())
    print(f"manifest: {man['format']} v{man['version']} tensors={len(man['tensors'])} "
          f"total={man['total_size']:,}")

    # --- structural checks ---
    bad_align = [e["name"] for e in man["tensors"]
                 if any(p["offset"] % man["align"] for p in e["planes"].values())]
    print(f"  plane alignment ({man['align']}B): {'OK' if not bad_align else 'FAIL ' + str(bad_align[:5])}")
    assert os.path.getsize(packp) == man["total_size"], "pack size != manifest total_size"

    g = G.GGUF(model)
    pack = open(packp, "rb")

    order = sorted(man["tensors"], key=lambda e: e["src_offset"])
    total = 0
    failures = []
    for i, ent in enumerate(order):
        src = g.raw(ent["name"])
        got = reassemble(pack, ent)
        if len(got) != len(src) or got != src:
            failures.append(ent["name"])
        total += len(src)
        if i % 60 == 0 or i == len(order) - 1:
            print(f"\r  verifying [{i+1:3d}/{len(order)}] {total/2**30:6.2f} GiB "
                  f"failures={len(failures)}", end="", flush=True)
    print()

    # --- semantic spot-check: dequantise a few tensors from both sides ---
    print("\n  dequant spot-check (pack vs source, fp32):")
    checks = ["token_embd.weight", "blk.0.ffn_gate.weight", "blk.2.attn_q.weight",
              "blk.2.ffn_gate_exps.weight", "blk.0.ffn_down.weight"]
    sem_ok = True
    for nm in checks:
        e = next(x for x in man["tensors"] if x["name"] == nm)
        t = g.tensors[nm]
        src_dq = G.dequantize(g.raw(nm), t.ggml_type, t.nelem)
        got_dq = G.dequantize(reassemble(pack, e), t.ggml_type, t.nelem)
        md = float(np.max(np.abs(src_dq - got_dq)))
        ok = md == 0.0
        sem_ok &= ok
        print(f"    {nm:32s} max|diff|={md:g}  {'OK' if ok else 'FAIL'}")

    print(f"\n  tensors verified : {len(order)}")
    print(f"  bytes verified   : {total:,} ({total/2**30:.3f} GiB)")
    print(f"  byte failures    : {len(failures)}")
    if failures:
        print("  FAILED:", failures[:10])
    ok = not failures and sem_ok and not bad_align
    print(f"\nGATE G1: {'PASS - pack is bit-exact and lossless' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
