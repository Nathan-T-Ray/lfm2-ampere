#!/usr/bin/env python3
"""Repack the GGUF into lfm25.pack + lfm25.manifest.json.

This is a pure, LOSSLESS data-layout transform — no requantisation, so gate C2
(quality) cannot be affected by it.

Motivation:
  * block_q4_0 is 18 bytes (2-byte half + 16-byte qs), so successive blocks are
    only 2-byte aligned.  Q4_0 -> planar split into `qs` (16 B/block) and
    `d` (2 B/block) makes every qs access 16-byte aligned/vectorisable.
  * block_q6_K is 210 bytes = ql[128] + qh[64] + scales[16] + half d[2], and
    128+64+16 = 208 = 13*16.  Within a 2048-wide row, block b starts at 210*b,
    so blocks 1..7 are only 2-byte aligned.  Splitting `d` out makes ql/qh/scales
    fully 16-byte aligned across all 128000 vocabulary rows.

Pack layout: 4096-byte header (page aligned) then every plane at a 256-byte
aligned offset.  The manifest is the source of truth for offsets.
"""
import argparse
import json
import os
import pathlib
import struct
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gguf as G

ALIGN = 256
HEADER_SIZE = 4096
MAGIC = b"LKPACK\x00\x00"
VERSION = 1


def align_up(x, a=ALIGN):
    return (x + a - 1) // a * a


def split_planes(raw, type_name, nblocks):
    """Return an ordered dict of plane_name -> bytes, all losslessly extracted."""
    if type_name == "F32":
        return [("data", raw)]
    if type_name == "Q4_0":
        blk = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 18)
        return [("qs", np.ascontiguousarray(blk[:, 2:18]).tobytes()),
                ("d", np.ascontiguousarray(blk[:, 0:2]).tobytes())]
    if type_name == "Q6_K":
        blk = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 210)
        return [("ql", np.ascontiguousarray(blk[:, 0:128]).tobytes()),
                ("qh", np.ascontiguousarray(blk[:, 128:192]).tobytes()),
                ("scales", np.ascontiguousarray(blk[:, 192:208]).tobytes()),
                ("d", np.ascontiguousarray(blk[:, 208:210]).tobytes())]
    raise NotImplementedError(type_name)


PLANE_GEOM = {  # type -> [(plane, bytes_per_block), ...]
    "F32":  [("data", 4)],
    "Q4_0": [("qs", 16), ("d", 2)],
    "Q6_K": [("ql", 128), ("qh", 64), ("scales", 16), ("d", 2)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("LK_MODEL"))
    ap.add_argument("--out", default=os.environ.get("LK_PACK"))
    ap.add_argument("--manifest", default=os.environ.get("LK_MANIFEST"))
    args = ap.parse_args()

    g = G.GGUF(args.model)
    v = g.verify_contiguity()
    if not v["exact"]:
        print("ERROR: source GGUF layout is not contiguous/exact", file=sys.stderr)
        return 2
    print(f"source: {args.model}")
    print(f"  tensors={len(g.tensors)} data_start={g.data_start:,} "
          f"tensor_bytes={v['total_bytes']:,} exact={v['exact']}")

    # ---- assign plane offsets --------------------------------------
    entries = []
    cur = HEADER_SIZE
    order = sorted(g.tensors.values(), key=lambda t: t.offset)
    for t in order:
        _, bsize, tsize = G.GGML_TYPES[t.ggml_type]
        nblocks = t.nelem // bsize
        cols = t.dims[0]
        rows = t.nelem // cols if cols else 1
        blocks_per_row = cols // bsize
        planes = {}
        # place planes largest-first as they appear in PLANE_GEOM
        for pname, pbytes in PLANE_GEOM[t.type_name]:
            size = nblocks * pbytes
            off = align_up(cur)
            planes[pname] = {
                "offset": off, "size": size,
                "bytes_per_block": pbytes,
                "row_stride": blocks_per_row * pbytes,
            }
            cur = off + size
        entries.append({
            "name": t.name,
            "ggml_type": t.ggml_type,
            "type_name": t.type_name,
            "dims": t.dims,           # ggml order: dims[0] is fastest-varying
            "nelem": t.nelem,
            "src_offset": t.offset,
            "src_nbytes": t.nbytes,
            "blocks": nblocks,
            "block_elems": bsize,
            "block_bytes": tsize,
            "cols": cols,
            "rows": rows,
            "blocks_per_row": blocks_per_row,
            "planes": planes,
        })
    total = align_up(cur)
    print(f"  pack planes: {len(entries)} tensors, total={total:,} bytes ({total/2**30:.3f} GiB)")

    manifest = {
        "format": "lfm25.pack",
        "version": VERSION,
        "align": ALIGN,
        "header_size": HEADER_SIZE,
        "data_size": total - HEADER_SIZE,
        "total_size": total,
        "source": os.path.basename(args.model),
        "source_gguf": {
            "version": g.version,
            "data_start": g.data_start,
            "tensor_bytes": v["total_bytes"],
            "n_tensors": len(g.tensors),
            "meta": {k: (val if not isinstance(val, list) else f"<list n={len(val)}>")
                     for k, val in sorted(g.kv.items())},
        },
        "tensors": entries,
    }

    # ---- headroom check --------------------------------------------
    st = os.statvfs(os.path.dirname(args.out))
    free = st.f_bavail * st.f_frsize
    if free < total + (1 << 30):
        print(f"ERROR: need {total/2**30:.2f} GiB, only {free/2**30:.2f} GiB free", file=sys.stderr)
        return 2

    # ---- write ------------------------------------------------------
    tmp = args.out + ".tmp"
    written = 0
    with open(tmp, "wb") as out:
        out.write(b"\x00" * HEADER_SIZE)
        for i, (t, ent) in enumerate(zip(order, entries)):
            raw = g.raw(t.name)
            assert len(raw) == ent["src_nbytes"]
            for pname, pdata in split_planes(raw, t.type_name, ent["blocks"]):
                off = ent["planes"][pname]["offset"]
                out.seek(off)
                out.write(pdata)
                written += len(pdata)
            if i % 40 == 0 or i == len(order) - 1:
                pct = 100.0 * (i + 1) / len(order)
                print(f"\r  [{i+1:3d}/{len(order)}] {pct:5.1f}%  {written/2**30:6.2f} GiB", end="", flush=True)
        out.truncate(total)
        # Header only.  The manifest is ~230 KiB, far larger than HEADER_SIZE,
        # so it lives in its own file; embedding it in the header would clobber
        # the planes of the first tensor.  (Gate G1 caught exactly that.)
        out.seek(0)
        out.write(MAGIC)
        out.write(struct.pack("<IIQQQ", VERSION, HEADER_SIZE, total, len(order), 0))
    print()
    os.replace(tmp, args.out)
    pathlib.Path(args.manifest).write_text(json.dumps(manifest, indent=1))
    print(f"wrote {args.out} ({os.path.getsize(args.out):,} bytes)")
    print(f"wrote {args.manifest} ({os.path.getsize(args.manifest):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
