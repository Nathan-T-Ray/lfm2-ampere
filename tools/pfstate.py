#!/usr/bin/env python3
"""Layer-by-layer agreement of the state a prompt leaves behind, batched vs
sequential prefill.

End-to-end logits cannot separate a structural bug from this model's chaos: a
1e-7 rounding change flips a near-tied expert somewhere, the flip propagates
through the KV cache and the conv state, and per-position NLL then moves by
tenths of a nat.  The state DOES separate them.  Layers before the first MoE
router (embed, layers 0-1 dense, layer 2's mixer) see only rounding-level
differences, so their K/V must agree to fp16 precision; a wrong position, mask,
tap or scale shows up there at O(1).  Deeper layers diverge only as routing
diverges, and the route log says by how much.

usage: tools/pfstate.py [--len 2000] [--corpus corpus_regular.bin]
"""
import argparse, os, pathlib, struct, subprocess, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENGINE = pathlib.Path(os.environ.get("LK_ENGINE", ROOT / "build" / "engine"))
ATTN_LAYERS = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38]
CONV_LAYERS = [i for i in range(40) if i % 4 != 2]


def run(ids, mode, tag):
    pb = pathlib.Path("/tmp/lk_pfs_%s.bin" % tag)
    pb.write_bytes(struct.pack("<%di" % len(ids), *ids))
    kv = pathlib.Path("/tmp/lk_pfs_%s.kv" % tag)
    rl = pathlib.Path("/tmp/lk_pfs_%s.route" % tag)
    env = dict(os.environ)
    env.update({"LK_PROMPT_BIN": str(pb), "LK_DUMP_KV": str(kv), "LK_ROUTE_LOG": str(rl),
                "LK_PREFILL": mode, "LK_MAX_SEQ": str(max(256, len(ids) + 64)),
                "CUDA_VISIBLE_DEVICES": env.get("LK_GPU", "0")})
    r = subprocess.run([str(ENGINE), "topk"], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        sys.exit("engine failed: " + r.stderr[-2000:])
    n = len(ids)
    raw = kv.read_bytes()
    per = n * 8 * 64
    kvh = np.frombuffer(raw[:10 * 2 * per * 2], dtype=np.float16).astype(np.float32).reshape(10, 2, n, 8, 64)
    hist = np.frombuffer(raw[10 * 2 * per * 2:], dtype=np.float32).reshape(30, 2048, 2)
    route = np.fromfile(rl, dtype=np.int32).reshape(n, 40, 4)
    return kvh, hist, route


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--len", type=int, default=2000)
    ap.add_argument("--corpus", default="corpus_regular.bin")
    a = ap.parse_args()
    blob = (ROOT / "ref" / a.corpus).read_bytes()
    ids = list(struct.unpack("<%di" % a.len, blob[:4 * a.len]))
    ks, hs, rs = run(ids, "seq", "seq")
    kb, hb, rb = run(ids, "batch", "batch")
    print("prompt of %d tokens; relative error = |batch - seq| / rms(seq)" % a.len)
    print("%5s %8s %12s %12s %12s %12s" % ("layer", "kind", "K rel max", "K rel rms", "V rel max", "V rel rms"))
    rows = []
    for al, il in enumerate(ATTN_LAYERS):
        out = []
        for kv in range(2):
            s, b = ks[al, kv], kb[al, kv]
            rms = float(np.sqrt((s * s).mean()))
            d = np.abs(b - s)
            out += [float(d.max()) / rms, float(np.sqrt((d * d).mean())) / rms]
        rows.append((il, "attn", *out))
    for ci, il in enumerate(CONV_LAYERS):
        s, b = hs[ci], hb[ci]
        rms = float(np.sqrt((s * s).mean()))
        d = np.abs(b - s)
        rows.append((il, "conv", float(d.max()) / rms, float(np.sqrt((d * d).mean())) / rms, float("nan"), float("nan")))
    for r in sorted(rows):
        print("%5d %8s %12.3e %12.3e %12.3e %12.3e" % r)
    print("\nrouting: fraction of prompt positions whose top-4 (ordered) matches, per MoE layer")
    same = (rs == rb).all(axis=2)          # [n, 40]
    line = []
    for il in range(2, 40):
        line.append("%d:%.3f" % (il, same[:-1, il].mean()))
    print("  " + "  ".join(line))


if __name__ == "__main__":
    main()
