#!/usr/bin/env python3
"""Gate the batched prefill against the sequential one on real prompts.

For each prompt length the engine ingests the same tokens twice -- once one
decode step per token (LK_PREFILL=seq, the reference) and once through the
batched path -- and dumps the FULL 128000-entry logit vector at the last
position.  Bit-identity is not expected (the GEMM sums blocks in a different
order and attention runs on fp16 tensor cores), so what is reported is how far
apart the two distributions are: KL divergence, top-1 / top-10 agreement, the
largest logit difference, and the correlation over the whole vocabulary.

usage: tools/pfcheck.py [--len 64,1000,5000] [--corpus corpus_regular.bin] [--chunk 4096,300]
"""
import argparse, os, pathlib, struct, subprocess, sys, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENGINE = pathlib.Path(os.environ.get("LK_ENGINE", ROOT / "build" / "engine"))


def run(ids, mode, chunk, tag):
    pb = pathlib.Path("/tmp/lk_pfcheck_%s.bin" % tag)
    pb.write_bytes(struct.pack("<%di" % len(ids), *ids))
    out = pathlib.Path("/tmp/lk_pfcheck_%s.logits" % tag)
    env = dict(os.environ)
    env.update({"LK_PROMPT_BIN": str(pb), "LK_LOGIT_DUMP": str(out), "LK_PREFILL": mode,
                "LK_PREFILL_CHUNK": str(chunk), "LK_MAX_SEQ": str(max(256, len(ids) + 64)),
                "CUDA_VISIBLE_DEVICES": env.get("LK_GPU", "0")})
    t0 = time.time()
    r = subprocess.run([str(ENGINE), "topk"], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        sys.exit("engine failed (%s): %s" % (tag, r.stderr[-2000:]))
    ing = [l for l in r.stderr.splitlines() if l.startswith("topk: ingested")]
    return np.fromfile(out, dtype=np.float32), (ing[-1] if ing else ""), time.time() - t0


def compare(a, b):
    def lsm(x):
        x = x.astype(np.float64); m = x.max(); return x - m - np.log(np.exp(x - m).sum())
    la, lb = lsm(a), lsm(b)
    pa = np.exp(la)
    kl = float((pa * (la - lb)).sum())
    t10a, t10b = set(np.argsort(-a)[:10]), set(np.argsort(-b)[:10])
    return {"kl": kl, "top1": int(a.argmax() == b.argmax()), "top10": len(t10a & t10b),
            "maxdiff": float(np.abs(a - b).max()), "corr": float(np.corrcoef(a, b)[0, 1]),
            "p_top1_a": float(pa.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--len", default="64,1000,5000")
    ap.add_argument("--corpus", default="corpus_regular.bin")
    ap.add_argument("--chunk", default="4096,300")
    ap.add_argument("--offset", type=int, default=0)
    a = ap.parse_args()
    blob = (ROOT / "ref" / a.corpus).read_bytes()
    corpus = list(struct.unpack("<%di" % (len(blob) // 4), blob))
    print("%7s %6s %10s %6s %6s %9s %10s %8s  %s" % ("len", "chunk", "KL", "top1", "top10", "maxdiff", "corr", "p(top1)", "ingest"))
    worst = 0.0
    for L in [int(x) for x in a.len.split(",")]:
        ids = corpus[a.offset:a.offset + L]
        ref, ing_s, _ = run(ids, "seq", 1, "seq")
        print("%7d %6s %10s %6s %6s %9s %10s %8s  %s" % (L, "seq", "-", "-", "-", "-", "-", "-", ing_s.replace("topk: ", "")))
        for c in [int(x) for x in a.chunk.split(",")]:
            got, ing_b, _ = run(ids, "batch", c, "b%d" % c)
            m = compare(ref, got)
            worst = max(worst, m["kl"])
            print("%7d %6d %10.2e %6d %6d %9.4f %10.7f %8.4f  %s" % (L, c, m["kl"], m["top1"], m["top10"], m["maxdiff"],
                                                               m["corr"], m["p_top1_a"], ing_b.replace("topk: ", "")))
    print("worst KL(seq || batch) = %.3e" % worst)


if __name__ == "__main__":
    main()
