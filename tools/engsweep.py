#!/usr/bin/env python3
"""The engine's prefill and decode across prompt lengths (same ids as llamasrv.py).

usage: tools/engsweep.py --ctx 64,256,4096 --ngen 64 --reps 2 --out /tmp/eng.json
"""
import argparse, json, os, pathlib, struct, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent
ENGINE = pathlib.Path(os.environ.get("LK_ENGINE", ROOT / "build" / "engine"))


def run(ids, ngen):
    pb = pathlib.Path("/tmp/lk_sweep_prompt.bin")
    pb.write_bytes(struct.pack("<%di" % len(ids), *ids))
    env = dict(os.environ)
    env.update({"LK_BENCH": "1", "LK_NGEN": str(ngen), "LK_PROMPT_BIN": str(pb),
                "CUDA_VISIBLE_DEVICES": env.get("LK_GPU", "0")})
    r = subprocess.run([str(ENGINE), "gen"], capture_output=True, text=True, env=env, timeout=7200)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            return dict(p.split("=") for p in line.split()[1:])
    sys.exit("engine failed:\n" + r.stdout[-1500:] + r.stderr[-1500:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="64,256,1024,4096,8192,16384,32768,65536,131040")
    ap.add_argument("--corpus", default="corpus_regular.bin")
    ap.add_argument("--ngen", type=int, default=64)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    blob = (ROOT / "ref" / a.corpus).read_bytes()
    ids = list(struct.unpack("<%di" % (len(blob) // 4), blob))
    rows = []
    print("%8s %10s %12s %10s  %s" % ("ctx", "prefill_s", "prefill_t/s", "decode_t/s", "mode"))
    for c in [int(x) for x in a.ctx.split(",")]:
        best = None
        for _ in range(a.reps):
            kv = run(ids[:c], a.ngen)
            r = dict(ctx=c, prompt_n=int(kv["prompt_n"]), prefill_s=float(kv["prefill_s"]),
                     prefill_tps=float(kv["prefill_tok_s"]), decode_tps=float(kv["tok_s"]),
                     mode=kv.get("prefill_mode", "?"))
            if best is None: best = r
            else:
                if r["prefill_s"] < best["prefill_s"]: best.update(prefill_s=r["prefill_s"], prefill_tps=r["prefill_tps"])
                best["decode_tps"] = max(best["decode_tps"], r["decode_tps"])
        rows.append(best)
        print("%8d %10.3f %12.1f %10.2f  %s" % (c, best["prefill_s"], best["prefill_tps"], best["decode_tps"], best["mode"]))
        sys.stdout.flush()
        pathlib.Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
