#!/usr/bin/env python3
"""llama.cpp's own prefill and decode numbers, from llama-server's timings.

Starts llama-server on LK_ORACLE_GPU with ONE slot (so the context is not
split across slots) and the given extra flags, then for each prompt length:
detokenizes the first N ids of the corpus through the server itself (so the
model sees the same token ids the engine is given), sends it with
cache_prompt=false, and records prompt_ms / prompt_per_second /
predicted_per_second exactly as llama.cpp reports them.  Best of --reps.

usage: tools/llamasrv.py --ctx 256,4096 --flags "-ub 2048" --tag ub2048 --out /tmp/x.json
"""
import argparse, json, os, pathlib, shlex, signal, struct, subprocess, sys, time, urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRV = os.environ.get("LK_ORACLE", str(pathlib.Path.home() / ".unsloth/llama.cpp/build/bin/llama-server"))
GGUF = os.environ.get("LK_MODEL", str(pathlib.Path.home() / "qwen-serving/models/LFM2.5-24B-A2B-QAD-Q4_0.gguf"))
PORT = int(os.environ.get("LK_ORACLE_PORT", "18099"))


def post(path, payload, timeout=7200):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path), data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="256,1024,4096,16384,32768,131040")
    ap.add_argument("--corpus", default="corpus_regular.bin")
    ap.add_argument("--flags", default="")
    ap.add_argument("--tag", default="default")
    ap.add_argument("--ngen", type=int, default=32)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    blob = (ROOT / "ref" / a.corpus).read_bytes()
    ids = list(struct.unpack("<%di" % (len(blob) // 4), blob))
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("LK_ORACLE_GPU", "1")
    env["LD_LIBRARY_PATH"] = os.environ.get("LK_OLLAMA_CUDA", "/usr/local/lib/ollama/cuda_v13") + ":" + env.get("LD_LIBRARY_PATH", "")
    cmd = [SRV, "-m", GGUF, "--host", "127.0.0.1", "--port", str(PORT), "-ngl", "99", "-c", "132096",
           "-np", "1", "--no-webui"] + shlex.split(a.flags)
    log = open("/tmp/llamasrv_%s.log" % a.tag, "w")
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        for _ in range(240):
            try:
                with urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=2) as r:
                    if r.status == 200: break
            except Exception:
                time.sleep(1)
        else:
            sys.exit("server did not come up; see /tmp/llamasrv_%s.log" % a.tag)
        rows = []
        print("llama.cpp %s  flags=[%s]" % (a.tag, a.flags))
        print("%8s %8s %10s %12s %10s" % ("ctx", "prompt_n", "prefill_s", "prefill_t/s", "decode_t/s"))
        for c in [int(x) for x in a.ctx.split(",")]:
            text = post("/detokenize", {"tokens": ids[:c]})["content"]
            best = None
            for _ in range(a.reps):
                d = post("/completion", {"prompt": text, "n_predict": a.ngen, "temperature": 0.0,
                                         "cache_prompt": False, "seed": 1234, "ignore_eos": True})
                t = d["timings"]
                r = dict(ctx=c, prompt_n=t["prompt_n"], prefill_s=t["prompt_ms"] / 1000.0,
                         prefill_tps=t["prompt_per_second"], decode_tps=t["predicted_per_second"],
                         gen=t["predicted_n"], tag=a.tag, flags=a.flags)
                if best is None or r["prefill_s"] < best["prefill_s"]: best = dict(r, decode_tps=max(r["decode_tps"], best["decode_tps"]) if best else r["decode_tps"])
                elif r["decode_tps"] > best["decode_tps"]: best["decode_tps"] = r["decode_tps"]
            rows.append(best)
            print("%8d %8d %10.3f %12.1f %10.2f" % (c, best["prompt_n"], best["prefill_s"], best["prefill_tps"], best["decode_tps"]))
            sys.stdout.flush()
            pathlib.Path(a.out).write_text(json.dumps(rows, indent=1))
    finally:
        p.send_signal(signal.SIGINT)
        try: p.wait(timeout=30)
        except Exception: p.kill()


if __name__ == "__main__":
    main()
