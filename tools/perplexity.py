#!/usr/bin/env python3
"""Gate C2: teacher-forced perplexity, liquid-kernel vs llama.cpp.

Both engines are fed IDENTICAL token IDs, so this is a like-for-like
comparison.  The engine scores every position (build/engine ppl); the oracle
is queried position by position with n_probs=100 and we read off the logprob
of the actual next token.  Positions where the target is outside the oracle's
top-100 are skipped (reported as 'unscored').
"""
import json, math, os, pathlib, sys, urllib.request
import numpy as np

REF = pathlib.Path(os.environ.get("LK_REF", pathlib.Path(__file__).resolve().parent.parent / "ref"))
ORACLE = "http://127.0.0.1:18099"


def post(path, payload, t=600):
    q = urllib.request.Request(ORACLE + path, data=json.dumps(payload).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(q, timeout=t).read())


def main():
    dump = sys.argv[1] if len(sys.argv) > 1 else "/tmp/lk_ppl.txt"
    ids = np.fromfile(REF / "ppl_corpus.bin", dtype=np.int32).tolist()
    lk = {}
    for line in open(dump):
        i, nll, am, tgt = line.split()
        lk[int(i)] = (float(nll), int(am) == int(tgt))
    pos = list(range(4, 300, 4))
    ok = 0; lo = []; lkn = []; agree = 0; unscored = 0
    for i in pos:
        if i not in lk: continue
        r = post("/completion", {"prompt": ids[:i + 1], "n_predict": 1, "n_probs": 100,
            "temperature": 0.0, "repeat_penalty": 1.0, "dry_multiplier": 0.0, "top_k": 0,
            "top_p": 1.0, "min_p": 0.0, "xtc_probability": 0.0, "seed": 1234,
            "post_sampling_probs": False, "cache_prompt": False})
        cp = r.get("completion_probabilities", [])
        if not cp: continue
        d = {e["id"]: e["logprob"] for e in cp[0]["top_logprobs"]}
        tgt = ids[i + 1]
        if tgt not in d: unscored += 1; continue
        ok += 1
        lo.append(-d[tgt]); lkn.append(lk[i][0])
        agree += (cp[0]["top_logprobs"][0]["id"] == tgt) == lk[i][1]
    a, b = sum(lo) / ok, sum(lkn) / ok
    pa, pb = math.exp(a), math.exp(b)
    print(f"positions compared: {ok}  (unscored: {unscored})")
    print(f"  llama.cpp     nll={a:.6f}  ppl={pa:.6f}")
    print(f"  liquid-kernel nll={b:.6f}  ppl={pb:.6f}")
    print(f"  perplexity delta = {100*abs(pb-pa)/pa:.3f}%")
    print(f"  per-position top-1 agreement: {agree}/{ok} = {100*agree/ok:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
