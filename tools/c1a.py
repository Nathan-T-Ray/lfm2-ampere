#!/usr/bin/env python3
"""Gate C1a: full-distribution agreement between liquid-kernel and llama.cpp.

Greedy-trajectory parity (C1) is a weak instrument: it only observes 128000
logits through one argmax, so a systematically shifted distribution can still
produce a long matching prefix and then diverge for reasons that look like
chaos.  C1a instead freezes the prompt, runs ONE forward pass on each engine
and compares the entire vocabulary distribution:

  * step-1 top-1 agreement on every suite case
  * Pearson correlation over all 128000 logits
  * top-5 set overlap and top-10 ordering agreement
  * KL(oracle || ours) and total-variation distance between the softmaxes
  * max |delta| on the top-64 logits (where the mass actually is)

All statistics are computed on raw logits (not softmaxed) for the correlation,
because softmax compresses exactly the large-magnitude region that carries the
decision.
"""
import json
import os
import pathlib
import subprocess
import sys

import numpy as np

ROOT    = pathlib.Path(__file__).resolve().parent.parent
REF     = pathlib.Path(os.environ.get("LK_REF", ROOT / "ref"))
ENGINE  = ROOT / "build" / "engine"
ORACLE  = ROOT / "build" / "oracle_logits"
GGUF    = os.environ.get("LK_MODEL", "/home/nathan/qwen-serving/models/LFM2.5-24B-A2B-QAD-Q4_0.gguf")
TMP     = pathlib.Path("/tmp/c1a")
TMP.mkdir(exist_ok=True)


def sh(cmd, env=None):
    # llama.cpp writes progress bars containing arbitrary bytes; strict
    # utf-8 decoding turns a successful run into a Python traceback.
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600,
                       errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[:2]} failed:\n{r.stdout[-1500:]}\n{r.stderr[-2500:]}")
    return r.stdout


def main():
    suite = json.loads((REF / "parity_suite.json").read_text())["cases"]
    only = sys.argv[1] if len(sys.argv) > 1 else None

    ckpt = REF / "oracle_logits.json"
    cache = json.loads(ckpt.read_text()) if ckpt.exists() else {}

    rows = []
    all_pear = []
    for case in suite:
        name = case["name"]
        if only and name != only:
            continue
        ids = case["prompt_ids"]

        # ---- oracle (llama.cpp), run on the same GPU the oracle server uses
        ob = TMP / f"{name}.oracle.bin"
        if not ob.exists():
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = os.environ.get("LK_ORACLE_GPU", "1")
            out = sh([str(ORACLE), GGUF, str(ob)] + [str(i) for i in ids], env=env)
            print("  " + out.strip())

        # ---- ours
        eb = TMP / f"{name}.engine.bin"
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = os.environ.get("LK_GPU", "0")
        env["LK_LOGIT_DUMP"] = str(eb)
        sh([str(ENGINE), "topk"] + [str(i) for i in ids], env=env)

        o = np.fromfile(ob, dtype=np.float32)
        e = np.fromfile(eb, dtype=np.float32)
        assert o.shape == e.shape, (o.shape, e.shape)
        n = o.size

        # Pearson over the full vocabulary
        pear = float(np.corrcoef(o.astype(np.float64), e.astype(np.float64))[0, 1])
        all_pear.append(pear)

        ot1, et1 = int(np.argmax(o)), int(np.argmax(e))
        ot = np.argsort(-o, kind="stable")
        et = np.argsort(-e, kind="stable")
        top5 = len(set(ot[:5].tolist()) & set(et[:5].tolist()))
        top10_ord = int(np.sum(ot[:10] == et[:10]))
        d64 = float(np.max(np.abs(o[ot[:64]] - e[ot[:64]])))

        # softmax distribution comparison (fp64, max-subtracted)
        def softmax(x):
            y = x.astype(np.float64)
            y -= y.max()
            np.exp(y, out=y)
            y /= y.sum()
            return y
        po, pe = softmax(o), softmax(e)
        eps = 1e-300
        kl = float(np.sum(po * (np.log(po + eps) - np.log(pe + eps))))
        tv = float(0.5 * np.sum(np.abs(po - pe)))

        rows.append(dict(name=name, n=len(ids), pearson=pear, top1_ok=ot1 == et1,
                         oracle_top1=ot1, engine_top1=et1, top5=top5,
                         top10_ord=top10_ord, maxabs_top64=d64, kl=kl, tv=tv))
        print(f"  {name:10s} n={len(ids):4d}  pearson={pear:.7f}  top1 {'OK ' if ot1==et1 else 'BAD'}"
              f"  ({ot1} vs {et1})  top5={top5}/5  top10ord={top10_ord}/10"
              f"  maxdelta@top64={d64:.4f}  KL={kl:.3e}  TV={tv:.3e}")

    print()
    ncases = len(rows)
    n_top1 = sum(r["top1_ok"] for r in rows)
    min_pear = min(r["pearson"] for r in rows)
    print(f"GATE C1a:")
    print(f"  step-1 top-1 agreement : {n_top1}/{ncases}"
          + ("  PASS" if n_top1 == ncases else "  FAIL"))
    print(f"  min Pearson over 128k  : {min_pear:.7f}"
          + ("  PASS" if min_pear >= 0.9999 else "  FAIL"))
    print(f"  mean Pearson           : {float(np.mean(all_pear)):.7f}")
    print(f"  min top-5 overlap      : {min(r['top5'] for r in rows)}/5")
    print(f"  max KL(oracle||ours)   : {max(r['kl'] for r in rows):.3e}")
    print(f"  max TV distance        : {max(r['tv'] for r in rows):.3e}")
    ok = (n_top1 == ncases) and (min_pear >= 0.9999)
    print(f"\n  RESULT: {'PASS' if ok else 'FAIL'}")
    (REF / "c1a_report.json").write_text(json.dumps(
        dict(gate="C1a", passed=ok, n_cases=ncases, top1_agree=n_top1,
             min_pearson=min_pear, mean_pearson=float(np.mean(all_pear)),
             cases=rows), indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
