#!/usr/bin/env python3
"""Gate C2: teacher-forced perplexity, and who is actually right.

Reports THREE numbers, not two:

  exact          the lm_head recomputed in float64 from the GGUF, applied to the
                 residual stream the engine dumped.  This is what the model
                 mathematically prescribes, so it is the only defensible
                 reference.  Comparing two engines to each other cannot say
                 which one is correct.
  liquid-kernel  our engine
  llama.cpp      the oracle

The original version of this gate scored 74 subsampled positions of a corpus
that llama.cpp itself had generated, and treated llama.cpp as ground truth.
Both choices are wrong: greedy-selected text is selection biased against any
other engine, and llama.cpp's own lm_head turns out to sit 17x further from
exact mathematics than ours does.  On this corpus the whole reported "delta" is
the oracle's error.
"""
import json, math, os, pathlib, subprocess, sys
import numpy as np

ROOT   = pathlib.Path(__file__).resolve().parent.parent
REF    = pathlib.Path(os.environ.get("LK_REF", ROOT / "ref"))
SYS    = ROOT / "tools"
# LK_ENGINE lets a variant binary be scored without rebuilding over build/engine
# -- the same convention tools/parity.py and tools/sweep.py already use.
ENGINE = pathlib.Path(os.environ.get("LK_ENGINE", ROOT / "build" / "engine"))
ORACLE = ROOT / "build" / "oracle_ppl"
GGUF   = os.environ.get("LK_MODEL", str(ROOT / "models" / "LFM2.5-24B-A2B-QAD-Q4_0.gguf"))
NL, H  = 41, 2048
EPS    = 1e-5


def sh(cmd, env=None):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=7200,
                       errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{r.stdout[-1500:]}\n{r.stderr[-2500:]}")
    return r.stdout


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else "ppl_natural.bin"
    n_exact = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    tokf = REF / corpus
    ids = np.fromfile(tokf, dtype=np.int32)
    n = len(ids)
    print(f"corpus: {corpus}  ({n} tokens -> {n-1} scored positions)")

    tmp = pathlib.Path("/tmp")
    odump, edump, ldump = tmp/"c2.oracle.txt", tmp/"c2.engine.txt", tmp/"c2.layers.bin"

    env = dict(os.environ)
    env["LK_BACKEND_DIR"] = os.environ.get("LK_BACKEND_DIR",
                                           str(pathlib.Path.home() / "llama.cpp" / "build" / "bin"))
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("LK_ORACLE_GPU", "0")
    if not odump.exists():
        print(sh([str(ORACLE), GGUF, str(tokf), str(odump)], env=env).strip())

    env2 = dict(os.environ)
    env2["CUDA_VISIBLE_DEVICES"] = os.environ.get("LK_GPU", "0")
    env2["LK_PPL_DUMP"] = str(edump)
    env2["LK_NO_GRAPH"] = "1"                 # the dump hooks are illegal in a capture
    env2["LK_DUMP_LAYERS"] = str(ldump)
    print(sh([str(ENGINE), "ppl", str(tokf)], env=env2).strip())

    eng = {int(p[0]): (float(p[1]), int(p[2])) for p in
           (l.split() for l in open(edump))}
    orc = {int(p[0]): (float(p[1]), int(p[2])) for p in
           (l.split() for l in open(odump) if not l.startswith("#"))}

    # ---- the exact reference -------------------------------------------
    sys.path.insert(0, str(SYS))
    import gguf as G
    g = G.GGUF(GGUF)
    wnorm = G.dequantize_tensor(g, "token_embd_norm.weight").astype(np.float64)
    W = G.dequantize_tensor(g, "token_embd.weight").astype(np.float32)
    raw = np.memmap(ldump, dtype=np.float32, mode="r")
    npos = raw.size // (NL * H)
    cand = [i for i in range(min(npos, n - 1)) if i in eng and i in orc]
    rng = np.random.default_rng(7)
    sel = sorted(rng.choice(cand, size=min(n_exact, len(cand)), replace=False).tolist())

    ex, en, orr, xe, xo = [], [], [], [], []
    for i in sel:
        x = np.array(raw[i*NL*H + (NL-1)*H: i*NL*H + NL*H], dtype=np.float64)
        inv = 1.0/np.sqrt(float((x*x).sum())/H + EPS)
        lg = (W @ ((x*inv)*wnorm).astype(np.float32)).astype(np.float64)
        m = lg.max(); lse = m + np.log(np.exp(lg-m).sum())
        tgt = int(ids[i+1])
        ex.append(-(lg[tgt]-lse)); en.append(eng[i][0]); orr.append(orc[i][0])
        xe.append(int(np.argmax(lg)) == tgt); xo.append(orc[i][1] == tgt)
    ex, en, orr = map(np.array, (ex, en, orr))

    print(f"\nover {len(sel)} paired positions:")
    print(f"  exact (float64 lm_head) nll={ex.mean():.6f}  ppl={math.exp(ex.mean()):.6f}")
    print(f"  liquid-kernel           nll={en.mean():.6f}  ppl={math.exp(en.mean()):.6f}")
    print(f"  llama.cpp               nll={orr.mean():.6f}  ppl={math.exp(orr.mean()):.6f}")
    de, do = abs(en.mean()-ex.mean()), abs(orr.mean()-ex.mean())
    print(f"\n  |liquid-kernel - exact| = {100*de/ex.mean():.4f}%   (sd per position {(en-ex).std(ddof=1):.6f})")
    print(f"  |llama.cpp     - exact| = {100*do/ex.mean():.4f}%   (sd per position {(orr-ex).std(ddof=1):.6f})")
    print(f"  oracle is {do/max(de,1e-12):.1f}x further from exact than the engine")

    # full-corpus engine vs oracle, for continuity with earlier reports
    common = sorted(set(eng) & set(orc))
    a = np.mean([eng[i][0] for i in common]); b = np.mean([orc[i][0] for i in common])
    print(f"\nfull corpus ({len(common)} positions), engine vs oracle:")
    print(f"  ppl {math.exp(a):.6f} vs {math.exp(b):.6f}  ->  "
          f"{100*abs(math.exp(a)-math.exp(b))/math.exp(b):.4f}% apart")
    print(f"  top-1 agreement: {sum(eng[i][1]==orc[i][1] for i in common)}/{len(common)}"
          f" = {100*sum(eng[i][1]==orc[i][1] for i in common)/len(common):.2f}%")
    print(f"\n  GATE C2 vs exact (<=0.1%): {'PASS' if 100*de/ex.mean() <= 0.1 else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
