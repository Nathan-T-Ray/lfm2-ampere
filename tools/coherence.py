#!/usr/bin/env python3
"""Coherence: does the engine still produce text that makes sense?

THE PROBLEM WITH THE PREVIOUS CLAIM.  "It looks good" was an eyeball on a
handful of generations, which cannot distinguish "correct" from "fluent and
subtly wrong".  A quantisation or attention bug that costs 0.5% perplexity
produces text that reads perfectly while getting arithmetic, code and closed-form
facts wrong.  So every item here is scored by a MACHINE, not by reading it.

THREE CATEGORIES, chosen because each fails for a different reason:

  factual / closed-form  A one-token answer with a unique correct value.  Fails
                         loudly if the lm_head or the final norm drifts.
  reasoning              Arithmetic whose answer is checkable by parsing a
                         number.  Fails if long-range state or the MoE router
                         degrades, and it fails MULTIPLICATIVELY -- one wrong
                         digit in the middle ruins the answer.
  code                   The generated source is EXTRACTED AND ACTUALLY RUN
                         against unit tests in a subprocess.  This is the only
                         item in the whole project that cannot be fooled by
                         plausible-looking output, and it is the category where
                         a subtly lobotomised model is most obviously caught:
                         code either passes its tests or it does not.

Both engines are greedy and get the identical prompt and the identical token
budget.  The engine and the oracle are compared to each other AND, where the
item has a ground truth, to that truth -- because two engines agreeing does not
mean either is right.

usage:
  tools/coherence.py --out ref/coherence.json
"""
import argparse
import json
import os
import pathlib
import re
import struct
import subprocess
import sys
import tempfile
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
PORT = int(os.environ.get("LK_ORACLE_PORT", "18099"))


def post(path, payload, timeout=3600):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path),
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def tok(t):  return post("/tokenize", {"content": t})["tokens"]
def detok(i): return post("/detokenize", {"tokens": i})["content"]


def engine_gen(binpath, ids, ngen):
    tmp = pathlib.Path("/tmp/lk_coh.bin")
    tmp.write_bytes(struct.pack("<%di" % len(ids), *ids))
    env = dict(os.environ)
    env.update({"LK_NGEN": str(ngen), "LK_PROMPT_BIN": str(tmp),
                "CUDA_VISIBLE_DEVICES": env.get("LK_GPU", "0")})
    env.pop("LK_BENCH", None)
    r = subprocess.run([str(binpath), "gen"], capture_output=True, text=True,
                       env=env, timeout=7200)
    if r.returncode != 0:
        raise RuntimeError("engine failed:\n" + r.stderr[-2000:])
    for line in r.stdout.splitlines():
        if line.startswith("out:"):
            return [int(x) for x in line[4:].split()]
    raise RuntimeError("no out: line\n" + r.stdout[-1500:])


def oracle_gen(text, ngen):
    d = post("/completion", {"prompt": text, "n_predict": ngen, "temperature": 0.0,
                             "cache_prompt": False, "seed": 1234,
                             "return_tokens": True})
    # `return_tokens` is not the default.  Without it llama-server returns an
    # EMPTY `tokens` list and the token-agreement column silently reports 0/0,
    # which reads as "no data" rather than as "the instrument never worked".
    return d["content"], (d.get("tokens") or [])


# ---------------------------------------------------------------------------
# CHAT FORMAT, AND WHY IT IS BAKED INTO THE PROMPT.
#
# This model is an instruct model with a visible reasoning block: asked a bare
# question it emits  thinking... and burns the whole token budget thinking.  Both
# engines must be given the identical token stream, and the engine has no chat
# template of its own, so the template is applied HERE and the resulting ids are
# handed to both.  Pre-filling an EMPTY think block (`` immediately after
# the assistant header) suppresses the reasoning block entirely and makes the
# model answer directly -- verified against the oracle before being written in.
# ---------------------------------------------------------------------------
def wrap(prompt):
    return ("<|im_start|>user\n" + prompt + "<|im_end|>\n"
            "<|im_start|>assistant\n\n")


# ---------------------------------------------------------------------------
# CODE EXECUTION.  Extraction is deliberately forgiving about the fence and
# deliberately strict about everything else: the code is run as-is, and a
# generation that does not even import fails.
# ---------------------------------------------------------------------------
def extract_code(text):
    m = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    if m:
        return max(m, key=len)
    # No fence: take everything from the first def/import to the end.
    m = re.search(r"^((?:from|import|def|class)\b.*)$", text, re.S | re.M)
    return m.group(1) if m else text


def run_code(code, tests, timeout=20):
    """Returns (passed, detail). Runs in a subprocess so a generated infinite
    loop or a syntax error cannot take the harness with it."""
    src = code + "\n\n" + tests + "\nprint('__OK__')\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, text=True,
                           timeout=timeout)
        ok = r.returncode == 0 and "__OK__" in r.stdout
        return ok, (r.stdout + r.stderr)[-300:]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    finally:
        os.unlink(path)


def num(text):
    """Last integer or float in the text -- models put the answer last."""
    m = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return m[-1] if m else None


P = []          # (category, prompt, ngen, checker)


def add(cat, prompt, ngen, check):
    P.append((cat, prompt, ngen, check))


# ---- factual: one unique answer, checkable by substring -------------------
# NO "Question: ... / Answer:" scaffolding.  MEASURED: appending that scaffolding
# pushes this model back into a  thinking block that consumes the entire token
# budget, and then BOTH engines fail -- which reads as a shared model defect and
# is actually the harness asking in a shape the chat template never produces.
# The bare question under the ChatML template answers directly.
for q, a in [("What is the capital of Australia?", "canberra"),
             ("What is the chemical symbol for gold?", "au"),
             ("Who wrote the play Romeo and Juliet?", "shakespeare"),
             ("What is the largest planet in our solar system?", "jupiter"),
             ("How many sides does a hexagon have?", "6")]:
    # Word boundaries, not plain substring: "au" is inside "because", and a
    # checker that passes for the wrong reason is worse than no checker.
    add("factual", q, 40,
        (lambda a: (lambda t: re.search(r"\b%s\b" % re.escape(a), t.lower()) is not None))(a))

# ---- reasoning: parse the number, compare to a computed truth -------------
for q, a in [("A train travels 60 miles per hour for 3 hours. How many miles does it travel?", 180),
             ("If 7 machines each produce 12 widgets per hour, how many widgets do they produce in 5 hours?", 420),
             ("A rectangle is 13 cm long and 7 cm wide. What is its area in square centimetres?", 91),
             ("Sarah has 48 apples. She gives away one quarter of them. How many apples does she have left?", 36),
             ("What is 17 multiplied by 23?", 391)]:
    add("reasoning", "%s Reply with only the number." % q, 64,
        (lambda a: (lambda t: num(t) is not None and abs(float(num(t)) - a) < 1e-6))(a))

# ---- code: extracted and EXECUTED ----------------------------------------
CODE = [
 ("Write a Python function `add(a, b)` that returns the sum of a and b. "
  "Output only the function inside a python code block.",
  "assert add(2,3)==5\nassert add(-1,1)==0\nassert add(0,0)==0"),
 ("Write a Python function `is_palindrome(s)` that returns True if the string s "
  "reads the same forwards and backwards, ignoring case. Output only the function "
  "inside a python code block.",
  "assert is_palindrome('Racecar')==True\nassert is_palindrome('hello')==False\n"
  "assert is_palindrome('')==True"),
 ("Write a Python function `fib(n)` returning the n-th Fibonacci number with "
  "fib(0)=0 and fib(1)=1. Output only the function inside a python code block.",
  "assert fib(0)==0\nassert fib(1)==1\nassert fib(10)==55\nassert fib(20)==6765"),
 ("Write a Python function `word_count(s)` that returns a dictionary mapping each "
  "lowercased word in s to how many times it appears. Split on whitespace. Output "
  "only the function inside a python code block.",
  "assert word_count('a b a')=={'a':2,'b':1}\nassert word_count('')=={}"),
 ("Write a Python function `flatten(lst)` that takes a list of lists and returns a "
  "single flat list. Output only the function inside a python code block.",
  "assert flatten([[1,2],[3]])==[1,2,3]\nassert flatten([])==[]\n"
  "assert flatten([[],[]])==[]"),
 ("Write a Python function `gcd(a, b)` returning the greatest common divisor of "
  "two positive integers. Output only the function inside a python code block.",
  "assert gcd(12,18)==6\nassert gcd(7,13)==1\nassert gcd(100,10)==10"),
]
for prompt, tests in CODE:
    add("code", prompt, 320, (lambda tests: (lambda t: run_code(extract_code(t), tests)))(tests))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=str(ROOT / "build" / "engine"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # THE ENGINE DOES NOT STOP AT EOS.  Given a fixed ngen it keeps generating
    # through <|im_end|> and <|endoftext|>, so the answer is correct and is then
    # followed by a degenerate repetition loop.  llama-server stops, so scoring
    # the raw text compares one clean answer against one clean answer plus
    # garbage and reports a token agreement that means nothing.  Both sides are
    # therefore CLIPPED AT THE FIRST EOS before scoring OR comparing; the raw
    # text is still recorded in the JSON so the loop stays visible.
    eos_ids = []
    for e in ("<|im_end|>", "<|endoftext|>"):
        t = tok(e)
        if len(t) == 1:
            eos_ids.append(t[0])
    if not eos_ids:
        print("WARNING: could not resolve EOS ids; agreement will be understated")

    def clip_ids(v):
        for i, x in enumerate(v):
            if x in eos_ids:
                return v[:i]
        return v

    def clip_txt(s):
        cut = len(s)
        for e in ("<|im_end|>", "<|endoftext|>"):
            j = s.find(e)
            if j >= 0:
                cut = min(cut, j)
        return s[:cut]

    rows = []
    print("%-10s %-6s %-6s %-9s %s" % ("category", "eng", "orc", "agree", "detail"))
    print("-" * 78)
    for cat, prompt, ngen, check in P:
        text = wrap(prompt)
        ids = tok(text)
        etoks = engine_gen(a.engine, ids, ngen)
        etxt = detok(etoks)
        otxt, otoks = oracle_gen(text, ngen)

        ec, oc = clip_txt(etxt), clip_txt(otxt)
        e_res = check(ec)
        o_res = check(oc)
        e_ok = e_res[0] if isinstance(e_res, tuple) else bool(e_res)
        o_ok = o_res[0] if isinstance(o_res, tuple) else bool(o_res)
        # Token agreement over the CLIPPED sequences.  This is the number that
        # says whether the two implementations are the same model; the raw one
        # mostly measures that llama-server honours EOS and we do not.
        ecl, ocl = clip_ids(etoks), clip_ids(otoks)
        k = min(len(ecl), len(ocl))
        agree = sum(1 for i in range(k) if ecl[i] == ocl[i])
        firstdiff = next((i for i in range(k) if ecl[i] != ocl[i]), k)

        detail = ""
        if isinstance(e_res, tuple) and not e_ok:
            detail = e_res[1].replace("\n", " ")[:60]
        rows.append(dict(category=cat, prompt=prompt, ngen=ngen,
                         engine_ok=e_ok, oracle_ok=o_ok,
                         engine_text=etxt[:400], oracle_text=otxt[:400],
                         agree=agree, common=k, first_diff=firstdiff, detail=detail))
        print("%-10s %-6s %-6s %2d/%2d  %s" %
              (cat, "PASS" if e_ok else "FAIL", "PASS" if o_ok else "FAIL",
               agree, k, detail))
        sys.stdout.flush()
        pathlib.Path(a.out).write_text(json.dumps(rows, indent=1))

    print()
    for cat in ("factual", "reasoning", "code"):
        r = [x for x in rows if x["category"] == cat]
        if not r:
            continue
        e = sum(x["engine_ok"] for x in r); o = sum(x["oracle_ok"] for x in r)
        ag = sum(x["agree"] for x in r); cm = sum(x["common"] for x in r)
        print("%-10s liquid-kernel %d/%d   llama.cpp %d/%d   token agreement %d/%d = %.1f%%"
              % (cat, e, len(r), o, len(r), ag, cm, 100.0 * ag / max(cm, 1)))
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
