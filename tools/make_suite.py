#!/usr/bin/env python3
"""Freeze the C1 parity suite as raw token IDs.

The tokenizer is used exactly ONCE here, at suite-creation time.  Every later
correctness check consumes frozen token IDs, so a tokenizer discrepancy can
never masquerade as a kernel bug.
"""
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from oracle import Oracle

REF = pathlib.Path(os.environ.get("LK_REF", pathlib.Path(__file__).resolve().parent.parent / "ref"))

# Each case is either a raw string prompt or a chat message list.
CASES = [
    ("factual",   [{"role": "user", "content": "What is the capital of France?"}]),
    ("math",      [{"role": "user", "content": "A train travels 60 km in 45 minutes. What is its average speed in km/h? Show your reasoning."}]),
    ("code",      [{"role": "user", "content": "Write a Python function that returns the nth Fibonacci number using iterative dynamic programming. Include a docstring."}]),
    ("agentic",   [{"role": "user", "content": "List the files in the current directory and then read the README."}]),
    ("reasoning", [{"role": "user", "content": "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies? Explain."}]),
    ("prose",     [{"role": "user", "content": "Describe the water cycle in three sentences."}]),
    ("longctx",   [{"role": "user", "content": "Here is a passage:\n\n" + ("The quick brown fox jumps over the lazy dog. " * 40) + "\n\nHow many times does the word 'fox' appear in the passage above?"}]),
]


def main():
    o = Oracle()
    print("oracle health:", o.health())
    REF.mkdir(parents=True, exist_ok=True)
    suite = []
    for name, messages in CASES:
        text = o.apply_template(messages)
        if text is None:
            text = "\n".join(m["content"] for m in messages)
        ids = o.tokenize(text)
        suite.append({"name": name, "messages": messages, "prompt_text": text, "prompt_ids": ids})
        print(f"  {name:10s} {len(ids):5d} tokens")
    out = REF / "parity_suite.json"
    out.write_text(json.dumps({"cases": suite}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
