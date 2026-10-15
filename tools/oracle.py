#!/usr/bin/env python3
"""Client for the llama.cpp oracle.

The oracle is the ONLY valid correctness reference for this checkpoint: the
QAD GGUF is a Quantization-Aware-Distillation retrain whose weights do not
match the public LiquidAI/LFM2.5-24B-A2B BF16 checkpoint (verified: token
embedding correlation -0.036).

`greedy()` deliberately disables every sampler so the chain reduces to a pure
argmax.  llama.cpp otherwise applies repeat/dry/top-k penalties *before* the
greedy pick, which would break exact parity comparisons.
"""
import json
import urllib.request

HOST = "127.0.0.1"
PORT = 18099


class Oracle:
    def __init__(self, host=HOST, port=PORT, timeout=1800):
        self.base = f"http://{host}:{port}"
        self.timeout = timeout

    def _req(self, path, payload=None):
        url = self.base + path
        if payload is None:
            req = urllib.request.Request(url)
        else:
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def health(self):
        return self._req("/health")

    def tokenize(self, text):
        return self._req("/tokenize", {"content": text})["tokens"]

    def detokenize(self, ids):
        return self._req("/detokenize", {"tokens": list(ids)})["content"]

    def apply_template(self, messages):
        """Returns the templated prompt string, or None if unsupported."""
        try:
            return self._req("/apply-template", {"messages": messages})["prompt"]
        except Exception:
            return None

    def greedy(self, prompt_ids, n_predict=128, n_probs=0, seed=1234):
        """Pure-argmax completion. Returns the raw server response dict."""
        p = {
            "prompt": list(prompt_ids),
            "n_predict": int(n_predict),
            "temperature": 0.0,
            "repeat_penalty": 1.0,   # disable: must not perturb argmax
            "dry_multiplier": 0.0,   # disable
            "top_k": 0,              # disable
            "top_p": 1.0,            # disable
            "min_p": 0.0,            # disable
            "xtc_probability": 0.0,  # disable
            "seed": int(seed),
            "return_tokens": True,
            "cache_prompt": False,
        }
        if n_probs:
            p["n_probs"] = int(n_probs)
            p["post_sampling_probs"] = False   # raw logits -> logprobs
        return self._req("/completion", p)

    @staticmethod
    def top_logprobs(resp, k=None):
        """Extract [(token_id, logprob), ...] per generated position."""
        out = []
        for pos in resp.get("completion_probabilities", []):
            tl = pos.get("top_logprobs", [])
            if k:
                tl = tl[:k]
            out.append([(e["id"], e["logprob"]) for e in tl])
        return out


if __name__ == "__main__":
    import sys
    o = Oracle()
    print("health:", o.health())
    ids = o.tokenize("The capital of France is")
    print("tokenize:", ids)
    r = o.greedy(ids, n_predict=8, n_probs=3)
    print("tokens:", r.get("tokens"))
    print("content:", repr(r.get("content")))
    t = r.get("timings", {})
    print("decode tok/s: %.2f" % t.get("predicted_per_second", 0))
    print("top_logprobs:", o.top_logprobs(r, 3)[:2])
