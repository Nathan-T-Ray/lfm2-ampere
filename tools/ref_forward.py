#!/usr/bin/env python3
"""Independent NumPy forward pass over lfm25.pack, for a SINGLE token.

Purpose: separate 'our CUDA does not do what we think' from 'we misunderstand
llama.cpp'.  It re-derives every layer from the pack with the same stated
semantics as src/kernels.cu; if the two agree layer by layer, the CUDA is
faithful and any remaining gap to llama.cpp is a semantic misunderstanding.

Attention is exact at one token: softmax over a single element is 1, so the
attention output is simply V for the corresponding KV head.
"""
import json, os, pathlib, struct, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gguf as G

MAN = os.environ["LK_MANIFEST"]
PACK = os.environ["LK_PACK"]
H, L, NH, HD, NKV, NE, TOPK = 2048, 40, 32, 64, 8, 64, 4
MOE_I, DENSE_I, VOCAB, EPS = 1536, 11776, 128000, 1e-5

_man = json.loads(pathlib.Path(MAN).read_text())
_ent = {e["name"]: e for e in _man["tensors"]}
_f = open(PACK, "rb")


def plane(name, p):
    e = _ent[name]
    o = e["planes"][p]["offset"]; s = e["planes"][p]["size"]
    _f.seek(o); return _f.read(s)


def q4(name, rows=None):
    """Dequantise a Q4_0 tensor to (N,K) float32; rows=(start,count) selects rows."""
    e = _ent[name]
    bpr = e["blocks_per_row"]; N = e["rows"]
    qs = np.frombuffer(plane(name, "qs"), np.uint8).reshape(N, bpr, 16)
    dd = np.frombuffer(plane(name, "d"), np.uint16).view(np.float16).astype(np.float32).reshape(N, bpr)
    if rows is not None:
        s, c = rows
        qs = qs[s:s + c]; dd = dd[s:s + c]; N = c
    lo = (qs & 0x0F).astype(np.float32) - 8
    hi = (qs >> 4).astype(np.float32) - 8
    w = np.concatenate([lo, hi], axis=2) * dd[:, :, None]
    return w.reshape(N, e["cols"])


def q6(name):
    e = _ent[name]
    nb = e["blocks"]
    raw = np.concatenate([
        np.frombuffer(plane(name, "ql"), np.uint8).reshape(nb, 128),
        np.frombuffer(plane(name, "qh"), np.uint8).reshape(nb, 64),
        np.frombuffer(plane(name, "scales"), np.uint8).reshape(nb, 16),
        np.frombuffer(plane(name, "d"), np.uint8).reshape(nb, 2)], axis=1)
    return G.dequantize(np.ascontiguousarray(raw), 14, e["nelem"]).reshape(e["rows"], e["cols"])


def q8_1(x):
    """ggml-cuda convention, per 32-element block: ds = (amax/127, RAW sum).

    CRITICAL: ggml stores ds as a half2 (make_half2(d, sum)), so BOTH d and the
    raw sum are rounded to fp16 before use.  Using full fp32 here instead makes
    the reference differ from the engine by ~1e-4 in every GEMV output, which
    then amplifies through 40 layers and looks exactly like an engine bug.
    Round to fp16 to stay comparable.
    """
    xb = np.asarray(x, np.float32).reshape(-1, 32)
    amax = np.max(np.abs(xb), axis=1)
    d = (amax / 127.0).astype(np.float32)
    safe = np.where(d > 0, d, np.float32(1.0))
    q = np.where(d[:, None] > 0, np.round(xb / safe[:, None]), 0.0).astype(np.int8)
    d16 = d.astype(np.float16).astype(np.float32)
    s16 = np.sum(xb, axis=1).astype(np.float32).astype(np.float16).astype(np.float32)
    return q, d16, s16


def q4_mv_q8(name, x, rows=None):
    """Q4_0 x Q8_1 matvec, reproducing dot_q4_0_q8_1 exactly:
           y[r] = sum_k d4[r,k] * (d8[k]*sumi[r,k] - 8*sum_raw[k])
    where sumi uses the raw 0..15 nibbles.  This is what makes the reference
    comparable with the CUDA engine (an exact-fp32 activation path would differ
    by the ~2% Q8_1 quantisation error and swamp any real bug)."""
    e = _ent[name]
    bpr = e["blocks_per_row"]; N = e["rows"]
    qs = np.frombuffer(plane(name, "qs"), np.uint8).reshape(N, bpr, 16)
    dd = np.frombuffer(plane(name, "d"), np.uint16).view(np.float16).astype(np.float32).reshape(N, bpr)
    if rows is not None:
        s0, c = rows
        qs = qs[s0:s0 + c]; dd = dd[s0:s0 + c]; N = c
    q, d8, rawsum = q8_1(x)                       # (K/32, 32), (K/32,), (K/32,)
    lo = (qs & 0x0F).astype(np.int32)
    hi = (qs >> 4).astype(np.int32)
    qb = q.astype(np.int32)                       # (bpr, 32)
    # lo/hi are (N,bpr,16); the 32 q8 lanes split into elements 0..15 (which
    # pair with the low nibbles) and 16..31 (high nibbles).
    sumi = np.einsum('nbj,bj->nb', lo, qb[:, :16]) + np.einsum('nbj,bj->nb', hi, qb[:, 16:])
    y = np.sum(dd * (d8[None, :] * sumi.astype(np.float32) - 8.0 * rawsum[None, :]), axis=1)
    return y.astype(np.float32)


def f32(name):
    """F32 tensor, shaped (rows, cols).  1-D norms stay flat."""
    e = _ent[name]
    a = np.frombuffer(plane(name, "data"), np.float32).copy()
    return a if e["rows"] == 1 else a.reshape(e["rows"], e["cols"])


def rmsnorm(x, w):
    return (x / np.sqrt(np.mean(x.astype(np.float64) ** 2) + EPS) * w).astype(np.float32)


def rope(v):
    """NEOX half-split at position 0 -> identity, kept explicit for clarity."""
    return v


def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v.astype(np.float64)))


def main():
    token = int(sys.argv[1]) if len(sys.argv) > 1 else 597
    cubes = np.fromfile("/tmp/cuda_layers.bin", dtype=np.float32).reshape(-1, H)

    emb = q6("token_embd.weight")
    x = emb[token].copy()
    snaps = [x.copy()]
    conv_hist = {}
    kc, vc = {}, {}

    for il in range(L):
        pre = f"blk.{il}."
        h = rmsnorm(x, f32(pre + "attn_norm.weight"))
        if il % 4 != 2:                                   # ---- ShortConv ----
            bcx = q4_mv_q8(pre + "shortconv.in_proj.weight", h)
            B, C, X = bcx[:H], bcx[H:2 * H], bcx[2 * H:]
            # conv.weight is [3, 2048] in ggml order: dim0 = tap is FASTEST, so
            # element (tap, ch) lives at flat index tap + ch*3.  Hence the taps
            # are the three strided slices below.
            cwf = f32(pre + "shortconv.conv.weight").reshape(-1)
            cw0, cw1, cw2 = cwf[0::3], cwf[1::3], cwf[2::3]
            h0, h1 = conv_hist.get(il, (np.zeros(H, np.float32), np.zeros(H, np.float32)))
            bx = (B * X).astype(np.float32)
            conv = h0 * cw0 + h1 * cw1 + bx * cw2
            conv_hist[il] = (h1, bx)
            y = (C * conv).astype(np.float32)
            x = (x + q4_mv_q8(pre + "shortconv.out_proj.weight", y)).astype(np.float32)
        else:                                             # ---- GQA ----
            q = q4_mv_q8(pre + "attn_q.weight", h)
            k = q4_mv_q8(pre + "attn_k.weight", h)
            v = q4_mv_q8(pre + "attn_v.weight", h)
            qn, kn = f32(pre + "attn_q_norm.weight"), f32(pre + "attn_k_norm.weight")
            q = np.concatenate([rmsnorm(q[i * HD:(i + 1) * HD], qn) for i in range(NH)])
            k = np.concatenate([rmsnorm(k[i * HD:(i + 1) * HD], kn) for i in range(NKV)])
            out = np.empty(H, np.float32)
            for hd in range(NH):                          # single token -> out = V
                out[hd * HD:(hd + 1) * HD] = v[(hd // (NH // NKV)) * HD:(hd // (NH // NKV) + 1) * HD]
            x = (x + q4_mv_q8(pre + "attn_output.weight", out)).astype(np.float32)

        h2 = rmsnorm(x, f32(pre + "ffn_norm.weight"))
        if il < 2:                                        # ---- dense FFN ----
            g = q4_mv_q8(pre + "ffn_gate.weight", h2)
            u = q4_mv_q8(pre + "ffn_up.weight", h2)
            inter = (g / (1.0 + np.exp(-g.astype(np.float64))) * u).astype(np.float32)
            x = (x + q4_mv_q8(pre + "ffn_down.weight", inter)).astype(np.float32)
        else:                                             # ---- MoE ----
            lg = f32(pre + "ffn_gate_inp.weight") @ h2    # [64,2048] @ [2048]
            pr = sigmoid(lg)
            sel = pr + f32(pre + "exp_probs_b.bias")
            idx = np.argsort(-sel, kind="stable")[:TOPK]
            w = pr[idx]
            ws = max(w.sum(), 6.103515625e-5)
            w = w / ws
            acc = np.zeros(H, np.float32)
            for kk, ex in enumerate(idx):
                base = ex * MOE_I
                g = q4_mv_q8(pre + "ffn_gate_exps.weight", h2, (base, MOE_I))
                u = q4_mv_q8(pre + "ffn_up_exps.weight",   h2, (base, MOE_I))
                inter = (g / (1.0 + np.exp(-g.astype(np.float64))) * u).astype(np.float32)
                dn = q4_mv_q8(pre + "ffn_down_exps.weight", inter, (ex * H, H))
                acc += np.float32(w[kk]) * dn
            x = (x + acc).astype(np.float32)
        snaps.append(x.copy())
        print(f"  layer {il:2d} done", flush=True)

    snaps = np.array(snaps)
    np.save("/tmp/ref_layers.npy", snaps)

    print(f"\n{'idx':>4} {'mean|d|':>10} {'max|d|':>10} {'rms cuda':>10} {'rms ref':>10}")
    for i in range(min(len(snaps), len(cubes))):
        d = np.abs(snaps[i] - cubes[i])
        rc = float(np.sqrt((cubes[i].astype(np.float64) ** 2).mean()))
        rr = float(np.sqrt((snaps[i].astype(np.float64) ** 2).mean()))
        tag = "  <-- residual BEFORE layer" if i < L else "  <-- final"
        print(f"{i:>4} {d.mean():>10.3e} {d.max():>10.3e} {rc:>10.5f} {rr:>10.5f}{tag}")


if __name__ == "__main__":
    main()
