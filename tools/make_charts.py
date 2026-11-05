#!/usr/bin/env python3
"""Generate the README charts from the raw bench JSON in bench/.

Stdlib only.  Writes light and dark SVGs into docs/media/:

    prefill_time_{light,dark}.svg    grouped bars, seconds to prefill a prompt
    decode_speed_{light,dark}.svg    decode tok/s against context length

    usage: python3 tools/make_charts.py

Data files (all measured 2026-10-22, same GPU, same corpus ids):
    bench/engine_v2_2026-10-22.json      this engine
    bench/llama_tuned_2026-10-22.json    llama.cpp b10342, -ub 4096 -b 4096
    bench/llama_default_2026-10-22.json  llama.cpp b10342, default flags

Colours are the Okabe-Ito colourblind-safe palette.
"""
import json
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "media"
OUT.mkdir(parents=True, exist_ok=True)

ENGINE = json.loads((ROOT / "bench" / "engine_v2_2026-10-22.json").read_text())
TUNED = json.loads((ROOT / "bench" / "llama_tuned_2026-10-22.json").read_text())
DEFAULT = json.loads((ROOT / "bench" / "llama_default_2026-10-22.json").read_text())

C_ENGINE = "#0072B2"   # blue
C_TUNED = "#E69F00"    # orange
C_DEFAULT = "#999999"  # grey

THEMES = {
    "light": dict(bg="#ffffff", fg="#1a1a1a", grid="#d8d8d8", dim="#666666"),
    "dark": dict(bg="#0d1117", fg="#e6edf3", grid="#30363d", dim="#8b949e"),
}

FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;")


def svg_doc(w, h, th):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" font-family="{FONT}">'
            f'<rect width="{w}" height="{h}" fill="{th["bg"]}"/>')


def fmt_s(v):
    return f"{v:.3f} s" if v < 1 else f"{v:.2f} s"


def kfmt(n):
    return {64: "64", 256: "256", 1024: "1k", 4096: "4k", 8192: "8k",
            16384: "16k", 32768: "32k", 65536: "64k",
            131040: "131k", 131072: "131k"}.get(n, str(n))


# ---------------------------------------------------------------------------
# Chart A: prefill time, grouped horizontal bars, log x scale.
# ---------------------------------------------------------------------------
def prefill_chart(theme_name):
    th = THEMES[theme_name]
    groups = [4096, 32768, 131040]
    series = [
        ("this engine", C_ENGINE, {r["ctx"]: r["prefill_s"] for r in ENGINE}),
        ("llama.cpp tuned", C_TUNED, {r["ctx"]: r["prefill_s"] for r in TUNED}),
        ("llama.cpp default", C_DEFAULT, {r["ctx"]: r["prefill_s"] for r in DEFAULT}),
    ]
    W, H = 860, 470
    L, R = 116, 120        # left margin (group labels), right (value labels)
    TOP = 74
    x0, x1 = math.log10(0.1), math.log10(40.0)

    def X(v):
        return L + (math.log10(v) - x0) / (x1 - x0) * (W - L - R)

    s = svg_doc(W, H, th)
    s += (f'<text x="{L}" y="30" font-size="17" font-weight="600" fill="{th["fg"]}">'
          f'Prompt processing time &#8212; seconds, lower is better</text>')
    # legend
    lx = L
    for name, color, _ in series:
        s += (f'<rect x="{lx}" y="44" width="13" height="13" rx="2" fill="{color}"/>'
              f'<text x="{lx + 19}" y="55" font-size="13" fill="{th["fg"]}">{esc(name)}</text>')
        lx += 19 + len(name) * 7.4 + 28

    bar_h, bar_gap, group_gap = 26, 5, 26
    y = TOP
    for g in groups:
        s += (f'<text x="{L - 12}" y="{y + (3 * bar_h + 2 * bar_gap) / 2 + 5}" '
              f'font-size="15" font-weight="600" text-anchor="end" fill="{th["fg"]}">'
              f'{kfmt(g)} tokens</text>')
        for name, color, data in series:
            v = data[g]
            w = X(v) - L
            s += (f'<rect x="{L}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="3" fill="{color}"/>')
            s += (f'<text x="{L + w + 8:.1f}" y="{y + bar_h / 2 + 5}" font-size="13.5" '
                  f'fill="{th["fg"]}">{fmt_s(v)}</text>')
            y += bar_h + bar_gap
        y += group_gap

    # gridlines + axis
    for tick in (0.1, 1, 10):
        tx = X(tick)
        s += (f'<line x1="{tx:.1f}" y1="{TOP - 6}" x2="{tx:.1f}" y2="{y - group_gap + 4}" '
              f'stroke="{th["grid"]}" stroke-width="1"/>')
        s += (f'<text x="{tx:.1f}" y="{y + 14}" font-size="12" text-anchor="middle" '
              f'fill="{th["dim"]}">{tick:g} s</text>')
    eng = {r["ctx"]: r["prefill_s"] for r in ENGINE}
    tnd = {r["ctx"]: r["prefill_s"] for r in TUNED}
    ratios = " / ".join(f"{tnd[g] / eng[g]:.2f}x at {kfmt(g)}" for g in groups)
    s += (f'<text x="{L}" y="{y + 38}" font-size="12.5" fill="{th["dim"]}">'
          f'Speedup over llama.cpp tuned: {ratios}</text>')
    s += "</svg>"
    return s


# ---------------------------------------------------------------------------
# Chart B: decode tok/s vs context, two lines.
# ---------------------------------------------------------------------------
def decode_chart(theme_name):
    th = THEMES[theme_name]
    W, H = 860, 420
    L, R, T, B = 70, 150, 66, 46
    ctxs = [r["ctx"] for r in ENGINE]
    x0, x1 = math.log10(min(ctxs)) - 0.12, math.log10(max(ctxs)) + 0.05
    y0, y1 = 0.0, 500.0

    def X(v):
        return L + (math.log10(v) - x0) / (x1 - x0) * (W - L - R)

    def Y(v):
        return T + (1 - (v - y0) / (y1 - y0)) * (H - T - B)

    s = svg_doc(W, H, th)
    s += (f'<text x="{L}" y="30" font-size="17" font-weight="600" fill="{th["fg"]}">'
          f'Decode speed &#8212; tokens per second, higher is better</text>')

    for gy in (100, 200, 300, 400, 500):
        s += (f'<line x1="{L}" y1="{Y(gy):.1f}" x2="{W - R}" y2="{Y(gy):.1f}" '
              f'stroke="{th["grid"]}" stroke-width="1"/>')
        s += (f'<text x="{L - 8}" y="{Y(gy) + 4:.1f}" font-size="12" text-anchor="end" '
              f'fill="{th["dim"]}">{gy}</text>')
    for c in (64, 256, 1024, 4096, 16384, 65536, 131040):
        s += (f'<line x1="{X(c):.1f}" y1="{T}" x2="{X(c):.1f}" y2="{H - B}" '
              f'stroke="{th["grid"]}" stroke-width="1" stroke-dasharray="2 3"/>')
        s += (f'<text x="{X(c):.1f}" y="{H - B + 18}" font-size="12" text-anchor="middle" '
              f'fill="{th["dim"]}">{kfmt(c)}</text>')
    s += (f'<text x="{(L + W - R) / 2}" y="{H - 8}" font-size="12.5" text-anchor="middle" '
          f'fill="{th["dim"]}">context length (tokens, log scale)</text>')

    for data, color, label, above in ((ENGINE, C_ENGINE, "this engine", True),
                                    (TUNED, C_TUNED, "llama.cpp tuned", False)):
        pts = " ".join(f"{X(r['ctx']):.1f},{Y(r['decode_tps']):.1f}" for r in data)
        s += (f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for r in data:
            s += (f'<circle cx="{X(r["ctx"]):.1f}" cy="{Y(r["decode_tps"]):.1f}" r="4" '
                  f'fill="{color}"/>')
        last = data[-1]
        ly = Y(last["decode_tps"])
        dy = (-16, -2) if above else (14, 28)
        s += (f'<text x="{X(last["ctx"]) + 12:.1f}" y="{ly + dy[0]:.1f}" '
              f'font-size="13.5" font-weight="600" fill="{color}">{esc(label)}</text>')
        s += (f'<text x="{X(last["ctx"]) + 12:.1f}" y="{ly + dy[1]:.1f}" '
              f'font-size="12" fill="{th["dim"]}">{last["decode_tps"]:.0f} tok/s</text>')

    e64 = ENGINE[0]["decode_tps"]
    t64 = TUNED[0]["decode_tps"]
    s += (f'<text x="{L}" y="{T - 8}" font-size="12.5" fill="{th["dim"]}">'
          f'{e64 / t64:.2f}x at short context, {ENGINE[-1]["decode_tps"] / TUNED[-1]["decode_tps"]:.2f}x at 131k</text>')
    s += "</svg>"
    return s


for name in ("light", "dark"):
    (OUT / f"prefill_time_{name}.svg").write_text(prefill_chart(name))
    (OUT / f"decode_speed_{name}.svg").write_text(decode_chart(name))
    print(f"wrote docs/media/prefill_time_{name}.svg docs/media/decode_speed_{name}.svg")
