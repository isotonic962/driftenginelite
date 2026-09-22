#!/usr/bin/env python3
"""Analyze eval/boundary_entropy.json against logs/prediction_boundary_entropy.txt."""
import json, statistics as st, itertools, sys
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "/workspace/driftenginelite/eval/boundary_entropy.json"))
rows = []
for r in d["results"]:
    for b, f, g in zip(r["base"], r["F"], r["G"]):
        rows.append(dict(text=r["text"], t=b["t"], w=r["words_at"][str(b["t"])], Hb=b["H"], HF=f["H"], HG=g["H"],
                         tb=b["top1"], tF=f["top1"], tG=g["top1"], lb=b["true_lp"], lF=f["true_lp"], lG=g["true_lp"]))
n = len(rows)
def sign_p(x):
    k = sum(v < 0 for v in x); m = sum(v != 0 for v in x)
    from math import comb
    p = sum(comb(m, i) for i in range(0, min(k, m - k) + 1)) / 2 ** m * 2
    return min(1.0, p)
print(f"{n} boundaries over {len(d['results'])} texts")
print(f"{'':18} {'base':>7} {'F':>7} {'G':>7}   {'F-base':>8} {'G-base':>8} {'G-F':>8}")
for lab, kb, kf, kg in (("entropy H (nats)", "Hb", "HF", "HG"), ("top-1 prob", "tb", "tF", "tG"), ("logp(true opening)", "lb", "lF", "lG")):
    mb, mf, mg = (st.median(r[k] for r in rows) for k in (kb, kf, kg))
    dF = [r[kf] - r[kb] for r in rows]; dG = [r[kg] - r[kb] for r in rows]; dGF = [r[kg] - r[kf] for r in rows]
    print(f"{lab:18} {mb:7.3f} {mf:7.3f} {mg:7.3f}   {st.median(dF):+8.3f} {st.median(dG):+8.3f} {st.median(dGF):+8.3f}   "
          f"share F<base {sum(v<0 for v in dF)/n:.2f} (sign p={sign_p(dF):.1e})  G<base {sum(v<0 for v in dG)/n:.2f} (p={sign_p(dG):.1e})  G<F {sum(v<0 for v in dGF)/n:.2f} (p={sign_p(dGF):.1e})")
print("\nby prefix length (words):")
for lo, hi in ((0, 250), (250, 500), (500, 750), (750, 10000)):
    sub = [r for r in rows if lo <= r["w"] < hi]
    if not sub: continue
    dF = [r["HF"] - r["Hb"] for r in sub]; dG = [r["HG"] - r["Hb"] for r in sub]
    print(f"  {lo:>4}-{hi if hi < 10000 else '':<5} n={len(sub):3}  base H {st.median(r['Hb'] for r in sub):.3f}  F-base {st.median(dF):+.3f}  G-base {st.median(dG):+.3f}  share F<base {sum(v<0 for v in dF)/len(sub):.2f}  G<base {sum(v<0 for v in dG)/len(sub):.2f}")
print("\nper text (median H):")
for r in d["results"]:
    if not r["base"]: continue
    m = lambda a: st.median(x["H"] for x in r[a])
    print(f"  {r['text']:6} n={r['n_boundaries']:3}  base {m('base'):.2f}  F {m('F'):.2f}  G {m('G'):.2f}")
