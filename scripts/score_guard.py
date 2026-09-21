#!/usr/bin/env python3
"""Score the guarded arm against the recorded unguarded arm with the canonical
instruments (functions imported from amplification_test.py / register_check.py
as committed in 99f99d1; AGRI list from register_robust.py)."""
import json, re, sys, statistics as st
sys.path.insert(0, "/workspace/driftenginelite/Drift-engine")
sys.path.insert(0, "/workspace/driftenginelite/Drift-engine/scripts")
from amplification_test import anaphora, int_pct, fisher, TA
from register_check import deloop
from engine.lexicon import PHYSICAL_VERBS

AGRI = {"plowed", "ploughed", "plows", "plow", "sowed", "sows", "sow",
        "harvested", "harvests", "harvest", "reaped", "reaps", "reap",
        "threshed", "harrowed", "seeded", "mowed", "mows", "mow", "scythed",
        "hammered", "hammers", "nailed", "nails", "sawed", "saws",
        "chopped", "chops", "dug", "digs", "dig", "loaded", "loads",
        "unloaded", "unloads", "hitched", "saddled", "mounted", "dismounted",
        "poured", "pours", "carried", "carries", "dragged", "drags",
        "buried", "buries", "bury", "filled", "fills", "fill"} & PHYSICAL_VERBS
FP = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}

CORPUS = "/workspace/final_training_corpus_v2_1_latest.json"
ARMS = {"unguarded": "/workspace/gen_v6_cap2560.json",
        "guarded": sys.argv[1] if len(sys.argv) > 1 else "/workspace/gen_v6_guard.json",
        "base": "/workspace/gen_base_control.json"}

corpus = json.load(open(CORPUS))
longs = [e["messages"][2]["content"] for e in corpus
         if e["messages"][1]["content"].strip() == "Write the next chapter."]
c_an = [anaphora(t) for t in longs]
C_P90 = sorted(r[0] for r in c_an)[int(0.9 * (len(c_an) - 1))]
C_MAX = max(r[0] for r in c_an); C_MAXRUN = max(r[1] for r in c_an)

def pctile(v, w):
    ref = [int_pct(" ".join(t.split()[:max(30, w)])) for t in longs]
    return 100.0 * (sum(r < v for r in ref) + 0.5 * sum(r == v for r in ref)) / len(ref)

def toks(t): return re.findall(r"[a-z']+", t.lower())

rows = {}
for arm, path in ARMS.items():
    d = json.load(open(path))
    print(f"\n=== {arm}  ({path})")
    print(f"{'i':>2} {'fin':>3} {'raw w':>6} {'delp w':>6} {'span ch':>7} {'anaph%':>7} {'run':>4} "
          f"{'int%':>5} {'int pct@W':>9} {'agri':>4} {'1p/1k':>6}")
    rows[arm] = []
    for r in d["results"]:
        t, span = deloop(r["text"])
        w = len(t.split()); tk = toks(t)
        a, run, ns = anaphora(t)
        ip = int_pct(t)
        row = dict(i=r["i"], fin=r["finish_reason"], raw=r["word_count"], w=w, span=span,
                   an=a, run=run, ip=ip, pc=pctile(ip, w),
                   agri=sum(x in AGRI for x in tk), ntok=len(tk),
                   fp=1000.0 * sum(x in FP for x in tk) / max(1, len(tk)),
                   looped=(r["finish_reason"] == "CAP" or span >= 200))
        rows[arm].append(row)
        print(f"{row['i']:>2} {row['fin']:>3} {row['raw']:>6} {w:>6} {span:>7} {a:>7.1f} {run:>4} "
              f"{ip:>5.1f} {row['pc']:>9.1f} {row['agri']:>4} {row['fp']:>6.1f}")

print(f"\ncorpus long-form anaphora: p90 {C_P90:.1f}  max {C_MAX:.1f}  max run {C_MAXRUN}")
print(f"\n{'arm':10} {'n':>2} {'EOS':>4} {'loop':>4} {'med raw w':>9} {'med delp w':>10} {'EOS w range':>12} "
      f"{'med an%':>7} {'mean an%':>8} {'>cMAX':>5} {'run>c':>5} {'med int%':>8} {'med pct':>7} {'>p90':>4} {'agri/1k':>7}")
for arm, rs in rows.items():
    eosw = [r["raw"] for r in rs if r["fin"] == "EOS"]
    print(f"{arm:10} {len(rs):>2} {sum(r['fin']=='EOS' for r in rs):>4} {sum(r['looped'] for r in rs):>4} "
          f"{st.median(r['raw'] for r in rs):>9.0f} {st.median(r['w'] for r in rs):>10.0f} "
          f"{(str(min(eosw))+'-'+str(max(eosw))) if eosw else '-':>12} "
          f"{st.median(r['an'] for r in rs):>7.1f} {st.mean(r['an'] for r in rs):>8.1f} "
          f"{sum(r['an']>C_MAX for r in rs):>5} {sum(r['run']>C_MAXRUN for r in rs):>5} "
          f"{st.median(r['ip'] for r in rs):>8.1f} {st.median(r['pc'] for r in rs):>7.1f} "
          f"{sum(r['pc']>90 for r in rs):>4} "
          f"{1000.0*sum(r['agri'] for r in rs)/sum(r['ntok'] for r in rs):>7.2f}")

u, g = rows["unguarded"], rows["guarded"]
n = len(g)
def fz(name, ku, kg):
    print(f"{name:34} unguarded {ku}/{len(u)}  guarded {kg}/{n}  Fisher p = {fisher(ku, len(u)-ku, kg, n-kg):.4f}")
fz("loop (CAP or >=200ch repeat)", sum(r["looped"] for r in u), sum(r["looped"] for r in g))
fz("EOS", sum(r["fin"]=="EOS" for r in u), sum(r["fin"]=="EOS" for r in g))
fz("anaphora > corpus MAX", sum(r["an"]>C_MAX for r in u), sum(r["an"]>C_MAX for r in g))
fz("EOS and 250-450 w (prediction band)", sum(r["fin"]=="EOS" and 250<=r["raw"]<=450 for r in u),
   sum(r["fin"]=="EOS" and 250<=r["raw"]<=450 for r in g))
ub = {r["i"]: r for r in u}
print("\nprediction check on the seeds that cap-looped unguarded:")
for r in g:
    if ub.get(r["i"], {}).get("fin") == "CAP":
        ok = r["fin"] == "EOS" and 250 <= r["raw"] <= 450
        print(f"  seed i={r['i']}: unguarded CAP {ub[r['i']]['raw']}w -> guarded {r['fin']} {r['raw']}w "
              f"anaph {r['an']:.1f}% run {r['run']}  {'IN BAND' if ok else 'out of band'}")
