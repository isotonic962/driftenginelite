"""Robustness pass for OPEN 1b. Two threats to the two surviving findings.

THREAT 1 -- zero inflation. D3 (agri-verb rate) and D1 (first-person rate) are
mostly zeros with a few large values. A median-difference permutation test is
close to powerless on that shape: register_scene_type.py returned p=1.0000 for
D1 purely because both medians were 0.0. The right test for a zero-inflated
variable is PRESENCE/ABSENCE, Fisher exact. Declaring this openly as a POST-HOC
test change, chosen after seeing the zero inflation and not before -- so it is
reported alongside the pre-registered median test, never instead of it.

THREAT 2 -- length. interiority_pct is a percentage over sentences. The base arm's
de-looped samples run 652-819 words; the scale-1.0 arm's run 146-720. Fewer
sentences means a coarser, noisier percentage, and could manufacture the
overshoot on its own. Control: recompute every metric on a LENGTH-MATCHED PREFIX
-- the first N words of every sample, N = the shortest de-looped sample. If the
effect survives on equal-length text it is not a length artifact.
"""
import json
import math
import re
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "/workspace/driftenginelite/Drift-engine")
from engine.texture import TextureAnalyzer
from engine.entropy import EntropyCalculator
from engine.drift import DriftScorer
from engine.lexicon import PHYSICAL_VERBS

TA, EC, DS = TextureAnalyzer(), EntropyCalculator(), DriftScorer()
J = json.load(open("/workspace/register_check.json"))
ARMS = ["0.00 base", "0.25 LoRA", "0.50 LoRA", "1.00 LoRA"]
GEN = {"0.00 base": "/workspace/gen_base_control.json",
       "0.25 LoRA": "/workspace/gen_v6_scale_down.json",
       "0.50 LoRA": "/workspace/gen_v6_scale_down.json",
       "1.00 LoRA": "/workspace/gen_v6_cap2560.json"}

AGRI = sorted({"plowed", "ploughed", "plows", "plow", "sowed", "sows", "sow",
               "harvested", "harvests", "harvest", "reaped", "reaps", "reap",
               "threshed", "harrowed", "seeded", "mowed", "mows", "mow", "scythed",
               "hammered", "hammers", "nailed", "nails", "sawed", "saws",
               "chopped", "chops", "dug", "digs", "dig", "loaded", "loads",
               "unloaded", "unloads", "hitched", "saddled", "mounted", "dismounted",
               "poured", "pours", "carried", "carries", "dragged", "drags",
               "buried", "buries", "bury", "filled", "fills", "fill"} & PHYSICAL_VERBS)
FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}


def fisher(a, b, c, d):
    """Two-sided Fisher exact on [[a,b],[c,d]] by summing tables <= observed prob."""
    n = a + b + c + d
    r1, r2, c1 = a + b, c + d, a + c

    def pr(x):
        if x < 0 or x > min(r1, c1) or (c1 - x) > r2:
            return 0.0
        return math.exp(sum(map(math.lgamma, [r1 + 1, r2 + 1, c1 + 1, n - c1 + 1]))
                        - sum(map(math.lgamma, [x + 1, r1 - x + 1, c1 - x + 1,
                                                r2 - c1 + x + 1, n + 1])))
    p0 = pr(a)
    return sum(pr(x) for x in range(0, min(r1, c1) + 1) if pr(x) <= p0 * (1 + 1e-9))


def deloop(text):
    n, lo, hi, best = len(text), 1, len(text) // 2, (0, -1, -1)
    while lo <= hi:
        mid = (lo + hi) // 2
        seen, found = {}, None
        for i in range(n - mid + 1):
            s = text[i:i + mid]
            if s in seen:
                found = (mid, seen[s], i)
                break
            seen[s] = i
        if found:
            best, lo = found, mid + 1
        else:
            hi = mid - 1
    span, a, b = best
    return text[:b] if span >= 200 and b > 0 else text


def texts(arm):
    d = json.load(open(GEN[arm]))
    if arm == "0.25 LoRA":
        return {f"s0.25#{r['i']}": r["text"] for r in d["results"] if r["scale"] == 0.25}
    if arm == "0.50 LoRA":
        return {f"s0.5#{r['i']}": r["text"] for r in d["results"] if r["scale"] == 0.5}
    if arm == "0.00 base":
        return {f"b#{r['i']}": r["text"] for r in d["results"]}
    return {f"v#{r['i']}": r["text"] for r in d["results"]}


ALL = {arm: {n: deloop(t) for n, t in texts(arm).items()} for arm in ARMS}
WEAK = ["0.00 base", "0.25 LoRA"]      # base + the dialled-down adapter

print("=" * 78)
print("THREAT 1 -- presence/absence, Fisher exact  [POST-HOC test choice]")
print("=" * 78)


def presence(metric, thresh):
    out = {}
    for arm in ARMS:
        hits = 0
        for n, t in ALL[arm].items():
            words = re.findall(r"[a-z']+", t.lower())
            nw = max(1, len(words))
            if metric == "agri":
                v = 1000 * sum(1 for w in words if w in AGRI) / nw
            else:
                v = 1000 * sum(1 for w in words if w in FIRST_PERSON) / nw
            hits += v > thresh
        out[arm] = (hits, len(ALL[arm]))
    return out


for metric, thresh, label in [("agri", 0.0, "any agricultural/craft verb at all"),
                              ("fp", 10.0, "first-person rate > 10 per 1000 words")]:
    p = presence(metric, thresh)
    print(f"-- {label}")
    for arm in ARMS:
        print(f"   {arm:<12} {p[arm][0]}/{p[arm][1]}")
    wa = sum(p[a][0] for a in WEAK)
    wn = sum(p[a][1] for a in WEAK)
    la, ln = p["1.00 LoRA"]
    pv = fisher(wa, wn - wa, la, ln - la)
    print(f"   POOLED(base+0.25) {wa}/{wn}  vs  1.00 LoRA {la}/{ln}   "
          f"Fisher exact two-sided p = {pv:.5f}")

print()
print("-- interiority_pct above the T_corpus p90 (13.9)")
corp = sorted(r["tex"]["interiority_pct"] for r in J["t_corpus"])
p90, cmax = float(np.percentile(corp, 90)), max(corp)
cnt = {}
for arm in ARMS:
    xs = [r["tex"]["interiority_pct"] for r in J["arms"][arm]]
    cnt[arm] = (sum(1 for x in xs if x > p90), sum(1 for x in xs if x > cmax), len(xs))
    print(f"   {arm:<12} above p90 {cnt[arm][0]}/{cnt[arm][2]}   "
          f"above corpus MAX ({cmax:.1f}) {cnt[arm][1]}/{cnt[arm][2]}")
wa = sum(cnt[a][0] for a in WEAK)
wn = sum(cnt[a][2] for a in WEAK)
print(f"   POOLED(base+0.25) {wa}/{wn} vs 1.00 LoRA {cnt['1.00 LoRA'][0]}/{cnt['1.00 LoRA'][2]}"
      f"   Fisher exact two-sided p = "
      f"{fisher(wa, wn-wa, cnt['1.00 LoRA'][0], cnt['1.00 LoRA'][2]-cnt['1.00 LoRA'][0]):.5f}")

print()
print("=" * 78)
print("THREAT 2 -- length-matched prefix")
print("=" * 78)
N = min(len(t.split()) for arm in ARMS for t in ALL[arm].values())
print(f"prefix length N = {N} words (the shortest de-looped sample in any arm)")
print(f"{'sample':<9} {'int%':>6} {'act%':>6} {'agri/1k':>8} {'fp/1k':>7} {'drift':>7}")
pref = {}
for arm in ARMS:
    pref[arm] = []
    print(f"-- {arm}")
    for n, t in ALL[arm].items():
        p = " ".join(t.split()[:N])
        tex = TA.analyze(p)
        tex["entropy"] = EC.analyze(p)["entropy"]
        words = re.findall(r"[a-z']+", p.lower())
        nw = max(1, len(words))
        ag = 1000 * sum(1 for w in words if w in AGRI) / nw
        fp = 1000 * sum(1 for w in words if w in FIRST_PERSON) / nw
        dr = DS.score(tex, entropy=tex["entropy"])["drift_score"]
        pref[arm].append(dict(int_=tex["interiority_pct"], act=tex["action_pct"],
                              ag=ag, fp=fp, drift=dr))
        print(f"{n:<9} {tex['interiority_pct']:>6.1f} {tex['action_pct']:>6.1f} "
              f"{ag:>8.2f} {fp:>7.1f} {dr:>7.2f}")

print()
print(f"{'arm':<12} {'int% med':>9} {'act% med':>9} {'agri>0':>8} {'fp>10':>7} {'drift med':>10}")
for arm in ARMS:
    r = pref[arm]
    print(f"{arm:<12} {np.median([x['int_'] for x in r]):>9.1f} "
          f"{np.median([x['act'] for x in r]):>9.1f} "
          f"{sum(1 for x in r if x['ag'] > 0)}/{len(r):<6} "
          f"{sum(1 for x in r if x['fp'] > 10)}/{len(r):<5} "
          f"{np.median([x['drift'] for x in r]):>10.2f}")

for key, label, thr in [("ag", "any agri verb in the first %d words" % N, 0.0),
                        ("int_", "interiority_pct > corpus p90 (13.9)", p90)]:
    wa = sum(1 for a in WEAK for x in pref[a] if x[key] > thr)
    wn = sum(len(pref[a]) for a in WEAK)
    la = sum(1 for x in pref["1.00 LoRA"] if x[key] > thr)
    ln = len(pref["1.00 LoRA"])
    print(f"  {label}: POOLED {wa}/{wn} vs 1.00 {la}/{ln}  "
          f"Fisher p = {fisher(wa, wn-wa, la, ln-la):.5f}")


# ---------------------------------------------------------------------------
# THREAT 2b -- the N=60 prefix cannot test the agri-verb finding.
# Agri verbs occur at ~5 per 1000 words in the base arm, so 60 words carries an
# expected count of 0.3. A presence/absence test on that has almost no power and
# its p=0.41 is an absence of evidence, not evidence of absence. The
# length-INVARIANT form of the same question is a rate comparison: pool every
# word in an arm and test the two rates against each other with the exact
# conditional binomial test (the standard two-sample Poisson rate test), which
# is unaffected by how the words are distributed across samples.
# ---------------------------------------------------------------------------
from math import comb

print()
print("=" * 78)
print("THREAT 2b -- pooled agri-verb RATE, exact conditional binomial (length-invariant)")
print("=" * 78)
pool = {}
for arm in ARMS:
    hits = tot = 0
    for t in ALL[arm].values():
        w = re.findall(r"[a-z']+", t.lower())
        tot += len(w)
        hits += sum(1 for x in w if x in AGRI)
    pool[arm] = (hits, tot)
    print(f"   {arm:<12} {hits:>3} agri verbs / {tot:>5} words = {1000*hits/tot:>5.2f} per 1k")

k1 = sum(pool[a][0] for a in WEAK); n1 = sum(pool[a][1] for a in WEAK)
k2, n2 = pool["1.00 LoRA"]
print(f"   POOLED(base+0.25) {k1}/{n1} = {1000*k1/n1:.2f}/1k   "
      f"vs 1.00 LoRA {k2}/{n2} = {1000*k2/n2:.2f}/1k")
# exact conditional binomial: given k1+k2 events, P(K1 >= k1) under H0 p = n1/(n1+n2)
K, p0 = k1 + k2, n1 / (n1 + n2)
tail = sum(comb(K, i) * p0**i * (1 - p0)**(K - i) for i in range(k1, K + 1))
print(f"   H0 rates equal -> K1 ~ Binom(K={K}, p={p0:.3f}); expected {K*p0:.1f}, observed {k1}")
print(f"   exact one-sided p = {tail:.3g}   (two-sided ~ {min(1, 2*tail):.3g})")
