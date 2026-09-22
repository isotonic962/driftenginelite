"""INVESTIGATION.md OPEN item 1c, part 2 — is the "amplifies past its own data"
claim an artifact of comparing unequal lengths, and does the same shape appear on
a second axis? Static, no GPU, no training run.

TWO TESTS, both pre-specified before results were inspected.

TEST A — LENGTH-MATCHED PERCENTILE RANK.
The fourth run scored each generation against the interiority_pct of WHOLE
long-form corpus entries (~1410 words). interiority_pct is a percentage over
classified sentences, so its sampling variance depends on how many sentences went
into it: a 1410-word entry has ~90, a 400-word generation ~25, a 150-word
generation ~9. Comparing the tail of a low-variance distribution against draws
from a high-variance one manufactures exceedances with no register difference at
all. Corpus prefix stats confirm the effect is large here: LONG entries have
whole-entry p90 13.9 but first-60-word p90 25.0.

The fix: score each generation of W words against the 138 corpus entries
truncated to their own first W words. Same statistic, same sentence budget, so
the percentile rank is honest. Reported per arm as a median percentile; 50 means
"sits exactly at the middle of the corpus at that length".

TEST B — ANAPHORA, the axis the termination failure actually runs on.
The runaway samples are anaphoric ladders ("He imagined... He imagined..."). That
is the amplification the standing task cares about, and unlike interiority it has
never been measured against the corpus at all. Metric: the fraction of adjacent
sentence pairs sharing their first two words, plus the longest run of
consecutive sentences with the same two-word opening. Computed on DE-LOOPED text
so a termination artifact cannot masquerade as a style result.

Reference sets are the whole corpus (702), not the 138 the register work used.
"""
import json
import sys
import statistics as st

sys.path.insert(0, "/workspace/driftenginelite/Drift-engine")
sys.path.insert(0, "/workspace/driftenginelite/Drift-engine/scripts")
from engine.texture import TextureAnalyzer
from register_check import deloop

CORPUS = "/workspace/final_training_corpus_v2_1_latest.json"
GENS = {
    "0.00 base": ("/workspace/gen_base_control.json", None),
    "0.25 LoRA": ("/workspace/gen_v6_scale_down.json", 0.25),
    "0.50 LoRA": ("/workspace/gen_v6_scale_down.json", 0.5),
    "1.00 LoRA": ("/workspace/gen_v6_cap2560.json", None),
}
LONG_USER_TURN = "Write the next chapter."

TA = TextureAnalyzer()


def int_pct(text):
    return TA.action_interiority_ratio(text)[1]


def n_sent(text):
    return len(TA._split_sentences(text))


def anaphora(text):
    """(adjacent-pair repeat rate %, longest same-opening run, n sentences)."""
    sents = TA._split_sentences(text)
    heads = [" ".join(TA._tokenize(s)[:2]) for s in sents]
    if len(heads) < 2:
        return 0.0, 1, len(sents)
    pairs = sum(1 for a, b in zip(heads, heads[1:]) if a and a == b)
    run = best = 1
    for a, b in zip(heads, heads[1:]):
        run = run + 1 if a and a == b else 1
        best = max(best, run)
    return 100.0 * pairs / (len(heads) - 1), best, len(sents)


def load_gens():
    out = {}
    for arm, (path, scale) in GENS.items():
        d = json.load(open(path))
        rs = [r for r in d["results"] if scale is None or r.get("scale") == scale]
        # words = the DE-LOOPED length. The scored text is de-looped, so the
        # length the corpus is matched to has to be the de-looped one; matching
        # to the raw cap length would compare a 400-word text against 2300-word
        # corpus prefixes.
        out[arm] = []
        for r in rs:
            t = deloop(r["text"])[0]
            out[arm].append(dict(i=r["i"], words=len(t.split()),
                                 raw_words=r["word_count"],
                                 finish=r["finish_reason"], text=t))
    return out


def main():
    corpus = json.load(open(CORPUS))
    longs = [e["messages"][2]["content"] for e in corpus
             if e["messages"][1]["content"].strip() == LONG_USER_TURN]
    every = [e["messages"][2]["content"] for e in corpus]
    arms = load_gens()

    print("=" * 78)
    print("TEST A — interiority_pct percentile rank against LENGTH-MATCHED corpus")
    print("=" * 78)
    print(f"{'arm':11s} {'#':>3s} {'raw w':>6s} {'delp w':>7s} {'sent':>5s} {'int%':>6s} "
          f"{'corpus median @W':>17s} {'corpus p90 @W':>14s} {'pctile':>7s}")
    arm_pctiles = {}
    for arm, samples in arms.items():
        ps = []
        for s in samples:
            w = max(30, s["words"])
            ref = sorted(int_pct(" ".join(t.split()[:w])) for t in longs)
            v = int_pct(s["text"])
            # percentile rank, midrank for ties
            below = sum(1 for r in ref if r < v)
            ties = sum(1 for r in ref if r == v)
            p = 100.0 * (below + 0.5 * ties) / len(ref)
            ps.append(p)
            med = st.median(ref)
            p90 = sorted(ref)[int(0.9 * (len(ref) - 1))]
            print(f"{arm:11s} {s['i']:3d} {s['raw_words']:6d} {s['words']:7d} "
                  f"{n_sent(s['text']):5d} "
                  f"{v:6.1f} {med:17.1f} {p90:14.1f} {p:7.1f}")
        arm_pctiles[arm] = ps
    print()
    print(f"{'arm':11s} {'n':>3s} {'median pctile':>14s}   samples above the length-matched p90")
    for arm, ps in arm_pctiles.items():
        print(f"{arm:11s} {len(ps):3d} {st.median(ps):14.1f}   "
              f"{sum(1 for p in ps if p > 90)}/{len(ps)}")
    print("\n  50 = indistinguishable from the corpus at that length.")
    print("  Compare with the fourth run's unmatched result: 6/10 above p90, 4/10 above MAX.")

    print()
    print("=" * 78)
    print("TEST B — ANAPHORA (de-looped): adjacent sentences sharing their first 2 words")
    print("=" * 78)
    print(f"{'set':22s} {'n':>4s} {'median rate%':>13s} {'mean rate%':>11s} "
          f"{'p90':>6s} {'max':>6s} {'med longest run':>16s} {'max run':>8s}")

    def block(name, texts):
        rows = [anaphora(t) for t in texts]
        rates = [r[0] for r in rows]
        runs = [r[1] for r in rows]
        srt = sorted(rates)
        print(f"{name:22s} {len(rows):4d} {st.median(rates):13.1f} {st.mean(rates):11.1f} "
              f"{srt[int(0.9 * (len(srt) - 1))]:6.1f} {max(rates):6.1f} "
              f"{st.median(runs):16.1f} {max(runs):8d}")
        return rates, runs

    c_rates, c_runs = block("CORPUS long-form", longs)
    block("CORPUS all 702", every)
    for arm, samples in arms.items():
        rates, runs = block(f"GEN {arm}", [s["text"] for s in samples])
        srt = sorted(c_rates)
        p90 = srt[int(0.9 * (len(srt) - 1))]
        cmax = max(c_rates)
        cmaxrun = max(c_runs)
        print(f"{'':22s}      above long-form p90 ({p90:.1f}): "
              f"{sum(1 for r in rates if r > p90)}/{len(rates)}   "
              f"above long-form MAX ({cmax:.1f}): {sum(1 for r in rates if r > cmax)}/{len(rates)}   "
              f"run > corpus max run ({cmaxrun}): {sum(1 for r in runs if r > cmaxrun)}/{len(runs)}")

    print()
    print("  Worst anaphoric run in each arm, with the sentence openings:")
    for arm, samples in arms.items():
        worst = max(samples, key=lambda s: anaphora(s["text"])[1])
        sents = TA._split_sentences(worst["text"])
        heads = [" ".join(TA._tokenize(x)[:2]) for x in sents]
        run = best = 1
        end = 0
        for j in range(1, len(heads)):
            run = run + 1 if heads[j] and heads[j] == heads[j - 1] else 1
            if run > best:
                best, end = run, j
        seg = sents[end - best + 1:end + 1] if best > 1 else []
        print(f"\n  {arm} sample #{worst['i']} ({worst['words']} w, "
              f"{worst['finish']}) longest run = {best}")
        for x in seg[:6]:
            print(f"      {x[:96]}")

    test_c(arms, longs)


def fisher(a, b, c, d):
    """Two-sided Fisher exact on [[a,b],[c,d]]."""
    from math import comb
    n = a + b + c + d
    r1, c1 = a + b, a + c
    def p(x):
        return comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)
    p0 = p(a)
    lo = max(0, c1 - (n - r1))
    hi = min(r1, c1)
    return sum(p(x) for x in range(lo, hi + 1) if p(x) <= p0 + 1e-12)


def perm_test(xs, ys, iters=200000):
    """Exact-if-small permutation test on the difference of means."""
    from itertools import combinations
    pool = xs + ys
    obs = abs(sum(xs) / len(xs) - sum(ys) / len(ys))
    idx = range(len(pool))
    cnt = tot = 0
    for c in combinations(idx, len(xs)):
        a = [pool[i] for i in c]
        b = [pool[i] for i in idx if i not in c]
        tot += 1
        if abs(sum(a) / len(a) - sum(b) / len(b)) >= obs - 1e-12:
            cnt += 1
    return obs, cnt / tot, tot


def test_c(arms, longs):
    """TEST C — does anaphora predict failure to terminate, WITHIN the scale-1.0 arm?

    The fourth run showed interiority does NOT (p=0.472), so register and
    termination were declared two casualties of one cause but not one mechanism.
    Anaphora is the other candidate and has never been tested this way. Measured
    on DE-LOOPED text, so the verbatim repeat that defines a runaway is removed
    before scoring — a CAP sample scores high here only if its NON-repeated
    prefix is already laddering.
    """
    print()
    print("=" * 78)
    print("TEST C — within the scale-1.0 arm, does anaphora predict CAP?")
    print("=" * 78)
    samples = arms["1.00 LoRA"]
    eos = [anaphora(s["text"])[0] for s in samples if s["finish"] == "EOS"]
    cap = [anaphora(s["text"])[0] for s in samples if s["finish"] != "EOS"]
    print(f"  EOS n={len(eos)} anaphora rate% {[round(x, 1) for x in eos]}  "
          f"mean {sum(eos) / len(eos):.1f}")
    print(f"  CAP n={len(cap)} anaphora rate% {[round(x, 1) for x in cap]}  "
          f"mean {sum(cap) / len(cap):.1f}")
    obs, p, tot = perm_test(eos, cap)
    print(f"  |mean difference| = {obs:.1f} pts, exact permutation p = {p:.3f} "
          f"({tot} splits)")
    runs_e = [anaphora(s["text"])[1] for s in samples if s["finish"] == "EOS"]
    runs_c = [anaphora(s["text"])[1] for s in samples if s["finish"] != "EOS"]
    print(f"  longest same-opening run  EOS {runs_e}  CAP {runs_c}")
    obs, p, tot = perm_test([float(x) for x in runs_e], [float(x) for x in runs_c])
    print(f"  |mean difference| = {obs:.1f} sentences, exact permutation p = {p:.3f}")

    print()
    print("  Arm-level tests against the corpus (pooled base+0.25 vs scale 1.0):")
    c_rates = [anaphora(t)[0] for t in longs]
    cmax = max(c_rates)
    pooled = [anaphora(s["text"])[0] for a in ("0.00 base", "0.25 LoRA")
              for s in arms[a]]
    one = [anaphora(s["text"])[0] for s in arms["1.00 LoRA"]]
    a = sum(1 for r in pooled if r > cmax)
    c = sum(1 for r in one if r > cmax)
    print(f"    above the long-form corpus MAXIMUM ({cmax:.1f}%): "
          f"pooled {a}/{len(pooled)} vs scale-1.0 {c}/{len(one)}   "
          f"Fisher exact p = {fisher(a, len(pooled) - a, c, len(one) - c):.4f}")


if __name__ == "__main__":
    main()
