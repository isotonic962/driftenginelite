"""INVESTIGATION.md OPEN item 1b — settle the register question on text already
on disk. Static, no GPU, no training run, no sign-off needed.

WHY THIS EXISTS
---------------
Every measurement in this investigation so far scores *stopping*: EOS vs CAP,
word count, longest repeated substring. None scores *voice*. That is the binding
gap before any gentler-retrain is spent: a weaker adapter that terminates
cleanly but writes like the stock base model would score as a total success on
the current bar and would in fact have bought nothing, because the LoRA exists
to buy register.

The specific `[unverified]` claim from the third run, quoted from EXPERIMENT_LOG:
"the 0.25 prose reads closer to the target register than the base control does."
That was an impression from reading six samples. This script tests it.

TWO TARGETS, NOT ONE — read before interpreting any number below
---------------------------------------------------------------
"The target register" is ambiguous in this project and the ambiguity is load
bearing, so both targets are measured separately and never averaged:

  T_moberg  the DESIGN's target. DriftScorer's corridors are derived from 16
            full chapters of Moberg's The Emigrants, and verify_scorer.py holds
            10 hand-measured Moberg chapter texture vectors. Independent of the
            training corpus entirely.

  T_corpus  what the LoRA was ACTUALLY trained on. The 138 long-form entries of
            final_training_corpus_v2_1_latest.json — Larsson, Backman, Lapidus,
            Haruf, Lagerlof, Nesser. ESTABLISHED already records that 62% of
            these contain modern-life vocabulary and 8% contain 1840s rural
            vocabulary.

An adapter can move toward T_corpus and away from T_moberg at the same time.
Nothing measured so far would have detected that. It is the first thing to look
for in the output.

PRE-REGISTERED METRICS — fixed before any result was inspected
--------------------------------------------------------------
M1  DriftScorer drift_score + components, via the project's own TextureAnalyzer
    and EntropyCalculator. Lower = closer to T_moberg. Caveat from ESTABLISHED
    carried forward: drift.py still holds the 16-chapter corridors, not the
    88-chapter recalibration, so RANKING is valid and absolute values are not.

M2  Moberg-chapter z-distance. The 10 verify_scorer chapter vectors are an
    empirical reference cloud in the same 6-feature space. Per feature,
    z = (x - mean_moberg) / sd_moberg; distance = mean |z| over the 6 features.
    Corridor-independent second view of the same question, so M1 and M2 agreeing
    means more than either alone. Mean-absolute rather than Mahalanobis on
    purpose: n=10 cannot support a 6x6 covariance estimate.

M3  Corpus-likeness. tf-idf cosine to the centroid of the 138 T_corpus long-form
    assistant texts. Corpus entries are scored LEAVE-ONE-OUT (centroid recomputed
    without the entry being scored) so the in-distribution band is honest and not
    inflated by an entry's own contribution to its own target.

M4  Modern-life probe rate per 1000 words. The narrow probe is exactly the three
    words ESTABLISHED already used (car, phone, computer) so the number is
    comparable to the corpus register scan on record; the wide probe adds obvious
    contemporary-life nouns. Fixed list, written before results were seen.

M5  Keyness by log-odds-ratio with an informative Dirichlet prior (Monroe,
    Colaresi & Quinn 2008), each arm against the pooled other generation arms.
    This is the un-riggable one: it picks the distinctive vocabulary itself
    rather than letting the analyst choose a lexicon after seeing which arm he
    wants to win. If the 0.25 arm's top keyness words are labor/rural and the
    base arm's are contemporary-urban, the claim survives a test it could have
    failed.

THE LOOP CONFOUND
-----------------
Five of the ten scale-1.0 samples and one of three scale-0.5 samples run away
into a verbatim loop to the cap. A loop is a TERMINATION artifact; leaving it in
distorts entropy (repeated text is lower entropy) and sentence_rhythm, and would
let a termination failure masquerade as a register result. Every metric is
therefore computed on the DE-LOOPED text: if the longest repeated substring is
>= 200 chars, the text is cut at the start of its second occurrence. Raw values
are printed alongside so the correction is visible rather than silent.
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

CORPUS = "/workspace/final_training_corpus_v2_1_latest.json"
GEN_BASE = "/workspace/gen_base_control.json"
GEN_SCALE = "/workspace/gen_v6_scale_down.json"
GEN_V6 = "/workspace/gen_v6_cap2560.json"

DELOOP_MIN_CHARS = 200
TEXTURE_KEYS = ["action_pct", "interiority_pct", "neutral_pct",
                "dialogue_density", "figurative_density", "sentence_rhythm"]

# verify_scorer.py's 10 hand-measured Moberg chapters -- the T_moberg cloud.
MOBERG = [
    dict(action_pct=29.5, interiority_pct=3.8,  neutral_pct=66.7, dialogue_density=3.8,  figurative_density=0.06, sentence_rhythm=11.24, entropy=7.988),
    dict(action_pct=24.4, interiority_pct=8.9,  neutral_pct=66.7, dialogue_density=2.2,  figurative_density=0.0,  sentence_rhythm=8.40,  entropy=7.369),
    dict(action_pct=34.2, interiority_pct=5.3,  neutral_pct=60.5, dialogue_density=10.5, figurative_density=0.0,  sentence_rhythm=15.20, entropy=7.575),
    dict(action_pct=38.7, interiority_pct=0.0,  neutral_pct=61.3, dialogue_density=0.0,  figurative_density=0.17, sentence_rhythm=9.47,  entropy=7.291),
    dict(action_pct=17.0, interiority_pct=17.0, neutral_pct=66.0, dialogue_density=4.3,  figurative_density=0.0,  sentence_rhythm=10.46, entropy=7.617),
    dict(action_pct=29.5, interiority_pct=13.6, neutral_pct=56.8, dialogue_density=13.6, figurative_density=0.17, sentence_rhythm=12.02, entropy=8.065),
    dict(action_pct=31.3, interiority_pct=8.4,  neutral_pct=60.3, dialogue_density=4.6,  figurative_density=0.08, sentence_rhythm=9.39,  entropy=8.073),
    dict(action_pct=19.5, interiority_pct=9.8,  neutral_pct=70.7, dialogue_density=6.5,  figurative_density=0.08, sentence_rhythm=9.86,  entropy=8.037),
    dict(action_pct=34.0, interiority_pct=9.4,  neutral_pct=56.6, dialogue_density=15.1, figurative_density=0.0,  sentence_rhythm=8.24,  entropy=7.570),
    dict(action_pct=22.5, interiority_pct=2.5,  neutral_pct=75.0, dialogue_density=10.0, figurative_density=0.12, sentence_rhythm=11.35, entropy=7.504),
]

# M4 -- fixed before results were seen.
MODERN_NARROW = ["car", "cars", "phone", "phones", "telephone",
                 "computer", "computers"]
MODERN_WIDE = MODERN_NARROW + [
    "television", "tv", "radio", "email", "internet", "website", "mobile",
    "cigarette", "cigarettes", "coffee", "refrigerator", "fridge", "elevator",
    "apartment", "office", "traffic", "bus", "train", "taxi", "airport",
    "plastic", "電", "supermarket", "police", "hospital", "doctor", "hotel",
    "camera", "screen", "battery", "electric", "electricity", "engine",
    "motor", "truck", "gasoline", "petrol", "factory", "bank", "credit",
]

STOP = set("""a an the and or but if then than that this these those of in on at to for
from by with without into onto over under up down out off again further once here there
when where why how all any both each few more most other some such no nor not only own
same so too very can will just should now i me my myself we our ours ourselves you your
yours he him his she her hers it its they them their theirs what which who whom am is are
was were be been being have has had having do does did doing would could shall may might
must as about after before during while against between through above below because until
s t don t ll re ve d m o y""".split())


# ----------------------------------------------------------------- text utils
def longest_repeat_offsets(text):
    """(span, first_index, second_index) of the longest repeated substring."""
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
    return best


def deloop(text):
    """Cut at the second occurrence of a >=200ch repeat. See THE LOOP CONFOUND."""
    span, a, b = longest_repeat_offsets(text)
    if span >= DELOOP_MIN_CHARS and b > 0:
        return text[:b], span
    return text, span


def content_tokens(text):
    return [w for w in re.findall(r"[a-z']+", text.lower())
            if w not in STOP and len(w) > 2]


# ------------------------------------------------------------------ M1 and M2
TA, EC, DS = TextureAnalyzer(), EntropyCalculator(), DriftScorer()

MOB_MEAN = {k: float(np.mean([m[k] for m in MOBERG])) for k in TEXTURE_KEYS}
MOB_SD = {k: float(np.std([m[k] for m in MOBERG], ddof=1)) for k in TEXTURE_KEYS}


def texture_of(text):
    t = TA.analyze(text)
    t["entropy"] = EC.analyze(text)["entropy"]
    return t


def drift_of(t):
    return DS.score(t, entropy=t["entropy"])["drift_score"]


def moberg_z(t):
    """M2. Mean |z| across the 6 texture features vs the Moberg chapter cloud."""
    return float(np.mean([abs(t[k] - MOB_MEAN[k]) / MOB_SD[k] for k in TEXTURE_KEYS]))


# ------------------------------------------------------------------------- M3
def build_tfidf(corpus_docs):
    """Vocabulary + idf from the T_corpus long-form entries only."""
    dfs = Counter()
    for d in corpus_docs:
        dfs.update(set(d))
    vocab = {w: i for i, w in enumerate(sorted(w for w, c in dfs.items() if c >= 3))}
    n = len(corpus_docs)
    idf = np.zeros(len(vocab))
    for w, i in vocab.items():
        idf[i] = math.log((1 + n) / (1 + dfs[w])) + 1.0
    return vocab, idf


def vectorize(tokens, vocab, idf):
    v = np.zeros(len(idf))
    for w, c in Counter(tokens).items():
        i = vocab.get(w)
        if i is not None:
            v[i] = (1 + math.log(c)) * idf[i]      # sublinear tf
    nrm = np.linalg.norm(v)
    return v / nrm if nrm else v


# ------------------------------------------------------------------------- M5
def keyness(target_tokens, background_tokens, top=18, alpha0=1000.0):
    """Log-odds ratio, informative Dirichlet prior (Monroe/Colaresi/Quinn 2008).

    Returns the words most distinctive of `target` against `background`, chosen
    by the procedure rather than by the analyst. z-scores are the usual
    delta / sqrt(var) with the prior drawn from the pooled counts.
    """
    yi, yj = Counter(target_tokens), Counter(background_tokens)
    pooled = yi + yj
    n_pool = sum(pooled.values())
    ni, nj = sum(yi.values()), sum(yj.values())
    out = []
    for w, cw in pooled.items():
        if cw < 6:
            continue
        a_w = alpha0 * cw / n_pool
        li = math.log((yi[w] + a_w) / (ni + alpha0 - yi[w] - a_w))
        lj = math.log((yj[w] + a_w) / (nj + alpha0 - yj[w] - a_w))
        var = 1.0 / (yi[w] + a_w) + 1.0 / (yj[w] + a_w)
        out.append((( li - lj) / math.sqrt(var), w, yi[w], yj[w]))
    out.sort(reverse=True)
    return out[:top]


# ------------------------------------------------------------------ load arms
def load_arms():
    arms = {}
    b = json.load(open(GEN_BASE))
    arms["0.00 base"] = [(f"b#{r['i']}", r["text"], r["finish_reason"], r["word_count"])
                         for r in b["results"]]
    s = json.load(open(GEN_SCALE))
    for sc in (0.25, 0.5):
        arms[f"{sc:.2f} LoRA"] = [(f"s{sc}#{r['i']}", r["text"], r["finish_reason"], r["word_count"])
                                  for r in s["results"] if r["scale"] == sc]
    v = json.load(open(GEN_V6))
    arms["1.00 LoRA"] = [(f"v#{r['i']}", r["text"], r["finish_reason"], r["word_count"])
                         for r in v["results"]]
    return arms, {"base_gen": b, "scale_gen": s, "v6_gen": v}


def main():
    arms, meta = load_arms()
    corpus = json.load(open(CORPUS))
    longs = [e for e in corpus
             if e["messages"][1]["content"].strip() == "Write the next chapter."]
    corpus_texts = [e["messages"][2]["content"] for e in longs]
    corpus_ids = [e["id"] for e in longs]

    print("=" * 78)
    print("PROVENANCE")
    print("=" * 78)
    print(f"  corpus (T_corpus)   : {CORPUS}")
    print(f"                        {len(corpus)} records, {len(longs)} long-form")
    print(f"  base control        : adapter={meta['base_gen'].get('adapter')!r} "
          f"n={len(arms['0.00 base'])}")
    print(f"  scale sweep         : adapter={meta['scale_gen'].get('adapter')} "
          f"corpus={meta['scale_gen'].get('corpus')}")
    print(f"  variant F (s=1.0)   : adapter={meta['v6_gen'].get('adapter')} "
          f"corpus={meta['v6_gen'].get('corpus')}")
    print(f"  T_moberg cloud      : {len(MOBERG)} hand-measured chapters "
          f"from engine/verify_scorer.py")

    # ---- per-sample M1 / M2 / M4 -------------------------------------------
    rows = {}
    for arm, samples in arms.items():
        rows[arm] = []
        for name, text, fin, wc in samples:
            dl, span = deloop(text)
            t_raw, t_dl = texture_of(text), texture_of(dl)
            toks = content_tokens(dl)
            nw = max(1, len(dl.split()))
            rows[arm].append(dict(
                name=name, fin=fin, wc=wc, wc_dl=len(dl.split()),
                looped=span >= DELOOP_MIN_CHARS, span=span,
                tex=t_dl, drift_dl=drift_of(t_dl), drift_raw=drift_of(t_raw),
                mz=moberg_z(t_dl), toks=toks,
                mod_n=1000 * sum(1 for w in re.findall(r"[a-z]+", dl.lower())
                                 if w in MODERN_NARROW) / nw,
                mod_w=1000 * sum(1 for w in re.findall(r"[a-z]+", dl.lower())
                                 if w in MODERN_WIDE) / nw,
            ))

    corpus_rows = []
    for cid, ct in zip(corpus_ids, corpus_texts):
        t = texture_of(ct)
        nw = max(1, len(ct.split()))
        corpus_rows.append(dict(
            name=cid, tex=t, drift_dl=drift_of(t), mz=moberg_z(t),
            toks=content_tokens(ct),
            mod_n=1000 * sum(1 for w in re.findall(r"[a-z]+", ct.lower())
                             if w in MODERN_NARROW) / nw,
            mod_w=1000 * sum(1 for w in re.findall(r"[a-z]+", ct.lower())
                             if w in MODERN_WIDE) / nw,
        ))

    # ---- M3 tf-idf, leave-one-out for corpus -------------------------------
    corpus_tok = [r["toks"] for r in corpus_rows]
    vocab, idf = build_tfidf(corpus_tok)
    C = np.vstack([vectorize(t, vocab, idf) for t in corpus_tok])
    total = C.sum(axis=0)
    n_c = len(C)
    for i, r in enumerate(corpus_rows):                    # leave-one-out
        cen = (total - C[i]) / (n_c - 1)
        nrm = np.linalg.norm(cen)
        r["cos"] = float(C[i] @ cen / nrm) if nrm else 0.0
    centroid = total / n_c
    centroid = centroid / np.linalg.norm(centroid)
    for arm in rows:
        for r in rows[arm]:
            r["cos"] = float(vectorize(r["toks"], vocab, idf) @ centroid)

    # ---- report -------------------------------------------------------------
    def med(xs):
        return float(np.median(xs)) if xs else float("nan")

    print()
    print("=" * 78)
    print("M1/M2/M3/M4 BY ARM  (all on de-looped text; medians)")
    print("=" * 78)
    print(f"{'arm':<12} {'n':>3} {'drift':>7} {'|z|Mob':>7} {'cosCorp':>8} "
          f"{'mod/1k':>7} {'act%':>6} {'int%':>6} {'dial%':>6} {'fig':>5} {'rhy':>6} {'H':>6}")
    print("-" * 78)

    def line(label, rs):
        print(f"{label:<12} {len(rs):>3} {med([r['drift_dl'] for r in rs]):>7.2f} "
              f"{med([r['mz'] for r in rs]):>7.2f} {med([r['cos'] for r in rs]):>8.3f} "
              f"{med([r['mod_w'] for r in rs]):>7.2f} "
              f"{med([r['tex']['action_pct'] for r in rs]):>6.1f} "
              f"{med([r['tex']['interiority_pct'] for r in rs]):>6.1f} "
              f"{med([r['tex']['dialogue_density'] for r in rs]):>6.1f} "
              f"{med([r['tex']['figurative_density'] for r in rs]):>5.2f} "
              f"{med([r['tex']['sentence_rhythm'] for r in rs]):>6.2f} "
              f"{med([r['tex']['entropy'] for r in rs]):>6.2f}")

    mob_rows = [dict(drift_dl=drift_of(m), mz=moberg_z(m), cos=float('nan'),
                     mod_w=float('nan'), tex=m) for m in MOBERG]
    line("T_moberg", mob_rows)
    line("T_corpus", corpus_rows)
    print("-" * 78)
    for arm in ["0.00 base", "0.25 LoRA", "0.50 LoRA", "1.00 LoRA"]:
        line(arm, rows[arm])
    print()
    print("  drift   = M1 DriftScorer, LOWER is closer to T_moberg (ranking only)")
    print("  |z|Mob  = M2 mean |z| vs the 10-chapter Moberg cloud, LOWER is closer")
    print("  cosCorp = M3 tf-idf cosine to the T_corpus centroid, HIGHER is closer")
    print("            (T_corpus row is leave-one-out; it is the in-distribution band)")
    print("  mod/1k  = M4 wide modern-life probe hits per 1000 words")

    print()
    print("=" * 78)
    print("PER-SAMPLE")
    print("=" * 78)
    print(f"{'sample':<9} {'fin':>4} {'wc':>5} {'wcDL':>5} {'loop':>5} "
          f"{'drift':>7} {'|z|Mob':>7} {'cosCorp':>8} {'modN':>6} {'modW':>6}")
    for arm in ["0.00 base", "0.25 LoRA", "0.50 LoRA", "1.00 LoRA"]:
        print(f"-- {arm}")
        for r in rows[arm]:
            print(f"{r['name']:<9} {r['fin']:>4} {r['wc']:>5} {r['wc_dl']:>5} "
                  f"{'YES' if r['looped'] else '-':>5} {r['drift_dl']:>7.2f} "
                  f"{r['mz']:>7.2f} {r['cos']:>8.3f} {r['mod_n']:>6.2f} {r['mod_w']:>6.2f}")

    # ---- M5 keyness ---------------------------------------------------------
    print()
    print("=" * 78)
    print("M5 KEYNESS -- each arm vs the pooled OTHER generation arms")
    print("   (log-odds ratio, informative Dirichlet prior; the procedure picks")
    print("    the words, not the analyst)")
    print("=" * 78)
    pool = {arm: [w for r in rows[arm] for w in r["toks"]] for arm in rows}
    for arm in ["0.00 base", "0.25 LoRA", "0.50 LoRA", "1.00 LoRA"]:
        bg = [w for a, ws in pool.items() if a != arm for w in ws]
        ks = keyness(pool[arm], bg)
        print(f"-- {arm}  ({len(pool[arm])} content tokens)")
        print("   " + ", ".join(f"{w}({z:.1f})" for z, w, _, _ in ks))

    print()
    print("=" * 78)
    print("M5b KEYNESS -- generations vs T_corpus long-form")
    print("=" * 78)
    ctoks = [w for r in corpus_rows for w in r["toks"]]
    for arm in ["0.00 base", "0.25 LoRA", "1.00 LoRA"]:
        ks = keyness(pool[arm], ctoks, top=14)
        print(f"-- OVER-used by {arm} relative to T_corpus")
        print("   " + ", ".join(f"{w}({z:.1f})" for z, w, _, _ in ks))

    # ---- machine-readable ---------------------------------------------------
    out = {
        "corpus": CORPUS,
        "deloop_min_chars": DELOOP_MIN_CHARS,
        "arms": {arm: [{k: v for k, v in r.items() if k != "toks"} for r in rs]
                 for arm, rs in rows.items()},
        "t_corpus": [{k: v for k, v in r.items() if k != "toks"} for r in corpus_rows],
        "moberg_mean": MOB_MEAN, "moberg_sd": MOB_SD,
    }
    with open("/workspace/register_check.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote /workspace/register_check.json")


if __name__ == "__main__":
    main()
