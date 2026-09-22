#!/usr/bin/env python3
"""First-word opening reuse -- the instrument the twelfth run introduced post hoc,
written down so the next comparison is pre-registered.

Definition. Split de-looped text (register_check.deloop, the canonical
de-looper) into sentences at whitespace following . ! ? " or the right curly
quote; drop sentences with no word. The opening is the first [\\w']+ token,
lower-cased. A sentence REUSES if its opening equals the opening of any of the
previous WINDOW (=4) sentences. Rate = reusing sentences / (sentences - 1), in
percent. Texts with fewer than 6 sentences are skipped (nan).

Why this and not the two-word anaphora rate: the opening guard forbids two-word
openings inside the same window, so on guarded arms the two-word rate is
constrained by construction; the one-word rate is not. The cycles the guard
passes (twelfth run #2, #8, #9) are one-word ladders.

Reference on the corpus chapter targets is printed first; each arm is scored
per sample and as median / share above the corpus p90.
"""
import json, re, sys, statistics as st
sys.path.insert(0, "/workspace/driftenginelite/Drift-engine")
sys.path.insert(0, "/workspace/driftenginelite/Drift-engine/scripts")
from register_check import deloop

WINDOW = 4
SENT = re.compile(r'(?<=[.!?"”])\s+')
WORD = re.compile(r"[\w']+")

def reuse(text, window=WINDOW):
    sents = [s for s in SENT.split(text) if WORD.search(s)]
    if len(sents) < 6: return float("nan")
    first = [WORD.findall(s.lower())[0] for s in sents]
    return 100.0 * sum(first[i] in first[max(0, i - window):i] for i in range(1, len(first))) / (len(first) - 1)

CORPUS = "/workspace/final_training_corpus_v2_1_latest.json"
ARMS = [("variant F (unguarded)", "/workspace/gen_v6_cap2560.json"),
        ("F + ngram guard", "/workspace/gen_v6_guard.json"),
        ("variant G", "/workspace/driftenginelite/eval/gen_v7_variantG.json"),
        ("G + opening guard", "/workspace/driftenginelite/eval/gen_v7_variantG_openguard.json"),
        ("G @ epoch 1", "/workspace/driftenginelite/eval/gen_v7_ckpt62.json"),
        ("base n=4 (Aug)", "/workspace/gen_base_control.json")] + \
       [(a, p) for a, p in (x.split("=", 1) for x in sys.argv[1:])]

if __name__ == "__main__":
    corpus = json.load(open(CORPUS))
    ch = [e["messages"][2]["content"] for e in corpus if e["messages"][1]["content"].strip() == "Write the next chapter."]
    c = sorted(reuse(t) for t in ch)
    p90 = c[int(0.9 * (len(c) - 1))]
    print(f"corpus chapters n={len(c)}: median {st.median(c):.1f}  p90 {p90:.1f}  max {max(c):.1f}   (window {WINDOW})")
    print(f"\n{'arm':24} {'n':>2} {'per-sample':52} {'median':>6} {'>p90':>5}")
    for name, path in ARMS:
        rs = json.load(open(path))["results"]
        v = [reuse(deloop(r["text"])[0]) for r in rs]
        ok = [x for x in v if x == x]
        print(f"{name:24} {len(ok):>2} {' '.join(f'{x:4.0f}' for x in v):52} {st.median(ok):>6.1f} {sum(x > p90 for x in ok):>2}/{len(ok)}")
