#!/usr/bin/env python3
"""Build the variant-H corpus: v2_3_ch3x minus the high-reuse tail of the brief
branch. The sixteenth run's addendum located the sentence-shape prior in the
brief targets whose first-word reuse (window 4, the twelfth-run instrument)
exceeds the chapter branch's p90 (29.2): 37% of scorable briefs sit above it,
and their p90 (57) is where variant F's generations sit. This drops exactly
those briefs -- scorable (>= 6 sentences) AND reuse > chapter p90 -- and keeps
everything else byte-identical, including the x3 chapter duplication. ONE
CHANGE vs. the variant-G corpus. Never overwrites.

The reuse instrument and the p90 reference are imported from
first_word_reuse.py, not re-implemented, so the filter is the same measurement
the addendum reported (raw target text, no deloop -- corpus-target treatment)."""
import json, hashlib, os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from first_word_reuse import reuse

SRC = "/workspace/final_training_corpus_v2_3_ch3x.json"
DST = "/workspace/final_training_corpus_v2_4_briefclean.json"
CHAPTER_REF = "/workspace/final_training_corpus_v2_1_latest.json"
assert not os.path.exists(DST), "refusing to overwrite"

# chapter p90 exactly as first_word_reuse.py computes it
ref = json.load(open(CHAPTER_REF))
ch_ref = sorted(reuse(e["messages"][2]["content"]) for e in ref
                if e["messages"][1]["content"].strip() == "Write the next chapter.")
P90 = ch_ref[int(0.9 * (len(ch_ref) - 1))]
assert abs(P90 - 29.2) < 0.1, f"chapter p90 moved: {P90}"

src = json.load(open(SRC))
assert len(src) == 978
is_chapter = lambda e: e["messages"][1]["content"].strip() == "Write the next chapter."
out, dropped = [], []
for e in src:
    if is_chapter(e):
        out.append(e); continue
    r = reuse(e["messages"][2]["content"])
    if r == r and r > P90:          # scorable and above the chapter p90
        dropped.append((e["id"], round(r, 1)))
    else:
        out.append(e)

n_ch = sum(is_chapter(e) for e in out)
n_br = len(out) - n_ch
print(f"chapter p90 (v2_1, window 4): {P90:.1f}")
print(f"kept {len(out)} = {n_ch} chapter + {n_br} brief;  dropped {len(dropped)} briefs")
kept_r = [reuse(e["messages"][2]["content"]) for e in out if not is_chapter(e)]
kept_ok = sorted(x for x in kept_r if x == x)
print(f"kept-brief reuse: scorable {len(kept_ok)}  median {st.median(kept_ok):.1f}  "
      f"max {max(kept_ok):.1f}  share>p90 {sum(x > P90 for x in kept_ok)/len(kept_ok):.2f}")
json.dump(out, open(DST, "w"), ensure_ascii=False, indent=1)
sha = hashlib.sha256(open(DST, "rb").read()).hexdigest()[:16]
print(f"wrote {DST}  sha256[:16] {sha}")
json.dump(dict(p90=P90, dropped=dropped, n_out=len(out), sha16=sha),
          open("/workspace/driftenginelite/eval/corpus_v2_4_dropped.json", "w"), indent=1)
