#!/usr/bin/env python3
"""Brief-branch regression spot-check for a retrained variant (charter verdict
discipline): the three held-out briefs from the sixth run, scored with the same
instruments, against the sixth run's recorded variant-F cells
(EOS 3/3, in-band 3/3, 33/57/96 words, anaphora 0.0, max run 2, int% 0.0)."""
import json, sys
sys.path.insert(0, "/workspace/driftenginelite/Drift-engine")
sys.path.insert(0, "/workspace/driftenginelite/Drift-engine/scripts")
from amplification_test import anaphora, int_pct
from register_check import deloop

REC_F = [dict(b=1, fin="EOS", w=33), dict(b=2, fin="EOS", w=57), dict(b=3, fin="EOS", w=96)]
d = json.load(open(sys.argv[1]))
print(f"{'b':>2} {'fin':>3} {'raw w':>6} {'delp w':>6} {'span':>5} {'band':>5} {'> ':>3} {'anaph%':>7} {'run':>4} {'int%':>5}   user")
for r in d["briefs"]:
    t, span = deloop(r["text"]); a, run, ns = anaphora(t); ip = int_pct(t)
    print(f"{r['i']:>2} {r['finish_reason']:>3} {r['word_count']:>6} {len(t.split()):>6} {span:>5} "
          f"{str(r['in_corpus_brief_band']):>5} {str(r['has_blockquote_marker']):>3} {a:>7.1f} {run:>4} {ip:>5.1f}   {r['user']}")
print("\nrecorded variant F (sixth run, same briefs, same seeds):",
      ", ".join(f"B{x['b']} {x['fin']} {x['w']}w" for x in REC_F), "| anaph 0.0, max run 2, int% 0.0, 3/3 in band")
for r in d["briefs"]:
    print(f"\n--- brief {r['i']} ({r['user']}) ---\n{r['text']}")
