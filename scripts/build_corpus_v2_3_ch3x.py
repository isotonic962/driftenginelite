#!/usr/bin/env python3
"""Build the variant-G corpus: v2_2_bq (v2_1 with the '> ' contamination stripped)
plus the 138 chapter entries repeated x3 -> 414 chapter + 564 brief = 978.
Never overwrites. Output sha256[:16] on the pod: aa3b644fd68a00b6."""
import json, copy, os
SRC = "/workspace/final_training_corpus_v2_2_bq.json"
DST = "/workspace/final_training_corpus_v2_3_ch3x.json"
assert not os.path.exists(DST), "refusing to overwrite"
c = json.load(open(SRC)); out = list(c)
ch = [e for e in c if e["messages"][1]["content"].strip() == "Write the next chapter."]
assert len(c) == 702 and len(ch) == 138
for k in (2, 3):
    for e in ch:
        d = copy.deepcopy(e); d["id"] = f"{e['id']}#x{k}"; out.append(d)
json.dump(out, open(DST, "w"), ensure_ascii=False, indent=1)
print(len(out), "entries")
