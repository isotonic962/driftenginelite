#!/usr/bin/env python3
"""Actual sentence-opening distribution of texts (free, CPU). Same sentence split
and first-word rule as first_word_reuse.py. Reports the pronoun-opening share
(he she i it they we you) and the top openings, per text set. Written after the
seventeenth run's probe falsified the brief-tail hypothesis, to ask whether the
adapters' pronoun-heavy greedy mode is simply the corpus's own opening
distribution."""
import json, re, sys, collections
sys.path.insert(0, "/workspace/driftenginelite/Drift-engine"); sys.path.insert(0, "/workspace/driftenginelite/Drift-engine/scripts")
from register_check import deloop
SENT = re.compile(r'(?<=[.!?"”])\s+'); WORD = re.compile(r"[\w']+")
PRON = {"he", "she", "i", "it", "they", "we", "you"}
def openings(text):
    return [WORD.findall(s.lower())[0] for s in SENT.split(text) if WORD.search(s)]
corpus = json.load(open("/workspace/final_training_corpus_v2_3_ch3x.json"))
dropped = {i for i, _ in json.load(open("eval/corpus_v2_4_dropped.json"))["dropped"]}
is_ch = lambda e: e["messages"][1]["content"].strip() == "Write the next chapter."
sets = {
 "chapter targets (138)": [e["messages"][2]["content"] for e in corpus if is_ch(e) and "#x" not in e["id"]],
 "kept briefs (479)":     [e["messages"][2]["content"] for e in corpus if not is_ch(e) and e["id"] not in dropped],
 "dropped briefs (85)":   [e["messages"][2]["content"] for e in corpus if e["id"] in dropped],
 "base gens (14)":        [r["text"] for r in json.load(open("/workspace/gen_base_control.json"))["results"]] +
                          [r["text"] for r in json.load(open("eval/gen_base_n10_cap2560.json"))["results"]],
 "G gens, de-looped (10)":[deloop(r["text"])[0] for r in json.load(open("eval/gen_v7_variantG.json"))["results"]],
 "F gens, de-looped (10)":[deloop(r["text"])[0] for r in json.load(open("/workspace/gen_v6_cap2560.json"))["results"]],
}
print(f"{'set':26} {'sentences':>9} {'pronoun%':>8}  top openings")
for name, texts in sets.items():
    o = [w for t in texts for w in openings(t)]
    c = collections.Counter(o)
    print(f"{name:26} {len(o):>9} {100*sum(w in PRON for w in o)/len(o):>8.1f}  " + " ".join(f"{k}:{100*v/len(o):.0f}%" for k, v in c.most_common(6)))
