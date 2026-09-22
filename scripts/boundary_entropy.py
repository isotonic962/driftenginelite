#!/usr/bin/env python3
"""Boundary-entropy probe (fourteenth run, next step 1; prediction in
logs/prediction_boundary_entropy.txt). Teacher-force clean chapter-length texts
(the 14 base controls) under the chapter prompt through base / F / G and record,
at every sentence boundary, the entropy of the next-token distribution and the
top-1 probability. Same prefix, same position, three models: paired by design.
Costs no generations."""
import json, math, re, sys, time
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gen_guarded import BASE, SYSTEM, USER, SYSTEM_SHA
import hashlib
assert hashlib.sha256(SYSTEM.encode()).hexdigest() == SYSTEM_SHA
F_ADAPTER = "/workspace/drift_sft_out_v6/adapter"
G_ADAPTER = "/workspace/drift_sft_out_v7/adapter"
OUT = "/workspace/driftenginelite/eval/boundary_entropy.json"
texts = [(f"aug{r['i']}", r["text"]) for r in json.load(open("/workspace/gen_base_control.json"))["results"]] + \
        [(f"sep{r['i']}", r["text"]) for r in json.load(open("/workspace/driftenginelite/eval/gen_base_n10_cap2560.json"))["results"]]
NONBOUNDARY = "nonboundary" in sys.argv[1:]
if NONBOUNDARY: OUT = OUT.replace(".json", "_nb.json")
if len(sys.argv) > 1 and sys.argv[1].startswith("corpus:"):
    # mirror-confound arm: N corpus chapter targets (adapter-manifold text), fixed-seed sample
    import random
    N = int(sys.argv[1].split(":")[1])
    corpus = json.load(open("/workspace/final_training_corpus_v2_2_bq.json"))
    ch = [(e["id"], e["messages"][2]["content"]) for e in corpus if e["messages"][1]["content"].strip() == "Write the next chapter."]
    random.Random(20260922).shuffle(ch)
    texts = ch[:N]
    OUT = f"/workspace/driftenginelite/eval/boundary_entropy_corpus{N}.json"

import torch
from unsloth import FastLanguageModel
from peft import PeftModel
model, tok = FastLanguageModel.from_pretrained(BASE, max_seq_length=4096, load_in_4bit=True, dtype=None)
model = PeftModel.from_pretrained(model, G_ADAPTER, adapter_name="G")
model.load_adapter(F_ADAPTER, adapter_name="F")
FastLanguageModel.for_inference(model)
END = re.compile(r'[.!?"”]\s*$')

def boundaries(ids, plen):
    """token indices t (>= plen) such that the token at t starts a new sentence:
    decoded text up to t ends in sentence-final punctuation, token t starts ' <word>'."""
    strs = tok.convert_ids_to_tokens(ids)
    out = []
    for t in range(plen + 1, len(ids)):
        s = tok.convert_tokens_to_string([strs[t]])
        if not (s.startswith(" ") and len(s) > 1 and (s[1].isalpha() or s[1] in '"“\'')): continue
        prev = tok.decode(ids[max(plen, t - 12):t])
        if END.search(prev): out.append(t)
    return out

def run(arm, ids, pos):
    with torch.no_grad():
        logits = model(ids).logits[0]
    out = []
    for t in pos:
        lp = torch.log_softmax(logits[t - 1].float(), -1)
        p = lp.exp()
        H = -(p * lp).sum().item()
        top = p.max().item()
        out.append(dict(t=t, H=H, top1=top, true_lp=lp[ids[0, t]].item()))
    return out

results = []
msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]
prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
plen = tok(prefix, return_tensors="pt").input_ids.shape[1]
assert plen == 54
for name, text in texts:
    t0 = time.time()
    ids = tok(prefix + text, return_tensors="pt").input_ids.to(model.device)
    pos = boundaries(ids[0].tolist(), plen)
    nb = []
    if NONBOUNDARY:
        # control: word-initial (space-led alphabetic) tokens that do NOT start a sentence, same count as boundaries
        import random
        strs = tok.convert_ids_to_tokens(ids[0].tolist()); bset = set(pos)
        cand = [t for t in range(plen + 1, ids.shape[1]) if t not in bset and
                (lambda x: x.startswith(" ") and len(x) > 1 and x[1].isalpha())(tok.convert_tokens_to_string([strs[t]]))]
        nb = sorted(random.Random(20260922).sample(cand, min(len(cand), len(pos))))
    pos_all = sorted(set(pos) | set(nb))
    words_at = {t: len(tok.decode(ids[0, plen:t]).split()) for t in pos_all}
    rec = dict(text=name, n_tokens=ids.shape[1], n_boundaries=len(pos), nonboundary=nb,
               words_at={str(t): w for t, w in words_at.items()})
    pos = pos_all
    with model.disable_adapter():
        rec["base"] = run("base", ids, pos)
    for arm in ("F", "G"):
        model.set_adapter(arm)
        rec[arm] = run(arm, ids, pos)
    # falsifier: the two adapters differ from base and from each other at the first boundary
    if pos:
        assert rec["F"][0]["H"] != rec["base"][0]["H"] and rec["G"][0]["H"] != rec["F"][0]["H"]
    results.append(rec)
    med = lambda a: sorted(x["H"] for x in rec[a])[len(pos) // 2] if pos else float("nan")
    print(f"{name:6} {ids.shape[1]:5} tok  {len(pos):3} boundaries  median H  base {med('base'):.2f}  F {med('F'):.2f}  G {med('G'):.2f}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(dict(texts=[n for n, _ in texts], F=F_ADAPTER, G=G_ADAPTER, base=BASE, results=results), open(OUT, "w"))
print("Saved", OUT, flush=True)
