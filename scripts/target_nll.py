#!/usr/bin/env python3
"""Teacher-forced NLL on training targets -- the static stand-in for the brief
spot-check while the generation budget is spent (60/60). For each arm, the
mean per-token NLL of the assistant target given system+user, on three fixed
samples (seed 20260922) of corpus entries:
  kept     briefs retained in v2_4 (trained on by G and H)
  dropped  the 85 briefs removed from v2_4 (trained on by G only)
  chapter  chapter targets (trained on by both, x3)
Reads: if H's NLL on kept briefs is within ~0.05 nats of G's, the brief branch
is not visibly changed by the filter; H > G on dropped briefs is the sanity
check that the filter took effect (H never saw them).
Usage: target_nll.py [H=/path] [n=20]"""
import json, random, sys, time, statistics as st
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gen_guarded import BASE
ADAPTERS = {"G": "/workspace/drift_sft_out_v7/adapter"}
N = 20
for a in sys.argv[1:]:
    k, v = a.split("=", 1)
    if k == "n": N = int(v)
    else: ADAPTERS[k] = v
corpus = json.load(open("/workspace/final_training_corpus_v2_3_ch3x.json"))
dropped_ids = {i for i, _ in json.load(open("/workspace/driftenginelite/eval/corpus_v2_4_dropped.json"))["dropped"]}
is_ch = lambda e: e["messages"][1]["content"].strip() == "Write the next chapter."
sets = dict(kept=[e for e in corpus if not is_ch(e) and e["id"] not in dropped_ids],
            dropped=[e for e in corpus if e["id"] in dropped_ids],
            chapter=[e for e in corpus if is_ch(e) and "#x" not in e["id"]])
rng = random.Random(20260922)
for k in sets: rng.shuffle(sets[k]); sets[k] = sets[k][:N]

import torch
from unsloth import FastLanguageModel
from peft import PeftModel
model, tok = FastLanguageModel.from_pretrained(BASE, max_seq_length=4096, load_in_4bit=True, dtype=None)
names = list(ADAPTERS)
model = PeftModel.from_pretrained(model, ADAPTERS[names[0]], adapter_name=names[0])
for n in names[1:]: model.load_adapter(ADAPTERS[n], adapter_name=n)
FastLanguageModel.for_inference(model)

def nll(e):
    msgs = [{"role": m["role"], "content": m["content"]} for m in e["messages"]]
    full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    pre = tok.apply_chat_template(msgs[:2], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(full, return_tensors="pt").input_ids.to(model.device)
    plen = tok(pre, return_tensors="pt").input_ids.shape[1]
    with torch.no_grad():
        lp = torch.log_softmax(model(ids).logits[0, plen - 1:-1].float(), -1)
    tgt = ids[0, plen:]
    return -lp[torch.arange(len(tgt)), tgt].mean().item(), len(tgt)

out = {}
for setname, es in sets.items():
    for arm in ["base"] + names:
        vals = []
        for e in es:
            if arm == "base":
                with model.disable_adapter(): v, n = nll(e)
            else:
                model.set_adapter(arm); v, n = nll(e)
            vals.append(v)
        out[f"{setname}/{arm}"] = vals
        print(f"{setname:8} {arm:5} n={len(vals):2}  mean NLL {st.mean(vals):.3f}  median {st.median(vals):.3f}", flush=True)
json.dump(dict(adapters=ADAPTERS, n=N, ids={k: [e["id"] for e in v] for k, v in sets.items()}, nll=out),
          open("/workspace/driftenginelite/eval/target_nll.json", "w"), indent=1)
print("Saved eval/target_nll.json")
