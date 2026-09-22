#!/usr/bin/env python3
"""Boundary-MODE probe (sixteenth run, next step; static, costs no generations).

The sixteenth run established that the adapter's most probable continuation at
a sentence boundary is a rung: at T=0.4 every sample enters a verbatim cycle
within 62-204 words. The boundary-entropy probe (fifteenth run) measured the
SPREAD at boundaries; this measures the MODE. Teacher-force the same 14 clean
base-written chapter texts under the chapter prompt through base / F / G (/ H)
and record, at every sentence boundary, the top-1 next token -- the sentence
opening the model would take greedily -- plus its probability and the entropy.
Same prefix, same position, all arms: paired by design.

Per-arm aggregates (the pre-registered instruments):
  mode_reuse4   share of boundaries whose greedy opening equals the greedy
                opening at any of the previous 4 boundaries of the same text
                (the first-word-reuse instrument, applied to the mode instead
                of to sampled text; pooled over texts, denominator n_b - 1)
  top3_share    the 3 most frequent greedy openings' share of all boundaries
  n_distinct    distinct greedy openings per 100 boundaries
  pronoun_share share of greedy openings in {he she i it they we you}
  median_H      median boundary entropy (replicates the fifteenth run)

Usage: boundary_mode.py [H=/path/to/adapter]   (H is optional; base/F/G always)
Output: eval/boundary_mode.json (per-boundary records) + printed table."""
import json, re, sys, time, hashlib, collections, statistics as st
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gen_guarded import BASE, SYSTEM, USER, SYSTEM_SHA
assert hashlib.sha256(SYSTEM.encode()).hexdigest() == SYSTEM_SHA
ADAPTERS = {"F": "/workspace/drift_sft_out_v6/adapter", "G": "/workspace/drift_sft_out_v7/adapter"}
for a in sys.argv[1:]:
    k, p = a.split("=", 1); ADAPTERS[k] = p
OUT = "/workspace/driftenginelite/eval/boundary_mode.json"
texts = [(f"aug{r['i']}", r["text"]) for r in json.load(open("/workspace/gen_base_control.json"))["results"]] + \
        [(f"sep{r['i']}", r["text"]) for r in json.load(open("/workspace/driftenginelite/eval/gen_base_n10_cap2560.json"))["results"]]
PRON = {"he", "she", "i", "it", "they", "we", "you"}

import torch
from unsloth import FastLanguageModel
from peft import PeftModel
model, tok = FastLanguageModel.from_pretrained(BASE, max_seq_length=4096, load_in_4bit=True, dtype=None)
names = list(ADAPTERS)
model = PeftModel.from_pretrained(model, ADAPTERS[names[0]], adapter_name=names[0])
for n in names[1:]:
    model.load_adapter(ADAPTERS[n], adapter_name=n)
FastLanguageModel.for_inference(model)
END = re.compile(r'[.!?"”]\s*$')

def boundaries(ids, plen):
    strs = tok.convert_ids_to_tokens(ids)
    out = []
    for t in range(plen + 1, len(ids)):
        s = tok.convert_tokens_to_string([strs[t]])
        if not (s.startswith(" ") and len(s) > 1 and (s[1].isalpha() or s[1] in '"“\'')): continue
        prev = tok.decode(ids[max(plen, t - 12):t])
        if END.search(prev): out.append(t)
    return out

def run(ids, pos):
    with torch.no_grad():
        logits = model(ids).logits[0]
    out = []
    for t in pos:
        lp = torch.log_softmax(logits[t - 1].float(), -1)
        p = lp.exp()
        H = -(p * lp).sum().item()
        top5 = torch.topk(p, 5)
        out.append(dict(t=t, H=H, top1=int(top5.indices[0]), top1_str=tok.decode([int(top5.indices[0])]),
                        top1_p=float(top5.values[0]),
                        top5=[(tok.decode([int(i)]), float(v)) for i, v in zip(top5.indices, top5.values)],
                        true=int(ids[0, t]), true_lp=lp[ids[0, t]].item()))
    return out

results = []
msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]
prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
plen = tok(prefix, return_tensors="pt").input_ids.shape[1]
assert plen == 54
arms = ["base"] + names
for name, text in texts:
    t0 = time.time()
    ids = tok(prefix + text, return_tensors="pt").input_ids.to(model.device)
    pos = boundaries(ids[0].tolist(), plen)
    rec = dict(text=name, n_tokens=ids.shape[1], n_boundaries=len(pos))
    with model.disable_adapter():
        rec["base"] = run(ids, pos)
    for arm in names:
        model.set_adapter(arm)
        rec[arm] = run(ids, pos)
    if pos:  # falsifier: adapters live and distinct at the first boundary
        assert all(rec[a][0]["H"] != rec["base"][0]["H"] for a in names)
    results.append(rec)
    print(f"{name:6} {ids.shape[1]:5} tok  {len(pos):3} boundaries  " +
          "  ".join(f"{a} top1={rec[a][0]['top1_str']!r}" for a in arms) + f"  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(dict(texts=[n for n, _ in texts], adapters=ADAPTERS, base=BASE, arms=arms, results=results), open(OUT, "w"))
print("Saved", OUT, flush=True)

# ---- aggregates
print(f"\n{'arm':5} {'n_b':>5} {'mode_reuse4':>11} {'top3_share':>10} {'distinct/100':>12} {'pronoun':>8} {'median_H':>8} {'median_top1p':>12}   top-5 greedy openings")
for a in arms:
    reuse_n = reuse_d = 0; opens = []; Hs = []; ps = []
    for rec in results:
        g = [x["top1"] for x in rec[a]]
        opens += [x["top1_str"] for x in rec[a]]
        Hs += [x["H"] for x in rec[a]]; ps += [x["top1_p"] for x in rec[a]]
        for i in range(1, len(g)):
            reuse_d += 1; reuse_n += g[i] in g[max(0, i - 4):i]
    c = collections.Counter(opens); n = len(opens)
    top3 = sum(v for _, v in c.most_common(3)) / n
    pron = sum(o.strip().lower() in PRON for o in opens) / n
    print(f"{a:5} {n:>5} {100*reuse_n/reuse_d:>11.1f} {100*top3:>10.1f} {100*len(c)/n:>12.1f} {100*pron:>8.1f} {st.median(Hs):>8.2f} {st.median(ps):>12.2f}   " +
          " ".join(f"{k.strip()!r}:{v}" for k, v in c.most_common(5)))
