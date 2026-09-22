#!/usr/bin/env python3
"""Find T* such that variant G's median boundary entropy on the 14 base texts at T*
equals base's median boundary entropy at T=0.7 (the recorded sampling temperature).
Saves the per-boundary logits' entropies on a T grid to eval/match_temperature.json."""
import json, re, sys, torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gen_guarded import BASE, SYSTEM, USER
from unsloth import FastLanguageModel
from peft import PeftModel
G_ADAPTER = "/workspace/drift_sft_out_v7/adapter"
texts = [r["text"] for r in json.load(open("/workspace/gen_base_control.json"))["results"]] + \
        [r["text"] for r in json.load(open("/workspace/driftenginelite/eval/gen_base_n10_cap2560.json"))["results"]]
model, tok = FastLanguageModel.from_pretrained(BASE, max_seq_length=4096, load_in_4bit=True, dtype=None)
model = PeftModel.from_pretrained(model, G_ADAPTER); FastLanguageModel.for_inference(model)
END = re.compile(r'[.!?"”]\s*$')
def boundaries(ids, plen):
    strs = tok.convert_ids_to_tokens(ids); out = []
    for t in range(plen + 1, len(ids)):
        s = tok.convert_tokens_to_string([strs[t]])
        if not (s.startswith(" ") and len(s) > 1 and (s[1].isalpha() or s[1] in '"“\'')): continue
        if END.search(tok.decode(ids[max(plen, t - 12):t])): out.append(t)
    return out
msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]
prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
plen = tok(prefix, return_tensors="pt").input_ids.shape[1]
TS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
def H(logits, T):
    lp = torch.log_softmax(logits.float() / T, -1); return -(lp.exp() * lp).sum(-1)
acc = {"base": {T: [] for T in TS}, "G": {T: [] for T in TS}}
for text in texts:
    ids = tok(prefix + text, return_tensors="pt").input_ids.to(model.device)
    pos = boundaries(ids[0].tolist(), plen)
    with torch.no_grad():
        with model.disable_adapter(): lb = model(ids).logits[0][[t - 1 for t in pos]]
        lg = model(ids).logits[0][[t - 1 for t in pos]]
    for T in TS:
        acc["base"][T] += H(lb, T).tolist(); acc["G"][T] += H(lg, T).tolist()
import statistics as st
target = st.median(acc["base"][0.7])
print(f"base median boundary H at T=0.7: {target:.3f} nats  (n={len(acc['base'][0.7])})")
for T in TS:
    print(f"  T={T:.2f}  G median H {st.median(acc['G'][T]):.3f}   base median H {st.median(acc['base'][T]):.3f}")
best = min(TS, key=lambda T: abs(st.median(acc["G"][T]) - target))
print(f"T* (G matches base@0.7): {best}")
json.dump(dict(target=target, grid={str(T): dict(base=st.median(acc["base"][T]), G=st.median(acc["G"][T])) for T in TS}, T_star=best),
          open("/workspace/driftenginelite/eval/match_temperature.json", "w"), indent=1)
