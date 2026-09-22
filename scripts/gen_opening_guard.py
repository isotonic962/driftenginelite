#!/usr/bin/env python3
"""The opening-guard run: chapter-prompt samples with a SENTENCE-OPENING guard,
seed-paired to the recorded unguarded arm (gen_v6_cap2560.json) and to the
n-gram guard arm (gen_v6_guard.json).

Motivation (docs/EXPERIMENT_LOG.md, runs 7-9): the ladder is a model-agnostic
induction capture with no repeat pressure in the weights; the adapter's only
contribution is upstream of the first rung. `no_repeat_ngram_size` blocks the
verbatim cycle (exit B) but passes the ladder itself, which is fuzzy: "He was
afraid of the dark. He was afraid of himself." shares an opening, not a 6-gram.
This guard blocks the first rung instead: a new sentence may not reuse the
two-word opening of any of the previous --window sentences.

Static cost, measured before spending budget (canonical TextureAnalyzer
splitter, window 4): the rule would touch 3.7% of corpus long-form sentences,
0.6% of base's chapter generations, 34.6% of the adapter's de-looped ones.

One-change discipline: sampling config byte-identical to the recorded arms;
the ONLY change is the logits processor (plus --ngram if explicitly asked for).
Falsifiers F1/F2 as in gen_guarded.py. F4: with --window 0 the processor is a
no-op and is not installed.
"""
import argparse, json, re, sys, time, hashlib

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gen_guarded import (BASE, ADAPTER, SYSTEM, SYSTEM_SHA, USER, PREFIX_TOKENS,
                         MASTER_SEED, KW, sha16, max_repeat_span, anaphora)

_SENT = re.compile(r'(?<=[.!?"”\n])\s+')
TAIL_TOKENS = 512   # decoded context per step; 4 sentences + the current one never need more
_WORD = re.compile(r"[\w']+")


def build_vocab_maps(tok):
    """word -> token ids that ARE that word: space-led (start a new word) and
    bare (continue/start without a space)."""
    led, bare = {}, {}
    strs = tok.batch_decode([[i] for i in range(len(tok))])
    for i, s in enumerate(strs):
        m = re.fullmatch(r"(\s*)([\w']+)", s)
        if not m: continue
        (led if m.group(1) else bare).setdefault(m.group(2).lower(), []).append(i)
    return led, bare, strs


def make_processor(tok, prompt_len, window, log):
    import torch
    from transformers import LogitsProcessor
    led, bare, _ = build_vocab_maps(tok)

    class OpeningGuard(LogitsProcessor):
        def __call__(self, input_ids, scores):
            # Only the tail is needed (window sentences + the current one); decoding
            # the whole text every step is O(n^2) and costs ~50 ms/step at 1.2k tokens.
            start = max(prompt_len, input_ids.shape[1] - TAIL_TOKENS)
            text = tok.decode(input_ids[0, start:], skip_special_tokens=True)
            sents = [s for s in _SENT.split(text)]
            if len(sents) < 2: return scores
            # a trailing boundary means the current sentence is empty
            cur = "" if re.search(r'[.!?"”\n]\s+$', text) else sents[-1]
            prev = [s for s in (sents if cur == "" else sents[:-1]) if s.strip()]
            banned = set()
            for s in prev[-window:]:
                ws = _WORD.findall(s.lower())
                if len(ws) >= 2: banned.add((ws[0], ws[1]))
            if not banned: return scores
            ws = _WORD.findall(cur.lower())
            ids = []
            if len(ws) == 1:
                # first word written; forbid the token that would complete a banned opening
                for w1, w2 in banned:
                    if w1 == ws[0]:
                        ids += led.get(w2, [])
                        if not cur[-1:].isalnum(): ids += bare.get(w2, [])
                        # w2 split across tokens: forbid nothing here, caught below
            elif len(ws) == 2 and cur[-1:].isalnum():
                # second word in progress: forbid the piece that completes it
                for w1, w2 in banned:
                    if w1 == ws[0] and w2.startswith(ws[1]) and w2 != ws[1]:
                        ids += bare.get(w2[len(ws[1]):], [])
            if ids:
                log["interventions"] += 1
                scores[0, torch.tensor(ids, device=scores.device)] = float("-inf")
            return scores

    return OpeningGuard()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--window", type=int, default=4, help="previous sentences whose openings are forbidden; 0 disables")
    ap.add_argument("--ngram", type=int, default=0, help="also set no_repeat_ngram_size (a second change; default off)")
    ap.add_argument("--scale", type=float, default=1.0, help="LoRA scaling multiplier (1.0 = recorded arms)")
    ap.add_argument("--adapter", default=ADAPTER)
    ap.add_argument("--user", default=USER)
    ap.add_argument("--out", default="/workspace/gen_v6_openguard.json")
    ap.add_argument("--master-seed", type=int, default=MASTER_SEED)
    ap.add_argument("--max-new-tokens", type=int, default=2560)
    args = ap.parse_args()

    assert hashlib.sha256(SYSTEM.encode()).hexdigest() == SYSTEM_SHA, "F1: system prompt drifted"
    import torch
    from unsloth import FastLanguageModel
    from peft import PeftModel
    from transformers import LogitsProcessorList
    model, tok = FastLanguageModel.from_pretrained(BASE, max_seq_length=4096,
                                                   load_in_4bit=True, dtype=None)
    model = PeftModel.from_pretrained(model, args.adapter)
    FastLanguageModel.for_inference(model)
    if args.scale != 1.0:
        n = 0
        for m in model.modules():
            if hasattr(m, "scaling") and isinstance(m.scaling, dict):
                for k in m.scaling: m.scaling[k] *= args.scale; n += 1
        print(f"scaled {n} LoRA layers by {args.scale}", flush=True)

    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": args.user}]
    prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    ids = tok(prefix, return_tensors="pt").input_ids.to(model.device)
    if args.user == USER:
        assert ids.shape[1] == PREFIX_TOKENS, f"F1: prefix is {ids.shape[1]} tokens"
    with torch.no_grad():
        live = model(ids).logits[0, -1]
        with model.disable_adapter():
            dis = model(ids).logits[0, -1]
    delta = (live - dis).abs().max().item()
    assert delta > 0, "F2: adapter arm is not live"
    print(f"falsifier F1: system sha ok, prefix {ids.shape[1]} tokens", flush=True)
    print(f"falsifier F2: adapter live, max |logit delta| = {delta:.3f}", flush=True)

    results = []
    for i in range(1, args.n + 1):
        t0 = time.time()
        seed = args.master_seed + i
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        log = dict(interventions=0)
        kw = dict(KW, max_new_tokens=args.max_new_tokens,
                  eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
        if args.ngram: kw["no_repeat_ngram_size"] = args.ngram
        if args.window:
            kw["logits_processor"] = LogitsProcessorList(
                [make_processor(tok, ids.shape[1], args.window, log)])
        with torch.no_grad():
            out = model.generate(ids, **kw)[0][ids.shape[1]:]
        finish = "EOS" if out[-1].item() == tok.eos_token_id else "CAP"
        text = tok.decode(out, skip_special_tokens=True)
        words = text.split()
        span = max_repeat_span(words)
        novel = len(words) - span
        rate, run, nsent = anaphora(" ".join(words[:novel]) if span else text)
        results.append(dict(i=i, seed=seed, finish_reason=finish, new_tokens=len(out),
                            word_count=len(words), max_repeat_span=span, novel_words=novel,
                            anaphora_rate_standin=round(rate, 1),
                            max_same_open_run_standin=run, n_sent=nsent,
                            guard_interventions=log["interventions"],
                            sha256=sha16(text), text=text))
        print(f"[{i}/{args.n}] seed {seed}  {finish}  {len(words)}w  repeat_span {span}  "
              f"anaph~{rate:.1f}%  run~{run}  guard fired {log['interventions']}x  "
              f"({time.time() - t0:.0f}s)", flush=True)
        json.dump(dict(variant="F", adapter=args.adapter, base=BASE, system_sha256=SYSTEM_SHA,
                       user=args.user, kw=KW, master_seed=args.master_seed,
                       max_new_tokens=args.max_new_tokens, opening_guard_window=args.window,
                       no_repeat_ngram_size=args.ngram, lora_scale=args.scale,
                       falsifier_adapter_live=delta, results=results),
                  open(args.out, "w"), indent=1)
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
