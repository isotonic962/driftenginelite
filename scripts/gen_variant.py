#!/usr/bin/env python3
"""Evaluate a retrained adapter variant: N unguarded chapter-prompt samples with
the recorded arms' sampling config (so `score_guard.py` compares it directly to
variant F's gen_v6_cap2560.json), then the sixth run's three held-out briefs as
the brief-branch regression spot-check the charter's verdict discipline requires.

Nothing here changes sampling: temperature 0.7, min_p 0.05, repetition_penalty
1.05, cap 2560, per-sample seed = master + i, no guard of any kind.
Falsifiers F1 (system sha, 54-token chapter prefix) and F2 (adapter live) run
before any budget is spent. Seed-pairing with the recorded arms is known broken
on this host (tenth run, F3), so comparisons are unpaired.
"""
import argparse, hashlib, json, sys, time

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gen_guarded import (BASE, SYSTEM, SYSTEM_SHA, USER, PREFIX_TOKENS, MASTER_SEED,
                         KW, sha16, max_repeat_span, anaphora)

BRIEFS = ["Einar has asked Elsa whether she is all right.",
          "Astrid has just told Bertil something difficult.",
          "Sigrid has called Lars to come to bed."]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--briefs", type=int, default=3)
    ap.add_argument("--master-seed", type=int, default=MASTER_SEED)
    ap.add_argument("--max-new-tokens", type=int, default=2560)
    args = ap.parse_args()
    assert hashlib.sha256(SYSTEM.encode()).hexdigest() == SYSTEM_SHA, "F1: system prompt drifted"

    import torch
    from unsloth import FastLanguageModel
    from peft import PeftModel
    model, tok = FastLanguageModel.from_pretrained(BASE, max_seq_length=4096,
                                                   load_in_4bit=True, dtype=None)
    model = PeftModel.from_pretrained(model, args.adapter)
    FastLanguageModel.for_inference(model)

    def encode(user):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
        return tok(p, return_tensors="pt").input_ids.to(model.device)

    ids = encode(USER)
    assert ids.shape[1] == PREFIX_TOKENS, f"F1: prefix is {ids.shape[1]} tokens"
    with torch.no_grad():
        live = model(ids).logits[0, -1]
        with model.disable_adapter():
            dis = model(ids).logits[0, -1]
    delta = (live - dis).abs().max().item()
    assert delta > 0, "F2: adapter arm is not live"
    print(f"falsifier F1: system sha ok, prefix {PREFIX_TOKENS} tokens", flush=True)
    print(f"falsifier F2: adapter live, max |logit delta| = {delta:.3f}", flush=True)

    def generate(ids, seed):
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        kw = dict(KW, max_new_tokens=args.max_new_tokens,
                  eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
        with torch.no_grad():
            out = model.generate(ids, **kw)[0][ids.shape[1]:]
        finish = "EOS" if out[-1].item() == tok.eos_token_id else "CAP"
        return tok.decode(out, skip_special_tokens=True), finish, len(out)

    def record(i, seed, text, finish, ntok, **extra):
        words = text.split()
        span = max_repeat_span(words)
        novel = len(words) - span
        rate, run, nsent = anaphora(" ".join(words[:novel]) if span else text)
        return dict(i=i, seed=seed, finish_reason=finish, new_tokens=ntok,
                    word_count=len(words), max_repeat_span=span, novel_words=novel,
                    anaphora_rate_standin=round(rate, 1), max_same_open_run_standin=run,
                    n_sent=nsent, sha256=sha16(text), text=text, **extra)

    results, briefs = [], []
    def save():
        json.dump(dict(adapter=args.adapter, base=BASE, system_sha256=SYSTEM_SHA, user=USER,
                       kw=KW, master_seed=args.master_seed, max_new_tokens=args.max_new_tokens,
                       falsifier_adapter_live=delta, results=results, briefs=briefs),
                  open(args.out, "w"), indent=1)

    for i in range(1, args.n + 1):
        t0 = time.time()
        text, finish, ntok = generate(ids, args.master_seed + i)
        r = record(i, args.master_seed + i, text, finish, ntok); results.append(r); save()
        print(f"[chapter {i}/{args.n}] {finish}  {r['word_count']}w  repeat_span {r['max_repeat_span']}  "
              f"anaph~{r['anaphora_rate_standin']}%  run~{r['max_same_open_run_standin']}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    for b, user in enumerate(BRIEFS[:args.briefs], 1):
        t0 = time.time()
        text, finish, ntok = generate(encode(user), args.master_seed + b)
        r = record(b, args.master_seed + b, text, finish, ntok, user=user,
                   in_corpus_brief_band=9 <= len(text.split()) <= 171,
                   has_blockquote_marker="> " in text)
        briefs.append(r); save()
        print(f"[brief {b}] {finish}  {r['word_count']}w  band {r['in_corpus_brief_band']}  "
              f"'> ' {r['has_blockquote_marker']}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
