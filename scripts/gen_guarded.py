#!/usr/bin/env python3
"""The guard run: ten chapter-prompt samples with a repetition guard, seed-paired
to the recorded unguarded arm (gen_v6_cap2560.json).

Motivation and prediction are on record in docs/EXPERIMENT_LOG.md (eighth and
ninth runs): the anaphora instrument is closed, a sampler guard is argued only
on cheap-intervention grounds (loops in 5/10 recorded chapter samples), and it
must be scored on LOOP INCIDENCE and REGISTER, not ELEV. Prediction: exit-B
cap-loops become clean EOS stops at ~250-450 words; length is NOT recovered
(sustainment is a training-side problem, OPEN 3).

One-change discipline: sampling config is byte-identical to the recorded arm
(do_sample, temperature 0.7, min_p 0.05, repetition_penalty 1.05, top_k 0,
top_p 1.0, cap 2560, master seed 20260826, per-sample seed = master + i);
the ONLY change is `no_repeat_ngram_size` (default 6). Register scoring stays
with the canonical pod instruments (register_check.py / amplification_test.py)
over the saved texts; the anaphora numbers printed here are a stand-in using
the fifth run's definitions and are labelled as such.

Falsifiers, run before any budget is spent:
  F1  system prompt sha256 == ed40b81d... and chapter prefix == 54 tokens,
      token-identical to every recorded arm.
  F2  adapter is live: max |logits(adapter) - logits(disabled)| > 0 on the
      prefix; disabled arm is bit-identical base.
  F3  (costs one sample, --skip-repro to spend nothing) guard OFF at seed
      master+1 must reproduce recorded gen1 exactly (sha256 79a6a7f0e0b64ac2).
      If it does not, seed-pairing with the recorded arm does not hold on this
      host; the run continues — the verdict needs loop incidence, not pairing —
      but the JSON records pairing as broken.
"""
import argparse, hashlib, json, re, sys, time

BASE    = "unsloth/Qwen3-14B-unsloth-bnb-4bit"
ADAPTER = "/workspace/drift_sft_out_v6/adapter"
SYSTEM  = ("Write in the mode of objective physical realism. Describe actions, "
           "environments, and labor with precision.\n"
           "Output is continuous prose. No headers, labels, or formatting.")
SYSTEM_SHA = "ed40b81db82c04bf1145a7d362c439b189f671588e03d6f8f7ea41eab5a8426a"
USER    = "Write the next chapter."
PREFIX_TOKENS = 54
MASTER_SEED = 20260826
KW = dict(do_sample=True, temperature=0.7, min_p=0.05, repetition_penalty=1.05,
          top_k=0, top_p=1.0)
GEN1_SHA = "79a6a7f0e0b64ac2"

def sha16(s): return hashlib.sha256(s.encode()).hexdigest()[:16]

def max_repeat_span(words):
    # Longest word-span occurring twice (rolling-hash binary search); matches the
    # recorded bookkeeping: novel_words = word_count - max_repeat_span.
    def has(L):
        if L == 0: return True
        seen = {}
        h = 0; B = 1000003; M = (1 << 61) - 1; pw = pow(B, L - 1, M)
        ids = {}
        w = [ids.setdefault(x, len(ids)) for x in words]
        for i, x in enumerate(w):
            h = (h * B + x) % M
            if i >= L: h = (h - w[i - L] * pw * B) % M
            if i >= L - 1:
                j = seen.setdefault(h, i - L + 1)
                if j != i - L + 1 and words[j:j + L] == words[i - L + 1:i + 1]:
                    return True
        return False
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if has(mid): lo = mid
        else: hi = mid - 1
    return lo

_SENT = re.compile(r'(?<=[.!?"”])\s+')
def anaphora(text):
    # Fifth run's definitions, stand-in implementation: share of adjacent sentence
    # pairs sharing their first two words, and the longest run of consecutive
    # same-opening sentences. Canonical numbers come from amplification_test.py.
    sents = [s for s in _SENT.split(text) if s.strip()]
    opens = []
    for s in sents:
        ws = re.findall(r"[\w']+", s.lower())
        if len(ws) >= 2: opens.append((ws[0], ws[1]))
    if len(opens) < 2: return 0.0, 1, len(opens)
    same = [a == b for a, b in zip(opens, opens[1:])]
    rate = 100.0 * sum(same) / len(same)
    run = best = 1
    for m in same:
        run = run + 1 if m else 1
        best = max(best, run)
    return rate, best, len(opens)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="samples to generate (the budget)")
    ap.add_argument("--ngram", type=int, default=6, help="no_repeat_ngram_size; 0 disables the guard")
    ap.add_argument("--out", default="/workspace/gen_v6_guard.json")
    ap.add_argument("--gens", default="/workspace/gen_v6_cap2560.json",
                    help="recorded unguarded arm, for the paired table")
    ap.add_argument("--master-seed", type=int, default=MASTER_SEED)
    ap.add_argument("--max-new-tokens", type=int, default=2560)
    ap.add_argument("--skip-repro", action="store_true", help="skip falsifier F3 (saves one sample)")
    args = ap.parse_args()

    assert hashlib.sha256(SYSTEM.encode()).hexdigest() == SYSTEM_SHA, "F1: system prompt drifted"

    import torch
    from unsloth import FastLanguageModel
    from peft import PeftModel
    model, tok = FastLanguageModel.from_pretrained(BASE, max_seq_length=4096,
                                                   load_in_4bit=True, dtype=None)
    model = PeftModel.from_pretrained(model, ADAPTER)
    FastLanguageModel.for_inference(model)

    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]
    prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    ids = tok(prefix, return_tensors="pt").input_ids.to(model.device)
    assert ids.shape[1] == PREFIX_TOKENS, f"F1: prefix is {ids.shape[1]} tokens, recorded arms used {PREFIX_TOKENS}"

    with torch.no_grad():
        live = model(ids).logits[0, -1]
        with model.disable_adapter():
            dis = model(ids).logits[0, -1]
    delta = (live - dis).abs().max().item()
    assert delta > 0, "F2: adapter arm is not live"
    print(f"falsifier F1: system sha ok, prefix {PREFIX_TOKENS} tokens")
    print(f"falsifier F2: adapter live, max |logit delta| = {delta:.3f}")

    def generate(seed, ngram):
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        kw = dict(KW, max_new_tokens=args.max_new_tokens,
                  eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
        if ngram: kw["no_repeat_ngram_size"] = ngram
        with torch.no_grad():
            out = model.generate(ids, **kw)[0][ids.shape[1]:]
        finish = "EOS" if out[-1].item() == tok.eos_token_id else "CAP"
        return tok.decode(out, skip_special_tokens=True), finish, len(out)

    pairing = None
    if not args.skip_repro:
        t0 = time.time()
        text, _, _ = generate(args.master_seed + 1, ngram=0)
        pairing = (sha16(text) == GEN1_SHA)
        print(f"falsifier F3: guard-off seed {args.master_seed + 1} sha {sha16(text)} "
              f"{'== recorded gen1 — seed-pairing holds' if pairing else '!= ' + GEN1_SHA + ' — PAIRING BROKEN on this host (run continues)'}"
              f" ({time.time() - t0:.0f}s)")

    results = []
    for i in range(1, args.n + 1):
        t0 = time.time()
        text, finish, ntok = generate(args.master_seed + i, ngram=args.ngram)
        words = text.split()
        span = max_repeat_span(words)
        novel = len(words) - span
        rate, run, nsent = anaphora(" ".join(words[:novel]) if span else text)
        results.append(dict(i=i, seed=args.master_seed + i, finish_reason=finish,
                            new_tokens=ntok, word_count=len(words), max_repeat_span=span,
                            novel_words=novel, anaphora_rate_standin=round(rate, 1),
                            max_same_open_run_standin=run, n_sent=nsent,
                            sha256=sha16(text), text=text))
        print(f"[{i}/{args.n}] seed {args.master_seed + i}  {finish}  {len(words)}w  "
              f"repeat_span {span}  anaph~{rate:.1f}%  run~{run}  ({time.time() - t0:.0f}s)")

    try:
        rec = {r["i"]: r for r in json.load(open(args.gens))["results"]}
    except Exception:
        rec = {}
    print(f"\n{'i':>2} {'unguarded':>18} {'guarded':>18}")
    for r in results:
        u = rec.get(r["i"])
        left = f"{u['finish_reason']} {u['word_count']}w span {u['max_repeat_span']}" if u else "-"
        print(f"{r['i']:>2} {left:>18} {r['finish_reason']} {r['word_count']}w span {r['max_repeat_span']:>5}")
    loops = sum(1 for r in results if r["finish_reason"] == "CAP" or
                r["max_repeat_span"] > 0.2 * r["word_count"])
    eos_band = sum(1 for r in results if r["finish_reason"] == "EOS")
    print(f"\nloops {loops}/{len(results)} (recorded arm: 5/10)   EOS {eos_band}/{len(results)}")
    print("Score register and canonical anaphora with register_check.py / "
          "amplification_test.py over the saved texts before drawing the verdict.")

    json.dump(dict(variant="F", adapter=ADAPTER, base=BASE, system_sha256=SYSTEM_SHA,
                   user=USER, kw=KW, master_seed=args.master_seed,
                   max_new_tokens=args.max_new_tokens, no_repeat_ngram_size=args.ngram,
                   falsifier_adapter_live=delta, falsifier_prefix_tokens=PREFIX_TOKENS,
                   falsifier_seed_pairing=pairing, results=results),
              open(args.out, "w"), indent=1)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    sys.exit(main())
