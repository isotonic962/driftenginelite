# Short-output investigation — working notes

**Shared state between the RunPod session and the chat session.** Neither polls;
a human triggers both. Protocol:

- **Before starting work:** `git pull origin claude/short-output-investigation-iose1e`
  and read this file. It is the current state of the investigation.
- **After any run that produces a number:** append the result under RESULTS,
  move any hypothesis it settles into KILLED with the evidence, and push.
- **Never delete a killed hypothesis.** The record of what was ruled out and why
  is the point — it is what stops the next session re-running a dead test.
- Numbers only. If a claim has no run behind it, mark it `[unverified]`.

Last updated: 2026-08-26, after the RUN 0 scaling falsifier.

---

## THE PROBLEM

Variant E answers `"Write the next chapter."` with 354 words. Every one of the
139 training instances of that exact string produced 792–1494 words
(median 1225). No training example on that key was under 792 words.

---

## ESTABLISHED

### Corpus: `final_training_corpus_1.json`

| | |
|---|---|
| records | 739 |
| distinct system prompts | 1, byte-identical across all records (sha256 `ed40b81d…`, 169 chars) |
| distinct user turns | 567 |
| group-size histogram | `{1: 532, 2: 34, 139: 1}` |

System prompt, verbatim:

```
Write in the mode of objective physical realism. Describe actions, environments, and labor with precision.
Output is continuous prose. No headers, labels, or formatting.
```

| user turn | count | median | min | max |
|---|---|---|---|---|
| `"Write the next chapter."` | 139 | 1225 w | 792 w | 1494 w |
| all 566 others (600 records) | 1–2 each | 51 w | 4 w | 171 w |

Cleanly bimodal, **no overlap** — shortest long-form (792 w) is 4.6× the longest
short-form (171 w). Three independent definitions select the identical set:
L0/L1-prefixed IDs, user turn == `"Write the next chapter."`, assistant ≥200 words.

Assistant texts beginning with `"Chapter"`: **2 of 739** (L057, L058 — both
long-form, both violate the system prompt's no-headers line). 0.27%.

### Training — variant E

attention-only q/k/v/o · 2 epochs · dropout 0.05 · r=16 alpha=16 (scale 1.0) ·
`max_seq_length=2560` · output `/workspace/drift_sft_out_v5`

### Eval harness

system = the corpus's own two-line prompt, byte-identical (sha256 verified
in-script) · user = `"Write the next chapter."` · observed ~0.85 words/token

---

## RESULTS

### Cap 1200 — determinism probe

| run | seed | tokens | stopped on EOS | words | sha |
|---|---|---|---|---|---|
| A | 1234 | 432 | yes | 354 | `98ff94d4c843998a` |
| B | 1234 | 432 | yes | 354 | `98ff94d4c843998a` |
| C | fresh RNG | 1200 | **no — hit cap** | 1017 | `e8efd847…` |
| D | fresh RNG | 528 | yes | 423 | `487448cb…` |

A == B exactly; C ≠ D ≠ A. Sampling is live and seed-reproducible, not a stuck
decode. **C was censored by the cap** — the strongest sample in this table is
left-truncated and its true length is unknown.

Run A also printed a `Chapter 27` header, and its final 149 words are a
character-for-character repeat of the passage closing paragraph one.

### Cap 2048 — RUN 0, scaling falsifier (seed 1234, one sample per scale)

- All three shas differ → **LoRA scaling is live in the forward pass**, not just
  set on the attribute. The sweep measures something real.
- Scale 1.0 @ seed 1234 reproduces cap-1200 run A **byte-for-byte**
  (`98ff94d4c843998a`, 432 tok, 354 w) → harness validated end-to-end against the
  prior experiment; raising the cap cannot alter a sample that stopped on EOS at 432.

| scale | words | max repeat span |
|---|---|---|
| 1.0 | 354 | 100 |
| 1.5 | 126 | — |
| 2.0 | 109 | 2 |

**Word count falls as scale rises.** n=1 per arm at one seed.

### In flight

RUN 1 (10 samples @ scale 1.0, cap 2048) and RUN 2 (sweep, 4 each @ 1.0/1.5/2.0).
Log: `/workspace/gen_v5_cap2048.log` · JSON: `/workspace/gen_v5_cap2048.json`

---

## KILLED

1. **Eval system prompt ≠ trained system prompt.** Byte-identical, sha256 match.
   The re-run with SYSTEM read verbatim from the corpus was a confirmed no-op.
2. **Key dilution.** `{1: 532, 2: 34, 139: 1}` — the key is exclusive to the 139
   long-form entries. Nothing else carries it.
3. **Corpus header contamination.** 2 of 739. Noise, not a training signal.
   The `Chapter 27` in output is base-model behavior surviving both the LoRA and
   an explicit prompt instruction — which is evidence *for* weak adapter influence.
4. **Adapter under-applied / needs higher scale.** Scale up *shortens* output
   (354 → 126 → 109). The direction is opposite to the hypothesis.
5. **Degenerate or stuck decode.** A == B but C ≠ D ≠ A. Sampling varies.
6. **Training-side truncation.** 1494 w ≈ 2000 tok, fits inside `max_seq_length=2560`.
7. **Inference cap.** Was a real confound at 1200 (it censored C). Now 2048,
   which is ~1740 words at the observed rate and clears the 1494 training max.

---

## LIVE HYPOTHESIS

**The adapter learned brevity, and scale amplifies it.**

126 and 109 words land *inside* the short-form training range (median 51,
max 171). Scale 1.0 sits at 354, between the two modes.

600 of 739 EOS tokens in the corpus sit at the end of a short response. Every
example contributes exactly one EOS regardless of length, so the stop prior is
governed by **entry count** (81% short) while register is governed by **token
share** (83.5% long). The key being exclusive did not gate EOS.

The loop fits the same picture: at scale 1.0 the adapter is weak and output
drifts into base-model degeneration (long verbatim repeats); at 2.0 the adapter
dominates, no loop, but it terminates early because early termination is what it
learned. Scale is a dial between base degeneration and trained brevity. Neither
end is a chapter.

---

## OPEN — in priority order

1. **How was the training loss normalized — per-sequence or per-token?**
   If per-sequence, the 600 short examples outweigh the 139 long ones 4.3:1 in
   gradient, and the 83.5% token share never reached the optimizer. That would
   explain the whole result and it is a flag, not a corpus rebuild.
   *Static read of the training script. No GPU. Do this first.*
2. **Add scale 0.5 to the sweep.** Turns a two-point trend into a line. If word
   count keeps rising as scale falls, the adapter is unambiguously the source.
3. **What are the 600 short entries for?** If they preserve general
   instruction-following they stay but must be a much smaller fraction. If
   vestigial, they are diluting the only behavior being trained for.
4. **Register check `[unverified]`.** Is the long-form prose actually 1840s
   Småland third person? Not directly inspected. L057/L058 openings
   (`Chapter 32 - Thursday, April 7`, PROLOGUE/EPILOGUE markers) look like a
   different source. Read the prose of 6–8 long-form entries.

If the sweep confirms the live hypothesis, the fix is rebalancing by **entry
count**, not token count: downsample short-form hard, upsample long-form to
match, or mask loss on the short entries entirely.

---

## ENGINE-SIDE (this repo — separate from the LoRA)

Shipped on this branch (`9a9a362`): `finish_reason` / `word_count` /
`output_tokens` / `template_leak` captured on both backends and logged;
telemetry `ALTER TABLE` migration (`CREATE TABLE IF NOT EXISTS` silently no-ops,
so columns added later never appeared and every `log_event` raised);
chat-template leak detection with a warning instead of a silent strip;
`MAX_TOKENS` in config; explicit chapter-length line in the anchor;
scene injection reworded to chapters.

**Untested risk:** if length is bound to the literal string
`"Write the next chapter."`, it will not transfer. `run_drift_pipeline` passes
arbitrary `user_input` straight through, so `generate("Karl walks to the field")`
may lose the behavior entirely with an otherwise working LoRA. Test by running
the eval a second time with a differently-phrased request — the gap between the
two numbers predicts engine behavior better than the eval score does.

**Known, not fixed:** `MemoryWindow` replays prior outputs as few-shot length
exemplars (short-output ratchet); `rolling_baseline.py:65` crashes on a NULL
`action_pct`; the `prevent_truncation` param set is selected by high interiority
rather than by short output; `total_scenes=40` against a ten-chapter arc means
one `generate()` per chapter never leaves the silent band and the escalation
curve never fires.
