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

Last updated: 2026-08-26 06:30, after RUN 2 (partial). Read THE REAL PROBLEM first.

---

## THE REAL PROBLEM

**The model composes a chapter, reaches the end, and writes it again verbatim.**

Every long sample in RUN 2 has `frac = 0.5` exactly — the output is two copies of
the same content — and runs to the token cap rather than stopping. Word count was
never measuring length. Novel content per sample is ~290 / 696 / 614 / 930 / 844
words against a training min of 792 and median 1225: slightly short, but in the
right neighborhood.

This is a **missing-stop** problem, not a length problem. Everything below that
reads as a length finding should be re-read in that light.

Original framing, superseded: variant E answered `"Write the next chapter."` with
354 words where all 139 training instances of that string produced 792–1494 words
(median 1225).

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

### Long-form provenance — the 139 are not homogeneous

| range | n | how it was trimmed | endings |
|---|---|---|---|
| L001–L109 | 109 | by word count, no regard for scene boundaries | mostly mid-paragraph |
| the other 30 | 30 | trailing incomplete sentences removed | clean |

**This is the most likely mechanical cause of the missing stop** and it was not
known when the earlier hypotheses were formed. An entry that ends mid-sentence
places its EOS at a narratively arbitrary position; 109 of 139 examples teaching
that would produce exactly the observed bimodality — EOS fires at a random early
point, or fails to fire where closure actually is.

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

### Cap 2048 — RUN 2, adapter-scale sweep (4 samples per arm)

| scale | words | stop | maxrepeat | frac |
|---|---|---|---|---|
| 1.0 #1 | 290 | EOS | 4 | 0.014 |
| 1.0 #2 | 1391 | CAP | 695 | **0.5** |
| 1.0 #3 | 1228 | CAP | 614 | **0.5** |
| 1.0 #4 | 1859 | CAP | 929 | **0.5** |
| 1.5 #1 | 81 | EOS | 6 | 0.074 |
| 1.5 #2 | 123 | EOS | 6 | 0.049 |
| 1.5 #3 | 89 | EOS | 5 | 0.056 |
| 1.5 #4 | 12 | EOS | 0 | 0.0 |
| 2.0 #1 | 1688 | CAP | 844 | **0.5** |

`frac = 0.5` on five independent samples. Every sample that goes long does so by
duplicating itself and then hitting the cap; every sample that stops on EOS is
clean and short. The two behaviours are the same defect seen from both sides.

**Interrupted after 2.0 #1** — the pod recycled (`env@5c1e2d7f5b1a` →
`env@e9b248f85411`) and `/workspace/gen_v5_cap2048.json` was never written.
The log is the only copy: `/workspace/gen_v5_cap2048.log`. **Copy it off the box.**

### Cap 2048 — RUN 1 (10 samples @ scale 1.0)

Only #10 recovered from the visible log: 447 tok, EOS, 414 words.
**Samples #1–#9 are in the log and have not been extracted.** This is the real
scale-1.0 distribution and the most valuable unread data on the box.

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
8. **Scale as a brevity dial (the successor to 4).** Killed by RUN 2: the
   relationship is non-monotonic — 1.0 goes long, 1.5 gives 81/123/89/12 words,
   2.0 goes long again. The falsifier's 354 → 126 → 109 was three unlucky draws
   at n=1, as flagged at the time. There is no dial. Stop chasing scale.
5. **Degenerate or stuck decode.** A == B but C ≠ D ≠ A. Sampling varies.
6. **Training-side truncation.** 1494 w ≈ 2000 tok, fits inside `max_seq_length=2560`.
7. **Inference cap.** Was a real confound at 1200 (it censored C). Now 2048,
   which is ~1740 words at the observed rate and clears the 1494 training max.

---

## LIVE HYPOTHESIS

**The model never learned where a chapter ends.** Four candidate mechanisms,
all answered by one check (see below):

1. **EOS at arbitrary positions.** 109 of 139 long-form entries were trimmed by
   word count and end mid-paragraph, so their EOS sits at a narratively
   meaningless place. *Best fit for the observed bimodality* — it predicts both
   the random early stop and the failure to stop at real closure. The other 30
   entries were trimmed to complete sentences.
2. **EOS absent from the targets.** Collator never appended `eos_token_id`.
   Predicts mostly never-stops, so it fits the loop but not the clean early stops.
3. **EOS masked out of the labels** (`-100` at that position). Same effect as 2.
4. **EOS truncated away by `max_seq_length=2560`.** 1494 w ≈ 2000 tok on average
   and fits — but that is an average, and any entry crossing 2560 loses its EOS
   to truncation. Would silently affect only the longest entries.

Either way, samples that *do* stop are the base model's EOS firing rather than a
learned one — which is why every short sample is clean and every long one loops.

### THE CHECK — one pass, answers all four

For each of the 139 long-form records, tokenized **exactly as the trainer does**
(same chat template, same `max_seq_length`, same collator), report:

1. total token count, and whether it hit 2560
2. whether the final content token is `<|im_end|>` / `eos_token_id`
3. whether the label at that position is unmasked (not `-100`)
4. the last ~80 characters of the decoded assistant text
5. whether it ends on terminal punctuation

Then cross-tabulate by ID range: **L001–L109 vs the other 30.**

Static, no GPU. Note that "do the loops concentrate in L001–L109" is *not*
runnable as stated — loops occur at inference and carry no training-entry
provenance. Inspecting how the entries end is the observable form of the same
question.

Still possible as a *contributing* factor, not the primary: 600 of 739 EOS tokens
in the corpus sit at the end of a short response. Every example contributes one
EOS regardless of length, so the stop prior is governed by **entry count** (81%
short) while register is governed by **token share** (83.5% long).

---

## OPEN — in priority order

0. **Copy `/workspace/gen_v5_cap2048.log` off the box.** The container has already
   been recycled once mid-run and the JSON was lost with it. Do this first.
1. **Run THE CHECK** (see LIVE HYPOTHESIS) over all 139 long-form records.
   One pass, four hypotheses, no GPU.
2. **Extract RUN 1 samples #1–#9 from the log.**
3. **How was the training loss normalized — per-sequence or per-token?**
   If per-sequence, the 600 short examples outweigh the 139 long ones 4.3:1 in
   gradient, and the 83.5% token share never reached the optimizer. That would
   explain the whole result and it is a flag, not a corpus rebuild.
   *Static read of the training script. No GPU. Do this first.*
4. **What are the 600 short entries for?** If they preserve general
   instruction-following they stay but must be a much smaller fraction. If
   vestigial, they are diluting the only behavior being trained for.
5. **Register check `[unverified]`.** Is the long-form prose actually 1840s
   Småland third person? Not directly inspected. L057/L058 openings
   (`Chapter 32 - Thursday, April 7`, PROLOGUE/EPILOGUE markers) look like a
   different source. Read the prose of 6–8 long-form entries.

Do **not** spend more GPU time on scale sweeps — that hypothesis is dead (KILLED 8).
The remaining open items are all static reads.

**None of the four mechanisms requires rebuilding the corpus.** EOS absent or
masked → collator config. EOS at mid-sentence → re-trim the 109 to the last
complete paragraph, a text operation on material already in hand. Truncated at
2560 → raise the limit. The material is fine; how it was terminated is not.

If the EOS check comes back clean and the defect is instead the stop prior, the
fix is rebalancing by **entry count**, not token count: downsample short-form
hard, upsample long-form to match, or mask loss on the short entries entirely.

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
