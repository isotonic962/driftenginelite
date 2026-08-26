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

Last updated: 2026-08-26 09:40. **Variant F eval complete: 10/10. Verdict below.**
NEXT JOB: the chapter-boundary slicer. See VERDICT.
Read DESIGN INTENT, THE REAL PROBLEM, and CORPUS VERSIONS first.

---

## DESIGN INTENT — read before proposing corpus changes

The LoRA is **deliberately not trained on Moberg**. It is trained on Swedish
literature influenced by him or on him; setting comes from the anchor
(Småland 1840s, Karl/Kristina/Per/Anders) and texture is steered by the
DriftScorer corridors. The goal is a story *like* his, not his story.

Two proposals have already been made and withdrawn against this: "the corpus is
the wrong books" and "rebuild from 88 whole Moberg chapters". Both would train
the model to reproduce Moberg's own text, which is the thing the design avoids.
Do not re-propose them.

## READING THE ACTUAL TEXTS — variant F samples (2026-08-26, corrects the verdict below)

The ten generated texts were read directly (gen_v6_cap2560.json, committed on
this branch). Three findings, one of which corrects an earlier claim.

### 1. The frac=0.5 interpretation was WRONG — there is no "chapter then repeat"

Earlier notes said the model "composes a chapter, reaches the end, and writes it
again." The texts show otherwise. The capped samples are **short-period
sentence loops**, not whole-text doublings: the model writes 150-450 words of
real prose, then falls into a single sentence or small block repeated to the cap
("He thought about how he had felt when he had been with her." ~150 times;
"I have never had a moment of clarity since I was twenty." ~180 times).
frac lands on 0.5 because any long periodic tail makes the longest repeated
substring ≈ half the text — it was never evidence of duplication of a whole
chapter. Loop onset, measured per capped sample:

| # | words | composed before loop | loop period |
|---|---|---|---|
| 2 | 2354 | **447** | 13-word sentence |
| 3 | 1861 | **200** | 46-word dialogue block |
| 6 | 2151 | (quasi-loop with variation — crisps/beer/biscuits ritual, no strict period) |
| 7 | 2362 | **146** | 12-word sentence |
| 10 | 2263 | **233** | 47-word block |

### 2. True composition length is ~150-450 words on ALL ten samples

EOS samples: 200, 252, 263, 374, 405 words. Loopers before onset: 146-447.
**No sample sustains composition anywhere near the training minimum of 893.**
So the model does not "compose long and fail to stop" — it composes SHORT and
then either stops (base EOS) or loops. Two distinct defects:
- **no sustained composition** past ~450 words, despite 138 training examples
  of 893-1500-word sustained prose;
- **no stop at closure** — the alternative to stopping is a loop, not more chapter.

The loops are the corpus's own style turned pathological: the clean samples use
controlled anaphora ("He thought of the lake. He thought of the girl…" — and
end), the loopers enter the same device and never escape. Incremental
repetition is a real feature of the v2_1 sources; a weak adapter reproduces the
device without the exit.

### 3. The style transfer WORKED

All ten are spare, physical, restrained, domestic — recognizably the v2_1
register (Petterson/Haruf line). No headers this time (variant E printed
"Chapter 27"; every F sample starts in-scene). Sample 5 is a complete miniature
with a genuine ending: "In the morning she packed her suitcase and left. He
didn't go with her." The corpus swap did exactly what it was supposed to do to
the voice. Register is no longer an open problem.

### Implications for next moves

- The **slicer remains necessary** (EOS placement) but is now known to be
  **insufficient alone**: it does not explain composition dying at ~300 words.
- **Sustainment** points back at adapter capacity: attention-only q/k/v/o,
  r=alpha=16, 2 epochs. The scale sweep ruled out inference-time scaling, not
  training strength. Next training cycle should pair the sliced corpus with
  MLP targets (gate/up/down) and/or 3 epochs — one cycle, both fixes, then the
  same 10-sample eval.
- **Cheap inference mitigation available now:** the loops are 12-47-word
  periods that repetition_penalty=1.05 cannot break. A DRY sampler /
  no-repeat-ngram constraint at generation time would sever exactly this
  failure mode and costs no training. Worth adding to the eval harness as a
  separate arm (same samples with and without) to see what the model does when
  the loop exit is forced.

## VERDICT — variant F (2026-08-26 09:40) — length/stop numbers still valid; interpretation superseded above

**The corpus-selection fix was not sufficient. The word-count slicing is the
defect. The chapter-boundary slicer is the confirmed next job.**

Full 10-sample eval, cap 2560, scale 1.0, fresh RNG per sample:

| | variant E (control, cap 2048) | variant F (v2_1, cap 2560) |
|---|---|---|
| stopped on EOS | 5/10 | 5/10 |
| ran to cap, frac saturated at 0.5 | 5/10 | 5/10 |
| word count in training range | 0/10 | **0/10** |

frac: [0.032, 0.5, 0.5, 0.04, 0.025, 0.5, 0.5, 0.034, 0.028, 0.5]
word_count: min 200 / med 1133 / max 2362 · novel_words: min 195 / med 660 / max 1181
(train ref: n=138, min 893, med 1422, max 1500)

The bimodal 5/5 pattern is **identical** across both variants. Two matched
10-sample arms, identical eval settings, one variable changed (corpus v1 ->
v2_1: cleaner selection, better endings, no Swedish, no front matter) — and the
stopping behaviour did not move at all. What v1 and v2_1 share is word-count
slicing. That isolates the slicing as the cause about as cleanly as an
experiment of this size can.

"novel in range: 5/10" in the summary is the saturation artifact (novel = cap/2
on every capped sample — 1181 ≈ 2362/2); wc in range 0/10 is the honest number.
`first_repeat_start` was not computed in-run; it can still be extracted from the
saved sample texts in `/workspace/gen_v6_cap2560.json` with no GPU.

### What the slicer must do (spec)

Cut the same v2_1 sources at their **chapter boundaries** instead of at word
counts, so every long-form entry ends where its author ended it:
- use the structural markers in the raw sources (chapter headings, part breaks
  — the ones stripped from v2 during cleanup) to locate boundaries;
- strip the heading itself from the assistant text after cutting;
- keep the exclusive `"Write the next chapter."` key, the single system prompt,
  and the short-form set unchanged;
- verify: seam-continuity test near 0%, endings at narrative closes, tokenized
  max under `max_seq_length`, then retrain with hyperparameters identical to E/F.

## RESUME HERE (superseded — eval completed; kept for the infrastructure notes)

Variant F (v2_1 corpus, hyperparameters identical to E) is **trained**. Eval is
**2 of 10 done**. Pick up by running samples 3-10 — #1 and #2 are in
`/workspace/gen_v6_cap2560.log` and must not be redone.

Three things to change before relaunching:

1. **Launch detached.** Two runs have now been lost to infrastructure: the
   container recycled at 06:26 (`5c1e2d7f5b1a` -> `e9b248f85411`) and the eval
   was SIGHUP'd at 09:08 when the browser connection dropped, because it ran in
   the session's background shell. Use
   `tmux new-session -d -s eval '<cmd> >> <log> 2>&1'`, or `setsid nohup ... &`.
2. **Append each sample as a JSON line on completion.** Both lost runs died
   before the final array write and had to be scraped back out of the log.
3. **Add `first_repeat_start`** and stop reading `novel`. See below.

### `frac = 0.5` is metric saturation, not a measurement

The longest span that can appear twice in a text of length T is T/2. When the
whole output is one block repeated, `maxrepeat` hits that ceiling and `frac`
reads exactly 0.500 **at any cap**. So `novel = words - maxrepeat` collapses to
cap/2 and carries no information about what the model composed.

This already produced one false positive: variant F #2 reported
`novel=1177, novel_in[893,1500]=True`, which looks like a chapter of trained
length. 1177 is half of 2560. Raising the cap to 4096 would have "produced" a
1900-word chapter by the same arithmetic.

Replace with **`first_repeat_start`** — the token index at which the output
first begins reproducing earlier text. That is the composition length, and it is
cap-independent. If it comes back at 1100-1200 on the loopers, the model is
writing chapter-length prose and only failing to terminate, which is the good
version of this outcome.

## RESULTS — variant F (v2_1 corpus, cap 2560)

Pre-flight PASS. 702 records tokenized as the trainer does (Qwen3-14B template,
`add_generation_prompt=False`): long-form n=138, min 1145 / median 1778 / p95
1967 / **max 2176** tokens against `max_seq_length=2560`. **Zero entries
truncated.** Longest is L006 (1470 w -> 2176 tok). All 16 hyperparameters diffed
identical to `train_drift_sft_v5.py`; only `DATASET_PATH` and `OUTPUT_DIR` differ.

| # | tok | stop | words | maxrepeat | frac | note |
|---|---|---|---|---|---|---|
| 1 | 416 | EOS | 374 | 12 | 0.032 | clean, below the 893 floor |
| 2 | 2560 | CAP | 2354 | 1177 | 0.500 | saturated — see above |

Both variant E modes survive into F. 8 samples outstanding.

### CONTROL ARM — variant E, cap 2048, 10 samples (recovered from log, zero GPU)

**5/10 CAP with frac exactly 0.5 · 5/10 EOS and short · 0/10 in the training
range.** Perfectly bimodal. Variant E never once produced a chapter of trained
length that terminated on its own. This is the matched baseline for variant F.

## CORPUS VERSIONS

| | v1 | v2 english | **v2_1 (training on this)** |
|---|---|---|---|
| records | 739 | 704 | 702 |
| long-form | 139 | 140 | 138 |
| long-form total words | 164,244 | 168,855 | **187,472** |
| median entry | 1225 | 1225 | **1422** |
| wholly-Swedish records | 39 | 0 | 0 |
| structural headings mid-text | 5 | 0 | 0 |
| drift median / max | 5.19 / 33.17 | 5.10 / 33.17 | **4.87 / 25.64** |
| pairs continuing across the seam | 72% | 74% | **67%** |

v2_1 sources: Haruf (18), Lagerlöf *Gösta Berling* + *Nils Holgersson* (19),
Bengtsson *Long Ships* (11), Petterson *Out Stealing Horses* (11), Boye
*Kallocain* (10), Larsson (10), Lapidus (10), Strindberg *Röda rummet* (9),
and roughly 16 Moberg entries (L053-L068). **The Moberg inclusion is
unconfirmed against the stated design intent below — ask before acting on it.**
Nesser is gone; he was the 33.17.

**Slicing is still word-count based in v2_1.** 37% of entries fall in the single
1450-1499 bin against a hard stop at 1500 — that is a raised word target, not a
chapter boundary. But endings read materially better by eye than in v2, and the
"ends mid-flow" regex used earlier over-flags (it counts any final clause
containing as/and/when), so discount that metric. The seam test is the reliable
one.

**DECISION (2026-08-26): train on v2_1 rather than build the chapter-boundary
slicer first.** The selection change is large enough to measure, and the run
discriminates: if `frac` drops off 0.5 the improved endings were sufficient and
the slicer is unnecessary; if `frac` stays pinned at 0.5 the slicer becomes the
next job with evidence behind it. Cheaper than building it on spec.

Hyperparameters held identical to variant E so the corpus is the only variable.
Pre-flight check: tokenize all 138 long-form entries and confirm none exceeds
`max_seq_length=2560` — the median rose to 1422 words, and anything truncated
loses its EOS, reintroducing the bug on exactly the longest entries.

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

### What the corpus actually contains — measured, not assumed

~39 source runs across the 139 long-form entries: Backman *Ove*, Söderberg
*Doctor Glas*, Lapidus *Easy Money*, Haruf *Plainsong*, Larsson *Millennium*,
Lagerlöf *Nils Holgersson* / *Gösta Berling*, Nesser *Van Veeteren*, Strindberg
*Röda rummet*, and one Moberg entry (L139).

Register scan: 62% contain modern-life vocabulary (car 84, phone 31, computer 16),
8% contain 1840s rural vocabulary. **This is expected under the design** — setting
is the anchor's job, not the corpus's. It is not a defect on its own.

### DriftScorer run over the training corpus (median drift per source run)

The project's own instrument, pointed at its own training data for the first time.

| run | n | drift | source |
|---|---|---|---|
| L139 | 1 | **1.37** | Moberg |
| L069–L079 | 11 | 2.67 | Lagerlöf, *Nils Holgersson* |
| L002–L010 | 9 | 3.67 | Backman, *Ove* |
| L011–L026 | 16 | 3.97 | Lapidus, *Easy Money* |
| L040–L061 | 22 | 5.82 | Larsson, *Millennium* |
| L085–L090 | 6 | **15.82** | Nesser, *Van Veeteren* |
| L091 | 1 | **24.87** | Nesser |

Corpus-wide long-form drift: median 5.19, min 0.07, max 33.17.

Two results worth keeping: **Moberg's own entry scores third-best of 39 runs**,
which validates the corridors; and the contemporary crime/comic material
(Lapidus, Backman) scores *well* while the psychological crime (Nesser) scores
worst. The corridors discriminate on technique, not period — judging the corpus
by author or publication date gives the wrong answer. An earlier eyeball critique
of the selection was wrong on exactly this point.

**Caveat:** `drift.py` still holds the 16-chapter corridors, not the 88-chapter
recalibration. Ranking is valid; absolute values are not. Re-run with current numbers.

### CONTAMINATION — 29 records to drop

- **22 untranslated Swedish records.** Long-form: L111, L119, L129, L137, L138.
  Short-form: 17 in the R259–R296 range. For an English-target model this is
  straight contamination, and it is why they score 20+ — the English
  `PHYSICAL_VERBS` lexicon cannot parse them, so every texture metric collapses
  toward zero and the drift score measures nothing.
- **7 front-matter records.** L057, L058, L069, L072, L103 carry structural
  headings mid-text (`PROLOGUE`, `PART 1`, `Book 6`); L023 and L064 are editorial
  or critical prose. L058 is *not* a maths textbook — it is the tail of one
  *Millennium* novel running into the prologue and mathematical epigraph of the
  next. A slicing artifact, not a foreign document.

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

**The model never learned where a chapter ends, because no training entry ends
where a chapter ends.**

CONFIRMED by direct analysis of the corpus — this is no longer a hypothesis:

- 85 of 138 consecutive long-form pairs share proper nouns across the seam. The
  entries are sequential ~1200-word **slices of continuous novels**, not chapters.
- 58 of 139 end on a continuation clause; 15 end inside dialogue.
- 96% end on terminal punctuation and the length distribution is smooth with no
  cap pile-up — so the *trimming* was competent. The defect is that a word-count
  cut point is not a narrative close.

EOS was therefore trained onto arbitrary boundaries. The model learned that a
chapter ends somewhere around 1200 words for no discernible reason, which
produces exactly the observed behaviour: EOS fires early at a random point, or
fails to fire where closure should be and the model loops to the cap.

Superseded sub-hypotheses (the mechanism was right, the cause was not the
collator):

1. **EOS at arbitrary positions.** 109 of 139 long-form entries were trimmed by
   word count and end mid-paragraph, so their EOS sits at a narratively
   meaningless place. *Best fit for the observed bimodality* — it predicts both
   the random early stop and the failure to stop at real closure. The other 30
   entries were trimmed to complete sentences.
2. ~~**EOS absent from the targets.**~~ **KILLED** by the v2_1 pre-flight:
   `<|im_end|>` sits as the penultimate token in 702/702 records, and in
   **739/739 for variant E's corpus too** (the Qwen template appends a trailing
   newline after it). EOS was always in the targets, on both corpora.
3. **EOS masked out of the labels** (`-100` at that position). Not verified, but
   the token is present, so this is the only remaining collator-side candidate.
4. ~~**EOS truncated away by `max_seq_length`.**~~ **KILLED** by the same
   pre-flight: max 2176 tokens against a 2560 limit, zero entries truncated.

Either way, samples that *do* stop are the base model's EOS firing rather than a
learned one — which is why every short sample is clean and every long one loops.

The EOS-config checks below are **no longer the priority** — the corpus explains
the behaviour without them. Run them only if the re-slice fails to fix it.

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

0. **THE FIX — one training cycle, not three.** A retrain is unavoidable because
   the 22 Swedish records must come out, so everything else rides along with it
   at near-zero marginal cost:
   1. drop the 29 contaminated records
   2. **re-slice the same sources at their chapter boundaries** instead of at word
      counts — same books, same selection principle, but every entry then ends
      where its author ended it
   3. optionally drop the worst runs by drift score (Nesser)
   4. train once, re-eval at cap 2048

   **No-retrain stopgap, available today:** the model already produces
   chapter-shaped prose (696 / 614 / 930 novel words) and then duplicates it.
   `frac` and `maxrepeat` locate the seam exactly — generate to 2048 and truncate
   where the duplication starts. That yields a clean ~700-word chapter from
   variant E as it stands and unblocks engine work. It gives clean *text*, not a
   well-formed *ending* — the same arbitrary boundary, moved to inference.

1. **Copy `/workspace/gen_v5_cap2048.log` off the box.** The container has already
   been recycled once mid-run and the JSON was lost with it. Do this first.
2. **Extract RUN 1 samples #1–#9 from the log.**
3. **Re-run the DriftScorer table above with the 88-chapter corridors.**
4. **How was the training loss normalized — per-sequence or per-token?**
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
