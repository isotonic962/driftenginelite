# Autonomous run log — Aug 26 2026

Lab notebook from seven headless Claude Code sessions on the RunPod pod,
running against `AUTONOMOUS_TASK.md` under `--max-turns 100`. Copied here from
`/workspace/driftenginelite/Drift-engine/EXPERIMENT_LOG.md`.

**Snapshot caveat.** This copy was taken before commit `6781c1b` ("Fix a
boundary rule that made the anaphora instrument blind to the ladder"). The
anaphora figures in the later entries come from the pre-fix instrument, which
required both boundary and candidate to be the space-prefixed token form and so
silently dropped every paragraph-initial sentence — 22.7% of the chapter corpus,
and 194 of 197 sentences in `gen10`, the single clearest specimen of the
phenomenon. The EOS-hazard results are unaffected. Re-run pending.

**Headline findings across the seven runs:**

1. EOS is not the defect. The adapter *installed* a stop decision the base model
   lacks — terminal P(EOS) on brief targets 0.481 vs base 0.000 — and on chapter
   text at the chapter prompt it sustains to E[stop] 1264 words of a true 1420,
   S(250) = 0.998.
2. The short-stop schedule is prompt-keyed, not global. Same 138 chapter texts,
   only the user turn swapped: premature mass 0.256 → 0.995, E[stop] 1264 → 215,
   paired 138/138, median delta −918 words. Read as difference-in-differences —
   base loses 150 words on the same swap, the adapter loses 1049.
3. The anaphoric ladder is the single cause of both failure modes. Not two bugs:
   one degeneration with two exits. Exit A gives a correct EOS on genuinely
   finished but degenerate text; exit B gives an EOS-suppressing attractor.

Consequence: "make the model emit EOS at the end of a long response" is the
wrong target. Fixing EOS would only convert exit-B loops into exit-A stops at
~300 words.

---

Contemporary domestic-relational, ten for ten.

The eval's system prompt is the corpus's own: *"Write in the mode of objective
physical realism. Describe actions, environments, and labor with precision."*
Base obeys it 4/4. Scale 0.25 obeys it 3/3. **Scale 1.00 obeys it 0/10.**

### Two mechanical discriminators, and what survived being attacked

"I read the openings" is exactly the kind of impression that put 1b on the list,
so: three discriminators declared before running — **D1** first-person pronoun rate
(a closed function-word class, no lexicon choice at all), **D2** fraction of
characters inside quotation marks, **D3** agricultural/craft verb rate using the
AGRI subset of the project's own `engine/lexicon.py` PHYSICAL_VERBS, printed in
full in the script for audit.

**D2 is a null** (p=0.31–0.43) — dialogue does not separate the arms.
**D1 is a null on the pre-registered test** (p=0.30) and stays null: the scale-1.0
arm is bimodal, 5 of 10 samples are heavily first-person (57–192 per 1k) and 5 are
at zero, which no median test can see, and the post-hoc presence test (>10/1k) is
1/7 vs 5/10, p=0.304. First-person narration is not the discriminator.

**D3 survives everything thrown at it.** Presence: **7/7** of base+0.25 samples
contain at least one agricultural/craft verb, against **1/10** at scale 1.0
(Fisher exact p=0.00041). That test is length-biased in base's favour — base
samples are longer — so the length-invariant form, pooling every word in an arm
and using the exact conditional binomial rate test:

```
0.00 base   18 agri verbs / 3008 words = 5.98 per 1k
0.25 LoRA    6 agri verbs / 1744 words = 3.44 per 1k
0.50 LoRA    0 agri verbs /  762 words = 0.00 per 1k
1.00 LoRA    4 agri verbs / 3266 words = 1.22 per 1k

POOLED(base+0.25) 5.05/1k  vs  1.00 LoRA 1.22/1k
K1 ~ Binom(K=28, p=0.593): expected 16.6, observed 24;  exact p = 0.005 (two-sided)
```

The adapter at full strength cuts labour vocabulary roughly four-fold.

### The finding that matters: the adapter *overshoots its own training data*

T_corpus long-form interiority_pct: median **7.0**, p90 **13.9**, max **21.4**.
That sits inside the Moberg corridor (7.0–12.0). **The training corpus is not
interiority-heavy.** So scale-1.0 output at median 19.4 is not something the
adapter inherited.

| arm | n | median int% | above corpus p90 (13.9) | above corpus MAX (21.4) |
|---|---|---|---|---|
| 0.00 base | 4 | 1.7 | 0/4 | 0/4 |
| 0.25 LoRA | 3 | 0.0 | 0/3 | 0/3 |
| 0.50 LoRA | 3 | 20.0 | 2/3 | 1/3 |
| 1.00 LoRA | 10 | 19.4 | **6/10** | **4/10** |

Pooled base+0.25 0/7 vs scale-1.0 6/10, Fisher exact **p=0.0345**.

Length was the obvious confound — interiority_pct is a percentage over sentences
and the de-looped scale-1.0 samples are shorter (146–720 w) than base's (652–819
w). Recomputed on a **length-matched 60-word prefix** of every sample: pooled 0/7
vs 5/10, Fisher **p=0.0441**. It is not a length artifact.

Four of ten scale-1.0 samples exceed the interiority of **all 138 long-form
training entries**. And the high-interiority corpus entries are scattered
(L016, L026, L035, L059, L060, L064, L086, L127 — across the Lapidus, Larsson and
Nesser ranges), 14 of 138 total, so there is no interiority-heavy source run to
drop. **This is amplification, not imitation** — the same signature as the
anaphoric attractor, which is also a frame amplified far past anything in the data.

### One hypothesis tested and killed

Given all of the above it was tempting to conclude the loop *is* the interiority
frame — that termination failure and register failure are one variable. **They are
not.** Classifying each sample's longest repeated substring with the project's own
lexicons gives 3 interiority frames, 2 physical, 1 neither among the six runaway
loops — mixed, not uniform. And within the scale-1.0 arm, interiority_pct does not
predict CAP at all: EOS n=5 mean 23.2 vs CAP n=5 mean 24.2, exact permutation
**p=0.472**. The across-all-20 version of that test (p=0.034) is pure arm
confounding and should not be quoted. Register and termination are two separate
casualties of the same adapter pressure, not one mechanism.

### Prose — the same prompt, the same seed family, three adapter strengths

**base b#4 (652 w, EOS):**

> The morning sun had barely breached the horizon when the first sounds of
> industry stirred the town from its slumber. A distant engine roared to life,
> followed by the rhythmic clatter of metal against metal, the hiss of steam, and
> the muffled shouts of men coordinating their tasks. The air was thick with the
> scent of coal smoke and damp earth…

**s=0.25 #3 (584 w, EOS):**

> The sun had not yet risen when the first of the workers arrived. They came in
> silence, their boots crunching on the frost-laced grass, their breath curling in
> the air like smoke from a chimney. Some carried bundles wrapped in oilskin,
> others carried tools, and still others carried nothing at all, save for the
> weight of their own exhaustion.

**s=1.00 v#4 (405 w, EOS):**

> 'You're not going to come back,' she said. 'I don't know that yet.' 'I want you
> to come back.' 'You have a right to ask me that,' he said, 'but I can't promise
> you anything. I'm not sure I can get over this.' He looked at her and thought,
> This is the first time I've been able to say that to anyone.

And the final sentences of **s=1.00 v#9 (252 w, EOS)** — one of the five samples
that terminates cleanly, so this is the scale-1.0 arm at its *best*:

> He imagined the man falling down dead. He imagined the blood flowing out of him.
> He imagined the pain and the fear in the man's eyes. He imagined the silence
> after the shot. Then he opened his eyes again. He looked at the man standing in
> front of him. He looked into his eyes. He saw the fear there. He saw the pain.
> He saw the suffering. He saw the death that was coming. And then he pulled the
> trigger.

An anaphoric ladder built out of interiority verbs, ending a chapter. The clean
EOS and the attractor are in the same six sentences.

### Verdict

**No intervention was made, so there is no CLEAN/IMPROVED/WORSE verdict to give.**
As a finding: **RETRACTION plus a stronger replacement.**

Retracted: *"scale 0.25 reads closer to the target register than base."* Base and
scale 0.25 are indistinguishable on every scored register metric (p=0.23–0.83),
and where they do differ measurably — labour vocabulary — base is the *higher* of
the two (5.98 vs 3.44 per 1k). Third run's `[unverified]` flag was correct to
be there.

Replacing it, and this is the load-bearing result: **at full strength the adapter
changes register measurably and in the wrong direction on the design's own axis.**
Interiority up from 1.7 to 19.4, past the T_corpus p90 in 6/10 samples and past
the T_corpus maximum in 4/10, surviving a length-matched control (p=0.044).
Labour vocabulary down four-fold (p=0.005). Prompt adherence on "objective
physical realism… labor with precision" from 4/4 to 0/10. Meanwhile M3 says the
output is no closer to the training corpus at any strength.

So the LoRA is not buying its corpus's register. It is amplifying one axis of it
past anything the corpus contains — which is structurally the same failure as the
anaphoric attractor, and lands the output further from T_moberg on the metric the
project itself weights most heavily (`interiority_pct`, WEIGHTS 1.0, second only
to figurative_density).

### What this does to the retrain decision — read before spending the run

OPEN 1b was raised to check whether a gentler adapter might terminate cleanly and
buy nothing. **That worry is now realised rather than averted**, and the dose-
response is a trade with no measured good operating point:

- Dial the adapter **down** → termination returns (3/3 EOS at 0.25) and register
  becomes statistically indistinguishable from no adapter at all.
- Dial it **up** → register changes measurably, in the wrong direction, and
  termination collapses (5/10 EOS at 1.0).

The honest caveat, stated because it cuts against the conclusion: **inference-time
scaling is not the same operation as retraining at a lower learning rate.**
Multiplying a trained delta by 0.25 shrinks every direction uniformly; training
more gently can find a *different* solution, not a scaled one. So this does not
formally rule OPEN 2 out. It does mean OPEN 2 should no longer be scored on
termination alone, and it raises the prior that a gentler variant F lands
somewhere on the same trade curve.

### Next step

1. **Whatever is run next must be scored on register as well as termination.**
   That is now possible and was not before this session: `register_check.py` plus
   `register_robust.py` give interiority-vs-corpus-p90, the agri-verb rate test,
   and the keyness tables, all static and free. A CLEAN verdict on termination
   alone is no longer sufficient evidence that a run bought anything.
2. **The strongest remaining question is free and static, and it is upstream of
   the retrain:** the corpus sits at interiority median 7.0, inside the Moberg
   corridor, and the adapter reaches 19.4. What in the training setup amplifies
   one texture axis roughly threefold past its own data? That is answerable
   against `train_drift_sft_v6.py` and the corpus with no GPU — loss masking over
   the prompt vs the response, the 81%-short entry-count imbalance interacting
   with token share, or `packing`. **Do this before OPEN 2.**
3. OPEN 2 (retrain gentler) unchanged and still needs sign-off. If it is
   authorised: one change, new OUTPUT_DIR, **save intermediate checkpoints**, and
   score both axes.

Budget this session: 0/2 training runs, 0/6 generations, 0 GPU-seconds.

---

## 2026-08-26 (fifth run) — OPEN 1c: the training setup does not contain the amplifier, and the amplification is much larger than interiority

**Adapter under test:** `/workspace/drift_sft_out_v6/adapter` (variant F), the
newest, unchanged. **No training run was made (0 of 2). No generations were made
(0 of 6). 0 GPU-seconds.** Everything below is static: the corpus JSON, the saved
`training_args.bin`, the installed library source, and text already on disk.

This is INVESTIGATION.md OPEN item 1c, which the fourth run named as the lead
question and put ahead of the retrain: *"the corpus sits at interiority median
7.0 and the adapter reaches 19.4 — what in the training setup amplifies one
texture axis roughly threefold past its own data?"* Candidates named there: loss
masking over prompt vs response, the entry-count/token-share imbalance, `packing`,
LR schedule.

Scripts added: `scripts/corpus_texture_full.py`, `scripts/amplification_test.py`,
`scripts/check_padding_free.py`. Outputs copied to `logs/`.

### Step 1 — termination in the training data

Unchanged and still closed. EOS present, untruncated (max templated entry 2176
tokens against `max_length` 2560), unmasked. Re-confirmed incidentally by
`check_padding_free.py`: in a real collated micro-batch, 2189 of 2191 label
positions are not `-100`, and the label at the `<|im_end|>` position is `151645`,
the EOS id itself. The standing task's step-1 branch "EOS present and
untruncated → the data is fine, say so and stop" remains the branch this
investigation is on. No corpus edit was made and none is proposed.

### The config the training script never set — and what it does and does not do

`train_drift_sft_v6.py` sets neither `packing` nor `padding_free`. Loading the
run's own saved `drift_sft_out_v6/checkpoint-88/training_args.bin`:

```
padding_free                     = True        <- NOT set by the script
packing                          = False
max_length                       = 2560
completion_only_loss             = None        -> resolves False for a
                                                  dataset_text_field dataset
assistant_only_loss              = False
average_tokens_across_devices    = True
per_device_train_batch_size      = 2 ; gradient_accumulation_steps = 8
```

unsloth 2026.8.19 auto-enables padding-free: `_should_auto_padding_free`
(`unsloth/trainer.py:118`) checks only an env var and `packing`, **not** the
attention implementation. Every micro-batch of 2 examples was therefore flattened
into one sequence with **no `attention_mask`** — TRL omits it deliberately so
flash-attention will derive `cu_seq_lens` from `position_ids` instead.

That looked like a real find, because the training banner reads
`FA [Xformers = 0.0.35. FA2 = False]` and unsloth falls back to **sdpa**, which is
not in TRL's `FLASH_ATTENTION_VARIANTS` allowlist. If sdpa ignored `position_ids`,
the second example in every micro-batch would have attended across an
`<|im_end|>` into the first example's text — the model trained 1404 times on
"prose continues past the end token", which is precisely the observed failure.

**It is not what happens.** Built the real collator on a real (short, long) pair
and ran the real mask builder:

```
example lengths          : 139 + 2052 = 2191 tokens
input_ids shape          : (1, 2191)      <- one flat row, not 2
attention_mask present   : False
position_ids at the seam : [136, 137, 138, 0, 1, 2]
label at seam start      : -100 (MASKED)
find_packed_sequence_indices -> segment ids 0..1, seam-1=0, seam=1
token 144 (example 1) may attend to token 134 (example 0): False
token 144 (example 1) may attend to token 140 (example 1): True
```

transformers 5.5.0 handles this on the sdpa path: when `attention_mask is None`
and `position_ids` is present, `_preprocess_mask_arguments` calls
`find_packed_sequence_indices` and `and_masks` a packed-sequence mask into the
causal mask (`masking_utils.py:978`). Cross-example attention is blocked.
Padding-free here is **mathematically equivalent to padded batching**. Recorded
because it is exactly the kind of thing a later session would re-derive from the
banner and the allowlist and get wrong.

### Every mechanism 1c named, measured

| candidate | measured | amplifier? |
|---|---|---|
| `packing` on | `packing=False` in the saved args | no |
| padding-free contamination | isolated, verified on the real mask builder | no |
| EOS absent / truncated / masked | present, 2176 < 2560, unmasked | no |
| loss masks the response only | loss covers **everything**; prompt region is 13.7% of all gradient tokens, and 8.1% is the *same* 37-token system prompt repeated 702x per epoch | no |
| entry-count vs token-share imbalance | loss is token-weighted (`num_items_in_batch`, `average_tokens_across_devices=True`), so entry count never enters the gradient weighting | no |
| train/inference template mismatch | the generation prefix is a **byte-identical** prefix of the training text, empty `<think>` block included | no |
| the corpus is interiority-heavy somewhere | pooled interiority sentence rate: long-form 6.7%, brief 7.1%, all 702 **6.8%** | no |

**Correction to a number on record.** INVESTIGATION.md states long-form is 83.5%
of tokens. Tokenized as the trainer does, it is **75.5%** for corpus v2_1
(240784 / 319111) and **72.5%** for corpus_1. Entry share is 19.7% / 18.8%. The
direction of the imbalance is right; the magnitude on record is not.

### The 80% of the corpus that had never been register-measured

`register_check.py` defines T_corpus as the 138 records whose user turn is
`"Write the next chapter."`. The other **564 records — 80.3% of the corpus —**
had never been scored on any register metric. If they were interiority-heavy, the
adapter would be imitating its data rather than amplifying it.

```
subset                                n  median    mean    p90    max   pooled-sentence rate
LONG  (= register_check T_corpus)   138     7.0     7.6   13.9   21.4                  6.7%
BRIEF (never measured until now)    564     0.0     8.2   33.3  100.0                  7.1%
ALL 702 entries                     702     0.0     8.1   25.0  100.0                  6.8%
```

They are not. The corpus runs at ~6.8% interiority sentences however you slice it;
BRIEF's fat tail is small-sample noise (a 3-sentence entry scores 0 or 33 or 100).
The adapter's 19.4 is not hiding in the unmeasured majority.

### The length-matched redo — the claim survives, one sub-claim does not

The fourth run scored whole generations (146–720 w de-looped) against whole
corpus entries (~1410 w). `interiority_pct` is a percentage over classified
sentences, so its variance depends on the sentence budget: ~90 sentences for a
corpus entry, ~25 for a 400-word generation. Corpus prefixes confirm the effect is
large — long-form entries have whole-entry p90 **13.9** but first-60-word p90
**25.0**. Comparing a fat-tailed small-n draw against a thin-tailed large-n tail
manufactures exceedances out of nothing.

Redone properly: each generation of W words scored against the same 138 entries
truncated to *their* first W words, reported as a percentile rank.

```
arm           n  median pctile   above the length-matched p90
0.00 base     4           16.5   0/4
0.25 LoRA     3            7.2   0/3
0.50 LoRA     3           89.5   1/3
1.00 LoRA    10           87.5   5/10
```

**The amplification survives.** Scale 1.0 sits at the 87.5th percentile of its own
corpus at its own length, against base at 16.5 and scale 0.25 at 7.2. What does
**not** survive is the sharper phrasing: "above the corpus MAXIMUM in 4/10" was a
length artifact and should not be quoted again — 6/10 above p90 becomes 5/10 once
the de-looped length is the one matched (the first cut of this test matched raw
cap length against de-looped text; the bug is fixed in the script).

### The finding that matters — anaphora, and it is an order of magnitude larger

Interiority was never the best axis. The runaway samples are **anaphoric
ladders**, and that had never been measured against the corpus at all. Metric:
share of adjacent sentence pairs sharing their first two words, plus the longest
run of consecutive same-opening sentences. De-looped, so a verbatim repeat cannot
masquerade as a style result — a CAP sample scores high only if its *non-repeated*
prefix is already laddering.

```
set                     n  median%  mean%    p90    max   med run  max run
CORPUS long-form      138      1.1    1.6    3.3    8.9       2.0        3
CORPUS all 702        702      0.0    2.8    3.3  100.0       1.0        5
GEN 0.00 base           4      0.0    0.0    0.0    0.0       1.0        1
GEN 0.25 LoRA           3      4.4    4.9    4.4   10.3       2.0        3
GEN 0.50 LoRA           3     33.3   30.8   33.3   59.1       2.0       14
GEN 1.00 LoRA          10     24.5   23.3   39.1   58.8       4.5       21
```

7 of 10 scale-1.0 samples exceed the **maximum** anaphora rate of all 138
long-form training entries; 6 of 10 exceed the longest same-opening run found
anywhere in the corpus. Pooled base+0.25 1/7 vs scale-1.0 7/10, Fisher exact
**p = 0.0498**. The base model is at **0.0% in 4/4** — this behaviour does not
exist before the adapter and barely exists in the data.

Where interiority is amplified ~3x past a corpus median of 7.0, anaphora is
amplified ~20x past a corpus median of 1.1, from a base model that shows none of
it. And it is dose-dependent on the same curve as termination: 0.0 → 4.9 → 30.8 →
23.3 mean rate as the adapter is dialled up.

### Prose — the same failure at three adapter strengths

**base #1 (788 w, EOS)** — longest same-opening run in the whole arm: **1**.

**s=0.25 #2 (318 w, EOS), longest run 3** — at the corpus maximum, not past it:

> She was going to leave the forest behind today. She was going to find a place
> where the trees were not so tall and the sun was not so far away. She was going
> to find a place where she could start over.

**s=1.00 #1 (374 w, EOS), longest run 21** — past anything in the corpus by a
factor of seven:

> I thought about how she was going to cook for me and how she was going to serve
> it. I thought about what she would say. I thought about what I would say. I
> thought about everything we were going to do together. I thought about the
> things we were going to talk about. I thought about the things I was going to
> ask her.

Note this sample **terminates cleanly**. The ladder is not a symptom of failing to
stop, which is the next result.

### One more hypothesis tested and killed

It is tempting to say the ladder *is* the termination failure. Within the
scale-1.0 arm it does not predict it:

```
EOS n=5  anaphora rate% [58.8, 34.2, 0.0, 22.7, 31.0]  mean 29.4
CAP n=5  anaphora rate% [26.3,  0.0, 0.8, 20.0, 39.1]  mean 17.3
|diff| 12.1 pts, exact permutation p = 0.381
longest run  EOS [21, 6, 1, 4, 5]  CAP [7, 1, 2, 2, 10]  p = 0.595
```

If anything the terminating samples ladder *harder*. This is the same shape as the
fourth run's interiority-vs-CAP null (p=0.472) and it means the same thing:
**style amplification and termination failure are two casualties of one cause,
not one mechanism.** Fixing the ladder is not guaranteed to fix the stopping.

### Verdict

**No intervention was made, so there is no CLEAN / IMPROVED / WORSE verdict.**
As a finding: **ESTABLISHED, and it closes 1c in the negative.**

The training setup does not contain the amplifier. Every mechanism 1c named is
now measured and dead, including one — auto-enabled padding-free — that nobody
knew was on and that would have been a complete explanation had the mask builder
not handled it. The corpus does not contain it either: not in the 138 that were
measured, not in the 564 that were not, on either axis. Meanwhile the
amplification itself is real, survives length matching, and is far larger than the
axis it was found on: ~20x on anaphora against ~3x on interiority, from a base
model that shows 0.0% anaphora in 4/4 samples.

That leaves exactly one explanation standing for a base capability being
destroyed and a base-absent frame being manufactured: **how hard the adapter
presses**. 1c was raised to find a cheaper cause than OPEN 2 and it found none.

### Next step

1. **OPEN 2 (retrain gentler) is now the only live intervention**, and it is no
   longer *demoted behind* anything. It still needs sign-off — lower LR
   (2e-4 → 5e-5), **or** 1 epoch, **or** lower rank; one change, new OUTPUT_DIR,
   **save intermediate checkpoints**.
2. **Score anaphora, not just interiority and EOS.** `scripts/amplification_test.py`
   gives the rate, the longest run and the length-matched percentile, all static
   and free. It is the sharpest discriminator this investigation has: base 0.0,
   corpus max 8.9, adapter median 24.5. A gentler adapter that lands inside the
   corpus band on anaphora has bought something measurable even if EOS is
   unchanged.
3. **`padding_free` should be pinned explicitly in whatever is trained next**, not
   because it did harm — it demonstrably did not — but because a library default
   silently changed the collation of a run that was documented as
   "byte-identical hyperparameters to variant E". Set it to whatever E used and
   record it. If a future unsloth or transformers pairing loses the packed-sequence
   mask, the same silent default becomes the failure it looked like today.
4. Unchanged: OPEN 3 (entry-count rebalance) demoted, OPEN 4 (chapter re-slice) held.

Budget this session: 0/2 training runs, 0/6 generations, 0 GPU-seconds.

---

## 2026-08-26 (sixth run) — the termination failure is prompt-specific. The adapter closes a response cleanly 3/3; it is the chapter branch it never learned.

**Adapter under test:** `/workspace/drift_sft_out_v6/adapter` (variant F), the
newest, unchanged. **No training run was made (0 of 2). Six generations were made
(6 of 6), ~161 GPU-seconds of sampling.** No corpus edit, no config edit, no
intervention of any kind — so there is no CLEAN / IMPROVED / WORSE verdict to
give.

Scripts added: `scripts/gen_prompt_conditioning.py`,
`scripts/score_prompt_conditioning.py`, `scripts/length_schedule.py`.
Outputs and the raw JSON copied to `logs/`.

### Step 1 — termination in the training data

Unchanged and still closed, for the fourth session running. EOS present,
untruncated (max templated entry 2176 tokens against `max_length` 2560),
unmasked. Nothing below re-derives it.

### The static finding that motivated spending the generation budget

The corpus has exactly **two** user-turn shapes, and they are **perfectly
confounded with target length**. Same system prompt for all 702 records
(sha256 `ed40b81d…`):

```
user turn                     n      assistant words        id prefix
"Write the next chapter."   138      893 - 1500  (med 1420)  L
a one-line scene brief      564        9 -  171  (med   52)  S/R/H/P/K/N/B/T/C
```

**Zero overlap between the two length distributions.** The adapter was trained on
a perfect prompt→length rule.

Every number this investigation has on record — base CLEAN 4/4, variant F
EOS 5/10, the dose-response curve, the ~20x anaphora amplification, the register
work — was measured at **one** of those two strings. The other branch, 80.3% of
the entry count, had never been sampled. It is also the branch the engine
actually uses: `run_drift_pipeline` passes arbitrary `user_input` straight
through, which INVESTIGATION.md's ENGINE-SIDE section flags as "Untested risk".

### The change made

**None.** This run buys the two missing cells of a 2x2 whose other two cells were
already on disk and free:

```
                   "Write the next chapter."        held-out scene brief
adapter s=1.0      n=10  EOS 5/10, 127-564w         <- BOUGHT, n=3
base    s=0.0      n=4   EOS 4/4,  652-819w         <- BOUGHT, n=3
```

Three held-out briefs, each an instantiation of one of the three commonest
templates in the 564 (14x / 11x / 7x) using corpus names in a pairing that never
occurs — form in-distribution, exact string asserted absent from all 531 corpus
user turns. So this tests "did it learn form→length", not "does an unseen name
confuse it". Each brief was run in **both** arms at the **same seed**, so the
comparison is paired rather than merely matched in n. Cap 2560, temperature 0.7,
min_p 0.05, repeat_penalty 1.05 — identical to every recorded arm.

Two falsifiers ran first, both deterministic, both costing none of the six:

```
max |logits(scale=0.0) - logits(disable_adapter())| = 0.000e+00   <- the base cell IS base
max |logits(scale=1.0) - logits(scale=0.0)|         = 24.750      <- the adapter cell IS live
chapter prompt still tokenizes to 54 tokens                       <- same template as the recorded arms
```

The first matters because the base cell was produced by zeroing the LoRA rather
than loading a second model. It is bit-identical, not approximately identical.

### Step 4 — the three verification generations, and their paired controls

```
arm      B  seed      tok   finish  words  in corpus brief band  >30ch repeat
adapter  1  20260827   46   EOS       33   yes                   no
adapter  2  20260828   68   EOS       57   yes                   no
adapter  3  20260829  119   EOS       96   yes                   no
base     1  20260827  259   EOS      205   no                    no
base     2  20260828  530   EOS      416   no                    no
base     3  20260829  305   EOS      247   no                    no
```

**Adapter 3/3 EOS, 3/3 inside the corpus brief band, 3/3 no repeat.** The
seed-matched base is 0/3 in band and 2.6x-7.3x longer on the same three prompts.

This does **not** meet the standing task's CLEAN bar, and it is important to say
why rather than round it up: the bar requires 600-1500 words, and these are
33-96. But the bar was written for the chapter prompt. At a scene brief, 600
words would be the failure — the corpus teaches 52.

Texture, de-looped, scored with the same instruments as the fourth and fifth runs:

```
set                             words                  anaph%  maxrun   int%
CORPUS brief targets (n=564)    median 52, 9-171           0.0      5     0.0
CORPUS chapter targets (n=138)  median 1420, 893-1500      1.1      3     7.0
NEW  adapter @ BRIEF   (n=3)    [33, 57, 96]               0.0      2     0.0
NEW  base    @ BRIEF   (n=3)    [205, 247, 416]            0.0      1     7.7
REC  adapter @ CHAPTER (n=10)   127-564 novel             24.5     21    19.4
REC  base    @ CHAPTER (n=4)    [652, 713, 788, 819]       0.0      1        -
```

**The ~20x anaphora amplification and the ~3x interiority amplification do not
appear at the brief prompt.** At a brief the adapter sits on its corpus median on
both axes (0.0 / 0.0). The amplification established by the fourth and fifth runs
is real, and it is now known to be **conditional on the chapter prompt**, which no
previous run could have seen.

### Prose — the thing being tested

All three adapter generations, complete, because at this length "the final five
sentences" is most of the sample.

**adapter B1 (33 w, EOS)** — `"Einar has asked Elsa whether she is all right."`

> "Are you all right?" Einar asked. "I don't know," she said. She didn't know what
> she was feeling. He looked at her, and he seemed to be looking for something in
> her face.

**adapter B3 (96 w, EOS)** — `"Sigrid has called Lars to come to bed."`

> Lars had been up since four a.m., working on a manuscript that had been due two
> weeks earlier. He had finished it at seven-thirty, then gone out for a run, then
> spent the rest of the morning cleaning the apartment. When he came back, Sigrid
> was still asleep. He had made coffee, eaten toast, read the paper, and then at
> one o'clock, when he could no longer bear the silence, he had taken the
> manuscript downstairs, put it in his office, and told the secretary to send it
> to the publisher. Then he had returned home.

**base B1 (205 w, EOS), same prompt, same seed** — for contrast, opening only:

> The room was dimly lit, the glow from a single overhead bulb casting long
> shadows across the worn wooden floorboards. […] Her voice, when it came, was low
> and steady, like the creak of a settling house. […] The room held its breath,
> waiting.

The adapter is doing its job here. "casting long shadows", "like the creak of a
settling house", "The room held its breath" is exactly the simile-laden register
the LoRA exists to remove, and the adapter removes it. The fourth run's finding
that a weak adapter is register-indistinguishable from base was measured at the
**chapter** prompt; at a brief the two are not remotely alike.

### What this does to the LIVE HYPOTHESIS

The live hypothesis has been, since the second run: *fine-tuning is overwriting
the base model's ability to close a response.*

**The general form of that is now false.** The adapter closes a response cleanly
3/3 — by EOS, on target length, no loop, and inside its corpus band on both
texture axes. Nothing was destroyed. A destroyed closure capability cannot be
recovered by changing the user turn.

The sceptical alternative — "the adapter just writes short everywhere, so nothing
is conditioned" — is separable and fails. Scored as distance from the length the
corpus teaches **for that branch**:

```
model      @brief  /target      @chapter  /target     own ratio
corpus          52   1.00x          1420   1.00x         27.3x
base           247   4.75x           750   0.53x          3.0x
adapter         57   1.10x           258   0.18x          4.5x
```

(adapter @chapter is the de-looped median; `deloop` reproduces the fifth run's
`true_novel_words` median exactly, 257.5.)

A uniform shortening would miss both targets by a similar factor. This does not.
Relative to base the adapter **corrects** the brief branch — 4.75x of target down
to 1.10x — and moves the chapter branch **the wrong way**, from base's 0.53x down
to 0.18x. One branch was learned. The other was not merely unlearned; the adapter
is further from it than the model it started from.

So the surviving statement is narrower and more useful than the one it replaces:
**the adapter learned a short-form stopping schedule and applies it to both
branches.** That is correct for 564 of 702 entries and catastrophic for 138. The
dose-response result (third run) is untouched and now reads more naturally: dial
the adapter down and you dial down a learned short-stop schedule, which is why
chapter length climbs back toward base.

### This un-demotes OPEN 3, and the fifth run's rebuttal does not cover it

The fifth run demoted the entry-count hypothesis on the grounds that the loss is
token-weighted (`num_items_in_batch`, `average_tokens_across_devices=True`), so
entry count never enters the gradient weighting. That argument is correct and it
still stands **for the interiority amplification it was made about**. It does not
reach the stop decision, for a specific reason: **EOS is one token per entry
regardless of how long the entry is.** Token-weighting therefore gives the 564
demonstrations of "stop at ~52 words" and the 138 demonstrations of "stop at
~1420 words" equal weight *per demonstration* — a 4.1:1 count advantage to the
short schedule that token-weighting does nothing to correct. That is an
entry-count effect on stopping specifically, and it is untested.

Stated as a hypothesis, not a finding: 138 examples is few for a 1420-word
structure, and where the chapter branch is under-learned the model falls back on
a stop prior the brief branch dominates.

### A corpus defect that only the brief branch could reveal

adapter B2 emitted markdown mid-sentence:

> She was not crying, she was not screaming, she was not **>** collapsing in a
> heap on the floor.

It is in the training data. **107 of the 564 brief targets (19.0%) contain `> `,
500 occurrences in total** — always preceded by a space, never by a newline, so
these are markdown blockquote lines flattened into single-line strings when the
corpus was built. **0 of the 138 chapter targets contain it.** The system prompt
says "No headers, labels, or formatting"; 19% of the brief targets contradict it.
Five runs of chapter-prompt evaluation could not have seen this, because the
contamination is entirely in the branch nobody sampled.

Cheap to fix and it is a corpus edit, so it needs sign-off like every other one.
Nothing was changed.

### Verdict

**No intervention was made, so no CLEAN / IMPROVED / WORSE.** As a finding:
**ESTABLISHED, and it falsifies the general form of the live hypothesis.**

The model has not lost the ability to end a chapter — it has never had the
ability to *write* one. It ends a scene perfectly, on target length, in the right
register, in the right texture, 3/3. Given the chapter prompt it applies the same
short schedule, runs out at ~258 novel words against a 1420-word target, and
either fires EOS there or loops to the cap. "Does not know how to end a chapter"
should be restated as **"does not know how to sustain one"**, and the anaphoric
ladder is what filling the gap looks like from the inside.

Two things previously believed are narrowed rather than killed: the amplification
(fourth and fifth runs) is real but conditional on the chapter prompt; the
capacity/pressure reading (second and third runs) survives only as "pressure
enforces a learned short-stop schedule", not as "pressure destroyed closure".

### Next step

1. **The EOS hazard curve — free, static, deterministic, no sampling, no
   sign-off, and it tests the claim directly.** Teacher-force each of the 138
   chapter targets through base and through the adapter and read P(EOS) at every
   position. The claim above predicts the adapter puts a hazard bump at ~50-250
   words that base does not, on text where the true EOS is at ~1420. It is
   forward passes over text already on disk. **Do this before OPEN 2** — it
   converts "learned short-stop schedule" from an inference about medians into a
   measured curve, and it will show whether the schedule is prompt-keyed or
   global.
2. **OPEN 2 (retrain gentler) is no longer obviously the right intervention.** If
   the hazard curve confirms a short-stop schedule, the targeted fix is the
   branch imbalance (OPEN 3, un-demoted above) or masking loss on the brief
   branch's EOS — not lowering the LR, which would weaken the brief branch the
   adapter currently gets right. Still needs sign-off; the point is that the
   sign-off should now be asked for a different change.
3. **Re-measure the engine's real prompts.** The engine sends brief-shaped
   `user_input`, so the production path is the branch that works. The
   ENGINE-SIDE "Untested risk" is now tested and the answer is: length does not
   transfer, and that is by design in the corpus. The engine will get ~57-word
   scenes, not chapters, from a working LoRA.
4. **The `> ` contamination** (107/564 brief targets, 500 occurrences) — propose
   stripping it. Corpus edit, needs sign-off.
5. Unchanged: OPEN 4 (chapter re-slice) held; OPEN 10 (pin `padding_free`) still
   applies to whatever is trained next.

Budget this session: 0/2 training runs, **6/6 generations**, ~161 GPU-seconds.

---

## 2026-08-26 (seventh run) — the EOS hazard curve. The stop decision is not broken and is not a schedule at the chapter prompt; the anaphoric ladder is the single cause of both failure modes

Adapter under test: `/workspace/drift_sft_out_v6/adapter` (variant F, r=alpha=16,
attention-only q/k/v/o, 160 LoRA layers, scaling 1.0). Corpus
`/workspace/final_training_corpus_v2_1_latest.json`, system sha256 `ed40b81d…`.

**No generations were sampled this session.** Every number below comes from
forward passes over text that was already on disk. Budget: **0/6 generations,
0/2 training runs**, ~26 GPU-minutes.

New scripts: `scripts/eos_hazard.py` (committed last session, run here),
`scripts/eos_hazard_selftext.py`, `scripts/report_eos_hazard_selftext.py`.
Outputs in `logs/eos_hazard.log`, `logs/eos_hazard_selftext.log`.

### Step 1 — termination in the training data

Unchanged, still closed, fifth session running. EOS present, untruncated (max
templated entry 2176 tokens against `max_length` 2560), unmasked. The standing
task's step-2 branch is therefore **"EOS PRESENT and untruncated"**, and its
instruction for that branch — say so plainly rather than invent a fix — is why
no intervention was made. None of the four legal single changes (append EOS,
raise MAX_SEQ_LEN, drop over-long entries, `packing=False`) has anything to act
on, and this run establishes that the defect is not in the EOS token at all.

### What was measured

Teacher-force text through the model and read `p_eos(t) = softmax(logits[t])[151645]`
at every position. The text is held fixed, so any difference between arms is the
stop decision and nothing else — no sampling noise, no length feedback, no loop
dynamics. `S(k) = prod(1 - p_eos)` up to word k is the probability it has not yet
stopped; `E[stop]` is the expected stopping word with the true ending as fallback.

Four falsifiers, all deterministic, all run before any measurement, all passed in
both scripts:

```
max |logits(scale=0.0) - logits(disable_adapter())| = 0.000e+00   <- base arm IS base, bit-identical
max |logits(scale=1.0) - logits(scale=0.0)|         = 14.375      <- adapter arm IS live
chapter generation prefix                           = 54 tokens   <- same string every recorded arm used
max |wordmap - len(text.split())| over 840 entries  = 0 words     <- word positions are exact
```

### Arms 1-3 — the corpus arms (`scripts/eos_hazard.py`, 1467s, 840 entries x 2 scales)

```
arm              model    n    true w  premature mass  E[stop] w  terminal P(EOS)
chapter@chapter  base     138   1420   0.064            1343       0.011
chapter@chapter  adapter  138   1420   0.256            1264       0.057

brief@brief      base     564     52   0.000              51       0.000
brief@brief      adapter  564     52   0.381              41       0.481

chapter@brief    base     138   1420   0.223            1193       0.018
chapter@brief    adapter  138   1420   0.995             215       0.293
```

`chapter@brief` is the prompt-key cell: the **same 138 chapter texts**, the only
change being the user turn, swapped from `"Write the next chapter."` to a
held-out scene brief. Sampling can never run this comparison, because changing
the prompt changes the text and length stops being a control.

Survival — P(still writing) at word k, median over entries:

```
arm              model    S(50)   S(100)  S(250)  S(500)  S(1000)
chapter@chapter  base     1.000   1.000   1.000   1.000   0.986
chapter@chapter  adapter  1.000   1.000   0.998   0.984   0.895
brief@brief      base     1.000   1.000   1.000   1.000   1.000
brief@brief      adapter  0.738   0.623   0.619   0.619   0.619
chapter@brief    base     1.000   1.000   0.995   0.965   0.861
chapter@brief    adapter  0.903   0.625   0.222   0.076   0.016
```

Mean per-token hazard, pooled over every position of every entry, adapter arm:

```
words      chapter@chapter   brief@brief   chapter@brief    (chapter@brief / chapter@chapter)
  0-24     2.11e-07          3.78e-03      1.23e-03          5800x
 25-49     1.42e-05          1.27e-02      4.77e-03           336x
 50-74     7.06e-05          1.27e-02      6.95e-03            98x
100-124    6.10e-05          1.02e-02      5.42e-03            89x
250-274    7.62e-05             -          4.18e-03            55x
```

### H1 is falsified. The prompt-key test resolves, and it resolves hard.

The pre-registered H1 was: *on the chapter targets the adapter puts EOS hazard
mass at ~50-250 words where the true EOS is at ~1420, and base does not.*

It does not. **`chapter@chapter` adapter S(250) = 0.998.** Given the chapter
prompt and in-distribution chapter text, the adapter has essentially zero
probability of stopping before word 250, its hazard in the first 25 words is
*two orders of magnitude below base's*, and its premature mass sits **late** —
median peak at word 945, with the curve ramping into the real ending at
1300-1424. E[stop] is 1264 against a true 1420, versus base's 1343. That is a
mildly early stopper, correctly shaped, not a short-stop schedule.

The schedule is real, and it lives **entirely in the user turn**. Same text, same
positions, only the prompt string changed: premature mass 0.256 → 0.995, E[stop]
1264 → 215 words, S(250) 0.998 → 0.222, paired **138/138 entries**, median paired
E[stop] delta **-918 words**. Base moves too on the same swap (0.064 → 0.223,
E[stop] 1343 → 1193), which is the off-distribution confound doing its work and
is why this is read as a difference-in-differences: base loses 150 words, the
adapter loses 1049.

The `brief@brief` control also comes out the right way round, and is worth
stating on its own. Base terminal P(EOS) at the true end of a scene brief is
**0.000** — under teacher forcing base essentially never wants to stop. The
adapter's is **0.481**. The LoRA *installed* a stop decision that base does not
have. It installed it correctly, keyed to the brief prompt, which is what
`brief@brief` E[stop] 41 against a true 52 shows.

### Consequence: the sixth run's headline is half retracted

The sixth run's surviving statement was: *"the adapter learned a short-form
stopping schedule and applies it to both branches."*

- **First clause: CONFIRMED, and now measured rather than inferred.** The
  short-form schedule exists (brief@brief terminal P 0.481 vs base 0.000).
- **Second clause: FALSE.** It does *not* apply it to the chapter branch.
  S(250) = 0.998 there. The schedule is prompt-keyed, not positional or global.

That inference was drawn from sampled medians (adapter@chapter novel median
258 words), which is a legitimate thing to have concluded from sampled data and
is simply wrong. The ~258-word sampled median is **not** the stop schedule
firing. Which raises the question this run had to answer next.

### Arm 4 — the self-text arm, the control that decides how to read the above

If the adapter sustains to ~1264 words on corpus chapter text, why do its own
chapter generations die at ~258 novel words or loop to the cap? Teacher forcing
on corpus text cannot see this, because it holds the prefix on the corpus
manifold — precisely the thing sampling does not do. So the same instrument was
pointed at the adapter's **own ten recorded chapter generations** (fourth run,
`gen_v6_cap2560.json`, 5 EOS / 5 CAP, unmodified, on disk since). Free, static,
no sampling.

**P1 — were the recorded EOS events real stop decisions, or sampling luck?**
Real decisions. Terminal P(EOS) at the last word the model actually wrote:

```
gen  end   words   adapter terminal P   base terminal P
5    EOS     200   0.2062               0.0037
9    EOS     252   0.9678               0.8904
8    EOS     263   0.3427               0.1625
1    EOS     374   0.7522               0.2645
4    EOS     405   0.7214               0.7187
                   median 0.7214        median 0.2645
2,3,6,7,10  CAP  1861-2362  all 0.0000  all 0.0000
```

**P2 — is the loop an EOS-suppressing attractor?** The loop entry point is the
first word from which the remainder is verbatim-repeated earlier material
(longest duplicated k-gram; reproduces the fourth run's recorded
`max_repeat_span` exactly on all five looping texts, 1177/930/1075/1181/1131).

Per-entry the within-text ratio is noisy and does **not** establish P2: adapter
loop/novel hazard ratio median 0.316, suppressed in only 3/5; base 15.6,
suppressed in 1/5. Reported as not established at n=5.

The pooled curve is the stronger evidence, and it is unambiguous. Mean per-token
hazard over the five CAP generations by word position:

```
words      CAP/adapter   CAP/base       words       CAP/adapter   CAP/base
150-174    2.32e-04      3.12e-06       800-824     2.53e-05      9.62e-06
300-324    3.08e-04      1.54e-04      1200-1224    6.80e-06      1.76e-06
600-624    2.08e-04      8.76e-05      1800-1824    1.53e-06      2.38e-07
                                       2200-2224    1.60e-06      1.16e-07
```

**The adapter's stop hazard decays ~190x from its peak as the loop runs**, and
keeps decaying to the cap. The adapter stays 2-14x above base throughout, so this
is not the adapter being uniquely trapped — but both are trapped, and a
generation whose per-token P(EOS) is 1.6e-6 at word 2200 cannot terminate. "Ran
to the cap" is not "forgot to stop"; it is **stopped being able to stop**.

And the contrast that matters: on the generations that *did* end, adapter hazard
reaches **1.04e-02 at words 275-299**, against **8.56e-05** for the CAP
generations in that same 25-word band — a **121x** separation, and **34x** even
against the CAP population's own lifetime peak (3.08e-04, words 300-324). By word
~275 the two populations have already parted.

### Prose — and this is where the two arms become one finding

The hazard peak on the EOS generations sits at words 275-400. Here is gen 4
(405 w, EOS) at its peak, `p_eos = 0.346` at word 281:

> He had lost his faith, and he had lost his will. He didn't know what else to
> lose. He had nothing more to lose. He was afraid of the dark. He was afraid of
> himself. He was afraid of everything. He didn't know how much longer he could

and its final sentences, terminal `p_eos = 0.721`:

> He was alone in the world, and he was afraid of being alone. He didn't know
> what to do. He didn't know who he was. He didn't know who he was anymore. He
> was afraid of the dark. He was afraid of himself. He was afraid of everything.
> **And then he cried.**

gen 9 (252 w, EOS), terminal `p_eos = 0.968`:

> He looked into his eyes. He saw the fear there. He saw the pain. He saw the
> suffering. He saw the death that was coming. **And then he pulled the trigger.**

gen 1 (374 w, EOS), terminal `p_eos = 0.752`:

> I thought about how long we were going to be together. I thought about how long
> we were going to be apart. I thought about how long we were going to be together
> again. I thought about how long it was going to take before we were together
> again. I thought about everything that was going to happen between us.

Now gen 10 (2263 w, CAP), words 200-262, loop entry at word 233:

> He thought about how much he had wanted to talk about it. He thought about how
> much his father-in-law had wanted to talk about it. He thought about how much
> they had talked about it. He thought about how much they had wanted to talk
> about it. He thought about how much he had wanted to talk about it. He thought
> about

and its final words, 2000 words later, `p_eos = 0.0000`:

> about how much he had wanted to talk about it. He thought about how much his
> father-in-law had wanted to talk about it. He thought about how much they had
> talked about it. He thought about how much they had

**It is the same text in both populations.** Every one of these ten generations
is climbing the anaphoric ladder the fourth and fifth runs measured at ~20x
corpus rate. The ladder is what the adapter emits when it has to sustain a
chapter and has nothing to sustain it with. The two observed failure modes are
just the ladder's two exits:

- **Exit A — cadence.** The ladder lands an "And then he cried." / "And then he
  pulled the trigger." closure, the model reads a finished piece, hazard jumps to
  0.2-0.97, it emits EOS at 200-405 words. This is the "stops at ~100 words"
  failure, and **the stop decision is correct** — the text really has ended.
- **Exit B — cycle.** The ladder's period closes on itself before any cadence
  arrives. The prefix is now verbatim self-repetition, hazard decays two orders
  of magnitude, and it runs to the cap. This is the "1200 words of a repeated
  block" failure, and **the stop decision is also not wrong** — nothing in a
  cycle looks like an ending.

### Verdict

**ESTABLISHED, with a partial RETRACTION of the sixth run.** No intervention was
made, so no CLEAN / IMPROVED / WORSE.

Three statements now rest on deterministic measurement rather than on sampled
medians:

1. **The adapter's EOS behaviour is not broken.** It installed a stop decision
   base does not have (brief terminal P 0.481 vs 0.000), keyed correctly to the
   prompt that asks for a short answer, and on real chapter text at the chapter
   prompt it sustains to E[stop] 1264 of a true 1420 with S(250) = 0.998.
2. **The short-stop schedule is prompt-keyed, not global.** 138/138 paired,
   -918 words on a user-turn swap alone. The sixth run's "applies it to both
   branches" is retracted.
3. **The anaphoric ladder is the single cause of both failure modes.** Not two
   bugs — one degeneration with two exits. Exit A produces a correct EOS on
   genuinely finished (bad) text; exit B produces an EOS-suppressing attractor.

The consequence for the standing task is direct and worth stating without
hedging: **"make the model emit EOS at the end of a long response" is the wrong
target.** The model emits EOS accurately for the text in front of it. Fixing EOS
would only convert exit-B loops into exit-A stops at ~300 words. The defect is
that the adapter cannot generate 1400 words of chapter without falling into
anaphora, and the fourth and fifth runs already localised that: ~20x corpus
anaphora rate, dose-dependent in LoRA scale, absent at the brief prompt, and not
caused by anything in the training setup (`padding_free`, token-weighting, and
packing all cleared in the fifth run).

### Next step

1. **Do not retrain for EOS.** None of the standing task's four legal changes
   applies, and this run shows the token is not the defect. The next intervention
   must target the ladder.
2. **The one measurement that would name the intervention, and it is free.**
   Teacher-force the 138 chapter targets and read the *anaphora* hazard rather
   than the EOS hazard: at each position, P(the model would begin the same
   sentence-opening n-gram it just used), base vs adapter. Same instrument, same
   text, no sampling, no sign-off. If the adapter's anaphora probability is
   elevated on **corpus** chapter text — where the ladder never actually happens
   because teacher forcing supplies the real continuation — then the pressure is
   in the weights and the fix is training-side (OPEN 3 branch rebalancing, or
   lower LoRA scale per the third run's dose-response). If it is **not** elevated
   on corpus text, the ladder is purely a self-conditioning runaway, and the fix
   is generation-side after all — which would be the first thing to reopen the
   sampling question the standing task closed, and it would need to be argued for
   explicitly.
3. **OPEN 3 (branch rebalancing) still needs sign-off** and is still the leading
   training-side candidate: 564 brief entries vs 138 chapter entries, one EOS
   demonstration each, a 4.1:1 count advantage to the short schedule that the
   loss's token-weighting does not correct. This run does not test it. It does
   remove one argument *for* it — the schedule is not leaking into the chapter
   branch — so the case for OPEN 3 is now "138 examples is too few to learn a
   1420-word structure", not "the brief branch is contaminating the chapter one".
4. **Unchanged and still open:** the `> ` markdown contamination (107/564 brief
   targets, 500 occurrences) — corpus edit, needs sign-off. OPEN 4 (chapter
   re-slice) held. OPEN 10 (pin `padding_free`) applies to whatever is trained next.
root@82746977abdd:/#

---

# Ninth run — base-control cross-measurement

**Verdict: the eighth run's pre-loop finding is retracted. So is the
contamination counter-reading. Both fail for the same reason.**

Run on a 4090; numerics identical to the L4 arms (same weights, same bf16).
Data: `docs/anaphora_basectl.json`.

## Falsifiers

| check | value | pass |
|---|---|---|
| base arm vs adapter-disabled | 0.000e+00 | yes |
| adapter arm live | 14.25 max logit delta | yes |
| route noise, median abs | 0.0158 nats | yes — effect is 1–3 nats |

## What was measured

Base's four recorded chapter generations (`gen_base_control.json`), teacher-forced
through the adapter with the same instrument. These are the right control: long,
chapter-shaped, produced by neither model under test in a way that could have
memorised them, and — verified in `basectl_loop` — completely clean:

```
bc1  788 words  finish EOS  repeat_span 0  loop_entry None
bc2  819 words  finish EOS  repeat_span 0  loop_entry None
bc3  713 words  finish EOS  repeat_span 0  loop_entry None
bc4  652 words  finish EOS  repeat_span 0  loop_entry None
```

## Result

**ELEV is positive on clean base chapter text: median +0.937 across 94
boundaries, 56.4% of them positive.**

Per text: bc1 +0.299, bc2 −1.602, bc3 +3.527, bc4 +2.925.

That is the same sign and the same order of magnitude as the pre-loop cell —
the single positive cell out of 7,481 boundaries that the eighth run built the
whirlpool story on. It appears here on four texts that never ladder and
terminate cleanly.

So positive ELEV is what this instrument reads on off-manifold chapter-length
text in general. It does not mark a pre-loop boundary. Neither ELEV's sign nor
repeat hazard distinguishes a pre-loop boundary from healthy prose.

**Both prior readings are falsified, not one.** The whirlpool story does not
hold. The contamination explanation — that the pre-loop prefixes were already
laddering and that is what produced the signal — does not hold either, because
the effect is present on prefixes that demonstrably are not laddering.

## Absolute hazards close it harder

```
adapter P(ana1)   1.5e-07 … 3.0e-06
base    P(ana1)   7.6e-13 … 4.9e-12
```

Five orders of magnitude apart in relative terms, and five orders of magnitude
below anything that affects sampling. There is no generatively meaningful
elevation in either direction.

## Consequences

1. **This line of measurement is closed.** Three runs went into the anaphora
   instrument. It has said what it can. No further anaphora-hazard runs.
2. **A sampler guard can still be argued**, but only on cheap-intervention
   grounds — loops occurred in 5 of 10 chapter samples. If run, it must be
   scored on **loop incidence and register**, not on ELEV or the pre-loop cell.
3. **Loops and length stay divorced.** Unchanged by this run.

## New observation, unrelated to the anaphora question

The four base controls are 652–819 words and every one finishes by EOS. The
adapter produces 100–450.

The fine-tune did not fail to reach chapter length. It **cut** the length of a
base model that was already writing 700-word chapters and ending them properly.

That reframes OPEN 3. The question is not "teach the model to write long" but
"stop the adapter shortening output by two-thirds" — and LoRA scale is already
known to be dose-dependent here from the third run. Worth testing scale before
paying for a corpus rebuild and a training run.
