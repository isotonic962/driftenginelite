# Autonomous run log — Aug 26 2026

Lab notebook from seven headless Claude Code sessions on the RunPod pod,
running against `AUTONOMOUS_TASK.md` under `--max-turns 100`. Copied here from
`/workspace/driftenginelite/Drift-engine/EXPERIMENT_LOG.md`. The eighth entry
was analyzed off-pod from the uploaded `anaphora_hazard.json` (preserved at
`eval/anaphora_hazard.json`) and its run log (`logs/anaphora_hazard_run.log`);
the ninth ran on a 4090 with its data at `docs/anaphora_basectl.json` and a
verification note appended off-pod.
Two sessions analyzed that JSON in parallel — this entry, and the
ANAPHORA-HAZARD VERDICT section that commit `a825aed` added to INVESTIGATION.md
on `claude/short-output-investigation-iose1e`. They agree on every measured
number; where their verdict sentences differed, the entry below is the
reconciled verdict and supersedes both.

**Snapshot caveat.** This copy was taken before commit `6781c1b` ("Fix a
boundary rule that made the anaphora instrument blind to the ladder"). The
anaphora figures in the later entries come from the pre-fix instrument, which
required both boundary and candidate to be the space-prefixed token form and so
silently dropped every paragraph-initial sentence — 22.7% of the chapter corpus,
and 194 of 197 sentences in `gen10`, the single clearest specimen of the
phenomenon. The EOS-hazard results are unaffected. Re-run pending.
*(The eighth entry below ran on the post-fix instrument: chapter-arm drop rate
0.1%, and `gen10` contributes 40/40 boundaries.)*

**Headline findings across the nine runs:**

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

4. The ladder pressure is not in the weights in any generative sense. On corpus
   chapter text the adapter's repeat hazard is up 8x only at the floor (6e-7 →
   6e-6) with the tail at parity (~0.02 expected rungs per chapter under both
   models), and the elevation is not repeat-specific — control openings are
   boosted more (ELEV −0.64). On the adapter's own laddered text, base is
   equally trapped (mean hazard 0.26 vs 0.25). The ladder is a model-agnostic
   capture; the adapter's contribution is upstream of the first rung. The
   ninth run closes the instrument from the other side: on clean off-manifold
   chapter text (base's own generations) the adapter's repeat hazard is also
   nil (median 5.4e-7, max 1.3e-3), so there is no weight-borne seed an
   exact-repeat probe can find, and the eighth run's pre-loop cell is retracted
   as evidence — its elevation came from prefixes that were already laddering.

Consequence: "make the model emit EOS at the end of a long response" is the
wrong target. Fixing EOS would only convert exit-B loops into exit-A stops at
~300 words. And "retrain to remove the repeat pressure" is equally wrong — the
eighth run shows there is no repeat pressure to remove. Loops and length are
formally divorced: loops → a sampling-time repetition guard (to be validated,
predicted outcome ~250–450-word clean stops); length → training on the chapter
branch (OPEN 3), which remains the sustain deficit's only fix.

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

---

## 2026-08-27 (eighth run) — the anaphora hazard. The repeat pressure is not in the weights in any generative sense; the ladder is a model-agnostic capture that the adapter only has to seed

Adapter under test: `/workspace/drift_sft_out_v6/adapter` (variant F, r=alpha=16,
attention-only q/k/v/o, 160 LoRA layers, scaling 1.0). Corpus
`/workspace/final_training_corpus_v2_1_latest.json`, generations
`/workspace/gen_v6_cap2560.json`, system sha256 `ed40b81d…`.

**No text was generated this session.** Every number is a forward pass over text
already on disk. Budget: **0/6 generations, 0/2 training runs**, ~72 GPU-minutes
(4310s). Raw output preserved in-repo at `eval/anaphora_hazard.json` (sha256
`fe02d9c6…`, 7481 boundaries, byte-identical to the pod's
`/workspace/anaphora_hazard.json`); run log at `logs/anaphora_hazard_run.log`.
This entry was written off-pod from those two files, in parallel with a second
session's analysis of the same JSON (`a825aed`, INVESTIGATION.md on
`claude/short-output-investigation-iose1e`). The two analyses agree on every
number; this entry folds both readings into one verdict.

This is the seventh run's next-step #2, the measurement it said "would name the
intervention": teacher-force the corpus and read the *anaphora* hazard rather
than the EOS hazard — at each sentence boundary, P(the model would begin the
same sentence-opening n-gram it just used), base vs adapter.

### Step 1 — termination in the training data

Unchanged, still closed, sixth session running. Nothing below touches it.

### What was measured

At each of up to 40 sentence boundaries per entry of teacher-forced text, the
joint log-probability of a K=6-token candidate continuation spliced in place of
the true one:

- **ANA1** — the opening 6 tokens of the sentence that just ended. `exp(ana1)`
  is the per-boundary anaphora hazard: the probability that the model, handed
  this exact prefix, would start the next sentence by repeating the opening it
  just used.
- **CTRL** — the opening of a sentence from the same entry ≥8 sentences away
  (`ctrl_min_sent_dist=8`). The pool contains both token forms — space-prefixed
  400 / paragraph-initial 147 — i.e. the post-`6781c1b` boundary rule; the bug
  that blinded the pre-fix instrument is confirmed gone (chapter arm dropped 13
  of 13,538 sentences = 0.1% vs 22.7% pre-fix; `gen10`, which pre-fix lost
  194/197 sentences, now contributes 40/40 boundaries).
- **ELEV** — `(ana1 − ctrl)_adapter − (ana1 − ctrl)_base`, the difference-in-
  differences: the adapter's *repeat-specific* preference net of how much each
  model likes an in-register sentence opening generally.

Three arms: `chapter@chapter` (138 entries, 5288 boundaries), `brief@brief`
(493 of 564 usable, 1855), `selftext` — the ten recorded chapter generations
from the fourth run (338; loop-entry points reproduce the recorded repeat spans
1177/930/1075/1181/1131 exactly).

Four falsifiers, all passed before measurement: scale-0 vs `disable_adapter()`
bit-identical (0.000e+00); scale-1 vs scale-0 live (max 14.375); the chapter
prefix token-identical to every recorded arm (54 tokens); and the spliced route
vs the plain teacher-forced route agreeing to zero-mean bf16 reduction noise,
worst median|Δ| 0.0375 nats on a 6-token joint — common-mode in ELEV, since both
ANA1 and CTRL use the spliced route in both models.

### F1 — on corpus chapter text the floor is up ~8x; the tail, which is where ladders start, is at parity

```
chapter@chapter                  adapter      base
median hazard / boundary         6.1e-06      6.1e-07    (Δ +2.08 nats, higher in 138/138 entries, p=6e-42)
mean hazard / boundary           4.8e-04      6.0e-04
P(hazard > 1e-2)                 1.1%         1.2%
max hazard                       0.115        0.304
expected rungs / 40 boundaries   0.019        0.024
per-entry mean ratio             median 1.25x, adapter higher in 87/138, p=0.003
```

The pre-registered word "elevated" splits in two, and the split is the finding.
At the **median** boundary the adapter's repeat hazard is 8x base's, in every
single entry — the pressure is detectably in the weights. But 8x of 6e-7 is
6e-6, and generation does not sample medians: a ladder needs a rung actually
drawn, which happens in the **tail**, and the tail is at parity — mean hazard
slightly *below* base, identical mass above 1e-2, base holding the maximum.
Expected sentence-opening repeats per chapter: ~0.02 under both models. Handed
corpus chapter text, the adapter would ladder essentially never, and so would
base.

### F2 — and the floor elevation is not anaphora-specific

Raw boost vs base, median nats, alongside the same-entry control opening:

```
arm              ana1     ctrl     ELEV (dd)   entries ELEV<0
chapter@chapter  +2.08    +2.83    -0.641      113/138   p=1.5e-14
brief@brief      +6.30    +9.52    -2.842      400/493   p=2e-46
selftext         +4.50    +5.50    -1.759      —
```

Whatever opening you splice in, the adapter boosts it — and it boosts a
*different* same-entry opening by more than it boosts the repeat. Net of that,
the adapter's repeat-vs-control odds are about **half** of base's (e^-0.64) on
chapter text. The obvious deflator — same-entry controls are memorized training
text — cannot explain the sign: ELEV is negative on `selftext` too, where
nothing is memorized. What the adapter installs is mass concentration onto
in-register sentence openings *as a class*, not a repeat operator.

### F3 — on the adapter's own generations, base is exactly as trapped

```
selftext (338 boundaries)        adapter      base
mean hazard / boundary           0.246        0.260
P(hazard > 0.1)                  28.1%        28.7%
P(hazard > 1e-2)                 34.3%        32.5%
```

Per generation, mean hazard per boundary:

```
gen  finish  words   adapter    base       P>0.1 (a / b)
1    EOS      374    1.8e-01    2.6e-01    40.0% / 36.7%
2    CAP     2354    7.8e-01    8.0e-01    82.5% / 82.5%
3    CAP     1861    3.5e-05    1.9e-07     0.0% /  0.0%
4    EOS      405    2.8e-02    5.4e-02    10.3% / 15.4%
5    EOS      200    4.4e-03    1.5e-03     0.0% /  0.0%
6    CAP     2151    2.4e-05    3.1e-07     0.0% /  0.0%
7    CAP     2362    9.3e-01    9.3e-01    92.5% / 95.0%
8    EOS      263    2.4e-03    1.8e-03     0.0% /  0.0%
9    EOS      252    2.1e-03    7.8e-05     0.0% /  0.0%
10   CAP     2263    2.1e-01    2.2e-01    22.5% / 22.5%
```

The same floor-vs-tail structure as F1, now on the adapter's own text. Wherever
the hazard is generatively large — the sentence-opening ladders and loops of
gens 1, 2, 4, 7, 10 — **base teacher-forced on the same text matches the
adapter almost exactly**, to the percentage point at the deep loops (gen2 0.78
vs 0.80, gen7 0.93 vs 0.93, saturating at ≈1.0 under both). Where the adapter
is 10–100x above base (gens 3, 6, 9), the absolute level is 1e-5 to 1e-3 —
floor again, generatively irrelevant; gens 3 and 6 loop on a period that is not
a sentence-opening repeat, which is why this instrument reads them low. And the
laddered text is if anything *more* predictable to base than to its author:
median 6-token true-continuation logprob −0.04 (base) vs −0.20 (adapter).

Conditioned on text that already ladders, any model of this class is captured —
this is induction copying, not adapter weights. It is the same result as the
seventh run's base-also-trapped hazard decay, now measured on the anaphora side.

### F4 — position

The raw adapter-vs-base gap is largest in the first 100 words (+3.35 nats,
decaying to +1.9 by mid-chapter) while absolute hazard stays flat at ~e^-12;
ELEV runs −0.91 → −0.28 across the chapter. The short-form pull is strongest
where the brief branch's opening moves are most applicable — still floor-level
everywhere.

### The pre-registered branch, resolved — mechanism to one side, policy to the other

The seventh run posed a binary: elevated on corpus text → the pressure is in
the weights, fix training-side; not elevated → "the ladder is purely a
self-conditioning runaway, and the fix is generation-side after all."

The measurement lands on the second branch's **mechanism** and rejects its
**policy inference**:

1. *Repeat-specific pressure visible on corpus text* — **falsified**. The only
   elevation is an 8x floor shift at absolute levels five orders of magnitude
   below generative relevance (F1), and it is subsumed by a larger, non-specific
   concentration onto in-register openings (F2, ELEV negative in 113/138).
2. *Self-conditioning runaway* — **confirmed and sharpened**: the runaway is
   model-agnostic capture. Once rungs exist in the prefix, base's hazard equals
   the adapter's (F3), and the seventh run already showed neither model can
   stop from inside it.
3. The remaining question is the **seeding**: the sampler is not what differs
   between base (0/4 ladders when sampling) and the adapter (7/10) — the text
   each model writes before the first rung is. The strongest evidence in this
   dataset about that region is the pre-loop cell, and the parallel analysis
   (`a825aed`) is right that it is the birthplace of every loop and comes out
   the adapter's way in a diff-in-diff: at boundaries on the adapter's own text
   before loop entry, the repeat option sits **0.78 nats below the true
   continuation for the adapter vs 5.69 for base** (~1/2–1/3 vs ~1/300 relative
   odds, ELEV +0.64 — the only positive cell in 7,481 boundaries). Two caveats
   keep this suggestive rather than established: it is n=30, and those prefixes
   are already laddering (the fourth run measured ~24% anaphora in the novel
   prefixes of these very texts), so the cell cannot distinguish "the adapter's
   own clean prose attracts repeats" from "already-laddering prose attracts
   repeats". Also, the ~1/3 figure is odds against the single most likely
   continuation, not a repeat probability — the median hazard in that cell is
   ~2e-4. The uncontaminated version of this measurement is free (next step 1).

### Verdict

**ESTABLISHED.** No intervention was made, so no CLEAN / IMPROVED / WORSE.

Three statements now on deterministic footing: the adapter carries no
generatively meaningful anaphora pressure on corpus chapter text — floor up 8x,
tail at parity, ~0.02 expected rungs per chapter under both models; the ladder,
once entered, is a model-agnostic attractor — base matches the adapter's hazard
on the adapter's own laddered text to the percentage point; and therefore the
adapter's distinctive contribution to the 7/10-vs-0/4 sampling gap lives
entirely **upstream of the first rung**, in the off-corpus register it drifts
into when asked to sustain a chapter it never learned (sixth run) and in the
opening-concentration F2 measures.

Consequence for interventions — the reconciled position of both analyses:

- **"Retrain to remove the repeat pressure" is off the table.** Both write-ups
  agree: this run shows there is no repeat pressure in the weights to remove.
- **A repetition guard at sampling time (`no_repeat_ngram_size≈6` or a DRY
  sampler) is a justified test, and the sampling question the standing task
  closed is formally reopenable on this measurement** — that is `a825aed`'s
  reading, adopted here. But it is loop mitigation, not the fix, and the
  shared prediction goes on record now: it converts exit-B cap-loops into
  exit-A-like clean stops at ~250–450 words, because sustainment is untouched.
  One mechanical caveat this entry adds: an n-gram guard blocks the *verbatim*
  cycle, but the ladder itself is fuzzy repetition ("He was afraid of the
  dark. He was afraid of himself.") where only the opening tokens repeat — a
  small n also forbids legitimate prose (the corpus has natural runs up to 3,
  plus dialogue tags), a large n may pass the ladder. So guarded output must
  be scored on anaphora with the fifth run's instrument, not just on length
  and EOS.
- **What a guard does NOT do: create 1400-word chapters.** Loops → sampler
  guard; length → training (OPEN 3). The two problems are now formally
  divorced, and the causal chain — can't sustain the chapter register → drifts
  toward brief-register openings → concentrated openings collide →
  model-agnostic capture → two exits — has exactly one unmeasured link left,
  the seeding, and it is free to measure before any budget is spent.

### Next step

1. **Close the seeding link, free and static, first.** Teacher-force base's
   four recorded chapter generations (652–819 w, 0.0% anaphora, on disk since
   the fourth run) through both models with this same instrument. Base's text
   is the uncontaminated version of the pre-loop cell: long, chapter-shaped,
   ladder-free, and memorized by nobody. If the adapter's hazard on *that*
   text is elevated in the tail — not the floor — `a825aed`'s whirlpool
   reading is confirmed on clean text and the chain above closes; if it is
   floor-only there too, the seed is in the register drift itself and OPEN 3
   gains its sharpest argument yet. Either way the result sharpens the
   prediction for step 2 at zero cost.
2. **The guard run — ten chapter-prompt samples with `no_repeat_ngram_size≈6`
   or DRY, same seeds, cap 2560 — is the agreed next spend.** It needs
   sign-off (it reopens the sampling question, which both analyses now agree
   this measurement justifies). Score it on length, EOS, register, *and*
   anaphora (fifth run's instrument), with the prediction on record: short,
   clean, right-register pieces at ~250–450 words. That output is the honest
   baseline for the only remaining decision — whether ~300-word scenes are
   enough for the engine to build chapters from, or whether OPEN 3 pays for
   one more training run to buy length.
3. **OPEN 3 (branch rebalancing) unchanged** and still the only path to
   1400-word chapters; this run neither strengthens nor weakens its case — the
   sustain deficit, not repeat pressure, remains the thing a retrain has to
   buy.
4. **Unchanged and still open:** the `> ` markdown contamination (107/564 brief
   targets, 500 occurrences) — corpus edit, needs sign-off. OPEN 4 (chapter
   re-slice) held. OPEN 10 (pin `padding_free`) applies to whatever is trained
   next.

Budget this session: 0/2 training runs, 0/6 generations, ~72 GPU-minutes.

---

## 2026-08-27 (ninth run) — base-control cross-measurement


**Verdict: the eighth run's pre-loop finding is retracted. So is the
contamination counter-reading. Both fail for the same reason.**

Run on a 4090; numerics identical to the L4 arms (same weights, same bf16).
Data: `docs/anaphora_basectl.json`.

### Falsifiers

| check | value | pass |
|---|---|---|
| base arm vs adapter-disabled | 0.000e+00 | yes |
| adapter arm live | 14.25 max logit delta | yes |
| route noise, median abs | 0.0158 nats | yes — effect is 1–3 nats |

### What was measured

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

### Result

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

### Absolute hazards close it harder

```
adapter P(ana1)   1.5e-07 … 3.0e-06
base    P(ana1)   7.6e-13 … 4.9e-12
```

Five orders of magnitude apart in relative terms, and five orders of magnitude
below anything that affects sampling. There is no generatively meaningful
elevation in either direction.

### Consequences

1. **This line of measurement is closed.** Three runs went into the anaphora
   instrument. It has said what it can. No further anaphora-hazard runs.
2. **A sampler guard can still be argued**, but only on cheap-intervention
   grounds — loops occurred in 5 of 10 chapter samples. If run, it must be
   scored on **loop incidence and register**, not on ELEV or the pre-loop cell.
3. **Loops and length stay divorced.** Unchanged by this run.

### New observation, unrelated to the anaphora question

The four base controls are 652–819 words and every one finishes by EOS. The
adapter produces 100–450.

The fine-tune did not fail to reach chapter length. It **cut** the length of a
base model that was already writing 700-word chapters and ending them properly.

That reframes OPEN 3. The question is not "teach the model to write long" but
"stop the adapter shortening output by two-thirds" — and LoRA scale is already
known to be dose-dependent here from the third run. Worth testing scale before
paying for a corpus rebuild and a training run.

### Off-pod verification note (appended on merge, same JSON)

Every number the entry reports reproduces from `docs/anaphora_basectl.json`:
ELEV median +0.937, 53/94 boundaries positive, per-text medians
+0.299 / −1.602 / +3.527 / +2.925, per-text median hazards adapter
1.5e-7 – 3.0e-6 vs base 7.6e-13 – 4.4e-12, all three falsifiers as stated.
Three qualifications go on record with it, none of which reopens the
instrument:

1. **"ELEV is positive on clean base text" is not statistically established —
   and the retraction is stronger without it.** 53/94 is a boundary-level sign
   test of p=0.26, and the boundaries cluster in four texts (3 of 4 positive,
   p=0.625), whose medians swing from −1.6 to +3.5. The defensible statement
   is that ELEV on off-manifold text is **indistinguishable from zero with
   ±3-nat per-text swings** — which kills the pre-loop cell's +0.64 (n=30, one
   cell of 7,481) as evidence just as surely, and without resting on a sign
   that four texts cannot pin down. The root cause is now measurable: the
   base-side term of every relative metric is unstable across text types
   (median `b_ana1`: −14.3 on corpus text, −13.1 on the adapter's generations,
   **−27.4 here** — base essentially never re-uses an opening in its own
   prose). ELEV and the ana−true diff-in-diffs inherit that 13-nat swing, so
   no base-relative quantity from this instrument is comparable across arms.
   Only the adapter's absolute hazard is.

2. **"Neither ELEV's sign nor repeat hazard distinguishes a pre-loop boundary"
   overstates by half, and the half matters.** True for ELEV. Not true for
   hazard: on these clean texts the adapter's median repeat hazard is 5.4e-7
   and the repeat sits 8.4 nats below the true continuation; at the pre-loop
   boundaries it was 2.1e-4 — **~400x higher** — and 0.78 nats below true. So
   the contamination reading is not dead alongside the whirlpool reading; it
   is what this run's own numbers leave standing: clean prefixes (anyone's)
   read nil, and the pre-loop cell read high **because its prefixes were
   already laddering** — self-conditioning, the eighth run's F3, now bracketed
   from both sides. What died in both readings is any *weight-borne seed
   measurable by an exact-repeat probe*: adapter hazard on clean off-manifold
   text (mean 4.2e-5, max 1.3e-3) is below sampling relevance, exactly as the
   entry concludes.

3. **The closing observation is the sixth run's, and the scale suggestion has
   a prior curve against it.** Base at 0.53x of chapter target vs adapter at
   0.18x — "further from it than the model it started from" — is on record in
   the sixth run's table. The third and fifth runs already walked the scale
   axis at the chapter prompt: 0.25 restores termination but is register-null
   (indistinguishable from base on every scored metric), 0.5 is already
   laddering (mean anaphora 30.8, max run 14). The only untested window is
   (0.25, 0.5), and both measured endpoints argue it is empty. A scale sweep
   is cheap and legal to run, but it re-spends generations where the recorded
   dose-response already gave the answer twice; if it runs anyway, score it
   with the fifth run's instruments and say what (0.25, 0.5) point it tests.

Consequences 1–3 above stand as written: instrument closed, guard argued only
on loop-incidence grounds, loops and length divorced. One boundary on the
closure for the record: this instrument probed the *exact previous opening*
(K=6). The first-rung question it cannot see — whether the adapter
concentrates sentence-opening mass enough to make *any* same-opening collision
likely (the eighth run's F2 boost, +2.8 to +9.5 nats on control openings) — is
a different, also-free probe (opening-distribution entropy at boundaries).
Noted as optional; nothing above depends on it.

---

## 2026-09-21 (tenth run) — the guard run. The n-gram guard kills the verbatim loop and converts it into a paraphrase ladder; the recorded prediction is falsified on length and on "clean"

Adapter under test: `/workspace/drift_sft_out_v6/adapter` (variant F), unchanged.
**Ten guarded chapter-prompt samples plus one guard-off repro sample (11
generations), 0 training runs, ~16 GPU-minutes on the pod's L4.** Script
`scripts/gen_guarded.py` as committed (`aa9d410`), run unmodified with
`HF_HOME=/workspace/huggingface-cache` in the environment. Data:
`eval/gen_v6_guard.json`; run log `logs/gen_guarded_run.log`; scorer output
`logs/score_guard_tenth.log`.

### The change made

One: `no_repeat_ngram_size=6`. Sampling config otherwise byte-identical to the
recorded unguarded arm (temperature 0.7, min_p 0.05, repetition_penalty 1.05,
cap 2560, master seed 20260826).

### Falsifiers

```
F1  system sha ed40b81d…, chapter prefix 54 tokens                      pass
F2  adapter live, max |logit delta| = 14.688                             pass
F3  guard-off seed 20260827 sha 2cf9f351bf4c5669 != recorded 79a6a7f0…   FAIL
```

**Seed-pairing does not hold on this host.** Everything below is an unpaired
10-vs-10 comparison; no per-seed before/after claim is made.

### The scorer, and its own falsifier

`amplification_test.py` / `register_check.py` / `register_robust.py` were not on
this branch; restored byte-identical from `99f99d1` (sha256 checked).
`scripts/score_guard.py` imports their `deloop`, `anaphora`, `int_pct`, `fisher`
and the AGRI list. Pointed at the recorded arms it reproduces every number on
record: anaphora median 24.5 / mean 23.3, 7/10 above corpus max, 6/10 runs past
corpus max run, interiority 19.4 at pctile 87.5, 5/10 above length-matched p90,
agri 1.22/1k; base 0.0 / 16.5 / 5.98.

### Result

```
 i fin  raw w  span ch  anaph%  run  int%  int pct@W  agri
 1 EOS    490       26    26.9    5  55.6      100.0     0
 2 EOS   2134       37    56.4   53  62.2      100.0     0
 3 EOS    284       24    12.5    3  11.8       76.1     0
 4 CAP   2022       31    51.6   55   2.1        9.4     0
 5 CAP   2295       39    60.5   15  35.8      100.0     0
 6 EOS    793       32    83.9   41   0.0        4.0     0
 7 EOS    962       36    47.4   11  25.2       99.3     0
 8 EOS    393       26    62.3   20   0.0       13.0     0
 9 EOS    892       29    11.9    3  16.2       91.3     0
10 EOS   1013       29    15.2    3   2.1       15.9     0

arm        n  EOS  loop  med delp w  EOS w range  med an%  mean an%  >corpus MAX  run>3  int pctile  >p90  agri/1k
unguarded 10    5     5         258      200-405     24.5      23.3         7/10   6/10        87.5  5/10     1.22
guarded   10    8     2         927     284-2134     49.5      42.9        10/10   7/10        83.7  5/10     0.00
base       4    4     0         750      652-819      0.0       0.0          0/4    0/4        16.5   0/4     5.98
```

(`loop` = CAP or a ≥200-char repeat. Under the guard the longest repeated
substring anywhere is 39 characters, so both guarded "loops" are cap hits, not
verbatim cycles.)

### Against the prediction on record

The eighth/ninth-run prediction: *exit-B cap-loops become clean EOS stops at
~250–450 words; length is NOT recovered.*

1. **Verbatim loops: gone, by construction.** CAP 5/10 → 2/10, EOS 5/10 → 8/10.
   At n=10 unpaired this is Fisher p = 0.35 — direction as predicted, not
   established.
2. **"Stops at ~250–450 words": FALSIFIED.** EOS lengths 284–2134, median 842;
   2/10 samples land in the band (unguarded arm: 4/10). The guard did not convert
   exit B into exit A at the ladder's usual cadence point.
3. **"Length is not recovered": false in the letter, true in the spirit.**
   De-looped median 258 → 927 words. But the added length is ladder. Anaphora
   median 24.5 → 49.5, 10/10 above the corpus maximum of 8.9, longest
   same-opening runs of 53, 55 and 41 sentences against a corpus max of 3 and an
   unguarded max of 21.
4. **"Clean": FALSIFIED, and this is the finding.** The eighth run's mechanical
   caveat ("a large n may pass the ladder") is confirmed at full strength. With
   the verbatim period forbidden, the model stays on the ladder and paraphrases
   each rung. #6, final 150 words:

   > They were not real. They were not actual. They were not true. They were not
   > genuine. They were not authentic. They were not real in any way. They existed
   > not at all. They did not exist at all.

   #2, final words before EOS at 2134:

   > He did nothing except think for thousands of weeks. Then he stopped thinking
   > for thousands of weeks. He did nothing at all for thousands of weeks.

   #4 runs to the cap on "For everything that is X and everything that is
   un-X." for 55 consecutive sentences. This is the seventh run's exit B with
   the period stretched by a thesaurus — the same capture, and the EOS hazard
   evidently stays suppressed inside it just as it did in the verbatim cycle.
5. **Register: not improved.** Interiority percentile 83.7 vs 87.5, 5/10 above
   the length-matched p90 in both arms. Agri/craft verbs 0 in 10,278 words
   (unguarded 1.22/1k, base 5.98/1k).

Two samples are what a fix would look like: **#9 (892 w) and #10 (1013 w)** end
by EOS with a longest run of 3 (= corpus max) — longer than any base control
and in the plain register ("He said goodbye and left. I sat there for a while,
looking out the window at the street outside. It was snowing."). Their anaphora
rates (11.9, 15.2) are still above the corpus maximum. 2/10 is an existence
proof that variant F can sustain ~1000 words when it does not seed a ladder; it
is not a rate anyone can ship.

### Verdict

**WORSE on the axis that matters, IMPROVED on the one it targeted.** Verbatim
loops 5/10 → 0/10 and EOS 5/10 → 8/10 (n.s.); ladder rate doubled, 10/10 past
the corpus maximum, register unchanged. `no_repeat_ngram_size=6` alone is not a
usable fix and should not go into the engine as one.

What it establishes: the verbatim cycle was never the disease, only its most
compressible form. Remove it and the capture persists as fuzzy anaphora, which
is what runs 5, 7 and 8 said the underlying object was. The part of the
prediction that failed is the assumption that a blocked cycle would fall through
to exit A; it falls through to a longer ladder instead.

### Next step

1. **Charter item 2 is already done**: `/workspace/final_training_corpus_v2_2_bq.json`
   (Aug 26) is v2_1 with all 500 `> ` occurrences stripped — 107 entries differ,
   0 brief targets still contain it, 702 entries, chapter branch untouched.
   Verified this session.
2. **Charter item 3 — the retrain — is next.** One change, new OUTPUT_DIR,
   checkpoints saved, `padding_free` pinned. Score with `scripts/score_guard.py`
   (termination + anaphora + register in one pass) and spot-check 3 briefs.
3. **An untested sampler idea this run points at, static cost already
   measured:** a *sentence-opening* guard — forbid reusing the two-word opening
   of the previous 4 sentences — targets the first rung rather than the
   verbatim period. It would touch 3.7% of corpus long-form sentences, 0.6% of
   base's generations and 34.6% of the adapter's de-looped ones.
   `scripts/gen_opening_guard.py` is drafted and **unrun and untested**; it is
   parked behind the charter order.

Budget this series: 0/2 training runs, **11/60 generations**, ~16 GPU-minutes.

---

## 2026-09-22 (eleventh run) — variant G, the chapter-branch retrain. The sustain deficit moves: the ladder now captures at ~770 words instead of ~260, and three samples are the first chapter-length stops on record; termination is not recovered and the ladder is unchanged

**Adapter under test:** `/workspace/drift_sft_out_v7/adapter` (variant G), new.
**One training run (1 of 2), 13 generations (10 chapter + 3 brief; series total
24/60), 62.5 GPU-minutes training + ~26 GPU-minutes sampling on the pod's L4.**
Scripts: `scripts/train_drift_sft_v7.py`, `scripts/build_corpus_v2_3_ch3x.py`,
`scripts/gen_variant.py` (all at `07c8e19`), `scripts/score_guard.py` (label
argument added, numbers unchanged), `scripts/score_briefs.py` (new). Data:
`eval/gen_v7_variantG.json`; logs `logs/gen_variantG_run.log`,
`logs/score_variantG_eleventh.log`, `logs/score_briefs_variantG.log`; training
log `/workspace/drift_sft_v7_train.log` (pod only, 47 KB).

### The change made

One: the corpus. `final_training_corpus_v2_3_ch3x.json` (sha256 `aa3b644fd68a00b6`,
978 entries) is v2_2_bq (v2_1 with the 500 `> ` occurrences stripped, charter
item 2) with the 138 chapter entries repeated x3: 414 chapter + 564 brief, so the
chapter branch is 42.3% of entries and roughly 88% of assistant tokens, against
19.7% / ~80% in F. Every hyperparameter is F's; `padding_free=True` and
`packing=False` are now pinned to the values F actually ran with (OPEN 10
closed); checkpoints at steps 31/62/93/124 kept. Pre-flight re-asserted in the
script: max templated entry 2176 tokens, 0 truncated. 124 steps, 62.5 min, peak
13.2 GB, final train loss 2.433 (F: 2.379 over 88 steps — not comparable, the
token mix changed).

### Falsifiers

```
F1  system sha ed40b81d…, chapter prefix 54 tokens               pass
F2  adapter live, max |logit delta| = 15.500                      pass
F3  adapter_model.safetensors sha 048c043e… != F's 41d93976…      pass
F4  memorisation: shared 8-grams with the corpus (all 702 targets)
      G 0 / 8,359   F 0 / 2,797   base 0 / 2,980                  pass
```

F4 is the check the x3 recipe owed: six passes over each chapter (3 copies x 2
epochs) reproduce no 8-word span of any training target in 10 generations.

Seed-pairing with the recorded arms is broken on this host (tenth run, F3), so
everything below is unpaired 10-vs-10.

### Result — chapter prompt

```
 i fin  raw w delp w span ch  anaph%  run  int% int pct@W agri  1p/1k
 1 CAP   2062   2062      80    85.6  153   0.0       0.0    0  212.7
 2 CAP   1935    885    5245     4.0    2  15.4      91.3    3   49.4
 3 CAP   2153    735    5030    37.7   18   1.9      13.0    0   60.9
 4 CAP   2192    593    5349    45.0    9  14.3      84.8    0   10.1
 5 CAP   2101    788    5430     8.3    2   7.7      55.4    1   38.1
 6 CAP   1766    604    4111    22.9    5   6.1      46.7    0   61.6
 7 EOS   1206   1206      69    13.7    4  15.6      93.5    0   36.3
 8 EOS    803    758     255    34.6    9   1.3       8.7    0  101.3
 9 CAP   2181    471    6239    59.3   18  38.2      99.3    0   17.0
10 EOS   1362   1312     238    21.9   12  24.3     100.0    0   80.1

arm         n  EOS loop med raw w med delp w  EOS w range med an% mean an% >cMAX run>c med int% med pct >p90 agri/1k
variant F  10    5    5      1133        258      200-405    24.5     23.3     7     6     19.4    87.5    5    1.22
variant G  10    3    9      1998        773     803-1362    28.8     33.3     8     8     11.0    70.1    4    0.41
base        4    4    0       750        750      652-819     0.0      0.0     0     0      1.7    16.5    0    5.98
```

(`delp w` = words before the first ≥200-char verbatim repeat, i.e. how far the
sample gets before the cycle closes. #1 is not a verbatim cycle: it counts
"…the time when I published my eighteenth book. …my nineteenth book…" up to the
hundred-and-thirty-eighth, so the exact-repeat instrument reads span 80 on a
2,062-word ladder; it is a loop by cap only. Without it G's de-looped median is
758.)

1. **Sustain — the axis this run targeted — IMPROVED, and it is the one
   established result.** Words before capture: median 258 → 773, exact
   Mann-Whitney one-sided **p = 3.8e-5** (dropping #1: 758, p = 7.6e-5). Every
   G sample gets further than F's median; G's minimum (471) is above F's
   third quartile. The three EOS stops are at 803, 1206 and 1362 words against
   F's 200–405; all three are longer than every base control (652–819), and #10
   is within 5% of the corpus chapter median (1420). These are the first
   chapter-length EOS stops any adapter has produced in this series.
2. **Termination — not recovered, if anything worse.** EOS 5/10 → 3/10, cap
   7/10 (Fisher p = 0.65). The capture still happens; it happens ~500 words
   later. The seventh run's two exits are both still present: #7, #8, #10 are
   exit A (EOS on a finished-but-degenerate tail — #8 and #10 both end inside a
   ladder, only #7 ends in prose), and the seven cap hits are exit B.
3. **Ladder — unchanged.** Anaphora median 28.8 vs 24.5 (two-sided p = 0.34),
   8/10 above the corpus maximum of 8.9, 8/10 with a same-opening run past the
   corpus max of 3. Training on three times as much clean chapter text (corpus
   anaphora median 1.1) did not move the adapter's ladder rate at all. This is
   the prediction of runs 8–9 ("the ladder pressure is not in the weights in
   any generative sense") holding under a retrain: more chapter data bought
   more chapter before the capture, not less capture.
4. **Register — mixed, n.s.** Interiority median 19.4 → 11.0, percentile
   87.5 → 70.1 (two-sided p = 0.29; corpus median 7.0), so the direction is
   toward the corpus. Agri/craft verbs 0.41/1k (F 1.22, base 5.98): the
   labour vocabulary is still gone. Prompt adherence to "objective physical
   realism" is not restored.

### Result — brief branch (the charter's regression check)

```
 b fin  raw w  span  band  '> '  anaph%  run  int%
 1 EOS     48    12  yes   no      0.0    1  25.0   Einar has asked Elsa whether she is all right.
 2 EOS     25     6  yes   no      0.0    1   0.0   Astrid has just told Bertil something difficult.
 3 EOS     41    16  yes   no      0.0    1   0.0   Sigrid has called Lars to come to bed.
recorded F (sixth run, same briefs, same seeds): EOS 33/57/96 w, 3/3 in band, anaph 0.0, run 2, int% 0.0
```

3/3 EOS, 3/3 inside the corpus brief band (9–171), no repeat, no `> ` marker
(the v2_2_bq strip held). **Not regressed.** Brief 1's 25% interiority is one
sentence of four ("She felt confused, but not in a bad way") and is within the
brief corpus's range.

### Prose

**#7 (1206 w, EOS)** — the one sample that ends in prose. Opening:

> They had just got out of bed when the doorbell rang. Mrs. Svedmyr opened it.
> She was wearing a red bathrobe over her nightgown and her hair was still wet
> from the shower she had taken before going to sleep. "Good morning," said the
> man on the threshold. "I wonder if I might talk to you for a moment?" She
> looked at him. He looked very tired and unshaven. He was a middle-aged man,
> about her husband's age, but he didn't seem to have been born in Sweden. His
> clothes were not especially good but they were clean, and his shoes were worn
> and polished.

and its close, still leaning on the ladder but stopping:

> She thought that he was lying to her, or that he had lost his mind. She
> thought that he was trying to make her feel afraid. […] "You must believe
> me," he said. "I promise you that I am telling the truth." Mrs. Svedmyr
> looked at him. She did not answer. She did not know what to say. She did not
> believe him, and she did not know what to do. She looked at the man. He
> looked at her.

**#10 (1362 w, EOS)** — exit A inside the ladder, the seventh run's pattern at
three times the length:

> I thought about how I wanted to lie down on the bed and sleep for a couple of
> hours and not think about anything else. I thought about how I wanted to lie
> down on the bed and sleep for a couple of hours and not think about anything
> else. I thought about how I wanted to lie down on the bed and sleep for a
> couple of hours and not think about anything else.

**#9 (cap)**, the last novel words before the cycle closes at 471:

> She was afraid. She was lost. She was desperate. She could not find anything
> to believe in. She could not find any reason to believe in anything anymore.
> She had lost everything. She had lost her father. She had lost her mother.
> She had lost everything.

### Verdict

**IMPROVED on the targeted axis (sustain), brief branch not regressed; WORSE
or unchanged on termination and ladder.** By the charter's discipline this is
IMPROVED — G beats F on the axis the intervention targeted without regressing
the production path — but it is not shippable at the chapter prompt: 7/10 cap
hits versus F's 5/10, and the three stops are exit A.

What it establishes: **length and the ladder are separable in training, as
runs 7–9 said they were in measurement.** One corpus change tripled the
pre-capture length (p = 4e-5) and left the ladder rate exactly where it was
(p = 0.34). OPEN 3 has done what it could do; the sustain deficit is no longer
the binding constraint. The binding constraint is the capture, and the
evidence of ten runs is consistent that no corpus edit reaches it.

Two instrument notes for the record. (a) #1's ordinal ladder is invisible to
the exact-repeat instrument (span 80 on 2,062 words); had it ended by EOS it
would have scored as a clean 2,000-word sample. The `looped` criterion needs
the anaphora run as a second trigger (run > corpus max of 3 for ≥ N
sentences), not just CAP-or-≥200-char. (b) The G loss curve is not comparable
to F's — 88% of tokens are now chapter text, which carries higher per-token
loss than briefs.

### Next step

1. **The sentence-opening guard on variant G** (`scripts/gen_opening_guard.py`,
   `--adapter /workspace/drift_sft_out_v7/adapter`, window 4, no n-gram
   guard). Rationale: G moved the capture point to ~770 words; the tenth run
   showed the n-gram guard passes the ladder because it blocks the period not
   the rung. The opening guard blocks the first rung. Prediction on record:
   **loop rate 9/10 → ≤ 5/10; EOS ≥ 6/10 with a median EOS length ≥ 800; the
   guard fires ≥ 20x per sample** (34.6% of F's de-looped sentences would be
   touched). Falsifier: if the guard fires but cap-loops persist at ≥ 7/10, the
   capture is not rung-initiated and the guard idea is dead. Cost: 10
   generations, 0 training runs.
2. **The last training run is held.** Nothing in this result says what a
   second corpus change would buy; a candidate only if (1) shows the guard
   converts G's captures into stops, in which case the remaining question is
   whether a further chapter upweight (x5, or 3 epochs) moves the capture
   point again.
3. OPEN 4 (chapter re-slice) held. `> ` contamination closed (v2_2_bq is in
   G). OPEN 10 closed (pinned).

Budget this series: **1/2 training runs, 24/60 generations**, ~88 GPU-minutes
this session.

---

## 2026-09-22 (twelfth run) — the sentence-opening guard on variant G. Cap-loops 7/10 → 4/10 and EOS 3/10 → 6/10 at chapter length (n.s., paired 4 up / 1 down); the ladder is relocated, not removed — first-word opening reuse is unchanged and the capture finds a longer period

**Adapter under test:** `/workspace/drift_sft_out_v7/adapter` (variant G),
unchanged. **Ten guarded chapter-prompt samples (series total 34/60), 0
training runs, ~20 GPU-minutes.** Script `scripts/gen_opening_guard.py`
(drafted in the tenth run; two edits before running: it now decodes only the
last 512 tokens per step instead of the whole text — a cost fix, the rule is
unchanged — and it was unit-tested offline against the tokenizer, where it
bans the space-led completion of a repeated first word and, as a known gap,
misses a second word that the tokenizer splits). Data
`eval/gen_v7_variantG_openguard.json`; run log
`logs/gen_openguard_variantG_run.log`; scorer `logs/score_openguardG_twelfth.log`.

### The change made

One: a logits processor that forbids a new sentence from reusing the two-word
opening of any of the previous 4 sentences (window 4, no n-gram guard, no
scale change). Sampling otherwise byte-identical to the eleventh run.

### Falsifiers

```
F1  system sha ed40b81d…, prefix 54 tokens                              pass
F2  adapter live, max |logit delta| = 15.500 (= eleventh run)            pass
F3  seed-pairing with the eleventh run: every seed shares an identical
    prefix with its unguarded twin (13–543 words) until the guard first
    fires                                                                pass
```

F3 passes here where it failed against the August arms: same host, same
weights, same seeds. So this comparison, uniquely in the series, is
**paired**. Comparisons with F and base remain unpaired.

### Result

```
 i fin  raw w delp w span ch  anaph%  run  int% int pct@W agri  1p/1k  guard fired   unguarded twin
 1 EOS    157    157      13     0.0    1   0.0      27.2    0  101.9        5        CAP 2062 (ordinal ladder)
 2 CAP   2160   1031    5599     0.0    1  12.5      83.7    4    0.0      142        CAP 1935
 3 CAP   1956   1540    1988    33.3   11   2.1       9.4    1   88.5      289        CAP 2153
 4 EOS    887    604    1450     0.0    1   9.4      65.2    0  115.9       48        CAP 2192
 5 EOS    926    836     419     4.4    2  21.7      97.1    0   37.1       52        CAP 2101
 6 EOS    486    486      26     9.3    4   2.3      19.9    3   52.0       22        CAP 1766
 7 EOS   1423   1302     586     5.8    5   6.7      46.0    0   30.6       75        EOS 1206
 8 CAP   1824    513    4535    16.5    6   0.0       8.3    0   93.7      270        EOS  803
 9 CAP   1783   1079    3865     0.0    1  11.4      78.3    0   51.0      116        CAP 2181
10 EOS    861    861      33     4.4    2  10.1      73.9    0   87.1       40        EOS 1362

arm           n  EOS CAP loop* med delp w  EOS w range  med an%  >cMAX  med int%  med pct  >p90  agri/1k  first-word reuse
variant F    10    5   5    5        258      200-405     24.5    7/10     19.4     87.5     5     1.22     70.0
variant G    10    3   7    9        773     803-1362     28.8    8/10     11.0     70.1     4     0.41     41.3
G + guard    10    6   4    7        848     157-1423      4.4    3/10      8.1     55.6     1     0.94     42.2
base          4    4   0    0        750      652-819      0.0    0/4       1.7     16.5     0     5.98     32.7
corpus chapters (n=138)                                    1.1 (max 8.9)      7.0                             18.3 (p90 29.2, max 37.2)
```

(`loop*` = CAP or a ≥200-char verbatim repeat. `first-word reuse` = share of
sentences whose first word equals the first word of one of the previous 4
sentences, on de-looped text — a post-hoc instrument, defined this run because
the two-word anaphora rate is what the guard forbids and so cannot serve as
its evidence.)

Against the prediction on record (eleventh run, next step 1):

| prediction | result | |
|---|---|---|
| loop* 9/10 → ≤ 5/10 | 7/10 | **failed** |
| EOS ≥ 6/10 | 6/10 | met, at the boundary |
| median EOS length ≥ 800 | 874 (157, 486, 861, 887, 926, 1423) | met |
| guard fires ≥ 20x per sample | 9/10 (#1 fired 5x and stopped at 157) | met |
| falsifier: cap-loops persist ≥ 7/10 → not rung-initiated | cap 4/10 | not triggered |

1. **Termination — direction right, not established.** Paired by seed: 4
   seeds go CAP → EOS (#1, #4, #5, #6), 1 goes EOS → CAP (#8), 5 unchanged.
   Exact McNemar on the 5 discordant pairs p = 0.375; unpaired Fisher on
   3/10 vs 6/10 p = 0.37. Cap-loops 7/10 → 4/10 (p = 0.37). Three of the six
   EOS samples (#4, #5, #7) still contain a ≥200-char verbatim block before
   the stop — exit A inside a cycle, the seventh run's pattern — so on the
   loop* criterion the improvement is 9/10 → 7/10 (p = 0.58).
2. **Length — held.** De-looped median 848 (G 773, p = 0.58); EOS lengths
   157–1423, median 874. #7 stops at 1423 words, the corpus chapter median to
   within three words. Against F (258, EOS 200–405) the eleventh run's
   sustain result stands under the guard.
3. **The ladder moved; it did not go away. This is the finding.** The
   two-word anaphora rate falls 28.8 → 4.4 and 8/10 → 3/10 past the corpus
   max — *by construction*, and not to be quoted. On the instrument the
   guard does not touch, first-word reuse is 41.3 → 42.2 (paired-seed
   two-sided p = 0.58), still above base (32.7) and the corpus p90 (29.2).
   The four caps are cycles the rule cannot see: #2 is an eight-sentence
   period in which every sentence opens "He …" with a different second word
   ("He sat at the table … He looked around the room … He wanted to remember
   it … He got up and went …"), #8 and #9 likewise. Same result as the tenth
   run's n-gram guard, one level up: forbid the verbatim period and the
   capture paraphrases; forbid the two-word rung and the capture varies the
   second word and lengthens the period past the window.
4. **Register — best numbers in the series, all n.s.** Interiority median
   8.1, percentile 55.6, 1/10 past p90 (G: 11.0 / 70.1 / 4; corpus median
   7.0; two-sided p = 0.44 vs G). Agri 0.94/1k. Not evidence of anything on
   its own; noted because it is the first arm to sit inside the corpus's
   interiority band.
5. **A post-hoc observation the eleventh run missed.** On first-word reuse,
   F → G is 70.0 → 41.3, unpaired two-sided p = 0.018. The retrain *did*
   reduce the single-word ladder that the two-word instrument (p = 0.34)
   could not see. `[post-hoc instrument, one comparison, not pre-registered]`
   — it goes on the list of things a pre-registered replication would test,
   not into the headline.

### Prose

**#1 (157 w, EOS)** — whole. The seed that unguarded ran an ordinal ladder to
2,062 words; the guard fired five times and it closed a scene:

> …It means that I can't write about anything else. You are right. It might
> have been better if I hadn't said anything at all. But now I can't stop
> thinking about it. You have no reason to fear anything. You will write about
> what interests you, and it will be fine. If you like, I will help you look up
> some sources on your own. Thank you. That would be nice. I have to go.
> Goodbye.

**#7 (1423 w, EOS)** — same opening as the eleventh run's #7 (paired), and it
ends by repeating one dialogue paragraph verbatim and then stopping:

> "She will," the man said. "She has a very strong mind, and she is determined
> to recover. I know she will recover. But I don't want to lose her. I need
> your help, Mrs. Svedmyr. Please help me."

**#2 (cap)**, the cycle the guard cannot see:

> He sat at the table with a cup of coffee in front of him. He looked around
> the room, thinking about the dream he had had. He wanted to remember it, but
> he couldn't. He got up and went to work in the kitchen. He didn't remember
> any dreams that night either. He felt frustrated, because he couldn't
> remember the dream. He worked in the garden the next day, but he couldn't
> remember any dreams.

### Verdict

**IMPROVED on termination in direction only (paired 4 up / 1 down, p = 0.375),
length held, ladder relocated not removed.** The guard is not a fix. Combined
with the tenth run it establishes a pattern worth stating as such: **every
sampler-side constraint so far is absorbed by the capture at the next level
of abstraction** — verbatim period → paraphrase; two-word rung → one-word rung
with a longer period. A constraint that reaches the first-word level would
touch 18% of corpus chapter sentences (median) and cannot be applied.

Engine note: `engine/drift_engine.py` samples the local backend at
`repeat_penalty=1.1`, `max_tokens=2048`, and `rolling_baseline.py` varies
temperature/penalty per state (0.55–0.78 / 1.03–1.08). None of the recorded
arms were run at those settings; nothing measured here transfers to the engine
without a run at the engine's own sampler, and the opening guard is an HF
logits processor that the llama.cpp backend cannot host as written.

### Next step

1. **No second guard variant.** Two sampler runs have both been absorbed; a
   third (wider window, first-word rule) is predicted to be absorbed or to
   exceed the corpus's own reuse rate, and would spend 10 generations to
   learn which.
2. **The last training run, if spent, should target the capture's period
   rather than the branch mix.** The one thing that has moved the ladder at
   all is G's corpus change (first-word reuse 70 → 41, post-hoc). Candidates,
   one per run, cheapest first: (a) 1 epoch on v2_3 (halves exposure; tests
   whether the capture strengthens with epochs — free to falsify against
   `checkpoint-62`, which is epoch 1 of this run, at 10 generations and no
   training); (b) chapter x3 with the brief branch dropped to x0.5 (tests
   whether the brief branch's short-stop schedule is what fires exit A inside
   the ladder). **(a) via checkpoint-62 first: it costs no training run.**
3. **Engine-side, when an adapter is worth shipping:** run 10 samples at the
   engine's own sampler settings before changing anything in `drift_engine.py`.

Budget this series: **1/2 training runs, 34/60 generations**, ~108
GPU-minutes this session. **Push is blocked on this pod** (no GitHub
credential: `could not read Username for 'https://github.com'`); the tenth,
eleventh and twelfth runs are committed locally on `claude/new-session-z1flke`
and need the owner to push.

---

## 2026-09-22 (thirteenth run) — checkpoint-62, variant G at epoch 1. The capture and the sustain are both already there at one epoch; the epoch count is not the lever, and a 1-epoch retrain is dead before it is paid for

**Adapter under test:** `/workspace/drift_sft_out_v7/checkpoint-62` (variant G,
epoch 1.0 of 2, step 62 of 124; `adapter_model.safetensors` sha `756273cf…`,
distinct from the final `048c043e…`). **Ten chapter samples (series total
44/60), 0 training runs, no briefs (a checkpoint is not a ship candidate),
~24 GPU-minutes.** Prediction pre-registered and committed before sampling in
`logs/prediction_ckpt62.txt` (`c3f7c2e`). Data `eval/gen_v7_ckpt62.json`; run
log `logs/gen_ckpt62_run.log`; scorer `logs/score_ckpt62_thirteenth.log`.
`scripts/gen_variant.py` unmodified, unguarded, same seeds.

### The question

Six passes over each chapter (3 copies x 2 epochs) — does exposure strengthen
the capture, so that the last training run should be 1 epoch on v2_3?

### Falsifiers

```
F1  system sha ed40b81d…, prefix 54 tokens                          pass
F2  adapter live, max |logit delta| = 15.500                         pass*
F3  weights differ from the final adapter: sha differs, and the
    seed-paired sample #1 shares 0 words of prefix with G's #1        pass
```

*F2 reads the same 15.500 as the final adapter. That is a bf16 coincidence at
one prompt position (step 0.125 at that magnitude), not shared weights — F3 is
what rules the latter out, and it is the check to run whenever F2 repeats a
number.

### Result

```
 i fin  raw w delp w span ch  anaph%  run  int% int pct@W agri  1p/1k
 1 CAP   2331    682    6130    36.4   16  41.8     100.0    2    0.0
 2 CAP   2180   1090    4904    31.5   12   1.8      10.5    0   22.2
 3 EOS    721    721     188     7.0    2  25.0      98.6    4    0.0
 4 EOS    837    837     118    13.4    3   1.0       7.2    1   35.5
 5 EOS   1166   1166      67    15.2    7   7.0      52.2    1   40.4
 6 CAP   2265    221    5725    93.3   27   9.7      65.2    0  128.3
 7 EOS    900    593    1504    10.7    3   0.0       7.2    0   46.5
 8 CAP   1286     32    3680    25.0    2   0.0      43.1    0   93.8
 9 CAP   1981    734    5647    27.1   11   1.0       8.3    0   57.9
10 CAP   2138    939    5951    46.8   24   4.8      34.8    0   51.1

arm             n  EOS CAP loop* med delp w  EOS w range  med an%  >cMAX  med int%  med pct  >p90  agri/1k  first-word reuse
variant F      10    5   5    5        258      200-405     24.5    7/10     19.4     87.5     5     1.22     70.0
G @ epoch 1    10    4   6    7        728      721-1166    26.0    9/10      3.3     38.9     2     1.11     51.9
G @ epoch 2    10    3   7    9        773      803-1362    28.8    8/10     11.0     70.1     4     0.41     41.3
base            4    4   0    0        750      652-819      0.0    0/4       1.7     16.5     0     5.98     32.7
```

Against the pre-registered prediction:

| prediction | result | |
|---|---|---|
| cap-loops 5–7/10 | 6/10 | met |
| de-looped median 400–700 (between F and G) | 728 | missed high — it is G's number, not halfway |
| falsifier for a 1-epoch run: caps ≥ 7/10 | 6/10 vs G's 7/10, Fisher p = 1.0 | not triggered, and not distinguishable |
| "the run to make": caps ≤ 3/10 and median ≥ 600 | 6/10 | **not met** |

1. **Sustain is bought by the first epoch.** De-looped median 728 vs 773 at
   epoch 2 (two-sided p = 0.36); EOS lengths 721–1166 vs 803–1362. Every
   chapter-length figure the eleventh run reported is already present at
   step 62.
2. **The capture is bought by the first epoch too.** Cap 6/10 vs 7/10, EOS
   4/10 vs 3/10, anaphora 26.0 vs 28.8, 9/10 vs 8/10 past the corpus max.
   Nothing here moves between epoch 1 and 2. (#8 caps after 32 novel words
   — the shortest pre-capture prefix any G arm has produced — and #6 after
   221, so the epoch-1 checkpoint has the *wider* spread, if anything.)
3. **First-word reuse 51.9, between F (70.0) and G (41.3).** The one axis
   that appears to move with exposure moves in the direction of *less*
   ladder with more training, not more. `[post-hoc instrument, third
   comparison; a pre-registered replication is owed before it carries
   weight]`
4. **Register, n.s.:** interiority median 3.3, percentile 38.9 — below the
   corpus median (7.0) and the lowest of any adapter arm; agri 1.11/1k.

### Prose

**#5 (1166 w, EOS)**, ending — the exit-A signature, with the seventh run's
"Goodnight" cadence:

> 'You're not going anywhere. I'm going to get the police.' 'Oh, you're going
> to get the police. I'm not afraid of the police. I'm not afraid of anyone.
> I'm going to leave. I'm going to go home now. Goodnight. Goodnight, doctor.
> I'm not afraid of you. I'm not afraid of anyone. Goodnight. Goodnight.
> Goodnight.'

**#3 (721 w, EOS)**, ending — a single sentence repeated four times, then stop:

> She smiled as she thought about how much she would miss him when he was old
> enough to leave home. She thought about how much she would miss him when he
> was old enough to leave home. The mother smiled as she thought about how
> much she would miss him when he was old enough to leave home. The mother
> smiled as she thought about how much she would miss him when he was old
> enough to leave home.

### Verdict

**ESTABLISHED: the epoch count is not the lever.** Epoch 1 and epoch 2 of
variant G are indistinguishable on capture rate, pre-capture length, EOS
length, and two-word anaphora (every p ≥ 0.36). The eleventh run's candidate
(a) — "1 epoch on v2_3" — would reproduce a checkpoint that is already on
disk, and is withdrawn. The last training run stays unspent.

What this adds to the picture of runs 11–12: the chapter-branch upweight
buys its sustain gain within the first 62 steps and adds nothing after; the
capture is present from the first checkpoint sampled and no amount of the
same data changes it. Together with the eighth/ninth runs (no repeat pressure
in the weights) and the tenth/twelfth (sampler constraints are absorbed at
the next level), the series' evidence now points one way: **the capture is a
property of the base model's long-context behaviour under this system
prompt that the adapter can only delay** — G delays it by ~500 words — and
neither more of the same data nor a local sampler rule removes it.

### Next step

1. **The last training run: hold it** until there is a candidate that
   targets something other than the branch mix or exposure. The one
   unexplored training-side variable that runs 11–13 leave open is the
   candidate (b) of the twelfth run — the brief branch's contribution to
   exit A (its short-stop schedule firing inside a ladder). It is a real
   question, but it targets the *stop*, and the caps are the larger failure.
2. **Before any further spend, two free measurements:** (i) pre-register and
   replicate the first-word-reuse instrument on the existing arms (F, G,
   G+guard, ckpt-62, base, corpus) with the boundary rules written down, so
   the F → G → ckpt-62 ordering (70 → 52 → 41) either survives or dies
   without costing a generation; (ii) the base model at the chapter prompt at
   cap 2560 with the *same* seeds as G — the four base controls on record
   were 652–819 words, all EOS, and they are the only arm that never
   captures; whether that holds at n=10 is the question every "the capture is
   base's" sentence above rests on. **(ii) costs 10 generations and is the
   best use of the next 10.**
3. Engine note from the twelfth run stands: nothing here has been sampled at
   the engine's own settings.

Budget this series: **1/2 training runs, 44/60 generations**, ~132
GPU-minutes this session. Push still blocked (no credential on the pod).

**Correction note appended to the thirteenth run (2026-09-22, same session).**
The verdict paragraph's sentence *"the capture is a property of the base
model's long-context behaviour under this system prompt that the adapter can
only delay"* is **retracted** by the fourteenth run below: base at the chapter
prompt, n = 10, same seeds, caps 0/10 and repeats nothing. The adapter does
not delay a capture the base would suffer anyway; it introduces one the base
never enters. The eighth run's formulation stands unchanged — a model-agnostic
capture *once seeded*, which the adapter seeds and the base does not.

---

## 2026-09-22 (fourteenth run) — base at the chapter prompt, n = 10, variant G's seeds. 10/10 EOS, 0 repeats, 466–995 words. The capture is the adapter's; the base never enters it

**Arm:** the base model, produced by loading variant G and scaling all 160
LoRA layers to 0 (`scripts/gen_opening_guard.py --scale 0 --window 0`, with
the sixth run's falsifier: scale 0 must be bit-identical to
`disable_adapter()`). **Ten chapter samples (series total 54/60), 0 training
runs, ~10 GPU-minutes.** Same seeds as the eleventh–thirteenth runs, cap 2560,
sampler byte-identical. Data `eval/gen_base_n10_cap2560.json`; run log
`logs/gen_base_n10_run.log`; scorer `logs/score_base_n10_fourteenth.log`;
first-word instrument `logs/first_word_reuse_all_arms.log`.

### Falsifiers

```
F1  system sha ed40b81d…, prefix 54 tokens                              pass
F2  scale 0 == disable_adapter(): max |logit delta| = 0.000e+00          pass
```

### Result

```
 i fin  raw w delp w span ch  anaph%  run  int% int pct@W agri  1p/1k
 1 EOS    780    780      25     0.0    1   0.0       4.0    3    0.0
 2 EOS    514    514      18     0.0    1   3.0      22.8    2    0.0
 3 EOS    508    508      23     0.0    1   3.1      23.6    1    0.0
 4 EOS    466    466      15     0.0    1   9.5      66.3    3    0.0
 5 EOS    509    509      16     0.0    1   3.0      22.8    1    0.0
 6 EOS    995    995      19     0.0    1   1.5       9.8    2    0.0
 7 EOS    547    547      17     0.0    1   0.0       8.0    6    0.0
 8 EOS    546    546      21     0.0    1   2.8      22.5    2    0.0
 9 EOS    816    816      27     0.0    1   2.3      17.4    3    1.2
10 EOS    901    901      24     0.0    1   1.6      11.6    2    0.0

arm             n  EOS CAP loop* med delp w  EOS w range  med an%  >cMAX  med int%  med pct  >p90  agri/1k  first-word reuse (>corpus p90)
base n=10      10   10   0    0        546      466-995      0.0    0/10      2.5     19.9     0     3.77     37.6  (5/10)
base n=4 (Aug)  4    4   0    0        750      652-819      0.0    0/4       1.7     16.5     0     5.98     32.7  (2/4)
variant F      10    5   5    5        258      200-405     24.5    7/10     19.4     87.5     5     1.22     70.0  (10/10)
variant G      10    3   7    9        773      803-1362    28.8    8/10     11.0     70.1     4     0.41     41.3  (10/10)
G @ epoch 1    10    4   6    7        728      721-1166    26.0    9/10      3.3     38.9     2     1.11     51.9  (10/10)
G + guard      10    6   4    7        848      157-1423     4.4*   3/10      8.1     55.6     1     0.94     42.2  (9/10)
corpus chapters (n=138)                893-1500 (med 1420)   1.1 (max 8.9)      7.0                            18.3  (p90 29.2)
```

1. **Base never captures.** 0/10 cap, longest repeated substring 15–27
   characters, two-word anaphora 0.0 in every sample, same-opening run 1 in
   every sample. Pooled with the August controls, 0/14. Against G's 7/10 cap
   Fisher p = 0.0031 (pooled 0/14 vs 7/10, p = 0.00035); against F's 5/10,
   p = 0.033. This is the contrast the whole series has been assuming and
   had measured only at n = 4.
2. **Base is short.** Median 546 words (August n = 4: 750), 466–995, 0/10 at
   the corpus chapter median of 1420. So the sustain deficit *is* real for
   the base too — G's EOS stops (803–1362) and G+guard's (median 874) are
   longer than base's — but base's shortfall is a clean stop at 0.4x target,
   the adapters' is a capture at 0.5–0.9x.
3. **Register.** Base: interiority median 2.5 (percentile 19.9), agri
   3.77/1k, first-person 0 in 9 of 10 — the prompt's "objective physical
   realism … labor with precision" obeyed; sample #6 ends: *"He adjusted his
   pack and set off once more, the path ahead uncertain but the purpose
   clear."* Every adapter arm sits above base on interiority and below it on
   labour vocabulary, as the fourth/fifth runs found.
4. **First-word reuse, the post-hoc instrument, with base at n = 10.**
   Base 37.6, itself above the corpus p90 in 5/10 — so the base's own
   sentence-opening habit is well above the corpus's, and *variant G (41.3)
   is at base level on this measure while F (70.0) is not.* Ordering
   F 70 > ckpt-62 52 > G 41 ≈ base 38 > corpus 18. Still post hoc; now
   written down in `scripts/first_word_reuse.py` with its boundary rules, so
   the next arm scores it pre-registered.

### Verdict

**ESTABLISHED, and a retraction.** The capture is introduced by the adapter,
not delayed by it: base 0/14, every adapter arm ≥ 4/10. The thirteenth run's
"base's long-context behaviour that the adapter can only delay" is withdrawn
in place above. What survives from runs 8–13, restated with this control in
hand:

- The base at this prompt writes 466–995 words and stops cleanly. It does not
  ladder and does not cycle.
- Fine-tuning on the chapter branch (F, G) buys 300–500 more words of chapter
  before a capture the base never enters; more chapter data moves the
  capture point (258 → 773), not the capture rate (5/10 → 7/10, n.s.).
- No sampler rule tried (n-gram, opening) removes the capture; each is
  absorbed at the next level.
- No exact-repeat probe finds the seed in the weights (runs 8–9), and the
  epoch count does not change it (run 13).

So the thing the adapter adds that base lacks is not "repeat pressure" and
not "a stop schedule"; it is whatever makes a 700-word prefix of the
adapter's own prose a ladder-entry state when a 700-word prefix of base's is
not. The eighth run's "upstream of the first rung" is still the right
location, and the free probe it named (opening-distribution entropy at
sentence boundaries, adapter vs base, on base's own clean text) is the
instrument that would see it. **That probe is the next free measurement, and
it costs no generations: teacher-force the 14 base controls through F, G and
base and compare next-token entropy at every sentence boundary.**

### Next step

1. **Free:** the boundary-entropy probe above, on `gen_base_control.json` +
   `eval/gen_base_n10_cap2560.json` (14 clean chapter-length texts), three
   arms (base, F, G). Prediction to register before running: the adapter's
   sentence-opening distribution at boundaries is lower-entropy than base's on
   the same prefix, by more on prefixes ≥ 500 words than < 500, and G's is
   between F's and base's.
2. **The last training run is held** until (1) says what the adapter is
   doing at boundaries; if it shows concentration, the corpus question is
   *which* entries carry the concentrated openings (the sixth run's "He/She
   + verb" chapter shapes are the candidate), and that is a corpus edit worth
   the run.
3. **Six generations remain in the series** (54/60). Reserve them for the
   brief-branch spot-check of whatever the last training run produces.

Budget this series: **1/2 training runs, 54/60 generations**, ~145
GPU-minutes this session. Push still blocked on the pod; seven commits since
`3a57ed7` await the owner.

---

## 2026-09-22 (fifteenth run) — the boundary-entropy probe. Prediction falsified in the opposite direction: the adapters do not concentrate sentence-opening mass, they flatten it, by +1.1 to +1.4 nats over base on every text type, with a +0.5-nat boundary-specific excess

**Free measurement, 0 generations, ~8 GPU-minutes.** Prediction pre-registered
in `logs/prediction_boundary_entropy.txt` (`a78ef8b`) before the run. Scripts
`scripts/boundary_entropy.py`, `scripts/analyze_boundary_entropy.py`. Data
`eval/boundary_entropy.json` (14 base texts), `eval/boundary_entropy_corpus40.json`
(40 corpus chapters, fixed-seed sample), `eval/boundary_entropy_nb.json` (the 14
base texts with a paired mid-sentence control). Analyses under `logs/analyze_*`.

### What was measured

Teacher-forcing under the chapter prompt, three arms on the same tokens — base
(`disable_adapter()`), F, G loaded as two named adapters on one model — at
every sentence boundary (token starting a new sentence after `. ! ? " ”`):
Shannon entropy of the next-token distribution, top-1 probability, and log-prob
of the opening actually written. Paired per boundary by construction. Two text
sets, because the ninth run showed base-relative measures swing with whose
manifold the text is on: base's own 14 clean chapter generations (adapter
off-manifold) and 40 corpus chapter targets (adapter on-manifold, base
off-manifold). If a sign holds on both, it is not the manifold.

### Result

```
                                  base      F      G    F-base  G-base   share F<base  share G<base
14 base texts, 483 boundaries
  entropy H (nats)               0.916  2.204  2.428   +1.261  +1.429       0.01          0.00
  top-1 prob                     0.699  0.446  0.416   -0.215  -0.258       0.90          0.92
  logp(true opening)            -0.413 -1.149 -1.289   -0.541  -0.671       0.92          0.94
40 corpus chapters, 2802 boundaries
  entropy H (nats)               2.368  3.641  3.520   +1.151  +1.046       0.00          0.01
  top-1 prob                     0.392  0.250  0.249   -0.118  -0.123       0.95          0.93
  logp(true opening)            -3.027 -2.891 -2.813   +0.021  +0.084       0.49          0.46

by prefix length (corpus, F-base / G-base):  0-250 +1.48/+1.25   250-500 +1.20/+1.07   500-750 +1.18/+1.05   750+ +1.05/+0.96

mid-sentence control (14 base texts, 483 word-initial non-boundary positions, paired):
  arm    boundary H   non-boundary H   boundary excess
  base      0.916          0.653           +0.263
  F         2.204          1.582           +0.621
  G         2.428          1.764           +0.664
  F-base   +1.261         +0.781           boundary-specific +0.480
  G-base   +1.429         +0.966           boundary-specific +0.463
```

Against the prediction:

| prediction | result | |
|---|---|---|
| adapter entropy < base at the median boundary | +1.26 / +1.43 nats, share below base 0.01 / 0.00 | **falsified, opposite sign** |
| gap larger at ≥ 500 words than below | gap *shrinks* with prefix length (+1.5 → +1.0) | falsified |
| G between F and base | G is above F on base text (+0.15), below F on corpus text (−0.11) | not supported |
| falsifier: median paired delta ≥ 0 → concentration is not the mechanism | triggered on both text sets | **the concentration reading is dead** |

1. **The adapters flatten the next-token distribution everywhere, and more
   at sentence boundaries.** Mid-sentence +0.8 / +1.0 nats over base; at
   boundaries +1.3 / +1.4, a boundary-specific excess of ~0.5 nats in both
   adapters. Top-1 mass at a boundary 0.70 → 0.42 on base text, 0.39 → 0.25
   on corpus text.
2. **It is not the manifold.** The sign and size are the same on the
   adapter's own training targets as on base's generations. The only
   quantity that flips is logp(true opening): on base text the adapters are
   worse at predicting base's openings (−0.5 / −0.7 nats, expected), and on
   corpus text they are *no better than base* at predicting the corpus's own
   openings (+0.02 / +0.08, share 0.49 / 0.46). Six passes over each chapter
   did not teach G the corpus's sentence openings; they spread the mass.
3. **The eighth run's F2 "boost on control openings" (+2.8 to +9.5 nats) is
   this, seen from one side.** A flatter distribution raises the log-prob of
   every low-probability continuation, repeats and controls alike, which is
   exactly what "the elevation is not repeat-specific" said. There is no
   concentration to find because there is none.
4. **Consequence for the ladder, stated as a hypothesis** `[unverified]`: at
   T = 0.7 the base's boundary distribution has ~0.55 nats and the adapter's
   ~1.25 (`logs/match_temperature.log`); the sampler is drawing sentence
   openings from a distribution more than twice as spread. Runs 8–9 found no
   repeat mass on clean prefixes, so the flattening does not seed a rung
   directly; the candidate reading is that it walks the text off any model's
   manifold faster, into the region where the eighth run's self-conditioning
   takes over. That reading has a cheap test, and the sixteenth run below is
   it.

### Verdict

**ESTABLISHED (falsification).** The adapter's contribution at sentence
boundaries is a +1.1–1.4 nat entropy increase relative to base, present on
both text manifolds, larger at boundaries than mid-sentence, and not
accompanied by any better fit to the corpus's openings. The "concentration
upstream of the first rung" hypothesis carried since the eighth run is
withdrawn.

### Next step

The entropy-matched temperature test, pre-registered in
`logs/prediction_entropy_matched_T.txt` and run as the sixteenth entry:
sample variant G at the T* where its boundary entropy on the base texts
equals base's at 0.7. `scripts/match_temperature.py` gives **T* = 0.4**
(G 0.484 nats at 0.40, 0.617 at 0.45; base 0.548 at 0.70).

---

## 2026-09-22 (sixteenth run) — variant G at the entropy-matched temperature T* = 0.4. 6/6 cap, cycles close after 62–204 words. The flattening is not the ladder's cause; it is what delays the ladder. The adapter's low-temperature path is a cycle

**Adapter:** variant G, unchanged. **Six chapter samples (series total 60/60 —
the generation budget is spent), 0 training runs, ~17 GPU-minutes.**
Prediction pre-registered in `logs/prediction_entropy_matched_T.txt`
(`b7a8c1b`); T* = 0.4 from `scripts/match_temperature.py` (`eval/match_temperature.json`),
committed before sampling. `scripts/gen_opening_guard.py --window 0
--temperature 0.4`, otherwise the recorded sampler (min_p 0.05, repetition
penalty 1.05, cap 2560). Seeds 20260827–32, paired with G's #1–6 at T = 0.7.
Data `eval/gen_v7_variantG_T04.json`; run log `logs/gen_variantG_T04_run.log`;
scorer `logs/score_G_T04_sixteenth.log`.

### Falsifiers

```
F1  system sha ed40b81d…, prefix 54 tokens                    pass
F2  adapter live, max |logit delta| = 15.500                   pass
F3  seed-paired with G @ 0.7 (same host, same weights)         by construction
```

### Result

```
 i fin  raw w delp w span ch  anaph%  run  int%  agri   novel words before the cycle   G @ 0.7, same seed
 1 CAP   2367    142    5952    33.3    4   0.0     0    143                            CAP, novel 2062 (ordinal ladder)
 2 CAP   2141    100    5123    16.7    2  28.6     0    100                            CAP, novel  885
 3 CAP   2240    159    4818   100.0   11   0.0     0    159                            CAP, novel  735
 4 CAP   1697     71    4066     0.0    1   0.0     0     71                            CAP, novel  593
 5 CAP   2132    204    5714    55.0   11   9.5     0    204                            CAP, novel  788
 6 CAP   1743     62    4554     0.0    1   0.0     0     62                            CAP, novel  604

arm                 n  EOS CAP  med novel w   novel range   med an%  first-word reuse
G @ T=0.4           6    0   6        121        62-204      25.0     68.8  (5/6 > corpus p90)
G @ T=0.7 (same 6)  6    0   6        762       593-2062     ~30      41   (median of the 10)
base @ T=0.7       10   10   0        546       466-995       0.0     37.6
```

Against the prediction:

| prediction | result | |
|---|---|---|
| if flattening is the mechanism: caps ≤ 2/6 | 6/6 | **falsified** |
| falsifier: caps ≥ 4/6 → flattening is not the mechanism | 6/6 | **triggered** |
| first-word reuse < 41 | 68.8 | falsified, and back at F's level |

1. **Matching the adapter's boundary entropy to base's makes the capture
   ~6x faster, not slower.** Paired by seed, every sample's cycle closes
   earlier at 0.4 than at 0.7 (62–204 vs 593–2062 words; 6/6, sign test
   p = 0.03). The cycles are short and verbatim from the outset: #6 is
   "He looked up again. … He looked down again. …" from word 62; #4 from
   word 71.
2. **So the +1.2 nats is protective.** The adapter's *mode* at a boundary —
   what it would say greedily — is already the rung; the spread the
   fifteenth run measured is the sampling noise that keeps the text off that
   path for a few hundred words. Reduce the noise and the path is taken at
   once. This is why runs 8–9 found no repeat mass on clean prefixes: the
   rung is not a high-probability *repeat*, it is the highest-probability
   *shape* ("He looked …", "She thought that …", "I'm helping you see
   …") — the seventh/tenth runs' fuzzy ladder — and the exact-repeat probe
   measured the wrong thing.
3. **Base at T = 0.7 never takes such a path** (0/14), with a boundary
   distribution that is *sharper* than the adapter's at 0.4 (0.55 vs 0.48
   nats median). Sharpness is not the variable; what the sharp distribution
   is centred on is.
4. **Register at T = 0.4:** interiority 0.0, agri 0.00, first-word reuse
   68.8 (F's level, up from G's 41). The prose collapses into the ladder
   before any register is established.

### Verdict

**WORSE (6/6 vs the same seeds' 6/6 cap at 0.7, but at one-sixth the
length), and ESTABLISHED as mechanism:** the adapter's most probable
continuation at a sentence boundary is a rung. Temperature is not a lever
downward; whether it is one *upward* (0.8–0.9: more noise, later capture,
at the cost of coherence) is untested and is the obvious cheap question the
budget no longer covers.

Across the sixteen runs the picture is now closed enough to state:

- Base: 466–995 words, clean EOS, 0/14 captures, obeys the register prompt.
- Any adapter from this corpus at the chapter prompt: a mode that is a
  ladder, reached within ~100 words greedily and within ~250 (F) to ~770
  (G) words at T = 0.7; nothing sampler-side removes it (n-gram, opening
  guard, temperature down), nothing exposure-side changes it (epoch 1 =
  epoch 2), and more chapter data only moves the entry point.
- The brief branch (33–96 words, 3/3 EOS, corpus band) is unaffected and
  remains the production path.

### Next step

1. **The last training run is the only budget left, and the evidence now
   says what it should test.** Not the branch mix, not exposure, not
   epochs: the *mode*. The corpus's chapter targets have a first-word reuse
   of 18.3 (p90 29.2) and the adapter's is 41–70 — the adapter has learned
   a sentence-shape prior the corpus does not have. Two candidates, one run:
   (a) **drop the brief branch entirely** (chapter-only, 138 x 3, same
   hyperparameters) — tests whether the 564 short, dialogue-heavy,
   "He said / She said"-shaped briefs are the source of the shape prior
   (brief targets: median 52 words, dialogue-dense; the ladders are built of
   exactly their sentence forms); (b) LoRA on the MLP projections as well
   as attention — off the table, it is a HARD REQ. **(a)** is the run.
   Prediction to register before it: chapter-only first-word reuse < 41
   and cap-loops < 7/10 at T = 0.7; brief-branch behaviour will regress (it
   is the branch being removed), so this is a diagnostic run, not a ship
   candidate, and the charter's brief spot-check is expected to fail.
2. **Free, before that run:** first-word reuse on the brief targets
   themselves (window 4 across the 564), to see whether the shape prior is
   visible in the data the adapter got. If the brief branch scores ≥ 40,
   (a) is well-motivated; if it scores near the chapter branch's 18, the
   prior is not in the data and (a) is weaker.
3. **Engine:** unchanged advice. Nothing here was sampled at the engine's
   settings; the engine's `repeat_penalty 1.1` and rolling temperature
   0.55–0.78 sit in a range this series has now measured at both ends (0.4:
   worse; 0.7: as recorded).

Budget this series: **1/2 training runs, 60/60 generations** — the
generation budget is spent. ~184 GPU-minutes this session. Push still
blocked on the pod.

**Addendum (same session, free): first-word reuse in the training data
itself** (`logs/first_word_reuse_brief_targets.log`, window 4, same rules).

```
set                          n   scorable  median   p90   share > chapter p90 (29.2)
brief targets              564      230     22.2   57.1        0.37
brief targets, blocks of 10 57       57     24.5   42.3        0.39
chapter targets            138      138     18.3   29.2        0.09
```

The brief branch's median is close to the chapter branch's, but its tail is
not: 37% of scorable briefs exceed the chapter branch's p90, and the brief p90
(57) is where F's generations sit (median 70) and above G's (41). The shape
prior is visible in the data, in the tail of the brief branch. Next step 1(a)
— chapter-only — is motivated; a gentler variant that keeps the production
branch, **dropping only the briefs above 29.2 first-word reuse (~85 of 230
scorable)**, is the version that would not regress the brief spot-check by
construction, and is the better candidate for the single remaining run.
