# Metric interconnections — correlation analysis on the 16-chapter benchmark

Source: analysis run by Stride, Aug 22–23 2026, on the corrected 16-chapter
*Emigrants* benchmark (original `lexicon.py` PHYSICAL_VERBS, patched
`_split_sentences()`). Recorded here because it lived only in Slack.

Note on provenance: corridors have since been recalibrated across 88 chapters
spanning three books (commit `cbe6ba8`). The correlations below are computed on
the 16-chapter single-book sample and have not been recomputed on the wider set.
The relationships are likely to hold; the exact coefficients may move.

---

## 1. Action subcategory split

`action_pct` as originally measured conflates five functionally different
behaviours. Splitting them:

| category | what it is |
|---|---|
| Labor / Struggle | sustained physical work |
| Displacement | body carrying suppressed interiority |
| Gestural / Social | nods, glances, handshakes |
| Routine / Maintenance | incidental household motion |
| Involuntary / Reflexive | body acting against will |

**Measured across 16 chapters:**

```
action_pct_clean  (Labor + Displacement + Involuntary)
  mean 14.3%   p10 8.4%   p90 18.5%

action_pct_raw    (all physical verbs, original method)
  mean 24.3%   p10 15.1%  p90 28.6%
```

Roughly 10pp of inflation comes from gestural and routine sentences.

**Chapter 10 (AT HOME AND AWAY) is the clean illustration:** raw action 37.3%,
clean action 8.4%. Almost entirely gestural. A raw-action corridor passes this
chapter; a clean-action corridor catches it.

Per-chapter breakdown:

```
Ch  Title                          labor  displ  gest  rout  invol   int   neu  act_raw  act_cln
 1  KING IN HIS STONE KINGDOM        5.9   12.5   2.8   4.8    1.3  10.3  62.4    27.4     19.7
 2  KARL OSKAR AND KRISTINA          3.9   10.9   2.5   3.7    1.2   9.0  68.8    22.2     16.0
 3  "SUITABLE CHASTISEMENT"          4.6   11.4   5.3   5.5    0.6  10.8  61.8    27.4     16.6
 4  THE AMERICA CHEST                5.1   11.2   4.1   3.6    0.5  10.9  64.6    24.5     16.8
 5  ONE EMIGRANT PAYS NO FARE        5.9    9.3   5.3   4.6    0.6  10.9  63.6    25.6     15.7
 6  THE CHARLOTTA OF KARLSHAMN       3.1   10.4   6.2   4.3    1.0  10.1  64.9    25.0     14.5
 7  A CARGO OF DREAMS                1.4    5.5   2.0   3.5    0.0   8.9  78.7    12.4      6.9
 8  HAPPENINGS ON BOARD THE SHIP     5.9   10.1   4.0   5.8    2.1   9.6  62.3    28.1     18.2
 9  A BUSHEL OF EARTH FROM SWEDEN    6.0    6.6   4.8   5.4    0.0  10.2  67.1    22.8     12.6
10  AT HOME AND AWAY                 2.4    6.0  19.3   9.6    0.0   9.6  53.0    37.3      8.4
11  STORIES ON THE AFTERDECK         4.2    8.9   5.8   6.3    0.5  12.6  61.8    25.7     13.6
12  IT WAS CALLED SHIP-SICKNESS      1.0    7.8   2.1   3.1    1.0  15.1  69.8    15.1      9.9
13  STORY TOLD AT THE MAIN HATCH     6.6    6.6   2.4   4.8    0.0   6.6  72.9    20.5     13.3
14  PEASANTS AT SEA                  4.7    9.0   1.3   3.8    0.0   9.0  72.2    18.8     13.7
15  A LONG NIGHT                     5.9   11.1   4.9   5.2    1.5   9.6  61.7    28.6     18.5
16  SAILING TOWARD MIDSUMMER         5.6    9.2   7.2   6.0    0.0  10.0  61.8    28.1     14.9
```

**Involuntary is rare but near-universal:** mean 0.64%, p90 1.5%, non-zero in
most chapters. Proposed as a one-sided floor signal rather than a corridor — 0%
involuntary sustained across several scenes is a drift tell.

Not currently in `telemetry_logger.py`. Would need adding before any of this can
be used at runtime.

---

## 2. Correlation matrix

Strongest relationships, |r| >= 0.5:

```
act_raw  vs neutral    r = -0.954   near-perfect inverse; gestural/routine inflating act_raw
gest     vs rout       r = +0.890   both noise categories, move together
act_cln  vs displ      r = +0.895   displacement dominates meaningful action
gest     vs neu        r = -0.757   gestural eats neutral — the inflation mechanism
rout     vs neu        r = -0.775   routine eats neutral — same mechanism
displ    vs entropy    r = +0.790   displacement correlates with lexical richness
act_cln  vs entropy    r = +0.682   meaningful action = richer vocabulary
displ    vs invol      r = +0.651   displacement and involuntary move together
act_cln  vs invol      r = +0.632   involuntary tracks meaningful action
dial     vs entropy    r = +0.545   dialogue = lexical complexity, not simplicity
dial     vs fig        r = +0.529   dialogue chapters carry more figurative language
labor    vs int        r = -0.420   inverse — see below
```

### The displacement + involuntary + entropy cluster

Chapters high in displacement are also high in involuntary, entropy, and
dialogue. That co-occurrence is the empirical signature of a narratively loaded
chapter — not action versus dialogue, but displacement (body carrying suppressed
interiority) arriving together with involuntary motion and lexical richness.

**Implication for any pressure signal:** track the cluster, not `action_pct` or
`dialogue_density` individually.

### labor vs interiority, r = -0.420

The chapters with the most sustained labor carry the least named interiority.
When Karl Oskar is working hardest he is processing least. That is not absence
of feeling — it is the baseline holding.

This is the measured form of the load-bearing-consistency reading below.

---

## 3. Three predictions from computational stylometrics, tested

**Dialogue density vs lexical diversity, predicted inverse.**
Result r = +0.545 — wrong direction. Probably true at sentence level (repetitive
rustic speech does reduce local TTR) but at chapter level the high-dialogue
chapters are also the most narratively complex: multiple characters, multiple
registers, theological debate alongside domestic argument. Chapter-level signal
swamps the sentence-level effect. Testing it properly needs sentence-level TTR.

**Action vs sensory vocabulary (land tactile, ocean olfactory).**
Result r = +0.324 — direction weakly confirmed, but the land/ocean split does not
appear in `figurative_density` at all (both 0.065 mean). The sensory shift is
real in the text; `figurative_density` is the wrong instrument. Would need a
dedicated smell/taste/touch lexicon.

**Action vs dialogue, predicted inverse (crisis in silence).**
Result r = +0.543 — wrong direction at chapter level. Ch13 (STORY TOLD AT THE
MAIN HATCH) matches the prediction cleanly at 0.0% dialogue with moderate action,
but most chapters mix both. The pattern is real at scene level; the chapter-level
instrument is too coarse to see it.

---

## 4. Intra-chapter structure

Each chapter split into opening/middle/ending thirds:

```
Segment    act    int    neu   dial
Opening   24.2    8.2   67.6   11.1
Middle    20.2    8.5   71.3   10.8
Ending    20.6   10.6   68.8    9.7
```

Consistent across all 16:

- **Opening action is highest.** Moberg front-loads physical grounding, then the
  chapter opens up. This is the opposite of a climax-in-middle template.
- **Interiority rises toward endings**, 8.2% → 10.6%. The one consistent
  structural signal.
- **Dialogue drops slightly at endings**, 11.1% → 9.7%.

The predicted universal template (documentary opening → action climax →
stasis ending) does not hold at aggregate level. It is genre-dependent within
the book: documentary chapters (Ch7, Ch13) follow it cleanly, crisis chapters
(Ch15, the hemorrhage) invert it with action building toward the ending.

Ch13 is the extreme case:

```
Opening: action 40.0%  dialogue 0.0%  neutral 54.5%
Middle:  action 27.3%  dialogue 0.0%  neutral 70.9%
Ending:  action  5.4%  dialogue 0.0%  neutral 87.5%   <- maximum stasis
```

### The interiority rise is a closing register, not generic reflection

The +2.4pp is specifically Moberg's closing liturgical register — rhythmic
phrase-pairs, alliterative refrains, Old Testament cadence. *"So ended the day
when Robert Nilsson tried to take his first steps on the road to America."*
*"Such is the lot of home-staying farmhands."*

`interiority_pct` catches these correctly, because they contain interiority verbs
("he knew", "she felt"), but cannot distinguish them from ordinary mid-chapter
interior narration. The closing register has a distinct joint signature:
interiority up, dialogue near zero, action down, neutral flat or up, sentence
rhythm slowing.

Flagged as a possible future detector — narrow, high-signal, distinctly Moberg.
Not worth building until generation can produce the register at all.

---

## 5. Reframe: load-bearing consistency, not emotional oscillation

Moberg's emotional register does not oscillate. It is load-bearing consistency.
Karl Oskar's baseline — provide, endure, act — does not change under pressure.
Surface metrics fluctuate with harder seasons and worse harvests; the behavioural
response stays the same. He gets up and works.

So a pressure signal is not measuring "is he close to breaking". It is measuring
**the gap between surface pressure and behavioural response**. Wide and sustained
gap — high external pressure, unchanged behavioural output — is Moberg working
correctly. The gap closing, pressure producing proportional behavioural change,
is a different kind of novel.

`involuntary_pct` is the most sensitive instrument for this, because it is the
only place the body breaks from baseline without the character choosing to. Those
moments are rare precisely because the baseline is strong. A late-arc spike is
not the character breaking — it is the load briefly exceeding what the baseline
can absorb.

Practical consequence: the trigger condition is not "pressure exceeded threshold"
but "the gap between what is happening and how he responds has become
narratively incredible". That requires tracking behavioural consistency across
scenes, not pressure level at a scene.

A middle layer of emotional themes (hope, belonging, obligation, resentment) was
considered and rejected as unnecessary here. The spectra are already encoded in
how the bottom-layer metrics relate across scenes. What is missing is the
interaction model over time with relative normalisation — not a separate
classifier. Moberg is not Dostoevsky.

---

## 6. Telemetry gaps this analysis exposes

Current columns cannot distinguish story states with similar metric profiles:

- High `dialogue_density` could be debt negotiation, family argument about
  emigrating, storytelling on the afterdeck, or a quiet conversation about the
  children. Identical in telemetry, very different narratively.
- High `action_pct` could be harvest labour, a fight, or frantic packing.
- High `interiority_pct` could be grief, planning, or nostalgia.

Candidate additions, none yet built:

- `dialogue_reciprocity` — is speech answered or absorbed?
- `gap_content_type` — what follows dialogue: action, interiority, or neutral?
- `action_subcategory` — the five-way split above; benchmarked, not in telemetry
- `dialogue_x_interiority` — both elevated simultaneously reads differently from
  either alone

With these, similar surface readings could produce different outcomes: high
dialogue + high reciprocity + action gap is a conflict scene; high dialogue + low
reciprocity + interiority gap is a character processing pressure alone.

---

## 7. Relative normalisation

Any pressure function normalising against fixed corridor ceilings uses the same
denominators every run, so the curve shape can come out similar across arcs
regardless of content.

Normalising against the arc's own rolling mean instead makes the signal measure
deviation from *this story's* established rhythm:

```python
# instead of
action_signal = texture['action_pct'] / 27.0          # fixed ceiling

# use
recent = mean(last 5 scenes' action_pct from telemetry)
action_signal = (texture['action_pct'] - recent) / max(recent, 1.0)
```

This matters more given the cluster finding above — the displacement/involuntary/
entropy signature is only meaningful against the arc's own baseline, not a
universal anchor.
