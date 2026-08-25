"""
Tests for the turn-scale scorer corrections (length-aware entropy floor +
short-turn interiority granularity dead zone).

Run:  python test_turn_scale_scorer.py   (or pytest)

Invariant being protected: when the texture dict carries NO length metadata
(n_tokens / n_sentences), scoring is bit-identical to the pre-patch scorer —
all chapter-scale benchmarks (verify_scorer.py) are unaffected.
"""

import math
import sys

sys.path.insert(0, ".")
from engine.drift import DriftScorer
from engine.texture import TextureAnalyzer
from engine.entropy import EntropyCalculator

scorer = DriftScorer()
ta = TextureAnalyzer()
ec = EntropyCalculator()

FAILURES = []


def check(name, cond, detail=""):
    status = "ok" if cond else "FAIL"
    print(f"  [{status}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


def legacy_score(texture, entropy=0.0):
    """The pre-patch algorithm, reproduced for equivalence checking."""
    drift = 0.0
    for key, target in scorer.ONE_SIDED.items():
        drift += max(0.0, texture.get(key, 0.0) - target) * scorer.WEIGHTS[key]
    for key, (low, high) in scorer.CORRIDORS.items():
        a = texture.get(key, 0.0)
        p = (low - a) if a < low else (a - high) if a > high else 0.0
        drift += p * scorer.WEIGHTS[key]
    drift += max(0.0, scorer.ENTROPY_FLOOR - entropy) * 0.3
    return round(drift, 4)


# --- 1. metadata-free dicts score exactly as before -------------------------
print("1. legacy equivalence (no length metadata -> bit-identical)")
CASES = [
    ({"action_pct": 29.5, "interiority_pct": 3.8, "neutral_pct": 66.7,
      "dialogue_density": 3.8, "figurative_density": 0.06,
      "sentence_rhythm": 11.24}, 7.988),          # Moberg ch1 (verify_scorer)
    ({"action_pct": 70.4, "interiority_pct": 9.3, "neutral_pct": 20.3,
      "dialogue_density": 1.9, "figurative_density": 0.51,
      "sentence_rhythm": 8.26}, 9.1),             # Engine burial scene
    ({"action_pct": 10.0, "interiority_pct": 30.0, "neutral_pct": 40.0,
      "dialogue_density": 30.0, "figurative_density": 3.0,
      "sentence_rhythm": 4.0}, 6.0),              # worst case
]
for i, (tex, ent) in enumerate(CASES):
    got = scorer.score(dict(tex), entropy=ent)["drift_score"]
    want = legacy_score(tex, ent)
    check(f"case {i}", got == want, f"got={got} want={want}")

# --- 2. long text: effective floor unchanged --------------------------------
print("2. long text keeps the 8.05 floor")
tex = dict(CASES[0][0], n_tokens=50000, n_sentences=400)
r = scorer.score(tex, entropy=7.988)
check("floor@50k", r["entropy_floor_effective"] == 8.05,
      f"eff={r['entropy_floor_effective']}")

# --- 3. short healthy passage: entropy penalty vanishes ---------------------
print("3. short healthy passage no longer pays the length tax")
S326 = ("The official read the decision aloud. The woman listened. When he "
        "finished she took the letter from him, folded it, put it back in the "
        "envelope, and put the envelope in her bag. She did not say anything. "
        "She stood up. She said thank you. She left.")
tex = ta.analyze(S326)
ent = ec.shannon_entropy(S326)
patched = scorer.score(dict(tex), entropy=ent)
stripped = {k: v for k, v in tex.items() if k not in ("n_tokens", "n_sentences")}
unpatched = scorer.score(stripped, entropy=ent)
check("entropy penalty = 0", patched["entropy_component"] == 0.0,
      f"component={patched['entropy_component']}")
check("score strictly improves", patched["drift_score"] < unpatched["drift_score"],
      f"{patched['drift_score']} < {unpatched['drift_score']}")

# --- 4. degenerate repetition still penalized -------------------------------
print("4. degenerate loop output still pays")
loop = "He walked to the barn and set the pail down. " * 30
tex = ta.analyze(loop)
ent = ec.shannon_entropy(loop)
cap = math.log2(tex["n_tokens"])
r = scorer.score(dict(tex), entropy=ent)
check("entropy well below cap", cap - ent > scorer.ENTROPY_LEN_MARGIN,
      f"delta={round(cap-ent,2)}")
check("penalty > 0", r["entropy_component"] > 0.0,
      f"component={r['entropy_component']}")

# --- 5. interiority granularity dead zone -----------------------------------
print("5. interiority dead zone under 9 sentences")
base = {"action_pct": 22.0, "neutral_pct": 70.0, "dialogue_density": 5.0,
        "figurative_density": 0.0, "sentence_rhythm": 10.0}
one_of_eight = dict(base, interiority_pct=12.5, n_sentences=8, n_tokens=400)
r = scorer.score(one_of_eight, entropy=8.1)
check("1/8 interior -> dev 0", r["interiority_pct_dev"] == 0.0,
      f"dev={r['interiority_pct_dev']}")
zero_of_eight = dict(base, interiority_pct=0.0, n_sentences=8, n_tokens=400)
r = scorer.score(zero_of_eight, entropy=8.1)
check("0/8 interior -> dev 0", r["interiority_pct_dev"] == 0.0,
      f"dev={r['interiority_pct_dev']}")
two_of_eight = dict(base, interiority_pct=25.0, n_sentences=8, n_tokens=400)
r = scorer.score(two_of_eight, entropy=8.1)
check("2/8 interior -> ceiling penalty", r["interiority_pct_dev"] == 13.0,
      f"dev={r['interiority_pct_dev']}")
chapter = dict(base, interiority_pct=3.0, n_sentences=100, n_tokens=2500)
r = scorer.score(chapter, entropy=8.1)
check("chapter-scale low side unchanged", r["interiority_pct_dev"] == 4.0,
      f"dev={r['interiority_pct_dev']}")

# --- 6. analyzer provides the metadata --------------------------------------
print("6. TextureAnalyzer emits length metadata")
tex = ta.analyze(S326)
check("n_tokens present", isinstance(tex.get("n_tokens"), int) and tex["n_tokens"] > 0)
check("n_sentences present", isinstance(tex.get("n_sentences"), int) and tex["n_sentences"] > 0)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("all checks passed")
