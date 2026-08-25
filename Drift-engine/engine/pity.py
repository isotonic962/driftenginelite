# Pity system: guaranteed-release regulator for prolonged flatness.
#
# WHY THIS EXISTS: every mechanism in the engine regulates drift DOWNWARD
# (corridor penalties, downweight_expressive, the old censor stack before
# it was removed). Nothing guarantees an upward correction when output is
# clean-but-flat for many consecutive turns. RollingBaseline's
# "allow_texture" is memoryless -- it reads 3 rows and nudges temperature
# once, with no memory of HOW LONG flatness has persisted, no escalation,
# and no guarantee. The pity system adds that missing side: a streak
# counter over "starved" turns, a soft ramp (gacha soft-pity: parameter
# pressure grows with the streak), and a hard release (gacha hard-pity:
# a guaranteed one-scene prompt intervention) followed by a refractory
# period so the release cannot oscillate.
#
# TRIGGER DESIGN -- relative values with absolute rails (decided in
# design discussion, Aug 2026):
#   * ABSOLUTE RAILS: a turn is "starved" per-channel against the Moberg
#     corridor FLOORS from DriftScorer.CORRIDORS (single source of truth,
#     imported -- do not duplicate numbers here). Floors are one-sided on
#     purpose: pity only ever responds to deficit. Excess (above corridor
#     ceiling) is RollingBaseline's job and BREAKS the streak.
#   * RELATIVE TRIGGER: variance collapse over the recent telemetry
#     window. If the last `window` drift scores are near-identical
#     (pstdev < monotony_eps) the engine is in a monotone lock, and even
#     a mildly starved turn (>= 1 channel) counts as flat. This is the
#     self-relative signal: it survives model swaps and corridor
#     recalibration because it measures "nothing is moving", not a level.
#     NOTE: z-scores against the window are deliberately NOT used -- with
#     a flat engine the window variance is ~0 and a z-score is undefined.
#     Variance collapse inverts that failure mode into the signal itself.
#
# BASELINE CONTAMINATION: hard-release turns are tagged pity_fired=1 in
# engine_logs, and RollingBaseline excludes them from its window --
# otherwise a successful release spikes interiority, the baseline reads
# the spike, slams downweight_expressive, flatness returns, pity builds
# again: a limit cycle. Release turns are measured and logged (we want
# to know if the release WORKED) but never steer the nudge loop.
#
# PERSISTENCE: SceneCounter pattern -- one-row state table in
# moberg_telemetry.db, survives kernel restarts. To wipe pity state
# without wiping telemetry: PityController(...).reset().
#
# CADENCE (defaults soft_start=3, hard_at=6, refractory=2):
#   turns 1-2 flat   -> stage "building", params untouched
#   turns 3-5 flat   -> stage "soft", temperature ramps base -> soft_temp_cap,
#                       repeat_penalty ramps toward 1.02
#   6th flat turn    -> next generation is stage "hard": release directive
#                       injected into the system prompt, temperature 0.92,
#                       repeat_penalty 1.01, counter reset, refractory set
#   refractory turns -> stage "refractory", no accumulation, params pass
#                       through (covers the release turn itself + 1 more)
#
# The hard-release directive is composed from WHICH channels were starved
# (rhythm / figurative / dialogue / interiority), so the release asks for
# exactly the texture that is missing, one scene only, inside the anchor's
# register -- it never licenses named emotion.

import sqlite3
from statistics import pstdev

from .drift import DriftScorer


class PityController:
    """
    Scene/turn-level flatness streak tracker with soft ramp and
    guaranteed hard release.

    Two hooks, called by DriftEngine.process():
        pre_turn(params)  -- BEFORE generation. Returns (params, injection,
                             meta). May modulate generation parameters
                             (soft) or inject a release directive (hard).
        observe(texture)  -- AFTER measurement. Updates the streak counter
                             from this turn's texture. Returns meta for
                             telemetry.
    """

    # Channels checked for starvation, with floors taken from the
    # DriftScorer corridors. figurative_density has no corridor floor
    # (scorer is one-sided on it), but a whole scene at exactly 0.0 is
    # a flatness tell even though it is never penalized -- so pity
    # treats 0.0 as starved. action_pct is deliberately excluded:
    # the observed flat failure mode is action-dense prose, and low
    # action already routes to allow_texture via RollingBaseline.
    STARVE_CHANNELS = ("figurative_density", "interiority_pct",
                       "sentence_rhythm", "dialogue_density")

    def __init__(
        self,
        db_path="moberg_telemetry.db",
        soft_start=3,
        hard_at=6,
        refractory=2,
        window=5,
        flat_channels_min=3,
        monotony_eps=0.15,
        soft_temp_cap=0.88,
        release_temp=0.92,
        release_repeat_penalty=1.01,
    ):
        self.db_path = db_path
        self.soft_start = soft_start
        self.hard_at = hard_at
        self.refractory = refractory
        self.window = window
        self.flat_channels_min = flat_channels_min
        self.monotony_eps = monotony_eps
        self.soft_temp_cap = soft_temp_cap
        self.release_temp = release_temp
        self.release_repeat_penalty = release_repeat_penalty

        corridors = DriftScorer.CORRIDORS
        self.floors = {
            "interiority_pct": corridors["interiority_pct"][0],
            "sentence_rhythm": corridors["sentence_rhythm"][0],
            "dialogue_density": corridors["dialogue_density"][0],
        }
        self.ceilings = {
            "interiority_pct": corridors["interiority_pct"][1],
            "dialogue_density": corridors["dialogue_density"][1],
            "figurative_density": DriftScorer.ONE_SIDED["figurative_density"],
        }

        self._init_db()

    # ------------------------------------------------------------------
    # persistence (SceneCounter pattern: one row, id=1, same db)
    # ------------------------------------------------------------------

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS pity_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                counter INTEGER NOT NULL DEFAULT 0,
                refractory INTEGER NOT NULL DEFAULT 0,
                starved TEXT NOT NULL DEFAULT ''
            )
        """)
        c.execute(
            "INSERT OR IGNORE INTO pity_state (id, counter, refractory, starved) "
            "VALUES (1, 0, 0, '')"
        )
        conn.commit()
        conn.close()

    def _get_state(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT counter, refractory, starved FROM pity_state WHERE id = 1")
        row = c.fetchone()
        conn.close()
        if not row:
            return 0, 0, []
        counter, refractory, starved = row
        starved_list = [s for s in starved.split(",") if s]
        return counter, refractory, starved_list

    def _set_state(self, counter, refractory, starved):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            "UPDATE pity_state SET counter = ?, refractory = ?, starved = ? WHERE id = 1",
            (counter, refractory, ",".join(starved)),
        )
        conn.commit()
        conn.close()

    def reset(self):
        """Wipe pity state. Deliberate use only -- e.g. new arc / SCENE.reset()."""
        self._set_state(0, 0, [])

    # ------------------------------------------------------------------
    # flatness measurement
    # ------------------------------------------------------------------

    def starved_channels(self, texture):
        """Channels below the Moberg corridor floor for this turn."""
        starved = []
        if texture.get("figurative_density", 0.0) == 0.0:
            starved.append("figurative_density")
        for key, floor in self.floors.items():
            if texture.get(key, 0.0) < floor:
                starved.append(key)
        return starved

    def excess_channels(self, texture):
        """Channels above corridor ceiling -- texture arrived (maybe too much)."""
        return [
            key for key, ceiling in self.ceilings.items()
            if texture.get(key, 0.0) > ceiling
        ]

    def _monotone(self):
        """
        Relative trigger: variance collapse across the last `window`
        drift scores in engine_logs. Includes pity-released turns on
        purpose -- monotony is a property of the actual output stream,
        unlike the nudge-loop baseline which must exclude them.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute(
                "SELECT drift_score FROM engine_logs ORDER BY id DESC LIMIT ?",
                (self.window,),
            )
            rows = [r[0] for r in c.fetchall() if r[0] is not None]
            conn.close()
        except Exception:
            return False
        if len(rows) < self.window:
            return False  # cold start: no relative claim without a full window
        return pstdev(rows) < self.monotony_eps

    # ------------------------------------------------------------------
    # release directive
    # ------------------------------------------------------------------

    _DIRECTIVE_PARTS = {
        "sentence_rhythm": (
            "vary sentence length -- set one long, accumulating sentence "
            "against short declaratives"
        ),
        "figurative_density": (
            "allow a single figurative comparison drawn from labor, "
            "weather, or the animals"
        ),
        "dialogue_density": (
            "allow a brief exchange of spoken words if characters are present"
        ),
        "interiority_pct": (
            "let one thought surface, tied to a physical object, "
            "without naming an emotion"
        ),
    }

    def _release_directive(self, starved):
        """
        Composed from the starved channels so the release asks for exactly
        the texture that is missing. Stays inside the anchor's register:
        no named emotion, one scene only.
        """
        channels = [s for s in self.STARVE_CHANNELS if s in starved]
        if not channels:
            channels = list(self.STARVE_CHANNELS)
        parts = "; ".join(self._DIRECTIVE_PARTS[c] for c in channels)
        return (
            f"[This scene only -- let the prose breathe: {parts}. "
            "Then return to plain physical narration.]\n"
        )

    # ------------------------------------------------------------------
    # hooks
    # ------------------------------------------------------------------

    def pre_turn(self, params):
        """
        Called BEFORE generation with the params RollingBaseline chose.
        Returns (params, injection, meta). params is always a copy --
        the baseline's dicts are never mutated.

        Soft ramp only ever raises temperature / lowers repeat_penalty
        relative to what the baseline chose; it cannot fight an active
        damping action because damped turns are not flat and the streak
        would not have built.
        """
        counter, refractory, starved = self._get_state()
        out = dict(params)

        if refractory > 0:
            return out, "", {
                "stage": "refractory", "fired": False,
                "counter": counter, "released": [],
            }

        if counter >= self.hard_at:
            # Guaranteed release. Reset streak, arm refractory (which
            # covers this release turn itself -- observe() decrements).
            directive = self._release_directive(starved)
            out["temperature"] = max(out.get("temperature", 0.7), self.release_temp)
            out["repeat_penalty"] = min(
                out.get("repeat_penalty", 1.05), self.release_repeat_penalty
            )
            self._set_state(0, self.refractory, [])
            return out, directive, {
                "stage": "hard", "fired": True,
                "counter": counter, "released": starved or list(self.STARVE_CHANNELS),
            }

        if counter >= self.soft_start:
            t = min(1.0, (counter - self.soft_start + 1)
                    / max(1, self.hard_at - self.soft_start))
            base_temp = out.get("temperature", 0.7)
            base_rp = out.get("repeat_penalty", 1.05)
            out["temperature"] = round(
                max(base_temp, base_temp + t * (self.soft_temp_cap - base_temp)), 3
            )
            out["repeat_penalty"] = round(
                min(base_rp, base_rp + t * (1.02 - base_rp)), 3
            )
            return out, "", {
                "stage": "soft", "fired": False,
                "counter": counter, "released": [],
            }

        stage = "building" if counter > 0 else "idle"
        return out, "", {
            "stage": stage, "fired": False,
            "counter": counter, "released": [],
        }

    def observe(self, texture):
        """
        Called AFTER measurement with this turn's TextureAnalyzer dict.
        Updates the streak. Returns meta for telemetry logging.
        """
        counter, refractory, _prev = self._get_state()
        starved = self.starved_channels(texture)
        excess = self.excess_channels(texture)

        if refractory > 0:
            # Release turn / cooldown: measured, logged, never counted.
            self._set_state(counter, refractory - 1, starved)
            return {
                "flat": False, "counter": counter,
                "starved": starved, "excess": excess,
                "refractory": refractory - 1, "monotone": False,
            }

        if excess:
            # Texture arrived (possibly too much -- baseline's problem).
            # Streak broken either way.
            self._set_state(0, 0, [])
            return {
                "flat": False, "counter": 0,
                "starved": starved, "excess": excess,
                "refractory": 0, "monotone": False,
            }

        monotone = self._monotone()
        flat = (len(starved) >= self.flat_channels_min) or (
            len(starved) >= 1 and monotone
        )

        if flat:
            counter += 1
            self._set_state(counter, 0, starved)
        else:
            counter = 0
            self._set_state(0, 0, [])

        return {
            "flat": flat, "counter": counter,
            "starved": starved, "excess": excess,
            "refractory": 0, "monotone": monotone,
        }
