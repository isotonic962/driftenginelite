# INTEGRATION CHECKLIST:
# 1. controller.py — update run_drift_pipeline() signature and system_message build (see below)
# 2. main.py — instantiate SCENE, pass to generate(), handle advance_scene=True (see below)
# 3. telemetry_logger.py — add scene_number column to engine_logs (future pass)

# Scene counter with SQLite persistence and proportional escalation curve.
# Option 1 confirmed: 40 scenes map to 10 Moberg chapter bands (4 scenes per chapter).
# Existing flat DriftScorer corridors unchanged — position-aware corridors are a future pass.

import sqlite3

class SceneCounter:
    """
    Tracks current scene position across kernel restarts via SQLite.

    Escalation curve scales proportionally to total_scenes:
        0–60%:  silent (no injection)
        61–80%: material pressure
        81–90%: breaking point
        91–100%: departure / endpoint line

    At total_scenes=40: silent 1-24, pressure 25-32, breaking 33-36, departure 37-40.
    At total_scenes=10: silent 1-6, pressure 7-8, breaking 9, departure 10.
    Same proportions at both scales — confirmed, not an artifact.

    advance_scene=True in generate() is the ONLY canonical advance path.
    Do not call advance() directly from anywhere else.

    Option-1-vs-option-2 resolved: Option 1.
    40 scenes = 10 chapter bands of 4 scenes each.
    current_chapter() maps scene position to Moberg chapter (1-10)
    for future position-aware corridor work. Not wired into DriftScorer yet.
    """

    def __init__(
        self,
        total_scenes=40,
        endpoint="aboard ship to America",
        db_path="moberg_telemetry.db",
    ):
        self.total = total_scenes
        self.endpoint = endpoint
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS scene_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                current_scene INTEGER NOT NULL DEFAULT 1
            )
        """)
        c.execute(
            "INSERT OR IGNORE INTO scene_state (id, current_scene) VALUES (1, 1)"
        )
        conn.commit()
        conn.close()

    @property
    def current(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT current_scene FROM scene_state WHERE id = 1")
        row = c.fetchone()
        conn.close()
        return row[0] if row else 1

    def advance(self):
        """
        Increment scene by 1, capped at total_scenes.
        Called exclusively via generate(advance_scene=True) in main.py.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            "UPDATE scene_state SET current_scene = MIN(current_scene + 1, ?) WHERE id = 1",
            (self.total,),
        )
        conn.commit()
        conn.close()
        return self.current

    def reset(self):
        """Reset to scene 1. Use deliberately — wipes arc progress."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("UPDATE scene_state SET current_scene = 1 WHERE id = 1")
        conn.commit()
        conn.close()

    def current_chapter(self):
        """
        Maps current scene to a Moberg chapter number (1-10).
        Reserved for future position-aware DriftScorer corridors.
        Not wired into scoring yet — validate flat corridors first.
        At total_scenes=40: scenes 1-4 -> ch1, 5-8 -> ch2, ... 37-40 -> ch10.
        """
        scenes_per_chapter = self.total / 10
        chapter = min(10, int((self.current - 1) / scenes_per_chapter) + 1)
        return chapter

    def inject(self):
        """
        Returns the arc-position injection string per escalation curve, or
        empty string. Thresholds are proportional to self.total — confirmed
        correct behavior, not an artifact. Breaking-point band is always 10%
        of total: 1 scene wide at total=10, 4 scenes wide at total=40.

        The injected line is worded in CHAPTERS, not scenes, even though the
        counter itself is scene-based. This string is prepended to the system
        prompt, so it is the first thing the model reads, and the anchor
        directly below it asks for one full chapter per response. Saying
        "Scene 25 of 40" there primed a scene-sized unit of output against a
        chapter-sized instruction — the two halves of the system prompt
        disagreed about how much prose a turn is. current_chapter() already
        maps position onto the anchor's ten-chapter arc, so it is used here.
        """
        n = self.current
        t = self.total
        ch = self.current_chapter()

        silent_cutoff = t * 0.60
        pressure_cutoff = t * 0.80
        breaking_point = t * 0.90

        if n <= silent_cutoff:
            return ""
        elif n <= pressure_cutoff:
            return f"[Chapter {ch} of 10. The departure approaches.]\n"
        elif n <= breaking_point:
            return f"[Chapter {ch} of 10. Breaking point. The decision cannot be deferred.]\n"
        else:
            return f"[Chapter {ch} of 10. {self.endpoint}.]\n"
