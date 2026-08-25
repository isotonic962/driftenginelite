import sqlite3


class RollingBaseline:
    """
    Reads recent telemetry to recommend generation parameters.

    Axes are now texture-based (action_pct, interiority_pct)
    replacing the old volatility/entropy axes.

    Parameter sets tuned against Moberg benchmark
    (temperature 0.7 produced target profile).
    """

    def __init__(self, db_path="moberg_telemetry.db", window=3):
        self.db_path = db_path
        self.window = window

        self.action_spike = 35.0
        self.action_floor = 12.0
        self.interiority_spike = 15.0

        self.params = {
            "downweight_expressive": {
                "temperature": 0.55,
                "repeat_penalty": 1.08
            },
            "prevent_truncation": {
                "temperature": 0.65,
                "repeat_penalty": 1.05
            },
            "allow_texture": {
                "temperature": 0.78,
                "repeat_penalty": 1.03
            },
            "default": {
                "temperature": 0.70,
                "repeat_penalty": 1.05
            },
        }

    def get_window(self):
        # Pity-released turns (pity_fired=1) are EXCLUDED from the nudge
        # window: a successful hard release spikes interiority/figurative
        # by design, and letting the baseline read that spike would slam
        # downweight_expressive, re-flatten the output, and rebuild the
        # pity streak -- a limit cycle. Released turns are still logged
        # and still count for pity's own monotony window (engine/pity.py),
        # which measures the actual output stream.
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            try:
                c.execute("""
                    SELECT action_pct, interiority_pct
                    FROM engine_logs
                    WHERE COALESCE(pity_fired, 0) = 0
                    ORDER BY id DESC
                    LIMIT ?
                """, (self.window,))
                rows = c.fetchall()
            except sqlite3.OperationalError:
                # Older db without the pity_fired column.
                c.execute("""
                    SELECT action_pct, interiority_pct
                    FROM engine_logs
                    ORDER BY id DESC
                    LIMIT ?
                """, (self.window,))
                rows = c.fetchall()
            conn.close()
            return rows
        except Exception:
            return []

    def get_rolling_averages(self):
        rows = self.get_window()
        if not rows:
            return None, None

        weights = [3, 2, 1][:len(rows)]
        total_weight = sum(weights)
        avg_action = sum(r[0] * w for r, w in zip(rows, weights)) / total_weight
        avg_interiority = sum(r[1] * w for r, w in zip(rows, weights)) / total_weight
        return round(avg_action, 2), round(avg_interiority, 2)

    def recommend(self):
        avg_action, avg_interiority = self.get_rolling_averages()

        if avg_action is None:
            return self.params["default"], "default", None, None

        high_action = avg_action > self.action_spike
        low_action = avg_action < self.action_floor
        high_interiority = avg_interiority > self.interiority_spike

        if high_action and high_interiority:
            action = "downweight_expressive"
        elif high_interiority:
            action = "prevent_truncation"
        elif low_action and not high_interiority:
            action = "allow_texture"
        else:
            action = "default"

        return self.params[action], action, avg_action, avg_interiority