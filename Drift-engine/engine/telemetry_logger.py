import sqlite3
import datetime

class TelemetryLogger:
    """
    NOTE ON SCHEMA: sentiment_raw, volatility_raw, and mode are retired
    metrics -- DriftAnalyzer stopped producing sentiment/volatility when
    SentimentAnalyzer was removed from the pipeline, and DriftModeClassifier
    was removed in the same cleanup pass that removed BehaviorController.

    Rather than back-fill 0.0 (which is indistinguishable from a genuine
    zero reading under the old scorer and would corrupt any historical
    comparison), these columns are left NULL going forward. The columns
    themselves are kept in the schema for backward compatibility with
    existing committed telemetry data (SQLite can't cheaply drop columns
    pre-3.35 without a full table rebuild) -- but log_event() no longer
    reads analysis["sentiment"] / analysis["volatility"] at all, so there
    is no KeyError risk regardless of what DriftAnalyzer returns.
    """

    def __init__(self, db_path="moberg_telemetry.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Creates the telemetry table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS engine_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                user_prompt TEXT,
                final_output TEXT,
                sentiment_raw REAL,
                volatility_raw REAL,
                entropy_raw REAL,
                drift_score REAL,
                engine_state TEXT,
                mode TEXT,
                figurative_density REAL,
                action_pct REAL,
                interiority_pct REAL,
                neutral_pct REAL,
                dialogue_density REAL,
                sentence_rhythm REAL,
                prompt_echo REAL
            )
        ''')
        # Pity columns, added via ALTER for existing dbs (same
        # keep-the-schema-compatible policy as the retired sentiment
        # columns above). pity_fired is read back by RollingBaseline
        # to exclude release turns from the nudge window.
        for col, coltype in [
            ("pity_counter", "INTEGER"),
            ("pity_stage", "TEXT"),
            ("pity_fired", "INTEGER"),
        ]:
            try:
                c.execute(f"ALTER TABLE engine_logs ADD COLUMN {col} {coltype}")
            except sqlite3.OperationalError:
                pass  # column already exists
        conn.commit()
        conn.close()

    def log_event(self, prompt, output, analysis, drift_score, state,
                  texture=None, pity=None):
        """Call this at the very end of your engine.process() loop."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        timestamp = datetime.datetime.now().isoformat()

        c.execute('''
            INSERT INTO engine_logs (
                timestamp, user_prompt, final_output,
                sentiment_raw, volatility_raw, entropy_raw,
                drift_score, engine_state, mode,
                figurative_density, action_pct, interiority_pct, neutral_pct,
                dialogue_density, sentence_rhythm, prompt_echo,
                pity_counter, pity_stage, pity_fired
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp, prompt, output,
            None, None, analysis.get("entropy", 0.0),
            drift_score, state, None,
            texture.get("figurative_density", 0) if texture else 0,
            texture.get("action_pct", 0) if texture else 0,
            texture.get("interiority_pct", 0) if texture else 0,
            texture.get("neutral_pct", 0) if texture else 0,
            texture.get("dialogue_density", 0) if texture else 0,
            texture.get("sentence_rhythm", 0) if texture else 0,
            texture.get("prompt_echo", 0) if texture else 0,
            pity.get("counter") if pity else None,
            pity.get("stage") if pity else None,
            (1 if pity.get("fired") else 0) if pity else 0,
        ))

        conn.commit()
        conn.close()
