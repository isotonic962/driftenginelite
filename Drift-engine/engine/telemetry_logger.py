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

    LENGTH COLUMNS: word_count / output_tokens / finish_reason /
    template_leak were added because nothing in the pipeline could see
    output length. Every texture metric is a ratio normalized by sentence
    or token count, so a 115-word chapter and a 900-word one produce
    identical drift scores. finish_reason is what separates "the model
    ended the chapter" from "we hit max_tokens" from "the server ran out
    of context" -- only the first implicates the LoRA.
    """

    # Columns added after the first databases were created. CREATE TABLE IF
    # NOT EXISTS silently no-ops on an existing table, so these never
    # appeared in older files and every INSERT naming them raised
    # OperationalError. _migrate() adds whatever is missing.
    ADDED_COLUMNS = {
        "neutral_pct":    "REAL",
        "word_count":     "INTEGER",
        "output_tokens":  "INTEGER",
        "finish_reason":  "TEXT",
        "template_leak":  "INTEGER",
    }

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
                prompt_echo REAL,
                word_count INTEGER,
                output_tokens INTEGER,
                finish_reason TEXT,
                template_leak INTEGER
            )
        ''')
        self._migrate(conn)
        conn.commit()
        conn.close()

    def _migrate(self, conn):
        """
        Add any column missing from a pre-existing engine_logs table.

        Names come from ADDED_COLUMNS, not from caller input, so the
        f-string here is not an injection surface -- SQLite does not accept
        bound parameters for identifiers in ALTER TABLE.
        """
        c = conn.cursor()
        existing = {row[1] for row in c.execute("PRAGMA table_info(engine_logs)")}
        for name, decl in self.ADDED_COLUMNS.items():
            if name not in existing:
                c.execute(f"ALTER TABLE engine_logs ADD COLUMN {name} {decl}")

    def log_event(self, prompt, output, analysis, drift_score, state, texture=None,
                  word_count=None, output_tokens=None, finish_reason=None,
                  template_leak=None):
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
                word_count, output_tokens, finish_reason, template_leak
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            word_count, output_tokens, finish_reason,
            None if template_leak is None else int(template_leak)
        ))

        conn.commit()
        conn.close()
