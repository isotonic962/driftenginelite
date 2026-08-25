# Dev verification for engine/pity.py + wiring (same spirit as
# engine/verify_scorer.py: synthetic inputs, hand-checked expectations).
# Run from repo root: python test_pity.py
# Uses a throwaway db -- never touches moberg_telemetry.db.

import os
import sqlite3
import tempfile

from engine.pity import PityController
from engine.rolling_baseline import RollingBaseline
from engine.telemetry_logger import TelemetryLogger

FLAT = {  # action-dense, everything else starved (the observed 14B failure mode)
    "figurative_density": 0.0, "action_pct": 40.0, "interiority_pct": 0.0,
    "neutral_pct": 60.0, "dialogue_density": 0.0, "sentence_rhythm": 4.0,
    "prompt_echo": 5.0,
}
TEXTURED = {  # inside all corridors
    "figurative_density": 0.10, "action_pct": 22.0, "interiority_pct": 9.0,
    "neutral_pct": 69.0, "dialogue_density": 8.0, "sentence_rhythm": 10.5,
    "prompt_echo": 5.0,
}
EXCESS = {  # over the interiority ceiling
    "figurative_density": 0.30, "action_pct": 22.0, "interiority_pct": 20.0,
    "neutral_pct": 58.0, "dialogue_density": 8.0, "sentence_rhythm": 10.5,
    "prompt_echo": 5.0,
}
MILD = {  # only dialogue starved -- flat ONLY under monotone lock
    "figurative_density": 0.10, "action_pct": 22.0, "interiority_pct": 9.0,
    "neutral_pct": 69.0, "dialogue_density": 0.0, "sentence_rhythm": 10.5,
    "prompt_echo": 5.0,
}

BASE_PARAMS = {"temperature": 0.70, "repeat_penalty": 1.05}

passed = failed = 0

def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")

def fresh_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    return path


# ---------------------------------------------------------------- streak
print("[1] streak accumulation / reset")
db = fresh_db()
p = PityController(db_path=db)

m = p.observe(FLAT)
check("flat turn counts", m["flat"] and m["counter"] == 1, str(m))
check("starved channels detected", len(m["starved"]) == 4, str(m["starved"]))
m = p.observe(FLAT)
check("streak builds", m["counter"] == 2)
m = p.observe(TEXTURED)
check("textured turn resets streak", (not m["flat"]) and m["counter"] == 0, str(m))

p.observe(FLAT); p.observe(FLAT)
m = p.observe(EXCESS)
check("excess breaks streak", m["counter"] == 0 and m["excess"], str(m))

# ---------------------------------------------------------------- stages
print("[2] stage progression + soft ramp")
p.reset()
_, inj, meta = p.pre_turn(BASE_PARAMS)
check("idle at counter 0", meta["stage"] == "idle" and inj == "")

p.observe(FLAT)
_, inj, meta = p.pre_turn(BASE_PARAMS)
check("building below soft_start", meta["stage"] == "building")

p.observe(FLAT); p.observe(FLAT)  # counter = 3
prm, inj, meta = p.pre_turn(BASE_PARAMS)
check("soft at soft_start", meta["stage"] == "soft" and inj == "")
check("soft raises temperature", BASE_PARAMS["temperature"] < prm["temperature"] <= p.soft_temp_cap,
      str(prm))
check("soft lowers repeat_penalty", prm["repeat_penalty"] < BASE_PARAMS["repeat_penalty"], str(prm))
check("baseline params not mutated", BASE_PARAMS["temperature"] == 0.70)

p.observe(FLAT); p.observe(FLAT)  # counter = 5
prm5, _, _ = p.pre_turn(BASE_PARAMS)
check("ramp increases with streak", prm5["temperature"] > prm["temperature"], f'{prm} vs {prm5}')

# ---------------------------------------------------------------- hard fire
print("[3] hard release + refractory")
p.observe(FLAT)  # counter = 6 = hard_at
prm, inj, meta = p.pre_turn(BASE_PARAMS)
check("hard fires at hard_at", meta["stage"] == "hard" and meta["fired"])
check("directive injected", inj.startswith("[") and "This scene only" in inj, inj[:60])
check("directive targets starved channels", "figurative comparison" in inj and "thought" in inj)
check("release params", prm["temperature"] == p.release_temp
      and prm["repeat_penalty"] == p.release_repeat_penalty, str(prm))

m = p.observe(TEXTURED)  # the release turn itself
check("release turn not counted (refractory)", m["counter"] == 0 and m["refractory"] == 1, str(m))
_, inj, meta = p.pre_turn(BASE_PARAMS)
check("still refractory next turn", meta["stage"] == "refractory" and inj == "")
m = p.observe(FLAT)
check("refractory expires", m["refractory"] == 0)
m = p.observe(FLAT)
check("accumulation resumes after refractory", m["counter"] == 1, str(m))

# ---------------------------------------------------------------- persistence
print("[4] persistence across restarts")
p2 = PityController(db_path=db)
c, r, s = p2._get_state()
check("counter survives re-instantiation", c == 1, f"counter={c}")
os.unlink(db)

# ---------------------------------------------------------------- monotony
print("[5] monotone lock (relative trigger)")
db = fresh_db()
TelemetryLogger(db_path=db)  # create engine_logs schema
conn = sqlite3.connect(db)
for _ in range(5):  # variance collapse: five near-identical drift scores
    conn.execute("INSERT INTO engine_logs (drift_score, action_pct, interiority_pct) VALUES (3.4, 22.0, 9.0)")
conn.commit(); conn.close()

p = PityController(db_path=db)
m = p.observe(MILD)
check("mildly starved counts flat under monotone lock", m["monotone"] and m["flat"], str(m))

db2 = fresh_db()
TelemetryLogger(db_path=db2)
p = PityController(db_path=db2)
m = p.observe(MILD)
check("mildly starved NOT flat without full window (cold start)",
      (not m["monotone"]) and (not m["flat"]), str(m))
os.unlink(db2)

# ---------------------------------------------------------------- baseline exclusion
print("[6] RollingBaseline excludes released turns")
conn = sqlite3.connect(db)
conn.execute("INSERT INTO engine_logs (drift_score, action_pct, interiority_pct, pity_fired) VALUES (1.0, 25.0, 30.0, 1)")
conn.commit(); conn.close()

rb = RollingBaseline(db_path=db, window=3)
rows = rb.get_window()
check("pity_fired row excluded", all(r[1] != 30.0 for r in rows), str(rows))
check("window still fills from clean rows", len(rows) == 3, str(rows))
os.unlink(db)

# ---------------------------------------------------------------- logger columns
print("[7] telemetry pity columns (fresh db + ALTER on old db)")
db = fresh_db()
# simulate an OLD db: schema without pity columns
conn = sqlite3.connect(db)
conn.execute("CREATE TABLE engine_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, user_prompt TEXT, final_output TEXT, sentiment_raw REAL, volatility_raw REAL, entropy_raw REAL, drift_score REAL, engine_state TEXT, mode TEXT, figurative_density REAL, action_pct REAL, interiority_pct REAL, neutral_pct REAL, dialogue_density REAL, sentence_rhythm REAL, prompt_echo REAL)")
conn.commit(); conn.close()

tl = TelemetryLogger(db_path=db)  # should ALTER in the pity columns
tl.log_event("p", "o", {"entropy": 8.5}, 2.1, 1.4, texture=TEXTURED,
             pity={"counter": 4, "stage": "soft", "fired": False})
conn = sqlite3.connect(db)
row = conn.execute("SELECT pity_counter, pity_stage, pity_fired FROM engine_logs").fetchone()
conn.close()
check("pity columns logged on migrated db", row == (4, "soft", 0), str(row))
os.unlink(db)

# ---------------------------------------------------------------- end-to-end
print("[8] end-to-end wiring through DriftEngine.process")
db = fresh_db()
os.chdir(tempfile.mkdtemp())  # keep default-path dbs out of the repo

from engine.drift_engine import DriftEngine

class FakeClient:
    def __init__(self):
        self.last_messages = None
        self.last_kwargs = None
    def chat(self, messages, **kwargs):
        self.last_messages = messages
        self.last_kwargs = kwargs
        return ("He set the axe against the wall. He crossed the yard. "
                "He lifted the pail. He carried it to the barn. "
                "He closed the door. He walked back.")

client = FakeClient()
eng = DriftEngine(model_client=client)
eng.pity = PityController(db_path=db, hard_at=2, refractory=1, soft_start=1)

msgs = [{"role": "system", "content": "ANCHOR"}, {"role": "user", "content": "scene"}]
r1 = eng.process("scene", messages=[dict(m) for m in msgs])
check("pity meta in result", "pity" in r1 and r1["pity"]["stage"] == "idle", str(r1.get("pity")))
check("flat output counted", r1["pity"]["counter"] == 1, str(r1["pity"]))

r2 = eng.process("scene", messages=[dict(m) for m in msgs])  # counter 1 -> soft
r3 = eng.process("scene", messages=[dict(m) for m in msgs])  # counter 2 = hard_at -> fire
check("hard fired end-to-end", r3["pity"]["stage"] == "hard" and r3["pity"]["fired"], str(r3["pity"]))
check("directive prepended to system message",
      client.last_messages[0]["content"].startswith("[This scene only"),
      client.last_messages[0]["content"][:60])
check("anchor preserved after injection", client.last_messages[0]["content"].endswith("ANCHOR"))
check("release temperature reached model call",
      client.last_kwargs.get("temperature") == eng.pity.release_temp, str(client.last_kwargs))
os.unlink(db)

print(f"\n{passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)
