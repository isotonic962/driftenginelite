# Updated to instantiate SceneCounter and pass it through the pipeline.
# advance_scene=True is the ONLY canonical path to increment the scene counter.

from engine.controller import run_drift_pipeline, engine as _engine
from engine.scene_counter import SceneCounter

with open("engine/prompts/system_anchor.txt", "r") as f:
    ANCHOR_TEXT = f.read()

# SceneCounter persists to moberg_telemetry.db — survives kernel restarts.
# Same db as telemetry_logger, different table (scene_state vs engine_logs).
# To wipe scene position without wiping telemetry: call SCENE.reset() explicitly.
SCENE = SceneCounter(total_scenes=40, endpoint="aboard ship to America")


def generate(user_input, advance_scene=False):
    """
    Main entry point for all generation calls.

    advance_scene=True: increments scene counter BEFORE generation,
    so the new scene's inject() string is already in the system prompt
    for this turn. Pass True only on deliberate chapter/scene advancement.

    advance_scene=False (default): counter unchanged, same scene continues.
    Use for all retries, corrections, and continuation turns.
    """
    if advance_scene:
        SCENE.advance()

    output = run_drift_pipeline(
        user_input,
        ANCHOR_TEXT,
        scene_counter=SCENE,
    )

    print("Engine:", output)
    print(f"  [SCENE] {SCENE.current}/{SCENE.total} (chapter {SCENE.current_chapter()}/10)")
    print(f"  [DRIFT STATE] {round(_engine.state.get_state(), 3)}")

    # Pity state for this turn (streak of flat turns -> soft ramp -> hard release)
    counter, refractory, _ = _engine.pity._get_state()
    print(f"  [PITY] counter={counter}/{_engine.pity.hard_at} refractory={refractory}")

    return output


if __name__ == "__main__":
    # REPL mode — development/testing only.
    # Scene counter will NOT advance interactively from this loop.
    # For production (Kaggle notebook), call generate() directly from cells:
    #   generate("input")                        — continue current scene
    #   generate("input", advance_scene=True)     — advance then generate
    #   print(SCENE.current, SCENE.current_chapter())  — check position

    print(f"Drift Engine [DEV] — scene {SCENE.current}/{SCENE.total}")
    print("NOTE: advance_scene not available in REPL. Use notebook cells for arc control.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            print("(empty input skipped)")
            continue
        generate(user_input)
