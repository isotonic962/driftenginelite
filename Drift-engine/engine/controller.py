# Updated to accept scene_counter and prepend inject() to system_message.
# advance_scene param removed — advance happens in main.py before this call.

from engine.drift_engine import DriftEngine, LocalModelClient
from engine.telemetry_logger import TelemetryLogger
from engine.memory import MemoryWindow
from engine.texture import TextureAnalyzer

client = LocalModelClient()
engine = DriftEngine(model_client=client)
logger = TelemetryLogger()
texture_analyzer = TextureAnalyzer()
memory = MemoryWindow(size=3)


def run_drift_pipeline(user_input, anchor_text, scene_counter=None):
    """
    scene_counter: SceneCounter instance or None.
    If provided, scene_counter.inject() is prepended to the system prompt.
    Advance logic lives in main.py — this function never calls advance().
    """
    scene_injection = scene_counter.inject() if scene_counter is not None else ""

    system_message = scene_injection + anchor_text

    messages = [{"role": "system", "content": system_message}]

    for exchange in memory.get_texts():
        messages.append({"role": "user", "content": exchange["user"]})
        messages.append({"role": "assistant", "content": exchange["assistant"]})

    messages.append({"role": "user", "content": user_input})

    result = engine.process(user_input, messages=messages, anchor_text=anchor_text)

    final_text = result["response"]
    raw_analysis = result["analysis"]
    texture_data = result["texture"]
    final_drift_score = result["drift_components"]["drift_score"]
    current_state = result["state"]

    memory.add({"user": user_input, "assistant": final_text})

    logger.log_event(
        prompt=user_input,
        output=final_text,
        analysis=raw_analysis,
        drift_score=final_drift_score,
        state=current_state,
        texture=texture_data,
        word_count=result.get("word_count"),
        output_tokens=result.get("output_tokens"),
        finish_reason=result.get("finish_reason"),
        template_leak=result.get("template_leak"),
    )

    return final_text
