import requests
import os
import anthropic
from dotenv import load_dotenv
load_dotenv()

from .analyze import DriftAnalyzer
from .drift import DriftScorer
from .drift_state import DriftState
from .pity import PityController
from .quadrant import QuadrantClassifier
from .rolling_baseline import RollingBaseline
from .texture import TextureAnalyzer


class LocalModelClient:
    """
    Model client supporting both Anthropic API and local llama.cpp server.
    Set API_BACKEND=anthropic in .env to use Anthropic, otherwise defaults
    to local llama.cpp endpoint.
    """
    def __init__(self, base_url=None, model=None, api_key=None):
        self.backend  = os.getenv("API_BACKEND", "local")
        self.base_url = base_url or os.getenv("API_BASE", "http://127.0.0.1:8080/v1")
        self.api_key  = api_key  or os.getenv("API_KEY", "")
        self.model    = model    or os.getenv(
            "API_MODEL",
            "claude-haiku-4-5" if self.backend == "anthropic" else "qwen2.5-3b-instruct-q4_k_m"
        )
        if self.backend == "anthropic":
            self._client = anthropic.Anthropic(api_key=self.api_key)

    def chat(self, messages, temperature=0.7, repeat_penalty=1.1):
        if self.backend == "anthropic":
            return self._chat_anthropic(messages, temperature)
        return self._chat_local(messages, temperature, repeat_penalty)

    def _chat_anthropic(self, messages, temperature):
        system_content = ""
        conversation = []
        for m in messages:
            if m["role"] == "system":
                system_content = m["content"]
            else:
                conversation.append(m)

        if not conversation:
            conversation = [{"role": "user", "content": "Begin."}]

        kwargs = {
            "model": self.model,
            "max_tokens": 2048,
            "temperature": temperature,
            "messages": conversation,
        }
        if system_content:
            kwargs["system"] = system_content

        response = self._client.messages.create(**kwargs)
        return response.content[0].text

    def _chat_local(self, messages, temperature, repeat_penalty):
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "repeat_penalty": repeat_penalty,
            "max_tokens": 2048,
            "min_p": 0.05,
        }
        r = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        if content.startswith("assistant\n"):
            content = content[len("assistant\n"):]
        return content


class DriftEngine:
    """
    Measurement-and-nudge pipeline (post-cleanup).

    REMOVED in this pass (Qwen2.5-era post-hoc censor stack, built to
    control a much weaker model and never re-validated against the 14B
    model or real Moberg prose):
        - ConstraintDetector / CycleCounter / OutputTruncator
          (truncation was already a dead stub before this cleanup)
        - BehaviorController (post-generation sentence stripping --
          benchmarked at a 15.3% false-positive rate against real
          Moberg chapters; this was the last live consumer of the
          broken detector)
        - RegisterCheck (already disabled, formalized here)
        - DriftModeClassifier / DriftRecovery (existed only to drive
          BehaviorController's mode dispatch; orphaned once it's gone)
        - SentimentAnalyzer (only remaining caller was OutputTruncator)

    KEPT: pure measurement (entropy, texture) and the corridor-based
    DriftScorer, which is the actual "how close to the Moberg baseline"
    signal -- this is the thing the nudge loop should be built around
    going forward, not a rules engine that edits model output after
    the fact.

    Output now passes through unmodified. The engine measures and
    logs; it does not silently rewrite what the model wrote.
    """

    def __init__(self, model_client, alpha=0.3):
        self.model = model_client
        self.analyzer = DriftAnalyzer()
        self.scorer = DriftScorer()
        self.state = DriftState(alpha=alpha)
        self.quadrant = QuadrantClassifier()
        self.baseline = RollingBaseline()
        self.texture = TextureAnalyzer()
        self.pity = PityController()

    def process(self, text, messages=None, anchor_text=""):
        if messages is None:
            messages = [{"role": "user", "content": text}]

        params, action, avg_action, avg_int = self.baseline.recommend()

        # Pity hook (pre-generation): may ramp params (soft) or inject a
        # one-scene release directive (hard). See engine/pity.py header.
        params, pity_injection, pity_pre = self.pity.pre_turn(params)
        if pity_injection:
            messages = [dict(m) for m in messages]
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] = pity_injection + messages[0]["content"]
            else:
                messages.insert(
                    0, {"role": "system", "content": pity_injection}
                )

        response_text = self.model.chat(messages, **params)

        texture_data = self.texture.analyze(response_text, text)
        raw_analysis = self.analyzer.analyze(response_text)
        entropy = raw_analysis.get("entropy", 0.0)

        drift_info = self.scorer.score(texture_data, entropy=entropy)
        drift_score = drift_info["drift_score"]

        state_value = self.state.update(drift_score)

        # NOTE: previously this passed `entropy` as the second argument
        # here, but QuadrantClassifier.classify() expects interiority_pct
        # (range 0-17, spike=15.0) -- entropy (range ~7-9) was silently
        # being treated as interiority. Fixed to pass the correct field.
        quadrant = self.quadrant.classify(
            texture_data.get("action_pct", 0.0),
            texture_data.get("interiority_pct", 0.0),
        )

        # Pity hook (post-measurement): update the flatness streak from
        # this turn's texture. Runs AFTER telemetry-window reads inside
        # pre_turn/_monotone but BEFORE this turn is logged -- so the
        # monotony window never includes the turn being judged.
        pity_post = self.pity.observe(texture_data)
        pity_meta = {
            "stage": pity_pre["stage"],
            "fired": pity_pre["fired"],
            "released": pity_pre["released"],
            "counter": pity_post["counter"],
            "flat": pity_post["flat"],
            "starved": pity_post["starved"],
            "excess": pity_post["excess"],
            "monotone": pity_post["monotone"],
            "temperature": params.get("temperature"),
            "repeat_penalty": params.get("repeat_penalty"),
        }

        return {
            "analysis": raw_analysis,
            "texture": texture_data,
            "drift_components": drift_info,
            "state": state_value,
            "raw_response": response_text,
            "response": response_text,
            "quadrant": quadrant,
            "baseline_action": action,
            "pity": pity_meta,
        }


if __name__ == "__main__":
    client = LocalModelClient()
    engine = DriftEngine(model_client=client)

    result = engine.process("Hello there")
    print(result["response"])

    while True:
        user_input = input("> ")
        if user_input.strip().lower() in ["exit", "quit"]:
            break

        result = engine.process(user_input)
        print(result["response"])

        tex = result["texture"]
        drift = result["drift_components"]
        print(
            f'  [TEXTURE] fig={tex["figurative_density"]} '
            f'act={tex["action_pct"]} int={tex["interiority_pct"]} '
            f'dial={tex["dialogue_density"]} rhythm={tex["sentence_rhythm"]} '
            f'echo={tex["prompt_echo"]}'
        )
        print(
            f'  [DRIFT] score={drift["drift_score"]:.3f} '
            f'int_dev={drift["interiority_pct_dev"]:.2f} '
            f'fig_dev={drift["figurative_density_dev"]:.3f} '
            f'dial_dev={drift["dialogue_density_dev"]:.2f}'
        )
