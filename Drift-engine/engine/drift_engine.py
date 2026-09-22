import requests
import os
import anthropic
from dotenv import load_dotenv
load_dotenv()

from .analyze import DriftAnalyzer
from .drift import DriftScorer
from .drift_state import DriftState
from .quadrant import QuadrantClassifier
from .rolling_baseline import RollingBaseline
from .texture import TextureAnalyzer


class LocalModelClient:
    """
    Model client supporting both Anthropic API and local llama.cpp server.
    Set API_BACKEND=anthropic in .env to use Anthropic, otherwise defaults
    to local llama.cpp endpoint.

    chat() returns a dict, not a bare string:
        {"text", "finish_reason", "output_tokens", "template_leak"}

    finish_reason is the only thing that distinguishes "the model ended the
    chapter" from "we cut it off at max_tokens" from "the server ran out of
    context". Those three have three different fixes and only the first one
    implicates the LoRA, so it is captured on every call rather than
    discarded. Same for output_tokens -- every texture metric is a ratio
    normalized by length, so without this the pipeline cannot see length
    at all.
    """

    # Role/template markers that must never appear in generation output.
    # If they do, the served chat template does not match the one the model
    # was trained under -- see _clean().
    LEAK_PREFIXES = ("assistant\n", "assistant\r\n", "assistant:")
    LEAK_MARKERS  = ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "</s>")

    _leak_warned = False

    def __init__(self, base_url=None, model=None, api_key=None, max_tokens=None):
        self.backend  = os.getenv("API_BACKEND", "local")
        self.base_url = base_url or os.getenv("API_BASE", "http://127.0.0.1:8080/v1")
        self.api_key  = api_key  or os.getenv("API_KEY", "")
        # max_tokens is a ceiling on chapter length, so it belongs in config
        # rather than hard-coded in two places. Raise it and the llama.cpp
        # server's -c together -- raising either alone just moves the wall.
        self.max_tokens = int(max_tokens or os.getenv("MAX_TOKENS", "2048"))
        self.model    = model    or os.getenv(
            "API_MODEL",
            "claude-haiku-4-5" if self.backend == "anthropic" else "qwen2.5-3b-instruct-q4_k_m"
        )
        if self.backend == "anthropic":
            self._client = anthropic.Anthropic(api_key=self.api_key)

    def _clean(self, text):
        """
        Strip leaked chat-template role markers, and report whether any were
        found.

        A leak is not cosmetic. If llama.cpp is emitting "assistant\n" into
        generation output then the template it built the prompt with is not
        the one the model was trained under, which means the model is also
        being stopped on the wrong end-of-turn token -- a direct cause of
        short output. Stripping keeps TextureAnalyzer's sentence
        classification clean; the returned flag is what tells you the server
        itself needs fixing (--jinja, or an explicit --chat-template).
        """
        cleaned = text or ""
        leaked = False

        while True:
            stripped = cleaned.lstrip()
            hit = next(
                (p for p in self.LEAK_PREFIXES if stripped.lower().startswith(p)),
                None,
            )
            if hit is None:
                break
            cleaned = stripped[len(hit):]
            leaked = True

        for marker in self.LEAK_MARKERS:
            if marker in cleaned:
                cleaned = cleaned.replace(marker, "")
                leaked = True

        if leaked and not LocalModelClient._leak_warned:
            print(
                "[WARN] chat-template leak: role/template markers appeared in "
                "generation output. The served template does not match the "
                "model's own, which means the stop token does not either. "
                "Relaunch llama.cpp with --jinja or an explicit --chat-template "
                "before drawing any conclusion about the LoRA."
            )
            LocalModelClient._leak_warned = True

        return cleaned.strip(), leaked

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
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            "messages": conversation,
        }
        if system_content:
            kwargs["system"] = system_content

        response = self._client.messages.create(**kwargs)

        # Join every text block rather than taking content[0] -- indexing the
        # first block silently drops the response if anything non-text leads.
        text = "".join(
            b.text for b in response.content if getattr(b, "type", None) == "text"
        )
        cleaned, leaked = self._clean(text)

        return {
            "text": cleaned,
            "finish_reason": response.stop_reason,
            "output_tokens": getattr(response.usage, "output_tokens", None),
            "template_leak": leaked,
        }

    def _chat_local(self, messages, temperature, repeat_penalty):
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "repeat_penalty": repeat_penalty,
            "max_tokens": self.max_tokens,
            "min_p": 0.05,
        }
        r = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
        )
        r.raise_for_status()
        data = r.json()
        choice = data["choices"][0]
        cleaned, leaked = self._clean(choice["message"].get("content"))
        usage = data.get("usage") or {}

        return {
            "text": cleaned,
            "finish_reason": choice.get("finish_reason"),
            "output_tokens": usage.get("completion_tokens"),
            "template_leak": leaked,
        }


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
        self.last = None

    def process(self, text, messages=None, anchor_text=""):
        if messages is None:
            messages = [{"role": "user", "content": text}]

        params, action, avg_action, avg_int = self.baseline.recommend()
        completion = self.model.chat(messages, **params)

        response_text  = completion["text"]
        finish_reason  = completion.get("finish_reason")
        output_tokens  = completion.get("output_tokens")
        template_leak  = completion.get("template_leak", False)
        word_count     = len(response_text.split())

        # Length is invisible to every other signal in this pipeline: the
        # texture metrics are all ratios normalized by sentence or token
        # count, so identical prose at 900 words and 115 words scores the
        # same drift. This is the only place a short chapter is detectable.
        if finish_reason in ("length", "max_tokens"):
            print(
                f"[WARN] generation stopped at the {self.model.max_tokens}-token "
                f"cap ({word_count} words) -- the chapter was cut off, not "
                f"finished. Raise MAX_TOKENS and the llama.cpp -c together."
            )

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

        result = {
            "analysis": raw_analysis,
            "texture": texture_data,
            "drift_components": drift_info,
            "state": state_value,
            "raw_response": response_text,
            "response": response_text,
            "quadrant": quadrant,
            "baseline_action": action,
            "word_count": word_count,
            "output_tokens": output_tokens,
            "finish_reason": finish_reason,
            "template_leak": template_leak,
        }

        # Kept so callers that only receive the final string (main.generate)
        # can still read this turn's length/stop metadata.
        self.last = result
        return result


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
            f'  [LENGTH] words={result["word_count"]} '
            f'tokens={result["output_tokens"]} '
            f'finish={result["finish_reason"]}'
        )
        print(
            f'  [DRIFT] score={drift["drift_score"]:.3f} '
            f'int_dev={drift["interiority_pct_dev"]:.2f} '
            f'fig_dev={drift["figurative_density_dev"]:.3f} '
            f'dial_dev={drift["dialogue_density_dev"]:.2f}'
        )
