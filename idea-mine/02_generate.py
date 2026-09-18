#!/usr/bin/env python
"""Step 2: raw-completion idea recombination with a BASE model.

Each sample = 3 random failed patents (title + abstract, one of them with its first claim)
+ 1 random retro snippet, joined by blank lines, ending with the fixed suffix from config.yaml.
No chat template, no instructions, no system prompt: the text is fed to the model as-is.

Output is appended to runs/YYYY-MM-DD.jsonl. Sample i of a given date always has the same seed
and the same sources, so rerunning the same day only fills in what is missing.
"""

import argparse
import math
import os
import random
import re
import sys
import time
from pathlib import Path

from common import (
    HERE,
    append_jsonl,
    failures_file,
    load_config,
    log,
    parse_date,
    read_jsonl,
    retro_file,
    run_file,
    stable_seed,
    today_str,
)

BAD_MODEL_WORDS = ("instruct", "chat", "coder", "math", "-vl", "audio", "omni", "reward", "embed", "rerank", "guard")


# ----------------------------------------------------------------------------- model resolution


def _looks_like_base_qwen(d):
    name = str(d).lower()
    if "qwen" not in name or any(w in name for w in BAD_MODEL_WORDS):
        return False
    if not (d / "config.json").exists():
        return False
    if not any(d.glob("*.safetensors")) and not any(d.glob("*.bin")):
        return False
    return True


def find_local_model(cfg):
    mcfg = cfg["model"]
    if mcfg.get("path"):
        p = Path(os.path.expanduser(mcfg["path"]))
        if not p.exists():
            raise SystemExit(f"model.path {p} does not exist")
        return str(p)
    want = mcfg["name"].split("/")[-1].lower()
    candidates = []
    for root in mcfg.get("local_search_roots", []):
        root = Path(os.path.expanduser(root))
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            depth = len(Path(dirpath).relative_to(root).parts)
            if depth > 5:
                dirnames[:] = []
                continue
            if "config.json" in filenames and _looks_like_base_qwen(Path(dirpath)):
                candidates.append(Path(dirpath))
    if not candidates:
        return None
    exact = [c for c in candidates if want in str(c).lower().replace("--", "/")]
    pick = sorted(exact or candidates, key=lambda p: str(p))[0]
    return str(pick)


def resolve_model(cfg):
    local = find_local_model(cfg)
    if local:
        log(f"[model] using local base model: {local}")
        return local
    name = cfg["model"]["name"]
    log(f"[model] no local Qwen base checkpoint found; will download {name} from HuggingFace")
    return name


def assert_base_model(path_or_name):
    low = path_or_name.lower()
    if any(w in low for w in ("instruct", "chat")):
        raise SystemExit(f"refusing to use a non-base model: {path_or_name}")


# ----------------------------------------------------------------------------- prompt construction


def failure_block(f, with_claim):
    block = f"{f['title'].strip()}\n\n{f['abstract'].strip()}"
    if with_claim and f.get("first_claim"):
        block += f"\n\n{f['first_claim'].strip()}"
    return block


def build_sample(date, index, cfg, failures, retro, count_tokens):
    """Deterministically choose sources and build the raw prompt for sample `index` of `date`."""
    gcfg = cfg["generation"]
    seed = stable_seed(date, index, gcfg.get("seed_salt", 0))
    rng = random.Random(seed)
    fidx = rng.sample(range(len(failures)), 3)
    claim_slot = rng.randrange(3)
    ridx = rng.randrange(len(retro))
    fails = [failures[i] for i in fidx]
    snippet = retro[ridx]
    suffix = gcfg["suffix"]
    limit = gcfg["max_prompt_tokens"]

    retro_text = snippet["text"].strip()
    words = retro_text.split()
    while True:
        parts = [failure_block(f, k == claim_slot) for k, f in enumerate(fails)]
        parts.append(" ".join(words) if len(words) < len(retro_text.split()) else retro_text)
        prompt = "\n\n".join(parts) + suffix
        if count_tokens(prompt) <= limit or len(words) <= 40:
            break
        words = words[: max(40, int(len(words) * 0.8))]
    return {
        "id": f"{date}-{index:04d}",
        "index": index,
        "seed": seed,
        "source_ids": [f["id"] for f in fails] + [snippet["id"]],
        "claim_source_id": fails[claim_slot]["id"],
        "prompt": prompt,
    }


# ----------------------------------------------------------------------------- backends


def gpu_vram_gb():
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 2**30
    except Exception:
        pass
    return 0.0


def choose_backend(cfg):
    gcfg = cfg["generation"]
    want = gcfg.get("backend", "auto")
    vram = gpu_vram_gb()
    try:
        import vllm  # noqa: F401

        have_vllm = True
    except Exception:
        have_vllm = False
    if want == "vllm":
        if not have_vllm:
            raise SystemExit("backend=vllm requested but vllm is not installed")
        return "vllm", vram
    if want == "transformers":
        return "transformers", vram
    if have_vllm and vram >= gcfg["vllm_min_vram_gb"]:
        return "vllm", vram
    return "transformers", vram


class VLLMBackend:
    def __init__(self, model, cfg):
        from vllm import LLM, SamplingParams

        gcfg = cfg["generation"]
        self.SamplingParams = SamplingParams
        self.llm = LLM(
            model=model,
            dtype="bfloat16",
            max_model_len=gcfg["max_prompt_tokens"] + gcfg["max_new_tokens"] + 64,
            gpu_memory_utilization=gcfg["vllm_gpu_memory_utilization"],
            seed=0,
        )
        self.tok = self.llm.get_tokenizer()
        self.gcfg = gcfg

    def count_tokens(self, text):
        return len(self.tok.encode(text, add_special_tokens=False))

    def generate(self, samples):
        g = self.gcfg
        params = [
            self.SamplingParams(
                temperature=g["temperature"], min_p=g["min_p"], top_p=1.0, top_k=-1,
                repetition_penalty=1.0, max_tokens=g["max_new_tokens"], seed=s["seed"], logprobs=1,
            )
            for s in samples
        ]
        outs = self.llm.generate([s["prompt"] for s in samples], params, use_tqdm=True)
        results = []
        for o in outs:
            c = o.outputs[0]
            lps = []
            for tid, lp in zip(c.token_ids, c.logprobs or []):
                if lp and tid in lp:
                    lps.append(lp[tid].logprob)
            mean_lp = float(sum(lps) / len(lps)) if lps else float("nan")
            results.append({"output": c.text, "n_tokens": len(c.token_ids), "mean_logprob": mean_lp,
                            "finish_reason": c.finish_reason})
        return results


class TransformersBackend:
    def __init__(self, model, cfg, vram):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        gcfg = cfg["generation"]
        self.gcfg = gcfg
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kwargs = {}
        if torch.cuda.is_available():
            if vram >= gcfg["transformers_bf16_min_vram_gb"]:
                kwargs.update(dtype=torch.bfloat16, device_map="cuda")
                log(f"[model] transformers bf16 on GPU ({vram:.1f} GiB)")
            else:
                from transformers import BitsAndBytesConfig

                kwargs.update(
                    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.bfloat16),
                    device_map="cuda",
                )
                log(f"[model] transformers 4-bit (bitsandbytes) on GPU ({vram:.1f} GiB)")
        else:
            kwargs.update(dtype=torch.float32)
            log("[model] WARNING: no GPU found; running transformers on CPU (very slow for a 7B model)")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
        except TypeError:  # transformers < 4.56 spelled the dtype argument differently
            if "dtype" in kwargs:
                kwargs["torch_dtype"] = kwargs.pop("dtype")
            self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def count_tokens(self, text):
        return len(self.tok.encode(text, add_special_tokens=False))

    def generate(self, samples):
        torch = self.torch
        g = self.gcfg
        results = []
        bs = g["batch_size"]
        for start in range(0, len(samples), bs):
            batch = samples[start:start + bs]
            enc = self.tok([s["prompt"] for s in batch], return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.device)
            torch.manual_seed(batch[0]["seed"])
            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    do_sample=True,
                    temperature=g["temperature"],
                    min_p=g["min_p"],
                    top_p=1.0,
                    top_k=0,
                    repetition_penalty=1.0,
                    max_new_tokens=g["max_new_tokens"],
                    output_logits=True,
                    return_dict_in_generate=True,
                    pad_token_id=self.tok.pad_token_id,
                )
            gen = out.sequences[:, enc["input_ids"].shape[1]:]
            # raw (pre-temperature) log-probabilities of the sampled tokens
            logits = torch.stack(out.logits, dim=1).float()  # [B, T, V]
            logp = torch.log_softmax(logits, dim=-1)
            chosen = logp.gather(-1, gen.unsqueeze(-1)).squeeze(-1)  # [B, T]
            for b, s in enumerate(batch):
                ids = gen[b].tolist()
                n = len(ids)
                eos = self.tok.eos_token_id
                for t, tid in enumerate(ids):
                    if tid == eos or tid == self.tok.pad_token_id:
                        n = t
                        break
                lps = chosen[b, :n].tolist()
                text = self.tok.decode(ids[:n], skip_special_tokens=True)
                results.append({
                    "output": text,
                    "n_tokens": n,
                    "mean_logprob": float(sum(lps) / n) if n else float("nan"),
                    "finish_reason": "stop" if n < len(ids) else "length",
                })
            log(f"[gen] {min(start + bs, len(samples))}/{len(samples)}")
        return results


# ----------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--date", default=None, help="run date YYYY-MM-DD (default: today)")
    ap.add_argument("--n", type=int, default=None, help="samples for this date (default: config n_samples)")
    ap.add_argument("--chunk", type=int, default=32, help="samples generated between log flushes")
    args = ap.parse_args()
    cfg = load_config(args.config)
    date = parse_date(args.date) if args.date else today_str()
    n_total = args.n or cfg["generation"]["n_samples"]

    failures = read_jsonl(failures_file(cfg))
    retro = read_jsonl(retro_file(cfg))
    if len(failures) < 3 or not retro:
        raise SystemExit("corpus missing or too small; run 01_pull_data.py first")
    failures.sort(key=lambda r: r["id"])
    retro.sort(key=lambda r: r["id"])

    rf = run_file(cfg, date)
    done = {r["index"] for r in read_jsonl(rf)}
    todo = [i for i in range(n_total) if i not in done]
    log(f"[run] date={date} target={n_total} done={len(done)} todo={len(todo)} -> {rf}")
    if not todo:
        return

    model = resolve_model(cfg)
    assert_base_model(model)
    backend_name, vram = choose_backend(cfg)
    log(f"[backend] {backend_name} (VRAM {vram:.1f} GiB)")
    backend = VLLMBackend(model, cfg) if backend_name == "vllm" else TransformersBackend(model, cfg, vram)

    t0 = time.time()
    for start in range(0, len(todo), args.chunk):
        idxs = todo[start:start + args.chunk]
        samples = [build_sample(date, i, cfg, failures, retro, backend.count_tokens) for i in idxs]
        results = backend.generate(samples)
        for s, r in zip(samples, results):
            row = {
                "id": s["id"], "date": date, "index": s["index"], "seed": s["seed"],
                "source_ids": s["source_ids"], "claim_source_id": s["claim_source_id"],
                "model": model, "backend": backend_name,
                "prompt_tokens": backend.count_tokens(s["prompt"]),
                "output": r["output"], "n_tokens": r["n_tokens"], "mean_logprob": r["mean_logprob"],
                "finish_reason": r["finish_reason"],
            }
            append_jsonl(rf, row)
        log(f"[run] {start + len(idxs)}/{len(todo)} written ({time.time() - t0:.0f}s)")
    log(f"[run] complete: {rf}")


if __name__ == "__main__":
    main()
