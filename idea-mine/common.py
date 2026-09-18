"""Shared helpers for the idea-mine pipeline scripts."""

import datetime as _dt
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def log(*args):
    print(*args, file=sys.stderr, flush=True)


def load_config(config_path):
    config_path = Path(config_path).resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = config_path.parent
    resolved = {}
    for key, rel in cfg["paths"].items():
        p = Path(os.path.expanduser(str(rel)))
        resolved[key] = p if p.is_absolute() else (base / p)
    cfg["paths"] = resolved
    cfg["_base"] = base
    return cfg


def today_str():
    return _dt.date.today().isoformat()


def parse_date(s):
    return _dt.date.fromisoformat(s).isoformat()


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def write_jsonl_atomic(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def write_text_atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def stable_seed(*parts):
    """Deterministic 31-bit seed from arbitrary string parts."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def sha1_short(text, n=12):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def compile_keywords(patterns):
    return [re.compile(p, re.IGNORECASE) for p in patterns]


def keyword_hits(text, compiled):
    """Number of distinct keyword regexes that match the text."""
    return sum(1 for rx in compiled if rx.search(text))


def word_count(text):
    return len(text.split())


def truncate_words(text, max_words):
    """Cut text to at most max_words, preferring a sentence or paragraph boundary."""
    words = text.split()
    if len(words) <= max_words:
        return text
    cut = " ".join(words[:max_words])
    # prefer to end at the last sentence terminator in the second half of the cut
    m = None
    for m in re.finditer(r"[.!?](\s|$)", cut):
        pass
    if m and m.end() > len(cut) * 0.5:
        cut = cut[: m.end()].rstrip()
    return cut


def run_file(cfg, date):
    return cfg["paths"]["runs"] / f"{date}.jsonl"


def score_file(cfg, date):
    return cfg["paths"]["scores"] / f"{date}.jsonl"


def digest_file(cfg, date):
    return cfg["paths"]["digests"] / f"{date}.md"


def failures_file(cfg):
    return cfg["paths"]["data"] / "failures.jsonl"


def retro_file(cfg):
    return cfg["paths"]["data"] / "retro.jsonl"


def hf_token():
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return os.environ.get("HF_TOKEN")
