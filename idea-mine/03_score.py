#!/usr/bin/env python
"""Step 3: novelty scoring.

Corpus texts (failure title+abstract, retro snippets) are embedded once with the configured
sentence-embedding model and cached under cache/. Each generated sample is embedded and its
novelty = cosine distance to its NEAREST corpus neighbour.

Quality band: samples are ranked by mean token logprob; the most predictable trim_fraction
(default 10%) and the least predictable trim_fraction are dropped. Survivors are ranked by
novelty descending. Nothing else filters samples. Writes scores/YYYY-MM-DD.jsonl.
"""

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np

from common import (
    HERE,
    failures_file,
    load_config,
    log,
    parse_date,
    read_jsonl,
    retro_file,
    run_file,
    score_file,
    today_str,
    write_jsonl_atomic,
)


def corpus_texts(cfg):
    items = []
    for f in read_jsonl(failures_file(cfg)):
        items.append((f["id"], f"{f['title']}\n\n{f['abstract']}"))
    for r in read_jsonl(retro_file(cfg)):
        items.append((r["id"], r["text"]))
    return items


def load_embedder(cfg):
    from sentence_transformers import SentenceTransformer

    name = cfg["scoring"]["embedding_model"]
    log(f"[embed] loading {name}")
    return SentenceTransformer(name)


def embed(model, texts, batch_size=64):
    return np.asarray(
        model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=len(texts) > 500),
        dtype=np.float32,
    )


def corpus_embeddings(cfg, model):
    items = corpus_texts(cfg)
    if not items:
        raise SystemExit("empty corpus; run 01_pull_data.py first")
    h = hashlib.sha256()
    h.update(cfg["scoring"]["embedding_model"].encode())
    for i, t in items:
        h.update(i.encode())
        h.update(t.encode())
    digest = h.hexdigest()
    cache = cfg["paths"]["cache"] / "corpus_embeddings.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if str(z["hash"]) == digest:
            log(f"[embed] corpus cache hit ({len(z['ids'])} texts)")
            return list(z["ids"]), z["emb"]
    log(f"[embed] embedding corpus ({len(items)} texts) ...")
    emb = embed(model, [t for _, t in items])
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp.npz")
    np.savez(tmp, ids=np.array([i for i, _ in items]), emb=emb, hash=np.array(digest))
    tmp.replace(cache)
    return [i for i, _ in items], emb


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--date", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    date = parse_date(args.date) if args.date else today_str()

    samples = read_jsonl(run_file(cfg, date))
    if not samples:
        raise SystemExit(f"no samples in {run_file(cfg, date)}; run 02_generate.py first")
    model = load_embedder(cfg)
    ids, cemb = corpus_embeddings(cfg, model)

    valid = [s for s in samples if s["n_tokens"] > 0 and s["output"].strip()
             and isinstance(s["mean_logprob"], (int, float)) and math.isfinite(s["mean_logprob"])]
    empty = [s for s in samples if s not in valid]
    log(f"[score] {len(samples)} samples, {len(valid)} valid, {len(empty)} empty/invalid")

    semb = embed(model, [s["output"] for s in valid]) if valid else np.zeros((0, cemb.shape[1]), np.float32)
    sims = semb @ cemb.T
    nearest = sims.argmax(axis=1) if len(valid) else np.array([], int)
    novelty = 1.0 - sims.max(axis=1) if len(valid) else np.array([])

    frac = cfg["scoring"]["trim_fraction"]
    k = int(round(frac * len(valid)))
    order = sorted(range(len(valid)), key=lambda i: valid[i]["mean_logprob"])  # ascending: salad first
    band = {}
    for pos, i in enumerate(order):
        if pos < k:
            band[i] = "drop_low_logprob"
        elif pos >= len(order) - k:
            band[i] = "drop_high_logprob"
        else:
            band[i] = "keep"

    rows = []
    for i, s in enumerate(valid):
        rows.append({
            "id": s["id"], "seed": s["seed"], "source_ids": s["source_ids"],
            "novelty": float(novelty[i]), "nearest_id": str(ids[nearest[i]]),
            "mean_logprob": s["mean_logprob"], "n_tokens": s["n_tokens"], "band": band[i],
        })
    for s in empty:
        rows.append({"id": s["id"], "seed": s["seed"], "source_ids": s["source_ids"], "novelty": None,
                     "nearest_id": None, "mean_logprob": s["mean_logprob"], "n_tokens": s["n_tokens"],
                     "band": "drop_empty"})
    keep = sorted((r for r in rows if r["band"] == "keep"), key=lambda r: -r["novelty"])
    for rank, r in enumerate(keep, 1):
        r["rank"] = rank
    rows.sort(key=lambda r: (r.get("rank") is None, r.get("rank") or 0, r["id"]))
    out = score_file(cfg, date)
    write_jsonl_atomic(out, rows)
    log(f"[score] kept {len(keep)} of {len(valid)} (trimmed {k} at each end); wrote {out}")
    if keep:
        log(f"[score] novelty range among kept: {keep[-1]['novelty']:.3f} .. {keep[0]['novelty']:.3f}")


if __name__ == "__main__":
    main()
