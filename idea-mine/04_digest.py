#!/usr/bin/env python
"""Step 4: write digests/YYYY-MM-DD.md.

Top N (config digest.top_n) surviving samples by novelty. Per entry: the generated text
verbatim, novelty score, seed, then the four source references as one line each. Nothing else.
"""

import argparse
import re

from common import (
    HERE,
    digest_file,
    failures_file,
    load_config,
    log,
    parse_date,
    read_jsonl,
    retro_file,
    run_file,
    score_file,
    today_str,
    write_text_atomic,
)


def fence_for(text):
    longest = max((len(m.group(0)) for m in re.finditer(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def source_line(sid, failures, retro):
    if sid in failures:
        f = failures[sid]
        return f"- {sid} — {f['title']} ({f.get('cpc') or f.get('ipc') or ''}, {f.get('filing_year', '')})"
    if sid in retro:
        r = retro[sid]
        where = ", ".join(x for x in (r.get("group") or r.get("source"), str(r.get("date") or "")[:16]) if x)
        return f"- {sid} — {r.get('subject') or '(no subject)'} ({where})"
    return f"- {sid}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--date", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    date = parse_date(args.date) if args.date else today_str()

    scores = [r for r in read_jsonl(score_file(cfg, date)) if r.get("band") == "keep"]
    if not scores:
        raise SystemExit(f"no scores for {date}; run 03_score.py first")
    samples = {s["id"]: s for s in read_jsonl(run_file(cfg, date))}
    failures = {f["id"]: f for f in read_jsonl(failures_file(cfg))}
    retro = {r["id"]: r for r in read_jsonl(retro_file(cfg))}

    top = sorted(scores, key=lambda r: r["rank"])[: cfg["digest"]["top_n"]]
    parts = [f"# idea-mine digest {date}\n"]
    for r in top:
        text = samples[r["id"]]["output"]
        fence = fence_for(text)
        body = text if text.endswith("\n") else text + "\n"
        parts.append(f"## {r['rank']}\n\n{fence}\n{body}{fence}\n\n"
                     f"novelty: {r['novelty']:.4f} · seed: {r['seed']}\n\n"
                     + "\n".join(source_line(sid, failures, retro) for sid in r["source_ids"]) + "\n")
    out = digest_file(cfg, date)
    write_text_atomic(out, "\n".join(parts))
    log(f"[digest] wrote {out} ({len(top)} entries)")


if __name__ == "__main__":
    main()
