#!/usr/bin/env python
"""Step 1: build the corpus.

  data/failures.jsonl  negative-outcome patents (HUPD) in control systems / measurement
  data/retro.jsonl     vintage technical forum posts (UTZOO Usenet archive) + optional Echo88 magazine text

Idempotent and resumable: every source writes per-source finished files under data/
(hupd/<year>.jsonl, retro_utzoo.jsonl, retro_echo88.jsonl). A rerun skips anything that is
already finished and only rebuilds the merged files. Use --force to redo a source.
"""

import argparse
import io
import json
import random
import re
import sys
import tarfile
import time
from collections import Counter
from itertools import islice
from pathlib import Path

from common import (
    HERE,
    compile_keywords,
    failures_file,
    hf_token,
    keyword_hits,
    log,
    read_jsonl,
    retro_file,
    sha1_short,
    truncate_words,
    word_count,
    write_jsonl_atomic,
)

# ----------------------------------------------------------------------------- HUPD


def first_independent_claim(claims_text):
    """Return the text of claim 1 (everything up to the start of claim 2)."""
    if not claims_text:
        return ""
    text = claims_text.strip()
    start = re.search(r"(?:^|\n)\s*1\s*[.)]\s+", text)
    if not start:
        return truncate_words(text, 350)
    body = text[start.end():]
    end = re.search(r"\n\s*2\s*[.)]\s+", body)
    if not end:
        # single-line claims blob: claim 2 follows on the same line
        end = re.search(r"(?<=[.;])\s+2\s*[.)]\s+(?=[A-Z])", body)
    claim = body[: end.start()] if end else body
    claim = re.sub(r"\s+", " ", claim).strip()
    return truncate_words(claim, 350)


def _pick_col(columns, candidates):
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def iter_tar_files(fileobj):
    """Yield (member_name, bytes) for every regular file in a (possibly nested) tar stream."""
    with tarfile.open(fileobj=fileobj, mode="r|*") as tf:
        for member in tf:
            if not member.isfile():
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            name = member.name
            if name.endswith((".tar.gz", ".tgz", ".tar")):
                yield from iter_tar_files(f)
            else:
                yield name, f.read()


def open_hf_stream(repo, filename, token):
    import requests
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers

    url = hf_hub_url(repo, filename, repo_type="dataset")
    headers = build_hf_headers(token=token)
    resp = requests.get(url, headers=headers, stream=True, timeout=120, allow_redirects=True)
    resp.raise_for_status()
    resp.raw.decode_content = False
    size = int(resp.headers.get("Content-Length") or 0)
    return resp, size


def hupd_record(j, meta):
    title = (j.get("title") or "").strip()
    abstract = re.sub(r"\s+", " ", (j.get("abstract") or "")).strip()
    if not title or not abstract:
        return None
    appno = str(j.get("application_number") or meta["application_number"])
    return {
        "id": f"hupd:{appno}",
        "source": "hupd",
        "application_number": appno,
        "title": title,
        "abstract": abstract,
        "first_claim": first_independent_claim(j.get("claims") or ""),
        "background": re.sub(r"[ \t]+", " ", (j.get("background") or "")).strip()[: meta["background_max_chars"]],
        "cpc": j.get("main_cpc_label") or meta.get("cpc") or "",
        "ipc": j.get("main_ipcr_label") or j.get("main_ipc_label") or meta.get("ipc") or "",
        "decision": meta.get("decision") or j.get("decision") or "",
        "filing_year": meta.get("filing_year"),
    }


def pull_hupd(cfg, force):
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download

    hcfg = cfg["data"]["hupd"]
    data_dir = cfg["paths"]["data"]
    hdir = data_dir / "hupd"
    hdir.mkdir(parents=True, exist_ok=True)
    out = failures_file(cfg)
    selection_path = hdir / "selection.json"
    token = hf_token()
    repo = hcfg["repo"]

    api = HfApi(token=token)
    files = api.list_repo_files(repo, repo_type="dataset")
    year_files = {}
    for f in files:
        m = re.search(r"(?:^|/)(\d{4})\.tar\.gz$", f)
        if m:
            year_files[int(m.group(1))] = f
    all_years_file = next((f for f in files if re.search(r"all[-_]years\.tar(\.gz)?$", f)), None)
    feathers = sorted(f for f in files if f.endswith(".feather"))
    log(f"[hupd] repo files: {len(files)}; per-year tarballs: {sorted(year_files)}; "
        f"all-years: {all_years_file}; metadata: {feathers}")

    # ---- selection from metadata (cheap) --------------------------------------------------
    if selection_path.exists() and not force:
        selection = json.loads(selection_path.read_text())
        log(f"[hupd] reusing selection.json ({sum(len(v) for v in selection['wanted'].values())} rows, "
            f"prefixes={selection['prefixes']})")
    else:
        if not feathers:
            raise SystemExit("[hupd] no metadata .feather found in the repo; cannot pre-filter. "
                             "Inspect the dataset layout and adapt pull_hupd().")
        meta_local = hf_hub_download(repo, feathers[-1], repo_type="dataset", token=token)
        df = pd.read_feather(meta_local)
        log(f"[hupd] metadata rows={len(df)} columns={list(df.columns)}")
        dcol = _pick_col(df.columns, ["decision"])
        ccol = _pick_col(df.columns, ["main_cpc_label", "cpc_label", "main_cpc"])
        icol = _pick_col(df.columns, ["main_ipcr_label", "main_ipc_label", "ipc_label", "ipcr_label"])
        acol = _pick_col(df.columns, ["application_number", "app_number", "application_id", "app_id"])
        fcol = _pick_col(df.columns, ["filing_date", "date_filed", "filing_year"])
        if not (dcol and acol and (ccol or icol) and fcol):
            raise SystemExit(f"[hupd] metadata columns not recognised: {list(df.columns)}")

        dec = df[dcol].astype(str).str.upper()
        log("[hupd] decision label counts:\n" + dec.value_counts().to_string())
        pats = [p.upper() for p in hcfg["negative_decision_patterns"]]
        neg = dec.apply(lambda s: any(p in s for p in pats))
        df = df[neg].copy()
        log(f"[hupd] negative-outcome rows: {len(df)}")

        code = df[ccol].astype(str).str.strip().str.upper() if ccol else pd.Series("", index=df.index)
        if icol:
            ipc = df[icol].astype(str).str.strip().str.upper()
            code = code.where(~code.isin(["", "NAN", "NONE"]), ipc)
        years = pd.to_datetime(df[fcol], errors="coerce").dt.year
        y0, y1 = hcfg["years"]
        in_years = years.between(y0, y1)

        def select(prefixes):
            return code.str.startswith(tuple(p.upper() for p in prefixes)) & in_years

        prefixes = list(hcfg["cpc_prefixes"])
        mask = select(prefixes)
        log(f"[hupd] rows with main code in {prefixes}: {int(mask.sum())}")
        if int(mask.sum()) < hcfg["widen_below"]:
            prefixes = list(hcfg["widened_cpc_prefixes"])
            mask = select(prefixes)
            log(f"[hupd] under {hcfg['widen_below']}; widened to {prefixes}: {int(mask.sum())}")
        sel = df[mask]
        if len(sel) > hcfg["max_rows"]:
            sel = sel.sample(n=hcfg["max_rows"], random_state=hcfg["seed"])
            log(f"[hupd] capped to max_rows={hcfg['max_rows']}")
        wanted = {}
        for appno, year, decision, c in zip(sel[acol].astype(str), years[sel.index], sel[dcol].astype(str),
                                            code[sel.index]):
            wanted.setdefault(str(int(year)), {})[appno] = {"decision": decision, "cpc": c}
        selection = {"prefixes": prefixes, "wanted": wanted}
        selection_path.write_text(json.dumps(selection))
        log(f"[hupd] selected {len(sel)} rows across years {sorted(wanted)}")

    # ---- per-year streaming extraction (resumable per year) --------------------------------
    wanted = selection["wanted"]
    pending = [y for y in sorted(wanted) if force or not (hdir / f"{y}.jsonl").exists()]
    log(f"[hupd] years to fetch: {pending} (already done: {sorted(set(wanted) - set(pending))})")

    def extract_from_stream(fileobj, wanted_by_year, label):
        got = {y: [] for y in wanted_by_year}
        n = 0
        t0 = time.time()
        for name, blob in iter_tar_files(fileobj):
            n += 1
            if n % 50000 == 0:
                log(f"[hupd] {label}: {n} members scanned, {sum(len(v) for v in got.values())} kept, "
                    f"{time.time() - t0:.0f}s")
            if not name.endswith(".json"):
                continue
            stem = Path(name).stem
            for y, apps in wanted_by_year.items():
                if stem in apps:
                    try:
                        j = json.loads(blob.decode("utf-8", errors="replace"))
                    except json.JSONDecodeError:
                        break
                    meta = dict(apps[stem])
                    meta.update(application_number=stem, filing_year=int(y),
                                background_max_chars=hcfg["background_max_chars"])
                    rec = hupd_record(j, meta)
                    if rec:
                        got[y].append(rec)
                    break
        return got

    def fetch_with_retries(filename, wanted_by_year, label):
        for attempt in range(1, 4):
            try:
                resp, size = open_hf_stream(repo, filename, token)
                log(f"[hupd] streaming {filename} ({size / 2**30:.2f} GiB) for {label}")
                with resp:
                    return extract_from_stream(resp.raw, wanted_by_year, label)
            except Exception as e:  # network hiccup: restart this file
                log(f"[hupd] attempt {attempt} for {filename} failed: {e!r}")
                time.sleep(10 * attempt)
        raise SystemExit(f"[hupd] giving up on {filename}")

    if pending:
        if year_files:
            for y in pending:
                if int(y) not in year_files:
                    log(f"[hupd] no tarball for year {y}; skipping")
                    write_jsonl_atomic(hdir / f"{y}.jsonl", [])
                    continue
                got = fetch_with_retries(year_files[int(y)], {y: wanted[y]}, f"year {y}")
                write_jsonl_atomic(hdir / f"{y}.jsonl", got[y])
                log(f"[hupd] year {y}: kept {len(got[y])} / {len(wanted[y])} wanted")
        elif all_years_file:
            got = fetch_with_retries(all_years_file, {y: wanted[y] for y in pending}, "all years")
            for y in pending:
                write_jsonl_atomic(hdir / f"{y}.jsonl", got[y])
                log(f"[hupd] year {y}: kept {len(got[y])} / {len(wanted[y])} wanted")
        else:
            raise SystemExit("[hupd] no per-year tarballs or all-years tarball found in the repo")

    rows = []
    for y in sorted(wanted):
        rows.extend(read_jsonl(hdir / f"{y}.jsonl"))
    write_jsonl_atomic(out, rows)
    log(f"[hupd] wrote {out} with {len(rows)} failures")
    return rows


# ----------------------------------------------------------------------------- generic HF row iteration


def iter_hf_rows(repo, token, split_hint=None):
    """Stream rows of a HuggingFace dataset as dicts, whatever its file layout."""
    from datasets import load_dataset

    try:
        ds = load_dataset(repo, streaming=True, token=token)
        if hasattr(ds, "keys"):
            names = list(ds.keys())
            name = split_hint if split_hint in names else names[0]
            log(f"[{repo}] streaming split '{name}' of {names}")
            ds = ds[name]
        for row in ds:
            yield row
        return
    except Exception as e:
        log(f"[{repo}] datasets streaming failed ({e!r}); falling back to raw file iteration")

    from huggingface_hub import HfApi, hf_hub_download

    files = HfApi(token=token).list_repo_files(repo, repo_type="dataset")
    data_files = [f for f in files if re.search(r"\.(parquet|jsonl|json|txt|csv)(\.gz)?$", f)]
    log(f"[{repo}] raw files: {len(data_files)}")
    for f in data_files:
        local = hf_hub_download(repo, f, repo_type="dataset", token=token)
        if f.endswith(".parquet"):
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(local)
            for batch in pf.iter_batches(batch_size=1024):
                for row in batch.to_pylist():
                    yield row
        elif re.search(r"\.jsonl?(\.gz)?$", f):
            import gzip

            opener = gzip.open if f.endswith(".gz") else open
            with opener(local, "rt", encoding="utf-8", errors="replace") as fh:
                if f.endswith(".json") or f.endswith(".json.gz"):
                    data = json.load(fh)
                    for row in (data if isinstance(data, list) else data.get("data", [])):
                        yield row
                else:
                    for line in fh:
                        line = line.strip()
                        if line:
                            yield json.loads(line)
        elif f.endswith(".csv"):
            import csv

            with open(local, newline="", encoding="utf-8", errors="replace") as fh:
                for row in csv.DictReader(fh):
                    yield row
        else:
            with open(local, encoding="utf-8", errors="replace") as fh:
                yield {"text": fh.read(), "file": f}


def detect_columns(sample_rows):
    """Guess text / newsgroup / subject / date columns from a sample of rows."""
    if not sample_rows:
        return {}
    keys = list(sample_rows[0].keys())
    str_lens = {}
    for k in keys:
        vals = [r.get(k) for r in sample_rows if isinstance(r.get(k), str)]
        if vals:
            str_lens[k] = sum(len(v) for v in vals) / len(vals)
    text_col = None
    for k in keys:  # prefer conventional names when they are long-ish
        if k.lower() in ("text", "body", "content", "message", "article", "post") and str_lens.get(k, 0) > 50:
            text_col = k
            break
    if text_col is None and str_lens:
        text_col = max(str_lens, key=str_lens.get)
    group_col = next((k for k in keys if re.search(r"newsgroup|group", k, re.I)), None)
    subject_col = next((k for k in keys if re.search(r"subject|title", k, re.I)), None)
    date_col = next((k for k in keys if re.search(r"^date|posted|timestamp", k, re.I)), None)
    return {"text": text_col, "group": group_col, "subject": subject_col, "date": date_col}


# ----------------------------------------------------------------------------- Usenet cleaning

HEADER_LINE = re.compile(r"^[A-Za-z][A-Za-z0-9-]*:(\s|$)")
ATTRIBUTION = re.compile(r"^(In article|In <|.*\b(writes|wrote|said)\s*:?\s*$|>)", re.I)
BANG_PATH = re.compile(r"\S+!\S+")
SIG_START = re.compile(r"^-{2,}\s*$")


def split_usenet_headers(text):
    """Split an article into (headers dict, body). Tolerates missing headers."""
    lines = text.replace("\r\n", "\n").split("\n")
    headers = {}
    i = 0
    last_key = None
    while i < len(lines):
        line = lines[i]
        if HEADER_LINE.match(line):
            key, _, val = line.partition(":")
            last_key = key.strip().lower()
            headers[last_key] = val.strip()
        elif line.startswith((" ", "\t")) and last_key:
            headers[last_key] += " " + line.strip()
        else:
            break
        i += 1
    if not headers:
        return {}, text
    if i < len(lines) and lines[i].strip() == "":
        i += 1
    return headers, "\n".join(lines[i:])


def clean_usenet_body(body):
    out = []
    for line in body.split("\n"):
        s = line.rstrip()
        if SIG_START.match(s):
            break
        if ATTRIBUTION.match(s.strip()):
            continue
        if BANG_PATH.search(s) and len(s) < 100 and " " not in s.strip():
            continue  # bang-path address lines
        if re.fullmatch(r"\s*[A-Za-z0-9._-]+@[A-Za-z0-9._-]+\s*", s):
            continue
        out.append(s)
    text = "\n".join(out)
    # drop trailing signature-like short lines (names, addresses, uucp paths)
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    while paras and (word_count(paras[-1]) < 8 or BANG_PATH.search(paras[-1]) or "@" in paras[-1]):
        paras.pop()
    text = "\n\n".join(re.sub(r"[ \t]+", " ", p) for p in paras)
    return text.strip()


def groups_of(row, cols, headers):
    val = None
    if cols.get("group") and row.get(cols["group"]):
        val = row[cols["group"]]
    elif headers.get("newsgroups"):
        val = headers["newsgroups"]
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return [str(g).strip().lower() for g in val]
    return [g.strip().lower() for g in re.split(r"[,\s]+", str(val)) if g.strip()]


def pull_utzoo(cfg, force):
    ucfg = cfg["data"]["utzoo"]
    out = cfg["paths"]["data"] / "retro_utzoo.jsonl"
    if out.exists() and not force:
        rows = read_jsonl(out)
        log(f"[utzoo] already done ({len(rows)} snippets); skip")
        return rows
    token = hf_token()
    repo = ucfg["repo"]
    kw = compile_keywords(cfg["data"]["keywords"])
    group_rx = [re.compile(p, re.I) for p in ucfg["group_patterns"]]

    rows_iter = iter_hf_rows(repo, token)
    sample = list(islice(rows_iter, 200))
    cols = detect_columns(sample)
    log(f"[utzoo] detected columns: {cols} (keys: {list(sample[0].keys()) if sample else None})")
    if not cols.get("text"):
        raise SystemExit("[utzoo] could not find a text column")

    def chain():
        yield from sample
        yield from rows_iter

    group_counter = Counter()
    tier_a, tier_b = [], []
    seen = set()
    n = 0
    t0 = time.time()
    min_hits = ucfg["min_keyword_hits"]
    any_group_info = False
    for row in chain():
        n += 1
        raw = row.get(cols["text"]) or ""
        if not isinstance(raw, str):
            continue
        headers, body = split_usenet_headers(raw)
        groups = groups_of(row, cols, headers)
        if groups:
            any_group_info = True
        if n <= ucfg["inspect_rows"]:
            group_counter.update(groups)
            if n == ucfg["inspect_rows"]:
                log(f"[utzoo] top newsgroups in first {n} rows:\n" +
                    "\n".join(f"   {g:40s} {c}" for g, c in group_counter.most_common(60)))
        if n % 100000 == 0:
            log(f"[utzoo] scanned {n} rows, candidates {len(tier_a)} (+{len(tier_b)} weak), {time.time() - t0:.0f}s")
        if groups and not any(rx.search(g) for g in groups for rx in group_rx):
            continue
        hits = keyword_hits(body, kw)
        need = min_hits if groups else min_hits + 1
        if hits < need - 1:
            continue
        text = clean_usenet_body(body)
        wc = word_count(text)
        if wc < ucfg["min_words"]:
            continue
        if wc > ucfg["max_words"]:
            text = truncate_words(text, ucfg["max_words"])
        h = sha1_short(text)
        if h in seen:
            continue
        seen.add(h)
        subject = (row.get(cols["subject"]) if cols.get("subject") else None) or headers.get("subject") or ""
        date = (row.get(cols["date"]) if cols.get("date") else None) or headers.get("date") or ""
        rec = {
            "id": f"utzoo:{h}",
            "source": "utzoo",
            "group": groups[0] if groups else "",
            "subject": re.sub(r"\s+", " ", str(subject)).strip()[:200],
            "date": str(date)[:40],
            "text": text,
            "word_count": word_count(text),
            "keyword_hits": hits,
        }
        (tier_a if hits >= need else tier_b).append(rec)
        if len(tier_a) >= ucfg["max_candidates"]:
            log(f"[utzoo] reached max_candidates after {n} rows")
            break
    if not any_group_info:
        log("[utzoo] WARNING: no newsgroup information found in the data; filtering was keyword-only")
    log(f"[utzoo] scanned {n} rows: {len(tier_a)} strong candidates, {len(tier_b)} weak")
    rng = random.Random(ucfg["seed"])
    chosen = list(tier_a)
    if len(chosen) < ucfg["min_snippets"]:
        log(f"[utzoo] fewer than min_snippets={ucfg['min_snippets']}; topping up from weaker matches")
        chosen.extend(tier_b)
    if len(chosen) > ucfg["target_snippets"]:
        chosen = rng.sample(chosen, ucfg["target_snippets"])
    chosen.sort(key=lambda r: r["id"])
    if len(chosen) < ucfg["min_snippets"]:
        log(f"[utzoo] WARNING: only {len(chosen)} snippets; consider loosening group_patterns/min_keyword_hits")
    write_jsonl_atomic(out, chosen)
    log(f"[utzoo] wrote {out} with {len(chosen)} snippets")
    return chosen


def pull_echo88(cfg, force):
    ecfg = cfg["data"]["echo88"]
    out = cfg["paths"]["data"] / "retro_echo88.jsonl"
    if not ecfg.get("enabled", True):
        return []
    if out.exists() and not force:
        rows = read_jsonl(out)
        log(f"[echo88] already done ({len(rows)} snippets); skip")
        return rows
    token = hf_token()
    repo = ecfg["repo"]
    try:
        from huggingface_hub import HfApi

        info = HfApi(token=token).dataset_info(repo)
        if getattr(info, "gated", False) and not token:
            raise PermissionError("gated, no token")
        # a real access check: listing files of a gated repo fails without access
        HfApi(token=token).list_repo_files(repo, repo_type="dataset")
    except Exception as e:
        log(f"[echo88] skipped (no access: {type(e).__name__})")
        write_jsonl_atomic(out, [])
        return []

    kw = compile_keywords(cfg["data"]["keywords"])
    rows_iter = iter_hf_rows(repo, token)
    sample = list(islice(rows_iter, 100))
    cols = detect_columns(sample)
    log(f"[echo88] detected columns: {cols}")
    if not cols.get("text"):
        log("[echo88] no text column; skipped")
        write_jsonl_atomic(out, [])
        return []

    def chain():
        yield from sample
        yield from rows_iter

    ucfg = cfg["data"]["utzoo"]
    chosen, seen = [], set()
    n = 0
    for row in chain():
        n += 1
        if n > ecfg["max_rows_scanned"] or len(chosen) >= ecfg["max_snippets"]:
            break
        raw = row.get(cols["text"])
        if not isinstance(raw, str) or keyword_hits(raw, kw) < ecfg["min_keyword_hits"]:
            continue
        # chunk into paragraphs windows of <= max_words
        paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
        buf, wc = [], 0
        chunks = []
        for p in paras:
            pw = word_count(p)
            if wc + pw > ucfg["max_words"] and buf:
                chunks.append("\n\n".join(buf))
                buf, wc = [], 0
            buf.append(p)
            wc += pw
        if buf:
            chunks.append("\n\n".join(buf))
        for chunk in chunks:
            if word_count(chunk) < ucfg["min_words"]:
                continue
            if keyword_hits(chunk, kw) < ecfg["min_keyword_hits"]:
                continue
            text = truncate_words(chunk, ucfg["max_words"])
            h = sha1_short(text)
            if h in seen:
                continue
            seen.add(h)
            title = (row.get(cols["subject"]) if cols.get("subject") else None) or text.split("\n")[0][:80]
            chosen.append({
                "id": f"echo88:{h}",
                "source": "echo88",
                "group": "",
                "subject": re.sub(r"\s+", " ", str(title)).strip()[:200],
                "date": str(row.get(cols["date"]) or "")[:40] if cols.get("date") else "",
                "text": text,
                "word_count": word_count(text),
                "keyword_hits": keyword_hits(text, kw),
            })
            if len(chosen) >= ecfg["max_snippets"]:
                break
    write_jsonl_atomic(out, chosen)
    log(f"[echo88] wrote {out} with {len(chosen)} snippets (scanned {n} rows)")
    return chosen


def merge_retro(cfg):
    data_dir = cfg["paths"]["data"]
    rows = read_jsonl(data_dir / "retro_utzoo.jsonl") + read_jsonl(data_dir / "retro_echo88.jsonl")
    write_jsonl_atomic(retro_file(cfg), rows)
    log(f"[retro] wrote {retro_file(cfg)} with {len(rows)} snippets")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--force", action="store_true", help="re-pull even if finished files exist")
    ap.add_argument("--only", choices=["hupd", "utzoo", "echo88"], help="run a single source")
    args = ap.parse_args()
    from common import load_config

    cfg = load_config(args.config)
    cfg["paths"]["data"].mkdir(parents=True, exist_ok=True)
    if args.only in (None, "hupd"):
        pull_hupd(cfg, args.force)
    if args.only in (None, "utzoo"):
        pull_utzoo(cfg, args.force)
    if args.only in (None, "echo88"):
        pull_echo88(cfg, args.force)
    merge_retro(cfg)
    log("[done] failures: %d  retro: %d" % (len(read_jsonl(failures_file(cfg))), len(read_jsonl(retro_file(cfg)))))


if __name__ == "__main__":
    main()
