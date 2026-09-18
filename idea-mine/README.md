# idea-mine

A local pipeline that uses a **base** (non-instruct) language model as a high-temperature idea
recombinator over a corpus of failed patents and vintage technical forum posts, in the domain
of control systems and measurement.

```
failed patents (HUPD, rejected)  ─┐
                                   ├─▶ raw prompt ─▶ Qwen2.5-7B (base, T=1.2, min_p=0.05) ─▶ runs/
vintage Usenet posts (UTZOO)     ─┘                                                        │
                                                                                            ▼
        digests/YYYY-MM-DD.md  ◀─  top 30 by novelty  ◀─  logprob band + nearest-neighbour ─┘
                                                            distance (bge-small-en-v1.5)
```

No chat template, no instructions, no persona, no LLM-as-judge. The digest shows the model's
raw output verbatim.

## Layout

| path | what |
| --- | --- |
| `config.yaml` | every knob (dataset filters, model, sampling, scoring, digest size) |
| `01_pull_data.py` | builds `data/failures.jsonl` and `data/retro.jsonl` (one-time, resumable) |
| `02_generate.py` | 300 raw completions per day → `runs/YYYY-MM-DD.jsonl` (resumable) |
| `03_score.py` | novelty + logprob band → `scores/YYYY-MM-DD.jsonl` |
| `04_digest.py` | top 30 verbatim → `digests/YYYY-MM-DD.md` |
| `nightly.sh` | runs steps 2–4 for today |
| `selftest.py` | offline check of the whole pipeline with synthetic data and a tiny model |

## Install (once)

Requires Linux, Python 3.10+, an NVIDIA GPU with a recent driver, and [uv](https://docs.astral.sh/uv/)
(`curl -LsSf https://astral.sh/uv/install.sh | sh` if you don't have it).

```bash
cd idea-mine
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install vllm          # optional but recommended for GPUs with >= 20 GB VRAM; skip on smaller cards
uv pip install bitsandbytes  # only needed for GPUs with < 18 GB VRAM (4-bit loading)
python selftest.py           # ~1 minute, no network needed; must end with "[selftest] ALL OK"
```

HuggingFace access: the HUPD and UTZOO datasets and the Qwen weights are public. Run
`huggingface-cli login` (or `export HF_TOKEN=...`) only if you want the optional gated Echo88
magazine text; without access it is skipped silently.

## Run the full pipeline once

```bash
source .venv/bin/activate
python 01_pull_data.py       # one-time corpus build; streams several GB from HuggingFace, resumable
python 02_generate.py        # 300 samples for today
python 03_score.py
python 04_digest.py
cat digests/$(date +%F).md
```

Step 1 prints the decision-label counts and newsgroup counts it found so you can see what the
filters matched. If the HUPD filter on G05B/G05D/G01D yields fewer than 1,500 rows it widens
to all of G05 and G01 automatically.

Model choice: `02_generate.py` scans `~/.cache/huggingface/hub`, `~/models`, `/models`,
`/opt/models`, `/data/models` for an existing Qwen **base** checkpoint (Instruct/Chat variants
are ignored and refused). If none is found it downloads `Qwen/Qwen2.5-7B` (~15 GB). Set
`model.path` in `config.yaml` to force a specific directory.

Backend choice: vLLM (bf16) if it is installed and the GPU has >= 20 GB VRAM; otherwise
transformers in bf16 (>= 18 GB) or 4-bit via bitsandbytes (smaller cards). Override with
`generation.backend`.

## Nightly routine

Two commands:

```bash
# 1. run it (generate → score → digest for today; safe to rerun, it only fills in what is missing)
/path/to/idea-mine/nightly.sh

# 2. schedule it at 02:00 every night (run once from inside the idea-mine directory)
(crontab -l 2>/dev/null; echo "0 2 * * * $PWD/nightly.sh >> $PWD/logs/nightly.log 2>&1") | crontab -
```

Each morning read `digests/YYYY-MM-DD.md`.

## Idempotence and resuming

- `01_pull_data.py` writes one finished file per source/year under `data/`; rerunning skips
  finished parts and only rebuilds the merged `failures.jsonl` / `retro.jsonl`. `--force` redoes
  a source, `--only hupd|utzoo|echo88` runs one source.
- `02_generate.py` derives sample *i* of a date from a stable seed (sources and sampling seed),
  appends each sample as soon as it is generated, and on rerun only generates the missing
  indices. `--date YYYY-MM-DD` and `--n N` override the date and count.
- `03_score.py` caches corpus embeddings in `cache/` keyed by corpus content and re-embeds only
  when the corpus changes. Scores and digests are rewritten in full (cheap).

## What is logged per sample (`runs/*.jsonl`)

`id`, `date`, `index`, `seed`, `source_ids` (3 patents + 1 snippet), `claim_source_id` (which
patent had its first claim included), `model`, `backend`, `prompt_tokens`, `output` (raw text),
`n_tokens`, `mean_logprob` (mean raw log-probability of the sampled tokens), `finish_reason`.

## Scoring rules

- Novelty = 1 − cosine similarity to the **nearest** corpus text (patent title+abstract or
  retro snippet), using `BAAI/bge-small-en-v1.5`.
- Quality band: rank by `mean_logprob`; drop the top 10% (most predictable, cliché) and the
  bottom 10% (least predictable, word salad). Empty outputs are dropped. Nothing else is filtered.
- Survivors ranked by novelty descending; the digest is the top 30.
