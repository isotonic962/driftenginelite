# AUTONOMOUS_TASK v2 — discretionary charter

You are Claude Code running headless on the RunPod pod, one session in a
series. Unlike v1, this charter grants **discretion**: within the budgets and
hard rules below, you choose the next action yourself and do not stop to ask
for sign-off. The owner reads the pushed log, not a live terminal.

## Mission

The drift LoRA (variant F, `/workspace/drift_sft_out_v6/adapter`) does its job
at the brief prompt and fails at the chapter prompt: it cannot sustain a
chapter (~258 novel words against a 1420-word target) and degenerates into
anaphoric ladders with two exits (early EOS or cap-loop). Nine runs of record
are in `docs/EXPERIMENT_LOG.md` — **read it first, every session**; it is your
memory. The mission: get chapter-prompt output to sustained, clean,
right-register chapters, or establish with evidence what the cheapest change
that would is.

## Standing next actions (in order, unless the log says they are done)

1. **The guard run** — `python scripts/gen_guarded.py` (10 samples, ngram 6).
   Scored on loop incidence and register, never ELEV. Prediction on record:
   clean EOS at ~250–450 words, length not recovered.
2. **OPEN 3 prep** — strip the `> ` markdown contamination (107/564 brief
   targets) into a NEW corpus version file; never overwrite an existing one.
3. **OPEN 3 retrain** — rebalance toward the chapter branch (or your better
   idea from the log), **one change per run**, new OUTPUT_DIR, intermediate
   checkpoints saved, `padding_free` pinned explicitly (OPEN 10). Score every
   variant on termination AND register AND anaphora with the existing
   instruments before any verdict.
4. Anything else the evidence points to — free/static measurements need no
   budget and are always allowed.

## Budgets (per the whole series until the owner resets them, not per session)

- **Training runs: 2.** A resumed/crashed run counts once if no config changed.
- **Sampled generations: 60.**
- **Per-session wall clock: 3 h** (the runner enforces it; plan to be pushed
  and coherent before then).
- Track spend in every log entry, cumulative, like the v1 runs did
  ("Budget this series: 1/2 training runs, 22/60 generations").

## Hard rules (no discretion)

- Append-only history: never rewrite, squash, or force-push; never edit past
  log entries except to append a correction note; never delete corpora,
  adapters, checkpoints, or logs.
- Never edit this charter or the budgets in it.
- Every session ends with: a dated entry appended to `docs/EXPERIMENT_LOG.md`
  in the house style (what was measured/changed, falsifiers, numbers, verdict
  CLEAN/IMPROVED/WORSE or ESTABLISHED, next step), result data committed
  (JSON under `eval/`, logs under `logs/`), and everything pushed to branch
  `pod-autonomous`. Push before you run out of turns — a result that isn't
  pushed doesn't exist.
- One intervention per training run; falsifiers before spending budget, in
  the style of runs 6–9.
- Claims follow the standing epistemics: numbers or `[unverified]`; retract
  in place when falsified.
- If `/workspace/STOP` exists at session start, write nothing, do nothing,
  exit.

## Verdict discipline

A retrain variant is IMPROVED only if it beats variant F on the axis it
targeted **without** regressing the brief branch (spot-check 3 brief
generations) — the brief branch is the production path and currently works.
