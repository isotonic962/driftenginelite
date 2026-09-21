#!/usr/bin/env bash
# Run N consecutive autonomous Claude Code sessions against AUTONOMOUS_TASK_v2.md.
# Usage: tmux new -s auto; bash scripts/autonomous_loop.sh [N]
# Stop it at any time with: touch /workspace/STOP  (takes effect at the next session boundary;
# the charter also makes a starting session exit immediately when the file exists).
set -uo pipefail
cd "$(dirname "$0")/.."

SESSIONS="${1:-4}"
MAX_TURNS="${MAX_TURNS:-150}"
SESSION_SECONDS="${SESSION_SECONDS:-10800}"   # 3 h, matches the charter
BRANCH="pod-autonomous"

git fetch origin && git checkout -B "$BRANCH" "origin/${BASE_BRANCH:-main}"
fails=0
for i in $(seq 1 "$SESSIONS"); do
  [ -f /workspace/STOP ] && { echo "STOP file present, ending loop"; break; }
  echo "=== autonomous session $i/$SESSIONS $(date -u +%FT%TZ) ==="
  git pull --ff-only origin "$BRANCH" 2>/dev/null || true
  timeout "$SESSION_SECONDS" claude -p "$(cat AUTONOMOUS_TASK_v2.md)" \
      --max-turns "$MAX_TURNS" --dangerously-skip-permissions \
      |& tee "logs/autonomous_$(date -u +%Y%m%dT%H%M).log"
  rc=$?
  # Belt and braces: the charter says the session pushes, but a session that
  # died mid-run must not strand its work on the pod.
  git add -A logs eval docs 2>/dev/null; git commit -m "autonomous session $i (exit=$rc)" 2>/dev/null
  git push -u origin "$BRANCH" || { sleep 4; git push -u origin "$BRANCH"; }
  if [ "$rc" -ne 0 ]; then fails=$((fails+1)); else fails=0; fi
  [ "$fails" -ge 2 ] && { echo "two consecutive failed sessions, stopping"; break; }
done
echo "loop done $(date -u +%FT%TZ)"
