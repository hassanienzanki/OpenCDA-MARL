#!/bin/bash
# Poll matrix progress on this machine and (optionally) one remote worker,
# every POLL_SEC, until every expected cell has metrics_history_ep$TARGET_EP.pkl
# or MAX_WAIT_SEC elapses.
#
# Everything is configured through the environment so the script carries no
# machine-specific paths:
#
#   LOCAL_DIR      metrics-history root here   (default: artifacts/metrics_history/matrix)
#   EXPECTED_LOCAL cells expected here         (default: 0 = "don't wait on local")
#   REMOTE_HOST    ssh target of a second worker, empty to skip
#   REMOTE_DIR     metrics-history root there
#   EXPECTED_REMOTE cells expected there       (default: 0)
#   TARGET_EP      episode number that marks a finished cell (default: 50)
#   POLL_SEC       seconds between polls       (default: 1800)
#   MAX_WAIT_SEC   safety timeout in seconds   (default: 72000 = 20 h)
#
# Usage:
#   EXPECTED_LOCAL=5 REMOTE_HOST=worker2 REMOTE_DIR=/path/artifacts/metrics_history/matrix \
#   EXPECTED_REMOTE=6 bash scripts/experiments/monitor_matrix.sh
#
# Exit codes: 0 = everything finished, 2 = timed out (progress printed).

set -u

LOCAL_DIR="${LOCAL_DIR:-artifacts/metrics_history/matrix}"
EXPECTED_LOCAL="${EXPECTED_LOCAL:-0}"
REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_DIR="${REMOTE_DIR:-$LOCAL_DIR}"
EXPECTED_REMOTE="${EXPECTED_REMOTE:-0}"
TARGET_EP="${TARGET_EP:-50}"
POLL_SEC="${POLL_SEC:-1800}"
MAX_WAIT_SEC="${MAX_WAIT_SEC:-72000}"

start_epoch=$(date +%s)
iter=0

while true; do
  iter=$((iter+1))
  ts=$(date "+%Y-%m-%d %H:%M:%S")
  elapsed=$(( $(date +%s) - start_epoch ))

  local_done=$(find "$LOCAL_DIR" -maxdepth 2 -name "metrics_history_ep${TARGET_EP}.pkl" 2>/dev/null | wc -l)
  local_dirs=$(find "$LOCAL_DIR" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)

  if [ -n "$REMOTE_HOST" ]; then
    if remote_check=$(ssh -o ConnectTimeout=10 -o BatchMode=yes "$REMOTE_HOST" \
        "find $REMOTE_DIR -maxdepth 2 -name 'metrics_history_ep${TARGET_EP}.pkl' 2>/dev/null | wc -l; \
         find $REMOTE_DIR -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l" 2>/dev/null); then
      remote_done=$(echo "$remote_check" | sed -n '1p')
      remote_dirs=$(echo "$remote_check" | sed -n '2p')
    else
      remote_done="?"
      remote_dirs="?"
    fi
    echo "[iter $iter $ts (+${elapsed}s)] local: ${local_done}/${EXPECTED_LOCAL} done (${local_dirs} dirs) | ${REMOTE_HOST}: ${remote_done}/${EXPECTED_REMOTE} done (${remote_dirs} dirs)"
  else
    remote_done="$EXPECTED_REMOTE"
    echo "[iter $iter $ts (+${elapsed}s)] local: ${local_done}/${EXPECTED_LOCAL} done (${local_dirs} dirs)"
  fi

  if [ "$local_done" -ge "$EXPECTED_LOCAL" ] 2>/dev/null && \
     [ "$remote_done" = "$EXPECTED_REMOTE" ] || \
     { [ "$remote_done" != "?" ] && [ "$remote_done" -ge "$EXPECTED_REMOTE" ] 2>/dev/null && \
       [ "$local_done" -ge "$EXPECTED_LOCAL" ] 2>/dev/null; }; then
    echo "ALL EXPECTED CELLS COMPLETE at $ts (took ${elapsed}s = $((elapsed/3600))h$((elapsed%3600/60))m)"
    exit 0
  fi

  if [ "$elapsed" -ge "$MAX_WAIT_SEC" ]; then
    echo "TIMEOUT after ${elapsed}s — exiting with current progress (local=${local_done}/${EXPECTED_LOCAL}, remote=${remote_done}/${EXPECTED_REMOTE})"
    exit 2
  fi

  sleep "$POLL_SEC"
done
