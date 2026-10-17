#!/bin/bash
# usage: run.sh <logname> <script.py> [args...]  -- runs one pipeline step from the repo root, log in $HERITAGE_ROOT/logs/<logname>.log
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
[ -f .venv/bin/activate ] && source .venv/bin/activate
export DATA_DIR=${DATA_DIR:-/workspace}
export HERITAGE_ROOT=${HERITAGE_ROOT:-$DATA_DIR/heritage_run}
export HF_HOME=${HF_HOME:-$HERITAGE_ROOT/hf_cache} PYTHONHASHSEED=0 OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
mkdir -p "$HERITAGE_ROOT/logs"
name=$1; shift
python -u "$@" > "$HERITAGE_ROOT/logs/$name.log" 2>&1
status=$?
echo "EXIT $status" >> "$HERITAGE_ROOT/logs/$name.log"
exit $status
