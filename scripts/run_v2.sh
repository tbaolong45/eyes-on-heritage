#!/bin/bash
# usage: run_v2.sh <logname> <python args...>   -- same as run.sh for the reviewed-label workspace (HERITAGE_ROOT=~/heritage_v2)
cd ~/heritage
source .venv/bin/activate
export HERITAGE_ROOT=$HOME/heritage_v2 PYTHONHASHSEED=0 HF_HOME=$HOME/heritage/hf_cache OMP_NUM_THREADS=2
name=$1; shift
python -u "$@" > $HERITAGE_ROOT/logs/$name.log 2>&1
echo "EXIT $?" >> $HERITAGE_ROOT/logs/$name.log
