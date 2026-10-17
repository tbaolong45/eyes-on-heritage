#!/bin/bash
# Whole pipeline: data -> tiles -> features -> embeddings -> members A, B, C, C3 (CV + leave-one-site-out) -> ensemble
# -> calibration -> nested audit -> final models.  Restartable: a step whose output exists is skipped.
# usage: bash scripts/run_all.sh            (inputs and output folder: see "Configure" in README.md)
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
export DATA_DIR=${DATA_DIR:-/workspace}
export HERITAGE_ROOT=${HERITAGE_ROOT:-$DATA_DIR/heritage_run}
R=scripts/run.sh; CD=$HERITAGE_ROOT/cache; REP=$HERITAGE_ROOT/reports

# ---- training labels: train_pairs (all labelled pairs), train_pairs_event (without the no_change pairs taken
#      before / after a change point's event) or years (the previous year_before / year_after rule) ----
PAIRS_FROM=${PAIRS_FROM:-train_pairs}
P="\"pairs_from\":\"$PAIRS_FROM\""
# ---- members (names start with their model kind: A_, B_, C_; the label source is part of the name) ----
L=$PAIRS_FROM
A=A_${L}_dinov2;  A_CFG="{\"agg\":\"top2\",\"dino\":\"dinov2-l_B1\",$P}"
B=B_${L}_dinov2;  B_CFG="{\"agg\":\"top2\",\"reverse\":false,\"emb\":\"dinov2-l_B1\",$P}"
C=C_${L}_dinov2;  C_CFG="{$P}"
C3=C_${L}_dinov3; C3_CFG="{\"bb\":\"dinov3s-l\",$P}"
SEEDS_AB=${SEEDS_AB:-0,1,2}        # CV seeds for A and B
SEEDS_C=${SEEDS_C:-0}              # CV seeds for C and C3 (each seed is ~10 fine-tunes)
FINAL_SEEDS=${FINAL_SEEDS:-0,1,2,3,4}
MEMBERS=${MEMBERS:-$A,$B,$C,$C3}
TAG=${TAG:-$PAIRS_FROM}

step() {  # step <name> <file proving it is done> <script> [args]
    local name=$1 done=$2; shift 2
    if [ -e "$done" ]; then echo "[skip] $name ($done exists)"; return; fi
    echo "[run ] $name   log: $HERITAGE_ROOT/logs/$name.log   $(date +%H:%M)"
    if ! bash $R "$name" "$@"; then echo "step $name failed:"; tail -25 "$HERITAGE_ROOT/logs/$name.log"; exit 1; fi
}
has() { [[ ",$MEMBERS," == *",$1,"* ]]; }

step p0     $CD/pair_labels.parquet            scripts/p0_data.py
step p1     $CD/quality.parquet                scripts/p1_tiles.py
step holes  $REP/.holes_ok                    scripts/check_holes_v2.py && touch $REP/.holes_ok
step featA  $CD/featA_tiles_pairs.parquet      scripts/p3_featA.py
step emb    $CD/dinopair_dinov2-l_B1.parquet   scripts/p4_embed.py dinov2-l B1
has $A  && step A  $CD/oof/$A.npz   scripts/exp_A.py $A  "$A_CFG"  $SEEDS_AB
has $B  && step B  $CD/oof/$B.npz   scripts/exp_B.py $B  "$B_CFG"  $SEEDS_AB
has $C  && step C  $CD/oof/$C.npz   scripts/exp_C.py $C  "$C_CFG"  $SEEDS_C
has $C3 && step C3 $CD/oof/$C3.npz  scripts/exp_C.py $C3 "$C3_CFG" $SEEDS_C
step ens    $REP/ens_$TAG.json                scripts/ensemble.py $MEMBERS --weights eq --out ens_$TAG
step calib  $REP/v2ens_$TAG.json              scripts/v2_ensemble.py $MEMBERS --tag $TAG
step audit  $REP/audit_ens_$TAG.json              scripts/audit_ensemble.py $MEMBERS --tag ens_$TAG
step final  $HERITAGE_ROOT/models/final/.done_$TAG scripts/train_final.py $REP/ens_$TAG.json --seeds $FINAL_SEEDS --calib $REP/v2ens_$TAG.json \
    && touch $HERITAGE_ROOT/models/final/.done_$TAG

echo; echo "===== nested validation (reports/audit_ens_$TAG.md) ====="; cat $REP/audit_ens_$TAG.md
echo; echo "final models: $HERITAGE_ROOT/models/final"
