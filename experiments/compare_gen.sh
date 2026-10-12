#!/bin/bash
# Compare generation methods. Usage: compare_gen.sh BIN CKPT TRAIN_TXT OUTDIR SEEDS THREADS PLAY...
# Each PLAY is "path|SPEAKER"; the first half of the play is the context scratchpad.
BIN=$1 CK=$2 TRAIN=$3 OUT=$4 SEEDS=$5 TH=$6; shift 6; mkdir -p "$OUT"
declare -A M
M[plain_t1]="--refine-epochs 0 --ttt-lr 0 --refine-rounds 0 --refine-cands 1 --refine-keep 1 --refine-final-temp 1"
M[plain_t07]="--refine-epochs 0 --ttt-lr 0 --refine-rounds 0 --refine-cands 1 --refine-keep 1 --refine-final-temp 0.7"
M[ttt_t1]="--refine-epochs 8 --ttt-lr 3e-4 --refine-rounds 0 --refine-cands 1 --refine-keep 1 --refine-final-temp 1"
M[ttt_t07]="--refine-epochs 8 --ttt-lr 3e-4 --refine-rounds 0 --refine-cands 1 --refine-keep 1 --refine-final-temp 0.7"
M[ttt_bo16]="--refine-epochs 8 --ttt-lr 3e-4 --refine-rounds 0 --refine-cands 16 --refine-keep 1 --refine-final-temp 1"
M[ttt_refine]="--refine-epochs 8 --ttt-lr 3e-4 --refine-rounds 3 --refine-cands 16 --refine-keep 4 --refine-steps 8 --refine-final-temp 1"
M[ttt_refine_edit]="--refine-epochs 8 --ttt-lr 3e-4 --refine-rounds 3 --refine-cands 16 --refine-keep 4 --refine-steps 8 --refine-final-temp 1 --refine-edits 64"
for play in "$@"; do path=${play%%|*}; spk=${play##*|}; name=$(basename "$path" .txt)
  n=$(wc -c < "$path"); head -c $((n/2)) "$path" > "$OUT/$name.ctx"
  for m in "${!M[@]}"; do for s in $(seq 1 "$SEEDS"); do
    "$BIN" --load "$CK" --refine 1 --refine-context "$OUT/$name.ctx" --prompt $'\n\n'"$spk"$':\n' --refine-len 400 --seed $s --threads "$TH" ${M[$m]} 2>/dev/null | grep refine_final >> "$OUT/$m.$name.jsonl"
  done; done; done
for m in "${!M[@]}"; do for play in "$@"; do path=${play%%|*}; name=$(basename "$path" .txt)
  echo "$m $name $(python3 "$(dirname "$0")/genmetrics.py" "$TRAIN" "$OUT/$name.ctx" < "$OUT/$m.$name.jsonl")"; done; done | sort > "$OUT/summary.txt"
