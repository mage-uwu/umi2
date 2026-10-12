#!/bin/bash
# Pod job: can the native memory learn long-range recall? Four arms, equal training time.
cd /w; TH=$(nproc); S=${RUN_SECONDS:-1800}
declare -A ARM=( [A_window]="" [B_carry]="--carry-windows 64" [C_carry_rep25]="--carry-windows 64 --repeat-p 0.25" [D_carry_rep50]="--carry-windows 64 --repeat-p 0.5" )
for a in A_window B_carry C_carry_rep25 D_carry_rep50; do
  ./nanotss --data input.txt --threads $TH --seconds $S ${ARM[$a]} --save out/$a.bin > out/$a.train.log 2>&1
  ./nanotss --load out/$a.bin --ttt-eval input.txt --threads $TH --ttt-stream 1 > out/$a.stream.jsonl
  python3 /w/repo/experiments/recall_test.py ./nanotss out/$a.bin input.txt > out/$a.recall.jsonl
done
