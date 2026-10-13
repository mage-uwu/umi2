#!/bin/bash
# Pod job: native long-range recall where the memory sees bytes (byte mode, no BLT patching).
cd /w; TH=$(nproc); S=${RUN_SECONDS:-1800}
declare -A ARM=( [b_D3_w64_rep25]="--depth 3 --width 64 --repeat-p 0.25" [b_D3_w192_rep25]="--depth 3 --width 192 --repeat-p 0.25"
                 [b_D3_w64_rep50g2k]="--depth 3 --width 64 --repeat-p 0.5 --repeat-maxgap 2048" [b_D7_w64_rep25]="--depth 7 --width 64 --repeat-p 0.25" )
for a in b_D3_w64_rep25 b_D3_w192_rep25 b_D3_w64_rep50g2k b_D7_w64_rep25; do
  ./nanotss --data input.txt --mode byte --conv-mix 0 --carry-windows 64 ${ARM[$a]} --threads $TH --seconds $S --save out/$a.bin > out/$a.train.log 2>&1
  ./nanotss --load out/$a.bin --ttt-eval input.txt --ttt-stream 1 > out/$a.stream.jsonl
  python3 /w/repo/experiments/recall_test.py ./nanotss out/$a.bin input.txt > out/$a.recall.jsonl
done
