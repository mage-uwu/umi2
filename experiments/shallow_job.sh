#!/bin/bash
# Pod job: deepest tree that still learns copying, then long-range BLT models with a shallow global tree.
cd /w; TH=$(nproc)
python3 -c "
import random
random.seed(1);open('copy32.txt','w').write(''.join((lambda s:s+'|'+s+'\n')(''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(32))) for i in range(40000)))"
for D in 4 5; do ./nanotss --data copy32.txt --mode byte --conv-mix 0 --depth $D --threads $TH --seconds 600 > out/copy_D$D.log 2>&1; done
for a in "w64:--width 64" "w192:--width 192"; do n=blt_D3_${a%%:*}; f=${a#*:}
  ./nanotss --data input.txt --depth 3 $f --carry-windows 64 --repeat-p 0.25 --threads $TH --seconds 1800 --save out/$n.bin > out/$n.train.log 2>&1
  ./nanotss --load out/$n.bin --ttt-eval input.txt --threads $TH --ttt-stream 1 > out/$n.stream.jsonl
  python3 /w/repo/experiments/recall_test.py ./nanotss out/$n.bin input.txt > out/$n.recall.jsonl
done
