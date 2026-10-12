#!/bin/bash
# Pod job: A/B whether tree routing blocks the plastic memory from learning to copy.
# Byte mode isolates the memory (no patching). D=0 makes each tree a single unrouted node.
cd /w; TH=$(nproc); S=${RUN_SECONDS:-600}
python3 -c "
import random
random.seed(1);open('copy32.txt','w').write(''.join((lambda s:s+'|'+s+'\n')(''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(32))) for i in range(40000)))"
run(){ name=$1; shift; ./nanotss --mode byte --conv-mix 0 --threads $TH --seconds $S "$@" > out/$name.log 2>&1; }
run copy_D7 --data copy32.txt --depth 7
run copy_D3 --data copy32.txt --depth 3
run copy_D0 --data copy32.txt --depth 0
run copy_D0_n64 --data copy32.txt --depth 0 --state-n 64
run rep_D7 --data input.txt --depth 7 --carry-windows 64 --repeat-p 0.5 --repeat-maxgap 128
run rep_D0 --data input.txt --depth 0 --carry-windows 64 --repeat-p 0.5 --repeat-maxgap 128
run nat_D7 --data input.txt --depth 7
run nat_D0 --data input.txt --depth 0
