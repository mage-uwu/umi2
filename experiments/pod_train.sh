#!/bin/bash
# Batch job for a CPU pod: build, train one checkpoint, serve results read-only on :8000.
# Env: RUN_SECONDS (training budget), TRAIN_ARGS (extra trainer flags), TAG (output name).
set -x
mkdir -p /w/out && cd /w
python3 -m http.server 8000 --directory /w/out >/dev/null 2>&1 &
curl -sSL -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
cp /w/repo/nanotss_blt.c . && cc -O3 -march=native -fopenmp nanotss_blt.c -lm -o nanotss 2>out/build.log
{ nproc; lscpu | grep -E 'Model name|avx512' | cut -c1-120; } > out/cpu.txt
./nanotss --check > out/check.log 2>&1
./nanotss --data input.txt --threads "$(nproc)" --seconds "${RUN_SECONDS:-7200}" ${TRAIN_ARGS} --save "out/${TAG:-model}.bin" > "out/${TAG:-model}.log" 2>&1
echo done > out/DONE
sleep infinity
