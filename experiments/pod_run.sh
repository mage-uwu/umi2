#!/bin/bash
# Batch job for a CPU pod: build this commit, run the committed experiment script $SCRIPT,
# and serve /w/out read-only on :8000. Writes out/DONE when the script exits.
mkdir -p /w/out && cd /w
python3 -m http.server 8000 --directory /w/out >/dev/null 2>&1 &
curl -sSL -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
cp /w/repo/nanotss_blt.c . && cc -O3 -march=native -fopenmp nanotss_blt.c -lm -o nanotss 2>out/build.log
bash "/w/repo/$SCRIPT" > out/run.log 2>&1
echo "exit $?" > out/DONE
sleep infinity
