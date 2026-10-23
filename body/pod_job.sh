#!/bin/bash
# Batch job for a CPU pod (run from a clone of the repo): build the body trainer, run train.c with
# $TRAIN_ARGS on all cores, and serve /w/out read-only on :8000. Writes out/DONE when training exits.
mkdir -p /w/out && cd /w
python3 -m http.server 8000 --directory /w/out >/dev/null 2>&1 &
B=/w/repo/body
cc -O3 -march=native -fopenmp $B/body.c $B/env.c $B/mref.c $B/policy.c $B/train.c -lm -o btrain 2> out/build.log
./btrain --threads $(nproc) --mref $B/data/motions_ref.bin --out out/policy.pol $TRAIN_ARGS > out/train.log 2>&1
echo "exit $?" > out/DONE
sleep infinity
