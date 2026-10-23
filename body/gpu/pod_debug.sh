#!/bin/bash
# Batch job: debug_env.py (GPU vs CPU environments, field by field). Serves /w/out on :8000.
mkdir -p /w/out; cd /w; python3 -m http.server 8000 --directory /w/out >/dev/null 2>&1 &
cd /w/repo/body/gpu && gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c -lm 2> /w/out/build.log
python3 debug_env.py > /w/out/debug.txt 2>&1; echo done > /w/out/DONE; sleep infinity
