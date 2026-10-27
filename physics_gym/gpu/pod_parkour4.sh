#!/bin/bash
# Batch job for a GPU pod (run from a clone of the repo, the file server already started on :8000 over /w/out): parkour run
# 4, long. From run 3's stage 1 (data/clean/warm/parkour3: 2% of clips unaided, a 0.51 hand-up) on the same 57 clips at x2
# strength (data/parkour_fit), the hand-up along the path only:
#   stage 1 (SECONDS_T, 4 h): the hand-up adapting (--assist-auto) to keep ASSIST_TARGET (0.5) of the clips finished: run
#            3's rule asked 0.6 and stalled at 0.51 with 49-61% finished;
#   stage 2 (STAGE2, 1 h; only if stage 1 ends above a 0.05 hand-up): what it reached faded to none over DECAY (40 min),
#            then on without.
# The training scan: tss_scan.cu as of 1d6722b (each sequence's own length) if scan_parity.py passes on this GPU, else
# run 3's (446a28d). No evaluation here (run 3's took 47 min on one CPU core of an idle A100): the checkpoints are pulled
# and evaluated elsewhere. Snapshots of stage 1's checkpoint every hour (pk1_h<n>), the logs as they go.
#   SECONDS_T (14400)  STAGE2 (3600)  DECAY (2400)  ASSIST (0.5)  ASSIST_TARGET (0.5)  FROM (parkour3)  ENVS (16384)  MB (16)
G=/w/repo/physics_gym/gpu; B=/w/repo/physics_gym; O=${OUT:-/w/out}; mkdir -p $O
TR=$B/data/parkour_fit
CL=$(python3 -c "import json; print(','.join(json.load(open('$TR' + '_bank.json'))['clips']))")
K="--task getup_imitate --style $TR.bin:$CL --room ${TR}_bank.json --steps 400 --strength 2 --start0 ${START0:-0.5} --drift ${DRIFT:-0.6} --tail 0.1 --assist-pos-only 1 --envs ${ENVS:-16384} --minibatches ${MB:-16}"
fail() { echo "$1" > $O/FAILED; exit 1; }
{ nvidia-smi; python3 -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"; nproc; } > $O/system.txt 2>&1
cd $G && gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c ../room.c -lm > $O/build.log 2>&1 || fail "build"
export TSS_FMAD=1
python3 -c "import gpu_env; gpu_env.cuda_lib(); print('built')" >> $O/build.log 2>&1 || fail "cuda build"
TSS_FMAD= python3 parity_env.py --n 64 --steps 300 --cases parkour > $O/parity.txt 2>&1
if ! python3 scan_parity.py $B/data/clean/warm/${FROM:-parkour3}.tss 2048 64 > $O/scan_parity.txt 2>&1; then
  echo "== parity failed or did not run: the scan as run 3 trained with (446a28d)" >> $O/scan_parity.txt
  (cd /w/repo && git checkout 446a28d -- physics_gym/gpu/tss_scan.cu physics_gym/gpu/nanotss_torch.py) >> $O/scan_parity.txt 2>&1 || fail "scan fallback"
fi
( n=0; while sleep 3600; do n=$((n + 1)); for e in tss head; do [ -f $O/pk1.$e ] && cp $O/pk1.$e $O/pk1_h$n.$e; done; done ) &
SNAP=$!
python3 ppo_gpu.py $K --load $B/data/clean/warm/${FROM:-parkour3} --assist ${ASSIST:-0.5} --assist-auto --assist-target ${ASSIST_TARGET:-0.5} \
    --seconds ${SECONDS_T:-14400} --out $O/pk1 > $O/train1.log 2>&1 || fail "stage 1"
kill $SNAP
A2=$(grep '"assist"' $O/train1.log | tail -1 | python3 -c "import sys, json; print(json.loads(sys.stdin.read())['checkpoint_assist'])")
echo "stage 1 checkpoint at hand-up $A2" > $O/stages.txt
if python3 -c "import sys; sys.exit(0 if $A2 > 0.05 else 1)"; then
  echo "stage 2: $A2 faded to none over ${DECAY:-2400} s" >> $O/stages.txt
  python3 ppo_gpu.py $K --load $O/pk1 --assist $A2 --assist-decay ${DECAY:-2400} --seconds ${STAGE2:-3600} --out $O/pk > $O/train.log 2>&1 || fail "stage 2"
fi
for f in pk1 pk; do [ -f $O/$f.tss ] && python3 $B/tools/tss2json.py $O/$f $O/$f.json; done
echo done > $O/DONE
