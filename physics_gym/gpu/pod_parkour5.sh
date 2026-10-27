#!/bin/bash
# Batch job for a GPU pod (run from a clone of the repo, the file server already started on :8000 over /w/out): parkour run
# 5, on a fixed budget. From run 4's checkpoint (data/clean/warm/parkour4a: a 0.36 hand-up, 45-50% of clips finished) on
# the same 57 clips at x2 strength, the hand-up along the path only, and now the clips drawn by how often they fail
# (--clip-focus: half by that, half uniformly; the hand-up judged over all clips alike):
#   stage 1 (SECONDS_T, 3 h 20 min): the hand-up adapting to keep ASSIST_TARGET (0.5) of the clips finished;
#   stage 2 (STAGE2, 30 min; only if stage 1 ends above a 0.05 hand-up): what it reached faded to none over it.
# The training scan as run 3 and 4 trained with (446a28d: 1d6722b's length-aware one fails scan_parity.py). If the run
# with --clip-focus dies in its first minutes, it starts again without. Snapshots of stage 1's checkpoint every 30 min
# (pk1_s<n>). A watchdog stops the pod (runpodctl, as RunPod pods have it) DEADLINE s after this starts, whatever happens.
#   SECONDS_T (12000)  STAGE2 (1800)  ASSIST (0.36)  ASSIST_TARGET (0.5)  FOCUS (0.5)  FROM (parkour4a)  DEADLINE (15600)
G=/w/repo/physics_gym/gpu; B=/w/repo/physics_gym; O=${OUT:-/w/out}; mkdir -p $O
( sleep ${DEADLINE:-15600}; echo "deadline: stopping the pod" >> $O/stages.txt; runpodctl stop pod $RUNPOD_POD_ID ) > $O/watchdog.log 2>&1 &
TR=$B/data/parkour_fit
CL=$(python3 -c "import json; print(','.join(json.load(open('$TR' + '_bank.json'))['clips']))")
K="--task getup_imitate --style $TR.bin:$CL --room ${TR}_bank.json --steps 400 --strength 2 --start0 ${START0:-0.5} --drift ${DRIFT:-0.6} --tail 0.1 --assist-pos-only 1 --envs ${ENVS:-16384} --minibatches ${MB:-16}"
fail() { echo "$1" > $O/FAILED; exit 1; }
{ nvidia-smi; python3 -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"; nproc; which runpodctl; } > $O/system.txt 2>&1
(cd /w/repo && git checkout 446a28d -- physics_gym/gpu/tss_scan.cu physics_gym/gpu/nanotss_torch.py) > $O/build.log 2>&1 || fail "scan checkout"
cd $G && gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c ../room.c -lm >> $O/build.log 2>&1 || fail "build"
export TSS_FMAD=1
python3 -c "import gpu_env; gpu_env.cuda_lib(); print('built')" >> $O/build.log 2>&1 || fail "cuda build"
TSS_FMAD= python3 parity_env.py --n 64 --steps 300 --cases parkour > $O/parity.txt 2>&1
( n=0; while sleep 1800; do n=$((n + 1)); for e in tss head; do [ -f $O/pk1.$e ] && cp $O/pk1.$e $O/pk1_s$n.$e; done; done ) &
SNAP=$!
S1="--load $B/data/clean/warm/${FROM:-parkour4a} --assist ${ASSIST:-0.36} --assist-auto --assist-target ${ASSIST_TARGET:-0.5} --seconds ${SECONDS_T:-12000} --out $O/pk1"
if ! python3 ppo_gpu.py $K $S1 --clip-focus ${FOCUS:-0.5} > $O/train1.log 2>&1; then
  [ $(grep -c '"iter"' $O/train1.log) -lt 10 ] || fail "stage 1"
  mv $O/train1.log $O/train1_focus_failed.log; echo "--clip-focus failed early: without it" >> $O/stages.txt
  python3 ppo_gpu.py $K $S1 > $O/train1.log 2>&1 || fail "stage 1 (no focus)"
fi
kill $SNAP
A2=$(grep '"assist"' $O/train1.log | tail -1 | python3 -c "import sys, json; print(json.loads(sys.stdin.read())['checkpoint_assist'])")
echo "stage 1 checkpoint at hand-up $A2" >> $O/stages.txt
if python3 -c "import sys; sys.exit(0 if $A2 > 0.05 else 1)"; then
  echo "stage 2: $A2 faded to none over ${STAGE2:-1800} s" >> $O/stages.txt
  python3 ppo_gpu.py $K --load $O/pk1 --assist $A2 --assist-decay ${STAGE2:-1800} --seconds ${STAGE2:-1800} --out $O/pk > $O/train.log 2>&1 || fail "stage 2"
fi
for f in pk1 pk; do [ -f $O/$f.tss ] && python3 $B/tools/tss2json.py $O/$f $O/$f.json; done
echo done > $O/DONE
