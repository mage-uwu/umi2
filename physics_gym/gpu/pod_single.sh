#!/bin/bash
# Batch job for a GPU pod (run from a clone of the repo, the file server already started on :8000 over /w/out): can one
# clip be learned on its own, from scratch, with no hand-up? Four runs side by side on the one GPU, each one clip of
# data/parkour_fit at x2 strength (a bank of that clip's room alone, $O/bank_<clip>.json), episodes ended as eval_course.py
# judges them (0.6 m drift, 0.1 s tail):
#   climb_p0_mlp  p0 (climbing terrain, the hardest in runs 3-5)  a float32 MLP actor 512x3 (--mlp)
#   climb_p4_mlp  p4 (climbing terrain)                            the same
#   vault_p6_mlp  p6 (dash vault)                                  the same: does the budget learn an easier move?
#   climb_p0_tss  p0                                               the nanotss stack the runs use (width 64, ternary)
# A climb the MLP cannot learn alone says the move is not doable for her as set up (retargeting, grip, strength); learnt by
# the MLP and not the stack, the stack's capacity. The checkpoints are pulled and judged elsewhere (tools/eval_course.py
# reads <out>.mlp.pt). A watchdog stops the pod DEADLINE s after this starts.
#   SECS (1320)  ENVS (4096)  MLP (512)  DEADLINE (2100)
G=/w/repo/physics_gym/gpu; B=/w/repo/physics_gym; O=${OUT:-/w/out}; mkdir -p $O
( sleep ${DEADLINE:-2100}; echo "deadline: stopping the pod" >> $O/stages.txt; runpodctl stop pod $RUNPOD_POD_ID ) > $O/watchdog.log 2>&1 &
TR=$B/data/parkour_fit
fail() { echo "$1" > $O/FAILED; exit 1; }
{ nvidia-smi; python3 -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"; nproc; } > $O/system.txt 2>&1
(cd /w/repo && git checkout 446a28d -- physics_gym/gpu/tss_scan.cu physics_gym/gpu/nanotss_torch.py) > $O/build.log 2>&1 || fail "scan checkout"
cd $G && gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c ../room.c -lm >> $O/build.log 2>&1 || fail "build"
export TSS_FMAD=1
python3 -c "import gpu_env; gpu_env.cuda_lib(); print('built')" >> $O/build.log 2>&1 || fail "cuda build"
python3 -c "
import json; b = json.load(open('$TR' + '_bank.json'))
for c in ['p0', 'p4', 'p6']:
    r = [x for x in b['rooms'] if x['clip'] == c][0]; json.dump(dict(b, clips=[c], rooms=[r]), open('$O/bank_' + c + '.json', 'w'))
" || fail "banks"
K="--task getup_imitate --steps 400 --strength 2 --start0 0.5 --drift 0.6 --tail 0.1 --envs ${ENVS:-4096} --minibatches 4 --seconds ${SECS:-1320}"
run() { python3 ppo_gpu.py $K --style $TR.bin:$1 --room $O/bank_$1.json $3 --out $O/$2 > $O/$2.log 2>&1 || echo "$2 failed" >> $O/stages.txt; }
run p0 climb_p0_mlp "--mlp ${MLP:-512}" & run p4 climb_p4_mlp "--mlp ${MLP:-512}" & run p6 vault_p6_mlp "--mlp ${MLP:-512}" & run p0 climb_p0_tss "" &
wait
echo done > $O/DONE
