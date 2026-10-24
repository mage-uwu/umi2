#!/bin/bash
# Batch job for a GPU pod (run from a clone of the repo, the file server already started on :8000 over /w/out):
# everything retrained after the contact fix (body.c static_friction: once per body and surface, judged against
# penetration depth, plus no bounce off contacts; a planted foot no longer slid down the ramps).
#  W: the walk (from data/clean/warm/walk_robust) in free-heading imitation with pushes, assist adapting from 0.05
#  Bal: Balance from scratch in the gym's physics, pushes to 60 N
#  A: go-to across the room bank from W, fresh goal/grip/probe inputs, goals ahead, 60% on raised surfaces
#  B: the same with goals anywhere
# Evaluations at the end; out/DONE.
G=/w/repo/physics_gym/gpu; B=/w/repo/physics_gym; O=/w/out; M=$B/data/clean/cmu_gait.bin
{ nvidia-smi; python3 -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"; } > $O/system.txt 2>&1
cd $G && gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c ../room.c -lm > $O/build.log 2>&1
python3 -c "import gpu_env; gpu_env.cuda_lib(); print('cuda environments built')" >> $O/build.log 2>&1
(cd $B && python3 room.py --bank 32 --difficulty 0.4:1.2 --out rooms_bank.json) >> $O/build.log 2>&1
python3 parity_env.py --n 128 --steps 300 --cases goto,balance,bank > $O/parity_env.txt 2>&1
ENVS=${ENVS:-4096}; RB=$B/rooms_bank.json
python3 ppo_gpu.py --task walk --free-heading 1 --push 80 --mref $M --room '' --load $B/data/clean/warm/walk_robust --assist 0.05 --assist-auto --envs $ENVS --seconds ${STAGE_W:-1200} --out $O/walk > $O/train_w.log 2>&1
for sig in 0 0.139 0.2; do echo "noise $sig: $(python3 $B/tools/eval_walk.py $O/walk $M 64 $sig | tail -1)"; done > $O/eval_walk.txt 2>&1
python3 ppo_gpu.py --task balance --push 60 --room '' --envs $ENVS --seconds ${STAGE_BAL:-900} --out $O/balance > $O/train_bal.log 2>&1
python3 ppo_gpu.py --task goto --mref $M --room $RB --load $O/walk --fresh-inputs goal,grip,probe --assist 0.15 --assist-auto --goal-cone 0.5 --goal-tops 0.6 --envs $ENVS --seconds ${STAGE_A:-2400} --out $O/terrain_a > $O/train_a.log 2>&1
python3 ppo_gpu.py --task goto --mref $M --room $RB --load $O/terrain_a --assist 0.1 --assist-auto --goal-tops 0.6 --envs $ENVS --seconds ${STAGE_B:-2400} --out $O/terrain_b > $O/train_b.log 2>&1
for s in walk balance terrain_a terrain_b; do [ -f $O/$s.tss ] && python3 $B/tools/tss2json.py $O/$s $O/$s.json; done
for s in terrain_a terrain_b; do
  [ -f $O/$s.tss ] || continue
  python3 eval_goto.py $O/$s --mref $M --room $RB --goal-tops 0.6 --n 256 > $O/eval_${s}_bank.txt 2>&1
  python3 eval_goto.py $O/$s --mref $M --room $B/room.json --n 256 > $O/eval_${s}_room.txt 2>&1
done
echo done > $O/DONE
