#!/bin/bash
# Batch job for a GPU pod (run from a clone of the repo, the file server already started on :8000 over /w/out):
# a walk that survives its own exploration noise, then go-to over the gym's terrain with probes (ROADMAP.md M2).
#  W: the clean walk prior (WALK_LOAD, default data/clean/warm/walk_robust_slick) trained on in free-heading imitation (ppo_gpu --free-heading 1:
#     the clip's style, not its path, so only falls end episodes and she must learn to recover) with pushes up to 80 N;
#     the prior falls in 98% of 20 s episodes under its own exploration noise (sigma 0.14), which go-to training needs.
#  A: go-to across the room bank graded 0.4x..1.2x (stairs, slopes, platforms, stones, gaps), 60% of goals up on raised
#     surfaces, goals ahead (30 degrees); the inputs the walk never saw vary (goal, grip, probes) start fresh.
#  B: the same with goals anywhere.
# The probes (env.h OBS_PROBE): the ground at 6 points ahead and 10 short rays at knee and hip height, so she can see
# a step or a box coming. The assist adapts throughout. Evaluations at the end; out/DONE.
G=/w/repo/physics_gym/gpu; B=/w/repo/physics_gym; O=/w/out; M=$B/data/clean/cmu_gait.bin
{ nvidia-smi; python3 -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"; } > $O/system.txt 2>&1
cd $G && gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c ../room.c -lm > $O/build.log 2>&1
python3 -c "import gpu_env; gpu_env.cuda_lib(); print('cuda environments built')" >> $O/build.log 2>&1
(cd $B && python3 room.py --bank 32 --difficulty 0.4:1.2 --out rooms_bank.json) >> $O/build.log 2>&1
python3 parity_env.py --n 128 --steps 300 --cases goto,balance,bank > $O/parity_env.txt 2>&1
ENVS=${ENVS:-4096}; SW=${STAGE_W:-2400}; SA=${STAGE_A:-2400}; SB=${STAGE_B:-3000}; RB=$B/rooms_bank.json
WL=${WALK_LOAD:-data/clean/warm/walk_robust_slick}
python3 ppo_gpu.py --task walk --free-heading 1 --push 80 --mref $M --room '' --load $B/$WL --assist 0.1 --assist-auto --envs $ENVS --seconds $SW --out $O/walk_robust > $O/train_w.log 2>&1
for sig in 0 0.139 0.2; do echo "noise $sig: $(python3 $B/tools/eval_walk.py $O/walk_robust $M 64 $sig | tail -1)"; done > $O/eval_walk_robust.txt 2>&1
python3 ppo_gpu.py --task goto --mref $M --room $RB --load $O/walk_robust --fresh-inputs goal,grip,probe --assist 0.15 --assist-auto --goal-cone 0.5 --goal-tops 0.6 --envs $ENVS --seconds $SA --out $O/terrain_a > $O/train_a.log 2>&1
python3 ppo_gpu.py --task goto --mref $M --room $RB --load $O/terrain_a --assist 0.1 --assist-auto --goal-tops 0.6 --envs $ENVS --seconds $SB --out $O/terrain_b > $O/train_b.log 2>&1
for s in walk_robust terrain_a terrain_b; do [ -f $O/$s.tss ] && python3 $B/tools/tss2json.py $O/$s $O/$s.json; done
for s in terrain_a terrain_b; do
  [ -f $O/$s.tss ] || continue
  python3 eval_goto.py $O/$s --mref $M --room $RB --goal-tops 0.6 --n 256 > $O/eval_${s}_bank.txt 2>&1
  python3 eval_goto.py $O/$s --mref $M --room $B/room.json --n 256 > $O/eval_${s}_room.txt 2>&1
done
echo done > $O/DONE
