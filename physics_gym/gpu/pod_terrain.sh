#!/bin/bash
# Batch job for a GPU pod (run from a clone of the repo, the file server already started on :8000 over /w/out):
# walking over the gym's terrain (ROADMAP.md M2). From the clean walk prior (data/clean/warm/walk_cmu2: the CMU walk,
# no Bandai Namco data), go-to across a bank of 32 stage-1 rooms graded in difficulty (room.py --difficulty 0.4:1.2:
# stair rise, slope angles, platform and stone heights, gaps scaled from 0.4x to 1.2x the design), with 60% of goals
# up on a raised surface (stair landings, platforms, foam, stepping stones, slope landings) so the way there crosses
# terrain. She feels it only through touch and muscle sense. The assist adapts (ppo_gpu --assist-auto). Stage A: goals
# ahead of her (within 30 degrees); stage B: anywhere. Each stage evaluated on the bank and on the design room
# (room.json), with the share of goals reached up on raised surfaces; out/DONE at the end.
G=/w/repo/physics_gym/gpu; B=/w/repo/physics_gym; O=/w/out; M=$B/data/clean/cmu_gait.bin
{ nvidia-smi; python3 -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"; } > $O/system.txt 2>&1
cd $G && gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c ../room.c -lm > $O/build.log 2>&1
python3 -c "import gpu_env; gpu_env.cuda_lib(); print('cuda environments built')" >> $O/build.log 2>&1
(cd $B && python3 room.py --bank 32 --difficulty 0.4:1.2 --out rooms_bank.json) >> $O/build.log 2>&1
python3 parity_env.py --n 128 --steps 300 --cases goto,bank > $O/parity_env.txt 2>&1
ENVS=${ENVS:-4096}; SA=${STAGE_A:-2400}; SB=${STAGE_B:-3000}; RB=$B/rooms_bank.json
python3 ppo_gpu.py --task goto --mref $M --room $RB --load $B/data/clean/warm/walk_cmu2 --assist 0.2 --assist-auto --goal-cone 0.5 --goal-tops 0.6 --envs $ENVS --seconds $SA --out $O/terrain_a > $O/train_a.log 2>&1
python3 ppo_gpu.py --task goto --mref $M --room $RB --load $O/terrain_a --assist 0.1 --assist-auto --goal-tops 0.6 --envs $ENVS --seconds $SB --out $O/terrain_b > $O/train_b.log 2>&1
for s in terrain_a terrain_b; do
  [ -f $O/$s.tss ] || continue
  python3 $B/tools/tss2json.py $O/$s $O/$s.json
  python3 eval_goto.py $O/$s --mref $M --room $RB --goal-tops 0.6 --n 256 > $O/eval_${s}_bank.txt 2>&1
  python3 eval_goto.py $O/$s --mref $M --room $B/room.json --n 256 > $O/eval_${s}_room.txt 2>&1
done
echo done > $O/DONE
