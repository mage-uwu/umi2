#!/bin/bash
# Batch job for a GPU pod (run from a clone of the repo, the file server already started on :8000 over /w/out):
# the unified navigation run. Go-to across a bank of 24 slope gyms (room.py --stage 2: ramps 6..20 degrees, stair
# pyramids, a ridge, a gradient hill, uneven tiles, a jump terrace, walls and pillars in the way; graded 0.5..1.15) and
# 8 stage-1 rooms, steering by the path helper (env.c nav_waypoint: the waypoint ~1 m along the shortest walkable path,
# progress measured along the path), long episodes of chained goals (half of them up on ramps, landings, steps), soft
# landings rewarded. From the contact-fix go-to (data/clean/warm/goto_contact) with the path inputs fresh.
#  U1: goals within 45 degrees ahead, 30 s episodes      U2: goals anywhere, 40 s episodes      U3: the same, longer
# Evaluations after each stage (eval_goto on the bank); out/DONE.
G=/w/repo/physics_gym/gpu; B=/w/repo/physics_gym; O=/w/out; M=$B/data/clean/cmu_gait.bin
{ nvidia-smi; python3 -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"; } > $O/system.txt 2>&1
cd $G && gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c ../room.c -lm > $O/build.log 2>&1
python3 -c "import gpu_env; gpu_env.cuda_lib(); print('cuda environments built')" >> $O/build.log 2>&1
(cd $B && python3 room.py --stage 2 --bank 24 --difficulty 0.5:1.15 --out slope_bank.json --web '' && python3 room.py --bank 32 --difficulty 0.6:1.0 --out rooms_bank.json --web '' \
  && python3 tools/merge_rooms.py rooms_nav.json slope_bank.json:24 rooms_bank.json:8) >> $O/build.log 2>&1
python3 parity_env.py --n 128 --steps 300 --cases goto,nav,balance > $O/parity_env.txt 2>&1
ENVS=${ENVS:-4096}; RN=$B/rooms_nav.json
ev() { python3 eval_goto.py $O/$1 --mref $M --room $RN --goal-tops 0.5 --n 256 > $O/eval_$1.txt 2>&1; }
python3 ppo_gpu.py --task goto --mref $M --room $RN --load $B/data/clean/warm/goto_contact --fresh-inputs path --assist 0.08 --assist-auto --goal-cone 0.8 --goal-tops 0.5 --steps 900 --envs $ENVS --seconds ${STAGE_1:-5400} --out $O/unified_1 > $O/train_1.log 2>&1
ev unified_1; python3 $B/tools/tss2json.py $O/unified_1 $O/unified_1.json
python3 ppo_gpu.py --task goto --mref $M --room $RN --load $O/unified_1 --assist 0.08 --assist-auto --goal-tops 0.5 --steps 1200 --envs $ENVS --seconds ${STAGE_2:-7200} --out $O/unified_2 > $O/train_2.log 2>&1
ev unified_2; python3 $B/tools/tss2json.py $O/unified_2 $O/unified_2.json
python3 ppo_gpu.py --task goto --mref $M --room $RN --load $O/unified_2 --assist 0.05 --assist-auto --goal-tops 0.5 --steps 1200 --envs $ENVS --seconds ${STAGE_3:-5400} --out $O/unified_3 > $O/train_3.log 2>&1
ev unified_3; python3 $B/tools/tss2json.py $O/unified_3 $O/unified_3.json
echo done > $O/DONE
