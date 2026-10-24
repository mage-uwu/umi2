#!/bin/bash
# Batch job for a GPU pod (run from a clone of the repo, the file server already started on :8000 over /w/out):
# jumping from platform to platform, from the CMU jumps (data/cmu_jump.bin: two running leaps, two standing forward
# jumps, two running jumps). Each room of data/jump_bank.json (tools/jump_course.mjs) is a takeoff and a landing
# platform built round one clip, the gap where its feet are off the ground, the tops 0.3 to 1.0 m up; she tracks the
# clip (task getup_imitate in a room: env.c course mode), from a random moment of it; dropping into the gap ends the
# episode. From the get-up tracker (it already acts round a clip), the hand-up toward the reference adapting down from 0.3.
G=/w/repo/physics_gym/gpu; B=/w/repo/physics_gym; O=/w/out; ST=$B/data/cmu_jump.bin:run_leap1,run_leap2,fjump1,fjump2,run_jump1,run_jump2; RB=$B/data/jump_bank.json
{ nvidia-smi; python3 -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"; } > $O/system.txt 2>&1
cd $G && gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c ../room.c -lm > $O/build.log 2>&1
python3 -c "import gpu_env; gpu_env.cuda_lib(); print('cuda environments built')" >> $O/build.log 2>&1
python3 parity_env.py --n 128 --steps 300 --cases jump,goto > $O/parity_env.txt 2>&1
python3 ppo_gpu.py --task getup_imitate --style $ST --room $RB --load $B/data/clean/warm/getup_track --assist 0.3 --assist-auto --envs ${ENVS:-4096} --seconds ${SECONDS_T:-4500} --out $O/jump > $O/train.log 2>&1
python3 $B/tools/eval_jump.py $O/jump $RB 384 > $O/eval_jump.txt 2>&1
[ -f $O/jump.tss ] && python3 $B/tools/tss2json.py $O/jump $O/jump.json
echo done > $O/DONE
