#!/bin/bash
# Batch job for a GPU pod (run from a clone of the repo, the file server already started on :8000 over /w/out):
# jumping from platform to platform, from the CMU jumps (data/cmu_jump.bin: two running leaps, two standing forward
# jumps, two running jumps). Each room of data/jump_bank.json (tools/jump_course.mjs) is a takeoff and a landing
# platform built round one clip, the gap where its feet are off the ground, the tops 0.3 to 1.0 m up; she tracks the
# clip (task getup_imitate in a room: env.c course mode), from a random moment of it; dropping into the gap ends the
# episode, and so does lying on a platform after a landing. Stage 2 (the defaults): the two standing jumps only (their
# takeoffs within her strength; the running ones need 1.5 to 4 times it at peak), from stage 1 (data/clean/warm/jump),
# starts mostly in the run-up, drifting up to 1 m from the clip allowed; the hand-up adapting down from 0.2.
# Stage 1 was: all six, from the get-up tracker (CLIPS=run_leap1,... BANK=jump_bank.json FROM=getup_track ASSIST=0.3).
G=/w/repo/physics_gym/gpu; B=/w/repo/physics_gym; O=/w/out; ST=$B/data/cmu_jump.bin:${CLIPS:-fjump1,fjump2}; RB=$B/data/${BANK:-jump_bank_standing.json}
{ nvidia-smi; python3 -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"; } > $O/system.txt 2>&1
cd $G && gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c ../room.c -lm > $O/build.log 2>&1
python3 -c "import gpu_env; gpu_env.cuda_lib(); print('cuda environments built')" >> $O/build.log 2>&1
python3 parity_env.py --n 128 --steps 300 --cases jump,goto > $O/parity_env.txt 2>&1
python3 ppo_gpu.py --task getup_imitate --style $ST --room $RB --load $B/data/clean/warm/${FROM:-jump} --assist ${ASSIST:-0.2} --assist-auto --envs ${ENVS:-4096} --seconds ${SECONDS_T:-4500} --out $O/jump > $O/train.log 2>&1
python3 $B/tools/eval_jump.py $O/jump $RB 384 > $O/eval_jump.txt 2>&1
[ -f $O/jump.tss ] && python3 $B/tools/tss2json.py $O/jump $O/jump.json
echo done > $O/DONE
