#!/bin/bash
# Batch job for a GPU pod (run from a clone of the repo, the file server already started on :8000 over /w/out):
# go-to with more sense of where to step. From unified_3, with the new inputs (env.h OBS_LOOK: the path 0.5 and 2 m on,
# the ground at 16 points round her) starting at zero weight, progress paid only up to 1.2 m/s and heading fast for a
# drop costing the excess speed (env.c goto_reward); over the same room bank as the unified run (24 slope gyms, 8
# stage-1 rooms), the hand-up adapting down from 5%. Then the go-to and path-following evaluations.
G=/w/repo/physics_gym/gpu; B=/w/repo/physics_gym; O=/w/out; M=$B/data/clean/cmu_gait.bin
{ nvidia-smi; python3 -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"; } > $O/system.txt 2>&1
cd $G && gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c ../room.c -lm > $O/build.log 2>&1
python3 -c "import gpu_env; gpu_env.cuda_lib(); print('cuda environments built')" >> $O/build.log 2>&1
(cd $B && python3 room.py --stage 2 --bank 24 --difficulty 0.5:1.15 --out slope_bank.json --web '' && python3 room.py --bank 32 --difficulty 0.6:1.0 --out rooms_bank.json --web '' \
  && python3 tools/merge_rooms.py rooms_nav.json slope_bank.json:24 rooms_bank.json:8) >> $O/build.log 2>&1
python3 parity_env.py --n 128 --steps 300 --cases goto,nav,balance > $O/parity_env.txt 2>&1
RN=$B/rooms_nav.json
python3 ppo_gpu.py --task goto --mref $M --room $RN --load $B/data/clean/warm/unified_3 --assist 0.05 --assist-auto --goal-tops 0.5 --steps 900 --envs ${ENVS:-4096} --seconds ${SECONDS_T:-4500} --out $O/aware > $O/train.log 2>&1
python3 eval_goto.py $O/aware --mref $M --room $RN --goal-tops 0.5 --n 256 > $O/eval_goto.txt 2>&1
python3 eval_goto.py $O/aware --mref $M --room $RN --goal-tops 0.5 --n 256 --assist 0.05 >> $O/eval_goto.txt 2>&1
python3 $B/tools/eval_nav.py $O/aware 256 > $O/eval_nav.txt 2>&1
[ -f $O/aware.tss ] && python3 $B/tools/tss2json.py $O/aware $O/aware.json
echo done > $O/DONE
