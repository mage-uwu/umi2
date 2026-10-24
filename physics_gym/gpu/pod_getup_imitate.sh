#!/bin/bash
# Batch job for a GPU pod (run from a clone of the repo, the file server already started on :8000 over /w/out):
# a get-up learned from the motion capture. I: imitation of the six CMU get-ups (from the back, face down, side) as a
# clip set (env.h TaskCfg.style): each episode one clip from a random point, once through and holding the standing end,
# no phase input (she reads from her body where in the motion she is), the hand-up toward the reference's pelvis
# adapting down to none. F: the get-up task from varied falls and all fours, warm from I, the clips as style (half the
# episodes starting in one of their frames), no hand-up. Each evaluated with no hand-up; out/DONE.
G=/w/repo/physics_gym/gpu; B=/w/repo/physics_gym; O=/w/out; ST=$B/data/cmu_ref.bin:getup_back1,getup_back2,getup_face1,getup_face2,getup_side1,getup_side2
{ nvidia-smi; python3 -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"; } > $O/system.txt 2>&1
cd $G && gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c ../room.c -lm > $O/build.log 2>&1
python3 -c "import gpu_env; gpu_env.cuda_lib(); print('cuda environments built')" >> $O/build.log 2>&1
python3 parity_env.py --n 128 --steps 300 --cases getup_imitate,getup,goto > $O/parity_env.txt 2>&1
ENVS=${ENVS:-4096}
python3 ppo_gpu.py --task getup_imitate --style $ST --room '' --assist 0.5 --assist-auto --envs $ENVS --seconds ${SECONDS_I:-2700} --out $O/getup_imit > $O/train_imit.log 2>&1
python3 $B/tools/eval_getup.py $O/getup_imit 64 8 > $O/eval_imit.txt 2>&1
python3 ppo_gpu.py --task getup --style $ST --style-start 0.5 --room '' --load $O/getup_imit --ent 0.001 --envs $ENVS --seconds ${SECONDS_F:-2400} --out $O/getup_ft > $O/train_ft.log 2>&1
python3 $B/tools/eval_getup.py $O/getup_ft 64 8 > $O/eval_ft.txt 2>&1
for s in getup_imit getup_ft; do [ -f $O/$s.tss ] && python3 $B/tools/tss2json.py $O/$s $O/$s.json; done
echo done > $O/DONE
