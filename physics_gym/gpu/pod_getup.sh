#!/bin/bash
# Batch job for a GPU pod (run from a clone of the repo, the file server already started on :8000 over /w/out):
# a natural get-up in the gym's physics. The viewer's get-up was a body/ policy run with no joint ranges, no muscle
# limits and actions x2.5: it righted itself with fast twitches. Here: the CMU get-up clips (from the back, face and
# side; data/cmu_ref.bin) as the style reward, 30% of episodes starting from one of their frames, the CMU joint
# ranges and muscle torque and slew limits, static friction. From scratch, an hour. out/DONE at the end.
G=/w/repo/physics_gym/gpu; B=/w/repo/physics_gym; O=/w/out; ST=$B/data/cmu_ref.bin:getup_back1,getup_back2,getup_face1,getup_face2,getup_side1,getup_side2
{ nvidia-smi; python3 -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"; } > $O/system.txt 2>&1
cd $G && gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c ../room.c -lm > $O/build.log 2>&1
python3 -c "import gpu_env; gpu_env.cuda_lib(); print('cuda environments built')" >> $O/build.log 2>&1
python3 parity_env.py --n 128 --steps 300 --cases getup > $O/parity_env.txt 2>&1
python3 ppo_gpu.py --task getup --style $ST --style-start 0.3 --room '' --envs ${ENVS:-4096} --seconds ${SECONDS_G:-3600} --out $O/getup > $O/train.log 2>&1
python3 $B/tools/eval_getup.py $O/getup 64 8 > $O/eval.txt 2>&1
[ -f $O/getup.tss ] && python3 $B/tools/tss2json.py $O/getup $O/getup.json
echo done > $O/DONE
