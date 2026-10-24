#!/bin/bash
# Batch job for a GPU pod (run from a clone of the repo, the file server already started on :8000 over /w/out):
# Balance retrained in the gym's physics (CMU joint ranges, muscle limits, static friction), from scratch, with
# pushes up to 60 N, so standing grips like everything else (the viewer's Balance was a body/ policy trained on
# slipping feet: with static friction it falls, without it it slides down the ramps). out/DONE at the end.
G=/w/repo/physics_gym/gpu; B=/w/repo/physics_gym; O=/w/out
{ nvidia-smi; python3 -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"; } > $O/system.txt 2>&1
cd $G && gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c ../room.c -lm > $O/build.log 2>&1
python3 -c "import gpu_env; gpu_env.cuda_lib(); print('cuda environments built')" >> $O/build.log 2>&1
python3 parity_env.py --n 128 --steps 300 --cases balance > $O/parity_env.txt 2>&1
python3 ppo_gpu.py --task balance --push ${PUSH:-60} --room '' --envs ${ENVS:-4096} --seconds ${SECONDS_B:-900} --out $O/balance > $O/train.log 2>&1
[ -f $O/balance.tss ] && python3 $B/tools/tss2json.py $O/balance $O/balance.json
echo done > $O/DONE
