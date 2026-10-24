#!/bin/bash
# Batch job for a GPU pod (run from a clone of the repo, the file server already started on :8000 over /w/out):
# the night's two runs back to back on one GPU.
#  1. a natural get-up, second try (pod_getup.sh's first stalled kneeling: the arms could not push her up, so they
#     are stronger now, muscle.h; and it never felt standing): the CMU get-up clips as style, a hand-up toward
#     standing fading from 0.5 to none over 40 min, then 20 min alone, an entropy bonus against settling early.
#  2. the unified navigation run (pod_unified.sh).
G=/w/repo/physics_gym/gpu; B=/w/repo/physics_gym; O=/w/out; ST=$B/data/cmu_ref.bin:getup_back1,getup_back2,getup_face1,getup_face2,getup_side1,getup_side2
{ nvidia-smi; python3 -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"; } > $O/system.txt 2>&1
cd $G && gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c ../room.c -lm > $O/build_getup.log 2>&1
python3 -c "import gpu_env; gpu_env.cuda_lib(); print('cuda environments built')" >> $O/build_getup.log 2>&1
python3 parity_env.py --n 128 --steps 300 --cases getup > $O/parity_getup.txt 2>&1
python3 ppo_gpu.py --task getup --style $ST --style-start 0.3 --room '' --assist 0.5 --assist-decay ${DECAY_G:-2400} --ent 0.002 --envs ${ENVS:-4096} --seconds ${SECONDS_G:-3600} --out $O/getup > $O/train_getup.log 2>&1
python3 $B/tools/eval_getup.py $O/getup 64 8 > $O/eval_getup.txt 2>&1
[ -f $O/getup.tss ] && python3 $B/tools/tss2json.py $O/getup $O/getup.json
echo done > $O/DONE_GETUP
bash $B/gpu/pod_unified.sh
