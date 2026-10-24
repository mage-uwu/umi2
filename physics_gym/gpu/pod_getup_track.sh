#!/bin/bash
# Batch job for a GPU pod (run from a clone of the repo, the file server already started on :8000 over /w/out):
# the get-up tracker, from wherever a fall leaves her. From the clip-set imitation (data/clean/warm/getup_imit), 40% of the
# episodes start from a real fall (limp and shoved, or on all fours) with the nearest frame of the six CMU get-ups taking
# over as the reference (env.c matched_start: the falls of a bank simulated once), the rest from a random frame of a
# clip; the hand-up toward the reference
# adapting down from 0.3. The viewer does the same matching when she falls and runs this policy along the clip.
G=/w/repo/physics_gym/gpu; B=/w/repo/physics_gym; O=/w/out; ST=$B/data/cmu_ref.bin:getup_back1,getup_back2,getup_face1,getup_face2,getup_side1,getup_side2
{ nvidia-smi; python3 -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"; } > $O/system.txt 2>&1
cd $G && gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c ../room.c -lm > $O/build.log 2>&1
python3 -c "import gpu_env; gpu_env.cuda_lib(); print('cuda environments built')" >> $O/build.log 2>&1
python3 parity_env.py --n 128 --steps 300 --cases getup_track,getup_imitate > $O/parity_env.txt 2>&1
python3 ppo_gpu.py --task getup_imitate --style $ST --style-start ${FALLS:-0.4} --room '' --load $B/data/clean/warm/getup_imit --assist 0.3 --assist-auto --envs ${ENVS:-4096} --seconds ${SECONDS_T:-3600} --out $O/getup_track > $O/train.log 2>&1
for f in 1 0; do echo "falls share $f: $(python3 $B/tools/eval_track.py $O/getup_track $f | tr '\n' ' ')"; done > $O/eval.txt 2>&1
[ -f $O/getup_track.tss ] && python3 $B/tools/tss2json.py $O/getup_track $O/getup_track.json
echo done > $O/DONE
