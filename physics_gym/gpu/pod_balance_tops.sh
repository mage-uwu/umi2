#!/bin/bash
# Batch job for a GPU pod (run from a clone of the repo, the file server already started on :8000 over /w/out):
# Balance that holds on ramps. From the flat Balance (data/clean/warm/balance, after the contact fix), half of the
# episodes start standing on the top of a room box (env.c top_start: ramps, stair landings, platforms, stones) across
# the graded room bank, pushes to 60 N. out/DONE at the end.
G=/w/repo/physics_gym/gpu; B=/w/repo/physics_gym; O=/w/out
{ nvidia-smi; python3 -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"; } > $O/system.txt 2>&1
cd $G && gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c ../room.c -lm > $O/build.log 2>&1
python3 -c "import gpu_env; gpu_env.cuda_lib(); print('cuda environments built')" >> $O/build.log 2>&1
(cd $B && python3 room.py --bank 32 --difficulty 0.4:1.2 --out rooms_bank.json) >> $O/build.log 2>&1
python3 parity_env.py --n 128 --steps 300 --cases balance,balance_tops > $O/parity_env.txt 2>&1
python3 ppo_gpu.py --task balance --push 60 --room $B/rooms_bank.json --goal-tops ${TOPS:-0.5} --load $B/data/clean/warm/balance --envs ${ENVS:-4096} --seconds ${SECONDS_B:-1200} --out $O/balance_tops > $O/train.log 2>&1
for t in 0 1; do echo "tops $t: $(python3 $B/tools/eval_balance.py $O/balance_tops $B/rooms_bank.json 64 $t 10 | tail -1)"; done > $O/eval.txt 2>&1
[ -f $O/balance_tops.tss ] && python3 $B/tools/tss2json.py $O/balance_tops $O/balance_tops.json
echo done > $O/DONE
