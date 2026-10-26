#!/bin/bash
# Batch job for a GPU pod (run from a clone of the repo, the file server already started on :8000 over /w/out): where the PPO
# update's time goes at scale, on the parkour course (gpu/pod_parkour.sh's settings). On an A100 the physics is fast enough
# (bench_env.py: 320K env steps/s at 16K environments) that the update takes 72-74% of a run, and it ran only 1.4x faster
# than on an A40: launch- or host-bound, not compute-bound. Here: one iteration profiled (ppo_gpu.py --profile) at 16384
# environments; 90 s runs with bigger minibatches (16 / 8 / 4 per pass) and at 32768 environments; and the in-place step
# kernel checked (parity_env.py) and timed (bench_env.py) with fused multiply-adds.
G=/w/repo/physics_gym/gpu; B=/w/repo/physics_gym; O=/w/out; TR=$B/data/parkour_fit
CL=$(python3 -c "import json; print(','.join(json.load(open('$TR' + '_bank.json'))['clips']))")
K="--task getup_imitate --style $TR.bin:$CL --room ${TR}_bank.json --load $B/data/clean/warm/parc1 --steps 400 --strength 2 --start0 0.5 --drift 0.6 --tail 0.1 --assist-pos-only 1 --assist 0.6"
{ nvidia-smi; python3 -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"; nproc; } > $O/system.txt 2>&1
cd $G && gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c ../room.c -lm > $O/build.log 2>&1
python3 -c "import gpu_env; gpu_env.cuda_lib(); gpu_env.build_cu('tss_scan.cu', 'libtss_scan.so'); print('built')" >> $O/build.log 2>&1
TSS_INPLACE=1 python3 parity_env.py --n 64 --steps 300 --cases parkour > $O/parity.txt 2>&1
TSS_INPLACE=1 TSS_FMAD=1 python3 bench_env.py 4096,16384,32768 > $O/bench.jsonl 2>> $O/bench.err
TSS_FMAD=1 python3 bench_env.py 16384 >> $O/bench.jsonl 2>> $O/bench.err
export TSS_INPLACE=1 TSS_FMAD=1
python3 ppo_gpu.py $K --envs 16384 --minibatches 16 --seconds 60 --profile 3 --out $O/prof > $O/prof.log 2>&1
for mb in 16 8 4; do python3 ppo_gpu.py $K --envs 16384 --minibatches $mb --seconds 90 --out $O/mb$mb > $O/mb$mb.log 2>&1; done
python3 ppo_gpu.py $K --envs 32768 --minibatches 16 --seconds 90 --out $O/n32k > $O/n32k.log 2>&1
echo done > $O/DONE
