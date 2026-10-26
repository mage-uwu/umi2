#!/bin/bash
# Batch job for a GPU pod (run from a clone of the repo, the file server already started on :8000 over /w/out): the training
# scan after the A100 profile (pod_parkour.sh PROFILE=3: k_ssm_fwd 6.2 s of a 13.7 s update, 7x the backward). Two changes:
# each block stops at its sequence's own length (the update's minibatches are episodes padded to the longest), and the
# forward's registers capped so more blocks fit an SM (147 registers x 256 threads: one block per SM, 8 warps waiting on
# __syncthreads). scan_parity.py at a minibatch's shape (2048 sequences x 64 steps, ragged) for the default build and caps of
# 128 and 80 (TSS_SCAN_REGS): the fused scan against the PyTorch loop on the valid steps, and the time of each. Then 60 s of
# ppo_gpu.py on the parkour course with the default and the fastest cap.
G=/w/repo/physics_gym/gpu; B=/w/repo/physics_gym; O=/w/out; TR=$B/data/parkour_fit; P=$B/data/clean/warm/parc1.tss
CL=$(python3 -c "import json; print(','.join(json.load(open('$TR' + '_bank.json'))['clips']))")
{ nvidia-smi; python3 -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"; } > $O/system.txt 2>&1
cd $G && gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c ../room.c -lm > $O/build.log 2>&1
python3 -c "import gpu_env; gpu_env.cuda_lib(); print('built')" >> $O/build.log 2>&1
for rg in "" 128 80; do echo "== regs ${rg:-default}" >> $O/scan.txt; TSS_SCAN_REGS=$rg python3 scan_parity.py $P 2048 64 >> $O/scan.txt 2>&1; done
K="--task getup_imitate --style $TR.bin:$CL --room ${TR}_bank.json --load $B/data/clean/warm/parc1 --steps 400 --strength 2 --start0 0.5 --drift 0.6 --tail 0.1 --assist-pos-only 1 --assist 0.6 --envs 16384 --minibatches 16 --seconds 60"
for rg in "" 80; do TSS_FMAD=1 TSS_SCAN_REGS=$rg python3 ppo_gpu.py $K --out $O/ppo${rg:-def} > $O/ppo${rg:-def}.log 2>&1; done
echo done > $O/DONE
