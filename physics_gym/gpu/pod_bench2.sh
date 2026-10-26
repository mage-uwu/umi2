#!/bin/bash
# Batch job for a GPU pod (run from a clone of the repo, the file server already started on :8000 over /w/out): the step and
# observation kernels without their per-thread copies of the Env. parity_env.py's parkour case for the default build and for
# TSS_INPLACE=1 (the arithmetic is the same: still bit for bit with the CPU); bench_env.py for default, in-place, fused
# multiply-adds and both, at 4K to 32K environments and block sizes 64 and 32; then 90 s of ppo_gpu.py with the fastest.
G=/w/repo/physics_gym/gpu; B=/w/repo/physics_gym; O=/w/out; TR=$B/data/parkour_fit
CL=$(python3 -c "import json; print(','.join(json.load(open('$TR' + '_bank.json'))['clips']))")
{ nvidia-smi; python3 -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"; } > $O/system.txt 2>&1
cd $G && gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c ../room.c -lm > $O/build.log 2>&1
python3 -c "import gpu_env; gpu_env.cuda_lib(); gpu_env.build_cu('tss_scan.cu', 'libtss_scan.so'); print('built')" >> $O/build.log 2>&1
python3 parity_env.py --n 64 --steps 300 --cases parkour > $O/parity.txt 2>&1
TSS_INPLACE=1 python3 parity_env.py --n 64 --steps 300 --cases parkour >> $O/parity.txt 2>&1
for v in "" "TSS_INPLACE=1" "TSS_FMAD=1" "TSS_INPLACE=1 TSS_FMAD=1" "TSS_INPLACE=1 TSS_FMAD=1 TSS_BLOCK=32"; do
  env $v python3 bench_env.py 4096,16384,32768 >> $O/bench.jsonl 2>> $O/bench.err
done
for n in 4096 16384; do
  env TSS_INPLACE=1 TSS_FMAD=1 python3 ppo_gpu.py --task getup_imitate --style $TR.bin:$CL --room ${TR}_bank.json --load $B/data/clean/warm/parc1 --steps 400 \
      --strength 2 --start0 0.5 --drift 0.6 --tail 0.1 --assist-pos-only 1 --assist 0.6 --envs $n --minibatches $((n / 1024)) --seconds 90 --out $O/ppo$n > $O/ppo$n.log 2>&1
done
echo done > $O/DONE
