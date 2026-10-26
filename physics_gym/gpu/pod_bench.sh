#!/bin/bash
# Batch job for a GPU pod (run from a clone of the repo, the file server already started on :8000 over /w/out): where the
# training throughput goes, before the next parkour run. bench_env.py (the environments alone) at 4K to 32K environments,
# for the default build and with TSS_FMAD=1 / TSS_BLOCK=128 / 32; then ppo_gpu.py on the parkour course for 90 s at 4096 and
# 16384 environments (steps per second, rollout and train shares). Run on an A40 and an A100 to see whether the physics is
# bound by double precision (the A40 runs it at 1/64 rate, the A100 at 1/2) or by memory (34 KB of stack per env thread).
G=/w/repo/physics_gym/gpu; B=/w/repo/physics_gym; O=/w/out; TR=$B/data/parkour_fit
CL=$(python3 -c "import json; print(','.join(json.load(open('$TR' + '_bank.json'))['clips']))")
{ nvidia-smi; python3 -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"; } > $O/system.txt 2>&1
cd $G && gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c ../room.c -lm > $O/build.log 2>&1
python3 -c "import gpu_env; gpu_env.cuda_lib(); gpu_env.build_cu('tss_scan.cu', 'libtss_scan.so'); print('built')" >> $O/build.log 2>&1
python3 bench_env.py 4096,8192,16384,32768 > $O/bench.jsonl 2>> $O/bench.err
TSS_FMAD=1 python3 bench_env.py 4096,16384 >> $O/bench.jsonl 2>> $O/bench.err
TSS_BLOCK=128 python3 bench_env.py 4096,16384 >> $O/bench.jsonl 2>> $O/bench.err
TSS_BLOCK=32 python3 bench_env.py 4096,16384 >> $O/bench.jsonl 2>> $O/bench.err
for n in 4096 16384; do
  python3 ppo_gpu.py --task getup_imitate --style $TR.bin:$CL --room ${TR}_bank.json --load $B/data/clean/warm/parc1 --steps 400 --strength 2 --start0 0.5 \
      --drift 0.6 --tail 0.1 --assist-pos-only 1 --assist 0.6 --envs $n --seconds 90 --out $O/ppo$n > $O/ppo$n.log 2>&1
done
echo done > $O/DONE
