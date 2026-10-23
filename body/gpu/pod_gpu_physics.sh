#!/bin/bash
# Batch job for a GPU pod (run from a clone of the repo): build the CUDA environments, check them against the CPU
# build, benchmark them, then a getup run of ppo_gpu.py with everything on the GPU. Serves /w/out read-only on
# :8000; writes out/DONE at the end.
mkdir -p /w/out && cd /w
python3 -m http.server 8000 --directory /w/out >/dev/null 2>&1 &
G=/w/repo/body/gpu; B=/w/repo/body; O=/w/out
{ nvidia-smi; which nvcc; nvcc --version; lscpu | grep -i "model name"; python3 -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"; } > $O/system.txt 2>&1
cd $G && gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c -lm 2> $O/build.log
python3 -c "import gpu_env; gpu_env.cuda_lib(); print('cuda environments built')" >> $O/build.log 2>&1
python3 parity_env.py --n 64 --steps 600 > $O/parity_env.txt 2>&1
python3 parity.py init:3 --device cuda --B 16 --L 64 --steps 3 > $O/parity.txt 2>&1
python3 bench_env.py > $O/bench_env.txt 2>&1
STYLE=$B/data/cmu_ref.bin:getup_back1,getup_back2,getup_face1,getup_face2,getup_side1,getup_side2
python3 ppo_gpu.py --task getup --style $STYLE --envs 4096 --horizon 64 --seconds 480 --out $O/getup_gpu > $O/ppo_gpu.log 2>&1
echo done > $O/DONE
sleep infinity
