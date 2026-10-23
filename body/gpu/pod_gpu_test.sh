#!/bin/bash
# Batch job for a GPU pod (run from a clone of the repo): parity against C on the GPU, kernel benchmarks,
# then the same short getup run with ppo_gpu.py and with ppo_tss.c. Serves /w/out read-only on :8000;
# writes out/DONE at the end.
mkdir -p /w/out && cd /w
python3 -m http.server 8000 --directory /w/out >/dev/null 2>&1 &
G=/w/repo/body/gpu; B=/w/repo/body; O=/w/out
{ nvidia-smi; nproc; lscpu | grep -i "model name\|flags" | cut -c1-300; python3 -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"; } > $O/system.txt 2>&1
cd $G && gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c -lm 2> $O/build.log
python3 parity.py init:3 --device cuda --B 16 --L 64 --steps 3 > $O/parity.txt 2>&1
python3 bench.py --device cuda > $O/bench.txt 2>&1
STYLE=$B/data/cmu_ref.bin:140_01,140_02,140_03,140_04,140_08,140_09
python3 ppo_gpu.py --task getup --style $STYLE --envs 2048 --horizon 64 --seconds 420 --threads $(nproc) --out $O/getup_gpu > $O/ppo_gpu.log 2>&1
cd $B && gcc -O3 -march=native -fopenmp -ffp-contract=off -o ppo_tss ppo_tss.c env.c body.c body_x8.c mref.c -lm 2>> $O/build.log
./ppo_tss --task getup --style $STYLE --envs 64 --horizon 64 --seconds 180 --threads $(nproc) --out $O/getup_c > $O/ppo_c.log 2>&1
echo done > $O/DONE
sleep infinity
