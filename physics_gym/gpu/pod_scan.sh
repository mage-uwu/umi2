#!/bin/bash
# Batch job for a GPU pod (run from a clone of the repo, the file server already started on :8000 over /w/out):
# the fused SSM-tree scan (tss_scan.cu): parity with the PyTorch loop (scan_parity.py), then the trainer's speed
# with it (the jump course from data/clean/warm/jump_stick, 150 s, as gpu/pod_bench.sh's baseline: 14,786 steps/s).
G=/w/repo/physics_gym/gpu; B=/w/repo/physics_gym; O=/w/out; ST=$B/data/cmu_jump.bin:fjump1,fjump2; RB=$B/data/jump_bank_standing.json
{ nvidia-smi; python3 -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"; } > $O/system.txt 2>&1
cd $G && gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c ../room.c -lm > $O/build.log 2>&1
python3 -c "import gpu_env; gpu_env.cuda_lib(); print('cuda environments built'); gpu_env.build_cu('tss_scan.cu', 'libtss_scan.so'); print('tss_scan built')" >> $O/build.log 2>&1
python3 scan_parity.py $B/data/clean/warm/jump_stick.tss 256 64 > $O/scan_parity.txt 2>&1
python3 scan_parity.py $B/data/clean/warm/aware.tss 128 64 >> $O/scan_parity.txt 2>&1
SCAN_PROF=1 python3 scan_parity.py $B/data/clean/warm/jump_stick.tss 1024 64 >> $O/scan_parity.txt 2>&1
python3 ppo_gpu.py --task getup_imitate --style $ST --room $RB --load $B/data/clean/warm/jump_stick --envs 4096 --seconds 150 --out $O/fused > $O/fused.log 2>&1
echo "fused, 4096 envs: $(grep '"iter"' $O/fused.log | tail -1 | grep -o '"iter": [0-9]*\|"steps_per_s": [0-9]*\|"rollout_share": [0-9.]*\|"train_share": [0-9.]*\|"episode_return": [0-9.]*' | tr '\n' ' ') $(grep -m1 -i 'error\|Traceback' $O/fused.log)" > $O/bench.txt
echo done > $O/DONE
