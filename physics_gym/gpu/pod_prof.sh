#!/bin/bash
# Batch job for a GPU pod (run from a clone of the repo, the file server already started on :8000 over /w/out): where the
# fused trainer's time goes. The fused forward + backward profiled by operator (scan_parity.py, SCAN_PROF) at the
# trainer's minibatch size, then one trainer iteration profiled (ppo_gpu.py --profile) on the jump course.
G=/w/repo/physics_gym/gpu; B=/w/repo/physics_gym; O=/w/out; ST=$B/data/cmu_jump.bin:fjump1,fjump2; RB=$B/data/jump_bank_standing.json
cd $G && gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c ../room.c -lm > $O/build.log 2>&1
python3 -c "import gpu_env; gpu_env.cuda_lib(); gpu_env.build_cu('tss_scan.cu', 'libtss_scan.so'); print('built')" >> $O/build.log 2>&1
SCAN_PROF=1 python3 scan_parity.py $B/data/clean/warm/jump_stick.tss 1024 64 > $O/scan_prof.txt 2>&1
python3 ppo_gpu.py --task getup_imitate --style $ST --room $RB --load $B/data/clean/warm/jump_stick --envs 4096 --seconds 60 --profile 2 --out $O/prof > $O/prof.log 2>&1
echo done > $O/DONE
