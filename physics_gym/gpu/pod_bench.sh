#!/bin/bash
# Batch job for a GPU pod (run from a clone of the repo, the file server already started on :8000 over /w/out):
# how fast the GPU trainer goes, and where its time goes. The jump course (from data/clean/warm/jump_stick) for 150 s
# per setting of environments, minibatches and epochs; torch.profiler on one iteration of the first and fourth.
G=/w/repo/physics_gym/gpu; B=/w/repo/physics_gym; O=/w/out; ST=$B/data/cmu_jump.bin:fjump1,fjump2; RB=$B/data/jump_bank_standing.json
{ nvidia-smi; python3 -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"; } > $O/system.txt 2>&1
cd $G && gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c ../room.c -lm > $O/build.log 2>&1
python3 -c "import gpu_env; gpu_env.cuda_lib(); print('cuda environments built')" >> $O/build.log 2>&1
run() {   # envs minibatches epochs [profile iteration]
  n=b_$1_$2_$3; python3 ppo_gpu.py --task getup_imitate --style $ST --room $RB --load $B/data/clean/warm/jump_stick --envs $1 --minibatches $2 --epochs $3 --seconds ${SECS:-150} ${4:+--profile $4} --out $O/$n > $O/$n.log 2>&1
  echo "envs $1 minibatches $2 epochs $3: $(grep '"iter"' $O/$n.log | tail -1 | grep -o '"iter": [0-9]*\|"steps_per_s": [0-9]*\|"rollout_share": [0-9.]*\|"train_share": [0-9.]*\|"episode_return": [0-9.]*' | tr '\n' ' ') $(grep -m1 -i 'error\|Traceback' $O/$n.log)" >> $O/bench.txt
}
run 4096 4 4 3; run 4096 1 4; run 16384 4 4; run 16384 1 4 2; run 16384 2 2; run 32768 2 2
echo done > $O/DONE
