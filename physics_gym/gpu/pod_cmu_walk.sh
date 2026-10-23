#!/bin/bash
# Batch job for a GPU pod (run from a clone of the repo, the file server already started on :8000 over /w/out):
# the clean walk prior (ROADMAP.md M1). Build; check the CUDA environments against the CPU ones (go-to, balance, and
# grip in the stage-1 room; the room drops); then train imitation of the CMU walk cycle (data/clean/cmu_gait.bin,
# tools/cmu_cycle.py) from scratch on the open floor under the CMU joint limits and muscles, the hand-up assist
# stepped 0.5 -> 0.3 -> 0.1 -> 0; then go-to in the stage-1 room from that prior. No Bandai Namco data anywhere.
# Each stage's best policy (.tss/.head) and its viewer JSON land in /w/out; out/DONE at the end.
G=/w/repo/physics_gym/gpu; B=/w/repo/physics_gym; O=/w/out; M=$B/data/clean/cmu_gait.bin
{ nvidia-smi; nvcc --version; python3 -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"; } > $O/system.txt 2>&1
cd $G && gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c ../room.c -lm > $O/build.log 2>&1
python3 -c "import gpu_env; gpu_env.cuda_lib(); print('cuda environments built')" >> $O/build.log 2>&1
python3 parity_env.py --n 128 --steps 300 --cases goto,balance,grip > $O/parity_env.txt 2>&1
ENVS=${ENVS:-4096}; S=${STAGE:-600}
python3 ppo_gpu.py --task walk --mref $M --room '' --assist 0.5 --envs $ENVS --seconds $S --out $O/walk_a > $O/train_walk_a.log 2>&1
python3 ppo_gpu.py --task walk --mref $M --room '' --load $O/walk_a --assist 0.3 --envs $ENVS --seconds $S --out $O/walk_b > $O/train_walk_b.log 2>&1
python3 ppo_gpu.py --task walk --mref $M --room '' --load $O/walk_b --assist 0.1 --envs $ENVS --seconds $S --out $O/walk_c > $O/train_walk_c.log 2>&1
python3 ppo_gpu.py --task walk --mref $M --room '' --load $O/walk_c --assist 0 --envs $ENVS --seconds $((2 * S)) --out $O/walk_d > $O/train_walk_d.log 2>&1
python3 ppo_gpu.py --task goto --mref $M --load $O/walk_d --assist 0.1 --goal-cone 0.5 --envs $ENVS --seconds $((2 * S)) --out $O/goto_clean > $O/train_goto_clean.log 2>&1
for s in walk_a walk_b walk_c walk_d goto_clean; do [ -f $O/$s.tss ] && python3 $B/tools/tss2json.py $O/$s $O/$s.json; done
[ -f $O/goto_clean.tss ] && python3 eval_goto.py $O/goto_clean --mref $M --n 256 > $O/eval_goto_clean.txt 2>&1
echo done > $O/DONE
