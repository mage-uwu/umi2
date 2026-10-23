#!/bin/bash
# Batch job for a GPU pod (run from a clone of the repo, the file server already started on :8000 over /w/out):
# the clean walk prior (ROADMAP.md M1). Build; check the CUDA environments against the CPU ones (go-to, balance, and
# grip in the stage-1 room; the room drops); then train imitation of the CMU walk cycle (data/clean/cmu_gait.bin,
# tools/cmu_cycle.py) from scratch on the open floor under the CMU joint limits and muscles, the hand-up assist
# adapting down from 0.5; then go-to across the bank of 32 stage-1 room variants (one per world) from that prior, its assist adapting down from 0.2. No Bandai Namco data anywhere.
# Each stage's best policy (.tss/.head) and its viewer JSON land in /w/out; out/DONE at the end.
G=/w/repo/physics_gym/gpu; B=/w/repo/physics_gym; O=/w/out; M=$B/data/clean/cmu_gait.bin
{ nvidia-smi; nvcc --version; python3 -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"; } > $O/system.txt 2>&1
cd $G && gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c ../room.c -lm > $O/build.log 2>&1
python3 -c "import gpu_env; gpu_env.cuda_lib(); print('cuda environments built')" >> $O/build.log 2>&1
(cd $B && python3 room.py --bank 32 --out rooms_bank.json) >> $O/build.log 2>&1
python3 parity_env.py --n 128 --steps 300 --cases goto,balance,grip,bank > $O/parity_env.txt 2>&1
ENVS=${ENVS:-4096}; WALK=${WALK_S:-2700}; GOTO=${GOTO_S:-2400}
# the assist adapts (ppo_gpu --assist-auto): it comes down only while she lasts most of each episode, and goes back up
# if she starts falling, so the curriculum never drops her off a cliff (run 3: every fixed step-down collapsed)
# WALK_LOAD (a policy prefix in the repo, e.g. data/clean/warm/walk_cmu) continues an earlier walk prior from WALK_ASSIST
WA=${WALK_ASSIST:-0.5}; WL=""; [ -n "$WALK_LOAD" ] && WL="--load $B/$WALK_LOAD"
python3 ppo_gpu.py --task walk --mref $M --room '' $WL --assist $WA --assist-auto --envs $ENVS --seconds $WALK --out $O/walk > $O/train_walk.log 2>&1
python3 ppo_gpu.py --task goto --mref $M --room $B/rooms_bank.json --load $O/walk --assist 0.2 --assist-auto --goal-cone 0.5 --envs $ENVS --seconds $GOTO --out $O/goto_clean > $O/train_goto_clean.log 2>&1
for s in walk goto_clean; do [ -f $O/$s.tss ] && python3 $B/tools/tss2json.py $O/$s $O/$s.json; done
[ -f $O/goto_clean.tss ] && python3 eval_goto.py $O/goto_clean --mref $M --n 256 > $O/eval_goto_clean.txt 2>&1
echo done > $O/DONE
