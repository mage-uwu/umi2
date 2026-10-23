#!/bin/bash
# Batch job for a GPU pod (run from a clone of the repo, the file server already started on :8000 over /w/out):
# build, check the CUDA environments against the CPU ones (go-to and the room), then train go-to on the GPU in the
# room: warm-started from the walk policy, with the hand-up assist stepped down 0.15 -> 0.05 -> 0. Each stage's best
# policy (.tss/.head) and its viewer JSON land in /w/out; out/DONE at the end.
G=/w/repo/physics_gym/gpu; B=/w/repo/physics_gym; O=/w/out
{ nvidia-smi; nvcc --version; python3 -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"; } > $O/system.txt 2>&1
cd $G && gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c ../room.c -lm > $O/build.log 2>&1
python3 -c "import gpu_env; gpu_env.cuda_lib(); print('cuda environments built')" >> $O/build.log 2>&1
python3 parity_env.py --n 32 --steps 300 --cases goto,balance > $O/parity_env.txt 2>&1
ENVS=${ENVS:-4096}; A=${STAGE_A:-900}; Bs=${STAGE_B:-900}; C=${STAGE_C:-2700}
python3 ppo_gpu.py --task goto --load $B/data/warm/walk --assist 0.15 --envs $ENVS --seconds $A --out $O/goto_a > $O/train_a.log 2>&1
python3 ppo_gpu.py --task goto --load $O/goto_a --assist 0.05 --envs $ENVS --seconds $Bs --out $O/goto_b > $O/train_b.log 2>&1
python3 ppo_gpu.py --task goto --load $O/goto_b --assist 0 --envs $ENVS --seconds $C --out $O/goto_c > $O/train_c.log 2>&1
for s in a b c; do [ -f $O/goto_$s.tss ] && python3 $B/tools/tss2json.py $O/goto_$s $O/goto_$s.json && python3 eval_goto.py $O/goto_$s --n 256 > $O/eval_$s.txt 2>&1; done
echo done > $O/DONE
