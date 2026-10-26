#!/bin/bash
# Batch job for a GPU pod (run from a clone of the repo, the file server already started on :8000 over /w/out): climbing.
# The CMU climbs of data/cmu_climb.bin (tools/cmu_fetch.py): up three steps (13_35, 13_36), up a stepstool (40_06), standing
# jumps over it (40_06, both ways), hoisting up a 0.6 m ledge with the hands (81_14, 81_15: climbing down it, played back),
# a standing jump up onto a 0.7 m platform (88_02). Each room of data/climb_bank.json (tools/climb_course.mjs) is built round
# one clip: the boxes its feet stand on, the stool it clears. She tracks the clip in its room (task getup_imitate: env.c
# course mode) from a random moment of it, from FROM (jump_stick, the jump tracker), the hand-up adapting down from ASSIST.
# Stage 2 in the same pod: the hand-up stage 1 reached faded to none on a schedule (DECAY s), then STAGE2 s in all without it
# (the jump chains stalled at a 1% hand-up under the adapting schedule, and leaned on it).
# Then tools/eval_course.py after each stage: each clip from its start, alone; done when she stands where the clip ends a
# second after it.
#   SECONDS_T (3600)  FROM (jump_stick)  ASSIST (0.3)  DECAY (1200)  STAGE2 (2700)
G=/w/repo/physics_gym/gpu; B=/w/repo/physics_gym; O=/w/out
CL=c13_35_30,c13_36_60,c40_06_30,c40_06_930,c40_06_1005,c81_14r,c81_15r,c88_02_15; ST=$B/data/cmu_climb.bin:$CL; RB=$B/data/climb_bank.json
{ nvidia-smi; python3 -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"; } > $O/system.txt 2>&1
cd $G && gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c ../room.c -lm > $O/build.log 2>&1
python3 -c "import gpu_env; gpu_env.cuda_lib(); gpu_env.build_cu('tss_scan.cu', 'libtss_scan.so'); print('built')" >> $O/build.log 2>&1
python3 ppo_gpu.py --task getup_imitate --style $ST --room $RB --load $B/data/clean/warm/${FROM:-jump_stick} --steps 400 \
    --assist ${ASSIST:-0.3} --assist-auto --envs 4096 --seconds ${SECONDS_T:-3600} --out $O/climb1 > $O/train1.log 2>&1
python3 $B/tools/eval_course.py $O/climb1 $RB $B/data/cmu_climb.bin 384 > $O/eval1.txt 2>&1
A2=$(grep '"assist"' $O/train1.log | tail -1 | python3 -c "import sys, json; print(max(0.005, json.loads(sys.stdin.read())['checkpoint_assist']))")
echo "stage 2 from hand-up $A2" >> $O/eval1.txt
python3 ppo_gpu.py --task getup_imitate --style $ST --room $RB --load $O/climb1 --steps 400 \
    --assist $A2 --assist-decay ${DECAY:-1200} --envs 4096 --seconds ${STAGE2:-2700} --out $O/climb > $O/train.log 2>&1
python3 $B/tools/eval_course.py $O/climb $RB $B/data/cmu_climb.bin 384 > $O/eval.txt 2>&1
for f in climb1 climb; do [ -f $O/$f.tss ] && python3 $B/tools/tss2json.py $O/$f $O/$f.json; done
echo done > $O/DONE
