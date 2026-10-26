#!/bin/bash
# Batch job for a GPU pod (run from a clone of the repo, the file server already started on :8000 over /w/out): jumps
# chained, as the viewer plays them. The standing-jump courses (data/jump_bank_standing.json) with env.h TaskCfg.chain on:
# after each landing, once she is steady, she is carried back onto the takeoff platform aimed by her feet (give or take
# CHAIN_YAW) and the clip starts again, so every jump after an episode's first starts from a real landing. From
# jump_stick (stage 3), episodes of 30 s, the hand-up adapting down from ASSIST.
# Before and after: tools/eval_chain.py (jumps landed in a row from each clip's start, the viewer's aim and a 0.25 rad error)
# and tools/eval_jump.py (single jumps). jump_stick on the CPU: 1.19 jumps in a row on average, 27% reaching 2, none 4.
#   SECONDS_T (4500)  FROM (jump_stick)  ASSIST (0.05)  CHAIN_YAW (0.25)
G=/w/repo/physics_gym/gpu; B=/w/repo/physics_gym; O=/w/out; ST=$B/data/cmu_jump.bin:fjump1,fjump2; RB=$B/data/jump_bank_standing.json
{ nvidia-smi; python3 -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"; } > $O/system.txt 2>&1
cd $G && gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c ../room.c -lm > $O/build.log 2>&1
python3 -c "import gpu_env; gpu_env.cuda_lib(); gpu_env.build_cu('tss_scan.cu', 'libtss_scan.so'); print('built')" >> $O/build.log 2>&1
FROM=$B/data/clean/warm/${FROM:-jump_stick}
python3 $B/tools/eval_chain.py $FROM $RB 384 30 0 > $O/eval_before.txt 2>&1
python3 ppo_gpu.py --task getup_imitate --style $ST --room $RB --load $FROM --chain 1 --chain-yaw ${CHAIN_YAW:-0.25} --steps 900 \
    --assist ${ASSIST:-0.05} --assist-auto --envs 4096 --seconds ${SECONDS_T:-4500} --out $O/chain > $O/train.log 2>&1
python3 $B/tools/eval_chain.py $O/chain $RB 384 30 0 > $O/eval.txt 2>&1
python3 $B/tools/eval_chain.py $O/chain $RB 384 30 0.25 >> $O/eval.txt 2>&1
python3 $B/tools/eval_jump.py $O/chain $RB 384 >> $O/eval.txt 2>&1
[ -f $O/chain.tss ] && python3 $B/tools/tss2json.py $O/chain $O/chain.json
echo done > $O/DONE
