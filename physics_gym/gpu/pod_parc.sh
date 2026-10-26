#!/bin/bash
# Batch job for a GPU pod (run from a clone of the repo, the file server already started on :8000 over /w/out): parkour from
# PARC (Xu, Shi, Yin, Peng, SIGGRAPH 2025; huggingface.co/datasets/mxucg/PARC, MIT), its source motions retargeted with
# tools/parc_to_ref.py: 64 clips to train on (data/parc_train*: all 6 wall climbs, all 11 vaults, 30 platform and 17 running
# clips, each on the terrain it was made for), 64 held out (data/parc_heldout*: 35 platform and 29 running clips, motions
# and terrains it never sees). She tracks each clip in its room (task getup_imitate: env.c course mode) from a random
# moment of it, from FROM; stage 1 with the hand-up adapting down from ASSIST, stage 2 with what it reached faded to none
# (DECAY s) and trained on without (STAGE2 s in all), as pod_climb.sh.
# tools/eval_course.py after each stage, on the clips trained on and on the held-out ones (each from its start, alone; done
# when she is where the clip ends a second after it).
#   SECONDS_T (3600)  FROM (jump_stick)  ASSIST (0.3)  DECAY (1200)  STAGE2 (2700)
G=/w/repo/physics_gym/gpu; B=/w/repo/physics_gym; O=/w/out
TR=$B/data/parc_train; HO=$B/data/parc_heldout
CL=$(python3 -c "import json; print(','.join(json.load(open('$TR' + '_bank.json'))['clips']))")
{ nvidia-smi; python3 -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"; } > $O/system.txt 2>&1
cd $G && gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c ../room.c -lm > $O/build.log 2>&1
python3 -c "import gpu_env; gpu_env.cuda_lib(); gpu_env.build_cu('tss_scan.cu', 'libtss_scan.so'); print('built')" >> $O/build.log 2>&1
python3 ppo_gpu.py --task getup_imitate --style $TR.bin:$CL --room ${TR}_bank.json --load $B/data/clean/warm/${FROM:-jump_stick} --steps 400 \
    --assist ${ASSIST:-0.3} --assist-auto --envs 4096 --seconds ${SECONDS_T:-3600} --out $O/parc1 > $O/train1.log 2>&1
{ python3 $B/tools/eval_course.py $O/parc1 ${TR}_bank.json $TR.bin 384; python3 $B/tools/eval_course.py $O/parc1 ${HO}_bank.json $HO.bin 384; } > $O/eval1.txt 2>&1
A2=$(grep '"assist"' $O/train1.log | tail -1 | python3 -c "import sys, json; print(max(0.005, json.loads(sys.stdin.read())['checkpoint_assist']))")
echo "stage 2 from hand-up $A2" >> $O/eval1.txt
python3 ppo_gpu.py --task getup_imitate --style $TR.bin:$CL --room ${TR}_bank.json --load $O/parc1 --steps 400 \
    --assist $A2 --assist-decay ${DECAY:-1200} --envs 4096 --seconds ${STAGE2:-2700} --out $O/parc > $O/train.log 2>&1
{ python3 $B/tools/eval_course.py $O/parc ${TR}_bank.json $TR.bin 384; python3 $B/tools/eval_course.py $O/parc ${HO}_bank.json $HO.bin 384;
  python3 $B/tools/eval_course.py $O/parc ${HO}_bank.json $HO.bin 384 0.05; } > $O/eval.txt 2>&1
for f in parc1 parc; do [ -f $O/$f.tss ] && python3 $B/tools/tss2json.py $O/$f $O/$f.json; done
echo done > $O/DONE
