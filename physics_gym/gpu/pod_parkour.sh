#!/bin/bash
# Batch job for a GPU pod (run from a clone of the repo, the file server already started on :8000 over /w/out): parkour at
# an athlete's strength. data/parkour_fit* (tools/merge_banks.py): the 50 PARC training clips her body can do with its
# muscles doubled, by tools/feasibility.py (reference motion against her strength: contact gaps, ballistic flight, joint
# torques), plus the CMU climbs but c88_02_15 (its flight is not free fall). The pilot (pod_parc.sh) left her at a 0.22
# hand-up and the climb runs at none: there a clip's start was drawn at random, and standing behind a clip while it went
# on paid (1 m of drift allowed), so neither learned whole clips. Here:
#   --strength 2   her muscles x2, strength and speed (env.h TaskCfg.strength): PARC's motions are parkour athletes';
#   --start0       this share of episodes from the clip's start, as eval_course.py judges;
#   --drift        the episode ends this far (m) off the reference (was 1.0 on a course);
#   --tail 0.1     each episode ends 0.1 s after its clip, as a time limit: most PARC clips end at a run (median 2.3 m/s),
#                  and holding the last frame for the rest of 13 s asked her to stop dead (eval_course.py now judges such
#                  clips as they end); the hand-up adapts on the share of clips finished;
#   --assist-pos-only 1  the hand-up only pulls along the reference path: the first run's policy (pk1, 80% of clips with a
#                  0.22 hand-up) passed none once the hand-up's rotational part was off, even at 0.5: it was being held
#                  upright, not balancing, which is why taking the hand-up away collapsed every run so far.
# Stage 1 adapts the hand-up down from ASSIST; stage 2 fades what it reached to none over DECAY s and trains on without
# (STAGE2 s in all). parity_env.py's parkour case first (GPU against CPU physics with the new settings); then
# tools/eval_course.py after each stage, each clip from its start alone at strength x2: the clips trained on, the
# feasible held-out PARC clips (data/parc_heldout_fit*) and all 64 held out.
#   SECONDS_T (7200)  FROM (parc1)  ASSIST (0.6)  DECAY (3600)  STAGE2 (5400)  START0 (0.5)  DRIFT (0.6)  OUT (/w/out)
#   ENVS (4096)  MB (4: minibatches per pass)  FMAD (0; 1: fused multiply-adds in the GPU physics, 13-38% faster, not bit for bit
#   with the CPU build; parity_env.py runs on the default build either way)
G=/w/repo/physics_gym/gpu; B=/w/repo/physics_gym; O=${OUT:-/w/out}; mkdir -p $O
TR=$B/data/parkour_fit; HF=$B/data/parc_heldout_fit; HA=$B/data/parc_heldout
CL=$(python3 -c "import json; print(','.join(json.load(open('$TR' + '_bank.json'))['clips']))")
K="--strength 2 --start0 ${START0:-0.5} --drift ${DRIFT:-0.6} --tail 0.1 --assist-pos-only 1"
{ nvidia-smi; python3 -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"; } > $O/system.txt 2>&1
cd $G && gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c ../room.c -lm > $O/build.log 2>&1
python3 -c "import gpu_env; gpu_env.cuda_lib(); gpu_env.build_cu('tss_scan.cu', 'libtss_scan.so'); print('built')" >> $O/build.log 2>&1
python3 parity_env.py --n 64 --steps 300 --cases parkour > $O/parity.txt 2>&1
[ "${FMAD:-0}" = 1 ] && export TSS_FMAD=1
ev() { python3 $B/tools/eval_course.py $1 ${TR}_bank.json $TR.bin 384 0 2; python3 $B/tools/eval_course.py $1 ${TR}_bank.json $TR.bin 384 0.2 2 1;
       python3 $B/tools/eval_course.py $1 ${HF}_bank.json $HF.bin 384 0 2; python3 $B/tools/eval_course.py $1 ${HA}_bank.json $HA.bin 384 0 2; }
python3 ppo_gpu.py --task getup_imitate --style $TR.bin:$CL --room ${TR}_bank.json --load $B/data/clean/warm/${FROM:-parc1} --steps 400 $K \
    --assist ${ASSIST:-0.6} --assist-auto --envs ${ENVS:-4096} --minibatches ${MB:-4} --seconds ${SECONDS_T:-7200} --out $O/pk1 > $O/train1.log 2>&1
ev $O/pk1 > $O/eval1.txt 2>&1
A2=$(grep '"assist"' $O/train1.log | tail -1 | python3 -c "import sys, json; print(max(0.005, json.loads(sys.stdin.read())['checkpoint_assist']))")
echo "stage 2 from hand-up $A2" >> $O/eval1.txt
python3 ppo_gpu.py --task getup_imitate --style $TR.bin:$CL --room ${TR}_bank.json --load $O/pk1 --steps 400 $K \
    --assist $A2 --assist-decay ${DECAY:-3600} --envs ${ENVS:-4096} --minibatches ${MB:-4} --seconds ${STAGE2:-5400} --out $O/pk > $O/train.log 2>&1
ev $O/pk > $O/eval.txt 2>&1
for f in pk1 pk; do [ -f $O/$f.tss ] && python3 $B/tools/tss2json.py $O/$f $O/$f.json; done
echo done > $O/DONE
