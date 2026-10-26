#!/bin/bash
# Batch job for a GPU pod (run from a clone of the repo, the file server already started on :8000 over /w/out): go-to with
# no gait prior (task goto_free: no clip plays, the actions act around the stand pose), its style from a learned motion
# discriminator over the CMU locomotion set (data/cmu_loco.bin, tools/amp_set.py), starting 80% of episodes from its
# frames. From scratch unless FROM is set; a hand-up (ASSIST, adapted down as episodes last) while she learns to stay up.
# Then eval_free.py: alone (no hand-up) and with a 10% hand-up, 256 episodes of 30 s.
#   SECS (7200)  FROM (a checkpoint prefix)  ASSIST (0.3)  AMPW (0.7, the style reward's weight)  OUT (free)
G=/w/repo/physics_gym/gpu; B=/w/repo/physics_gym; O=/w/out
SECS=${SECS:-7200}; ASSIST=${ASSIST:-0.3}; AMPW=${AMPW:-0.7}; OUT=${OUT:-free}
{ nvidia-smi; python3 -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"; } > $O/system.txt 2>&1
cd $G && gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c ../room.c -lm > $O/build.log 2>&1
python3 -c "import gpu_env; gpu_env.cuda_lib(); gpu_env.build_cu('tss_scan.cu', 'libtss_scan.so'); print('built')" >> $O/build.log 2>&1
LOAD=""; [ -n "$FROM" ] && LOAD="--load $FROM"
python3 ppo_gpu.py --task goto_free --amp $B/data/cmu_loco.bin --amp-w $AMPW --room '' --envs 4096 --seconds $SECS \
    --assist $ASSIST --assist-auto $LOAD --out $O/$OUT > $O/$OUT.log 2>&1
python3 eval_free.py $O/$OUT 256 30 0 > $O/eval.txt 2>&1
python3 eval_free.py $O/$OUT 256 30 0.1 >> $O/eval.txt 2>&1
echo done > $O/DONE
