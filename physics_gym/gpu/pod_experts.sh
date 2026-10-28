#!/bin/bash
# Batch job for a GPU pod (run from a clone of the repo, the file server already started on :8000 over /w/out): the Rooftops
# move experts (data/experts/rooftops.json; banks by tools/expert_bank.py, committed). Each: one clip, a float MLP from
# scratch (ppo_gpu.py --mlp), on 48 variants of its clip's room drawn by how often she fails each (--clip-focus with the
# hand-up held at none), with start noise (--init-noise), unaided, x2 strength. CONC of them side by side on the one GPU
# (4 used 13-25 GB each at 4 minibatches: 8 here), in waves until all are done. Snapshots of every checkpoint every
# 30 min (<name>_s<n>), each one's log as it goes; a watchdog stops the pod (runpodctl) DEADLINE s after this starts.
# The checkpoints are pulled and judged elsewhere (tools/expert_eval.py: held-out variants, and Rooftops itself).
#   EXPERTS (all in the manifest)  SECS (manifest's: 2400)  CONC (4)  ENVS (4096)  MB (8)  DEADLINE (6000)
G=/w/repo/physics_gym/gpu; B=/w/repo/physics_gym; O=${OUT:-/w/out}; mkdir -p $O
( sleep ${DEADLINE:-6000}; echo "deadline: stopping the pod" >> $O/stages.txt; runpodctl stop pod $RUNPOD_POD_ID ) > $O/watchdog.log 2>&1 &
fail() { echo "$1" > $O/FAILED; exit 1; }
{ nvidia-smi; python3 -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"; nproc; which runpodctl; } > $O/system.txt 2>&1
cd $G && gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c ../room.c -lm > $O/build.log 2>&1 || fail "build"
export TSS_FMAD=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -c "import gpu_env; gpu_env.cuda_lib(); print('built')" >> $O/build.log 2>&1 || fail "cuda build"
TSS_FMAD= python3 parity_env.py --n 64 --steps 300 --cases parkour,expert > $O/parity.txt 2>&1
# one line per expert: name clip clips.bin seconds init_noise mlp
python3 -c "
import json, sys; sys.path.insert(0, '$B/tools'); import expert_bank as eb
want = '${EXPERTS:-}'.split(',') if '${EXPERTS:-}' else None
for e in eb.experts('$B/data/experts/rooftops.json'):
    if not want or e['name'] in want: print(e['name'], e['clip'], e['clips'], ${SECS:-0} or e['seconds'], e['init_noise'], e['mlp'])
" > $O/experts.txt || fail "manifest"
( n=0; while sleep 1800; do n=$((n + 1)); for f in $O/*.mlp.pt; do [ -f "$f" ] && case $f in *_s[0-9]*) ;; *) b=${f%.mlp.pt}; cp $f ${b}_s$n.mlp.pt; cp $b.head ${b}_s$n.head;; esac; done; done ) &
SNAP=$!
train() {   # name clip clips seconds init_noise mlp
  echo "$1 started $(date -u +%T)" >> $O/stages.txt
  python3 $G/ppo_gpu.py --task getup_imitate --style $B/$3:$2 --room $B/data/experts/$1_train.json --steps 400 --strength 2 --start0 0.5 --drift 0.6 --tail 0.1 \
    --envs ${ENVS:-4096} --minibatches ${MB:-8} --mlp $6 --init-noise $5 --assist 0 --assist-auto --clip-focus 0.5 --seconds $4 --out $O/$1 > $O/$1.log 2>&1 \
    && echo "$1 done $(date -u +%T)" >> $O/stages.txt || echo "$1 FAILED $(date -u +%T)" >> $O/stages.txt
}
export -f train; export G B O ENVS MB
xargs -P ${CONC:-4} -L 1 bash -c 'train "$@"' _ < $O/experts.txt
kill $SNAP
echo done > $O/DONE
