#!/bin/bash
# Batch job for a CPU pod: build the body trainer and run an imitation curriculum for clip $CLIP, lowering the
# pelvis assist stage by stage ($ASSISTS, $STAGE_SECONDS each); every stage starts from the previous policy.
# Serves /w/out read-only on :8000; writes out/DONE at the end.
mkdir -p /w/out && cd /w
python3 -m http.server 8000 --directory /w/out >/dev/null 2>&1 &
B=/w/repo/body
cc -O3 -march=native -fopenmp $B/body.c $B/env.c $B/mref.c $B/policy.c $B/train.c -lm -o btrain 2> out/build.log
prev=""
for a in $ASSISTS; do
  ./btrain --threads $(nproc) --mref $B/data/motions_ref.bin --task $CLIP --assist $a --seconds $STAGE_SECONDS \
    --pairs 96 --sigma 0.01 --lr 0.001 --steps 300 $prev --out out/${CLIP}_a$a.pol > out/${CLIP}_a$a.log 2>&1
  echo "assist $a: $(grep done out/${CLIP}_a$a.log)" >> out/stages.log
  prev="--load out/${CLIP}_a$a.pol"
done
echo "exit 0" > out/DONE
sleep infinity
