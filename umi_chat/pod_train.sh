#!/bin/bash
# umi-1M-chat training run for the RunPod CPU pod "UMI" (8 vCPU).
# Clones this branch, builds, trains, evaluates (tuning the hybrid n-gram /
# conversation-graph weights), runs a multi-turn demo, and serves /root/ckpt
# on port 8888 (http) so the weights can be fetched through the RunPod proxy.
# Stage 2 then pulls the branch again and trains the umi-smart-internet widget
# model on data_web/, and compiles demo pages with both models.
# TRAIN_SECONDS / WEB_SECONDS (env) set the wall-clock training budgets.
export DEBIAN_FRONTEND=noninteractive GIT_TERMINAL_PROMPT=0
SECS=${TRAIN_SECONDS:-10800}
WEB_SECS=${WEB_SECONDS:-5400}
BRANCH=${UMI_BRANCH:-claude/jolly-wozniak-hodvrx}
OUT=/root/ckpt/umi-1M-chat.bin
mkdir -p /root/ckpt; exec > >(tee -a /root/ckpt/train.log) 2>&1
(cd /root/ckpt && nohup python3 -m http.server 8888 >/dev/null 2>&1 &)
echo "=== START $(date -u +%FT%TZ)  $(nproc) vCPU: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)"
apt-get update -qq && apt-get install -y -qq build-essential git >/dev/null
cd /root && for i in 1 2 3 4 5; do rm -rf umi2; git clone -q --depth 1 -b "$BRANCH" https://github.com/mage-uwu/umi2.git && break; sleep $((i * 5)); done
cd /root/umi2/umi_chat && git log --oneline -1
cc -O3 -march=native -fopenmp umi_chat.c -lm -o umi_chat || cc -O3 -fopenmp umi_chat.c -lm -o umi_chat || { echo "BUILD FAILED"; exit 1; }
(cd .. && cc -O3 -march=native -fopenmp nanotss_blt_mlm.c -lm -o nanotss && ./nanotss --check-mlm | tail -n 1)
echo "=== TRAIN ${SECS}s, $(nproc) threads, $(date -u +%T)"
./umi_chat train --data data --seconds "$SECS" --entropy-seconds 120 --threads "$(nproc)" --save "$OUT"
echo "=== EVAL $(date -u +%T)"
./umi_chat eval --data data --load "$OUT" --threads "$(nproc)" --eval-n 2000 | tee /root/ckpt/eval.txt
read -r BETA GAMMA < <(sed -n 's/^use: chat --beta \([0-9.]*\) --gamma \([0-9.]*\)$/\1 \2/p' /root/ckpt/eval.txt)
echo "${BETA:-8} ${GAMMA:-4}" > /root/ckpt/hybrid_weights.txt
echo "=== DEMO (beta ${BETA:-8} gamma ${GAMMA:-4})"
for convo in \
  "hi umi!|my name is Sam|what's my name?|how do i make rice not stick?|wait, really?|ok and then?|haha thanks|bye" \
  "who are you?|why are you so small?|tell me more|lol|what's your favorite food?|why?" \
  "i'm feeling kind of down today|work stuff mostly|yeah|thanks, that helps" \
  "who was napoleon?|why was he exiled?|and then what?|cool. how do i boil an egg?"; do
  ./umi_chat chat --data data --load "$OUT" --beta "${BETA:-8}" --gamma "${GAMMA:-4}" -v --ask "$convo"; echo
done
echo "=== STAGE 1 DONE $(date -u +%FT%TZ)"
echo "=== STAGE 2: umi-smart-internet $(date -u +%T)"
WEB=/root/ckpt/umi-smart-internet.bin
for i in 1 2 3 4 5; do git pull -q --ff-only origin "$BRANCH" && break; sleep $((i * 5)); done; git log --oneline -1
cc -O3 -march=native -fopenmp umi_chat.c -lm -o umi_chat || { echo "BUILD FAILED (stage 2)"; exit 1; }
if [ -f data_web/train.tsv ]; then
  ./umi_chat train --data data_web --seconds "$WEB_SECS" --entropy-seconds 60 --threads "$(nproc)" --save "$WEB"
  ./umi_chat eval --data data_web --load "$WEB" --threads "$(nproc)" --eval-n 1000 | tee /root/ckpt/eval_web.txt
  i=0
  for convo in \
    "hi umi!|my name is Sam|can you show me a recipe card for pancakes?|show me a dashboard for my week|thanks!" \
    "give me a bar chart of something fun|what's an http 404?|show me a colour palette for a calm website|bye" \
    "i want to build better habits|make me a habit tracker|explain photosynthesis as flashcards|cool, thanks umi"; do
    i=$((i+1)); ./umi_chat chat --data data --load "$OUT" --beta "${BETA:-8}" --gamma "${GAMMA:-4}" --web data_web --web-load "$WEB" --html "/root/ckpt/demo_$i.html" -v --ask "$convo"; echo
  done
else echo "no data_web/ on the branch; skipping stage 2"; fi
sha256sum /root/ckpt/*.bin > /root/ckpt/SHA256SUMS
echo "=== ALL DONE $(date -u +%FT%TZ)"
