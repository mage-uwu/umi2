#!/bin/bash
# Fine-tune umi-smart-internet on multi-turn widget conversations, on the RunPod CPU pod "UMI".
# Starts from the committed widget model, trains on data_web/ (single-turn widgets + widget
# conversation trees), evaluates (single-turn, follow-up, topic switch) and serves /root/ckpt
# on port 8888 so the weights can be fetched through the RunPod proxy.
# FT_SECONDS (env) sets the wall-clock budget.
export DEBIAN_FRONTEND=noninteractive GIT_TERMINAL_PROMPT=0
SECS=${FT_SECONDS:-5400}
BRANCH=${UMI_BRANCH:-claude/jolly-wozniak-hodvrx}
OUT=/root/ckpt/umi-smart-internet.bin
mkdir -p /root/ckpt; exec > >(tee -a /root/ckpt/train.log) 2>&1
(cd /root/ckpt && nohup python3 -m http.server 8888 >/dev/null 2>&1 &)
echo "=== START $(date -u +%FT%TZ)  $(nproc) vCPU: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)"
apt-get update -qq && apt-get install -y -qq build-essential git >/dev/null
cd /root && for i in 1 2 3 4 5; do rm -rf umi2; git clone -q --depth 1 -b "$BRANCH" https://github.com/mage-uwu/umi2.git && break; sleep $((i * 5)); done
cd /root/umi2/umi_chat && git log --oneline -1
cc -O3 -march=native -fopenmp umi_chat.c -lm -o umi_chat || { echo "BUILD FAILED"; exit 1; }
echo "=== FINE-TUNE ${SECS}s, $(nproc) threads, $(date -u +%T)"
./umi_chat train --data data_web --load weights/umi-smart-internet/umi-smart-internet.bin --seconds "$SECS" --lr 0.0015 --threads "$(nproc)" --save "$OUT"
echo "=== EVAL $(date -u +%T)"
./umi_chat eval --data data_web --load "$OUT" --threads "$(nproc)" --eval-n 1500 | tee /root/ckpt/eval_web.txt
sha256sum "$OUT" > "$OUT.sha256"
echo "=== ALL DONE $(date -u +%FT%TZ)"
