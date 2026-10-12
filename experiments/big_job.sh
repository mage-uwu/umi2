#!/bin/bash
# Pod job (run by pod_run.sh from /w): train the 4.2M-parameter width-192 model, then evaluate it.
cd /w; TH=$(nproc); B=./nanotss; D=/w/repo/data
$B --data input.txt --width 192 --threads $TH --seconds ${RUN_SECONDS:-7200} --save out/big2h.bin > out/big2h.log 2>&1
$B --load out/big2h.bin --ttt-eval input.txt --threads $TH > out/big_val_nottt.jsonl
$B --load out/big2h.bin --ttt-eval input.txt --threads $TH --ttt-lr 2e-4 --ttt-steps 2 > out/big_val_ttt.jsonl
$B --load out/big2h.bin --ttt-fit $D/faustus.txt,$D/tamburlaine2.txt,$D/spanish_tragedie.txt --ttt-epochs 16 --ttt-lr 3e-4 --threads $TH > out/big_fit.jsonl
for lr in 1e-4 2e-4; do $B --load out/big2h.bin --ttt-fit $D/faustus.txt,$D/tamburlaine2.txt,$D/spanish_tragedie.txt --ttt-epochs 16 --ttt-lr $lr --threads $TH > out/big_fit_lr$lr.jsonl; done
