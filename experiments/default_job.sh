#!/bin/bash
# Pod job: 20-minute baseline of the new default architecture (byte mode, pure SSM, per-tree keys/queries).
cd /w; TH=$(nproc); D=/w/repo/data; G=$D/faustus.txt,$D/tamburlaine2.txt,$D/spanish_tragedie.txt; B=out/default1200.bin
./nanotss --data input.txt --threads $TH --seconds ${RUN_SECONDS:-1200} ${TRAIN_ARGS} --save $B > out/default1200.log 2>&1
./nanotss --load $B --ttt-eval input.txt --threads $TH > out/val_nottt.jsonl
./nanotss --load $B --ttt-eval input.txt --threads $TH --match 1 > out/val_match.jsonl
./nanotss --load $B --ttt-eval input.txt --threads $TH --ttt-lr 2e-4 --ttt-steps 2 > out/val_ttt2.jsonl
./nanotss --load $B --ttt-eval input.txt --threads $TH --ttt-lr 2e-4 --ttt-steps 1 --match 1 > out/val_session.jsonl
./nanotss --load $B --ttt-eval input.txt --threads $TH --ttt-lr 2e-4 --ttt-steps 2 --ttt-control input.txt > out/val_ctrl.jsonl
./nanotss --load $B --ttt-fit $G --ttt-epochs 8 --ttt-lr 3e-4 --threads $TH > out/fit.jsonl
./nanotss --load $B --ttt-fit $G --ttt-epochs 8 --ttt-lr 3e-4 --threads $TH --match 1 > out/fit_match.jsonl
./nanotss --load $B --ttt-fit $G --ttt-frac 0.8 --ttt-epochs 8 --ttt-lr 3e-4 --threads $TH > out/ctx0.jsonl
./nanotss --load $B --ttt-fit $G --ttt-frac 0.8 --ttt-epochs 8 --ttt-lr 3e-4 --threads $TH --match 1 > out/ctx0_match.jsonl
python3 /w/repo/experiments/recall_test.py ./nanotss $B input.txt --ttt-stream 0 --ttt-lr 3e-4 --ttt-steps 2 --match 1 > out/recall_ttt_match.jsonl
