#!/bin/bash
# Pod job: byte-mode deep-tree LM (depth 7) with TTT as context memory.
cd /w; TH=$(nproc); D=/w/repo/data; G=$D/faustus.txt,$D/tamburlaine2.txt,$D/spanish_tragedie.txt
for s in 600 1200 2400; do ./nanotss --data input.txt --mode byte --conv-mix 0 --threads $TH --seconds $s --save out/byte$s.bin > out/byte$s.log 2>&1; done
best=$(for s in 600 1200 2400; do grep '"eval"' out/byte$s.log | python3 -c "import json,sys;print(json.load(sys.stdin)['val_bpc'],$s)"; done | sort -n | head -1 | cut -d' ' -f2)
echo $best > out/best.txt; B=out/byte$best.bin
./nanotss --load $B --ttt-eval input.txt --threads $TH > out/val_nottt.jsonl
./nanotss --load $B --ttt-eval input.txt --threads $TH --ttt-lr 2e-4 --ttt-steps 2 > out/val_ttt.jsonl
./nanotss --load $B --ttt-eval input.txt --threads $TH --ttt-lr 2e-4 --ttt-steps 2 --ttt-control input.txt > out/val_ctrl.jsonl
./nanotss --load $B --ttt-fit $G --ttt-epochs 16 --ttt-lr 3e-4 --threads $TH > out/fit.jsonl
for cl in 2048 8192 32768 0; do ./nanotss --load $B --ttt-fit $G --ttt-frac 0.8 --ttt-ctxlen $cl --ttt-epochs 16 --ttt-lr 3e-4 --threads $TH > out/ctx$cl.jsonl; done
python3 /w/repo/experiments/recall_test.py ./nanotss $B input.txt --ttt-stream 0 --ttt-lr 3e-4 --ttt-steps 2 > out/recall_ttt.jsonl
python3 /w/repo/experiments/recall_test.py ./nanotss $B input.txt > out/recall_native.jsonl
