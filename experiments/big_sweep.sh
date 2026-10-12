#!/bin/bash
# Pod job: find the training budget where the 4.2M width-192 model generalizes best.
cd /w; TH=$(nproc)
for s in 300 600 1200; do
  ./nanotss --data input.txt --width 192 --threads $TH --seconds $s --save out/big$s.bin > out/big$s.log 2>&1
done
