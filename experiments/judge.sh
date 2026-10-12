#!/bin/bash
# Judge-model fluency: BPC of each method's samples under a separate (larger) model, before and
# after that judge reads the same context with TTT. Usage: judge.sh BIN JUDGE CMPDIR THREADS
BIN=$1 J=$2 DIR=$3 TH=$4
for f in "$DIR"/*.jsonl; do b=$(basename "$f" .jsonl); m=${b%%.*}; play=${b#*.}; ctx="$DIR/$play.ctx"
  doc=$(mktemp); cat "$ctx" > "$doc"; python3 -c "import json,sys;sys.stdout.write(''.join('\n\n'+json.loads(l)['text'] for l in open(sys.argv[1])))" "$f" >> "$doc"
  frac=$(python3 -c "import os,sys;print(os.path.getsize(sys.argv[1])/os.path.getsize(sys.argv[2]))" "$ctx" "$doc")
  "$BIN" --load "$J" --ttt-fit "$doc" --ttt-frac "$frac" --ttt-epochs 8 --ttt-lr 3e-4 --threads "$TH" | python3 -c "
import json,sys
r={json.loads(l)['epoch']:json.loads(l)['query_bpc'] for l in sys.stdin}
print('$m $play', round(r[0],4), round(r[8],4))"
  rm -f "$doc"; done | sort
