#!/usr/bin/env python3
"""Native long-range recall: passage P, D bytes of other text, then P again, streamed through
the recurrent state with no test-time training. Reports BPC on the second copy of P against the
first copy; a model that remembers P across D bytes scores the second copy much lower.
Usage: recall_test.py BIN CKPT TEXT [extra flags...]   (TEXT: the last 10% is used, unseen in training)"""
import json, os, subprocess, sys, tempfile
binp, ck, text = sys.argv[1:4]; extra = sys.argv[4:]
raw = open(text, 'rb').read(); val = raw[int(len(raw) * .9):]
P, DS, NP = 256, [0, 256, 1024, 4096, 16384, 65536], 6
tmp = tempfile.mkdtemp()
for D in DS:
    r1, r2 = [], []
    for k in range(NP):
        a = (k * 9973) % (len(val) - P - 1)
        pas = val[a:a + P]; filler = (val[a + P:] + val[:a])[:D]  # other unseen text, never P itself
        lead = val[(a + len(val) // 2) % (len(val) - 64):][:64]
        doc = lead + pas + filler + pas; f = os.path.join(tmp, f'd{D}_{k}.txt'); open(f, 'wb').write(doc)
        s1, s2 = len(lead), len(lead) + P + D
        out = subprocess.run([binp, '--load', ck, '--ttt-eval', f, '--ttt-offset', '0', '--ttt-stream', '1',
                              '--ttt-range', f'{s1}:{s1+P}', '--ttt-range', f'{s2}:{s2+P}', '--threads', '1'] + extra,
                             capture_output=True, text=True).stdout
        rs = [json.loads(l)['bpc'] for l in out.splitlines() if '"ttt_range"' in l]
        r1.append(rs[0]); r2.append(rs[1])
    print(json.dumps({'gap': D, 'first_bpc': round(sum(r1) / NP, 4), 'second_bpc': round(sum(r2) / NP, 4),
                      'recall_gain': round((sum(r1) - sum(r2)) / NP, 4)}), flush=True)
