#!/usr/bin/env python3
"""merge_banks.py: clips and their course rooms from several banks into one (env.c course mode: room r for clip r).
    python3 tools/merge_banks.py OUT.bin OUT_bank.json CLIPS.bin:BANK.json[:FEASIBLE.json] [...] [--drop name,...]
With a FEASIBLE.json (tools/feasibility.py --json), only the clips it passes are taken."""
import json, struct, sys, numpy as np
sys.path.insert(0, __file__.rsplit('/', 2)[0] + '/gpu'); import gpu_env as ge
out, bank_out = sys.argv[1], sys.argv[2]; drop = set(sys.argv[sys.argv.index('--drop') + 1].split(',')) if '--drop' in sys.argv else set()
srcs = [a for a in sys.argv[3:] if ':' in a]; clips, rooms = [], []
for s in srcs:
    parts = s.split(':'); C = ge.load_mref(parts[0]); doc = json.load(open(parts[1])); ok = json.load(open(parts[2])) if len(parts) > 2 else None
    for r in doc['rooms']:
        k = r['clip']
        if k in drop or (ok is not None and not ok[k]['pass']): continue
        clips.append((k, C[k])); rooms.append(dict(r, source=r.get('source', parts[0].rsplit('/', 1)[-1] + ':' + k)))
assert len(clips) <= 64, f"{len(clips)} clips: env.h MAX_STYLE and body.h ROOM_BANK are 64"
with open(out, 'wb') as f:
    f.write(b'MREF'); f.write(struct.pack('<i', len(clips)))
    for k, c in clips:
        f.write(k.encode().ljust(16, b'\0')[:16]); f.write(struct.pack('<if', c['frames'], c['fps']))
        f.write(np.concatenate([c['root'], np.asarray(c['q']).reshape(c['frames'], -1)], 1).astype(np.float32).tobytes())
json.dump({'version': 1, 'source': ' + '.join(srcs), 'clips': [k for k, _ in clips], 'rooms': rooms}, open(bank_out, 'w'))
print(f'{out}, {bank_out}: {len(clips)} clips')
