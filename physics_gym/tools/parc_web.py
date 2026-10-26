#!/usr/bin/env python3
"""parc_web.py: the PARC clips and rooms for the viewer's PARC skill (web/parc_clips.json, web/parc_bank.json).
    python3 tools/parc_web.py
data/parc_train* and data/parc_heldout* (tools/parc_to_ref.py) in one file each: the training clips as they are (p0..),
the held-out ones renamed h0.. so the names stay apart; each room keeps its source file and says which set it is from."""
import base64, json, os, struct
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
out, rooms = [], []
for s, pre in (('train', 'p'), ('heldout', 'h')):
    raw = open(f'{B}/data/parc_{s}.bin', 'rb').read(); n = struct.unpack('<i', raw[4:8])[0]; o = 8; ren = {}
    for k in range(n):
        name = raw[o:o + 16].rstrip(b'\0').decode(); frames = struct.unpack('<i', raw[o + 16:o + 20])[0]; size = 24 + frames * (3 + 14 * 4) * 4
        new = pre + name[1:]; ren[name] = new; out.append(new.encode().ljust(16, b'\0') + raw[o + 16:o + size]); o += size
    for r in json.load(open(f'{B}/data/parc_{s}_bank.json'))['rooms']: rooms.append(dict(r, clip=ren[r['clip']], name='parc_' + ren[r['clip']], set=s))
mref = b'MREF' + struct.pack('<i', len(out)) + b''.join(out)
note = ('base64 of data/parc_train.bin and data/parc_heldout.bin (held-out clips renamed h0..): parkour clips from PARC (Xu, Shi, '
        'Yin, Peng, SIGGRAPH 2025; huggingface.co/datasets/mxucg/PARC, MIT), retargeted by tools/parc_to_ref.py; the reference '
        'poses the PARC tracker follows.')
json.dump({'format': 'MREF', 'note': note, 'b64': base64.b64encode(mref).decode()}, open(f'{B}/web/parc_clips.json', 'w'))
json.dump({'version': 1, 'source': 'PARC (huggingface.co/datasets/mxucg/PARC, MIT)', 'clips': [r['clip'] for r in rooms], 'rooms': rooms},
          open(f'{B}/web/parc_bank.json', 'w'), separators=(',', ':'))
print(f'web/parc_clips.json, web/parc_bank.json: {len(out)} clips')
