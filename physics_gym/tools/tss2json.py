#!/usr/bin/env python3
"""tss2json.py prefix out.json: a nanotss body policy (prefix.tss + prefix.head, from ppo_tss.c) as JSON for the viewer."""
import base64, json, struct, sys
pre, out = sys.argv[1], sys.argv[2]
tss, head = open(pre + '.tss', 'rb').read(), open(pre + '.head', 'rb').read()
d, obs, act = struct.unpack_from('<3i', head, 4)
ends = d * obs + d + act * d + 2 * act
head = head[:16 + 4 * ends + 8 * obs]   # the viewer needs no critic
json.dump({'type': 'nanotss', 'params': (len(tss) - 76) // 4 - 256 * d + ends,   # header: magic, real size, Cfg (52 bytes), extension; minus the unused byte embedding 'width': d,
           'tss': base64.b64encode(tss).decode(), 'head': base64.b64encode(head).decode()}, open(out, 'w'))
