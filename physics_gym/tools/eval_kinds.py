#!/usr/bin/env python3
"""eval_kinds.py: tools/eval_course.py's per-clip lines grouped by the kind of move each clip is (from its source in the
bank: PARC's runs over broken ground, platforms, vaults, climbing terrain; the CMU climbs), so a checkpoint's skills read
at a glance rather than as one number over all of them.

    python3 tools/eval_kinds.py BANK.json EVAL.txt [EVAL.txt ...]      (one column per evaluation file)
"""
import json, re, sys

KINDS = [("run", r"running"), ("vault", r"vault"), ("platform", r"platform"), ("climb terrain", r"climbing_up_down"), ("climb (CMU)", r"cmu_climb|^c\d")]


def kind(src):
    for k, pat in KINDS:
        if re.search(pat, src): return k
    return "other"


def main():
    bank, files = json.load(open(sys.argv[1])), sys.argv[2:]
    of = {r["clip"]: kind(r.get("source", r["clip"])) for r in bank["rooms"]}
    cols = []
    for f in files:
        t = {}
        for line in open(f):
            m = re.match(r"\s+(\S+): (\d+) of (\d+) done", line)
            if m and m.group(1) in of:
                k = of[m.group(1)]; d, n = t.get(k, (0, 0)); t[k] = (d + int(m.group(2)), n + int(m.group(3)))
        cols.append(t)
    kinds = [k for k, _ in KINDS] + ["other"]
    print(f"{'kind':15s} {'clips':>5s}  " + "  ".join(f"{f.split('/')[-1][:18]:>18s}" for f in files))
    for k in kinds:
        nclips = sum(1 for v in of.values() if v == k)
        if not nclips: continue
        cells = [f"{c.get(k, (0, 0))[0]:>4d} of {c.get(k, (0, 0))[1]:<4d} {100 * c.get(k, (0, 0))[0] / max(1, c.get(k, (0, 0))[1]):3.0f}%" for c in cols]
        print(f"{k:15s} {nclips:>5d}  " + "  ".join(f"{s:>18s}" for s in cells))


if __name__ == "__main__":
    main()
