#!/usr/bin/env python3
"""expert_eval.py: the Rooftops move experts judged (data/experts/rooftops.json): each one's checkpoint DIR/<name>
(.mlp.pt/.head, or .tss) on its 16 held-out variants and in Rooftops itself, on the obstacles its move is for
(tools/clip_in_world.py --align), N episodes each from the clip's start with start noise (EVAL_NOISE, 0.02 rad unless set),
unaided, x2 strength. Done: the clip's end reached (tools/eval_course.py); reached: her pelvis 0.6 m over the obstacle,
above it (up on it, over it, landed on it), at some point. A table to stdout and DIR/report.md, the numbers to
DIR/report.json; with --replays WEBDIR, each Rooftops placement's first episode as WEBDIR/expert_<name>_<k>.json.

    python3 tools/expert_eval.py DIR [NAME ...] [--n 64] [--map NAME=PREFIX ...] [--replays web/replays]
"""
import argparse, json, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

B = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(B, 'tools'))
import expert_bank as eb

LINE = re.compile(r"done (\d+) \((\d+)%\), ended down (\d+), up on its box (\d+)")


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, EVAL_NOISE=os.environ.get("EVAL_NOISE", "0.02")))
    m = LINE.search(r.stdout)
    return {"done": int(m[1]), "down": int(m[3]), "reached": int(m[4])} if m else {"error": (r.stderr or r.stdout)[-400:]}


def main():
    ap = argparse.ArgumentParser(); A = ap.add_argument
    A("dir"); A("names", nargs="*"); A("--n", type=int, default=64); A("--map", nargs="*", default=[]); A("--replays")
    A("--manifest", default=os.path.join(B, "data/experts/rooftops.json")); A("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    pre = dict(m.split("=", 1) for m in a.map)
    jobs = []
    for e in eb.experts(a.manifest):
        if a.names and e["name"] not in a.names: continue
        p = pre.get(e["name"], os.path.join(a.dir, e["name"]))
        if not (os.path.exists(p + ".mlp.pt") or os.path.exists(p + ".tss")): print(f"{e['name']}: no checkpoint at {p}"); continue
        held = os.path.join(os.path.dirname(a.manifest), f"{e['name']}_heldout.json")
        jobs.append((e["name"], "held-out variants", [sys.executable, os.path.join(B, "tools/eval_course.py"), p, held, os.path.join(B, e["clips"]), str(a.n), "0", "2"]))
        for k, (box, heading) in enumerate(e.get("rooftops", [])):
            cmd = [sys.executable, os.path.join(B, "tools/clip_in_world.py"), p, e["clip"], os.path.join(B, "data/ac_course_bank.json"), "0", "--align", box, str(heading), "--t0", "auto",
                   "--n", str(a.n // 2), "--clips", e["clips"], "--bank", e["bank"]]
            if a.replays: cmd += ["--replay", os.path.join(a.replays, f"expert_{e['name']}_{k}.json")]
            jobs.append((e["name"], f"Rooftops: {box}", cmd))
    with ThreadPoolExecutor(max(1, a.jobs)) as ex: res = list(ex.map(lambda j: run(j[2]), jobs))
    rows, out = [], {}
    for (nm, where, cmd), r in zip(jobs, res):
        n = int(cmd[cmd.index("--n") + 1]) if "--n" in cmd else a.n
        out.setdefault(nm, {})[where] = dict(r, episodes=n)
        rows.append(f"| {nm} | {where} | " + (f"{r['reached']}/{n} | {r['done']}/{n} |" if "error" not in r else f"error | {r['error'][-80:]!r} |"))
    table = "| expert | where | reached the obstacle | whole clip done |\n|---|---|---|---|\n" + "\n".join(rows)
    note = f"\n\nStart noise {os.environ.get('EVAL_NOISE', '0.02')} rad, unaided, x2 strength (tools/expert_eval.py).\n"
    if a.replays:   # the viewer's replay buttons (web/index.html reads replays/index.json)
        f = os.path.join(a.replays, "index.json"); idx = json.load(open(f)) if os.path.exists(f) else []
        for nm, where, cmd in jobs:
            if "--replay" in cmd and os.path.exists(cmd[cmd.index("--replay") + 1]):
                file = "replays/" + os.path.basename(cmd[cmd.index("--replay") + 1]); idx = [e for e in idx if e["file"] != file] + [{"label": f"{nm}: {where[len('Rooftops: '):]}", "file": file}]
        json.dump(idx, open(f, "w"), indent=1)
    print(table + note); os.makedirs(a.dir, exist_ok=True)
    open(os.path.join(a.dir, "report.md"), "w").write(table + note); json.dump(out, open(os.path.join(a.dir, "report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
