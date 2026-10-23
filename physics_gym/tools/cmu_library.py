#!/usr/bin/env python3
"""cmu_library.py: the whole CMU Graphics Lab Motion Capture Database as one pose library for the physics character.

Downloads every trial (ASF skeleton + AMC motion, mocap.cs.cmu.edu) a few at a time, converts it with
cmu_to_ref.convert (the character's 14 body rotations, root scaled to her hip height) at 10 Hz, and deletes the
download. Writes, under data/clean/ (not in git: about 100 MB):

  cmu_library.npz    q [F, 14, 4] float16 world rotations, root [F, 3] float16, trial [F] int16 (index into the json)
  cmu_library.json   per trial: id, subject, group, description, frames (a manifest; also used to pick skill clips)

Restartable: trials already converted are skipped (data/clean/cmu_parts/). The trial list comes from the site's
motion index (cmu_index.json, written by --index).

Terms, as stated on mocap.cs.cmu.edu: "This data is free for use in research projects. You may include this data
in commercially-sold products, but you may not resell this data directly, even in converted form."

    python3 tools/cmu_library.py --index        (fetch the motion index)
    python3 tools/cmu_library.py [--jobs 6]     (fetch and convert every trial, then pack)
"""
import argparse, html, json, os, re, subprocess, sys, tempfile, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "data", "clean"); PARTS = os.path.join(OUT, "cmu_parts")
sys.path.insert(0, HERE)
import cmu_to_ref as c2r

SITE = "http://mocap.cs.cmu.edu"


def get(url, path, tries=4):
    for k in range(tries):
        try:
            subprocess.run(["curl", "-sSf", "--max-time", "300", "-o", path, url], check=True); return True
        except subprocess.CalledProcessError:
            if k == tries - 1: return False


def index():
    t = urllib.request.urlopen(SITE + "/search.php?subjectnumber=%25&motion=%25", timeout=300).read().decode("latin-1")
    out = []; subj = None
    for r in re.findall(r"<TR[^>]*>(.*?)</TR>", t, re.S | re.I):
        m = re.search(r"Subject #(\d+)\s*\(([^)]*)\)", html.unescape(re.sub("<[^>]+>", "", r)))
        if m: subj = (int(m.group(1)), m.group(2).strip()); continue
        c = [html.unescape(re.sub("<[^>]+>", "", x)).strip() for x in re.findall(r"<TD[^>]*>(.*?)</TD>", r, re.S | re.I)]
        amc = re.search(r"/subjects/(\d+)/(\d+_\d+)\.amc", r)
        if subj and amc and len(c) > 2: out.append(dict(id=amc.group(2), subject=subj[0], group=subj[1], desc=c[2]))
    os.makedirs(OUT, exist_ok=True); json.dump(out, open(os.path.join(OUT, "cmu_index.json"), "w"), indent=0)
    print(len(out), "trials")


def one(tr, tmp):
    part = os.path.join(PARTS, tr["id"] + ".npz")
    if os.path.exists(part): return tr["id"], "cached"
    s = tr["id"].split("_")[0]; asf = os.path.join(tmp, s + ".asf"); amc = os.path.join(tmp, tr["id"] + ".amc")
    if not os.path.exists(asf) and not get(f"{SITE}/subjects/{s}/{s}.asf", asf): return tr["id"], "no asf"
    if not get(f"{SITE}/subjects/{s}/{tr['id']}.amc", amc): return tr["id"], "no amc"
    try:
        frames, fps, scale = c2r.convert(asf, amc, step=12)   # 120 Hz -> 10 Hz
        q = np.array([[qq for qq in f[1]] for f in frames], np.float32); r = np.array([f[0] for f in frames], np.float32)
        if not np.isfinite(q).all() or len(q) < 2: raise ValueError("bad frames")
        np.savez(part, q=q.astype(np.float16), root=r.astype(np.float16)); return tr["id"], f"{len(q)} frames"
    except Exception as e:
        return tr["id"], f"failed: {e}"
    finally:
        os.path.exists(amc) and os.remove(amc)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--index", action="store_true"); ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    if a.index: return index()
    trials = json.load(open(os.path.join(OUT, "cmu_index.json"))); os.makedirs(PARTS, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(a.jobs) as ex:
        for k, (tid, msg) in enumerate(ex.map(lambda t: one(t, tmp), trials)):
            print(f"{k + 1}/{len(trials)} {tid}: {msg}", flush=True)
    Q, R, T, man = [], [], [], []
    for k, tr in enumerate(trials):
        p = os.path.join(PARTS, tr["id"] + ".npz")
        if not os.path.exists(p): continue
        z = np.load(p); Q.append(z["q"]); R.append(z["root"]); T.append(np.full(len(z["q"]), len(man), np.int16))
        man.append(dict(tr, frames=int(len(z["q"]))))
    np.savez(os.path.join(OUT, "cmu_library.npz"), q=np.concatenate(Q), root=np.concatenate(R), trial=np.concatenate(T))
    json.dump({"source": SITE, "fps": 10, "terms": "free for research; may be included in commercially-sold products; may not be resold, even converted",
               "trials": man}, open(os.path.join(OUT, "cmu_library.json"), "w"), indent=0)
    print(f"packed {len(man)} trials, {sum(m['frames'] for m in man)} frames")


if __name__ == "__main__":
    main()
