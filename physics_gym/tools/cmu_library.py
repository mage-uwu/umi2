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
    python3 tools/cmu_library.py --realign [--name cmu30]   (parts converted before cmu_to_ref.py aligned the rest
        directions: apply each subject's alignment to them in place, then pack)
    python3 tools/cmu_library.py --fps 30 --name cmu30 --subset walk,run,jump,...   (trials whose description has a
        keyword, at most --per-keyword each, at 30 Hz: for inverse dynamics, which needs finer time steps)
"""
import argparse, html, json, os, re, subprocess, sys, tempfile, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "data", "clean"); PARTS = os.path.join(OUT, "cmu_parts"); STEP = 12   # 120 Hz -> 10 Hz
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
        frames, fps, scale = c2r.convert(asf, amc, step=STEP)
        q = np.array([[qq for qq in f[1]] for f in frames], np.float32); r = np.array([f[0] for f in frames], np.float32)
        if not np.isfinite(q).all() or len(q) < 2: raise ValueError("bad frames")
        np.savez(part, q=q.astype(np.float16), root=r.astype(np.float16), aligned=1); return tr["id"], f"{len(q)} frames"
    except Exception as e:
        return tr["id"], f"failed: {e}"
    finally:
        os.path.exists(amc) and os.remove(amc)


def qmul(a, b):
    return np.stack([a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(-1),
                     *(a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] + np.cross(a[..., 1:], b[..., 1:])).transpose(-1, *range(a.ndim - 1))], -1)


def realign(trials, tmp):
    """parts converted without the rest-direction alignment: q -> q A per body (A from the subject's ASF), in place"""
    asfs = {}; done = 0
    for tr in trials:
        part = os.path.join(PARTS, tr["id"] + ".npz")
        if not os.path.exists(part): continue
        z = np.load(part)
        if "aligned" in z.files: continue
        s = tr["id"].split("_")[0]
        if s not in asfs:
            f = os.path.join(tmp, s + ".asf")
            if not get(f"{SITE}/subjects/{s}/{s}.asf", f): print("no asf", s); continue
            bones, _, _ = c2r.parse_asf(f); asfs[s] = np.array([c2r.mat2quat(A) for A in c2r.alignment(bones)])
        q = z["q"].astype(np.float64); q = qmul(q, asfs[s][None]); q /= np.linalg.norm(q, axis=-1, keepdims=True)
        np.savez(part, q=q.astype(np.float16), root=z["root"], aligned=1); done += 1
    print(f"realigned {done} parts ({len(asfs)} subjects)")


def main():
    global PARTS, STEP
    ap = argparse.ArgumentParser(); ap.add_argument("--index", action="store_true"); ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--fps", type=int, default=10); ap.add_argument("--name", default="cmu"); ap.add_argument("--subset", default="")
    ap.add_argument("--per-keyword", type=int, default=60); ap.add_argument("--realign", action="store_true")
    a = ap.parse_args()
    if a.index: return index()
    STEP = 120 // a.fps; PARTS = os.path.join(OUT, a.name + "_parts")
    trials = json.load(open(os.path.join(OUT, "cmu_index.json"))); os.makedirs(PARTS, exist_ok=True)
    if a.subset:   # up to per-keyword trials for each keyword, spread over subjects (every k-th match)
        pick = []
        for kw in a.subset.split(","):
            m = [t for t in trials if kw in t["desc"].lower() and t not in pick]; k = max(1, len(m) // a.per_keyword)
            pick += m[::k][:a.per_keyword]
        trials = pick
    if a.realign:
        trials = [t for t in trials if os.path.exists(os.path.join(PARTS, t["id"] + ".npz"))]
        with tempfile.TemporaryDirectory() as tmp: realign(trials, tmp)
    else:
        with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(a.jobs) as ex:
            for k, (tid, msg) in enumerate(ex.map(lambda t: one(t, tmp), trials)):
                print(f"{k + 1}/{len(trials)} {tid}: {msg}", flush=True)
    Q, R, T, man = [], [], [], []
    for k, tr in enumerate(trials):
        p = os.path.join(PARTS, tr["id"] + ".npz")
        if not os.path.exists(p): continue
        z = np.load(p); Q.append(z["q"]); R.append(z["root"]); T.append(np.full(len(z["q"]), len(man), np.int16))
        man.append(dict(tr, frames=int(len(z["q"]))))
    np.savez(os.path.join(OUT, a.name + "_library.npz"), q=np.concatenate(Q), root=np.concatenate(R), trial=np.concatenate(T))
    json.dump({"source": SITE, "fps": a.fps, "terms": "free for research; may be included in commercially-sold products; may not be resold, even converted",
               "trials": man}, open(os.path.join(OUT, a.name + "_library.json"), "w"), indent=0)
    print(f"packed {len(man)} trials, {sum(m['frames'] for m in man)} frames")


if __name__ == "__main__":
    main()
