#!/usr/bin/env python3
"""Turn DeepSeek's umi-smart-internet data into tables for the widget model.

Each cached widget instance gets the ID (widget, instance, 0); widget 63 is the
reserved "plain" class (no widget), trained on ordinary chat messages from
data/train.tsv so umi knows when not to draw anything. Fragments are sanitised:
<script>, event handlers and external URLs are removed. Newlines are stored as
0x1C so a fragment fits on one TSV line.

  python3 prepare_web.py data/web data data_web
writes data_web/{train,test,answers}.tsv (same format as prepare.py) and design.css
"""
import glob, json, os, random, re, sys
from prepare import clean, tsv, TAG, NAME_TOKEN

SCRIPT = re.compile(r"<\s*script\b[\s\S]*?<\s*/\s*script\s*>|<\s*script\b[^>]*>", re.I)
HANDLER = re.compile(r"\s+on[a-z]+\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s>]+)", re.I)
EXTERNAL = re.compile(r"((?:src|href|xlink:href|action)\s*=\s*)(\"|')\s*(?:https?:|//|javascript:)[^\"']*\2", re.I)
CSSURL = re.compile(r"url\(\s*['\"]?\s*(?:https?:|//)[^)]*\)", re.I)
IMPORT = re.compile(r"@import[^;]*;", re.I)

def sanitize(h):
    h = SCRIPT.sub("", h); h = HANDLER.sub("", h); h = EXTERNAL.sub(r"\1\2#\2", h)
    h = CSSURL.sub("none", h); h = IMPORT.sub("", h)
    return NAME_TOKEN.sub(TAG, h).replace("\t", " ").replace("\r", "").replace("\n", "\x1c")

def main(webdir, chatdata, out, plain_n=4000, seed=0):
    rnd = random.Random(seed); os.makedirs(out, exist_ok=True)
    design = json.load(open(os.path.join(webdir, "design.json")))
    open(os.path.join(out, "design.css"), "w").write(IMPORT.sub("", CSSURL.sub("none", design["css"])))
    catalog = json.load(open(os.path.join(webdir, "catalog.json")))["widgets"][:63]
    answers, train, test = [], [], []
    for i, w in enumerate(catalog):
        p = os.path.join(webdir, f"w{i:02d}_{w['name']}.json")
        if not os.path.exists(p): continue
        seen = set()
        insts = [x for x in json.load(open(p))["instances"] if not (x["title"] in seen or seen.add(x["title"]))][:64]
        for j, x in enumerate(insts):
            h = sanitize(x["html"])
            if w["wraps_answer"] and "{{answer}}" not in h: continue
            cid = f"{i}.{j}.0"
            answers.append((cid, f"{w['name']}/{tsv(x['title'])[:60]}", h, h.replace(TAG, ""), "-"))
            reqs = [clean(r) for r in x["user_requests"] if r.strip()]
            train += [(cid, r) for r in reqs[:-1]]; test += [(cid, reqs[-1])] if reqs else []
    # the plain class: ordinary chat, no widget
    chat = sorted({l.rstrip("\n").split("\t")[3] for l in open(os.path.join(chatdata, "train.tsv"))})
    rnd.shuffle(chat)
    answers.append(("63.0.0", "plain/none", "", "", "-"))
    train += [("63.0.0", m) for m in chat[:plain_n]]; test += [("63.0.0", m) for m in chat[plain_n:plain_n + plain_n // 10]]
    owner = {}
    for c, m in train + test: owner.setdefault(m, set()).add(c)
    amb = {m for m, s in owner.items() if len(s) > 1}
    train = sorted({r for r in train if r[1] not in amb}); tr = {m for _, m in train}
    test = sorted({r for r in test if r[1] not in amb and r[1] not in tr})
    with open(os.path.join(out, "answers.tsv"), "w") as fp:
        for r in answers: fp.write("\t".join(r) + "\n")
    for name, rows in (("train", train), ("test", test)):
        with open(os.path.join(out, name + ".tsv"), "w") as fp:
            for c, m in rows: fp.write(f"{c}\t-\t-\t{tsv(m)}\n")
    print(f"widgets {len(catalog)}  instances {len(answers) - 1}  train {len(train)}  test {len(test)}  ambiguous dropped {len(amb)}")

if __name__ == "__main__":
    a = sys.argv[1:] + [None] * 3
    main(a[0] or "data/web", a[1] or "data", a[2] or "data_web")
