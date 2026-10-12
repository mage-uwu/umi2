#!/usr/bin/env python3
"""Turn DeepSeek's umi-smart-internet data into tables for the widget model.

Each cached widget instance gets the ID (widget, instance, 0); widget 63 is the
reserved "plain" class (no widget), trained on ordinary chat messages from
data/train.tsv so umi knows when not to draw anything. Fragments are sanitised:
<script>, event handlers and external URLs are removed. Newlines are stored as
0x1C so a fragment fits on one TSV line.

  python3 prepare_web.py data/web data data_web [data/web_convs]

Widget conversations (gen_web_convs.py) become trees like the chat ones: turn t of conversation k
gets the ID (64 + k // 64, k % 64, t), its parent is the widget it refines, and each request
carries the IDs of the last two widgets shown as context.
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
    h = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", h)  # stray control bytes (newlines are kept, then encoded)
    h = SCRIPT.sub("", h); h = HANDLER.sub("", h); h = EXTERNAL.sub(r"\1\2#\2", h)
    h = CSSURL.sub("none", h); h = IMPORT.sub("", h)
    return NAME_TOKEN.sub(TAG, h).replace("\t", " ").replace("\r", "").replace("\n", "\x1c")

def main(webdir, chatdata, out, convdir=None, plain_n=4000, seed=0):
    rnd = random.Random(seed); os.makedirs(out, exist_ok=True)
    design = json.load(open(os.path.join(webdir, "design.json")))
    open(os.path.join(out, "design.css"), "w").write(IMPORT.sub("", CSSURL.sub("none", design["css"])))
    catalog = json.load(open(os.path.join(webdir, "catalog.json")))["widgets"][:63]
    answers, train, test = [], [], []  # train/test rows: (id, ctx1, ctx2, message)
    for i, w in enumerate(catalog):
        p = os.path.join(webdir, f"w{i:02d}_{w['name']}.json")
        if not os.path.exists(p): continue
        seen = set()
        insts = [x for x in json.load(open(p))["instances"] if not (x["title"] in seen or seen.add(x["title"]))][:4096]
        for j, x in enumerate(insts):
            h = sanitize(x["html"])
            if w["wraps_answer"] and "{{answer}}" not in h: continue
            cid = f"{i}.{j % 64}.{j // 64}"
            answers.append((cid, f"{w['name']}/{tsv(x['title'])[:60]}", h, h.replace(TAG, ""), "-"))
            reqs = [clean(r) for r in x["user_requests"] if r.strip()]
            train += [(cid, "-", "-", r) for r in reqs[:-1]]; test += [(cid, "-", "-", reqs[-1])] if reqs else []
    nconv = nturn = 0
    if convdir and os.path.exists(os.path.join(convdir, "convs.json")):
        for k, cv in enumerate(json.load(open(os.path.join(convdir, "convs.json")))["conversations"][:64 * 64]):
            a, b = 64 + k // 64, k % 64; tix = {t["id"]: i for i, t in enumerate(cv["turns"][:64])}
            par = {t["id"]: t["parent"] for t in cv["turns"]}; nconv += 1
            for t in cv["turns"][:64]:
                cid = f"{a}.{b}.{tix[t['id']]}"; h = sanitize(t["html"]); nturn += 1
                p = t["parent"]; c1 = f"{a}.{b}.{tix[p]}" if p in tix else "-"
                pp = par.get(p, ""); c2 = f"{a}.{b}.{tix[pp]}" if pp in tix else "-"
                answers.append((cid, f"conv/{cv.get('opener', '')}/{tsv(t['title'])[:60]}", h, h.replace(TAG, ""), c1))
                reqs = [clean(r) for r in t["user_messages"] if r.strip()]
                train += [(cid, c1, c2, r) for r in reqs[:-1]]; test += [(cid, c1, c2, reqs[-1])] if reqs else []
    # the plain class: ordinary chat, no widget
    chat = sorted({l.rstrip("\n").split("\t")[3] for l in open(os.path.join(chatdata, "train.tsv"))})
    rnd.shuffle(chat)
    answers.append(("63.0.0", "plain/none", "", "", "-"))
    train += [("63.0.0", "-", "-", m) for m in chat[:plain_n]]; test += [("63.0.0", "-", "-", m) for m in chat[plain_n:plain_n + plain_n // 10]]
    # a (context, message) pair that leads to two different widgets is ambiguous: drop it
    owner = {}
    for c, c1, c2, m in train + test: owner.setdefault((c1, m), set()).add(c)
    amb = {k for k, v in owner.items() if len(v) > 1}
    train = sorted({r for r in train if (r[1], r[3]) not in amb}); tr = {(r[1], r[3]) for r in train}
    test = sorted({r for r in test if (r[1], r[3]) not in amb and (r[1], r[3]) not in tr})
    with open(os.path.join(out, "answers.tsv"), "w") as fp:
        for r in answers: fp.write("\t".join(r) + "\n")
    for name, rows in (("train", train), ("test", test)):
        with open(os.path.join(out, name + ".tsv"), "w") as fp:
            for c, c1, c2, m in rows: fp.write(f"{c}\t{c1}\t{c2}\t{tsv(m)}\n")
    ctx = sum(r[1] != "-" for r in train)
    print(f"widgets {len(catalog)}  instances {len(answers) - 1} ({nconv} widget conversations, {nturn} turns)  "
          f"train {len(train)} ({ctx} with context)  test {len(test)}  ambiguous dropped {len(amb)}")

if __name__ == "__main__":
    a = sys.argv[1:] + [None] * 4
    main(a[0] or "data/web", a[1] or "data", a[2] or "data_web", a[3])
