#!/usr/bin/env python3
"""Turn DeepSeek's raw data into umi-1M-chat training tables.

Every cached answer has a three-part ID (a, b, t):
  single-turn intents      a = domain (0..63), b = intent, t = 0
  conversation turns       a = 64 + conv // 64, b = conv % 64, t = turn index
so a follow-up shares (a, b) with the answer it follows and the model only has
to pick the next turn. Each example carries the IDs of the last two answers
umi gave (or none). Messages are lower-cased and user names are canonicalised
to the tag <regex>user_name</regex> with the same regexes the C runtime uses.

  python3 prepare.py data/raw data/convs data
writes data/{train,test}.tsv  (a.b.t \\t ctx1 \\t ctx2 \\t message; ctx "-" = none)
and    data/answers.tsv       (a.b.t \\t name \\t answer \\t answer_no_name \\t parent a.b.t; one row per variant)
"""
import glob, json, os, re, sys, collections

NAME_STRONG = re.compile(r"\b(my name is|my name's|name's|call me|you can call me|i go by)\s+([a-z][a-z'\-]*)(?!\w|>)", re.I)
TAG = "<regex>user_name</regex>"
NAME_TOKEN = re.compile(r"<\s*regex\s*>[^<]*<\s*/\s*regex\s*>|\{\s*(name|user|username|user_name)\s*\}", re.I)
NONE = "-"
# DeepSeek sometimes emits a punctuation escape with its high byte dropped (\u0014 for \u2014 and
# similar), leaving a control character plus stray digits in the text. Such variants are dropped
# when a clean one exists; otherwise the damaged sequence becomes a dash.
CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f][0-9]{0,2}")
def fix(s):
    if not CTRL.search(s): return s
    s = CTRL.sub(" \u2014 ", s)
    s = re.sub(r"(\s*\u2014\s*)+", " \u2014 ", s)
    return re.sub(r"[ ]{2,}", " ", s).strip()

def clean(msg):
    m = NAME_TOKEN.sub(TAG, fix(msg.strip()))
    m = NAME_STRONG.sub(lambda g: g.group(1) + " " + TAG, m)
    return re.sub(r"\s+", " ", m).lower()

def tsv(s): return fix(s).replace("\t", " ").replace("\n", " ")  # every text field: control bytes repaired
def ids(a, b, t): return f"{a}.{b}.{t}"

def main(raw, convdir, out):
    answers, rows = [], []  # rows: (split, id, ctx1, ctx2, msg)
    for f in sorted(glob.glob(os.path.join(raw, "[0-9][0-9]_*.json"))):
        di = int(os.path.basename(f)[:2]); d = json.load(open(f)); seen = set()
        for ii, it in enumerate([x for x in d["intents"] if not (x["intent"] in seen or seen.add(x["intent"]))][:64]):
            cid = ids(di, ii, 0)
            pairs = list(zip(it["answers"], it["answers_no_name"]))
            good = [p for p in pairs if not CTRL.search(p[0] + p[1])]
            for ans, ano in (good or [(fix(a), fix(b)) for a, b in pairs]):
                ans = NAME_TOKEN.sub(TAG, ans.strip())
                answers.append((cid, f"{d['domain']}/{it['intent']}", ans, NAME_TOKEN.sub("", ano.strip()) or ans.replace(TAG, "").strip(), NONE))
            rows += [("train", cid, NONE, NONE, clean(m)) for m in it["user_messages"]]
            rows += [("test", cid, NONE, NONE, clean(m)) for m in it["test_messages"]]
    k = 0; nturns = 0
    for f in sorted(glob.glob(os.path.join(convdir, "*.json"))) if convdir else []:
        d = json.load(open(f))
        for cv in d["conversations"]:
            if k >= 64 * 64: break
            a, b = 64 + k // 64, k % 64; k += 1
            tix = {t["id"]: i for i, t in enumerate(cv["turns"][:64])}; par = {t["id"]: t["parent"] for t in cv["turns"]}
            for t in cv["turns"][:64]:
                cid = ids(a, b, tix[t["id"]]); nturns += 1
                ans = NAME_TOKEN.sub(TAG, fix(t["assistant"].strip()))
                ano = NAME_TOKEN.sub("", fix(t["assistant_no_name"].strip())) or ans.replace(TAG, "").strip()
                p = t["parent"]; c1 = ids(a, b, tix[p]) if p in tix else NONE
                answers.append((cid, f"conv/{d['domain']}/{tsv(cv['topic'])[:60]}/{t['id']}", ans, ano, c1))
                pp = par.get(p, ""); c2 = ids(a, b, tix[pp]) if pp in tix else NONE
                msgs = [clean(m) for m in t["user_messages"]]
                for j, m in enumerate(msgs):  # last phrasing held out
                    rows.append(("test" if j == len(msgs) - 1 else "train", cid, c1, c2, m))
    # a (context, message) pair that leads to two different answers is ambiguous: drop it
    owner = collections.defaultdict(set)
    for s, cid, c1, c2, m in rows:
        if m: owner[(c1, m)].add(cid)
    amb = {key for key, v in owner.items() if len(v) > 1}
    keep = lambda r: r[4] and (r[2], r[4]) not in amb
    train = sorted({r[1:] for r in rows if r[0] == "train" and keep(r)})
    trainkeys = {(r[1], r[3]) for r in train}
    test = sorted({r[1:] for r in rows if r[0] == "test" and keep(r) and (r[1], r[3]) not in trainkeys})
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "answers.tsv"), "w") as fp:
        for r in answers: fp.write("\t".join(tsv(x) for x in r) + "\n")
    for name, rs in (("train", train), ("test", test)):
        with open(os.path.join(out, name + ".tsv"), "w") as fp:
            for cid, c1, c2, m in rs: fp.write(f"{cid}\t{c1}\t{c2}\t{tsv(m)}\n")
    L = sorted(len(r[3]) for r in train)
    ctx = sum(r[1] != NONE for r in train)
    print(f"answers {len({a[0] for a in answers})} ({len(answers)} variants; {k} conversations, {nturns} turns)  "
          f"train {len(train)} ({ctx} with context)  test {len(test)}  ambiguous dropped {len(amb)}  "
          f"msg len p50 {L[len(L)//2]} p99 {L[int(len(L)*.99)]} max {L[-1]}")

if __name__ == "__main__":
    a = sys.argv[1:] + [None] * 3
    main(a[0] or "data/raw", a[1] or "data/convs", a[2] or "data")
