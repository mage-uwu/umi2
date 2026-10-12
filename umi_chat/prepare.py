#!/usr/bin/env python3
"""Turn DeepSeek's raw intent files into umi-1M-chat training tables.

Each cached answer gets a two-byte ID: byte 1 picks the domain, byte 2 the
intent inside it (the model decides "which chunk do I retrieve" in that
order). Messages are lower-cased and user names are canonicalised to the
tag <regex>user_name</regex> with the same regexes the C runtime uses.

  python3 prepare.py data/raw data
writes data/{train,test}.tsv  (ID \t message) and data/answers.tsv
       (ID \t domain/intent \t answer \t answer_no_name; one row per variant)
"""
import glob, json, os, re, sys, collections

# 64 printable ID symbols; index i <-> ALPHA[i]. Must match umi_chat.c.
ALPHA = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+/"

# Strong name patterns: the captured word is always a name.
NAME_STRONG = re.compile(r"\b(my name is|my name's|name's|call me|you can call me|i go by)\s+([a-z][a-z'\-]*)(?!\w|>)", re.I)
TAG = "<regex>user_name</regex>"
NAME_TOKEN = re.compile(r"<\s*regex\s*>[^<]*<\s*/\s*regex\s*>|\{\s*(name|user|username|user_name)\s*\}", re.I)

def clean(msg):
    m = NAME_TOKEN.sub(TAG, msg.strip())
    m = NAME_STRONG.sub(lambda g: g.group(1) + " " + TAG, m)
    m = re.sub(r"\s+", " ", m).lower()
    return m

def main(raw, out):
    files = sorted(glob.glob(os.path.join(raw, "[0-9][0-9]_*.json")))
    answers, train, test = [], [], []
    owner = collections.defaultdict(set)  # message -> ids (to drop ambiguous phrasings)
    for f in files:
        di = int(os.path.basename(f)[:2])
        d = json.load(open(f))
        seen = set()
        for ii, it in enumerate([x for x in d["intents"] if not (x["intent"] in seen or seen.add(x["intent"]))][:64]):
            cid = ALPHA[di] + ALPHA[ii]
            for ans, ano in zip(it["answers"], it["answers_no_name"]):
                ans = NAME_TOKEN.sub(TAG, ans.strip())
                ano = NAME_TOKEN.sub("", ano.strip()) or ans.replace(TAG, "").strip()
                answers.append((cid, f"{d['domain']}/{it['intent']}", ans, ano))
            for m in it["user_messages"]:
                m = clean(m)
                if m: train.append((cid, m)); owner[m].add(cid)
            for m in it["test_messages"]:
                m = clean(m)
                if m: test.append((cid, m)); owner[m].add(cid)
    amb = {m for m, s in owner.items() if len(s) > 1}
    train = sorted({(c, m) for c, m in train if m not in amb})
    trainset = {m for _, m in train}
    test = sorted({(c, m) for c, m in test if m not in amb and m not in trainset})
    os.makedirs(out, exist_ok=True)
    tsv = lambda s: s.replace("\t", " ").replace("\n", " ")
    with open(os.path.join(out, "answers.tsv"), "w") as fp:
        for r in answers: fp.write("\t".join(tsv(x) for x in r) + "\n")
    for name, rows in (("train", train), ("test", test)):
        with open(os.path.join(out, name + ".tsv"), "w") as fp:
            for c, m in rows: fp.write(f"{c}\t{tsv(m)}\n")
    L = sorted(len(m) for _, m in train)
    print(f"domains {len(files)}  intents {len({a[0] for a in answers})}  answer variants {len(answers)}  train {len(train)}  test {len(test)}  "
          f"ambiguous dropped {len(amb)}  msg len p50 {L[len(L)//2]} p99 {L[int(len(L)*.99)]} max {L[-1]}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw", sys.argv[2] if len(sys.argv) > 2 else "data")
