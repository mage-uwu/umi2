#!/usr/bin/env python3
"""Coverage audit for the general-assistant corpus: how the trees spread over categories, topic sources,
request kinds and typing styles, how many openers repeat, and tree shape.

  python3 general/stats.py data_general
"""
import collections, glob, json, os, re, statistics as st, sys

def main(root):
    P = [json.loads(l) for f in sorted(glob.glob(os.path.join(root, "prompts", "*.jsonl"))) for l in open(f)]
    T = [json.loads(l) for f in sorted(glob.glob(os.path.join(root, "trees", "trees_*.jsonl"))) for l in open(f)]
    print(f"prompts {len(P)}   trees {len(T)}   turns {sum(len(t['turns']) for t in T)}   "
          f"user phrasings {sum(len(x['user_messages']) for t in T for x in t['turns'])}")
    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
    ops = collections.Counter(norm(p["opener"]) for p in P)
    print(f"distinct openers {len(ops)} ({len(ops)/max(1,len(P)):.1%});  repeated more than once: {sum(1 for c in ops.values() if c > 1)}")
    cat = lambda s: s["topic"].split(" > ")[0] if s["source"] != "hat" else "hat:" + s["topic"].split(":")[0]
    for name, key in (("source", lambda s: s["source"]), ("request", lambda s: s["wants_to"]), ("style", lambda s: s["typing_style"])):
        c = collections.Counter(key(t["seed"]) for t in T)
        print(f"\n{name}: " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
    c = collections.Counter(cat(t["seed"]) for t in T if t["seed"]["source"] == "backbone")
    v = sorted(c.values()); print(f"\nbackbone categories {len(c)}: trees per category min {v[0]} median {v[len(v)//2]} max {v[-1]}")
    n = [len(t["turns"]) for t in T]
    br = [sum(1 for k in collections.Counter(x["parent"] for x in t["turns"]).values() if k > 1) for t in T]
    L = [len(x["assistant"]) for t in T for x in t["turns"]]
    print(f"turns per tree mean {st.mean(n):.1f}; branch points mean {st.mean(br):.2f}; reply chars p10 {sorted(L)[len(L)//10]} "
          f"p50 {st.median(L):.0f} p90 {sorted(L)[9*len(L)//10]}")
    tag = sum("<regex>user_name</regex>" in x["assistant"] for t in T for x in t["turns"])
    print(f"replies using the name tag: {tag}")

if __name__ == "__main__": main(sys.argv[1] if len(sys.argv) > 1 else "data_general")
