#!/usr/bin/env python3
"""Fact-check pass over the general-assistant trees. DeepSeek (a different model from the one that wrote the
replies) judges each umi reply, with the user message it answers, and lists the ones that state something
false. Flagged turns go to data_general/flagged.txt (uid:turn); prepare.py leaves them and their subtrees out
of the cache, so no retraining is needed: the model simply never sees them as candidates.

  python3 general/factcheck.py --dry-run          # token count and cost estimate, no API calls
By default only replies that state something specific (numbers, quoted titles, proper nouns) are judged.
  python3 general/factcheck.py --budget 2.5       # run (restartable; judged turns are skipped)
"""
import argparse, glob, gzip, json, os, re, threading, time
from concurrent.futures import ThreadPoolExecutor
from orcall import call, prompt_for, Budget, DEEPSEEK

PRICE = {"in": 0.15, "out": 0.60}  # DeepSeek V4.1 Flash, $ per million tokens
SCHEMA_ITEM = {"type": "object", "additionalProperties": False, "required": ["i", "verdict"],
               "properties": {"i": {"type": "integer"},
                              "verdict": {"type": "string", "enum": ["ok", "false", "doubtful"],
                                          "description": "false: states something factually wrong (a wrong date, name, number, "
                                                         "title, step or claim) or invents something that does not exist; "
                                                         "doubtful: a specific claim you cannot confirm; ok: otherwise, including "
                                                         "opinions, jokes and general advice"}}}
def schema(n):
    return {"type": "object", "additionalProperties": False, "required": ["items"],
            "properties": {"items": {"type": "array", "minItems": n, "maxItems": n, "items": SCHEMA_ITEM,
                                     "description": "One verdict per reply in Context, in order, judging only umi's reply."}}}

# a reply worth checking states something specific: a number, a quoted title, or a proper noun mid-sentence
SPECIFIC = re.compile(r"\d|[\"“”«]|(?<=[a-z,;:] )[A-Z][a-z]+")
def turns(tdir):
    for f in sorted(glob.glob(os.path.join(tdir, "trees_*.jsonl.gz"))):
        for line in gzip.open(f, "rt"):
            t = json.loads(line)
            for x in t["turns"]: yield f"{t['uid']}:{x['id']}", x["user_messages"][0], x["assistant"]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--trees", default="data_general/trees")
    ap.add_argument("--out", default="data_general"); ap.add_argument("--budget", type=float, default=0)
    ap.add_argument("--per", type=int, default=40); ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--flag-doubtful", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--all", action="store_true", help="judge every reply, not only those with specific claims")
    a = ap.parse_args()
    judged_p = os.path.join(a.out, "factcheck.jsonl")
    judged = {json.loads(l)["key"] for l in open(judged_p)} if os.path.exists(judged_p) else set()
    todo = [x for x in turns(a.trees) if x[0] not in judged and (a.all or SPECIFIC.search(x[2]))]
    batches = [todo[i:i + a.per] for i in range(0, len(todo), a.per)]
    chars = sum(len(u) + len(r) + 40 for _, u, r in todo)
    est_in = chars / 3.6 + len(batches) * 700  # ~3.6 chars per token, plus schema per call
    est_out = len(todo) * 14
    cost = est_in / 1e6 * PRICE["in"] + est_out / 1e6 * PRICE["out"]
    print(f"{len(todo)} replies in {len(batches)} calls: ~{est_in/1e6:.1f}M input + {est_out/1e6:.1f}M output tokens, ~${cost:.2f}")
    if a.dry_run: return
    if a.budget <= 0: raise SystemExit("set --budget")
    b = Budget(a.budget); lock = threading.Lock(); fp = open(judged_p, "a"); t0 = time.time(); n = [0, 0]
    def one(batch):
        if not b.ok(reserve=0.01): return
        ctx = {"replies": [{"i": i, "user": u, "umi": r} for i, (_, u, r) in enumerate(batch)]}
        try: out = call(DEEPSEEK, schema(len(batch)), prompt_for(schema(len(batch)), ctx), b, max_tokens=4000, temperature=0)[0]
        except RuntimeError: return
        with lock:
            for it in out["items"]:
                if 0 <= it["i"] < len(batch):
                    fp.write(json.dumps({"key": batch[it["i"]][0], "verdict": it["verdict"]}) + "\n"); n[0] += 1
                    n[1] += it["verdict"] != "ok"
            fp.flush()
    with ThreadPoolExecutor(a.workers) as ex: list(ex.map(one, batches))
    bad = {"false"} | ({"doubtful"} if a.flag_doubtful else set())
    flagged = [json.loads(l)["key"] for l in open(judged_p) if json.loads(l)["verdict"] in bad]
    open(os.path.join(a.out, "flagged.txt"), "w").write("\n".join(flagged) + "\n")
    print(f"judged {n[0]} this run ({n[1]} not ok); flagged {len(flagged)} in total; ${b.spent:.3f}; {time.time()-t0:.0f}s")

if __name__ == "__main__": main()
