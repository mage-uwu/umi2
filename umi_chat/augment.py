#!/usr/bin/env python3
"""Second DeepSeek pass: more user phrasings for every existing intent.

Again only a schema goes out, plus DeepSeek's own intent names and its own
first phrasing of each (so it knows what the intent means). New phrasings
are appended to user_messages in data/raw/*.json in place.

  python3 augment.py --raw data/raw --budget 0.55
"""
import argparse, glob, json, os, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from gen_data import Budget, call, prompt_for, TAG

def schema(names, n):
    return {"type": "object", "additionalProperties": False, "required": ["intents"],
            "properties": {"intents": {"type": "array", "minItems": len(names), "maxItems": len(names),
                "description": "One entry per intent in Context, same order.",
                "items": {"type": "object", "additionalProperties": False, "required": ["intent", "more_user_messages"],
                          "properties": {"intent": {"type": "string", "enum": names},
                                         "more_user_messages": {"type": "array", "minItems": n, "maxItems": n, "items": {"type": "string"},
                                             "description": "New, varied user phrasings for this intent, unlike the example: other wordings, "
                                                            "lengths, registers, typos, slang, questions vs statements. No personal names; "
                                                            "use the literal tag " + TAG + " for the user's own name."}}}}}}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw"); ap.add_argument("--budget", type=float, default=0.55)
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--chunk", type=int, default=14)
    ap.add_argument("--workers", type=int, default=10); ap.add_argument("--flag", default="augmented")
    a = ap.parse_args(); budget = Budget(a.budget); t0 = time.time(); locks = {}
    jobs = []
    for f in sorted(glob.glob(os.path.join(a.raw, "[0-9][0-9]_*.json"))):
        d = json.load(open(f)); locks[f] = threading.Lock()
        for s in range(0, len(d["intents"]), a.chunk):
            if not all(it.get(a.flag) for it in d["intents"][s:s + a.chunk]): jobs.append((f, s))
    def run(f, s):
        with locks[f]: d = json.load(open(f))
        its = d["intents"][s:s + a.chunk]; names = [it["intent"] for it in its]
        ctx = {"domain": d["domain"], "intents": [{"intent": it["intent"], "example": it["user_messages"][0]} for it in its]}
        sc = schema(names, a.n); out, c, tot = call(sc, prompt_for(sc, ctx), budget)
        got = {o["intent"]: o["more_user_messages"] for o in out["intents"]}
        with locks[f]:
            d = json.load(open(f))
            for it in d["intents"][s:s + a.chunk]:
                if it["intent"] in got and not it.get(a.flag):
                    it["user_messages"] += got[it["intent"]]; it[a.flag] = True
            json.dump(d, open(f, "w"), indent=1)
        print(f"{os.path.basename(f)}[{s}] +{sum(map(len, got.values()))} msgs  ${c:.4f}  total ${tot:.4f}  {time.time()-t0:.0f}s", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for fu in as_completed([ex.submit(run, f, s) for f, s in jobs]):
            try: fu.result()
            except Exception as e: print("job failed:", e, flush=True)
    print(f"done: ${budget.spent:.4f} spent, {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
