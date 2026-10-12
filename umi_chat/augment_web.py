#!/usr/bin/env python3
"""More user requests for umi-smart-internet widget instances (schema + DeepSeek's own titles only).

  python3 augment_web.py --web data/web --budget 0.28
"""
import argparse, glob, json, os, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from gen_data import Budget, call, prompt_for

def schema(titles, n):
    return {"type": "object", "additionalProperties": False, "required": ["instances"],
            "properties": {"instances": {"type": "array", "minItems": len(titles), "maxItems": len(titles),
                "description": "One entry per instance in Context, same order.",
                "items": {"type": "object", "additionalProperties": False, "required": ["title", "more_user_requests"],
                          "properties": {"title": {"type": "string", "enum": titles},
                                         "more_user_requests": {"type": "array", "minItems": n, "maxItems": n, "items": {"type": "string"},
                                             "description": "New, varied things a user might type to get this widget instance: other wordings, "
                                                            "lengths, typos, slang; unlike the example."}}}}}}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--web", default="data/web"); ap.add_argument("--budget", type=float, default=0.28)
    ap.add_argument("--n", type=int, default=6); ap.add_argument("--chunk", type=int, default=16); ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(); budget = Budget(a.budget); t0 = time.time(); locks = {}; jobs = []
    files = sorted(glob.glob(os.path.join(a.web, "w[0-9][0-9]_*.json")))
    for f in files:
        d = json.load(open(f)); locks[f] = threading.Lock()
        for s in range(0, len(d["instances"]), a.chunk):
            if not all(x.get("augmented") for x in d["instances"][s:s + a.chunk]): jobs.append((f, s))
    jobs.sort(key=lambda j: j[1])  # spread the budget across widgets first
    def run(f, s):
        if not budget.ok(reserve=0.01): return
        with locks[f]: d = json.load(open(f))
        its = d["instances"][s:s + a.chunk]; titles = list(dict.fromkeys(x["title"] for x in its))
        ctx = {"widget": d["widget"], "instances": [{"title": t, "example": next(x["user_requests"][0] for x in its if x["title"] == t)} for t in titles]}
        sc = schema(titles, a.n); out, c, tot = call(sc, prompt_for(sc, ctx), budget)
        got = {o["title"]: o["more_user_requests"] for o in out["instances"]}
        with locks[f]:
            d = json.load(open(f))
            for x in d["instances"][s:s + a.chunk]:
                if x["title"] in got and not x.get("augmented"): x["user_requests"] += got[x["title"]]; x["augmented"] = True
            json.dump(d, open(f, "w"), indent=1)
        print(f"{os.path.basename(f)}[{s}]  ${c:.4f}  total ${tot:.4f}  {time.time()-t0:.0f}s", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for fu in as_completed([ex.submit(run, f, s) for f, s in jobs]):
            try: fu.result()
            except Exception as e: print("job failed:", e, flush=True)
    print(f"done: ${budget.spent:.4f} spent, {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
