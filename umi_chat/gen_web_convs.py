#!/usr/bin/env python3
"""DeepSeek writes branched *widget conversations* for umi-smart-internet.

A user asks for something visual, then keeps refining it ("make it a pie chart", "add the
weekend", "same thing for my budget"). Every turn is one complete HTML widget built on umi's
design system, plus five phrasings of the user's message; turns sharing a parent are branches.
Only a JSON schema goes out, plus DeepSeek's own earlier output (design notes, one catalog
widget as the opener, titles already used).

  python3 gen_web_convs.py --web data/web --out data/web_convs --budget 1.5
"""
import argparse, itertools, json, os, random, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from gen_data import Budget, call, prompt_for, TAG
from gen_web import RULES

def schema(avoid):
    desc = ("One branched conversation in which a user and umi build and refine visual widgets together, opening "
            "with the widget kind in Context. A tree of 7-9 turns: 1 root, depth 3-4, and at least 2 points where the "
            "user could continue in 2-3 different ways (turns sharing a parent). Later requests depend on the widget "
            "just shown: change the chart type or layout, add or remove data, filter, zoom into one part, compare with "
            "something else, switch units or period, turn it into a checklist/table/dashboard, simplify it. Short "
            "context-dependent requests ('make it a pie', 'add saturday', 'just the top 3') are encouraged. "
            "Every turn's html is the complete new widget, with real, consistent, accurate content. " + RULES)
    if avoid: desc += " Topics already covered (do not repeat): " + "; ".join(avoid)
    turn = {"type": "object", "additionalProperties": False, "required": ["id", "parent", "user_messages", "title", "html"],
            "properties": {
                "id": {"type": "string", "description": "t1, t2, ... unique"},
                "parent": {"type": "string", "description": "id of the turn this one refines, or empty string for the root"},
                "user_messages": {"type": "array", "minItems": 5, "maxItems": 5, "items": {"type": "string"},
                                  "description": "Five different phrasings of the user's request (vary wording, length, typos, slang). "
                                                 "No personal names; use the literal tag " + TAG + " for the user's own name."},
                "title": {"type": "string"},
                "html": {"type": "string", "description": "The complete widget fragment. Under 2500 characters."}}}
    return {"type": "object", "additionalProperties": False, "required": ["topic", "turns"],
            "properties": {"topic": {"type": "string"}, "turns": {"type": "array", "minItems": 7, "maxItems": 9,
                                                                  "description": desc, "items": turn}}}

def valid(c):
    ids = [t["id"] for t in c["turns"]]; seen = set()
    if len(set(ids)) != len(ids): return False
    for t in c["turns"]:
        if t["parent"] and t["parent"] not in seen: return False
        seen.add(t["id"])
    return any(not t["parent"] for t in c["turns"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--web", default="data/web"); ap.add_argument("--out", default="data/web_convs")
    ap.add_argument("--budget", type=float, default=1.5); ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); budget = Budget(a.budget); t0 = time.time()
    design = json.load(open(os.path.join(a.web, "design.json")))
    catalog = [w for w in json.load(open(os.path.join(a.web, "catalog.json")))["widgets"][:63] if not w["wraps_answer"]]
    lock = threading.Lock(); stop = threading.Event(); path = os.path.join(a.out, "convs.json")
    store = json.load(open(path)) if os.path.exists(path) else {"conversations": []}
    def one(w):
        if stop.is_set() or not budget.ok(reserve=0.03): stop.set(); return
        with lock: avoid = [c["topic"] for c in store["conversations"] if c["opener"] == w["name"]][-30:]
        sc = schema(avoid)
        ctx = {"design_notes": design["notes"], "opening_widget": {k: w[k] for k in ("name", "description", "slot_schema", "example_html")}}
        try: out, c, tot = call(sc, prompt_for(sc, ctx), budget, max_tokens=40000)
        except RuntimeError: stop.set(); return
        if not valid(out): print("invalid tree, skipped", flush=True); return
        out["opener"] = w["name"]
        with lock:
            store["conversations"].append(out); json.dump(store, open(path, "w"), indent=1)
            n = len(store["conversations"])
        print(f"{w['name']:<26} +{len(out['turns'])} turns  ({n} convs)  ${c:.4f}  total ${tot:.4f}  {time.time()-t0:.0f}s", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        futs = set(); it = itertools.cycle(catalog)
        while not stop.is_set():
            while len(futs) < a.workers * 2 and not stop.is_set():
                if not budget.ok(reserve=0.05): stop.set(); break
                futs.add(ex.submit(one, next(it)))
            if not futs: break
            done = next(as_completed(futs)); futs.discard(done)
            try: done.result()
            except Exception as e: print("call failed:", e, flush=True)
        for f in futs:
            try: f.result()
            except Exception as e: print("call failed:", e, flush=True)
    print(f"done: ${budget.spent:.4f} spent, {len(store['conversations'])} conversations, {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
