#!/usr/bin/env python3
"""DeepSeek writes whole branched conversations between a user and umi.

DeepSeek plays both sides. Each turn is one user message (five phrasings)
plus umi's reply; a turn's parent is the turn it follows, so several turns
sharing a parent are branches (different ways the user might continue).
Only a JSON schema goes out, plus DeepSeek's own persona, domain and the
conversation topics it already wrote for that domain.

  python3 gen_convs.py --raw data/raw --out data/convs --budget 2.0
"""
import argparse, json, os, threading, time, itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from gen_data import Budget, call, prompt_for, TAG

def conv_schema(n_conv, avoid):
    desc = (f"{n_conv} distinct branched conversations between a user and umi (persona in Context) about this "
            "domain. Each is a tree of 10-14 turns: 1-2 root turns, depth 3-5, and at least 3 points where the user "
            "could continue in 2-3 different ways (different turns with the same parent: follow-up questions, "
            "reactions like 'haha' or 'wait, really?', disagreement, 'tell me more', changing the subject, "
            "sharing something personal). Later user messages should depend on umi's previous reply; short "
            "context-dependent follow-ups ('why?', 'ok and then?', 'lol same') are encouraged. umi stays in "
            "character: clever, warm, fun, accurate, 1-3 sentences.")
    if avoid:
        desc += " Topics already covered (do not repeat): " + "; ".join(avoid)
    turn = {"type": "object", "additionalProperties": False,
            "required": ["id", "parent", "user_messages", "assistant", "assistant_no_name"],
            "properties": {
                "id": {"type": "string", "description": "t1, t2, ... unique within the conversation"},
                "parent": {"type": "string", "description": "id of the turn this one follows, or empty string for a root turn"},
                "user_messages": {"type": "array", "minItems": 5, "maxItems": 5, "items": {"type": "string"},
                                  "description": "Five different phrasings of the same user message (vary wording, length, "
                                                 "typos, slang). No personal names; use the literal tag " + TAG + " for the user's own name."},
                "assistant": {"type": "string", "description": "umi's reply. May address the user by the literal tag " + TAG + " where natural."},
                "assistant_no_name": {"type": "string", "description": "Empty string unless assistant contains the tag; "
                                                                       "then the same reply without it."}}}
    return {"type": "object", "additionalProperties": False, "required": ["conversations"],
            "properties": {"conversations": {"type": "array", "minItems": n_conv, "maxItems": n_conv, "description": desc,
                "items": {"type": "object", "additionalProperties": False, "required": ["topic", "turns"],
                          "properties": {"topic": {"type": "string"},
                                         "turns": {"type": "array", "minItems": 10, "maxItems": 14, "items": turn}}}}}}

def valid(conv):
    ids = [t["id"] for t in conv["turns"]]
    if len(set(ids)) != len(ids): return False
    seen = set()
    for t in conv["turns"]:  # parents must come earlier (a tree, no cycles)
        if t["parent"] and t["parent"] not in seen: return False
        seen.add(t["id"])
    return any(not t["parent"] for t in conv["turns"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw"); ap.add_argument("--out", default="data/convs")
    ap.add_argument("--budget", type=float, default=2.0); ap.add_argument("--per", type=int, default=2)
    ap.add_argument("--workers", type=int, default=12); ap.add_argument("--max-calls", type=int, default=100000)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    budget = Budget(a.budget); t0 = time.time()
    persona = json.load(open(os.path.join(a.raw, "persona.json")))
    domains = [{"name": "umi_core", "description": "umi itself and the conversation mechanics"}] + json.load(open(os.path.join(a.raw, "domains.json")))
    locks = {d["name"]: threading.Lock() for d in domains}
    stop = threading.Event(); calls = itertools.count()

    def path(d): return os.path.join(a.out, d["name"] + ".json")
    def load(d): return json.load(open(path(d))) if os.path.exists(path(d)) else {"domain": d["name"], "conversations": []}

    def one(d):
        if stop.is_set() or next(calls) >= a.max_calls: return 0
        with locks[d["name"]]: avoid = [c["topic"] for c in load(d)["conversations"]]
        sc = conv_schema(a.per, avoid[-40:])
        try:
            out, c, tot = call(sc, prompt_for(sc, {"persona": persona, "domain": d["name"], "description": d["description"]}), budget, max_tokens=24000)
        except RuntimeError as e:
            stop.set(); return 0
        good = [cv for cv in out["conversations"] if valid(cv)]
        with locks[d["name"]]:
            have = load(d); have["conversations"] += good; json.dump(have, open(path(d), "w"), indent=1)
        n = sum(len(cv["turns"]) for cv in good)
        print(f"{d['name']:<34} +{len(good)} convs {n} turns  ${c:.4f}  total ${tot:.4f}  {time.time()-t0:.0f}s", flush=True)
        return n

    # round-robin over domains until the budget runs out
    with ThreadPoolExecutor(a.workers) as ex:
        futs = set(); it = itertools.cycle(domains)
        while not stop.is_set():
            while len(futs) < a.workers * 2 and not stop.is_set():
                if not budget.ok(reserve=0.05): stop.set(); break
                futs.add(ex.submit(one, next(it)))
            done = next(as_completed(futs)); futs.discard(done)
            try: done.result()
            except Exception as e: print("call failed:", e, flush=True)
        for f in futs:
            try: f.result()
            except Exception as e: print("call failed:", e, flush=True)
    print(f"done: ${budget.spent:.4f} spent, {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
