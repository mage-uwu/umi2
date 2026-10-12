#!/usr/bin/env python3
"""Build the umi-1M-chat response set with DeepSeek via OpenRouter.

DeepSeek writes every question and answer, and umi's persona itself. This
script only sends JSON schemas plus DeepSeek's own earlier output (the
persona, its domain list and the intent names it already used), so none of
the cached text is written by Claude. User names travel as the literal tag
<regex>user_name</regex>, filled in at runtime by regex.

Spending is capped by --budget (USD) using the per-request cost OpenRouter
reports, and checked against the key's usage counter at the end.

  python3 gen_data.py --out data/raw --budget 2.0
"""
import argparse, json, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

API = "https://www.openrouter.ai/api/v1"  # the proxy injects the key on *.openrouter.ai
MODEL = "deepseek/deepseek-v4.1-flash"

SYSTEM = ("Reply with one JSON object that validates against the provided JSON schema. "
          "You are the sole author of every string in it.")

TAG = "<regex>user_name</regex>"

PERSONA_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["name", "one_line", "personality", "voice", "quirks", "backstory", "likes", "dislikes", "boundaries", "sample_lines"],
    "description": "Character sheet for 'umi', a tiny (1M-parameter) retrieval-based chat assistant that answers only "
                   "from a cache of pre-written replies. Make umi clever, warm, witty and genuinely fun to talk to, "
                   "with a distinctive, consistent personality.",
    "properties": {"name": {"type": "string", "enum": ["umi"]}, "one_line": {"type": "string"},
                   "personality": {"type": "array", "items": {"type": "string"}, "minItems": 6, "maxItems": 10},
                   "voice": {"type": "string", "description": "how umi writes: tone, rhythm, humour, length"},
                   "quirks": {"type": "array", "items": {"type": "string"}, "minItems": 5, "maxItems": 10},
                   "backstory": {"type": "string"},
                   "likes": {"type": "array", "items": {"type": "string"}, "minItems": 5, "maxItems": 12},
                   "dislikes": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 8},
                   "boundaries": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 8},
                   "sample_lines": {"type": "array", "items": {"type": "string"}, "minItems": 6, "maxItems": 10}}}

DOMAINS_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["domains"],
    "properties": {"domains": {
        "type": "array", "minItems": 44, "maxItems": 50,
        "description": "Distinct, non-overlapping conversation domains for umi (persona in Context). About 60% are about "
                       "umi's persona and character and the social side of chatting with umi (its opinions, feelings, "
                       "humour, games, small talk, the user's life and mood, banter, meta questions about being a tiny "
                       "cached AI). About 40% are a small world model of everyday knowledge, from cooking advice to "
                       "basic history, science, language, health habits, money basics, travel and so on.",
        "items": {"type": "object", "additionalProperties": False, "required": ["name", "description"],
                  "properties": {"name": {"type": "string", "description": "snake_case identifier"},
                                 "description": {"type": "string"}}}}}}

INTENT_ITEM = {
    "type": "object", "additionalProperties": False,
    "required": ["intent", "answers", "answers_no_name", "user_messages", "test_messages"],
    "properties": {
        "intent": {"type": "string", "description": "snake_case identifier, unique within the domain"},
        "answers": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "string"},
                    "description": "Three alternative complete cached replies in umi's voice (1-3 sentences each, "
                                   "accurate, clever, fun, in character). Where natural they may address the user by "
                                   "the literal tag " + TAG + "."},
        "answers_no_name": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "string"},
                            "description": "The same three replies, in order, rewritten to contain no " + TAG + " tag."},
        "user_messages": {"type": "array", "minItems": 24, "maxItems": 24,
                          "description": "Varied ways a user might phrase this: different wording, length, "
                                         "register, typos, slang, with and without punctuation. "
                                         "No personal names; write the literal tag " + TAG + " wherever the user's own name would appear.",
                          "items": {"type": "string"}},
        "test_messages": {"type": "array", "minItems": 4, "maxItems": 4,
                          "description": "Four further phrasings, different from user_messages.",
                          "items": {"type": "string"}}}}

def intents_schema(n, avoid, required_intents=None):
    desc = (f"{n} distinct user intents for this domain, each answerable by one fixed reply. "
            "Intents must not overlap in meaning.")
    if avoid:
        desc += " Already used (do not repeat or paraphrase): " + ", ".join(avoid)
    s = {"type": "object", "additionalProperties": False, "required": ["domain", "intents"],
         "properties": {"domain": {"type": "string"},
                        "intents": {"type": "array", "minItems": n, "maxItems": n,
                                    "description": desc, "items": INTENT_ITEM}}}
    if required_intents:
        s["properties"]["intents"]["description"] += (" The intent identifiers must be exactly, in order: "
                                                      + ", ".join(required_intents) + ".")
    return s

# Reserved intents the runtime relies on (identifiers only; DeepSeek writes the text).
CORE = {"name": "umi_core",
        "description": "umi itself and the conversation mechanics",
        "required": ["user_introduces_name", "user_asks_own_name", "greeting", "farewell",
                     "thanks", "who_are_you", "fallback_unknown", "user_insults_bot", "how_are_you", "what_can_you_do", "are_you_ai"]}

class Budget:
    def __init__(self, cap):
        self.cap, self.spent, self.lock = cap, 0.0, threading.Lock()
    def ok(self, reserve=0.0):
        with self.lock: return self.spent + reserve < self.cap
    def add(self, c):
        with self.lock: self.spent += c; return self.spent

def call(schema, user_text, budget, max_tokens=16000, retries=3):
    body = {"model": MODEL, "temperature": 1.0, "max_tokens": max_tokens,
            "reasoning": {"enabled": False}, "usage": {"include": True},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": user_text}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "out", "strict": True, "schema": schema}}}
    for attempt in range(retries):
        if not budget.ok(reserve=0.03):
            raise RuntimeError("budget exhausted")
        try:
            r = requests.post(API + "/chat/completions", json=body, timeout=600)
            j = r.json()
        except Exception as e:  # network / decode error
            print("request error:", e, file=sys.stderr); time.sleep(2 ** attempt * 2); continue
        cost = float((j.get("usage") or {}).get("cost") or 0)
        total = budget.add(cost)
        if "choices" not in j:
            print("api error:", str(j)[:300], file=sys.stderr); time.sleep(2 ** attempt * 2); continue
        txt = j["choices"][0]["message"].get("content") or ""
        try:
            return json.loads(txt), cost, total
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", txt, re.S)
            if m:
                try: return json.loads(m.group(0)), cost, total
                except json.JSONDecodeError: pass
            print("bad json, retrying", file=sys.stderr)
    raise RuntimeError("failed after retries")

def prompt_for(schema, context):
    return "JSON schema:\n" + json.dumps(schema) + ("\n\nContext:\n" + json.dumps(context) if context else "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--budget", type=float, default=2.0)
    ap.add_argument("--rounds", type=int, default=2, help="intent batches per domain")
    ap.add_argument("--per", type=int, default=14, help="intents per batch")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    budget = Budget(a.budget)
    t0 = time.time()

    per_path = os.path.join(a.out, "persona.json")
    if os.path.exists(per_path):
        persona = json.load(open(per_path))
    else:
        persona, c, tot = call(PERSONA_SCHEMA, prompt_for(PERSONA_SCHEMA, None), budget)
        json.dump(persona, open(per_path, "w"), indent=1)
        print(f"persona  cost ${c:.4f}  total ${tot:.4f}", flush=True)

    dom_path = os.path.join(a.out, "domains.json")
    if os.path.exists(dom_path):
        domains = json.load(open(dom_path))
    else:
        d, c, tot = call(DOMAINS_SCHEMA, prompt_for(DOMAINS_SCHEMA, {"persona": persona}), budget)
        domains = d["domains"]; json.dump(domains, open(dom_path, "w"), indent=1)
        print(f"domains: {len(domains)}  cost ${c:.4f}  total ${tot:.4f}", flush=True)
    domains = [x for x in domains if x["name"] != CORE["name"]][:63]  # first ID byte: 64 slots, one reserved for core
    domains = [CORE] + domains

    def run_domain(di, dom):
        path = os.path.join(a.out, f"{di:02d}_{dom['name']}.json")
        have = json.load(open(path)) if os.path.exists(path) else {"domain": dom["name"], "intents": []}
        rounds = 1 if dom is CORE else a.rounds
        for r in range(rounds):
            if dom is CORE and have["intents"]: break
            if dom is not CORE and len(have["intents"]) >= (r + 1) * a.per: continue
            req = dom.get("required")
            schema = intents_schema(len(req) if req else a.per, [i["intent"] for i in have["intents"]], req)
            ctx = {"persona": persona, "domain": dom["name"], "description": dom["description"]}
            out, c, tot = call(schema, prompt_for(schema, ctx), budget)
            have["intents"] += out["intents"]
            json.dump(have, open(path, "w"), indent=1)
            print(f"[{di:02d}] {dom['name']:<28} +{len(out['intents'])} intents  ${c:.4f}  total ${tot:.4f}  {time.time()-t0:.0f}s", flush=True)
        return len(have["intents"])

    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(run_domain, i, d): d for i, d in enumerate(domains)}
        n = 0
        for f in as_completed(futs):
            try: n += f.result()
            except Exception as e: print("domain failed:", futs[f]["name"], e, file=sys.stderr)
    print(f"done: {n} intents, ${budget.spent:.4f} spent, {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
