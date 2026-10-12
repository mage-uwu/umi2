#!/usr/bin/env python3
"""Phase 2: DeepSeek plays the user. Python samples each slot (where the topic comes from, who is typing, what
they want, how they type) so the spread is set here, not by the model's favourite topics. DeepSeek turns each
slot into a realistic opening message plus paraphrases, the user's real goal, and where they might go next.

Topic sources per slot:  backbone leaf 60% | hat pull 25% (1-2 random hat items) | crossover 15% (two leaves
from different categories). Categories are sampled uniformly, so small categories are not drowned out.

  python3 general/gen_prompts.py --topics data_general/topics --out data_general/prompts --budget 5
"""
import argparse, json, os, random, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from orcall import call, prompt_for, Budget, DEEPSEEK, TAG

# (label, weight) -- request kinds and typing styles; DeepSeek writes the actual words.
INTENTS = [("ask a factual question", 12), ("ask for an explanation of how or why something works", 10),
           ("ask how to do something step by step", 10), ("troubleshoot a problem they have right now", 7),
           ("ask for advice on a personal decision", 7), ("ask for a recommendation", 6), ("compare two options", 5),
           ("plan something (a trip, a meal, a schedule, a project)", 5), ("ask the assistant's opinion", 4),
           ("brainstorm ideas", 4), ("ask for something short to be written (a message, caption, toast, poem)", 4),
           ("ask to rewrite, shorten or fix some text they paste", 3), ("check whether their own understanding is right", 3),
           ("share something from their day and chat", 4), ("vent about something frustrating or sad", 3),
           ("joke around or play a word game", 2), ("ask to be quizzed or taught", 2), ("ask what a word or phrase means", 3),
           ("quick calculation, conversion or estimate", 3), ("ask for a fun fact or a short story about it", 3)]
STYLES = [("all lowercase, terse, no punctuation", 14), ("casual with a typo or two", 16), ("polite full sentences", 14),
          ("rambling, gives backstory first", 8), ("non-native English speaker", 7), ("enthusiastic", 5),
          ("frustrated or in a hurry", 6), ("just a few words, like a search query", 12), ("texting style with abbreviations", 6),
          ("precise and technical", 5), ("unsure, hedging, a bit shy", 4), ("a kid or teenager", 3)]
def wpick(rng, xs): return rng.choices([x for x, _ in xs], [w for _, w in xs])[0]

class Sampler:
    def __init__(self, topics, seed):
        bb = json.load(open(os.path.join(topics, "backbone.json")))["categories"]
        self.cats = [(c["name"], [(s["name"], l) for s in c["subtopics"] for l in s["leaves"]]) for c in bb if c["subtopics"]]
        self.hats = json.load(open(os.path.join(topics, "hats.json")))
        self.users = self.hats.pop("user_profiles"); self.rng = random.Random(seed)
    def leaf(self, exclude=None):
        c, ls = self.rng.choice([x for x in self.cats if x[0] != exclude]); s, l = self.rng.choice(ls)
        return c, f"{c} > {s} > {l}"
    def slot(self):
        r = self.rng.random()
        if r < .60: src, topic = "backbone", self.leaf()[1]
        elif r < .85:
            ks = self.rng.sample(sorted(self.hats), self.rng.choice([1, 1, 2]))
            src, topic = "hat", " + ".join(f"{k}: {self.rng.choice(self.hats[k])}" for k in ks)
        else:
            c1, t1 = self.leaf(); src, topic = "crossover", t1 + "  x  " + self.leaf(exclude=c1)[1]
        return {"source": src, "topic": topic, "user": self.rng.choice(self.users),
                "wants_to": wpick(self.rng, INTENTS), "typing_style": wpick(self.rng, STYLES)}

def prompts_schema(n):
    item = {"type": "object", "additionalProperties": False,
            "required": ["slot", "opener", "paraphrases", "user_goal", "follow_ups"],
            "properties": {
                "slot": {"type": "integer", "description": "index of the slot in Context"},
                "opener": {"type": "string", "description": "the user's first message to a chat assistant, exactly as this "
                           "user would type it in this typing style"},
                "paraphrases": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "string"},
                                "description": "four other ways the same user might have typed the same request (vary wording and length)"},
                "user_goal": {"type": "string", "description": "one line: what the user actually needs"},
                "follow_ups": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "string"},
                               "description": "three different directions this user would plausibly take after the first "
                                              "answer (dig deeper, push back, change subject, react, share, ask a related thing)"}}}
    return {"type": "object", "additionalProperties": False, "required": ["prompts"],
            "properties": {"prompts": {"type": "array", "minItems": n, "maxItems": n, "items": item,
                "description": "One item per slot in Context, in order. Each slot describes who is typing to a general "
                               "chat assistant, what they want and how they type, plus a topic seed. Write the messages "
                               "real people send: concrete, specific, with real details (numbers, names of things, "
                               "situations), never a template, and never naming the slot fields. A topic seed is "
                               "inspiration: for a combined seed (+ or x), find the most natural real question that "
                               "connects the parts; if the parts do not connect naturally, build the message around just the part this user "
                               "would most plausibly ask about. Never force an absurd link, and keep to one main request per message. The user profile shapes voice, "
                               "vocabulary and situation; mention their personal circumstances only when it matters to the "
                               "request (roughly one message in four). Use no personal names "
                               "for the user; write the literal tag " + TAG + " where they state their own name."}}}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--topics", default="data_general/topics")
    ap.add_argument("--out", default="data_general/prompts"); ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--per", type=int, default=12); ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-calls", type=int, default=10**6); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    b = Budget(a.budget); t0 = time.time(); smp = Sampler(a.topics, a.seed + len(os.listdir(a.out)))
    lock = threading.Lock(); stop = threading.Event(); n_ok = [0]
    outp = os.path.join(a.out, f"prompts_{int(time.time())}.jsonl"); fp = open(outp, "a")
    def one(k):
        with lock: slots = [smp.slot() for _ in range(a.per)]
        sc = prompts_schema(a.per)
        try: out = call(DEEPSEEK, sc, prompt_for(sc, {"slots": [dict(s, slot=i) for i, s in enumerate(slots)]}), b, max_tokens=8000)[0]
        except RuntimeError as e:
            if "budget" in str(e): stop.set()
            return
        got = 0
        with lock:
            for p in out["prompts"]:
                if not (0 <= p["slot"] < len(slots)) or not p["opener"].strip(): continue
                fp.write(json.dumps(dict(p, seed=slots[p["slot"]])) + "\n"); got += 1
            fp.flush(); n_ok[0] += got
        if k % 25 == 0: print(f"call {k}: total prompts {n_ok[0]}  ${b.spent:.3f}  {time.time()-t0:.0f}s", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        futs = set(); k = 0
        while not stop.is_set() and k < a.max_calls:
            while len(futs) < a.workers * 2 and k < a.max_calls and b.ok(reserve=0.03): futs.add(ex.submit(one, k)); k += 1
            if not futs: break
            done = next(as_completed(futs)); futs.discard(done)
            if not b.ok(reserve=0.03): stop.set()
        for f in futs: f.result()
    print(f"done: {outp}  ${b.spent:.4f}  {time.time()-t0:.0f}s")

if __name__ == "__main__": main()
