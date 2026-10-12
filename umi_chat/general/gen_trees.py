#!/usr/bin/env python3
"""Phase 3: gpt-oss-120b writes a branched conversation tree from each phase-2 prompt, playing both the user
(same typing style, steering by the prompt's follow-up directions; the user profile is withheld so umi cannot
know things the user never said) and umi (persona from
DeepSeek's character sheet, as a capable general assistant). The root turn's user message is the phase-2
opener; its phrasings are taken from phase 2, not regenerated.

Prompts are consumed in file order and each is used once (done ids in trees/done.txt), so the script can be
stopped and restarted, and can run while phase 2 is still writing.

  python3 general/gen_trees.py --raw data/raw --prompts data_general/prompts --out data_general/trees --budget 10
"""
import argparse, glob, json, os, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from orcall import call, prompt_for, Budget, GPTOSS, TAG

UMI = "umi's reply. May address the user by the literal tag " + TAG + " where natural."
NONAME = "Empty string unless assistant contains the tag; then the same reply without it."
def exchange(n_phr, what):
    return {"type": "object", "additionalProperties": False, "required": ["user_messages", "assistant", "assistant_no_name"],
            "properties": {"user_messages": {"type": "array", "minItems": n_phr, "maxItems": n_phr, "items": {"type": "string"},
                                             "description": what + " No personal names; use the literal tag " + TAG + " for the user's own name."},
                           "assistant": {"type": "string", "description": UMI},
                           "assistant_no_name": {"type": "string", "description": NONAME}}}

def tree_schema(n):
    # Branching is explicit: a main path plus alternative continuations hanging off it, which gpt-oss follows
    # far more reliably than free-form parent ids.
    spine = exchange(4, "Four phrasings of the same user message, in this user's typing style. In the first exchange the "
                        "first phrasing must be the opener verbatim.")
    branch = {"type": "object", "additionalProperties": False, "required": ["after", "turns"],
              "properties": {"after": {"type": "integer", "description": "1-based index of the main-path exchange this branch "
                                                                          "follows (the user reacts to that umi reply instead)"},
                             "turns": {"type": "array", "minItems": 1, "maxItems": 2,
                                       "items": exchange(3, "Three phrasings of a different way the user could have continued "
                                                            "(or, for a second turn, continues the branch).")}}}
    return {"type": "object", "additionalProperties": False, "required": ["conversations"],
            "properties": {"conversations": {"type": "array", "minItems": n, "maxItems": n,
                "description": "One conversation per prompt in Context, in order, between that user and umi (persona in "
                               "Context). main: the conversation as it happened, 5-7 exchanges starting from the prompt's "
                               "opener, where each user message reacts to umi's previous reply. branches: 3-4 alternative "
                               "continuations, each hanging off a different main-path exchange, where the user says "
                               "something else instead (the prompt's follow_ups, pushback, 'wait really?', 'why?', a joke, "
                               "a change of subject, 'thanks!'). Short context-dependent user messages are encouraged. The "
                               "user keeps their typing style throughout. umi is a genuinely capable general assistant in "
                               "her own voice: accurate and specific (real numbers, names, steps), complete enough to be "
                               "useful on its own, usually 2-4 sentences or a short numbered list when steps are needed, "
                               "never vague, honest when unsure, kind on emotional topics, and declines unsafe requests "
                               "briefly. umi knows only what the user has said in this conversation; persona catchphrases "
                               "are seasoning, at most one per conversation.",
                "items": {"type": "object", "additionalProperties": False, "required": ["prompt", "main", "branches"],
                          "properties": {"prompt": {"type": "integer", "description": "index of the prompt in Context"},
                                         "main": {"type": "array", "minItems": 5, "maxItems": 7, "items": spine},
                                         "branches": {"type": "array", "minItems": 3, "maxItems": 4, "items": branch}}}}}}

def to_turns(cv):
    """main + branches -> [{id, parent, user_messages, assistant, assistant_no_name}] with t1 the root."""
    turns, main = [], cv["main"]
    for i, x in enumerate(main): turns.append(dict(x, id=f"t{i+1}", parent=f"t{i}" if i else ""))
    k = len(main)
    for br in cv["branches"]:
        if not 1 <= br["after"] <= len(main): continue
        par = f"t{br['after']}"
        for x in br["turns"]:
            k += 1; turns.append(dict(x, id=f"t{k}", parent=par)); par = f"t{k}"
    return turns

def real(x): return len(x.strip().strip(".")) >= 2  # rejects "", "...", placeholders
def valid(turns):
    return len(turns) >= 6 and all(real(t["assistant"]) and t["user_messages"] and all(real(m) for m in t["user_messages"]) for t in turns)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--prompts", default="data_general/prompts"); ap.add_argument("--out", default="data_general/trees")
    ap.add_argument("--budget", type=float, default=10.0); ap.add_argument("--per", type=int, default=2)
    ap.add_argument("--workers", type=int, default=48); ap.add_argument("--max-calls", type=int, default=10**6)
    ap.add_argument("--wait-for", default="", help="pid of a running phase 2; wait for new prompts while it lives")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    persona = json.load(open(os.path.join(a.raw, "persona.json")))
    b = Budget(a.budget); t0 = time.time(); lock = threading.Lock(); stop = threading.Event()
    donep = os.path.join(a.out, "done.txt"); done = set(open(donep).read().split()) if os.path.exists(donep) else set()
    dfp = open(donep, "a"); outp = os.path.join(a.out, f"trees_{int(time.time())}.jsonl"); fp = open(outp, "a")
    stats = {"calls": 0, "trees": 0, "turns": 0, "bad": 0}; offsets = {}

    def fresh():  # unread prompts, as (uid, prompt); uid = file:line
        new = []
        for f in sorted(glob.glob(os.path.join(a.prompts, "*.jsonl"))):
            with open(f) as h:
                h.seek(offsets.get(f, (0, 0))[0]); ln = offsets.get(f, (0, 0))[1]
                while True:
                    line = h.readline()
                    if not line.endswith("\n"): break  # partial line still being written
                    uid = f"{os.path.basename(f)}:{ln}"; ln += 1
                    if uid not in done: new.append((uid, json.loads(line)))
                offsets[f] = (h.tell() - len(line.encode()) if line else h.tell(), ln)
        return new

    def alive(): return a.wait_for and os.path.exists(f"/proc/{a.wait_for}")

    def one(batch):
        ctx = {"persona": persona, "prompts": [{"prompt": i, "opener": p["opener"],
               "typing_style": p["seed"]["typing_style"], "user_goal": p["user_goal"], "follow_ups": p["follow_ups"]}
               for i, (_, p) in enumerate(batch)]}
        sc = tree_schema(len(batch))
        try: out = call(GPTOSS, sc, prompt_for(sc, ctx), b, max_tokens=8000)[0]
        except RuntimeError as e:
            if "budget" in str(e): stop.set()
            return
        with lock:
            stats["calls"] += 1
            for cv in out["conversations"]:
                i = cv.get("prompt")
                turns = to_turns(cv) if isinstance(i, int) and 0 <= i < len(batch) else []
                if not valid(turns): stats["bad"] += 1; continue
                uid, p = batch[i]
                turns[0]["user_messages"] = [p["opener"]] + p["paraphrases"]
                fp.write(json.dumps({"uid": uid, "seed": p["seed"], "user_goal": p["user_goal"], "turns": turns}) + "\n")
                stats["trees"] += 1; stats["turns"] += len(turns)
            for uid, _ in batch: dfp.write(uid + "\n")
            fp.flush(); dfp.flush()
            if stats["calls"] % 50 == 0:
                print(f"calls {stats['calls']}  trees {stats['trees']}  turns {stats['turns']}  invalid {stats['bad']}  "
                      f"${b.spent:.3f}  {time.time()-t0:.0f}s", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        futs, queue, k = set(), [], 0
        while not stop.is_set() and k < a.max_calls:
            if len(queue) < a.per * a.workers * 2: queue += fresh()
            while len(futs) < a.workers * 2 and len(queue) >= a.per and k < a.max_calls and b.ok(reserve=0.02):
                futs.add(ex.submit(one, queue[:a.per])); queue = queue[a.per:]; k += 1
            if not b.ok(reserve=0.02): break
            if not futs:
                if alive(): time.sleep(5); continue
                break
            done_f = next(as_completed(futs)); futs.discard(done_f)
            try: done_f.result()
            except Exception as e: print("call failed:", e, flush=True)
        for f in futs: f.result()
    print(f"done: {stats}  ${b.spent:.4f}  {time.time()-t0:.0f}s  -> {outp}")

if __name__ == "__main__": main()
