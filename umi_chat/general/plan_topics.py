#!/usr/bin/env python3
"""Phase 1: the topic graph. DeepSeek writes a backbone (categories -> subtopics -> leaves) and a set of
'hats' (independent lists of places, objects, eras, user profiles...) that phase 2 samples stochastically.
Only schemas go out; DeepSeek writes every topic string.

  python3 general/plan_topics.py --out data_general/topics --budget 0.6
"""
import argparse, json, os, time
from concurrent.futures import ThreadPoolExecutor
from orcall import call, prompt_for, Budget, DEEPSEEK

BACKBONE = {"type": "object", "additionalProperties": False, "required": ["categories"],
    "properties": {"categories": {"type": "array", "minItems": 56, "maxItems": 64,
        "description": "Top-level backbone of everything real people bring to a general-purpose chat assistant: "
                       "practical everyday life, how-to and repair, cooking, health, money, work and careers, school "
                       "subjects, science, math, history, geography, arts, language, technology and software, "
                       "relationships, parenting, pets, travel, hobbies, entertainment, sports, law and civic basics, "
                       "philosophy, creative writing, emotional support and small talk, and anything else common. "
                       "Together the categories should tile that space with little overlap.",
        "items": {"type": "object", "additionalProperties": False, "required": ["name", "description"],
                  "properties": {"name": {"type": "string", "description": "snake_case"}, "description": {"type": "string"}}}}}}

def subtopic_schema(n):
    return {"type": "object", "additionalProperties": False, "required": ["subtopics"],
        "properties": {"subtopics": {"type": "array", "minItems": n, "maxItems": n,
            "description": f"{n} distinct subtopics of the category in Context, spanning both the most common things "
                           "people ask about and the long tail. Each has 8 leaves: concrete, specific things a user "
                           "might ask or talk about (a named dish, a specific error, a specific battle, a specific "
                           "life situation), not generic labels.",
            "items": {"type": "object", "additionalProperties": False, "required": ["name", "leaves"],
                      "properties": {"name": {"type": "string"},
                                     "leaves": {"type": "array", "minItems": 8, "maxItems": 8, "items": {"type": "string"}}}}}}}

HATS = {
    "places": "specific places worldwide: cities, towns, regions, islands, rivers, seas, mountains, deserts, parks, neighbourhoods; famous and obscure, every continent",
    "everyday_objects": "everyday physical objects, tools, appliances, materials and household things",
    "foods": "specific foods, dishes, ingredients, drinks and snacks from many cuisines",
    "history": "specific historical events, eras, inventions, empires, movements and turning points from all periods and regions",
    "people": "notable historical or cultural figures (deceased or very widely known) from many fields and countries",
    "nature": "animals, plants, fungi, ecosystems, natural phenomena, weather and geology",
    "activities": "hobbies, sports, crafts, chores, games, outdoor activities and skills",
    "ideas": "abstract concepts, scientific ideas, mathematical notions, philosophical questions and social topics",
    "tech": "generic technologies, software tasks, file formats, programming concepts, devices and internet things",
    "occasions": "occasions, seasons, holidays, life events, and times of day or year",
    "user_profiles": "short descriptions of a person who might be typing to a chatbot right now: age range, situation, "
                     "where they are, what kind of day they are having (e.g. a tired night-shift nurse in Ohio). Very diverse.",
}
def hat_schema(name, desc, n, avoid):
    d = f"{n} items for the list '{name}': {desc}. Short strings, all distinct."
    if avoid: d += " Already listed (do not repeat): " + "; ".join(avoid)
    return {"type": "object", "additionalProperties": False, "required": ["items"],
            "properties": {"items": {"type": "array", "minItems": n, "maxItems": n, "description": d, "items": {"type": "string"}}}}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data_general/topics")
    ap.add_argument("--budget", type=float, default=0.6); ap.add_argument("--hat-rounds", type=int, default=3)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); b = Budget(a.budget); t0 = time.time()
    bp = os.path.join(a.out, "backbone.json")
    if not os.path.exists(bp):
        cats = call(DEEPSEEK, BACKBONE, prompt_for(BACKBONE, None), b)[0]["categories"]
        json.dump({"categories": [dict(c, subtopics=[]) for c in cats]}, open(bp, "w"), indent=1)
    bb = json.load(open(bp))
    def fill(c):
        if c["subtopics"]: return
        sc = subtopic_schema(36)
        c["subtopics"] = call(DEEPSEEK, sc, prompt_for(sc, {"category": c["name"], "description": c["description"]}), b, max_tokens=12000)[0]["subtopics"]
        print(f"{c['name']:<40} {len(c['subtopics'])} subtopics  ${b.spent:.3f}  {time.time()-t0:.0f}s", flush=True)
    with ThreadPoolExecutor(16) as ex: list(ex.map(fill, bb["categories"]))
    json.dump(bb, open(bp, "w"), indent=1)
    hp = os.path.join(a.out, "hats.json"); hats = json.load(open(hp)) if os.path.exists(hp) else {k: [] for k in HATS}
    for r in range(a.hat_rounds):  # each round extends every hat, avoiding a sample of what it already has
        def grow(k):
            sc = hat_schema(k, HATS[k], 120, hats[k][-150:])
            got = call(DEEPSEEK, sc, prompt_for(sc, None), b, max_tokens=12000)[0]["items"]
            seen = {x.lower() for x in hats[k]}; hats[k] += [x for x in got if x.lower() not in seen and not seen.add(x.lower())]
        with ThreadPoolExecutor(12) as ex: list(ex.map(grow, HATS))
        json.dump(hats, open(hp, "w"), indent=1)
        print(f"hats round {r+1}: " + ", ".join(f"{k} {len(v)}" for k, v in hats.items()) + f"  ${b.spent:.3f}", flush=True)
    leaves = sum(len(s["leaves"]) for c in bb["categories"] for s in c["subtopics"])
    print(f"done: {len(bb['categories'])} categories, {leaves} leaves, ${b.spent:.4f}, {time.time()-t0:.0f}s")

if __name__ == "__main__": main()
