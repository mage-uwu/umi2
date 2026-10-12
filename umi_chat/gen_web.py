#!/usr/bin/env python3
"""umi-smart-internet: DeepSeek builds umi's visual vocabulary.

Stage 1  a design system (one CSS stylesheet, light + dark) for umi's in-app interface
Stage 2  a catalog of client-side widgets, containers and dashboard layouts on it
Stage 3  many cached instances per widget: a ready HTML fragment plus the user
         requests that should retrieve it. Wrapper widgets hold an {{answer}} slot
         so any umi reply can be dressed in them.

Only JSON schemas go out, plus DeepSeek's own earlier output (persona, CSS,
catalog, titles already used). No <script>, no external resources.

  python3 gen_web.py --raw data/raw --out data/web --budget 2.0
"""
import argparse, itertools, json, os, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from gen_data import Budget, call, prompt_for, TAG

RULES = ("Static HTML only: no <script>, no event handlers, no external URLs, fonts or images. Interactivity only "
         "via <details>/<summary> or CSS. Graphics as inline <svg>. Use the design system's classes and CSS variables "
         "so it works in light and dark mode. Mobile-friendly. Address the user only via the literal tag " + TAG + ".")

DESIGN_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["name", "css", "notes"],
    "description": "A compact, beautiful design system for umi's own in-app interface (persona in Context): the chat "
                   "transcript plus visual widgets umi can show inline. " + RULES,
    "properties": {
        "name": {"type": "string"},
        "css": {"type": "string", "description": "One complete stylesheet: :root tokens (colours, spacing, radius, type scale) "
                "with a prefers-color-scheme dark override; base typography; .umi-page layout; chat bubbles (.msg.user, "
                ".msg.umi); and reusable component classes for cards, callouts, badges, stat tiles, tables, lists, "
                "timelines, progress bars, grids/dashboards, tabs and accordions (details), chart containers, quiz/flash "
                "cards, recipe cards, code blocks, quotes. Under 9000 characters."},
        "notes": {"type": "string", "description": "Short guide to the class names and tokens, for writing widgets."}}}

CATALOG_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["widgets"],
    "properties": {"widgets": {
        "type": "array", "minItems": 44, "maxItems": 52,
        "description": "Catalog of client-side widgets, containers and dashboard layouts umi can show, built on the design "
                       "system in Context. Cover: text containers (card, callout, quote, steps), data (table, stat tiles, "
                       "progress, timeline, comparison), SVG graphics (bar/line/pie/donut charts, gauges, simple "
                       "diagrams and illustrations), learning (flashcards, quiz, glossary), everyday (recipe card, "
                       "checklist, schedule, budget, habit tracker, weather-style summary), web-dev references (colour "
                       "palette, CSS snippet card, component spec, HTTP status table, HTML cheat sheet), and whole "
                       "dashboards. About 10 are wrappers that present any umi reply. " + RULES,
        "items": {"type": "object", "additionalProperties": False,
                  "required": ["name", "category", "description", "wraps_answer", "slot_schema", "example_html"],
                  "properties": {
                      "name": {"type": "string", "description": "snake_case, unique"},
                      "category": {"type": "string"},
                      "description": {"type": "string", "description": "what it is for and when umi should use it"},
                      "wraps_answer": {"type": "boolean", "description": "true if it presents an arbitrary umi reply via the literal {{answer}} slot"},
                      "slot_schema": {"type": "string", "description": "JSON-schema-like description of the content slots"},
                      "example_html": {"type": "string", "description": "one complete example fragment (wrappers contain {{answer}})"}}}}}}

def instances_schema(n, w, avoid):
    desc = (f"{n} distinct cached instances of the widget '{w['name']}' (spec in Context). ")
    if w["wraps_answer"]:
        desc += ("Each is a variant of the wrapper (different accent, icon, label or framing) whose html contains the "
                 "literal {{answer}} exactly once, and user_requests are ways a user asks for a reply to be shown that way. ")
    else:
        desc += ("Each is a complete, self-contained fragment with real, accurate, useful content on a different "
                 "everyday subject, and user_requests are what a user would type to get it. ")
    desc += RULES
    if avoid: desc += " Titles already used (do not repeat): " + "; ".join(avoid)
    return {"type": "object", "additionalProperties": False, "required": ["instances"],
            "properties": {"instances": {"type": "array", "minItems": n, "maxItems": n, "description": desc,
                "items": {"type": "object", "additionalProperties": False, "required": ["title", "user_requests", "html"],
                          "properties": {"title": {"type": "string"},
                                         "user_requests": {"type": "array", "minItems": 7, "maxItems": 7, "items": {"type": "string"},
                                                           "description": "Seven varied phrasings (wording, length, typos, slang)."},
                                         "html": {"type": "string", "description": "The fragment. Under 2500 characters."}}}}}}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw"); ap.add_argument("--out", default="data/web")
    ap.add_argument("--budget", type=float, default=2.0); ap.add_argument("--per", type=int, default=6)
    ap.add_argument("--max-per-widget", type=int, default=60); ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); budget = Budget(a.budget); t0 = time.time()
    persona = json.load(open(os.path.join(a.raw, "persona.json")))
    p = os.path.join(a.out, "design.json")
    if os.path.exists(p): design = json.load(open(p))
    else:
        design, c, tot = call(DESIGN_SCHEMA, prompt_for(DESIGN_SCHEMA, {"persona": persona}), budget, max_tokens=12000)
        json.dump(design, open(p, "w"), indent=1); print(f"design system {len(design['css'])} chars  ${c:.4f}", flush=True)
    p = os.path.join(a.out, "catalog.json")
    if os.path.exists(p): catalog = json.load(open(p))
    else:
        catalog, c, tot = call(CATALOG_SCHEMA, prompt_for(CATALOG_SCHEMA, {"persona": persona["one_line"], "design_notes": design["notes"], "css": design["css"]}), budget, max_tokens=60000)
        json.dump(catalog, open(p, "w"), indent=1); print(f"catalog {len(catalog['widgets'])} widgets  ${c:.4f}  total ${tot:.4f}", flush=True)
    widgets = catalog["widgets"][:63]
    locks = [threading.Lock() for _ in widgets]; stop = threading.Event()
    def path(i): return os.path.join(a.out, f"w{i:02d}_{widgets[i]['name']}.json")
    def load(i): return json.load(open(path(i))) if os.path.exists(path(i)) else {"widget": widgets[i]["name"], "instances": []}
    def one(i):
        if stop.is_set(): return
        w = widgets[i]
        with locks[i]: have = load(i)
        if len(have["instances"]) >= (a.max_per_widget if not w["wraps_answer"] else 12): return
        sc = instances_schema(a.per, w, [x["title"] for x in have["instances"]][-60:])
        ctx = {"design_notes": design["notes"], "widget": w, "umi_voice": persona["voice"]}
        try: out, c, tot = call(sc, prompt_for(sc, ctx), budget, max_tokens=30000)
        except RuntimeError: stop.set(); return
        with locks[i]:
            have = load(i); have["instances"] += out["instances"]; json.dump(have, open(path(i), "w"), indent=1)
        print(f"[w{i:02d}] {w['name']:<30} +{len(out['instances'])} ({len(have['instances'])})  ${c:.4f}  total ${tot:.4f}  {time.time()-t0:.0f}s", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        futs = set(); it = itertools.cycle(range(len(widgets))); idle = 0
        while not stop.is_set() and idle < len(widgets):
            while len(futs) < a.workers * 2 and not stop.is_set():
                if not budget.ok(reserve=0.06): stop.set(); break
                i = next(it); w = widgets[i]
                if len(load(i)["instances"]) >= (a.max_per_widget if not w["wraps_answer"] else 12):
                    idle += 1
                    if idle >= len(widgets): break
                    continue
                idle = 0; futs.add(ex.submit(one, i))
            if not futs: break
            done = next(as_completed(futs)); futs.discard(done)
            try: done.result()
            except Exception as e: print("call failed:", e, flush=True)
        for f in futs:
            try: f.result()
            except Exception as e: print("call failed:", e, flush=True)
    print(f"done: ${budget.spent:.4f} spent, {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
