"""Shared OpenRouter call for the general-assistant corpus (model per call, spend capped by Budget)."""
import json, re, sys, time, requests, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gen_data import API, SYSTEM, TAG, Budget, prompt_for  # noqa: F401

DEEPSEEK = "deepseek/deepseek-v4.1-flash"
GPTOSS = "openai/gpt-oss-120b"
REASONING = {DEEPSEEK: {"enabled": False}, GPTOSS: {"effort": "low"}}
# Measured on this schema: CoreWeave (cheapest) loops on whitespace under JSON mode until max_tokens, Google
# fills strings with "...", Venice has no JSON mode. Crusoe: valid, 8-11 s per call, ~$0.0007 per 2 trees; Mancer and DekaLLM
# are as cheap and valid but slower. Groq is valid at 3x the price.
PROVIDER = {GPTOSS: {"order": ["crusoe", "mancer", "dekallm"], "allow_fallbacks": False, "require_parameters": True}}

def call(model, schema, user_text, budget, max_tokens=16000, retries=3, temperature=1.0):
    body = {"model": model, "temperature": temperature, "max_tokens": max_tokens, "usage": {"include": True},
            "reasoning": REASONING[model],
            **({"provider": PROVIDER[model]} if model in PROVIDER else {}),
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_text}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "out", "strict": True, "schema": schema}}}
    for attempt in range(retries):
        if not budget.ok(reserve=0.02): raise RuntimeError("budget exhausted")
        try:
            j = requests.post(API + "/chat/completions", json=body, timeout=900).json()
        except Exception as e:
            print("request error:", e, file=sys.stderr, flush=True); time.sleep(2 ** attempt * 2); continue
        cost = float((j.get("usage") or {}).get("cost") or 0); total = budget.add(cost)
        if "choices" not in j:
            print("api error:", str(j)[:300], file=sys.stderr, flush=True); time.sleep(2 ** attempt * 2); continue
        if j["choices"][0].get("finish_reason") == "length":
            print("truncated, retrying", file=sys.stderr, flush=True); continue
        txt = j["choices"][0]["message"].get("content") or ""
        for cand in (txt, (re.search(r"\{.*\}", txt, re.S) or [None])[0]):
            if cand:
                try: return json.loads(cand), cost, total
                except json.JSONDecodeError: pass
        print("bad json, retrying", file=sys.stderr, flush=True)
    raise RuntimeError("failed after retries")
