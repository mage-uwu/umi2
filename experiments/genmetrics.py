#!/usr/bin/env python3
"""Independent generation metrics (not the generator's own likelihood).
wvr      : share of word tokens found in the reference vocabulary (train split + context)
bigram   : share of adjacent word pairs seen in the reference text
speakers : share of speaker headings (NAME:) that are real speakers in the reference
ctxnames : share of speaker headings that occur as speakers in the context (ICL use of names)
distinct2: distinct word bigrams / bigrams (higher = less repetitive)"""
import re, sys, json, collections
W = re.compile(r"[A-Za-z']+")
def words(t): return [w.lower().strip("'") for w in W.findall(t) if w.strip("'")]
def speakers(t): return set(m.strip() for m in re.findall(r"^([A-Z][A-Za-z' ]{1,30}):\s*$", t, re.M))
class Ref:
    def __init__(self, train_path, ctx_text=""):
        raw = open(train_path, encoding="latin-1").read(); raw = raw[:int(len(raw)*.9)] + "\n" + ctx_text
        ws = words(raw); self.vocab = set(ws); self.bi = set(zip(ws, ws[1:]))
        self.spk = speakers(raw); self.ctxspk = speakers(ctx_text)
    def score(self, text):
        ws = words(text); bg = list(zip(ws, ws[1:])); sp = re.findall(r"^([A-Z][A-Za-z' ]{1,30}):\s*$", text, re.M)
        return {"wvr": sum(w in self.vocab for w in ws)/max(1,len(ws)),
                "bigram": sum(b in self.bi for b in bg)/max(1,len(bg)),
                "speakers": sum(s.strip() in self.spk for s in sp)/max(1,len(sp)),
                "ctxnames": sum(s.strip() in self.ctxspk for s in sp)/max(1,len(sp)) if self.ctxspk else 0,
                "distinct2": len(set(bg))/max(1,len(bg)), "words": len(ws)}
def mean(rows):
    keys = rows[0].keys(); return {k: sum(r[k] for r in rows)/len(rows) for k in keys}
if __name__ == "__main__":
    train, ctx = sys.argv[1], (open(sys.argv[2]).read() if len(sys.argv) > 2 and sys.argv[2] != "-" else "")
    ref = Ref(train, ctx); rows = [ref.score(json.loads(l)["text"]) for l in sys.stdin if l.strip()]
    print(json.dumps({k: round(v, 4) for k, v in mean(rows).items()}))
