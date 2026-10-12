# umi-1M-chat

umi-1M-chat is a multi-turn retrieval chatbot built on the bidirectional masked-LM in
`../nanotss_blt_mlm.c`. umi never generates text token by token. Every reply is a cached
answer written by DeepSeek, and each turn is a typed decision about which one to retrieve.

```
[last reply ID][reply before that] GS <user message> SEP [MASK][MASK][MASK] END
```

* **Answer IDs** have three bytes, `(a, b, t)`. A single-turn intent is
  `(domain, intent, 0)`. A turn in a DeepSeek conversation tree is
  `(conversation hi, conversation lo, turn)`, so a follow-up shares its first two bytes
  with the reply it follows.
* **Retrieval** has two stages:
  1. A large hashed character 3–5-gram cache (TF-IDF over every answer's phrasings)
     proposes candidates. Every turn of the conversation trees in context is added to them.
  2. The MLM scores each candidate's ID through the masked slot. The reply is chosen by
     `log p_MLM + beta * cosine + gamma * [candidate directly follows umi's last reply]`.
     `eval` tunes `beta` and `gamma` on half of the test set.
* **Names**: a regex rewrites the user's name to the tag `<regex>user_name</regex>`
  before the model sees it. The same tag inside a cached answer is replaced with the
  remembered name. If no name is known, the answer's no-name variant is used.
* **Model**: BLT mode with 1,287,904 core parameters, plus the BLT hashed n-gram
  embeddings (`--ngram-bins`, excluded from the budget as upstream does) and a frozen
  42,720-parameter entropy patcher.

## Data (all written by DeepSeek V4.1 Flash via OpenRouter, about $3.95 in total)

The scripts send DeepSeek only JSON schemas plus its own earlier output. DeepSeek wrote:

* umi's persona (`data/raw/persona.json`)
* 49 domains plus the reserved `umi_core`, with 2,608 single-turn intents, 3 answer
  variants each, and about 112k user phrasings (`gen_data.py`, `augment.py`)
* 1,276 branched conversations with 15,376 turns, playing both the user and umi. Each user
  turn has 5 phrasings, and branches share a parent (`gen_convs.py`, in `data/convs/`).

`prepare.py` builds `data/train.tsv` and `data/test.tsv` (`a.b.t`, context IDs,
message) and `data/answers.tsv`. The test set holds out phrasings never seen in training.

```
python3 gen_data.py  --out data/raw --budget 1.75    # persona, domains, intents
python3 augment.py   --raw data/raw --budget 0.55    # more user phrasings
python3 gen_convs.py --raw data/raw --out data/convs --budget 2.0
python3 prepare.py data/raw data/convs data
cc -O3 -march=native -fopenmp umi_chat.c -lm -o umi_chat
./umi_chat train --data data --seconds 10800 --threads 8 --save umi-1M-chat.bin
./umi_chat eval  --data data --load umi-1M-chat.bin --eval-n 2000
./umi_chat chat  --data data --load umi-1M-chat.bin --beta 8 --gamma 8 -v
```

`pod_train.sh` is the start script for the RunPod CPU pod. It clones this branch, trains,
evaluates, runs multi-turn demos, and serves `/root/ckpt` on port 8888.
