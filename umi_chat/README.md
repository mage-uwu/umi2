# umi-1M-chat

A retrieval chatbot built on the bidirectional masked-LM in `../nanotss_blt_mlm.c`.
umi never generates text token by token. The MLM reads the user's message and fills a
masked two-byte answer slot (`<message> SEP [MASK] [MASK] END`). That is a typed
decision: byte 1 picks a domain, byte 2 an intent within it. The chosen cached reply
is then retrieved.

* **Model**: BLT mode, 1,287,904 core parameters plus a hashed 3..8-byte n-gram cache
  (`--ngram-bins`, excluded from the budget like upstream) and a frozen 42,720-parameter
  entropy patcher.
* **Data**: everything was written by DeepSeek V4.1 Flash via OpenRouter, for about $1.96
  in total. The scripts only send JSON schemas plus DeepSeek's own earlier output.
  DeepSeek wrote umi's persona (`data/raw/persona.json`), 49 domains plus the reserved
  `umi_core`, 2,608 intents with 3 answer variants each, and about 112k user phrasings.
* **Names**: user names never reach the model. A regex rewrites them to the tag
  `<regex>user_name</regex>`, and the same tag inside a cached answer is replaced with the
  remembered name. If no name is known, the answer's no-name variant is used.

```
python3 gen_data.py --out data/raw --budget 1.75   # persona, domains, intents
python3 augment.py --raw data/raw --budget 0.55    # more user phrasings
python3 prepare.py data/raw data                   # train/test/answers tables
cc -O3 -march=native -fopenmp umi_chat.c -lm -o umi_chat
./umi_chat train --data data --seconds 7200 --threads 8 --save umi-1M-chat.bin
./umi_chat eval  --data data --load umi-1M-chat.bin
./umi_chat chat  --data data --load umi-1M-chat.bin -v
```

`pod_train.sh` is the start script for the RunPod CPU pod. It clones this branch,
trains, evaluates, and serves `/root/ckpt` on port 8888.
