# umi-general

umi-1M-chat fine-tuned for 3 hours (105,443 steps, 4 CPU threads) on umi's own data plus the general-assistant
corpus (`general/`): 17,573 conversation trees, 480k training messages.

- Checkpoint: `umi-general.bin` (1,287,904 core parameters, BLT MLM), sha256 `4fb37db4d96f526175326ac39b0578647462aa0319930e096b8e47adff760d09`
- Data: `python3 prepare.py data/raw data/convs data_checked data_general/trees` (fact-checked cache: 133,481 replies).
  The model was trained on the unchecked build (`data_all`); the checked build keeps every surviving reply's ID, so it
  is used as is.
- Chat: `./umi_chat chat --data data_checked --load weights/umi-general/umi-general.bin --beta 64 --gamma 8`

## Held-out evaluation (`eval.txt`, fact-checked cache, n=1,523)

| | single-turn | follow-up | topic switch |
|---|---|---|---|
| umi-1M-chat on its own 23k-reply cache | 0.678 | 0.823 | 0.644 |
| **umi-general, 133k-reply cache** | **0.730** | **0.840** | **0.672** |
| umi-general, MLM only | 0.128 | 0.758 | |
| umi-general, n-gram only | 0.500 | 0.388 | |

Held-out messages are unseen phrasings of questions the cache covers.

## Questions the cache does not cover (`demo.txt`)

Retrieval returns the nearest cached reply with high confidence even when it answers a different question:
"what's 15% of 80" gets "25% of 80 is 20"; "how do i reverse a list in python" gets a reply about comments.
The confidence (p) does not separate these from good matches, so a near-miss gate (for example on n-gram overlap
with the matched question) is the next thing to build.
