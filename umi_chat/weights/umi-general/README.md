# umi-general

umi-1M-chat fine-tuned for 3 hours (105,443 steps, 4 CPU threads) on umi's own data plus the general-assistant
corpus (`general/`): 17,573 conversation trees, 480k training messages.

- Checkpoint: `umi-general.bin` (1,287,904 core parameters, BLT MLM), sha256 `4fb37db4d96f526175326ac39b0578647462aa0319930e096b8e47adff760d09`
- Data: `python3 prepare.py data/raw data/convs data_checked data_general/trees` (fact-checked cache: 133,481 replies).
  The model was trained on the unchecked build (`data_all`); the checked build keeps every surviving reply's ID, so it
  is used as is.
- Chat: `./umi_chat chat --data data_checked --load weights/umi-general/umi-general.bin --beta 64 --gamma 4` (near-miss gate on by default: `--gate 0.3 --gate-follow 0`)
- Browser: https://claude.ai/artifact/5BEfqA294fyAeMP6DoFqDT (version 5)

## Held-out evaluation (`eval.txt`, fact-checked cache, n=1,523)

| | single-turn | follow-up | topic switch |
|---|---|---|---|
| umi-1M-chat on its own 23k-reply cache | 0.678 | 0.823 | 0.644 |
| **umi-general, 133k-reply cache** | **0.730** | **0.840** | **0.672** |
| umi-general, MLM only | 0.128 | 0.758 | |
| umi-general, n-gram only | 0.500 | 0.388 | |

Held-out messages are unseen phrasings of questions the cache covers.

## Strict trees and the near-miss gate (`eval_gate.txt`, `eval_gate_loose.txt`, same 1,001 held-out messages)

Two changes after the first evaluation:

- **Strict trees.** A conversation turn below the root is a candidate only right after its own parent (the
  widget cache already worked this way). Before, "why does that work?" after a sourdough answer could take the
  reply cached for "why does that work?" in a phone-charging conversation. `--loose` restores the old behaviour.
- **Near-miss gate.** The reply is given only if enough of the message's informative words (IDF-weighted,
  words in more than 1 message in 200 count zero) appear in one of the questions it was cached for, and every
  number matches. Otherwise umi gives its "not in the cache" reply.

The gate was tuned without new data: for each held-out message, the conversation (or whole topic group) it
belongs to is hidden, so whatever comes back is a near miss by construction.

| | single-turn | follow-up | topic switch | uncovered topics refused (gate 0.3) | fresh replies correct when answered (gate 0.3) |
|---|---|---|---|---|---|
| loose | 0.651 | 0.876 | 0.582 | 46% | 0.552 |
| **strict (default)** | **0.667** | **0.905** | **0.640** | **79%** | **0.640** |

With the gate at 0.3, 83% of in-cache fresh messages are still answered. For follow-ups the gate only removed
good replies, so it is off there (`--gate-follow 0`). Hiding a single conversation is a weak test (many
conversations ask the same thing, so the "near miss" is often a fine answer); hiding the topic group is the
one that matches questions the cache really lacks.

## Questions the cache does not cover (`demo.txt`)

Retrieval returns the nearest cached reply with high confidence even when it answers a different question:
"what's 15% of 80" gets "25% of 80 is 20"; "how do i reverse a list in python" gets a reply about comments.
The confidence (p) does not separate these from good matches; the gate above does for most of them ("15% of 80"
is now refused). It still lets through near misses that share a key word: "how many cups in a liter" matches a
cached question about cups in a cake pan (word coverage 0.38).
