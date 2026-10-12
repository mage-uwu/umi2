# umi-smart-internet: widget model

**Current weights: the multi-turn fine-tune** (see the end of this file). The single-turn
model described first was its starting point.

This is a 1.5-hour run on the RunPod CPU pod "UMI", stage 2 of `pod_train.sh`. It was trained on `data_web/`: 3,757 cached HTML widget instances DeepSeek wrote, plus the "plain" (no widget) class.
The model has the same architecture as umi-1M-chat (1,287,904 core parameters, BLT with a hashed n-gram cache).

* Held-out requests, exact instance out of 3,758 (`eval_web.txt`): MLM 0.279, n-gram cache 0.386, **hybrid 0.409** (beta 32).
* `train.log` has both pod stages. `demo_*.html` are pages compiled on the pod with both models.
  They predate the topic-switch fix in `5b97149`, so some chat replies there stay in an old conversation thread.

```
./umi_chat chat --data data --load weights/umi-1M-chat/umi-1M-chat.bin --beta 32 --gamma 2 \
  --web data_web --web-load weights/umi-smart-internet/umi-smart-internet.bin --html umi.html -v
```

## Widget gate (after the "plain is always a candidate" fix)

These numbers come from 1,400 held-out widget requests and 641 held-out ordinary chat messages that should get no widget.

| gate | widget requests shown | right widget type when shown | chat messages given a widget |
|---|---|---|---|
| 0.35 (old default) | 97.4% | 67.7% | 27.6% |
| 0.60 | 87.0% | 70.0% | 20.9% |
| **0.90 (default)** | **63.6%** | **76.0%** | **11.7%** |

Before the fix, "plain" (no widget) was almost never in the n-gram top-16 candidate list, because its phrasings are 4,000 unrelated chat messages. At the old gate, 80.5% of chat messages got a widget.
Many of the remaining "chat" hits are useful: "best month to visit Japan" gets a table and "compound interest" gets a glossary.
With the fix, the exact-instance accuracy on the full widget test set rises from 0.409 to 0.487.

Wrapper widgets ("put that in a card", "make it a flashcard") restyle umi's *previous* reply. This happens only when the message points back at it ("that", "it", "this", ...) and the widget model's confidence is at least 0.5. Then the chat model is not asked and the conversation context stays where it was.


## Multi-turn fine-tune (current weights)

These weights were fine-tuned for 1.5 hours (50,838 steps, lr 0.0015) with `umi_chat train --load` on `data_web/`.
The data adds DeepSeek's 501 branched widget conversations (4,132 turns, $1.49, `gen_web_convs.py`). In each, a user asks for something visual and then refines it.
The UMI pod could not start (its host had no free vCPUs), so this run used 4 threads in the Claude Code container.
`finetune_multiturn.log` is the training log and `eval_multiturn.txt` the hybrid eval.

Held-out requests, exact widget out of 7,890 (eval-n 700 split):

| | single-turn request | refinement of the widget on screen | topic switch |
|---|---|---|---|
| previous model | 0.409 | 0.027 | 0.409 |
| fine-tuned (beta 32, gamma 4) | **0.447** | **0.873** | 0.420 |

Refinements reach 0.900 on the eval-n 500 split once the strict-tree rule below is active.

Decision rules in `turn()`:

* **Context:** the widget model sees the last two widgets shown.
* **Strict trees:** a non-root turn of a widget conversation can only follow its own parent, since it changes *that* widget.
* **Gate:** the decision is gated on the total non-"plain" probability, then the best widget is shown. At 0.995, 93.2% of widget requests get a widget, with 67% of the right type, and 10.0% of ordinary chat messages get one.
* **Refinements:** a widget that refines one on screen is the whole answer. The chat model is not asked.
* **Opener preference:** widgets that open a refinable conversation get a +1 bonus. On the eval this is free (single-turn 0.5125 → 0.5161, everything else unchanged).

fp16 packaging matches float32 on 99.75% of 4,069 held-out requests.
