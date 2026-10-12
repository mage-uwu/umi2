# umi-smart-internet: widget model

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
