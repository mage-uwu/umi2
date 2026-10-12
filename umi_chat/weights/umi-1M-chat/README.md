# umi-1M-chat: multi-turn model

This is a 3-hour run (201,639 steps) on the RunPod CPU pod "UMI" (8 vCPU, Threadripper 7960X), trained on the data at commit `0af3f3b` and later.
The model has 1,287,904 core parameters, in BLT mode with a hashed n-gram cache.

## Accuracy on held-out phrasings (the right answer out of 17,984 cached answers)

| retrieval | single-turn | follow-up | topic switch |
|---|---|---|---|
| MLM beam only, full test set (n=10,785 / 14,085) | 0.374 | 0.821 | – |
| n-gram cache only (eval-n 600 half) | 0.487 | 0.090 | – |
| hybrid as first shipped (beta 32, gamma 16) | 0.665 | 0.873 | ~0.20 |
| **hybrid, two-hypothesis scoring (beta 32, gamma 2)** | **0.678** | **0.823** | **0.644** |

"Topic switch" means a single-turn question asked right after a random cached conversation turn.
The first hybrid rule never left a conversation once it entered one. The current rule scores
turns that follow the context with the context, and scores everything else as a fresh message.
`eval.txt` is the pod's original eval; `eval_two_hypothesis.txt` is the re-tune.

```
./umi_chat chat --data data --load weights/umi-1M-chat/umi-1M-chat.bin --beta 32 --gamma 2 -v
```
