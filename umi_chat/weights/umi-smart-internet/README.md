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
