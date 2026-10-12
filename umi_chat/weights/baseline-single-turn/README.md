# umi-1M-chat: single-turn baseline

These weights come from 2 hours of training (105,396 steps) on the RunPod CPU pod "UMI" (8 vCPU, AMD Threadripper 7960X).
The model is the original single-turn version, with 2-byte answer IDs over 2,608 DeepSeek intents and no n-gram cache.

* Test (8,076 held-out phrasings): intent accuracy **0.4578**, domain accuracy **0.7494**.
* `train.log` has the full run log.
* It needs the code and data of commit `74cfeb1`, because its ID layout differs from the current multi-turn version:

```
git checkout 74cfeb1 -- umi_chat
cc -O3 -march=native -fopenmp umi_chat/umi_chat.c -lm -o umi_chat_v1
./umi_chat_v1 chat --data umi_chat/data --load umi_chat/weights/baseline-single-turn/umi-1M-chat.bin -v
```
