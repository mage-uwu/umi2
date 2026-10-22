# Context as test-time compute for nanotss (AR)

Branch `icl-ttt`, built on the autoregressive `nanotss_blt.c`. Everything runs on CPU: a 4-thread
container for local work, plus an 8-vCPU RunPod pod (cpu5c, $0.28/hr) for longer training.

## Bottom line

1. **The recurrent state does not turn long context into better predictions on real text.** A model
   trained on 256-byte windows drifts when streamed, and its BPC blows up after a few KB (the
   global SSM tree is the cause). With the drift fixed, streaming is stable but no better than a
   64-byte window.
2. **Test-time training (TTT) on the context does.** Each chunk of the context is scored, then
   the weights take a few optimizer steps on the newest window plus replay of earlier context.
   The scratchpad ends up stored in the weights, and context length becomes a compute knob: about
   −0.2 BPC for each 4× more context, still improving at 100 KB. One TTT epoch costs about 3 s per
   100 KB on 4 threads, so a 1 MB scratchpad is about 30 s per epoch.
3. **The 1.4M model with TTT beats a 3× larger 4.2M model without TTT** on every text tested,
   and on TinyShakespeare validation it even edges out the 4.2M model with TTT.
4. **Iterative refinement works when drafts decide what to study, not what to imitate.**
   Self-training on the model's own drafts made text worse. Using each draft to *retrieve* the
   most similar context passages and training on those (real text) gives the best context-faithful
   generations, at no loss of fluency versus plain best-of-K.

## Models and data

| Model | Params (budgeted) | Training | TinyShakespeare val BPC |
|---|---|---|---|
| `base300` | 1.29M (1.43M total at inference) | 300 s, 4 threads | 2.546 |
| `base2h` | same | 7200 s, 8 vCPU | 2.514 |
| `big600` (width 192) | 4.19M (4.53M total) | 600 s, 8 vCPU (best of a 300/600/1200/7200 s sweep) | 2.413 |
| `big2h` (width 192) | same | 7200 s, 8 vCPU | 3.132 (badly overfit: train 1.28) |

All models train on TinyShakespeare: the first 90% for training, the last 10% (111 KB) for
validation. Unseen test texts are three public-domain plays in `data/`: Marlowe's *Doctor
Faustus* and *Tamburlaine Part 2*, and Kyd's *The Spanish Tragedie* (307 KB in all,
normalized to the `NAME:` format). BPC is bits per byte, windowed with 64 bytes of left context,
unless stated otherwise.

## 1. Baseline ICL: the recurrent state

`--ttt-eval FILE --ttt-stream 1` streams the whole document through the recurrent state with
no resets.

| base300, TinyShakespeare val | BPC |
|---|---|
| windowed (64-byte context) | 2.548 |
| streamed, whole 111 KB | **5.920** (2.5 up to 2 KB, 3.7 at 4 KB, 5.8 at 16 KB, 6.7 at 64 KB) |
| streamed, reset entropy / encoder / decoder state every 1 KB | 5.91 / 5.94 / 5.90 |
| streamed, reset **global stack** state every 1 KB | 2.561 |
| streamed, reset global plastic-memory layer only | 4.97 |
| streamed, reset global SSM layer only | 2.60 |
| streamed with `--ssm-horizon 64` (inference cap on SSM retention) | 2.623 |

The global SSM tree's slow modes keep accumulating past the 64-patch horizon they were trained on.
The plastic memory is comparatively stable. `base2h` streams stably (2.532 vs 2.515 windowed),
so longer training fixes the drift, but streaming still adds nothing over the window.

## 2. Online TTT on the context

The context is scored chunk by chunk (192 bytes), and after each chunk TTT takes `--ttt-steps`
AdamW steps on 4 windows: the newest plus uniform replay of everything already read. No byte is
trained on before it has been scored.

| TinyShakespeare val | base300 | base2h | big600 |
|---|---|---|---|
| no TTT | 2.548 | 2.515 | 2.420 |
| **online TTT** (lr 2e-4, 2 steps per chunk) | 2.375 | **2.221** (−11.7%) | 2.231 |
| control: the same steps on random *training-set* windows | 2.562 | 2.518 | |

The control rules out "just more training": all of the gain comes from reading the context.
Sweep on base300 (lr 3e-4 unless noted): BPC stays between 2.375 and 2.40 for lr 1e-4 to 5e-4
(lr 3e-3 diverges to 2.60), batch 1 to 8 and chunk 64 or 192. Freezing the ternary matrices
gives 2.395, essentially the same. Muon is worse (2.475 vs 2.447 for AdamW at lr 1e-3).
Anchoring toward the loaded weights (decay 0.01) gives 2.380. Retrieval-chosen replay barely
moves online BPC (2.684 → 2.677 on *The Spanish Tragedie*).

## 3. Context → held-out query (`--ttt-fit`)

TTT runs for E epochs on the first part of each unseen play, then the rest is scored with the
weights frozen. Means over the three plays:

| | no TTT | 8 epochs |
|---|---|---|
| base2h, first half → second half | 3.352 | **2.498** (−25%) |
| big600, first half → second half | 3.167 | 2.479 |

### Test-time compute scaling: context length (base2h, query = last 20% of each play)

| context TTT reads | 0 | 2 KB | 8 KB | 32 KB | full (57–103 KB) |
|---|---|---|---|---|---|
| query BPC | 3.365 | 3.040 | 2.803 | 2.520 | **2.355** (−30%) |

For comparison, big600 scores 3.170 here without TTT and 2.331 with it on the full context.

### Where TTT stores the context (base300, mean gain on the plays, 8 epochs)

| TTT may change | all | byte embedding + hashed n-grams | decoder | hashed n-grams | encoder | dense attention/mixer | global stack |
|---|---|---|---|---|---|---|---|
| BPC gain | 0.552 | **0.427** | 0.367 | 0.310 | 0.300 | 0.277 | 0.122 |

The free hashed n-gram table (98K parameters) and the byte embedding carry 77% of the gain. They
act as a learned, context-adaptive n-gram cache. A 16× larger table (4096 bins) did not help:
it was worse before TTT and equal after (2.602 vs 2.598). Tripling the lexical step size was
slightly worse too.

## 4. Generation: iterative refinement

`--refine` reads the context with TTT, then runs rounds of K drafts. Scoring uses the adapted
model's mean NLL plus a repetition penalty. Two loop variants were tested:

* **Self-training** (`--refine-retr 0`): TTT steps on windows ending in the kept drafts, mixed
  with context replay. **This made text worse** (word validity 0.44 vs 0.60 on base300): the
  model reinforces its own errors.
* **Retrieval-guided** (`--refine-retr 1`): each kept draft queries a BM25-style word index of
  the context. TTT then trains only on the retrieved real passages, plus uniform replay, before
  the next round. The final round is best-of-K. Optional local edits (`--refine-edits`)
  resample high-surprisal spans and keep an edit only if the whole-draft score improves.

Evaluation uses 400-byte responses after `"\n\n<LEAD>:\n"`, with the first half of each play as
context: 3 plays × 4 seeds = 12 samples per method. The metrics never use the generator's own
likelihood. **wvr** is the share of words found in TinyShakespeare-train plus the context;
**bigram** is the share of word pairs seen there; **ctxbigram** is the share of word pairs that
occur in the context; **ctxnames** is the share of speaker headings that are speakers in the
context.

| base2h | wvr | bigram | ctxbigram | ctxnames | distinct-2 |
|---|---|---|---|---|---|
| plain, T=1 | 0.741 | 0.344 | 0.108 | 0 | 0.999 |
| plain, T=0.7 | **0.897** | **0.575** | 0.214 | 0 | 0.985 |
| TTT (8 epochs), T=0.7 | 0.834 | 0.462 | 0.262 | 0.47 | 0.987 |
| TTT + best-of-8, T=0.7 | 0.848 | 0.469 | 0.279 | **0.66** | 0.991 |
| **TTT + 3 retrieval rounds + best-of-8**, T=0.7 | 0.849 | 0.536 | **0.345** | 0.63 | 0.990 |
| same, 5 rounds, 4 epochs | 0.843 | 0.514 | 0.305 | 0.64 | 0.988 |
| same, 3 rounds, 4 epochs / 2 epochs / lr 1e-4 | 0.846 / 0.837 / 0.844 | 0.499 / 0.466 / 0.480 | 0.281 / 0.240 / 0.236 | 0.60 / 0.30 / 0.39 | |

Plain sampling never writes in the context's world (0% context names): it falls back to
TinyShakespeare characters. TTT + retrieval refinement uses 61% more context word pairs than
plain sampling. It does lose some general fluency (word validity 0.85 vs 0.90), and weaker TTT
does not win that back. On base300 the same comparison favored retrieval refinement on every
metric (0.784 / 0.427 / 0.247 / 0.57 vs plain T=0.7 at 0.786 / 0.416 / 0.154 / 0).

**Judge model.** `big600` (4.2M parameters, never the generator) scores each method's 12 samples. Lower BPC is better.
It scores them twice: as trained (*plain judge*), and after reading the same context for 8 TTT epochs (*context judge*).

| base2h samples | plain judge | context judge |
|---|---|---|
| **retrieval refine, 4 epochs, 3 rounds** | 2.725 | **2.194** |
| retrieval refine, 8 epochs, 3 rounds | 2.703 | 2.213 |
| retrieval refine, 4 epochs, 5 rounds | 2.673 | 2.228 |
| TTT + best-of-8 | 2.732 | 2.268 |
| retrieval refine, 4 epochs, lr 1e-4 | 2.603 | 2.287 |
| plain, T=0.7 | **2.136** | 2.293 |
| TTT, T=0.7 | 2.810 | 2.374 |
| retrieval refine, 2 epochs | 2.826 | 2.376 |
| plain, T=1 | 2.826 | 2.907 |

The context-aware judge ranks retrieval refinement first, at 0.10 BPC better than plain T=0.7 and
0.18 better than single-shot TTT. The plain judge prefers plain sampling, because plain samples
are ordinary TinyShakespeare text, which is exactly what that judge was trained on.

Samples (base2h, *Faustus* context, seed 1, prompt `FAUSTUS:`):

> **plain T=0.7:** *ESCALUS: Go to the busir to yet approve, but yet seisiat it will come in a
> corseen not tountenance shall know the corruption of the hamp...*
>
> **TTT + retrieval refine:** *MEPHIST: Tut, tell me of a neck, and there is not. ... MAPDIA:
> What art thou, man? FAUSTUS: Why, this is tell--fits, and do not so arch the prince for: This
> is well. ... MEPHIST: I pray thee hell. ... FAUSTUS: Well, I warrant.*

## 5. Native long-range memory: can the recurrent state learn it?

Recall test (`experiments/recall_test.py`): a 256-byte unseen passage, then D bytes of other unseen
text, then the same passage again, streamed with no TTT. The gain is the drop in BPC on the second
copy. Perfect recall would be about 2.4 BPC.

| Model (30 min, 8 vCPU unless noted) | Recall gain, gap 0 / 1K / 4K / 16K / 64K |
|---|---|
| BLT, D=7, independent windows | 0.05 / 0.08 / 0.07 / 0.07 / 0.06 |
| BLT, D=7, carried state (+ 25% / 50% repeats) | about 0 |
| BLT, global D=3, carried state + 25% repeats (0.29M and 1.23M) | about 0 |
| byte mode, D=3, carried + 25% repeats, 171K | 0.035 / 0.039 / 0.047 / 0.051 / 0.047 |
| byte mode, D=3, carried + 25% repeats, 512K | 0.033 / 0.049 / 0.059 / 0.021 / 0.014 |
| byte mode, D=7, carried + 25% repeats, 1.16M | 0.018 / 0.017 / 0.007 / 0.024 / 0.012 |
| *TTT on base2h (reference)* | *0.66 / 1.40 / 1.46 / 1.32 / 0.97* |

`--carry-windows N` carries the plastic-memory/SSM state and conv history across N consecutive
windows (truncated BPTT). `--repeat-p` re-copies earlier stream pieces into the training data.

**Tree depth is what blocks learnable memory.** Pure copy task (random 32-letter string, then an
exact copy; no memory ≈ 4.56 BPC, perfect ≈ 2.3), byte mode, 10 minutes:

| depth | 0 | 0 with n=64 | **3** | 4 | 5 | 7 |
|---|---|---|---|---|---|---|
| val BPC | 4.20 | 2.37 | **2.35** | 4.38 | 4.45 | 4.36 |

There's a sharp cliff between depth 3 and depth 4. With 255 nodes per tree, writes and later
reads seldom meet at the same node, so the routing has to learn to align them from sparse
gradients. Even a model that can copy, though, does not carry exact bytes across kilobytes. The
fast memory (about 1,000 key slots at depth 3) is far too small, and through BLT the global
memory only sees patch averages. **Conclusion: the context memory has to be TTT.**

Depth also trades against speed. Streaming speed (single thread): D=7 width 64 (1.29M params)
29.5K bytes/s; D=3 width 64 (0.29M) 35.5K bytes/s; D=3 width 192 (1.23M) 13.2K bytes/s. At equal
parameter count the deep tree is 2.2× faster, because each step evaluates only D+1 nodes.

## 6. Final design: byte-mode deep-tree LM + TTT memory

Byte mode (no BLT), depth 7, 1.16M parameters (`checkpoints/byte1200_d7_1p2M.bin`), trained for
1200 s on 8 vCPUs (best of 600 / 1200 / 2400 s: val 2.272 / **2.222** / 2.226).

| | BLT 1.4M (base2h) | BLT 4.2M (big600) | **byte 1.16M** |
|---|---|---|---|
| TinyShakespeare val, no TTT | 2.515 | 2.420 | **2.231** |
| val, online TTT (control on training text) | 2.221 (2.518) | 2.231 | **2.077** (2.242) |
| Unseen plays, half → half: no TTT → TTT | 3.352 → 2.498 | 3.167 → 2.477 | **2.978 → 2.275** |
| Plays, last 20%, context 0 / 2K / 8K / 32K / full | 3.365 / 3.040 / 2.803 / 2.520 / 2.355 | 3.170 / – / – / – / 2.331 | **2.970 / 2.709 / 2.492 / 2.271 / 2.141** |
| Recall with TTT, gap 0 / 1K / 4K / 16K / 64K | 0.66 / 1.40 / 1.46 / 1.32 / 0.97 | | 0.58 / 1.19 / 1.13 / 0.92 / 0.55 |
| Recall, native state only | ≈ 0 | | −0.02 to −0.10 |

The byte-mode deep-tree model is the best language model measured here, and it gains the most
from TTT on unseen text. With TTT it beats the 4.2M BLT model on every BPC test. Context still
scales like a compute knob: about −0.2 BPC per 4× more context, still falling at the full play
length. Its native state recall is slightly negative at long gaps, so for this model the
scratchpad has to live in the weights. Iterative refinement (`--refine`) still needs a BLT
checkpoint; porting it to byte mode is the remaining step.

## What did not work

* Recurrent streaming as long-context memory (section 1).
* Self-training on drafts, including with likelihood-guided edits.
* A larger hashed n-gram table, a lexical step multiplier, and Muon for TTT.
* Training 4.2M parameters for 2 hours on 1 MB of text (val 3.13): budget matters more than size here.
* Native long-range memory through carried state and repeat augmentation, at any tree depth (section 5).

## Commands

```
cc -O3 -march=native -fopenmp nanotss_blt.c -lm -o nanotss
# online TTT over a document (scores before training on it)
./nanotss --load model.bin --ttt-eval doc.txt --ttt-offset 0 --ttt-lr 2e-4 --ttt-steps 2
# read the first half for 8 epochs, score the rest
./nanotss --load model.bin --ttt-fit a.txt,b.txt --ttt-frac 0.5 --ttt-epochs 8 --ttt-lr 3e-4
# generate with the scratchpad: TTT, then 3 retrieval-guided rounds, best-of-8
./nanotss --load model.bin --refine 1 --refine-context ctx.txt --prompt $'\n\nFAUSTUS:\n' \
  --refine-epochs 8 --ttt-lr 3e-4 --refine-rounds 3 --refine-cands 8 --refine-keep 2 \
  --refine-retr 1 --refine-steps 8 --refine-temp 0.7 --refine-final-temp 0.7
# stable recurrent streaming
./nanotss --load model.bin --context-file ctx.txt --ssm-horizon 64 --sample 300
```

Harnesses: `experiments/compare_gen.sh` (generation methods), `experiments/genmetrics.py`,
`experiments/judge.sh`, `experiments/pod_run.sh` (batch pod runner). Checkpoints are in
`checkpoints/`.

## Next steps

* Train for TTT: short inner-loop TTT during pretraining so that the slow weights learn to be
  quick to adapt (meta-learning). This is the most likely route to larger gains.
* Port `--refine` (retrieval-guided refinement) to byte-mode checkpoints, then rerun the generation comparison.
* A much larger fast memory (wider fast matrices or a dedicated delta-rule layer) if native recall is still wanted; section 5 shows that depth and carried state alone are not enough.
* An LLM judge (DeepSeek via OpenRouter, blocked this session because no key is attached) and
  DeepSeek-written synthetic plays, for 1 MB scratchpad tests with controlled facts and names.

## 7. `--session`: one tool that reads, adapts online and generates

```
./nanotss --load checkpoints/byte1200_d7_1p2M.bin --session 1 [--session-speed fast|balanced|quality] [--session-score 0]
```

Stdin is the session. Plain lines are context: they are learned online one 192-byte chunk at a time
(scored first, then a TTT step) and added to the match model. Commands:
`@@gen N [temp] [cands]` (generate N bytes, best of `cands` drafts, match-assisted), `@@file PATH`,
`@@keep 0|1` (append generations to the context), `@@stats`, `@@reset`.

TTT acceleration (TinyShakespeare val, 111 KB, 4 threads; match model on unless noted):

| setting | val BPC | unseen play BPC | throughput |
|---|---|---|---|
| original: 2 steps / 256-byte windows, no match | 2.078 | – | 2.3 KB/s |
| quality preset (2 steps) | 2.060 | 2.290 | 2.4 KB/s |
| **balanced preset (1 step), default** | **2.053** | 2.318 | **4.1 KB/s** |
| fast preset (1 step / 128-byte windows) | 2.065 | 2.354 | 10 KB/s without scoring |

With the match model on, one TTT step per chunk is as good as two on familiar text: the match model
covers the exact repeats that the second step was learning. On unseen text the second step is still
worth about 0.03 BPC. Surprise-adaptive step counts (`--ttt-adaptive`) and truncated backward
(`--ttt-bwd-layers`) land on the same quality/cost curve rather than beating it.

## 8. Default architecture: pure SSM trees with per-tree keys/queries

`./nanotss --data input.txt --save model.bin` now trains, by default: byte mode, layers
SSM tree → gated FFF tree → SSM tree → gated FFF tree (no plastic layer), with the SSM key B and
query C computed **per tree** (`--ssm-groups 4`, a GQA analogue: previously one B/C was shared by
all four trees). 1.19M budgeted parameters. `--mode blt` restores the BLT stack and its defaults.

Evidence (byte mode, 4 threads):

| change | equal time (5 min) | equal steps |
|---|---|---|
| plastic layer → SSM (`--fw-mask 0`) | 2.389 → 2.361 | – |
| shared → per-tree SSM keys/queries | 2.379 → 2.373 | 2.416 → 2.380 (2,600 steps) |
| LTC update (input gate / state gate) | 2.361 → 2.381 / 2.484 | state gate: 2.462 → 2.489 (1,800 steps) |

Checkpoints written with a non-default group count or LTC mode use magic `TFX1`, which stores
`{ssm_groups, ssm_ltc}` after the config; older checkpoints load as groups 1, no LTC.
BLT checkpoints always use shared projections.

### Default update: Muon + SSM QK-norm + rotary

New byte models now also default to blockwise **Muon** for training (`--optimizer adamw` to opt
out; TTT always uses AdamW), **L2-normalized** SSM keys/queries (`--ssm-qknorm 1`) and **rotary**
SSM keys/queries (`--ssm-rope 1`), both stored in the TFX1 header. Checkpoint:
`checkpoints/muon_qk_rope1200_1p19M.bin`.

Smoke tests (byte mode, 4 threads, two seeds): Muon vs AdamW at 5 min 2.356 vs 2.382 (seed spread
0.005 vs 0.067); on Muon at 2,500 steps, QK-norm −0.019, rotary −0.022, both −0.031 (two batches),
structural position codes −0.017 alone but nothing on top of rotary.

20 minutes on 8 vCPU:

| BPC | previous default (AdamW) | Muon + QK-norm + rotary |
|---|---|---|
| val, no TTT | 2.215 | **2.206** |
| val, TTT 2 steps | 2.075 | **2.054** |
| val, session default (TTT 1 step + match) | 2.051 | **2.036** |
| unseen plays, half → half, no TTT / TTT | 3.005 / 2.266 | **2.967 / 2.259** |
| same with match | 2.477 / 2.161 | **2.469 / 2.154** |
| plays, full context → last 20%, TTT (with match) | **2.114 (2.048)** | 2.130 (2.053) |

The run completed 22% fewer steps (17,700 vs 22,600), and TTT on random training text still
helped it (2.211 → 2.201), so it is likely undertrained at 20 minutes.

### Universal-transformer looping (`--loops R`, `--layers N`): not adopted

The layer stack is applied R times per byte with shared weights (own state per pass); the
whole-model gradient is exact for R = 1–4 (`--check-loops`). Byte default, 4 threads, seed 0:

| layers × loops | params | val BPC, 5 min (steps) | val BPC, 1,300 steps (time) |
|---|---|---|---|
| 4 × 1 | 1.19M | **2.372** (2,517) | 2.456 (149 s) |
| 4 × 2 | 1.19M | 2.458 (1,365) | 2.442 (266 s) |
| 4 × 4 | 1.19M | 2.557 (769) | 2.433 (500 s) |
| 2 × 1 | 0.61M | – | 2.577 (73 s) |
| 2 × 2 | 0.61M | 2.417 (2,794) | 2.539 (133 s) |
| 2 × 4 | 0.61M | 2.536 (1,402) | 2.533 (252 s) |

Looping helps per step (−0.014 to −0.023 for 4 layers, −0.038 to −0.044 for 2 layers) but each extra
pass costs about 1.8× compute, so it loses at any fixed budget. It stays off by default; it is an
option for very small models (2 × 2 at 0.61M is within 0.045 of the 1.19M model at equal time).
Adaptive per-byte looping would be the version worth revisiting.
