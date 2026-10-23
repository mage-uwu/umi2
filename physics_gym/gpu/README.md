# GPU trainer for the nanotss body controllers

| file | what |
|---|---|
| `nanotss_torch.py` | the nanotss byte stack (`nanotss_blt.c`) in PyTorch: `.tss` load/save, forward, hand-ported backward, `tss_train_step`'s update |
| `ppo_gpu.py` | `ppo_tss.c` with the controller on the GPU and the physics on CPU cores; same flags, same `.tss` / `.head` files |
| `tssenv.c` → `libtssenv.so` | `env.c` + XPBD physics as a shared library (8 worlds per AVX-512 group, OpenMP over groups) |
| `parity.py` + `tss_parity.c` | runs the C training path and the torch one on the same batch and compares them |

## Compatibility with the C version

* **Checkpoints**: identical format. GPU-trained `.tss`/`.head` files run in `tss_run`, `ppo_tss --load` and the viewer, and C-trained ones continue here with `--load`.
* **Forward**: the ternary weights are rebuilt with the C rounding and the C order for summing each group's scale. Int8 codes times ternary signs are exact integer dot products (TF32 is off), and the transcendental functions are evaluated in double, as libm does it.
* **Backward**: a port of `layer_bwd` and `fff_bwd`, with no autograd inside the stack. It includes:
  * straight-through gradients;
  * the sampled routing credit, with the same xorshift stream per sequence (`tss_work_seed(step*1000003+b)`), so the same alternate walks are drawn;
  * reverse-time state adjoints, with the bf16 state tape of the AVX-512 build.
* **Update**: the same global clip. Adam runs on the dense fields and only on visited rows of the node fields, and blockwise Muon (16-row blocks of visited rows, 5 Newton-Schulz steps) runs on `w_in`, `w_out`, `proj` and `U`.

`python3 parity.py init:3` (fresh model) and `python3 parity.py some.tss --L 64` (trained get-up policy, 8 sequences, some starting from a saved state, uneven lengths) give these results on CPU:

| quantity | max relative difference vs C |
|---|---|
| init (`params_init`, same seed) | 0 (bit-identical) |
| forward output | 2e-7 |
| input gradient | 8e-6 |
| parameter gradients | ≤ 1.5e-5 |
| visited rows | identical |
| parameters after 3 Adam+Muon steps | ≤ 3e-4 (Adam turns tiny gradients into full steps, so it amplifies 1e-6 differences on near-zero gradients) |

The remaining differences come from float summation order. Streaming inference (one step at a time, as the rollouts run) is bit-identical to the sequence forward, and the rollout-vs-training log-prob gap (`lp_gap`) is about 3e-7, the same as in C.

Not ported, and loading such a checkpoint raises: fast-weight layers, `ssm_ltc`, rotary B/C, `n_loops > 1`, NorMuon. The body controllers use none of these.

## Running

```sh
cd body/gpu
gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c ../room.c -lm
python3 parity.py init:3 --device cuda          # check the GPU build against C first
python3 ppo_gpu.py --task getup --style ../data/cmu_ref.bin:getup_back1,getup_back2,getup_face1,getup_face2,getup_side1,getup_side2 \
    --envs 2048 --horizon 64 --seconds 3600 --out getup_gpu
```

The flags are `ppo_tss.c`'s flags, plus `--device`, `--gamma`, `--lam` and `--ent`.

## Scaling

The stack's walk is sequential over time and tree level, so one training pass is roughly a fixed number of small kernels (about 64 × 5 × 2 layers, forward and backward). Its cost barely depends on batch size until the GPU fills. That's the point: run thousands of environments and train each minibatch as one pass, rather than 64 environments on 8 threads as the C trainer does.

On CPU, PyTorch is slower than the C trainer (about 160 vs 3000 steps/s at 16 environments). The GPU speed has not been measured yet.
