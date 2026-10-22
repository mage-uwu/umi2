# ssm_tree.c

A single-file, dependency-free byte-level language model built only from **SSM-tree blocks**, with its trainer
(forward, hand-written backward, AdamW or blockwise Muon, OpenMP batch parallelism), evaluation (val BPB),
sampling and checkpoint save/load. About 600 lines of plain C, float32.

It is the default SSM layer of `nanotss_blt.c` (branch `icl-ttt`): routed binary trees whose nodes each hold a
selective-decay SSM state, per-tree-group keys/queries, QK-norm and rotary on B/C. The model is configured with
`--layers 4 --trees 4 --depth 7 --state-n 16 --state-p 16 --width 128`, which gives about 1.17M parameters.

Deliberately left out, compared with the parent file:
- the ternary weights, int8 activations and AVX-512 kernels (so it is several times slower);
- the FFF blocks (every layer here is an SSM-tree block);
- the sampled routing-credit gradient for unvisited branches (routing is hard; the router learns only through gelu(s));
- TTT, match model, BLT mode and the other tools.

The header comment of `ssm_tree.c` gives the block equations.

```sh
cc -O3 -march=native -fopenmp ssm_tree.c -lm -o ssm_tree
./ssm_tree --data input.txt --seconds 300 --threads 4 --save m.bin --sample 400
./ssm_tree --load m.bin --prompt "ROMEO:" --sample 400 --temperature 0.8

# finite-difference gradient check (double precision)
cc -O2 -DDOUBLE ssm_tree.c -lm -o ssm_tree_dbl && ./ssm_tree_dbl --check
```

Checks run so far: the gradient check gives a worst relative error of 6e-6 over 399 random parameters. A
90-second TinyShakespeare run reaches val 3.08 BPB at about 6.6K bytes/s on 4 shared threads.
