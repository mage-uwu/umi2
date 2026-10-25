"""scan_parity.py: the fused SSM-tree scan (tss_scan.cu) against the PyTorch loop it replaces, on a trained policy.

The same training forward and backward both ways (recurrent state, ragged lengths, routing-credit draws): the
outputs, every tape field, the input gradient, every parameter gradient and the carried state, as the largest
difference and the share of routing decisions that differ; then the time of one forward and backward each way.

    python3 scan_parity.py [policy.tss] [B] [L]
"""
import sys, os, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nanotss_torch as nt

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/clean/warm/jump_stick.tss")
B = int(sys.argv[2]) if len(sys.argv) > 2 else 256; L = int(sys.argv[3]) if len(sys.argv) > 3 else 64
dev = torch.device("cuda")
m = nt.TSS.load(path, device=dev); c = m.c
assert nt.scan_lib() is not None, "tss_scan.cu did not build"
gen = torch.Generator(device=dev); gen.manual_seed(5)
x = torch.randn(B, L, c.d, device=dev, generator=gen)
st0 = m.zero_state(B)
for v in st0["h"]: v.copy_(0.05 * torch.randn(v.shape, device=dev, generator=gen))
lens = torch.randint(L // 3, L + 1, (B,), device=dev, generator=gen); lens[0] = L
g = torch.randn(B, L, c.d, device=dev, generator=gen)
seeds = [1000003 * 7 + b for b in range(B)]


def run(fused):
    m.fused = fused; m.zero_grad()
    st = {k: [v.clone() for v in vs] for k, vs in st0.items()}
    torch.cuda.synchronize(); t0 = time.time()
    out, tape = m.forward(x, state=st, lens=lens, train=True, seeds=seeds)
    dx = m.backward(tape, g)
    torch.cuda.synchronize(); el = time.time() - t0
    return out, tape, dx, {f: m.G[f].clone() for f in nt.FIELDS}, st, el, m.hit.clone()


A = run(False); Fz = run(True)
valid = A[1]["valid"]
def diff(a, b, mask=None):
    a, b = a.float(), b.float()
    d = (a - b).abs()
    if mask is not None: d = d[mask]
    return float(d.max()) if d.numel() else 0.0, float(a.abs().max()) if a.numel() else 0.0
print(f"{path}: B {B}, L {L}, trees {c.T}, depth {c.D}, n {c.n}, p {c.p}")
print(f"output: max diff {diff(A[0], Fz[0])[0]:.2e} (of values to {diff(A[0], Fz[0])[1]:.2f})")
print(f"input gradient: max diff {diff(A[2], Fz[2])[0]:.2e} (of {diff(A[2], Fz[2])[1]:.2e})")
for l, (ta, tf) in enumerate(zip(A[1]["layers"], Fz[1]["layers"])):
    if ta.get("kind") != "ssm": continue
    vm = valid[:, :, None, None]
    flips = float(((ta["node"] != tf["node"]) & vm).float().sum() / (vm.float().sum() * ta["node"].shape[2] * ta["node"].shape[3]))
    line = [f"layer {l}: path rows differing {flips:.2e}"]
    for k in ("s", "dec", "R", "rs", "hprev", "as_", "aR", "bw"):
        if k in ta and ta[k] is not None: line.append(f"{k} {diff(ta[k], tf[k])[0]:.1e}")
    print("  " + ", ".join(line))
for f in nt.FIELDS:
    dmax, scale = diff(A[3][f], Fz[3][f])
    if scale > 0: print(f"  grad {f:8s} max diff {dmax:.2e} of {scale:.2e} ({dmax / scale:.1e} relative)")
print(f"carried state: max diff {max(diff(a, b)[0] for a, b in zip(A[4]['h'], Fz[4]['h'])):.2e}; visited rows equal: {bool((A[6] == Fz[6]).all())}")
ts = [run(True)[5] for _ in range(3)]; tl = [run(False)[5] for _ in range(2)]
print(f"forward + backward: PyTorch loop {min(tl):.3f} s, fused {min(ts):.3f} s ({min(tl) / min(ts):.1f}x)")
# inference as the rollouts run it: one step at a time with the state carried (stream=True, no tape)
def stream(fused, steps=16):
    m.fused = fused; st = {k: [v.clone() for v in vs] for k, vs in st0.items()}; outs = []
    with torch.no_grad():
        for t in range(steps): outs.append(m.forward(x[:, t:t + 1], state=st, stream=True)[:, 0])
    return torch.stack(outs, 1), st
oa, sa = stream(False); of, sf = stream(True)
print(f"stream inference, 16 steps: output max diff {diff(oa, of)[0]:.2e}, state max diff {max(diff(a, b)[0] for a, b in zip(sa['h'], sf['h'])):.2e}")
# where the fused forward + backward spends its time: the scan kernels against the PyTorch ops around them
if os.environ.get("SCAN_PROF"):
    run(True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof: run(True)
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))
