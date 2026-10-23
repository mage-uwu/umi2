"""bench.py: time the GPU trainer's pieces at the body controller's size (d 64, 4 trees, depth 4).

    python3 bench.py [--device cuda]
"""
import argparse, json, sys, os, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nanotss_torch as nt

ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda"); ap.add_argument("--L", type=int, default=64)
a = ap.parse_args(); dev = torch.device(a.device)
sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
m = nt.TSS(nt.Cfg(d=64, T=4, D=4, groups=4), device=dev, seed=1); opt = nt.Optimizer(m); d = m.c.d
for B in (64, 256, 1024, 4096):
    x = torch.randn(B, a.L, d, device=dev); w = torch.randn(B, a.L, d, device=dev)
    def once(step):
        xo, tape = m.forward(x, state=m.zero_state(B), train=True, seeds=[step * 1000003 + b for b in range(B)])
        u, r = nt.rmsnorm(xo, m.P["norm_f"]); g, dw = nt.rmsnorm_bwd(xo, m.P["norm_f"], r, w); m.G["norm_f"] += dw
        m.backward(tape, g); opt.step(B * a.L)
    once(1); sync(); t0 = time.time(); n = 3
    for s in range(n): once(s + 2)
    sync(); dt = (time.time() - t0) / n
    mem = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0
    print(json.dumps({"bench": "train_step", "B": B, "L": a.L, "seconds": round(dt, 3), "positions_per_s": round(B * a.L / dt), "max_mem_gb": round(mem, 2)}), flush=True)
for B in (256, 1024, 4096):
    st = m.zero_state(B); x = torch.randn(B, 1, d, device=dev)
    with torch.no_grad():
        m.forward(x, state=st, stream=True); sync(); t0 = time.time()
        for _ in range(20): m.forward(x, state=st, stream=True)
        sync()
    dt = (time.time() - t0) / 20
    print(json.dumps({"bench": "policy_step", "envs": B, "ms": round(dt * 1000, 2)}), flush=True)
