"""parity.py: the GPU trainer against nanotss_blt.c on one batch.

Builds tss_parity (the C training path with a test loss), feeds both the same sequences (some from a saved
recurrent state, some from zero, uneven lengths), and compares forward outputs, input gradients, every
parameter gradient, the visited rows and the parameters after a few tss_train_step updates.

    python3 parity.py [model.tss | init:SEED] [--device cuda] [--B 8] [--L 24] [--steps 3]
"""
import argparse, os, subprocess, sys, tempfile
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nanotss_torch as nt


def build(exe):
    src = os.path.join(HERE, "tss_parity.c")
    if not os.path.exists(exe) or os.path.getmtime(exe) < max(os.path.getmtime(src), os.path.getmtime(os.path.join(HERE, "../../nanotss_blt.c"))):
        subprocess.check_call(["gcc", "-O3", os.environ.get("PARITY_MARCH", "-march=native"), "-fopenmp", "-ffp-contract=off", "-o", exe, src, "-lm"])


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="init:3")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--B", type=int, default=8); ap.add_argument("--L", type=int, default=24)
    ap.add_argument("--steps", type=int, default=3); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tape", choices=("auto", "bf16", "fp32"), default="auto",
                    help="state tape precision on the torch side; auto = what the C build here uses (bf16 only with AVX-512)")
    a = ap.parse_args()
    exe = os.path.join(HERE, "tss_parity"); build(exe)
    tmp = tempfile.mkdtemp()
    init_path = os.path.join(tmp, "init.tss")
    # the model: C writes it (from its own seeded init, or a copy of the checkpoint), both sides load it
    Bn, L = a.B, a.L
    rng = np.random.default_rng(a.seed)
    if a.model.startswith("init:"):
        seed = int(a.model[5:])
        c0 = nt.Cfg(d=64, n_layer=4, T=4, D=4, n=16, p=16, K=3, groups=4, qknorm=1)
        P0 = nt.params_init(c0, (seed * 7919 + 88172645463325252) & 0xFFFFFFFFFFFFFFFF)
    m0 = nt.TSS.load(a.model, device="cpu") if not a.model.startswith("init:") else nt.TSS(c0, P0, device="cpu")
    c = m0.c; d = c.d
    S_all = (c.n_layer // 2) * c.T * c.NODES * c.n * c.p; HH_all = c.n_layer * (c.K - 1) * d
    lens = rng.integers(L // 3, L + 1, Bn).astype(np.int32); lens[0] = L
    has0 = (rng.random(Bn) < 0.6).astype(np.int32)
    X = rng.normal(0, 1, (Bn, L, d)).astype(np.float32)
    W = rng.normal(0, 1, (Bn, L, d)).astype(np.float32)
    for b in range(Bn): W[b, lens[b]:] = 0; X[b, lens[b]:] = 0
    H0 = (rng.normal(0, 0.3, (Bn, S_all)) * has0[:, None]).astype(np.float32)
    HH0 = (rng.normal(0, 0.5, (Bn, HH_all)) * has0[:, None]).astype(np.float32)
    inp, out = os.path.join(tmp, "in.bin"), os.path.join(tmp, "out.bin")
    with open(inp, "wb") as f:
        f.write(np.array([Bn, L], np.int32).tobytes()); f.write(lens.tobytes()); f.write(has0.tobytes())
        for arr in (X, W, H0, HH0): f.write(arr.tobytes())
    subprocess.check_call([exe, a.model, inp, out, str(a.lr), str(a.steps), init_path])
    # read C results
    z = c.sizes(); raw = open(out, "rb").read(); o = 0
    def take(n, dt=np.float32):
        nonlocal o; v = np.frombuffer(raw, dt, n, o); o += v.nbytes; return v
    XO = take(Bn * L * d).reshape(Bn, L, d); GX = take(Bn * L * d).reshape(Bn, L, d); loss_c = take(1, np.float64)[0]
    Gc = {f: take(z[f]) for f in nt.FIELDS}; hit_c = take(c.rows, np.uint8).astype(bool)
    losses_c = take(a.steps, np.float64); Pc = {f: take(z[f]) for f in nt.FIELDS}
    # the same on the torch side, from the checkpoint C wrote
    avx512 = "__AVX512F__" in subprocess.run(["gcc", os.environ.get("PARITY_MARCH", "-march=native"), "-dM", "-E", "-x", "c", "/dev/null"], capture_output=True, text=True).stdout
    bf = {"auto": None if avx512 else False, "bf16": True, "fp32": False}[a.tape]
    m = nt.TSS.load(init_path, device=a.device, bf16_tape=bf)
    print(f"C build: {'AVX-512 (bf16 state tape)' if avx512 else 'portable (fp32 state tape)'}; torch tape: {'bf16' if m.bf16_tape else 'fp32'}")
    if a.model.startswith("init:"):
        print(f"init: python params_init vs C   max rel {max(rel(P0[f], m.P[f].cpu().numpy()) for f in nt.FIELDS):.1e}")
    dev = m.device
    def run(step):
        st = m.state_from_c(H0, HH0)
        x = torch.tensor(X, device=dev)
        xo, tape = m.forward(x, state=st, lens=lens, train=True, seeds=[step * 1000003 + b for b in range(Bn)])
        nf = m.P["norm_f"]; u, r = nt.rmsnorm(xo, nf); Wt = torch.tensor(W, device=dev)
        loss = float((Wt * u).double().sum())
        g, dw = nt.rmsnorm_bwd(xo, nf, r, Wt); m.G["norm_f"] += dw
        gx = m.backward(tape, g)
        return xo, gx, loss
    xo, gx, loss = run(1)
    mask = (np.arange(L)[None, :] < lens[:, None])[..., None]
    print(f"forward    max rel {rel(xo.cpu().numpy() * mask, XO * mask):.2e}   loss {loss:.6f} vs C {loss_c:.6f}")
    print(f"input grad max rel {rel(gx.cpu().numpy() * mask, GX * mask):.2e}")
    hit_t = m.hit.cpu().numpy()
    print(f"visited rows: {hit_c.sum()} in C, {hit_t.sum()} here, {int((hit_c != hit_t).sum())} differ")
    worst = 0
    for f in nt.FIELDS:
        gc, gt = Gc[f], m.G[f].cpu().numpy()
        if np.abs(gc).max() == 0 and np.abs(gt).max() == 0: continue
        e = rel(gt, gc); worst = max(worst, e)
        print(f"  grad {f:8s} max rel {e:.2e}   |g| {np.linalg.norm(gc):.4g}")
    m.zero_grad()
    opt = nt.Optimizer(m, lr=a.lr)
    for s in range(1, a.steps + 1):
        _, _, ls = run(s); opt.step(int(lens.sum()))
        print(f"step {s}: loss {ls:.6f} vs C {losses_c[s - 1]:.6f}")
    pw = 0
    for f in nt.FIELDS:
        p0 = m0.P[f].cpu().numpy() if not a.model.startswith("init:") else P0[f]
        dc = Pc[f] - p0; dt = m.P[f].cpu().numpy() - p0
        if np.abs(dc).max() == 0 and np.abs(dt).max() == 0: continue
        e = rel(dt, dc); pw = max(pw, e)
        print(f"  update {f:8s} max rel {e:.2e}   |dp| {np.linalg.norm(dc):.3g}")
    print(f"worst gradient {worst:.2e}, worst update {pw:.2e}")


if __name__ == "__main__":
    main()
