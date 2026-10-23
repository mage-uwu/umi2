"""ppo_gpu.py: ppo_tss.c with the controller on the GPU (nanotss_torch.py) and the physics on CPU cores (libtssenv.so).

Same algorithm, flags, defaults and files as ppo_tss.c: the nanotss stack between an observation projection
and a tanh Gaussian head, PPO with GAE (a fall is terminal, the time limit bootstraps), minibatches of whole
chunks cut into segments at episode ends, each segment trained from the recurrent state it started in; the
stack updated by tss_train_step's rule (clip, Adam, blockwise Muon), the ends by Adam, an MLP critic.
Writes <out>.tss and <out>.head that ppo_tss.c --load, tss_run and the viewer read. What changes is scale:
thousands of environments per iteration and one batched training pass per minibatch.

    python3 ppo_gpu.py --task getup --style ../data/cmu_ref.bin:140_01,140_02 --envs 2048 --seconds 3600 --out getup
    python3 ppo_gpu.py --task getup --load getup --envs 2048 ...        (continue from a policy, as ppo_tss.c --load)

Rollout action noise comes from torch's generator, so runs are not step-for-step identical to the C trainer.
"""
import argparse, ctypes, json, math, os, struct, sys, time
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nanotss_torch as nt

F32 = np.float32
CH = 128


def lib():
    path = os.path.join(HERE, "libtssenv.so")
    if not os.path.exists(path):
        os.system(f"cd {HERE} && gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c -lm")
    L = ctypes.CDLL(path)
    L.pool_new.restype = ctypes.c_void_p
    L.pool_new.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p] + [ctypes.c_double] * 3 + [ctypes.c_int] + [ctypes.c_double] * 3 + [ctypes.c_int]
    for f in ("pool_obs", "pool_step", "pool_steps"): getattr(L, f).argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * (7 if f == "pool_step" else 1 if f == "pool_obs" else 0)
    return L


def ptr(a): return a.ctypes.data_as(ctypes.c_void_p)


class XorShift:   # ppo_tss.c's ur()
    def __init__(self, s): self.s = s & 0xFFFFFFFFFFFFFFFF
    def __call__(self):
        s = self.s; s ^= (s << 13) & 0xFFFFFFFFFFFFFFFF; s ^= s >> 7; s ^= (s << 17) & 0xFFFFFFFFFFFFFFFF; self.s = s
        return (s >> 11) * (1.0 / 9007199254740992.0)


class Core(torch.autograd.Function):
    """The nanotss stack as one autograd node; its backward is the C port, its parameter gradients go to model.G."""
    @staticmethod
    def forward(ctx, x, model, state, lens, seeds):
        out, tape = model.forward(x.detach(), state=state, lens=lens, train=True, seeds=seeds)
        ctx.model, ctx.tape = model, tape
        return out

    @staticmethod
    def backward(ctx, g):
        return ctx.model.backward(ctx.tape, g.contiguous()), None, None, None, None


def main():
    ap = argparse.ArgumentParser()
    A = ap.add_argument
    A("--out", default="tsspolicy"); A("--load"); A("--task", default="balance"); A("--mref", default=os.path.join(HERE, "../data/motions_ref.bin"))
    A("--seconds", type=float, default=600); A("--lr", type=float, default=3e-4); A("--head-lr", type=float, default=3e-4); A("--vlr", type=float, default=1e-3)
    A("--clip", type=float, default=0.2); A("--log-std", type=float, default=-1.6); A("--threads", type=int, default=0); A("--envs", type=int, default=64)
    A("--horizon", type=int, default=64); A("--epochs", type=int, default=4); A("--minibatches", type=int, default=4); A("--seed", type=int, default=0)
    A("--width", type=int, default=64); A("--depth", type=int, default=4); A("--trees", type=int, default=4)
    A("--push", type=float, default=0); A("--act-mul", type=float, default=0); A("--style", default=""); A("--style-start", type=float, default=0)
    A("--steps", type=int, default=0); A("--assist", type=float, default=0); A("--sway", type=float, default=0); A("--arm-tone", type=float, default=0)
    A("--gamma", type=float, default=0.95); A("--lam", type=float, default=0.95); A("--ent", type=float, default=0)
    A("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.threads: os.environ["OMP_NUM_THREADS"] = str(a.threads)
    E = lib(); obs_dim, act_dim, hz = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    E.tss_dims(ctypes.byref(obs_dim), ctypes.byref(act_dim), ctypes.byref(hz)); OBS, ACT, HZ = obs_dim.value, act_dim.value, hz.value
    pool = E.pool_new(a.envs, a.task.encode(), a.mref.encode(), a.style.encode(), a.assist, a.push, a.act_mul, a.steps, a.style_start, a.sway, a.arm_tone, a.seed)
    if not pool: sys.exit(1)
    dev = torch.device(a.device)

    # the stack
    if a.load:
        m = nt.TSS.load(a.load + ".tss", device=dev)
    else:
        m = nt.TSS(nt.Cfg(d=a.width, n_layer=4, T=a.trees, D=a.depth, n=16, p=16, K=3, groups=a.trees, qknorm=1, walks=2, temp=1, fff_gate=1), device=dev, seed=a.seed)
    c = m.c; d = c.d
    opt = nt.Optimizer(m, lr=a.lr)
    # the ends (Ends in ppo_tss.c) and the critic
    s = XorShift(a.seed * 31 + 7)
    Win = np.array([(2 * s() - 1) / math.sqrt(OBS) for _ in range(d * OBS)], np.float32).reshape(d, OBS)
    ends = {"Win": Win, "bin": np.zeros(d, F32), "Wo": np.zeros((ACT, d), F32), "bo": np.zeros(ACT, F32), "log_std": np.full(ACT, a.log_std, F32)}
    s = XorShift(a.seed + 5)
    W1 = np.array([(2 * s() - 1) / math.sqrt(OBS) for _ in range(CH * OBS)], F32).reshape(CH, OBS)
    W2 = np.array([(2 * s() - 1) / math.sqrt(CH) for _ in range(CH * CH)], F32).reshape(CH, CH)
    W3 = np.array([(2 * s() - 1) * 0.1 / math.sqrt(CH) for _ in range(CH)], F32)
    crit = {"W1": W1, "b1": np.zeros(CH, F32), "W2": W2, "b2": np.zeros(CH, F32), "W3": W3, "b3": np.zeros(1, F32)}
    mean, sd = np.zeros(OBS, F32), np.ones(OBS, F32); osum, osq, on = np.zeros(OBS), np.zeros(OBS), 0.0
    if a.load:
        raw = open(a.load + ".head", "rb").read()
        assert raw[:4] == b"TSPH" and struct.unpack("<3i", raw[4:16]) == (d, OBS, ACT), "head does not match"
        o = 16
        def take(n):
            nonlocal o; v = np.frombuffer(raw, F32, n, o).copy(); o += 4 * n; return v
        for k, n in (("Win", d * OBS), ("bin", d), ("Wo", ACT * d), ("bo", ACT), ("log_std", ACT)): ends[k] = take(n).reshape(ends[k].shape)
        mean, sd = take(OBS), take(OBS)
        if len(raw) - o >= 4 * sum(v.size for v in crit.values()):
            for k in ("W1", "b1", "W2", "b2", "W3", "b3"): crit[k] = take(crit[k].size).reshape(crit[k].shape)
        on = 1000; osum = mean.astype(np.float64) * on; osq = (sd.astype(np.float64) ** 2 + mean.astype(np.float64) ** 2) * on
        ends["log_std"][:] = a.log_std
    T_ = lambda v: torch.tensor(v, device=dev, requires_grad=True)
    EN = {k: T_(v) for k, v in ends.items()}; CR = {k: T_(v) for k, v in crit.items()}
    Em = {k: torch.zeros_like(v) for k, v in EN.items()}; Ev = {k: torch.zeros_like(v) for k, v in EN.items()}
    Cm = {k: torch.zeros_like(v) for k, v in CR.items()}; Cv = {k: torch.zeros_like(v) for k, v in CR.items()}

    def critic(o):
        h1 = torch.tanh(CR["b1"] + o @ CR["W1"].T); h2 = torch.tanh(CR["b2"] + h1 @ CR["W2"].T)
        return CR["b3"][0] + h2 @ CR["W3"]

    def head(x, nf):
        u, _ = nt.rmsnorm(x, nf)
        return torch.tanh(EN["bo"] + u @ EN["Wo"].T)

    def adam_dict(P, M, V, lr, scale, step):
        c1, c2 = F32(1 - math.pow(float(F32(.9)), step)), F32(1 - math.pow(float(F32(.999)), step))
        with torch.no_grad():
            for k in P:
                if P[k].grad is not None: nt.adam_(P[k].data, P[k].grad, M[k], V[k], lr, .9, .999, 1e-8, 0, F32(scale), c1, c2); P[k].grad = None

    nenv, hor = a.envs, a.horizon
    n_params = sum(v.numel() for f, v in m.P.items() if f != "embed")
    print(json.dumps({"event": "config", "task": a.task, "model": "nanotss byte stack (ssm-tree + fff, ternary STE) on " + str(dev),
                      "optimizer": "tss_train_step rule: blockwise muon + adam", "core_params": int(n_params), "io_params": int(sum(v.size for v in ends.values())),
                      "width": d, "trees": c.T, "depth": c.D, "envs": nenv, "horizon": hor}), flush=True)
    state = m.zero_state(nenv)
    O = np.zeros((nenv, hor + 1, OBS), F32); Araw = np.zeros((nenv, hor, ACT), F32); LP = np.zeros((nenv, hor), F32)
    VAL = np.zeros((nenv, hor + 1), F32); R = np.zeros((nenv, hor), F32); VTERM = np.zeros((nenv, hor), F32)
    DONE = np.zeros((nenv, hor), np.int32); FELL = np.zeros((nenv, hor), np.int32)
    obs = np.zeros((nenv, OBS), F32); rew = np.zeros(nenv, F32); done = np.zeros(nenv, np.int32); fell = np.zeros(nenv, np.int32)
    tobs = np.zeros((nenv, OBS), F32); epr = np.zeros(nenv, F32); epl = np.zeros(nenv, np.int32)
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed * 7919 + 1)
    erng = XorShift(a.seed * 7919 + 1)
    hstep = cstep = 0; t0 = time.time(); best = -1e30; it = 0; steps_total = 0; t_roll = 0.0; t_train = 0.0
    ep_ret_sum = ep_len_sum = 0.0; ep_count = 0
    norm_obs = lambda x: np.clip((x - mean) / sd, -5, 5).astype(F32)
    while time.time() - t0 < a.seconds:
        it += 1
        H0 = {k: [v.clone() for v in vs] for k, vs in state.items()}
        tr0 = time.time()
        with torch.no_grad():
            for t in range(hor + 1):
                E.pool_obs(pool, ptr(obs)); O[:, t] = norm_obs(obs)
                ot = torch.tensor(O[:, t], device=dev)
                VAL[:, t] = critic(ot).cpu().numpy()
                if t == hor: break
                x0 = ot @ EN["Win"].T + EN["bin"]
                x = m.forward(x0[:, None], state=state, stream=True)[:, 0]
                mu = head(x, m.P["norm_f"])
                sg = torch.exp(EN["log_std"]); z = torch.randn(mu.shape, device=dev, generator=gen)
                at = mu + sg * z
                LP[:, t] = (-0.5 * z.double() ** 2 - EN["log_std"].double() - 0.9189385332046727).sum(-1).cpu().numpy()
                Araw[:, t] = at.cpu().numpy()
                act = np.clip(Araw[:, t], -1, 1).astype(F32)
                E.pool_step(pool, ptr(act), ptr(rew), ptr(done), ptr(fell), ptr(tobs), ptr(epr), ptr(epl))
                R[:, t] = rew; DONE[:, t] = done; FELL[:, t] = fell
                te = np.nonzero(done & (1 - fell))[0]
                if len(te): VTERM[te, t] = critic(torch.tensor(norm_obs(tobs[te]), device=dev)).cpu().numpy()
                de = np.nonzero(done)[0]
                if len(de):
                    ep_ret_sum += float(epr[de].sum()); ep_len_sum += float(epl[de].sum()); ep_count += len(de)
                    di = torch.tensor(de, device=dev)
                    for v in state["h"] + state["hist"]: v[di] = 0
        t_roll += time.time() - tr0
        steps_total += nenv * hor
        for t in range(0, hor, 4):
            raw = O[:, t].astype(np.float64) * sd + mean; osum += raw.sum(0); osq += (raw * raw).sum(0); on += nenv
        # GAE
        ADV = np.zeros((nenv, hor), F32); RET = np.zeros((nenv, hor), F32); gg = np.zeros(nenv)
        for t in reversed(range(hor)):
            vnext = np.where(FELL[:, t] == 1, 0, np.where(DONE[:, t] == 1, VTERM[:, t], VAL[:, t + 1]))
            delta = R[:, t] + a.gamma * vnext - VAL[:, t]
            gg = delta + np.where(DONE[:, t] == 1, 0, a.gamma * a.lam * gg)
            ADV[:, t] = gg; RET[:, t] = gg + VAL[:, t]
        am = ADV.astype(np.float64).mean(); asd = math.sqrt(max((ADV.astype(np.float64) ** 2).mean() - am * am, 1e-8))
        ADV = ((ADV - am) / asd).astype(F32)
        # update
        trn0 = time.time()
        order = list(range(nenv)); clipped = 0; lpdiff = 0.0; lpn = 0; vloss = 0.0
        for ep in range(a.epochs):
            for e in range(nenv - 1, 0, -1):
                j = int(erng() * (e + 1)); order[e], order[j] = order[j], order[e]
            for bb in range(a.minibatches):
                e0, e1 = bb * nenv // a.minibatches, (bb + 1) * nenv // a.minibatches
                segs = []
                for e in order[e0:e1]:
                    ta = 0
                    while ta < hor:
                        tb = ta
                        while tb < hor - 1 and not DONE[e, tb]: tb += 1
                        segs.append((e, ta, tb - ta + 1)); ta = tb + 1
                ns = len(segs); Lm = max(L for _, _, L in segs)
                ee = np.array([s_[0] for s_ in segs]); tt = np.array([s_[1] for s_ in segs]); ll = np.array([s_[2] for s_ in segs])
                ti = tt[:, None] + np.arange(Lm)[None, :]; vm = np.arange(Lm)[None, :] < ll[:, None]; ti = np.where(vm, ti, 0)
                ob = torch.tensor(O[ee[:, None], ti] * vm[..., None], device=dev)
                acts = torch.tensor(Araw[ee[:, None], ti], device=dev); lpo = torch.tensor(LP[ee[:, None], ti], device=dev, dtype=torch.float64)
                adv = torch.tensor(ADV[ee[:, None], ti], device=dev, dtype=torch.float64); vmask = torch.tensor(vm, device=dev)
                first = torch.tensor(tt == 0, device=dev)
                st = {k: [torch.where(first.view(-1, *[1] * (v.dim() - 1)), v[torch.tensor(ee, device=dev)], torch.zeros(1, device=dev)) for v in vs] for k, vs in H0.items()}
                hstep += 1
                x0 = ob @ EN["Win"].T + EN["bin"]
                xo = Core.apply(x0, m, st, torch.tensor(ll, device=dev), [hstep * 1000003 + b for b in range(ns)])
                nf = m.P["norm_f"].detach().clone().requires_grad_(True)
                mu = head(xo, nf)
                sgm = torch.exp(EN["log_std"]); zz = (acts - mu) / sgm
                lp = (-0.5 * zz.double() ** 2 - EN["log_std"].double() - 0.9189385332046727).sum(-1)
                if ep == 0 and bb == 0: lpdiff += float((lp - lpo).abs()[vmask].sum()); lpn += int(vmask.sum())
                ratio = torch.exp(torch.where(vmask, lp - lpo, torch.zeros_like(lp)))
                s1 = ratio * adv; s2 = ratio.clamp(1 - a.clip, 1 + a.clip) * adv
                active = (s1 <= s2) | ((ratio > 1 - a.clip) & (ratio < 1 + a.clip))
                clipped += int((~active & vmask).sum())
                sur = torch.where(active, -s1, -torch.minimum(s1, s2).detach())
                loss = (sur * vmask).sum() - a.ent * EN["log_std"].sum() * float(vmask.sum())
                loss.backward()
                m.G["norm_f"] += nf.grad
                opt.step(int(ll.sum()))
                adam_dict(EN, Em, Ev, a.head_lr, 1.0 / int(ll.sum()), hstep)
                with torch.no_grad(): EN["log_std"].clamp_(-3, 0)
                # critic
                oc = torch.tensor(O[order[e0:e1], :hor].reshape(-1, OBS), device=dev); rc = torch.tensor(RET[order[e0:e1]].reshape(-1), device=dev)
                dv = critic(oc) - rc; (0.5 * (dv * dv).sum()).backward(); vloss += float((dv.detach() ** 2).sum())
                cstep += 1; adam_dict(CR, Cm, Cv, a.vlr, 1.0 / ((e1 - e0) * hor), cstep)
        t_train += time.time() - trn0
        if on > 10:
            mu_ = osum / on; var = osq / on - mu_ * mu_; mean = mu_.astype(F32); sd = np.sqrt(np.maximum(var, 1e-2)).astype(F32)
        rw = ep_ret_sum / ep_count if ep_count else 0; ln = ep_len_sum / ep_count / HZ if ep_count else 0
        sgm_ = float(torch.exp(EN["log_std"]).mean())
        if it % 5 == 1 or time.time() - t0 >= a.seconds:
            el = time.time() - t0
            print(json.dumps({"event": "iter", "iter": it, "seconds": round(el), "steps": steps_total, "episode_return": round(rw, 1), "episode_s": round(ln, 2),
                              "episodes": ep_count, "clip_frac": round(clipped / (a.epochs * nenv * hor), 3), "value_loss": round(vloss / (a.epochs * nenv * hor), 3),
                              "std": round(sgm_, 3), "lp_gap": float(f"{lpdiff / max(lpn, 1):.2e}"), "rollout_share": round(t_roll / el, 2), "train_share": round(t_train / el, 2),
                              "steps_per_s": round(steps_total / el)}), flush=True)
            if ep_count >= 8 and rw > best:
                best = rw; m.save(a.out + ".tss"); save_head(a.out + ".head", EN, mean, sd, d, OBS, ACT, CR)
            ep_ret_sum = ep_len_sum = 0.0; ep_count = 0
    print(json.dumps({"event": "done", "iterations": it, "steps": steps_total, "best_episode_return": round(best, 1)}), flush=True)


def save_head(path, EN, mean, sd, d, OBS, ACT, CR):
    with open(path, "wb") as f:
        f.write(b"TSPH"); f.write(struct.pack("<3i", d, OBS, ACT))
        for k in ("Win", "bin", "Wo", "bo", "log_std"): f.write(EN[k].detach().cpu().numpy().astype(F32).tobytes())
        f.write(mean.astype(F32).tobytes()); f.write(sd.astype(F32).tobytes())
        for k in ("W1", "b1", "W2", "b2", "W3", "b3"): f.write(CR[k].detach().cpu().numpy().astype(F32).tobytes())


if __name__ == "__main__":
    main()
