"""ppo_gpu.py: ppo_tss.c on the GPU, the controller (nanotss_torch.py) and the physics (gpu_env.GpuPool) both.

Same algorithm, flags, defaults and files as ppo_tss.c: the nanotss stack between an observation projection
and a tanh Gaussian head, PPO with GAE (a fall is terminal, the time limit bootstraps), minibatches of whole
chunks cut into segments at episode ends, each segment trained from the recurrent state it started in; the
stack updated by tss_train_step's rule (clip, Adam, blockwise Muon), the ends by Adam, an MLP critic.
Writes <out>.tss and <out>.head that ppo_tss.c --load, tss_run and the viewer read.

On a CUDA device everything stays there: the environments step as CUDA threads running body.c / env.c, and the
observations, actions, rollout buffers, advantages and training never leave the device. With --cpu-physics (or
on a CPU) the environments run in libtssenv.so instead.

    python3 ppo_gpu.py --task getup --style ../data/cmu_ref.bin:getup_back1,getup_back2,getup_face1,getup_face2,getup_side1,getup_side2 \\
        --envs 4096 --seconds 3600 --out getup
    python3 ppo_gpu.py --task getup --load getup ...        (continue from a policy, as ppo_tss.c --load)

Rollout action noise comes from torch's generator, so runs are not step-for-step identical to the C trainer.
"""
import argparse, json, math, os, struct, sys, time
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nanotss_torch as nt
import gpu_env as ge

F32 = np.float32
CH = 128


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


def save_head(path, EN, mean, sd, d, OBS, ACT, CR):
    with open(path, "wb") as f:
        f.write(b"TSPH"); f.write(struct.pack("<3i", d, OBS, ACT))
        for k in ("Win", "bin", "Wo", "bo", "log_std"): f.write(EN[k].detach().cpu().numpy().astype(F32).tobytes())
        f.write(mean.cpu().numpy().astype(F32).tobytes()); f.write(sd.cpu().numpy().astype(F32).tobytes())
        for k in ("W1", "b1", "W2", "b2", "W3", "b3"): f.write(CR[k].detach().cpu().numpy().astype(F32).tobytes())


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
    A("--device", default="cuda" if torch.cuda.is_available() else "cpu"); A("--cpu-physics", action="store_true")
    A("--assist-decay", type=float, default=0, help="the assist from --assist down to 0 in a straight line over this many seconds, then none (getting up, where no episode ends early and --assist-auto has nothing to go by); the checkpoint as --assist-auto's")
    A("--assist-auto", action="store_true", help="adapt the assist: down by 15%% (at least 0.005) while episodes last over 60%% of their length, up 0.02 (to --assist at most) under 30%%")
    A("--fresh-inputs", default="", help="on --load: inputs the loaded policy never saw vary (always zero in its task), as groups: goal, grip, probe, path, touch, "
      "muscle; their weights are zeroed and their normaliser starts afresh, so the policy starts out acting exactly as it did "
      "(e.g. 'goal,grip' when a walk imitation prior starts go-to)")
    A("--free-heading", type=int, default=0, help="imitation: imitate the clip's style, not its path (posture, limbs, speed in her own heading frame); only falls end episodes")
    A("--goal-tops", type=float, default=0, help="go-to: share of goals on top of room boxes (terrain); balance: share of episodes starting on top of one (ramps too); 0 = 0.3")
    A("--goal-cone", type=float, default=0, help="go-to: goals within this angle (rad) of her heading; 0 = any")
    A("--room", default=os.path.join(HERE, "../room.json"), help="room.json (physics_gym); '' for the empty floor")
    a = ap.parse_args()
    if a.threads: os.environ["OMP_NUM_THREADS"] = str(a.threads)
    dev = torch.device(a.device)
    Pool = ge.GpuPool if dev.type == "cuda" and not a.cpu_physics else ge.CpuPool
    pool = Pool(a.envs, task=a.task, mref=a.mref, style=a.style, assist=a.assist, push=a.push, act_mul=a.act_mul, steps=a.steps,
                style_start=a.style_start, sway=a.sway, arm_tone=a.arm_tone, goal_cone=a.goal_cone, goal_tops=a.goal_tops, free_heading=a.free_heading, seed=a.seed, device=dev, room=a.room)
    OBS, ACT, HZ = pool.OBS, pool.ACT, pool.HZ

    # the stack
    if a.load:
        m = nt.TSS.load(a.load + ".tss", device=dev)
    else:
        m = nt.TSS(nt.Cfg(d=a.width, n_layer=4, T=a.trees, D=a.depth, n=16, p=16, K=3, groups=a.trees, qknorm=1, walks=2, temp=1, fff_gate=1), device=dev, seed=a.seed)
    c = m.c; d = c.d
    opt = nt.Optimizer(m, lr=a.lr)
    # the ends (Ends in ppo_tss.c) and the critic, initialised as ppo_tss.c does
    s = XorShift(a.seed * 31 + 7)
    Win = np.array([(2 * s() - 1) / math.sqrt(OBS) for _ in range(d * OBS)], F32).reshape(d, OBS)
    ends = {"Win": Win, "bin": np.zeros(d, F32), "Wo": np.zeros((ACT, d), F32), "bo": np.zeros(ACT, F32), "log_std": np.full(ACT, a.log_std, F32)}
    s = XorShift(a.seed + 5)
    W1 = np.array([(2 * s() - 1) / math.sqrt(OBS) for _ in range(CH * OBS)], F32).reshape(CH, OBS)
    W2 = np.array([(2 * s() - 1) / math.sqrt(CH) for _ in range(CH * CH)], F32).reshape(CH, CH)
    W3 = np.array([(2 * s() - 1) * 0.1 / math.sqrt(CH) for _ in range(CH)], F32)
    crit = {"W1": W1, "b1": np.zeros(CH, F32), "W2": W2, "b2": np.zeros(CH, F32), "W3": W3, "b3": np.zeros(1, F32)}
    mean, sd = np.zeros(OBS, F32), np.ones(OBS, F32); on = 0.0; osum = np.zeros(OBS); osq = np.zeros(OBS)
    if a.load:
        raw = open(a.load + ".head", "rb").read()
        hd, hobs, hact = struct.unpack("<3i", raw[4:16])
        # a head with fewer inputs (a body/ policy: 216) continues here with the new inputs (goal, touch, grip) appended,
        # their weights zero, and fewer outputs (31: no grip) with the new ones zero, so it starts out acting as before
        assert raw[:4] == b"TSPH" and hd == d and hact <= ACT and hobs <= OBS, "head does not match"
        o = 16
        def take(n):
            nonlocal o; v = np.frombuffer(raw, F32, n, o).copy(); o += 4 * n; return v
        ends["Win"][:, :hobs] = take(d * hobs).reshape(d, hobs); ends["Win"][:, hobs:] = 0
        ends["bin"] = take(d); ends["Wo"][:] = 0; ends["Wo"][:hact] = take(hact * d).reshape(hact, d)
        ends["bo"][:] = 0; ends["bo"][:hact] = take(hact); take(hact)   # (log_std: reset below)
        mean[:hobs], sd[:hobs] = take(hobs), take(hobs)
        if len(raw) - o >= 4 * (sum(v.size for v in crit.values()) - CH * (OBS - hobs)):
            crit["W1"][:, :hobs] = take(CH * hobs).reshape(CH, hobs); crit["W1"][:, hobs:] = 0
            for k in ("b1", "W2", "b2", "W3", "b3"): crit[k] = take(crit[k].size).reshape(crit[k].shape)
        GROUPS = {"goal": (216, 220), "touch": (220, 234), "muscle": (234, 247), "grip": (247, 257), "probe": (257, 273), "path": (273, 277)}   # env.h OBS layout after OBS_BASE
        fresh = [GROUPS[g] for g in a.fresh_inputs.split(",") if g]
        for lo, hi in fresh:   # never varied in the prior's training: start them at zero effect, the normaliser at (0, 1)
            ends["Win"][:, lo:hi] = 0; crit["W1"][:, lo:hi] = 0; mean[lo:hi] = 0; sd[lo:hi] = 1
        if fresh: print(json.dumps({"event": "load", "fresh_inputs": a.fresh_inputs, "columns": sum(h - l for l, h in fresh)}), flush=True)
        if hobs < OBS or hact < ACT: print(json.dumps({"event": "load", "inputs": hobs, "outputs": hact, "widened_to": [OBS, ACT]}), flush=True)
        on = 1000.0; osum = mean.astype(np.float64) * on; osq = (sd.astype(np.float64) ** 2 + mean.astype(np.float64) ** 2) * on
        ends["log_std"][:] = a.log_std
    T_ = lambda v: torch.tensor(v, device=dev, requires_grad=True)
    EN = {k: T_(v) for k, v in ends.items()}; CR = {k: T_(v) for k, v in crit.items()}
    Em = {k: torch.zeros_like(v) for k, v in EN.items()}; Ev = {k: torch.zeros_like(v) for k, v in EN.items()}
    Cm = {k: torch.zeros_like(v) for k, v in CR.items()}; Cv = {k: torch.zeros_like(v) for k, v in CR.items()}
    mean = torch.tensor(mean, device=dev); sd = torch.tensor(sd, device=dev)
    osum = torch.tensor(osum, device=dev); osq = torch.tensor(osq, device=dev)

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
                      "physics": "cuda (body.c / env.c as device code)" if Pool is ge.GpuPool else "cpu (libtssenv)",
                      "optimizer": "tss_train_step rule: blockwise muon + adam", "core_params": int(n_params), "io_params": int(sum(v.size for v in ends.values())),
                      "width": d, "trees": c.T, "depth": c.D, "envs": nenv, "horizon": hor}), flush=True)
    state = m.zero_state(nenv)
    O = torch.zeros(nenv, hor + 1, OBS, device=dev); Araw = torch.zeros(nenv, hor, ACT, device=dev); LP = torch.zeros(nenv, hor, device=dev, dtype=torch.float64)
    VAL = torch.zeros(nenv, hor + 1, device=dev); R = torch.zeros(nenv, hor, device=dev); VTERM = torch.zeros(nenv, hor, device=dev)
    DONE = torch.zeros(nenv, hor, dtype=torch.bool, device=dev); FELL = torch.zeros(nenv, hor, dtype=torch.bool, device=dev)
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed * 7919 + 1)
    erng = XorShift(a.seed * 7919 + 1)
    hstep = cstep = 0; t0 = time.time(); best = -1e30; it = 0; steps_total = 0; t_roll = 0.0; t_train = 0.0
    cur_assist = a.assist; best_assist = 1e9
    ep_ret_sum = torch.zeros((), device=dev, dtype=torch.float64); ep_len_sum = torch.zeros_like(ep_ret_sum); ep_cnt = torch.zeros((), device=dev, dtype=torch.long)
    norm_obs = lambda x: ((x - mean) / sd).clamp(-5, 5)
    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)
    while time.time() - t0 < a.seconds:
        it += 1
        H0 = {k: [v.clone() for v in vs] for k, vs in state.items()}
        tr0 = time.time()
        with torch.no_grad():
            for t in range(hor + 1):
                ot = norm_obs(pool.obs()); O[:, t] = ot
                VAL[:, t] = critic(ot)
                if t == hor: break
                x0 = ot @ EN["Win"].T + EN["bin"]
                x = m.forward(x0[:, None], state=state, stream=True)[:, 0]
                mu = head(x, m.P["norm_f"])
                z = torch.randn(mu.shape, device=dev, generator=gen)
                Araw[:, t] = mu + torch.exp(EN["log_std"]) * z
                LP[:, t] = (-0.5 * z.double() ** 2 - EN["log_std"].double() - 0.9189385332046727).sum(-1)
                r = pool.step(Araw[:, t].clamp(-1, 1))
                done, fell = r["done"].bool(), r["fell"].bool()
                R[:, t] = r["rew"]; DONE[:, t] = done; FELL[:, t] = fell
                VTERM[:, t] = torch.where(done & ~fell, critic(norm_obs(r["tobs"])), torch.zeros_like(VTERM[:, t]))
                ep_ret_sum += r["ep_ret"].double().sum(); ep_len_sum += r["ep_len"].double().sum(); ep_cnt += done.long().sum()
                keep = (~done).float()
                for v in state["h"]: v.mul_(keep.view(-1, 1, 1, 1))
                for v in state["hist"]: v.mul_(keep.view(-1, 1, 1))
            sync()
        t_roll += time.time() - tr0
        steps_total += nenv * hor; goals_it = int((R > 1.9).sum())   # go-to: arrival bonuses (+2 on a step reward below 1) this iteration
        raw = O[:, 0:hor:4].double() * sd.double() + mean.double()
        osum += raw.sum((0, 1)); osq += (raw * raw).sum((0, 1)); on += raw.shape[0] * raw.shape[1]
        # GAE (a fall is terminal, the time limit bootstraps)
        with torch.no_grad():
            ADV = torch.zeros(nenv, hor, device=dev, dtype=torch.float64); gg = torch.zeros(nenv, device=dev, dtype=torch.float64)
            for t in reversed(range(hor)):
                vnext = torch.where(FELL[:, t], torch.zeros_like(VTERM[:, t]), torch.where(DONE[:, t], VTERM[:, t], VAL[:, t + 1])).double()
                delta = R[:, t].double() + a.gamma * vnext - VAL[:, t].double()
                gg = delta + torch.where(DONE[:, t], torch.zeros_like(gg), a.gamma * a.lam * gg)
                ADV[:, t] = gg
            RET = (ADV + VAL[:, :hor].double()).float()
            am = ADV.mean(); asd = torch.sqrt(torch.clamp((ADV ** 2).mean() - am * am, min=1e-8))
            ADV = (ADV - am) / asd
        # update: minibatches of whole chunks, cut into segments at episode ends
        trn0 = time.time()
        DONE_np = DONE.cpu().numpy()
        order = list(range(nenv)); clipped = torch.zeros((), device=dev, dtype=torch.long); lpdiff = torch.zeros((), device=dev, dtype=torch.float64); lpn = 0
        vloss = torch.zeros((), device=dev, dtype=torch.float64)
        for ep in range(a.epochs):
            for e in range(nenv - 1, 0, -1):
                j = int(erng() * (e + 1)); order[e], order[j] = order[j], order[e]
            for bb in range(a.minibatches):
                e0, e1 = bb * nenv // a.minibatches, (bb + 1) * nenv // a.minibatches
                rows = np.array(order[e0:e1]); Dm = DONE_np[rows]
                starts = np.zeros_like(Dm); starts[:, 0] = True; starts[:, 1:] = Dm[:, :-1]
                ri, ti = np.nonzero(starts); pos = ri * hor + ti
                end = np.minimum(np.append(pos[1:], len(rows) * hor), (ri + 1) * hor); ll = end - pos
                ee = rows[ri]; ns = len(ee); Lm = int(ll.max())
                eeT = torch.tensor(ee, device=dev); ttT = torch.tensor(ti, device=dev); llT = torch.tensor(ll, device=dev)
                tix = ttT[:, None] + torch.arange(Lm, device=dev)[None, :]; vmask = torch.arange(Lm, device=dev)[None, :] < llT[:, None]; tix = torch.where(vmask, tix, torch.zeros_like(tix))
                ob = O[eeT[:, None], tix] * vmask[..., None]
                acts = Araw[eeT[:, None], tix]; lpo = LP[eeT[:, None], tix]; adv = ADV[eeT[:, None], tix]
                first = (ttT == 0)
                st = {k: [torch.where(first.view(-1, *[1] * (v.dim() - 1)), v[eeT], torch.zeros((), device=dev)) for v in vs] for k, vs in H0.items()}
                hstep += 1
                x0 = ob @ EN["Win"].T + EN["bin"]
                xo = Core.apply(x0, m, st, llT, [hstep * 1000003 + b for b in range(ns)])
                nf = m.P["norm_f"].detach().clone().requires_grad_(True)
                mu = head(xo, nf)
                zz = (acts - mu) / torch.exp(EN["log_std"])
                lp = (-0.5 * zz.double() ** 2 - EN["log_std"].double() - 0.9189385332046727).sum(-1)
                if ep == 0 and bb == 0: lpdiff = lpdiff + (lp - lpo).abs()[vmask].sum().detach(); lpn += int(ll.sum())
                ratio = torch.exp(torch.where(vmask, lp - lpo, torch.zeros_like(lp)))
                s1 = ratio * adv; s2 = ratio.clamp(1 - a.clip, 1 + a.clip) * adv
                active = (s1 <= s2) | ((ratio > 1 - a.clip) & (ratio < 1 + a.clip))
                clipped += (~active & vmask).sum()
                sur = torch.where(active, -s1, -torch.minimum(s1, s2).detach())
                nsamp = int(ll.sum())
                loss = (sur * vmask).sum() - a.ent * EN["log_std"].sum() * nsamp
                loss.backward()
                m.G["norm_f"] += nf.grad
                opt.step(nsamp)
                adam_dict(EN, Em, Ev, a.head_lr, 1.0 / nsamp, hstep)
                with torch.no_grad(): EN["log_std"].clamp_(-3, 0)
                # critic
                rT = torch.tensor(rows, device=dev)
                dv = critic(O[rT, :hor].reshape(-1, OBS)) - RET[rT].reshape(-1)
                (0.5 * (dv * dv).sum()).backward(); vloss = vloss + (dv.detach().double() ** 2).sum()
                cstep += 1; adam_dict(CR, Cm, Cv, a.vlr, 1.0 / ((e1 - e0) * hor), cstep)
        sync(); t_train += time.time() - trn0
        if on > 10:
            mu_ = osum / on; var = osq / on - mu_ * mu_; mean = mu_.float(); sd = torch.sqrt(torch.clamp(var, min=1e-2)).float()
        if it % 5 == 1 or time.time() - t0 >= a.seconds:
            el = time.time() - t0; cnt = int(ep_cnt)
            rw = float(ep_ret_sum) / cnt if cnt else 0; ln = float(ep_len_sum) / cnt / HZ if cnt else 0
            print(json.dumps({"event": "iter", "iter": it, "seconds": round(el), "steps": steps_total, "episode_return": round(rw, 1), "episode_s": round(ln, 2),
                              "episodes": cnt, "goals_reached": goals_it, "clip_frac": round(int(clipped) / (a.epochs * nenv * hor), 3), "value_loss": round(float(vloss) / (a.epochs * nenv * hor), 3),
                              "std": round(float(torch.exp(EN["log_std"]).mean()), 3), "lp_gap": float(f"{float(lpdiff) / max(lpn, 1):.2e}"),
                              "rollout_share": round(t_roll / el, 2), "train_share": round(t_train / el, 2), "steps_per_s": round(steps_total / el)}), flush=True)
            if a.assist_decay > 0:   # the checkpoint: the lowest assist reached, the best return there
                if cnt >= 8 and (cur_assist < best_assist - 1e-9 or (abs(cur_assist - best_assist) < 1e-9 and rw > best)):
                    best_assist, best = cur_assist, rw; m.save(a.out + ".tss"); save_head(a.out + ".head", EN, mean, sd, d, OBS, ACT, CR)
                new = a.assist * max(0.0, 1 - el / a.assist_decay); new = new if new > 0.004 else 0.0
                if new != cur_assist: cur_assist = new; pool.set_assist(cur_assist)
                print(json.dumps({"event": "assist", "iter": it, "assist": round(cur_assist, 3), "checkpoint_assist": round(best_assist, 3)}), flush=True)
            elif a.assist_auto:   # the checkpoint: the lowest assist reached, the best return there
                if cnt >= 8 and (cur_assist < best_assist - 1e-9 or (abs(cur_assist - best_assist) < 1e-9 and rw > best)):
                    best_assist, best = cur_assist, rw; m.save(a.out + ".tss"); save_head(a.out + ".head", EN, mean, sd, d, OBS, ACT, CR)
                frac = ln * HZ / pool.steps if cnt else 0
                down = cur_assist - max(0.005, 0.15 * cur_assist); new = (down if down > 0.004 else 0.0) if frac > 0.6 else min(a.assist, cur_assist + 0.02) if frac < 0.3 else cur_assist
                if new != cur_assist: cur_assist = new; pool.set_assist(cur_assist)
                print(json.dumps({"event": "assist", "iter": it, "survival": round(frac, 2), "assist": round(cur_assist, 3), "checkpoint_assist": round(best_assist, 3)}), flush=True)
            elif cnt >= 8 and rw > best:
                best = rw; m.save(a.out + ".tss"); save_head(a.out + ".head", EN, mean, sd, d, OBS, ACT, CR)
            ep_ret_sum.zero_(); ep_len_sum.zero_(); ep_cnt.zero_()
    print(json.dumps({"event": "done", "iterations": it, "steps": steps_total, "best_episode_return": round(best, 1)}), flush=True)


if __name__ == "__main__":
    main()
