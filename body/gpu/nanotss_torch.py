"""nanotss_torch.py: the nanotss byte stack (nanotss_blt.c) on the GPU, kept 1:1 with the C version.

What "1:1" means here, and how it is kept:
  * checkpoints: .tss files are read and written in the C layout (TFX1/TFW1 header, Cfg, extension ints,
    the eleven fields in order), so a model trained here loads in the C trainer, tss_run and the browser.
  * forward: the same arithmetic. Ternary weights are rebuilt with the C rounding and the C summation order
    for the group scale; int8 activations use quant8's rounding; dot products of int8 codes with ternary
    weights are exact integers (TF32 is switched off), scaled the way pdot8 scales them. Erf, exp, sqrt,
    softplus and sigmoid are evaluated in double from the same float operands as the C library calls.
  * training: no autograd inside the stack. The backward is a port of layer_bwd / fff_bwd: straight-through
    gradients for the ternary weights and int8 activations, the sampled routing credit (alternate walks,
    the same xorshift stream per sequence as tss_work_seed / urand, so the same walks are sampled), the
    recurrent state adjoints in reverse time, with the state tape rounded to bf16 as the AVX-512 build does.
    The update is tss_train_step's: one global clip, Adam on the dense fields and on visited rows only,
    blockwise Muon (16-row blocks of visited rows, five quintic Newton-Schulz steps) on w_in / w_out / proj / U.
Differences that remain are float summation order (about 1e-6 relative), which can flip a routing decision
or an int8 rounding where a value sits exactly on the boundary. tools/parity.py measures them against C.

Supported: the configurations the body controllers use (SSM-tree + FFF layers, any width, trees, depth,
ssm_groups, qk-norm on or off, the FFF gate). Not ported: fast-weight layers (fw_mask), liquid time constants
(ssm_ltc), rotary B/C, n_loops > 1; loading such a checkpoint raises.
"""
import math
import os
import struct
import ctypes

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

FIELDS = ("embed", "norm_f", "norm", "conv", "w_in", "bias", "w_out", "A_log", "proj", "dt_bias", "U")
NODE_FIELDS = ("w_in", "bias", "w_out", "A_log")
F32 = np.float32


# ----------------------------------------------------------------------------------------------- config
class Cfg:
    FMT = "<10if2i"   # vocab d n_layer T D n p K group walks | temp | fw_mask fff_gate

    def __init__(self, vocab=256, d=64, n_layer=4, T=4, D=4, n=16, p=16, K=3, group=None, walks=2, temp=1.0,
                 fw_mask=0, fff_gate=1, groups=None, qknorm=1, rope=0, ltc=0, loops=1, pos_feat=0.0, dec_cap=1.0):
        self.vocab, self.d, self.n_layer, self.T, self.D, self.n, self.p, self.K = vocab, d, n_layer, T, D, n, p, K
        self.group = d if group is None else group
        self.walks, self.temp, self.fw_mask, self.fff_gate = walks, float(F32(temp)), fw_mask, fff_gate
        self.groups = T if groups is None else groups          # ssm_groups
        self.qknorm, self.rope, self.ltc, self.loops, self.pos_feat = qknorm, rope, ltc, loops, pos_feat
        self.dec_cap = float(F32(dec_cap))                       # ssm_dec_cap (1 unless --ssm-horizon)

    NODES = property(lambda c: (1 << (c.D + 1)) - 1)
    FNODES = property(lambda c: c.NODES + 2 * min(1 << c.D, 9))   # fff_nodes
    PDIM = property(lambda c: 2 * c.n * c.groups + 2 * c.p + c.T)

    def node_off(self, l):
        return self.T * ((l + 1) // 2 * self.NODES + l // 2 * self.FNODES)

    @property
    def rows(self):
        return self.node_off(self.n_layer)

    def sizes(self):
        nl, d, r = self.n_layer, self.d, self.rows
        return {"embed": self.vocab * d, "norm_f": d, "norm": nl * d, "conv": nl * self.K * d, "w_in": r * d, "bias": r,
                "w_out": r * d, "A_log": r, "proj": nl * self.PDIM * d, "dt_bias": nl * self.T, "U": nl * self.T * self.p * d}

    def check(self):
        bad = []
        if self.fw_mask: bad.append("fast-weight layers (fw_mask)")
        if self.ltc: bad.append("ssm_ltc")
        if self.rope: bad.append("rotary B/C (ssm_rope)")
        if self.loops != 1: bad.append("n_loops > 1")
        if self.d % self.group or self.T % self.groups: bad.append("group sizes")
        if bad: raise NotImplementedError("not ported to the GPU trainer: " + ", ".join(bad))


# ----------------------------------------------------------------------------------------------- float helpers (C arithmetic)
def exp_d(x):            # exp() of a float, in double, rounded to float
    return torch.exp(x.double()).float()


def sigm(x):             # 1/(1+exp(-x)) in double
    return (1.0 / (1.0 + torch.exp(-x.double()))).float()


def softplus(x):
    return torch.where(x > 20, x, torch.log1p(torch.exp(x.double())).float())


def gelu_rec(s):         # record_gelu: value and derivative from one erf, the C grouping
    e = torch.erf((s * F32(0.70710678118654752440)).double())
    y = ((F32(0.5) * s).double() * (1 + e)).float()
    ex = torch.exp(((F32(-0.5) * s) * s).double())
    dy = (0.5 * (1 + e) + s.double() * ex * float(F32(0.39894228040143267794))).float()
    return y, dy


def rmsnorm(x, w):
    d = x.shape[-1]
    v = (x * x).sum(-1) / F32(d) + F32(1e-5)
    r = (1.0 / torch.sqrt(v.double())).float()
    return (x * r[..., None]) * w, r


def rmsnorm_bwd(x, w, r, du):
    d = x.shape[-1]
    s = (w * du * x).sum(-1)
    k = r * r * r * s / F32(d)
    dx = (r[..., None] * w) * du - x * k[..., None]
    dw = (du * x * r[..., None]).reshape(-1, d).sum(0)
    return dx, dw


def quant8(x):
    m = x.abs().amax(-1).clamp_min(F32(1e-5))
    scale = F32(127) / m
    v = torch.round(x * scale[..., None]).clamp(-128, 127)
    return v / scale[..., None], v, 1 / scale


def ternary(W, g):
    """tss_ternary: absmean scale per group (summed in the C order), round(clamp(w/s)) * s."""
    R, d = W.shape
    Wg = W.view(R, d // g, g)
    s = torch.zeros(R, d // g, device=W.device)
    for i in range(g):
        s = s + Wg[..., i].abs()
    s = (s / F32(g)).clamp_min(F32(1e-8))
    t = torch.round((Wg / s[..., None]).clamp(-1, 1))
    return t.view(R, d), s, (t * s[..., None]).view(R, d)


class QMat:
    """A ternary matrix as the packed planes hold it: signs t (-1/0/1), group scales sc, dense Q = t*sc."""
    def __init__(self, W, g):
        self.g = g
        self.t, self.sc, self.Q = ternary(W, g)

    def rows(self, a, b):
        q = QMat.__new__(QMat); q.g = self.g; q.t, q.sc, q.Q = self.t[a:b], self.sc[a:b], self.Q[a:b]; return q

    def dot8(self, c8, xs):     # pdot8: exact integer dot of int8 codes with signs, per group times scale, times xs
        ng = self.sc.shape[1]
        if ng == 1:
            return (c8 @ self.t.T) * self.sc[:, 0] * xs[..., None]
        g = self.g; acc = 0
        for j in range(ng):
            acc = acc + self.sc[:, j] * (c8[..., j * g:(j + 1) * g] @ self.t[:, j * g:(j + 1) * g].T)
        return acc * xs[..., None]

    def dotf(self, v):          # pdotf: float vector against the signs, per group times scale
        ng = self.sc.shape[1]
        if ng == 1:
            return (v @ self.t.T) * self.sc[:, 0]
        g = self.g; acc = 0
        for j in range(ng):
            acc = acc + self.sc[:, j] * (v[..., j * g:(j + 1) * g] @ self.t[:, j * g:(j + 1) * g].T)
        return acc


# ----------------------------------------------------------------------------------------------- the routing-credit random stream
_lib = None


def _shim():
    global _lib
    if _lib is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libtssenv.so")
        _lib = ctypes.CDLL(path) if os.path.exists(path) else False
    return _lib or None


def urand_stream(seeds, n):
    """n successive urand() values for each Work seeded with tss_work_seed(seed): float32 [B, n]."""
    seeds = np.asarray(seeds, dtype=np.uint64)
    B = len(seeds); out = np.empty((B, n), np.float32)
    lib = _shim()
    if lib:
        lib.tss_urand(seeds.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(B), ctypes.c_int(n), out.ctypes.data_as(ctypes.c_void_p))
        return out
    with np.errstate(over="ignore"):
        s = seeds * np.uint64(0x9E3779B97F4A7C15) + np.uint64(1)
        mul, k = np.uint64(2685821657736338717), F32(1.0 / 9007199254740992.0)
        for i in range(n):
            s ^= s >> np.uint64(12); s ^= s << np.uint64(25); s ^= s >> np.uint64(27)
            out[:, i] = ((s * mul) >> np.uint64(11)).astype(np.float32) * k
    return out


class _Draws:
    """Per-sequence position in the stream; layers take their draws in the C order."""
    def __init__(self, seeds, n, device):
        self.u = torch.from_numpy(urand_stream(seeds, n)).to(device)
        self.off = torch.zeros(len(seeds), dtype=torch.long, device=device)

    def take(self, idx):        # idx [B, ...] offsets from this sequence's position
        B = idx.shape[0]
        flat = (idx.reshape(B, -1) + self.off[:, None]).clamp(max=self.u.shape[1] - 1)
        return self.u.gather(1, flat).view(idx.shape)


# ----------------------------------------------------------------------------------------------- C-compatible init
class _CRand:
    def __init__(self, s): self.s = s & 0xFFFFFFFFFFFFFFFF

    def rand01(self):
        s = self.s; s ^= s >> 12; s ^= (s << 25) & 0xFFFFFFFFFFFFFFFF; s ^= s >> 27; self.s = s
        return float(((s * 2685821657736338717) & 0xFFFFFFFFFFFFFFFF) >> 11) * (1.0 / 9007199254740992.0)

    def randn(self):
        u, v = self.rand01(), self.rand01()
        return math.sqrt(-2 * math.log(max(u, 1e-300))) * math.cos(6.283185307179586 * v)


def params_init(c, seed_state):
    """params_init with g_rng = seed_state: the same values as the C trainer for the same seed."""
    r = _CRand(seed_state); z = c.sizes(); P = {f: np.zeros(z[f], np.float32) for f in FIELDS}
    d, T, D, N, Pd, K, pv = c.d, c.T, c.D, c.NODES, c.PDIM, c.K, c.p
    k_in = F32(1 / math.sqrt(F32(d))); k_out = F32(1 / math.sqrt(F32(T * (D + 1)))); k_u = F32(k_out / math.sqrt(F32(pv)))
    uni = lambda k: F32((2 * r.rand01() - 1) * float(k))
    for i in range(c.vocab * d): P["embed"][i] = F32(0.02 * r.randn())
    P["norm_f"][:] = 1
    base_pdim = 2 * c.n * c.groups + 2 * c.p + c.T
    for l in range(c.n_layer):
        P["norm"][l * d:(l + 1) * d] = 1
        P["conv"][l * K * d:l * K * d + d] = 1
        for row in range(c.node_off(l), c.node_off(l + 1)):
            for i in range(d): P["w_in"][row * d + i] = uni(k_in)
            P["bias"][row] = uni(k_in)
            for i in range(d): P["w_out"][row * d + i] = uni(k_out)
            P["A_log"][row] = F32(math.log(1 + 15 * r.rand01()))
        for i in range(base_pdim * d): P["proj"][l * Pd * d + i] = uni(k_in)
        for t in range(T):
            dt = max(math.exp(r.rand01() * (math.log(0.1) - math.log(0.001)) + math.log(0.001)), 1e-4)
            P["dt_bias"][l * T + t] = F32(dt + math.log(-math.expm1(-dt)))
        for i in range(T * pv * d): P["U"][l * T * pv * d + i] = uni(k_u)
        if (l & 1) and c.fff_gate: P["U"][l * T * pv * d:l * T * pv * d + c.n * d] = 0
    return P


# ----------------------------------------------------------------------------------------------- the model
class TSS:
    def __init__(self, c, P=None, device="cuda", seed=0, bf16_tape=None):
        c.check()
        if bf16_tape is None:   # the C build keeps the state tape in bf16 where its AVX-512 kernels run (k->ok, p = 16)
            bf16_tape = c.d % 64 == 0 and c.group % 64 == 0 and c.p == 16
        self.c, self.device, self.bf16_tape = c, torch.device(device), bf16_tape
        if P is None:
            P = params_init(c, (seed * 7919 + 88172645463325252) & 0xFFFFFFFFFFFFFFFF)
        self.P = {f: torch.as_tensor(np.asarray(P[f], np.float32)).to(self.device).contiguous() for f in FIELDS}
        self.G = {f: torch.zeros_like(self.P[f]) for f in FIELDS}
        self.hit = torch.zeros(c.rows, dtype=torch.bool, device=self.device)
        self.requantize()

    # -- checkpoints (nanotss_blt.c save / load)
    @classmethod
    def load(cls, path, device="cuda", **kw):
        with open(path, "rb") as f:
            magic = f.read(4); rs, = struct.unpack("<i", f.read(4))
            if magic not in (b"TFX1", b"TFW1", b"TSFK", b"TFP1", b"TSFM") or rs != 4:
                raise ValueError(f"{path}: not a float32 nanotss checkpoint")
            if magic == b"TSFM":   # LegacyCfg: no fw_mask / fff_gate
                v = struct.unpack("<10if", f.read(44)) + (0, 0)
            else:
                v = struct.unpack(Cfg.FMT, f.read(52))
            e = struct.unpack("<4i", f.read(16)) if magic == b"TFX1" else (1, 0, 0, 0)
            c = Cfg(*v[:10], temp=v[10], fw_mask=v[11], fff_gate=v[12], groups=e[0], ltc=e[1] & 15,
                    loops=max(1, e[1] >> 4), qknorm=e[2] & 1, rope=(e[2] >> 1) & 1, pos_feat=e[3] / 1e6)
            c.group = v[8]; c.walks = v[9]
            z = c.sizes(); P = {}
            for fld in FIELDS:
                P[fld] = np.frombuffer(f.read(4 * z[fld]), np.float32).copy()
                if P[fld].size != z[fld]: raise ValueError(f"{path}: truncated at {fld}")
        return cls(c, P, device=device, **kw)

    def save(self, path):
        c = self.c
        ext = c.groups != 1 or c.ltc or c.qknorm or c.rope or c.pos_feat > 0 or c.loops > 1
        with open(path, "wb") as f:
            f.write(b"TFX1" if ext else b"TFW1"); f.write(struct.pack("<i", 4))
            f.write(struct.pack(Cfg.FMT, c.vocab, c.d, c.n_layer, c.T, c.D, c.n, c.p, c.K, c.group, c.walks, c.temp, c.fw_mask, c.fff_gate))
            if ext:
                f.write(struct.pack("<4i", c.groups, c.ltc | (c.loops << 4 if c.loops > 1 else 0), c.qknorm | (c.rope << 1), int(c.pos_feat * 1e6 + .5)))
            for fld in FIELDS: f.write(self.P[fld].detach().cpu().numpy().astype(np.float32).tobytes())

    def requantize(self):
        c, P = self.c, self.P
        self.q = {f: QMat(P[f].view(-1, c.d), c.group) for f in ("w_in", "w_out", "proj", "U")}
        self.ea = torch.exp(P["A_log"].double()).float()        # decay_rates

    # -- recurrent state (gen / carry layout: per even layer, T*NODES rows of n x p; conv history newest first)
    def zero_state(self, B):
        c = self.c
        return {"h": [torch.zeros(B, c.T * c.NODES, c.n, c.p, device=self.device) for _ in range(0, c.n_layer, 2)],
                "hist": [torch.zeros(B, max(c.K - 1, 0), c.d, device=self.device) for _ in range(0, c.n_layer, 2)]}

    def state_from_c(self, hs, hh):
        """carry_h / carry_hist (gen_snapshot) arrays [B, S_all], [B, n_layer*(K-1)*d] -> state dict."""
        c = self.c; B = hs.shape[0]; S = c.T * c.NODES * c.n * c.p
        hs = torch.as_tensor(hs, device=self.device).view(B, -1); hh = torch.as_tensor(hh, device=self.device).view(B, c.n_layer, max(c.K - 1, 0), c.d)
        return {"h": [hs[:, (l // 2) * S:(l // 2 + 1) * S].reshape(B, c.T * c.NODES, c.n, c.p).clone() for l in range(0, c.n_layer, 2)],
                "hist": [hh[:, l].clone() for l in range(0, c.n_layer, 2)]}

    # -- forward
    def forward(self, x, state=None, lens=None, train=False, seeds=None, stream=False):
        """x [B, L, d] -> [B, L, d]. state is updated in place (the SSM states; the conv history only when
        stream=True, as gen_core does). With train=True, returns (out, tape) for backward()."""
        c = self.c; B, L, _ = x.shape
        if state is None: state = self.zero_state(B)
        lens = torch.full((B,), L, dtype=torch.long, device=self.device) if lens is None else torch.as_tensor(lens, device=self.device)
        draws = None
        if train:
            nd = sum(L * c.T * (c.D if not (l & 1) else c.D + 1) for l in range(c.n_layer))
            draws = _Draws(np.zeros(B, np.uint64) if seeds is None else seeds, max(nd, 1), self.device)
        valid = torch.arange(L, device=self.device)[None, :] < lens[:, None]
        tapes = []
        for l in range(c.n_layer):
            if l & 1:
                x, tp = self._fff_fwd(l, x, train, draws, valid, lens)
            else:
                x, tp = self._ssm_fwd(l, x, state["h"][l // 2], state["hist"][l // 2], train, draws, valid, lens)
                if stream and c.K > 1:
                    state["hist"][l // 2] = tp.pop("ext").flip(1)[:, :c.K - 1].contiguous()
                else:
                    tp.pop("ext", None)
            tapes.append(tp)
        return (x, {"layers": tapes, "valid": valid}) if train else x

    def _ssm_fwd(self, l, x, h, hist, train, draws, valid, lens):
        c, P = self.c, self.P
        d, T, D, N, n, pv, K, G, Pd = c.d, c.T, c.D, c.NODES, c.n, c.p, c.K, c.groups, c.PDIM
        D1, TN, B, L = D + 1, c.T * c.NODES, x.shape[0], x.shape[1]
        dev = x.device; r0 = c.node_off(l)
        norm = P["norm"][l * d:(l + 1) * d]; conv = P["conv"][l * K * d:(l + 1) * K * d].view(K, d)
        u, rn = rmsnorm(x, norm)
        ext = torch.cat([hist.flip(1), u], 1) if K > 1 else u
        mx = conv[0] * u
        for j in range(1, K):
            mx = mx + conv[j] * ext[:, K - 1 - j:K - 1 - j + L]
        x8, c8, xs = quant8(mx)
        p = self.q["proj"].rows(l * Pd, (l + 1) * Pd).dot8(c8, xs)
        inv = None
        if c.qknorm:
            bc = p[..., :2 * G * n].reshape(B, L, 2, G, n)
            inv = (1.0 / torch.sqrt(((bc * bc).sum(-1) + F32(1e-6)).double())).float()
            p = torch.cat([(bc * inv[..., None]).reshape(B, L, 2 * G * n), p[..., 2 * G * n:]], -1)
        zdt = p[..., 2 * G * n + 2 * pv:] + P["dt_bias"][l * T:(l + 1) * T]
        dt = softplus(zdt)
        grp = torch.tensor([tr * G // T for tr in range(T)], device=dev)
        Bt = p[..., :G * n].reshape(B, L, G, n)[:, :, grp]; Ct = p[..., G * n:2 * G * n].reshape(B, L, G, n)[:, :, grp]
        V = p[..., 2 * G * n:2 * G * n + pv]; Z = p[..., 2 * G * n + pv:2 * G * n + 2 * pv]
        qin = self.q["w_in"].rows(r0, r0 + TN)
        s_lin = qin.dot8(c8, xs)
        bias = P["bias"][r0:r0 + TN]; ea = self.ea[r0:r0 + TN]
        node_t = torch.empty(B, L, T, D1, dtype=torch.long, device=dev)
        s_t = torch.empty(B, L, T, D1, device=dev); dec_t = torch.empty_like(s_t); R_t = torch.empty(B, L, T, D1, pv, device=dev)
        rs = torch.empty(B, L, T, pv, device=dev)
        hprev = torch.empty(B, L, T, D1, n, pv, device=dev, dtype=torch.bfloat16 if self.bf16_tape else torch.float32) if train else None
        AL = D * (D + 1) // 2
        if train:
            anode = torch.zeros(B, L, T, max(AL, 1), dtype=torch.long, device=dev)
            as_t = torch.zeros(B, L, T, max(AL, 1), device=dev); aR_t = torch.zeros(B, L, T, max(AL, 1), pv, device=dev)
            bw_t = torch.zeros(B, L, T, max(D, 1), device=dev)
            base = [sum(D - k for k in range(kk)) for kk in range(D)]
            sidx = (torch.arange(T, device=dev)[:, None] * D + torch.arange(D, device=dev)[None, :])   # [T, D]
        bi = torch.arange(B, device=dev)[:, None]; trN = torch.arange(T, device=dev) * N
        cap = c.dec_cap
        for t in range(L):
            node = torch.zeros(B, T, dtype=torch.long, device=dev)
            dtt, Btt, Ctt, Vt, Zt, sl = dt[:, t], Bt[:, t], Ct[:, t], V[:, t], Z[:, t], s_lin[:, t]
            gB = dtt[..., None] * Btt
            Racc = torch.zeros(B, T, pv, device=dev)
            for kk in range(D1):
                row = trN + node
                ha = h[bi, row]
                dec = exp_d(-(ea[row] * dtt))
                if cap < 1: dec = dec.clamp(max=cap)
                if train: hprev[:, t, :, kk] = ha
                hv = ha * dec[..., None, None]
                R = (Ctt[..., None] * hv).sum(-2)
                h[bi, row] = hv + gB[..., None] * Vt[:, None, None, :]
                s = (sl.gather(1, row) + bias[row]) + (R * Zt[:, None, :]).sum(-1)
                Racc = Racc + R
                node_t[:, t, :, kk] = row; s_t[:, t, :, kk] = s; dec_t[:, t, :, kk] = dec; R_t[:, t, :, kk] = R
                node = 2 * node + 1 + (s > 0).long()
            rs[:, t] = Racc
            if train and D > 0:
                st = s_t[:, t, :, :D]
                sg = sigm(st.abs() / F32(c.temp)); w = sg * (1 - sg)
                tot = torch.zeros(B, T, device=dev)
                for kk in range(D): tot = tot + w[..., kk]
                if c.walks <= 0:
                    bw = torch.ones_like(w)
                else:
                    q = torch.where(tot[..., None] > 0, (F32(c.walks) * w / tot[..., None]).clamp(max=1), torch.ones_like(w))
                    uu = draws.take((t * T * D + sidx)[None].expand(B, T, D))
                    bw = torch.where(uu < q, 1 / q, torch.zeros_like(q))
                bw_t[:, t] = bw
                a = 2 * (node_t[:, t, :, :D] - trN[:, None]) + 2 - (st > 0).long()
                for j in range(1, D1):
                    rowa = trN[:, None] + a[..., :j]
                    hr = h[bi[..., None], rowa]
                    deca = exp_d(-(ea[rowa] * dtt[..., None]))
                    aR = (Ctt[:, :, None, :, None] * hr).sum(-2) * deca[..., None]
                    asv = (sl.gather(1, rowa.reshape(B, -1)).view(B, T, j) + bias[rowa]) + (aR * Zt[:, None, None, :]).sum(-1)
                    ai = torch.tensor([base[kk] + j - kk - 1 for kk in range(j)], device=dev)
                    anode[:, t][..., ai] = rowa; as_t[:, t][..., ai] = asv; aR_t[:, t][:, :, ai] = aR
                    a = torch.cat([2 * a[..., :j] + 1 + (asv > 0).long(), a[..., j:]], -1)
        if train and D > 0: draws.off += lens * T * D
        gl, dgl = gelu_rec(s_t)
        A_out = torch.zeros(B, L, TN, device=dev).scatter_(2, node_t.view(B, L, -1), gl.view(B, L, -1))
        y = A_out @ self.q["w_out"].Q[r0:r0 + TN] + rs.reshape(B, L, T * pv) @ self.q["U"].Q[l * T * pv:(l + 1) * T * pv]
        tp = {"ext": ext}
        if train:
            tp.update(kind="ssm", xin=x, u=u, rn=rn, x8=x8, p=p, inv=inv, zdt=zdt, dt=dt, Bt=Bt, Ct=Ct, V=V, Z=Z, grp=grp,
                      node=node_t, s=s_t, dec=dec_t, R=R_t, rs=rs, hprev=hprev, gl=gl, dgl=dgl, anode=anode, as_=as_t, aR=aR_t, bw=bw_t)
        return x + y, tp

    def _fff_fwd(self, l, x, train, draws, valid, lens):
        c, P = self.c, self.P
        d, T, Df, N, n = c.d, c.T, c.D + 1, c.FNODES, c.n
        B, L, dev = x.shape[0], x.shape[1], x.device
        r0 = c.node_off(l); TN = T * N; Pd = c.PDIM
        u, rn = rmsnorm(x, P["norm"][l * d:(l + 1) * d])
        x8, c8, xs = quant8(u)
        s_lin = self.q["w_in"].rows(r0, r0 + TN).dot8(c8, xs)
        bias = P["bias"][r0:r0 + TN]
        trN = torch.arange(T, device=dev) * N
        a = torch.zeros(B, L, T, dtype=torch.long, device=dev)
        node = torch.full((B, L, T, Df + 1), -1, dtype=torch.long, device=dev); s_t = torch.zeros(B, L, T, Df + 1, device=dev)
        for j in range(Df + 1):
            ok = a < N
            row = trN + a.clamp(max=N - 1)
            s = s_lin.gather(2, row) + bias[row]
            node[..., j] = torch.where(ok, row, -1); s_t[..., j] = torch.where(ok, s, torch.zeros_like(s))
            a = torch.where(ok, 2 * a + 1 + (s > 0).long(), a)
        gl, dgl = gelu_rec(s_t)
        okn = node >= 0
        A_out = torch.zeros(B, L, TN, device=dev).scatter_add_(2, node.clamp(min=0).view(B, L, -1), torch.where(okn, gl, torch.zeros_like(gl)).view(B, L, -1))
        y = A_out @ self.q["w_out"].Q[r0:r0 + TN]
        tp = {}
        if c.fff_gate:
            rp, ru = l * Pd, l * T * c.p
            low = self.q["proj"].rows(rp, rp + n).dotf(x8)
            gg, dg = gelu_rec(low)
            gate = 2 * sigm(gg @ self.q["U"].Q[ru:ru + n])
            tp.update(raw=y, gate=gate, gg=gg, dg=dg)
            y = y * gate
        if train:
            AL = Df * (Df + 1) // 2
            loc = node[..., :Df] - trN[:, None]
            child = okn[..., :Df] & (2 * loc + 1 < N)
            sg = sigm(s_t[..., :Df].abs() / F32(c.temp)); w = torch.where(child, sg * (1 - sg), torch.zeros_like(sg))
            tot = torch.zeros(B, L, T, device=dev)
            for j in range(Df): tot = tot + w[..., j]
            if c.walks <= 0:
                bw = child.float()
            else:
                q = torch.where(tot[..., None] > 0, (F32(c.walks) * w / tot[..., None]).clamp(max=1), torch.ones_like(w))
                take = child & valid[:, :, None, None]
                idx = take.view(B, -1).long().cumsum(1) - 1
                uu = draws.take(idx.clamp(min=0)).view(B, L, T, Df)
                bw = torch.where(child & (uu < q), 1 / q, torch.zeros_like(q))
                draws.off += take.view(B, -1).sum(1)
            base = [sum(Df - k for k in range(kk)) for kk in range(Df)]
            anode = torch.full((B, L, T, AL), -1, dtype=torch.long, device=dev); as_t = torch.zeros(B, L, T, AL, device=dev)
            aa = 2 * loc + 2 - (s_t[..., :Df] > 0).long()
            alive = child.clone()
            for jj in range(1, Df + 1):
                ak = aa[..., :jj]; ok = alive[..., :jj] & (ak < N)
                rowa = trN[:, None] + ak.clamp(min=0, max=N - 1)
                asv = s_lin.gather(2, rowa.view(B, L, -1)).view(B, L, T, jj) + bias[rowa]
                ai = torch.tensor([base[kk] + jj - kk - 1 for kk in range(jj)], device=dev)
                anode[..., ai] = torch.where(ok, rowa, -1); as_t[..., ai] = torch.where(ok, asv, torch.zeros_like(asv))
                alive = torch.cat([ok, alive[..., jj:]], -1)
                aa = torch.cat([torch.where(ok, 2 * ak + 1 + (asv > 0).long(), ak), aa[..., jj:]], -1)
            tp.update(kind="fff", xin=x, rn=rn, x8=x8, node=node, s=s_t, gl=gl, dgl=dgl, anode=anode, as_=as_t, bw=bw)
        return x + y, tp

    # -- backward (layer_bwd / fff_bwd); gradients summed into self.G, visited rows into self.hit
    def zero_grad(self):
        for f in FIELDS: self.G[f].zero_()
        self.hit.zero_()

    def backward(self, tape, g):
        """g: dLoss/d(output) [B, L, d]. Returns dLoss/d(input)."""
        valid = tape["valid"]
        for l in reversed(range(self.c.n_layer)):
            tp = tape["layers"][l]
            g = self._fff_bwd(l, tp, g, valid) if l & 1 else self._ssm_bwd(l, tp, g, valid)
        return g

    def _credit(self, s, bw, on, altv, D, Dlv):
        """sampled routing credit added to ds at levels 0..D-1 (Dlv: levels of the on-path walk)."""
        c = self.c
        suf = torch.zeros_like(on)
        for j in range(Dlv - 1, 0, -1): suf[..., j - 1] = suf[..., j] + on[..., j]
        onk = suf[..., :D]
        kk_of = torch.tensor([kk for kk in range(D) for _ in range(D - kk)], device=on.device)
        alt = torch.zeros_like(onk).index_add_(-1, kk_of, altv)
        sv = s[..., :D]; sg = sigm(sv.abs() / F32(c.temp))
        return bw * torch.where(sv > 0, 1.0, -1.0) * (onk - alt) * sg * (1 - sg) / F32(c.temp)

    def _ssm_bwd(self, l, tp, g, valid):
        c, P, Gr = self.c, self.P, self.G
        d, T, D, N, n, pv, K, G, Pd = c.d, c.T, c.D, c.NODES, c.n, c.p, c.K, c.groups, c.PDIM
        D1, TN, B, L = D + 1, c.T * c.NODES, g.shape[0], g.shape[1]
        dev = g.device; r0 = c.node_off(l); rU = l * T * pv
        node, R, gl, dgl, s = tp["node"], tp["R"], tp["gl"], tp["dgl"], tp["s"]
        qU = self.q["U"].rows(rU, rU + T * pv); qout = self.q["w_out"].rows(r0, r0 + TN)
        drs = qU.dotf(g).view(B, L, T, pv)
        Gr["U"].view(-1, d)[rU:rU + T * pv] += tp["rs"].reshape(-1, T * pv).T @ g.reshape(-1, d)
        GO = qout.dotf(g)
        go = GO.gather(2, node.view(B, L, -1)).view(B, L, T, D1)
        on = gl * go + (drs[:, :, :, None, :] * R).sum(-1)
        ds = dgl * go
        if D > 0:
            an = tp["anode"]; ag, _ = gelu_rec(tp["as_"])
            altv = ag * GO.gather(2, an.view(B, L, -1)).view(an.shape) + (drs[:, :, :, None, :] * tp["aR"]).sum(-1)
            ds = torch.cat([ds[..., :D] + self._credit(s, tp["bw"], on, altv, D, D1), ds[..., D:]], -1)
        Z = tp["Z"]
        dZ = (ds[..., None] * R).sum((2, 3))
        dR = drs[:, :, :, None, :] + ds[..., None] * Z[:, :, None, None, :]
        nf = node.view(B, L, -1)
        DS = torch.zeros(B, L, TN, device=dev).scatter_(2, nf, ds.view(B, L, -1))
        A_out = torch.zeros(B, L, TN, device=dev).scatter_(2, nf, gl.view(B, L, -1))
        dx = DS @ self.q["w_in"].Q[r0:r0 + TN]
        Gr["bias"][r0:r0 + TN] += DS.sum((0, 1))
        Gr["w_out"].view(-1, d)[r0:r0 + TN] += torch.einsum("blr,bld->rd", A_out, g)
        Gr["w_in"].view(-1, d)[r0:r0 + TN] += torch.einsum("blr,bld->rd", DS, tp["x8"])
        vis = torch.zeros(B, L, TN, dtype=torch.bool, device=dev).scatter_(2, nf, True) & valid[:, :, None]
        self.hit[r0:r0 + TN] |= vis.any(1).any(0)
        # recurrent adjoints, reverse time
        lam = torch.zeros(B, TN, n, pv, device=dev)
        Bt, Ct, V, dt, dec_t, hprev = tp["Bt"], tp["Ct"], tp["V"], tp["dt"], tp["dec"], tp["hprev"]
        eaN = self.ea[r0:r0 + TN][node]
        dBt = torch.zeros(B, L, T, n, device=dev); dCt = torch.zeros_like(dBt); dV = torch.zeros(B, L, pv, device=dev)
        ddt = torch.zeros(B, L, T, device=dev); GA = torch.zeros(B, L, T, D1, device=dev)
        bi = torch.arange(B, device=dev)[:, None]
        for t in reversed(range(L)):
            rows = node[:, t].reshape(B, -1)
            la = lam[bi, rows].view(B, T, D1, n, pv)
            hp = hprev[:, t].float(); dec = dec_t[:, t]; dRv = dR[:, t]
            Btt, Ctt, Vt, gv = Bt[:, t], Ct[:, t], V[:, t], dt[:, t]
            laV = (la * Vt[:, None, None, None, :]).sum(-1)
            dBt[:, t] = gv[..., None] * laV.sum(2)
            dV[:, t] = (gv[..., None, None] * (la * Btt[:, :, None, :, None]).sum(-2)).sum((1, 2))
            dgain = (laV * Btt[:, :, None, :]).sum(-1)
            dCt[:, t] = (dec[..., None] * (hp * dRv[:, :, :, None, :]).sum(-1)).sum(2)
            dht = la + Ctt[:, :, None, :, None] * dRv[:, :, :, None, :]
            ddec = (dht * hp).sum((-1, -2))
            lam[bi, rows] = (dec[..., None, None] * dht).view(B, T * D1, n, pv)
            e = eaN[:, t]
            ddt[:, t] = dgain.sum(-1) + (ddec * dec * (-e)).sum(-1)
            GA[:, t] = ddec * dec * (-e) * gv[..., None]
        Gr["A_log"][r0:r0 + TN] += torch.zeros(TN, device=dev).index_add_(0, node.reshape(-1), GA.reshape(-1))
        # dense projections, convolution, norm
        zdt = tp["zdt"]
        dz = ddt * torch.where(zdt > 20, torch.ones_like(zdt), sigm(zdt))
        Gr["dt_bias"][l * T:(l + 1) * T] += dz.sum((0, 1))
        grp = tp["grp"]
        dBg = torch.zeros(B, L, G, n, device=dev).index_add_(2, grp, dBt); dCg = torch.zeros(B, L, G, n, device=dev).index_add_(2, grp, dCt)
        dbc = torch.cat([dBg.reshape(B, L, G * n), dCg.reshape(B, L, G * n)], -1)
        if c.qknorm:
            uu = tp["p"][..., :2 * G * n].reshape(B, L, 2, G, n); dv = dbc.view(B, L, 2, G, n)
            dbc = (tp["inv"][..., None] * (dv - uu * (uu * dv).sum(-1, keepdim=True))).reshape(B, L, 2 * G * n)
        dp = torch.cat([dbc, dV, dZ, dz], -1)
        Gr["proj"].view(-1, d)[l * Pd:(l + 1) * Pd] += torch.einsum("blp,bld->pd", dp, tp["x8"])
        dx = dx + dp @ self.q["proj"].Q[l * Pd:(l + 1) * Pd]
        u = tp["u"]; conv = P["conv"][l * K * d:(l + 1) * K * d].view(K, d)
        du = torch.zeros_like(u); gc = Gr["conv"][l * K * d:(l + 1) * K * d].view(K, d)
        for j in range(K):
            if j >= L: break
            gc[j] += (dx[:, j:] * u[:, :L - j]).reshape(-1, d).sum(0)
            du[:, :L - j] += conv[j] * dx[:, j:]
        dxn, dw = rmsnorm_bwd(tp["xin"], P["norm"][l * d:(l + 1) * d], tp["rn"], du)
        Gr["norm"][l * d:(l + 1) * d] += dw
        return g + dxn

    def _fff_bwd(self, l, tp, g, valid):
        c, P, Gr = self.c, self.P, self.G
        d, T, Df, N, n = c.d, c.T, c.D + 1, c.FNODES, c.n
        B, L, dev = g.shape[0], g.shape[1], g.device
        r0 = c.node_off(l); TN = T * N; Pd = c.PDIM
        x8 = tp["x8"]; dx = torch.zeros_like(g); gy = g
        if c.fff_gate:
            rp, ru = l * Pd, l * T * c.p
            gate, raw = tp["gate"], tp["raw"]
            gy = g * gate
            dz = g * raw * gate * (1 - gate / 2)
            dl = self.q["U"].rows(ru, ru + n).dotf(dz) * tp["dg"]
            Gr["U"].view(-1, d)[ru:ru + n] += tp["gg"].reshape(-1, n).T @ dz.reshape(-1, d)
            Gr["proj"].view(-1, d)[rp:rp + n] += dl.reshape(-1, n).T @ x8.reshape(-1, d)
            dx = dl @ self.q["proj"].Q[rp:rp + n]
        node, s, gl, dgl = tp["node"], tp["s"], tp["gl"], tp["dgl"]
        okn = node >= 0; nf = node.clamp(min=0)
        GO = self.q["w_out"].rows(r0, r0 + TN).dotf(gy)
        go = torch.where(okn, GO.gather(2, nf.view(B, L, -1)).view(node.shape), torch.zeros_like(gl))
        on = gl * go; ds = dgl * go
        an = tp["anode"]; ag, _ = gelu_rec(tp["as_"])
        altv = torch.where(an >= 0, ag * GO.gather(2, an.clamp(min=0).view(B, L, -1)).view(an.shape), torch.zeros_like(ag))
        ds = torch.cat([ds[..., :Df] + self._credit(s, tp["bw"], on, altv, Df, Df + 1), ds[..., Df:]], -1)
        ds = torch.where(okn, ds, torch.zeros_like(ds))
        DS = torch.zeros(B, L, TN, device=dev).scatter_add_(2, nf.view(B, L, -1), ds.view(B, L, -1))
        A_out = torch.zeros(B, L, TN, device=dev).scatter_add_(2, nf.view(B, L, -1), torch.where(okn, gl, torch.zeros_like(gl)).view(B, L, -1))
        Gr["w_out"].view(-1, d)[r0:r0 + TN] += torch.einsum("blr,bld->rd", A_out, gy)
        Gr["w_in"].view(-1, d)[r0:r0 + TN] += torch.einsum("blr,bld->rd", DS, x8)
        Gr["bias"][r0:r0 + TN] += DS.sum((0, 1))
        dx = dx + DS @ self.q["w_in"].Q[r0:r0 + TN]
        vis = (torch.zeros(B, L, TN, dtype=torch.long, device=dev).scatter_add_(2, nf.view(B, L, -1), okn.view(B, L, -1).long()) > 0) & valid[:, :, None]
        self.hit[r0:r0 + TN] |= vis.any(1).any(0)
        dxn, dw = rmsnorm_bwd(tp["xin"], P["norm"][l * d:(l + 1) * d], tp["rn"], dx)
        Gr["norm"][l * d:(l + 1) * d] += dw
        return g + dxn


# ----------------------------------------------------------------------------------------------- the update (tss_train_step after the reduction)
def adam_(p, g, m, v, lr, b1, b2, eps, wd, scale, c1, c2):
    gi = g * scale
    m.mul_(F32(b1)).add_(F32(1 - F32(b1)) * gi)
    v.mul_(F32(b2)).add_(F32(1 - F32(b2)) * gi * gi)
    upd = float(F32(lr)) * ((F32(wd) * p).double() + (m / F32(c1)).double() / (torch.sqrt((v / F32(c2)).double()) + float(F32(eps))))
    p.sub_(upd.float())


class Optimizer:
    """tss_train_step's update with byte_muon on (as ppo_tss.c runs it): clip, Adam, blockwise Muon, requantise."""
    def __init__(self, model, lr=3e-4, beta1=.9, beta2=.999, eps=1e-8, weight_decay=0, clip=1, muon=True,
                 muon_scale=3.0, muon_steps=5, muon_block=16, normuon=False):
        self.m, self.lr, self.b1, self.b2, self.eps, self.wd, self.clip = model, lr, beta1, beta2, eps, weight_decay, clip
        self.muon, self.muon_scale, self.muon_steps, self.blk, self.normuon = muon, muon_scale, muon_steps, muon_block, normuon
        self.step_n = 0
        self.M = {f: torch.zeros_like(t) for f, t in model.P.items()}
        self.V = {f: torch.zeros_like(t) for f, t in model.P.items()}
        if normuon: raise NotImplementedError("normuon")
        self._blocks()

    def _blocks(self):
        """Muon jobs as byte_muon_update builds them: per layer and tree, node rows in blocks; proj and U rows
        of each layer (odd layers: the gate's n rows) in blocks. Index tensors padded with -1."""
        c, dev, b = self.m.c, self.m.device, self.blk
        jobs = {"w_in": [], "w_out": [], "proj": [], "U": []}
        for l in range(c.n_layer):
            nn_ = c.FNODES if l & 1 else c.NODES
            for tr in range(c.T):
                r = c.node_off(l) + tr * nn_
                for at in range(0, nn_, b):
                    rr = list(range(r + at, r + min(at + b, nn_)))
                    jobs["w_in"].append(rr); jobs["w_out"].append(rr)
            for f, nrow in (("proj", c.PDIM), ("U", c.T * c.p)):
                active = (c.n if c.fff_gate else 0) if l & 1 else nrow
                for at in range(0, active, b):
                    jobs[f].append(list(range(l * nrow + at, l * nrow + min(at + b, active))))
        self.jobs = {f: torch.tensor([j + [-1] * (b - len(j)) for j in js], dtype=torch.long, device=dev).view(-1, b) for f, js in jobs.items() if js}

    def step(self, scored):
        """Apply the summed gradients in model.G (scored = number of scored positions). Returns the gradient norm."""
        m, c = self.m, self.m.c
        self.step_n += 1; st = self.step_n
        G, P, hit = m.G, m.P, m.hit
        nrm = sum((G[f].double() ** 2).sum() for f in FIELDS).item()
        inv = F32(1 / scored) if scored else F32(0)
        norm = math.sqrt(nrm) * float(inv)
        scale = inv
        if self.clip > 0 and norm > self.clip: scale = F32(scale * (self.clip / (norm + float(F32(1e-6)))))
        c1, c2 = F32(1 - math.pow(float(F32(self.b1)), float(F32(st)))), F32(1 - math.pow(float(F32(self.b2)), float(F32(st))))
        args = (self.lr, self.b1, self.b2, self.eps, self.wd, scale, c1, c2)
        rows = hit.nonzero().squeeze(1)
        muon_f = ("w_in", "w_out", "proj", "U") if self.muon else ()
        for f in FIELDS:
            if f in muon_f: continue
            if f in NODE_FIELDS:
                w = c.d if f in ("w_in", "w_out") else 1
                p, g, mm, vv = (t.view(-1, w)[rows] for t in (P[f], G[f], self.M[f], self.V[f]))
                adam_(p, g, mm, vv, *args)
                P[f].view(-1, w)[rows] = p; self.M[f].view(-1, w)[rows] = mm; self.V[f].view(-1, w)[rows] = vv
            else:
                adam_(P[f], G[f], self.M[f], self.V[f], *args)
        for f in muon_f:
            if f in self.jobs: self._muon(f, self.jobs[f], hit if f in ("w_in", "w_out") else None, scale)
        m.zero_grad()
        m.requantize()
        return norm

    def _muon(self, f, idx, hit, scale):
        d = self.m.c.d
        P, G, M = self.m.P[f].view(-1, d), self.m.G[f].view(-1, d), self.M[f].view(-1, d)
        pad = idx < 0; ix = idx.clamp(min=0)
        act = ~pad if hit is None else (~pad & hit[ix])
        g = G[ix] * scale
        mom = torch.where(act[..., None], F32(.95) * M[ix] + F32(.05) * g, M[ix])
        X = torch.where(act[..., None], F32(.95) * mom + F32(.05) * g, torch.zeros_like(g))
        sel = ix[act]; M[sel] = mom[act]
        norm = (X.double() ** 2).sum((1, 2))
        X = X * (1 / (torch.sqrt(norm) + 1e-7)).float()[:, None, None]
        for _ in range(self.muon_steps):
            A = X @ X.transpose(1, 2)
            Bm = F32(-4.775) * A + F32(2.0315) * (A @ A.transpose(1, 2))
            X = F32(3.4445) * X + Bm @ X
        r = act.sum(1)
        lr = F32(F32(self.lr) * F32(self.muon_scale)) * (0.2 * torch.sqrt(torch.clamp(r, min=d).double())).float()
        newp = P[ix] - ((F32(self.lr) * F32(self.wd)) * P[ix] + lr[:, None, None] * X)
        P[sel] = newp[act]
