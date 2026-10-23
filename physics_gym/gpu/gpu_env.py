"""gpu_env.py: the body environments as batches of torch tensors.

  GpuPool  every environment on the GPU, one CUDA thread each, running body.c / env.c compiled as device code
           (tssenv_kernels.cu -> libtssenv_cuda.so). Observations, actions, rewards and resets stay on the device.
  CpuPool  the same interface over libtssenv.so (the CPU build, 8 worlds per AVX-512 group): the reference.

Both set tasks up as ppo_tss.c does (same flags, same defaults) and reset finished environments with the same
seeds (seed*7919+1 + episode*7777 + env, episodes counted in environment order), so given the same actions the
two produce the same episodes; parity_env.py measures how closely.
"""
import ctypes, os, subprocess, struct
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BODY = os.path.dirname(HERE)
U64 = 0xFFFFFFFFFFFFFFFF
TASKS = {"balance": 0, "imitate": 1, "crawl": 2, "getup": 3, "goto": 4}


def load_mref(path):
    """MREF clips as mref_load reads them: root (frames x 3) and unit quaternions (frames x 14 x 4), in double."""
    raw = open(path, "rb").read()
    assert raw[:4] == b"MREF", path
    n, = struct.unpack("<i", raw[4:8]); o = 8; clips = {}
    for _ in range(n):
        name = raw[o:o + 16].split(b"\0")[0].decode(); frames, = struct.unpack("<i", raw[o + 16:o + 20]); fps, = struct.unpack("<f", raw[o + 20:o + 24]); o += 24
        v = np.frombuffer(raw, np.float32, frames * (3 + 4 * 14), o).reshape(frames, 3 + 4 * 14).astype(np.float64); o += v.nbytes // 2
        q = v[:, 3:].reshape(frames, 14, 4)
        nq = np.sqrt(q[..., 0] * q[..., 0] + q[..., 1] * q[..., 1] + q[..., 2] * q[..., 2] + q[..., 3] * q[..., 3])
        clips[name] = dict(frames=frames, fps=fps, root=np.ascontiguousarray(v[:, :3]), q=np.ascontiguousarray(q / nq[..., None]))
    return clips


def load_room(path):
    """room.json -> float64 [n, 13]: c, h, q (w x y z), mu, soft, grip per box, as body.c room_set takes them"""
    import json
    if not path: return np.zeros((0, 13))
    bx = json.load(open(path))["boxes"]
    return np.array([b["c"] + b["h"] + b["q"] + [b["mu"], b["soft"], b.get("grip", 0)] for b in bx], np.float64).reshape(-1, 13)


def task_setup(task, mref, style, assist, push, act_mul, steps, style_start, sway, arm_tone, goal_cone=0.0):
    """tssenv.c pool_new / ppo_tss.c: the TaskCfg values and the clips they name."""
    t = dict(task=0, clip=None, push_max=push, sway=sway, arm_tone=arm_tone, push_every=2.0, steps=300, init_noise=0.05,
             fixed_start=0, assist=assist, act_mul=act_mul, style=[], style_start=style_start, style_file=None, goal_cone=goal_cone)
    if task in ("crawl", "getup"):
        t["task"] = TASKS[task]; t["act_mul"] = act_mul or 2.5; t["steps"] = 300 if task == "crawl" else 240
        if style:
            f, names = style.split(":", 1); t["style_file"] = f; t["style"] = names.split(",")[:8]
    elif task == "goto":   # physics_gym: the walk clip's cycle is the phase input; walking poses are the style
        t["task"] = TASKS["goto"]; t["clip"] = "walk"; t["clip_file"] = mref; t["steps"] = 600
        if style: f, names = style.split(":", 1); t["style_file"] = f; t["style"] = names.split(",")[:8]
        else: t["style_file"] = mref; t["style"] = ["walk"]
    elif task != "balance":
        t["task"] = TASKS["imitate"]; t["clip"] = task; t["init_noise"] = 0.0; t["clip_file"] = mref
    if steps > 0: t["steps"] = steps
    return t


def _build_cuda(out):
    """nvcc from PATH, or the CUDA 13 compiler wheels (nvidia-cuda-nvcc, -crt, -nvvm, -runtime) fetched with pip."""
    nvcc = "nvcc"
    if subprocess.run(["which", "nvcc"], capture_output=True).returncode:
        d = os.path.join(HERE, ".cuda"); root = os.path.join(d, "nvidia", "cu13")
        if not os.path.exists(os.path.join(root, "bin", "nvcc")):
            os.makedirs(d, exist_ok=True)
            subprocess.check_call(["pip", "download", "-q", "--no-deps", "-d", d, "nvidia-cuda-nvcc", "nvidia-cuda-crt", "nvidia-nvvm", "nvidia-cuda-runtime"])
            for w in os.listdir(d):
                if w.endswith(".whl"): subprocess.check_call(["python3", "-m", "zipfile", "-e", os.path.join(d, w), d])
            for b in ("nvcc", "cudafe++", "ptxas", "fatbinary", "nvlink", "bin2c"): os.chmod(os.path.join(root, "bin", b), 0o755)
            os.chmod(os.path.join(root, "nvvm", "bin", "cicc"), 0o755)
        nvcc = os.path.join(root, "bin", "nvcc"); inc = ["-I" + os.path.join(root, "include"), "-L" + os.path.join(root, "lib")]
    else:
        inc = []
    cap = torch.cuda.get_device_capability(); arch = f"sm_{cap[0]}{cap[1]}"
    subprocess.check_call([nvcc, "-O3", "--fmad=false", "--expt-relaxed-constexpr", "-arch=" + arch, "-shared", "-Xcompiler", "-fPIC", *inc,
                           "-o", out, os.path.join(HERE, "tssenv_kernels.cu")])


_cuda = None
def cuda_lib():
    global _cuda
    if _cuda is None:
        out = os.path.join(HERE, "libtssenv_cuda.so")
        srcs = [os.path.join(HERE, "tssenv_kernels.cu")] + [os.path.join(BODY, f) for f in ("body.c", "body.h", "bmath.h", "joint_limits.h", "env.c", "env.h", "mref.c", "mref.h")]
        if not os.path.exists(out) or os.path.getmtime(out) < max(map(os.path.getmtime, srcs)): _cuda_build_once(out)
        L = ctypes.CDLL(out); vp, i, d = ctypes.c_void_p, ctypes.c_int, ctypes.c_double
        L.tss_init.restype = ctypes.c_char_p; L.tss_error.restype = ctypes.c_char_p; L.tss_set_style.restype = ctypes.c_char_p
        L.tss_make_clip.argtypes = [vp, ctypes.c_char_p, i, ctypes.c_float, vp, vp]
        L.tss_make_cfg.argtypes = [vp, i, vp, d, d, d, d, i, d, i, d, d, vp, i, d, d]
        L.tss_set_style.argtypes = [vp, vp, vp, i]
        L.tss_reset.argtypes = [vp, vp, i, vp, vp, vp]; L.tss_obs.argtypes = [vp, i, vp, vp]
        L.tss_set_room.restype = ctypes.c_char_p; L.tss_set_room.argtypes = [vp, i]
        L.tss_step.argtypes = [vp, i, vp, vp, vp, vp, vp, vp, vp]; L.tss_urand.argtypes = [vp, i, i, vp, vp]
        err = L.tss_init()
        if err: raise RuntimeError(err.decode())
        _cuda = L
    return _cuda


def _cuda_build_once(out):
    tmp = out + f".{os.getpid()}"; _build_cuda(tmp); os.replace(tmp, out)


def _dims(L):
    o, a, h, f = (ctypes.c_int() for _ in range(4)); L.tss_dims(*(ctypes.byref(x) for x in (o, a, h, f))); return o.value, a.value, h.value, f.value


def _ptr(t): return ctypes.c_void_p(t.data_ptr())


def urand_cuda(seeds, n, device):
    """the routing-credit stream on the GPU: float32 [B, n]"""
    L = cuda_lib(); s = torch.as_tensor(np.asarray(seeds, np.uint64).view(np.int64), device=device)
    out = torch.empty(len(s), n, device=device)
    L.tss_urand(_ptr(s), len(s), n, _ptr(out), ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream))
    return out


class _Pool:
    def _init_common(self, n, seed):
        self.n = n; self.erng = (seed * 7919 + 1) & U64
        self.episodes = n   # each initial reset counts as an episode (ppo_tss.c)


class GpuPool(_Pool):
    def __init__(self, n, task="balance", mref=os.path.join(BODY, "data/motions_ref.bin"), style="", assist=0, push=0, act_mul=0, steps=0,
                 style_start=0, sway=0, arm_tone=0, goal_cone=0, seed=0, device="cuda", room=None):
        self.L = L = cuda_lib(); self.dev = torch.device(device); self._init_common(n, seed)
        self.room = load_room(room); rb = np.ascontiguousarray(self.room)   # one room per process, as in body.c
        err = L.tss_set_room(rb.ctypes.data_as(ctypes.c_void_p), len(rb))
        if err: raise RuntimeError(err.decode())
        self.OBS, self.ACT, self.HZ, self.FEAT = _dims(L)
        t = task_setup(task, mref, style, assist, push, act_mul, steps, style_start, sway, arm_tone, goal_cone); self.steps = t["steps"]
        self._keep = []
        def dclip(c, name):
            root = torch.tensor(c["root"], device=self.dev); q = torch.tensor(c["q"], device=self.dev)
            buf = torch.empty(L.tss_clip_size(), dtype=torch.uint8, device=self.dev)
            L.tss_make_clip(_ptr(buf), name.encode(), c["frames"], c["fps"], _ptr(root), _ptr(q)); self._keep += [root, q, buf]; return buf
        clip = None
        if t["clip"]:
            cl = load_mref(t["clip_file"]); assert t["clip"] in cl, f"no clip {t['clip']} in {t['clip_file']}"; clip = dclip(cl[t["clip"]], t["clip"])
        style_bufs = []
        if t["style"]:
            cl = load_mref(t["style_file"]); feats = []
            for nm in t["style"]:
                assert nm in cl, f"no clip {nm}"; style_bufs.append(dclip(cl[nm], nm))
                feats.append(torch.empty(cl[nm]["frames"], self.FEAT, device=self.dev))
            self._keep += feats
            arr = lambda xs, ct: (ct * len(xs))(*xs)
            err = L.tss_set_style(arr([b.data_ptr() for b in style_bufs], ctypes.c_void_p), arr([cl[nm]["frames"] for nm in t["style"]], ctypes.c_int),
                                  arr([f.data_ptr() for f in feats], ctypes.c_void_p), len(style_bufs))
            if err: raise RuntimeError(err.decode())
        self.cfg = torch.empty(L.tss_cfg_size(), dtype=torch.uint8, device=self.dev)
        sp = (ctypes.c_void_p * 8)(*[b.data_ptr() for b in style_bufs])
        L.tss_make_cfg(_ptr(self.cfg), t["task"], ctypes.c_void_p(clip.data_ptr() if clip is not None else 0), t["push_max"], t["sway"], t["arm_tone"],
                       t["push_every"], t["steps"], t["init_noise"], t["fixed_start"], t["assist"], t["act_mul"], sp, len(style_bufs), t["style_start"], t["goal_cone"])
        self.envs = torch.empty(n * L.tss_env_size(), dtype=torch.uint8, device=self.dev)
        self.ar = torch.arange(n, device=self.dev, dtype=torch.int64)
        self.ep = torch.tensor(self.episodes, device=self.dev, dtype=torch.int64)
        seeds = torch.tensor(np.array([(self.erng + e * 1000003) & U64 for e in range(n)], np.uint64).view(np.int64), device=self.dev)
        L.tss_reset(_ptr(self.envs), _ptr(self.cfg), n, _ptr(seeds), None, self._st())
        self.ret = torch.zeros(n, dtype=torch.float64, device=self.dev)
        self.rew = torch.empty(n, device=self.dev); self.done = torch.empty(n, dtype=torch.int32, device=self.dev)
        self.fell = torch.empty_like(self.done); self.len = torch.empty_like(self.done); self.tobs = torch.zeros(n, self.OBS, device=self.dev)
        self._check()

    def _st(self): return ctypes.c_void_p(torch.cuda.current_stream(self.dev).cuda_stream)

    def _check(self):
        torch.cuda.synchronize(self.dev); e = self.L.tss_error()
        if e: raise RuntimeError(e.decode())

    def obs(self):
        out = torch.empty(self.n, self.OBS, device=self.dev); self.L.tss_obs(_ptr(self.envs), self.n, _ptr(out), self._st()); return out

    def step(self, act):
        """act [n, ACT] in [-1, 1] -> reward, done, fell, the observation reached at a time-limit end, and for finished
        episodes their return and length (0 elsewhere). Finished environments are reset."""
        act = act.to(self.dev, torch.float32).contiguous()
        self.L.tss_step(_ptr(self.envs), self.n, _ptr(act), _ptr(self.rew), _ptr(self.done), _ptr(self.fell), _ptr(self.len), _ptr(self.tobs), self._st())
        d = self.done.bool()
        self.ret += self.rew.double()
        ep_ret = torch.where(d, self.ret, torch.zeros_like(self.ret)).float(); ep_len = torch.where(d, self.len, torch.zeros_like(self.len))
        self.ret = torch.where(d, torch.zeros_like(self.ret), self.ret)
        ids = self.ep + torch.cumsum(d.long(), 0)                    # the episode number each finished env starts
        seeds = self.erng + ids * 7777 + self.ar                       # int64 arithmetic wraps like uint64
        self.L.tss_reset(_ptr(self.envs), _ptr(self.cfg), self.n, _ptr(seeds), _ptr(d.to(torch.uint8)), self._st())
        self.ep = self.ep + d.long().sum()
        return dict(rew=self.rew.clone(), done=self.done.clone(), fell=self.fell.clone(), tobs=self.tobs.clone(), ep_ret=ep_ret, ep_len=ep_len)


class CpuPool(_Pool):
    def __init__(self, n, task="balance", mref=os.path.join(BODY, "data/motions_ref.bin"), style="", assist=0, push=0, act_mul=0, steps=0,
                 style_start=0, sway=0, arm_tone=0, goal_cone=0, seed=0, device="cpu", room=None):
        path = os.path.join(HERE, "libtssenv.so")
        if not os.path.exists(path):
            subprocess.check_call(f"cd {HERE} && gcc -O3 -march=native -fopenmp -ffp-contract=off -shared -fPIC -o libtssenv.so tssenv.c ../env.c ../body.c ../body_x8.c ../mref.c ../room.c -lm", shell=True)
        self.L = L = ctypes.CDLL(path); self._init_common(n, seed); self.dev = torch.device(device)
        L.pool_new.restype = ctypes.c_void_p
        L.pool_new.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p] + [ctypes.c_double] * 3 + [ctypes.c_int] + [ctypes.c_double] * 3 + [ctypes.c_int, ctypes.c_double]
        L.pool_obs.argtypes = [ctypes.c_void_p] * 2; L.pool_step.argtypes = [ctypes.c_void_p] * 8; L.pool_steps.argtypes = [ctypes.c_void_p]
        o, a, h = ctypes.c_int(), ctypes.c_int(), ctypes.c_int(); L.tss_dims(ctypes.byref(o), ctypes.byref(a), ctypes.byref(h)); self.OBS, self.ACT, self.HZ = o.value, a.value, h.value
        self.room = load_room(room); rb = np.ascontiguousarray(self.room)
        L.room_set.argtypes = [ctypes.c_void_p, ctypes.c_int]
        if L.room_set(rb.ctypes.data_as(ctypes.c_void_p), len(rb)) < 0: raise RuntimeError("room_set failed")
        self.p = L.pool_new(n, task.encode(), mref.encode(), style.encode(), assist, push, act_mul, steps, style_start, sway, arm_tone, seed, goal_cone)
        if not self.p: raise RuntimeError("pool_new failed")
        self.steps = L.pool_steps(self.p)
        self.b = dict(rew=np.zeros(n, np.float32), done=np.zeros(n, np.int32), fell=np.zeros(n, np.int32), tobs=np.zeros((n, self.OBS), np.float32),
                      ep_ret=np.zeros(n, np.float32), ep_len=np.zeros(n, np.int32))

    def obs(self):
        o = np.zeros((self.n, self.OBS), np.float32); self.L.pool_obs(self.p, o.ctypes.data_as(ctypes.c_void_p)); return torch.from_numpy(o).to(self.dev)

    def step(self, act):
        a = np.ascontiguousarray(act.detach().cpu().numpy(), np.float32); b = self.b
        for v in (b["tobs"], b["ep_ret"], b["ep_len"]): v[...] = 0
        self.L.pool_step(self.p, a.ctypes.data_as(ctypes.c_void_p), *(b[k].ctypes.data_as(ctypes.c_void_p) for k in ("rew", "done", "fell", "tobs", "ep_ret", "ep_len")))
        return {k: torch.from_numpy(v.copy()).to(self.dev) for k, v in b.items()}
