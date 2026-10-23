"""debug_env.py: where the GPU build of the environments departs from the CPU build (field by field)."""
import ctypes, os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_env as ge
N = 14 * 14 + 14
g = ge.GpuPool(4, task="balance", seed=3); c = ge.CpuPool(4, task="balance", seed=3)
L, C = g.L, c.L
dc = torch.zeros(N, dtype=torch.float64, device="cuda"); L.tss_consts(ctypes.c_void_p(dc.data_ptr())); torch.cuda.synchronize()
hc = np.zeros(N); C.tss_consts_cpu(hc.ctypes.data_as(ctypes.c_void_p))
dd = dc.cpu().numpy(); bad = np.nonzero(dd != hc)[0]
print("constants differing:", [(int(i), dd[i], hc[i]) for i in bad[:20]])
ES = L.tss_env_size(); C.pool_env_size.restype = ctypes.c_int; print("sizeof(Env) gpu", ES, "cpu", C.pool_env_size())
C.pool_env_copy.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]; C.pool_env_set.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
C.pool_world_step.argtypes = [ctypes.c_void_p, ctypes.c_int]; L.tss_world_step.argtypes = [ctypes.c_void_p, ctypes.c_int]
def cpu_env(e):
    b = np.zeros(ES, np.uint8); C.pool_env_copy(c.p, e, b.ctypes.data_as(ctypes.c_void_p)); return b
def gpu_env(e): torch.cuda.synchronize(); return g.envs.view(-1, ES)[e].cpu().numpy()
FIELDS = ["x", "v", "w", "xprev"]  # Vec x,v,w,xprev then Quat q,qprev, inv_mass, inv_inertia: 24 doubles per Body
names = [f"b{i}.{f}{a}" for i in range(14) for f, n in (("x", 3), ("v", 3), ("w", 3), ("xprev", 3), ("q", 4), ("qprev", 4), ("im", 1), ("ii", 3)) for a in range(n)]
def report(tag, A, B):
    a = A[:14 * 24 * 8].view(np.float64); b = B[:14 * 24 * 8].view(np.float64); d = np.abs(a - b)
    i = np.argsort(-d)[:6]
    print(f"{tag}: bodies max diff {d.max():.3e}; worst " + ", ".join(f"{names[k]} {d[k]:.2e}" for k in i))
    rest = np.nonzero(A[14 * 24 * 8:] != B[14 * 24 * 8:])[0]
    print(f"    other Env bytes differing: {len(rest)} (first offsets {list((rest[:8] + 14 * 24 * 8))})")
report("after reset", gpu_env(0), cpu_env(0))
# the physics alone, from the same state: copy the CPU env into the GPU slot, one world_step on both
st = cpu_env(1); g.envs.view(-1, ES)[1].copy_(torch.from_numpy(st))
for k in range(3):
    L.tss_world_step(g.envs.data_ptr(), 1); C.pool_world_step(c.p, 1)
    report(f"world_step {k + 1} from identical state", gpu_env(1), cpu_env(1))
# a full env step with zero action from identical state
st = cpu_env(2); g.envs.view(-1, ES)[2].copy_(torch.from_numpy(st))
g.step(torch.zeros(4, g.ACT, device="cuda")); c.step(torch.zeros(4, g.ACT))
report("env_step (zero action) from identical state", gpu_env(2), cpu_env(2))
g._check()
