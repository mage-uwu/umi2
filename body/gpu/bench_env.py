"""bench_env.py: control steps per second of the GPU environments (random actions, resets included)."""
import json, os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_env as ge
D = os.path.join(ge.BODY, "data")
for task, kw in (("balance", dict(push=150)), ("getup", dict(style=f"{D}/cmu_ref.bin:getup_back1,getup_back2,getup_face1,getup_face2,getup_side1,getup_side2"))):
    for n in (1024, 4096, 16384):
        p = ge.GpuPool(n, task=task, seed=1, **kw); g = torch.Generator(device="cuda").manual_seed(0)
        act = lambda: torch.rand(n, p.ACT, device="cuda", generator=g) * 2 - 1
        for _ in range(5): p.step(act())
        torch.cuda.synchronize(); t0 = time.time(); k = 30
        for _ in range(k): p.obs(); p.step(act())
        torch.cuda.synchronize(); dt = time.time() - t0; p._check()
        print(json.dumps({"bench": "env_step", "task": task, "envs": n, "ms_per_step": round(1000 * dt / k, 2), "env_steps_per_s": round(n * k / dt)}), flush=True)
