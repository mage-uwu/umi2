"""bench_env.py: how fast the GPU environments step, alone (no policy), at several sizes: the parkour course of
gpu/pod_parkour.sh (PARC and CMU climbs, strength x2), random actions in [-1, 1], 30 steps to warm up, then STEPS timed.
    python3 bench_env.py [n,n,...] [STEPS]
One JSON line per size: environment steps per second (and per control step, ms). Builds of the environments other than
the default come from TSS_FMAD / TSS_BLOCK in the environment (gpu_env.py _variant)."""
import json, os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import gpu_env as ge
D = os.path.join(ge.BODY, "data"); B = json.load(open(f"{D}/parkour_fit_bank.json"))
sizes = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "4096,8192,16384,32768").split(",")]; steps = int(sys.argv[2]) if len(sys.argv) > 2 else 200
for n in sizes:
    pool = ge.GpuPool(n, task="getup_imitate", style=f"{D}/parkour_fit.bin:" + ",".join(B["clips"]), room=f"{D}/parkour_fit_bank.json", seed=3, assist=0.3,
                      steps=400, strength=2, start0=0.5, drift=0.6, tail=0.1, assist_pos_only=1)
    g = torch.Generator(device="cuda").manual_seed(1)
    act = lambda: torch.rand(n, pool.ACT, device="cuda", generator=g) * 2 - 1
    for _ in range(30): pool.obs(); pool.step(act())
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(steps): pool.obs(); pool.step(act())
    torch.cuda.synchronize(); dt = time.time() - t0
    print(json.dumps({"variant": "+".join(ge._variant()) or "default", "gpu": torch.cuda.get_device_name(0), "envs": n, "env_steps_per_s": round(n * steps / dt),
                      "ms_per_step": round(1000 * dt / steps, 2)}), flush=True)
    del pool; torch.cuda.empty_cache()
