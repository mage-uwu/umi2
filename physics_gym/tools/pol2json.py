"""Convert a .pol file written by train.c into JSON for body.js (weights as base64 float32)."""
import base64, json, struct, sys
OBS_DIM = 14 * 15 + 6
raw = open(sys.argv[1], "rb").read()
t, i, o, h, L, T, D, n, p = struct.unpack("<9i", raw[:36])
P = (len(raw) - 36) // 4 - 2 * OBS_DIM
b64 = lambda a, b: base64.b64encode(raw[a:b]).decode()
th0 = 36; mu0 = th0 + 4 * P; sd0 = mu0 + 4 * OBS_DIM
spec = {"type": ["mlp", "ssm", "ltc"][t], "in": i, "out": o, "hidden": h, "layers": L, "trees": T, "depth": D, "n": n, "p": p,
        "params": P, "theta": b64(th0, mu0), "mean": b64(mu0, sd0), "std": b64(sd0, sd0 + 4 * OBS_DIM)}
json.dump(spec, open(sys.argv[2], "w"))
print(spec["type"], P, "parameters")
