"""Convert the shader page's motions.json into per-body reference poses for the physics character.

Same retargeting as the page (index.html: restFrames / clipFrame): a bone's world rotation is
(clip frame) x (rest frame)^-1, where a frame is built from the bone's direction and a reference axis
across it. Body frames are world-aligned in the rest T-pose, so that world rotation is the body's.

Output (little endian): b"MREF", int32 clips; per clip: 16-byte name, int32 frames, float32 fps, then per
frame float32 root position (3, pelvis body centre) and 14 body world quaternions (w, x, y, z).

usage: python3 motions_to_ref.py motions.json skeleton.json out.bin
"""
import json, struct, sys
import numpy as np

BODIES = ["hips", "spine", "chest", "head", "leftUpperArm", "leftLowerArm", "rightUpperArm", "rightLowerArm",
          "leftUpperLeg", "leftLowerLeg", "leftFoot", "rightUpperLeg", "rightLowerLeg", "rightFoot"]
PELVIS_CENTER = np.array([0, 0.915, 0.005])
AIM = {
    "hips": ["hips", "spine"], "spine": ["spine", "chest"], "chest": ["chest", "neck"], "neck": ["neck", "head"], "head": ["neck", "head"],
    "leftShoulder": ["leftShoulder", "leftUpperArm"], "leftUpperArm": ["leftUpperArm", "leftLowerArm"], "leftLowerArm": ["leftLowerArm", "leftHand"], "leftHand": ["leftLowerArm", "leftHand"],
    "rightShoulder": ["rightShoulder", "rightUpperArm"], "rightUpperArm": ["rightUpperArm", "rightLowerArm"], "rightLowerArm": ["rightLowerArm", "rightHand"], "rightHand": ["rightLowerArm", "rightHand"],
    "leftUpperLeg": ["leftUpperLeg", "leftLowerLeg"], "leftLowerLeg": ["leftLowerLeg", "leftFoot"], "leftFoot": ["leftFoot", "leftToes"], "leftToes": ["leftFoot", "leftToes"],
    "rightUpperLeg": ["rightUpperLeg", "rightLowerLeg"], "rightLowerLeg": ["rightLowerLeg", "rightFoot"], "rightFoot": ["rightFoot", "rightToes"], "rightToes": ["rightFoot", "rightToes"],
}


def frame(d, s):
    x = np.asarray(d, float); x = x / np.linalg.norm(x)
    y = np.asarray(s, float); y = y - y.dot(x) * x; y = y / np.linalg.norm(y)
    return np.stack([x, y, np.cross(x, y)], axis=1)   # columns x, y, z


def mat2quat(m):
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1) * 2; return np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    i = int(np.argmax(np.diag(m)))
    if i == 0:
        s = np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2; return np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    if i == 1:
        s = np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2; return np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    s = np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2; return np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])


def main(mpath, spath, out):
    M = json.load(open(mpath)); S = {k: np.array(v) for k, v in json.load(open(spath)).items()}
    left = S["leftUpperLeg"] - S["rightUpperLeg"]; left[1] = 0; left /= np.linalg.norm(left)
    up = np.array([0, 1.0, 0]); fwd = np.cross(left, up)
    to_model = np.stack([left, up, fwd], axis=1)
    rest_inv = {}
    for name, (a, b) in AIM.items():
        if name not in BODIES: continue
        ref = fwd if any(k in name for k in ("Shoulder", "Arm", "Hand")) else left
        rest_inv[name] = frame(S[b] - S[a], ref).T
    leg = np.linalg.norm(S["leftUpperLeg"] - S["leftLowerLeg"]) + np.linalg.norm(S["leftLowerLeg"] - S["leftFoot"])
    hips_rest = S["hips"]
    bones = M["bones"]; clips = M["clips"]
    blob = [b"MREF", struct.pack("<i", len(clips))]
    for cname, c in clips.items():
        F = c["frames"]; ax = np.array(c["axes"]).reshape(F, len(bones), 6); hp = np.array(c["hips"]).reshape(F, 3)
        blob += [cname.encode()[:16].ljust(16, b"\0"), struct.pack("<if", F, M["fps"])]
        for f in range(F):
            R = {}
            for k, bn in enumerate(bones):
                if bn in BODIES: R[bn] = to_model @ frame(ax[f, k, :3], ax[f, k, 3:]) @ rest_inv[bn]
            hips = hips_rest + to_model @ np.array([hp[f, 0], 0, hp[f, 2]]) * leg
            hips[1] += (hp[f, 1] - M["stand"]) * leg
            root = hips + R["hips"] @ (PELVIS_CENTER - hips_rest)
            blob.append(struct.pack("<3f", *root))
            for bn in BODIES: blob.append(struct.pack("<4f", *mat2quat(R[bn])))
        print(cname, F, "frames", "root y range %.3f..%.3f" % (hp[:, 1].min(), hp[:, 1].max()))
    open(out, "wb").write(b"".join(blob))


if __name__ == "__main__":
    main(*sys.argv[1:4])
