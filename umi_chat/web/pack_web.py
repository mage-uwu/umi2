#!/usr/bin/env python3
"""Pack umi-1M-chat for the browser: fp16 n-gram table, gzipped text, base64 parts.

The hashed n-gram embedding table (25 MB of float32) is stored as fp16; everything else in the
checkpoint stays float32. The worker widens the table back to float32 before loading, so the
C code is unchanged. Also writes a float32 checkpoint with the same fp16 rounding, for eval.

  python3 pack_web.py CHECKPOINT DATA_DIR OUT_DIR OFFSET COUNT [NAME]
  (OFFSET COUNT from hash_offset.c; NAME defaults to umi-1M-chat and names the parts and
   NAME.manifest.json, so several models can share one OUT_DIR)
"""
import base64, gzip, json, os, sys
import numpy as np

PART = 12 * 1024 * 1024  # base64 characters per .txt part (under the 16 MB text-file limit; multiple of 4)

def b64_parts(raw, out, stem):
    b = base64.b64encode(raw)
    names = []
    for i in range(0, len(b), PART):
        name = f"{stem}.{i // PART}.txt"; names.append(name)
        open(os.path.join(out, name), "wb").write(b[i:i + PART])
    return names, len(b)

def main(ckpt, data, out, off, count, name="umi-1M-chat"):
    off, count = int(off), int(count)
    os.makedirs(os.path.join(out, "model"), exist_ok=True); os.makedirs(os.path.join(out, "data"), exist_ok=True)
    raw = open(ckpt, "rb").read()
    end = off + 4 * count
    table = np.frombuffer(raw, dtype="<f4", count=count, offset=off)
    half = table.astype("<f2")
    if not np.isfinite(half).all(): sys.exit("fp16 overflow in the n-gram table")
    err = np.abs(half.astype("<f4") - table)
    print(f"n-gram table: {count} values, max |x| {np.abs(table).max():.4g}, max fp16 error {err.max():.3g}, "
          f"mean relative error {np.mean(err / (np.abs(table) + 1e-12)):.2e}")
    # float32 checkpoint carrying the fp16 rounding, for eval
    rt = raw[:off] + half.astype("<f4").tobytes() + raw[end:]
    open(os.path.join(out, name + ".fp16-roundtrip.bin"), "wb").write(rt)
    # browser package: prefix (float32) + table (fp16) + suffix (float32), gzipped, base64 parts
    pack = gzip.compress(raw[:off] + half.tobytes() + raw[end:], 9, mtime=0)
    mparts, mlen = b64_parts(pack, os.path.join(out, "model"), name + ".fp16.gz")
    manifest = {"model": {"parts": ["model/" + n for n in mparts], "prefix": off, "count": count, "suffix": len(raw) - end,
                          "bytes": len(raw)}, "data": {}}
    total = mlen
    for table in ("answers", "train"):
        z = gzip.compress(open(os.path.join(data, table + ".tsv"), "rb").read(), 9, mtime=0)
        parts, n = b64_parts(z, os.path.join(out, "data"), f"{name}.{table}.tsv.gz")
        manifest["data"][table] = ["data/" + p for p in parts]; total += n
    manifest["download_bytes"] = total
    json.dump(manifest, open(os.path.join(out, name + ".manifest.json"), "w"), indent=1)
    print(f"model: {len(raw)/1e6:.1f} MB float32 -> {len(pack)/1e6:.1f} MB packed -> {mlen/1e6:.1f} MB base64; "
          f"total download {total/1e6:.1f} MB")

if __name__ == "__main__":
    main(*sys.argv[1:7])
