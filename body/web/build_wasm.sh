#!/bin/sh
# tss.wasm: the nanotss policy runtime (tss_wasm.c, which includes ../../nanotss_blt.c) for the viewer.
# Needs clang with the wasm32 target and wasi-libc (apt install wasi-libc).
cd "$(dirname "$0")" && clang --target=wasm32-wasi --sysroot=/usr -isystem /usr/include/wasm32-wasi -O2 -mexec-model=reactor \
  -nodefaultlibs -L/usr/lib/wasm32-wasi -Wl,--strip-all -o tss.wasm tss_wasm.c -lc -lm
