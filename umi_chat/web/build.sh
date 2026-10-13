#!/bin/bash
# Build umi.js + umi.wasm for the browser chat window (needs Emscripten: source emsdk_env.sh).
set -e
cd "$(dirname "$0")"
emcc -O3 -msimd128 umi_wasm.c -o umi.js \
  -sMODULARIZE=1 -sEXPORT_NAME=createUmi -sENVIRONMENT=worker \
  -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB -sINITIAL_MEMORY=256MB -sSTACK_SIZE=4MB \
  -sFORCE_FILESYSTEM=1 -sEXPORTED_FUNCTIONS=_umi_load,_umi_load_web,_umi_say,_umi_reset,_umi_set_gate,_malloc,_free \
  -sEXPORTED_RUNTIME_METHODS=FS,ccall,cwrap -lm
ls -la umi.js umi.wasm
