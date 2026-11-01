#!/usr/bin/env bash
# Atalho que configura o ambiente da GPU AMD e roda o python do venv.
# Uso: ./gpu.sh script.py [args...]
SP="$(cd "$(dirname "$0")" && pwd)"
ROCMDIR=${ROCM_SDK_CORE:-/path/to/_rocm_sdk_core}
export DEV=AMD
export ROCM_PATH="$ROCMDIR"
export HIP_DEVICE_LIB_PATH="$ROCMDIR/lib/llvm/amdgcn/bitcode"
export LD_LIBRARY_PATH="$SP/rocmlibs:$ROCMDIR/lib"
export PATH="$ROCMDIR/bin:$PATH"
exec "$SP/venv/bin/python" "$@"
