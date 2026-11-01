#!/usr/bin/env bash
# Treino na CPU (backend estável). Usa o clang que veio no ROCm SDK.
SP="$(cd "$(dirname "$0")" && pwd)"
ROCMDIR=${ROCM_SDK_CORE:-/path/to/_rocm_sdk_core}
export DEV=CPU
export THREADS="${THREADS:-24}"   # paraleliza os kernels de CPU nos núcleos
export PATH="$ROCMDIR/lib/llvm/bin:$PATH"
exec "$SP/venv/bin/python" "$@"
