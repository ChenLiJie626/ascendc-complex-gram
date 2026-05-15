#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

: "${ASCEND_HOME:=/usr/local/Ascend/ascend-toolkit/latest}"
if [ ! -d "${ASCEND_HOME}" ]; then
  echo "ASCEND_HOME not found: ${ASCEND_HOME}" >&2
  echo "This is an msOpGen-style skeleton. On an Ascend machine set:" >&2
  echo "  export ASCEND_HOME=/usr/local/Ascend/ascend-toolkit/latest" >&2
  exit 1
fi
source "${ASCEND_HOME}/set_env.sh"

cat <<'MSG'
This repository has been reorganized as an msOpGen-style source skeleton.
To produce an installable custom-op package, create/generate the real msOpGen
project for op ComplexGram, then merge this repository's op_proto/op_host/op_kernel
files into the generated project and use that generated build system.

Next files to wire in the generated project:
  op_proto/complex_gram.cpp
  op_host/complex_gram_tiling.cpp
  op_kernel/complex_gram_single_kernel.cpp
  op_kernel/complex_gram_reduce_kernel.cpp
  op_kernel/complex_gram_tiling.h
MSG
