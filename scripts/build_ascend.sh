#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

: "${ASCEND_HOME:=/usr/local/Ascend/ascend-toolkit/latest}"
if [ ! -d "$ASCEND_HOME" ]; then
  echo "ASCEND_HOME not found: $ASCEND_HOME" >&2
  echo "Set ASCEND_HOME to your ascend-toolkit path, for example:" >&2
  echo "  export ASCEND_HOME=/usr/local/Ascend/ascend-toolkit/latest" >&2
  exit 1
fi

source "$ASCEND_HOME/set_env.sh"

cat <<'MSG'
This repository contains the operator kernel/tiling source. To build a runnable
custom op, place these files into an msOpGen or official AscendC custom-op sample:

  op_kernel/complex_gram_fused.cpp
  op_kernel/complex_gram_tiling.h
  op_host/complex_gram_tiling.cpp

Then run the generated package build, typically:

  ./build.sh
  ./build_out/*.run

If you use a pure AscendC sample CMake, add op_kernel/complex_gram_fused.cpp to
the kernel target and pass the generated tiling buffer as GM_ADDR tiling.
MSG
