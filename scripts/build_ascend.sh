#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${ASCEND_HOME:=/usr/local/Ascend/ascend-toolkit/latest}"
source "${ASCEND_HOME}/set_env.sh"

cat <<'MSG'
This repository is now arranged as an msOpGen-style custom-op skeleton.
For a real build:
  1. Generate an operator project with msOpGen for op name ComplexGram.
  2. Copy/merge these directories into the generated project:
       op_proto/
       op_host/
       op_kernel/
       acl_test/
       python/
       scripts/
  3. Replace op_proto and op_host pseudo registration blocks with generated macros.
  4. Wire Matmul tiling and AIC/AIV SetFlag/WaitFlag wrappers for your CANN version.
  5. Run the generated ./build.sh and then scripts/run_ascend.sh <n>.
MSG
