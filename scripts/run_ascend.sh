#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE_ID="${1:-0}"
N="${2:-16}"

./build.sh
MY_OP_PKG=$(find ./build_out -maxdepth 1 -name "custom_opp_*.run" 2>/dev/null | head -1 || true)
if [[ -n "${MY_OP_PKG}" ]]; then
  bash "${MY_OP_PKG}"
fi
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/opp/vendors/customize/op_api/lib:${LD_LIBRARY_PATH:-}
bash caller/run.sh "${DEVICE_ID}" "${N}"
