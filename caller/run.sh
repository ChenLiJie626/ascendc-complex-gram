#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEVICE_ID="${1:-0}"
N="${2:-16}"
INPUT_DIR="${ROOT_DIR}/data/n${N}"
OUTPUT_DIR="${ROOT_DIR}/output/n${N}"

resolve_install_path() {
    if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then echo "${ASCEND_HOME_PATH}";
    elif [[ -n "${ASCEND_HOME:-}" ]]; then echo "${ASCEND_HOME}";
    elif [[ -d "${HOME}/Ascend/ascend-toolkit/latest" ]]; then echo "${HOME}/Ascend/ascend-toolkit/latest";
    else echo "/usr/local/Ascend/ascend-toolkit/latest"; fi
}
ASCEND_HOME_PATH="$(resolve_install_path)"
export ASCEND_HOME_PATH
if [[ -f "${ASCEND_HOME_PATH}/bin/setenv.bash" ]]; then
  source "${ASCEND_HOME_PATH}/bin/setenv.bash"
elif [[ -f "${ASCEND_HOME_PATH}/set_env.sh" ]]; then
  source "${ASCEND_HOME_PATH}/set_env.sh"
else
  echo "Cannot find Ascend setenv script under ${ASCEND_HOME_PATH}" >&2
  exit 1
fi
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib:${LD_LIBRARY_PATH:-}

cd "${ROOT_DIR}"
PYTHONPATH=. python3 python/gen_data.py --n "${N}" --out "${INPUT_DIR}" --dtype float16
mkdir -p "${OUTPUT_DIR}"
cmake -S caller -B caller/build
cmake --build caller/build -j
"${SCRIPT_DIR}/build/execute_complex_gram_op" "${DEVICE_ID}" "${N}" "${INPUT_DIR}" "${OUTPUT_DIR}"
PYTHONPATH=. python3 python/verify.py --n "${N}" --golden "${INPUT_DIR}" --output "${OUTPUT_DIR}"
