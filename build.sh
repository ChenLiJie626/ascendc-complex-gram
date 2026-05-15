#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

resolve_install_path() {
    if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then echo "${ASCEND_HOME_PATH}";
    elif [[ -n "${ASCEND_HOME:-}" ]]; then echo "${ASCEND_HOME}";
    elif [[ -d "${HOME}/Ascend/ascend-toolkit/latest" ]]; then echo "${HOME}/Ascend/ascend-toolkit/latest";
    else echo "/usr/local/Ascend/ascend-toolkit/latest"; fi
}
export ASCEND_HOME_PATH="$(resolve_install_path)"
if [[ -f "${ASCEND_HOME_PATH}/bin/setenv.bash" ]]; then
  source "${ASCEND_HOME_PATH}/bin/setenv.bash"
elif [[ -f "${ASCEND_HOME_PATH}/set_env.sh" ]]; then
  source "${ASCEND_HOME_PATH}/set_env.sh"
else
  echo "ERROR: cannot find Ascend environment under ${ASCEND_HOME_PATH}" >&2
  exit 1
fi

if command -v msopgen >/dev/null 2>&1; then
  echo "Using msopgen compile entry."
  msopgen compile -i "${SCRIPT_DIR}" -c "${ASCEND_HOME_PATH}"
else
  echo "msopgen command not found; falling back to generated CMake preset if available."
  cmake --preset default
  cmake --build --preset default
fi
