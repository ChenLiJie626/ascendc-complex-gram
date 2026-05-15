#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

log() { echo "[ComplexGram build] $*"; }
fail() { echo "[ComplexGram build][ERROR] $*" >&2; exit 1; }

log "project root: ${SCRIPT_DIR}"

resolve_install_path() {
    if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then echo "${ASCEND_HOME_PATH}";
    elif [[ -n "${ASCEND_HOME:-}" ]]; then echo "${ASCEND_HOME}";
    elif [[ -d "${HOME}/Ascend/ascend-toolkit/latest" ]]; then echo "${HOME}/Ascend/ascend-toolkit/latest";
    else echo "/usr/local/Ascend/ascend-toolkit/latest"; fi
}

export ASCEND_HOME_PATH="$(resolve_install_path)"
log "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"

if [[ -f "${ASCEND_HOME_PATH}/bin/setenv.bash" ]]; then
  # shellcheck source=/dev/null
  source "${ASCEND_HOME_PATH}/bin/setenv.bash"
elif [[ -f "${ASCEND_HOME_PATH}/set_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${ASCEND_HOME_PATH}/set_env.sh"
else
  fail "cannot find Ascend environment. Set ASCEND_HOME_PATH, for example:
  export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
  source \${ASCEND_HOME_PATH}/bin/setenv.bash
Then run from the project directory:
  cd ${SCRIPT_DIR}
  bash build.sh"
fi

log "checking tools..."
command -v cmake >/dev/null 2>&1 || fail "cmake not found"
if ! command -v msopgen >/dev/null 2>&1; then
  fail "msopgen not found in PATH after sourcing CANN environment.
This script must be run on an Ascend/CANN environment with msopgen installed.
For local non-NPU checks, use:
  bash scripts/run_local_checks.sh"
fi

log "msopgen: $(command -v msopgen)"
log "cleaning old build output"
rm -rf build build_out

log "running: msopgen compile -i ${SCRIPT_DIR} -c ${ASCEND_HOME_PATH}"
msopgen compile -i "${SCRIPT_DIR}" -c "${ASCEND_HOME_PATH}"

if ! find ./build_out -maxdepth 1 -name "custom_opp_*.run" | grep -q .; then
  fail "msopgen finished but no build_out/custom_opp_*.run was generated. Check the compile log above."
fi

log "build packages:"
find ./build_out -maxdepth 1 -name "custom_opp_*.run" -print
log "done"
