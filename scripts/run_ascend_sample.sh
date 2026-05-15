#!/usr/bin/env bash
set -euo pipefail

# Example command sequence after integrating the files into an official AscendC
# custom-op sample package. Adjust paths according to your generated project.

: "${ASCEND_HOME:=/usr/local/Ascend/ascend-toolkit/latest}"
source "$ASCEND_HOME/set_env.sh"

# 1. Generate input and golden data on host.
python3 python/gen_data.py --n 16 --out data

# 2. Build/install your custom op package.
# ./build.sh
# ./build_out/*.run

# 3. Run your ACL/AscendCL test application, then verify.
# ./build/complex_gram_acl_run --n 16 --input data --output output
# python3 python/verify.py --n 16 --golden data --output output

echo "Edit this script after wiring into your official AscendC sample runner."
