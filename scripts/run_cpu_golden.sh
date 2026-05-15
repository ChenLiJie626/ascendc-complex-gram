#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 test_complex_gram_golden.py
python3 test_tiling_plan.py
cmake -S . -B build_host -DCOMPLEX_GRAM_BUILD_HOST_TILING=ON
cmake --build build_host -j
./build_host/complex_gram_tiling 16
