#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 test_complex_gram_golden.py
python3 test_tiling_plan.py
mkdir -p build_host
c++ -std=c++17 -DCOMPLEX_GRAM_HOST_STANDALONE op_host/complex_gram.cpp -o build_host/complex_gram_tiling
./build_host/complex_gram_tiling 16
PYTHONPATH=. python3 python/gen_data.py --n 2 --out data/local_n2 --dtype float16
