# AscendC ComplexGram msOpGen Skeleton

This repository is organized as a realistic msOpGen/CANN custom operator skeleton for a fused complex Gram operator.

## Operator

Inputs:

- `Ar`: `[272, 256, 8*n]`, real part
- `Ai`: `[272, 256, 8*n]`, imaginary part

Outputs:

- `B`: `[17, 8*n, 8*n]`, norm output
- `BPlur`: `[17, 8*n, 8*n]`, real part of complex accumulation
- `BPlui`: `[17, 8*n, 8*n]`, imaginary part of complex accumulation
- `Bsum`: `[8*n, 8*n]`
- `Csum`: `[n, n]`

For every slice `s`, `A_s = Ar[s] + j*Ai[s]` and:

```text
P_s = A_s^H @ A_s
    = (Ar_s^T Ar_s + Ai_s^T Ai_s)
      + j(Ar_s^T Ai_s - Ai_s^T Ar_s)
```

For each group `g` of 16 slices:

```text
B[g]     = sum_i (real(P)^2 + imag(P)^2) / 16
BPlur[g] = sum_i real(P) / 16
BPlui[g] = sum_i imag(P) / 16
Bsum     = sum_g B[g] / 17
Csum[a,b]= sum_g B[g,a*8,b*8] / 17
```

## Directory layout

```text
op_proto/                 msOpGen op prototype / shape inference skeleton
op_host/                  host tiling skeleton
op_kernel/                AscendC kernels and shared tiling data
  complex_gram_single_kernel.cpp   AIC/AIV single-kernel producer-consumer template
  complex_gram_reduce_kernel.cpp   AIV reduce kernel for Bsum/Csum
  complex_gram_tiling.h            host/kernel tiling contract
acl_test/                 ACL runner skeleton
python/                   data generation and verification
scripts/                  local and Ascend run scripts
test_complex_gram_golden.py
test_tiling_plan.py
```

## Current implementation status

Implemented as source templates:

1. AIC/AIV single-kernel schedule with one cube mapped to two vectors.
2. Ping-pong workspace and SetFlag/WaitFlag wrapper locations.
3. Pure vector reduce kernel for `Bsum/Csum`.
4. Host tiling calculator and local tests.
5. Python input/golden generation and output verification scripts.

Still required on a real Ascend machine:

1. Generate a real msOpGen project for `ComplexGram`.
2. Replace `op_proto/complex_gram.cpp` pseudo registration with generated CANN macros.
3. Replace `op_host/complex_gram_tiling.cpp` pseudo tiling registration with your generated `gert::TilingContext` signature.
4. Wire AscendC Matmul tiling data for the `BM x BN x 256` GEMM tiles.
5. Replace the four SetFlag/WaitFlag wrappers in `op_kernel/complex_gram_single_kernel.cpp` with the exact HardEvent API from your CANN version.
6. Fill `acl_test/main.cpp` with real ACL memory allocation, kernel launch, and copy-back code.

## Local checks without CANN

```bash
cd /Users/chenlj/workspace/ascendc_complex_gram
bash scripts/run_local_checks.sh
```

Expected output includes:

```text
golden test passed
tiling plan tests passed
n=16 K=128 BM=32 BN=32
tileM=4 tileN=4 tasks=272
```

## Ascend integration flow

On an Ascend development machine:

```bash
cd /path/to/ascendc_complex_gram
export ASCEND_HOME=/usr/local/Ascend/ascend-toolkit/latest
source ${ASCEND_HOME}/set_env.sh
bash scripts/build_ascend.sh
```

Recommended real flow:

```bash
# 1. Generate msOpGen project for op ComplexGram.
# 2. Copy this skeleton's op_proto/op_host/op_kernel/acl_test/python/scripts into it.
# 3. Replace generated registration and tiling API placeholders.
# 4. Build package.
./build.sh

# 5. Generate input/golden and run ACL test.
bash scripts/run_ascend.sh 16

# 6. After ACL runner is wired to real NPU kernels, verify:
PYTHONPATH=. python3 python/verify.py --n 16 --golden data/n16 --output output/n16
```

## Notes

- `complex_gram_single_kernel.cpp` intentionally keeps SetFlag/WaitFlag calls in four wrappers so CANN-version-specific HardEvent names are isolated.
- `Bsum/Csum` are intentionally produced by a second vector reduce kernel because they need global completion of all `B[g,:,:]` tiles.
- Default input type in kernels is `half`; change `using InT = half` if your real input is float32 and your CANN/hardware Matmul path supports it.
