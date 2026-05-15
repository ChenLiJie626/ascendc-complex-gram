# complex_gram_fused AscendC kernel

This workspace contains an AscendC MIX AIC/AIV fused kernel for:

```text
Ar, Ai: half [272, 256, 8n]
B:      float [17, 8n, 8n]
BPlur:  float [17, 8n, 8n]
BPlui:  float [17, 8n, 8n]
Bsum:   float [8n, 8n]
Csum:   float [n, n]
```

For each `g in [0, 17)` and `inner in [0, 16)`, the AIC side computes four real GEMMs with Cube Matmul:

```text
RR = Ar^T * Ar
II = Ai^T * Ai
RI = Ar^T * Ai
IR = Ai^T * Ar

real = RR + II
imag = RI - IR
B     += real^2 + imag^2
BPlur += real
BPlui += imag for upper triangle, -imag for lower triangle
```

The AIV side waits for the AIC side with `CrossCoreWaitFlag`, performs the norm / triangular conjugate / reductions, then signals the AIC side with `CrossCoreSetFlag` so the temporary GEMM workspace can be reused.

## Files

```text
include/complex_gram_fused_tiling.h
src/complex_gram_fused_kernel.cpp
src/complex_gram_fused_tiling.cpp
```

## Kernel Arguments

```cpp
extern "C" __global__ __aicore__ void complex_gram_fused(
    GM_ADDR ar,
    GM_ADDR ai,
    GM_ADDR b,
    GM_ADDR bPlur,
    GM_ADDR bPlui,
    GM_ADDR bsum,
    GM_ADDR csum,
    GM_ADDR workspace,
    GM_ADDR tilingGm);
```

Use `GenerateTiling(...)` from `src/complex_gram_fused_tiling.cpp` to create `tilingGm`, get `actualBlockDim`, and get the required `workspaceSize`.

The workspace layout is:

```text
[ Matmul system workspace ][ tmpRR ][ tmpII ][ tmpRI ][ tmpIR ]
```

where each temporary matrix has `8n * 8n * sizeof(float)` bytes.

## Notes

- The kernel is written for `half` inputs and `float` outputs. If `Ar/Ai` are stored as `float`, convert them to `half` before launch or switch the Cube path to a supported dtype on your target.
- The kernel uses CrossCore flag IDs `8` and `9`. If your CANN version or surrounding fused code already reserves these IDs, change `FLAG_CUBE_DONE` and `FLAG_VEC_DONE` in `src/complex_gram_fused_kernel.cpp`.
- This is a direct-kernel style implementation. For framework-launch custom ops, keep the same kernel body and move `GenerateTiling` into the op-host tiling function.
- The current workspace does not include a local CANN/AscendC toolkit, so compile validation must be done on an Ascend development environment.
