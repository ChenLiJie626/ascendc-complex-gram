# AscendC ComplexGram Custom Operator

This repository has been reorganized to match the official `gitcode.com/Ascend/msopgen` C++ custom-op project style.

The important generated-project files are now:

```text
ComplexGram
├── build.sh
├── CMakeLists.txt
├── CMakePresets.json
├── framework/
│   ├── CMakeLists.txt
│   └── tf_plugin/CMakeLists.txt
├── op_host/
│   ├── CMakeLists.txt
│   └── complex_gram.cpp          # OpDef + InferShape + InferDataType + TilingFunc
├── op_kernel/
│   ├── CMakeLists.txt
│   ├── complex_gram.cpp          # kernel entry: complex_gram + complex_gram_do
│   └── complex_gram_tiling.h     # BEGIN_TILING_DATA_DEF / REGISTER_TILING_DATA_CLASS
├── caller/
│   ├── CMakeLists.txt
│   ├── main.cpp                  # ACLNN caller, includes aclnn_complex_gram.h
│   ├── exec.py
│   └── run.sh
├── python/
│   ├── gen_data.py
│   └── verify.py
├── scripts/
│   ├── run_local_checks.sh
│   └── run_ascend.sh
├── test_complex_gram_golden.py
└── test_tiling_plan.py
```

## Operator definition

Inputs:

```text
ar: fp16, ND, [272, 256, 8*n]
ai: fp16, ND, [272, 256, 8*n]
```

Outputs:

```text
b:     fp32, ND, [17, 8*n, 8*n]
bplur: fp32, ND, [17, 8*n, 8*n]
bplui: fp32, ND, [17, 8*n, 8*n]
bsum:  fp32, ND, [8*n, 8*n]
csum:  fp32, ND, [n, n]
```

Math:

```text
A_s = Ar_s + j * Ai_s
P_s = A_s^H @ A_s
    = (Ar_s^T Ar_s + Ai_s^T Ai_s)
      + j(Ar_s^T Ai_s - Ai_s^T Ar_s)

B[g]     = sum_i (real(P)^2 + imag(P)^2) / 16
BPlur[g] = sum_i real(P) / 16
BPlui[g] = sum_i imag(P) / 16
Bsum     = sum_g B[g] / 17
Csum[a,b]= sum_g B[g,a*8,b*8] / 17
```

## Official msOpGen-style entries

### Host entry

File:

```text
op_host/complex_gram.cpp
```

It contains the official-style pieces shown in Ascend/msopgen quick start:

```cpp
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context);
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context);
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context);
}

namespace ops {
class ComplexGram : public OpDef { ... };
OP_ADD(ComplexGram);
}
```

### Kernel entry

File:

```text
op_kernel/complex_gram.cpp
```

It now has the kernel entry and generated-stub call entry:

```cpp
extern "C" __global__ __aicore__ void complex_gram(
    GM_ADDR ar, GM_ADDR ai,
    GM_ADDR b, GM_ADDR bplur, GM_ADDR bplui,
    GM_ADDR bsum, GM_ADDR csum,
    GM_ADDR workspace, GM_ADDR tiling);

void complex_gram_do(uint32_t blockDim, void *l2ctrl, void *stream,
    uint8_t *ar, uint8_t *ai,
    uint8_t *b, uint8_t *bplur, uint8_t *bplui,
    uint8_t *bsum, uint8_t *csum,
    uint8_t *workspace, uint8_t *tiling);
```

Inside the kernel entry:

```cpp
GET_TILING_DATA(tiling_data, tiling);
KernelComplexGram op;
op.Init(..., tiling_data);
op.Process();
```

This fixes the previous problem: the project now has a real msOpGen-style call entry.

## AIC/AIV synchronization design

The kernel is organized as one AIC/AIV fused kernel:

```text
AIC cube c computes tile into workspace[c][buf]
    SetFlag ready0/ready1
AIV vector 2c and 2c+1 WaitFlag ready
    consume first/second half of tile
    write B/BPlur/BPlui
    SetFlag done0/done1
AIC waits done0/done1 before reusing ping-pong buffer
```

Mapping:

```text
cube 0  -> vector 0, 1
cube 1  -> vector 2, 3
...
cube 19 -> vector 38, 39
```

Flag wrappers are isolated in `op_kernel/complex_gram.cpp`:

```cpp
NotifyVecLaneReady
WaitVecLaneReady
NotifyCubeLaneDone
WaitCubeLaneDone
```

You must replace their `HardEvent` values with the exact values used by your installed CANN version's AIC/AIV fusion sample.

## Why Bsum/Csum are recomputed in the same kernel

A true global barrier across all AIV blocks is unsafe inside one kernel. To keep one official `complex_gram` entry producing all five outputs, `Bsum/Csum` are recomputed directly from `Ar/Ai` on the AIV side after the vector lane finishes its tile-consume loop. This is correct but not optimal.

Performance optimization later can replace this with a second reduce kernel or a formally supported global synchronization mechanism if your CANN version provides one.

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
blockDim=20 cubeBlockNum=20 vectorBlockNum=40
singleKernelWorkspace=10485760 bytes fullWorkspace=71303168 bytes
```

## Generate data locally

```bash
cd /Users/chenlj/workspace/ascendc_complex_gram
PYTHONPATH=. python3 python/gen_data.py --n 16 --out data/n16 --dtype float16
```

## Build and deploy on Ascend

On an Ascend machine:

```bash
cd /path/to/ascendc_complex_gram
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
source ${ASCEND_HOME_PATH}/bin/setenv.bash
bash build.sh
```

If `msopgen` is installed, `build.sh` calls:

```bash
msopgen compile -i . -c ${ASCEND_HOME_PATH}
```

If not, it falls back to CMake presets.

After a successful build, install the generated package:

```bash
MY_OP_PKG=$(find ./build_out -maxdepth 1 -name "custom_opp_*.run" | head -1)
bash ${MY_OP_PKG}
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib:$LD_LIBRARY_PATH
```

## Run with the official ACLNN caller

```bash
cd /path/to/ascendc_complex_gram
bash caller/run.sh 0 16
```

Arguments:

```text
caller/run.sh <device_id> <n>
```

The caller includes the generated ACLNN header:

```cpp
#include "aclnn_complex_gram.h"
```

and calls:

```cpp
aclnnComplexGramGetWorkspaceSize(ar, ai, b, bplur, bplui, bsum, csum, &workspaceSize, &executor);
aclnnComplexGram(workspace, workspaceSize, executor, stream);
```

## Important remaining Ascend-specific work

The structure and entries now follow msOpGen style, but real execution still requires CANN-version-specific integration:

1. Run `msopgen gen` on your Ascend machine for `ComplexGram`, or use this repository directly if your environment supports `msopgen compile` on this layout.
2. Replace `this->AICore().AddConfig("ascend910b")` in `op_host/complex_gram.cpp` with the exact line generated for your SoC.
3. Wire AscendC Matmul tiling for the tile shape `[BM,256] @ [256,BN] -> [BM,BN]`.
4. Replace `SetFlag/WaitFlag` `HardEvent` placeholders with your CANN version's official AIC/AIV fusion API.
5. Verify generated ACLNN header name. This repo assumes `aclnn_complex_gram.h` and `aclnnComplexGram*` functions, following msOpGen naming convention.

## Git status after this refactor

This refactor intentionally removes the previous non-official skeleton files and replaces them with official-style files under `op_host`, `op_kernel`, and `caller`.
