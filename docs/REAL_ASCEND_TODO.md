# Real Ascend Integration TODO

This project now follows the official `gitcode.com/Ascend/msopgen` C++ project layout:

```text
build.sh
CMakeLists.txt
CMakePresets.json
framework/
op_host/complex_gram.cpp
op_kernel/complex_gram.cpp
op_kernel/complex_gram_tiling.h
caller/
python/
scripts/
```

## 1. Generate or validate msOpGen project

Recommended command on an Ascend machine:

```bash
msopgen gen -i complex_gram_msopgen.json -c ai_core-<your_soc> -lan cpp -out ComplexGram
```

The generated official project should contain:

```text
op_host/<op>.cpp
op_kernel/<op>.cpp
op_kernel/<op>_tiling.h
op_host/CMakeLists.txt
op_kernel/CMakeLists.txt
framework/
CMakePresets.json
build.sh
```

This repository already mirrors that layout with `complex_gram` file names.

## 2. Host registration file

File:

```text
op_host/complex_gram.cpp
```

It contains:

```text
optiling::TilingFunc
gel::InferShape / InferDataType
ops::ComplexGram : public OpDef
OP_ADD(ComplexGram)
```

Required real-machine edits:

1. Replace `this->AICore().AddConfig("ascend910b")` with the exact `AddConfig` line generated for your SoC.
2. Confirm `context->SetBlockDim(20)` matches your official AIC/AIV fusion launch API.
3. Add Matmul tiling data if your CANN Matmul API requires extra tiling beyond `ComplexGramTilingData`.

## 3. Kernel entry file

File:

```text
op_kernel/complex_gram.cpp
```

It now contains the official callable entries:

```cpp
extern "C" __global__ __aicore__ void complex_gram(..., GM_ADDR workspace, GM_ADDR tiling);
void complex_gram_do(uint32_t blockDim, void *l2ctrl, void *stream, ..., uint8_t *workspace, uint8_t *tiling);
```

Required real-machine edits:

1. Replace the four AIC/AIV flag wrappers with the exact `SetFlag`/`WaitFlag` `HardEvent` values from your CANN version:

```text
NotifyVecLaneReady
WaitVecLaneReady
NotifyCubeLaneDone
WaitCubeLaneDone
```

2. Wire Matmul tiling for:

```text
[BM,256] @ [256,BN] -> [BM,BN]
```

3. Confirm AIC/AIV compile macros. The code currently uses:

```cpp
#if defined(__DAV_CUBE__)
    ProcessCube();
#else
    ProcessVector();
#endif
```

Adjust if your official fusion sample uses a different macro.

## 4. ACLNN caller

File:

```text
caller/main.cpp
```

It includes the generated custom op API:

```cpp
#include "aclnn_complex_gram.h"
```

and calls:

```cpp
aclnnComplexGramGetWorkspaceSize(ar, ai, b, bplur, bplui, bsum, csum, &workspaceSize, &executor);
aclnnComplexGram(workspace, workspaceSize, executor, stream);
```

If your generated header/function names differ, update `caller/main.cpp` accordingly.

## 5. Build, install, run

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
source ${ASCEND_HOME_PATH}/bin/setenv.bash
bash build.sh
MY_OP_PKG=$(find ./build_out -maxdepth 1 -name "custom_opp_*.run" | head -1)
bash ${MY_OP_PKG}
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib:$LD_LIBRARY_PATH
bash caller/run.sh 0 16
```

## 6. Verify

```bash
PYTHONPATH=. python3 python/verify.py --n 16 --golden data/n16 --output output/n16
```

## 7. Profile

Use `msprof` after numerical correctness passes. Focus on:

- cube utilization
- vector utilization
- AIC/AIV flag waiting time
- GM workspace bandwidth
- Matmul tile efficiency
- recomputed `Bsum/Csum` overhead
