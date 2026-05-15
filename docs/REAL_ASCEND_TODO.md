# Real Ascend Integration TODO

Use this checklist after generating the real `ComplexGram` project with `msOpGen`.

## 1. Generate base project

Generate an operator named `ComplexGram` with two inputs and five outputs:

Inputs:
- `ar`: float16/float32, `[272,256,K]`
- `ai`: float16/float32, `[272,256,K]`

Outputs:
- `b`: float32, `[17,K,K]`
- `bplur`: float32, `[17,K,K]`
- `bplui`: float32, `[17,K,K]`
- `bsum`: float32, `[K,K]`
- `csum`: float32, `[K/8,K/8]`

## 2. Merge source skeleton

Copy these folders into the generated project:

```text
op_proto/
op_host/
op_kernel/
acl_test/
python/
scripts/
```

## 3. Replace generated registration placeholders

Files needing CANN-version-specific registration macros:

```text
op_proto/complex_gram.cpp
op_host/complex_gram_tiling.cpp
```

Keep the helper functions, but replace the commented pseudo registration blocks with the exact functions emitted by your generated project.

## 4. Wire Matmul tiling

`op_kernel/complex_gram_single_kernel.cpp` uses AscendC Matmul for tiles:

```text
[BM,256] @ [256,BN] -> [BM,BN]
```

Default `BM=32`, `BN=32`. The host tiling side must generate and pass Matmul tiling data for this shape, including boundary/tail tiles if `K` is not divisible by `BM/BN`.

## 5. Wire AIC/AIV flags

Replace the four wrappers in `op_kernel/complex_gram_single_kernel.cpp`:

```text
NotifyVecLaneReady
WaitVecLaneReady
NotifyCubeLaneDone
WaitCubeLaneDone
```

Use the exact `SetFlag`/`WaitFlag` `HardEvent` names from your installed CANN version's AIC/AIV fusion sample.

## 6. Wire ACL runner

Fill TODOs in:

```text
acl_test/main.cpp
```

Required operations:

1. `aclInit`
2. `aclrtSetDevice`
3. create stream
4. allocate/copy inputs
5. allocate workspace/output/tiling
6. launch `complex_gram_single_kernel`
7. launch `complex_gram_reduce_kernel` on the same stream
8. copy outputs back
9. write output `.bin` files

## 7. Run verification

```bash
PYTHONPATH=. python3 python/gen_data.py --n 16 --out data/n16 --dtype float16
# build and run ACL binary
PYTHONPATH=. python3 python/verify.py --n 16 --golden data/n16 --output output/n16
```

## 8. Profile

Use `msprof` after numerical correctness passes. Focus on:

- cube utilization
- vector utilization
- AIC/AIV flag waiting time
- GM workspace bandwidth
- Matmul tile efficiency
- reduce kernel time for `Bsum/Csum`
