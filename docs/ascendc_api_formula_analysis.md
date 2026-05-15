# AscendC API 与复数 Gram 公式对应关系分析

本文档分析当前 `baremix_custom.cpp` / `baremix_custom_tiling.cpp` 中 AscendC API 调用与目标数学公式的对应关系，并解释当前现象：

- 输出前面部分有值；
- 前面有值的部分也和 `scripts/gen_data.py` 生成的 Python golden 不一致；
- 输出后面部分为 0。

结论先放前面：当前实现的数学拆解方向基本正确，但工程实现存在多个高风险逻辑点。尤其是 **tile 覆盖不完整** 和 **Matmul 中间结果布局/stride 与 AIV 紧凑读取假设不一致**，足以解释“前面有值但不对、后面为 0”的现象。

## 目标公式

输入复数矩阵：

```text
A = Ar + j * Ai
Ar/Ai shape = [17, 16, 256, 8n]
U = 8n
```

对于每个 `g in [0, 17)`：

```text
B_g = 0
BPlu_g = 0

for inner in [0, 16):
    H = A[g, inner]^H * A[g, inner]
    B_g    += norm(H)
    BPlu_g += H, lower triangle uses conjugate

B_g    /= 16
BPlu_g /= 16
Bsum   += B_g / 17
Csum   += B_g[::8, ::8] / 17
```

复数乘法展开：

```text
A^H * A
= (Ar^T - j Ai^T) * (Ar + j Ai)
= (Ar^T Ar + Ai^T Ai) + j * (Ar^T Ai - Ai^T Ar)
```

因此需要四个实数矩阵乘：

```text
RR = Ar^T * Ar
II = Ai^T * Ai
RI = Ar^T * Ai
IR = Ai^T * Ar

real = RR + II
imag = RI - IR
norm = real^2 + imag^2
```

Python golden 对应代码在 `scripts/gen_data.py`：

```python
rr = ar_slice.T @ ar_slice
ii = ai_slice.T @ ai_slice
ri = ar_slice.T @ ai_slice
ir = ai_slice.T @ ar_slice
real = rr + ii
imag = ri - ir
norm = real * real + imag * imag
```

## 当前 AIC 侧 API 与公式对应

当前 AIC 侧核心代码：

```cpp
using Mm = Matmul<
    MatmulType<TPosition::GM, CubeFormat::ND, float, true>,
    MatmulType<TPosition::GM, CubeFormat::ND, float, false>,
    MatmulType<TPosition::GM, CubeFormat::ND, float>,
    MatmulType<TPosition::GM, CubeFormat::ND, float>>;

REGIST_MATMUL_OBJ(pipe_, GetSysWorkSpacePtr(), mm_, &tiling_);

RunMatmul(ar, ar, tmpRR); // Ar^T * Ar
RunMatmul(ai, ai, tmpII); // Ai^T * Ai
RunMatmul(ar, ai, tmpRI); // Ar^T * Ai
RunMatmul(ai, ar, tmpIR); // Ai^T * Ar
```

`MatmulType<..., float, true>` 表示 A 矩阵按转置参与 Matmul，`MatmulType<..., float, false>` 表示 B 矩阵不转置。`SetAType(..., true)` / `SetTensorA(..., true)` 与公式中的 `Ar^T` / `Ai^T` 对应。

当前 offset 计算：

```cpp
offsetA = rowStart; // transA=true 时，A 原始矩阵为 [K, U]，按输出 M 方向平移列
offsetB = colStart; // B 原始矩阵为 [K, U]，按输出 N 方向平移列
offsetC = rowStart * U + colStart;
```

这与官方 Matmul 样例中 transA 的 offset 逻辑一致：`isTransA=true` 时，A 的 GM offset 不是 `rowStart * K`，而是 `rowStart`。

### AIC 侧潜在问题

1. `SetDim(blockDim)` 和实际 tile 网格可能不一致。

当前 host 侧：

```cpp
blockDim = min(ceil(U / 16) * ceil(U / 16), maxAic);
tilingApi.SetDim(blockDim);
tilingApi.GetTiling(tilingData);
```

kernel 侧：

```cpp
mBlocks = ceil(tiling.M / tiling.singleCoreM);
nBlocks = ceil(tiling.N / tiling.singleCoreN);
blockIdx -> (mIdx, nIdx)
```

如果 `mBlocks * nBlocks > launch blockDim`，只有前 `launch blockDim` 个 tile 被执行，后面的输出 tile 就保持 0。这与当前“后面都是 0”的现象高度一致。

需要验证：

```text
print:
  U
  blockDim
  tiling.M
  tiling.N
  tiling.singleCoreM
  tiling.singleCoreN
  ceil(M/singleCoreM) * ceil(N/singleCoreN)
```

必须满足：

```text
launch blockDim >= ceil(M/singleCoreM) * ceil(N/singleCoreN)
```

如果不满足，输出后半部分必然为 0。

2. 中间 workspace 被当作紧凑 `[16, U, U]` 读取，但 Matmul GM 输出布局未被显式验证。

当前 AIC 写：

```cpp
tmpBase = inner * U * U;
mm_.IterateAll(tmp[tmpBase + rowStart * U + colStart]);
```

当前 AIV 读：

```cpp
DataCopy(local, tmp[tmpBase + row * U + col], count);
```

这隐含假设：Matmul 的 GM ND 输出完全按紧凑 stride `U` 写入。

如果 Matmul API 因 ND 对齐、内部 tiling 或尾块处理，实际按对齐 stride 写，AIV 按 `U` 读取就会读错位置。这样会表现为：

```text
输出有值，但数值与 Python golden 不一致
```

需要验证 Matmul 输出的实际 GM stride。保守做法是给临时矩阵定义显式 aligned stride，例如：

```text
tmpStride = AlignUp(U, 8 or 16)
tmpBase   = inner * tmpStride * U
offsetC   = rowStart * tmpStride + colStart
```

AIV 也使用同一个 `tmpStride` 读：

```text
tmp[tmpBase + row * tmpStride + col]
```

当前实现没有这个 stride 参数，因此存在布局风险。

3. `n=1` 时 `U=8`，float 输入一行是 32B。

Cube Matmul 对 GM ND 输入的对齐很敏感。改成 float 输入后，`U=8` 时：

```text
float row bytes = 8 * 4 = 32B
```

行字节数已经满足 32B，对此前 half 输入下 `n=1` 只有 16B 的风险有缓解。但如果当前 CANN/硬件对 `ND` Matmul 的 N 维还有更高粒度要求，仍建议保留 pack/pad 的排查路径：

```text
UAligned = AlignUp(U, 16)
```

然后 Matmul 计算 `[K, UAligned] -> [UAligned, UAligned]`，AIV 只取前 `U * U` 有效区域。

## 当前 AIV 侧 API 与公式对应

当前 AIV 等待 AIC：

```cpp
CrossCoreWaitFlag(readyFlag);
ProcessGroup(g);
CrossCoreSetFlag<0x2, PIPE_MTE3>(doneFlag);
```

`KERNEL_TYPE_MIX_AIC_1_2` 下，AIV block 数是 AIC 的 2 倍。因此当前映射：

```cpp
aicBlockIdx = GetBlockIdx() / 2;
subIdx      = GetBlockIdx() % 2;
```

这表示两个 AIV 分别处理同一个 AIC tile 的前半行和后半行。

### AIV 公式映射

当前 AIV 每个 group、每个 row/col chunk：

```cpp
Duplicate(bLocal, 0)
Duplicate(prLocal, 0)
Duplicate(piLocal, 0)

for inner in [0, 16):
    rr = tmpRR[inner]
    ii = tmpII[inner]
    ri = tmpRI[inner]
    ir = tmpIR[inner]

    rr = rr + ii       // real
    ri = ri - ir       // imag
    ii = rr * rr
    ir = ri * ri
    ii = ii + ir       // norm

    bLocal  += norm
    prLocal += real
    piLocal += imag with lower triangle negated

bLocal  /= 16
prLocal /= 16
piLocal /= 16
```

对应公式：

```text
B     = average(real^2 + imag^2)
BPlur = average(real)
BPlui = average(imag or -imag)
```

这部分数学逻辑目前是正确方向。

### AIV 侧潜在问题

1. `Bsum` 在 tile 覆盖不全时只会累加已计算 tile。

当前：

```cpp
Bsum[row, col] += B_g[row, col] / 17
```

如果某些 tile 没执行，`Bsum` 对应区域保持 0。

2. `Csum` 只在 row 为 8 的倍数时更新。

当前：

```cpp
if row % 8 != 0 return;
for c in col..end step 8:
    Csum[row/8, c/8] += B_g[row, c] / 17
```

逻辑与公式：

```python
csum += bg[::8, ::8] / 17
```

一致。

但如果包含这些采样点的 tile 没执行，`Csum` 对应位置也保持 0。因此 `Csum` 后面为 0 同样可由 tile 覆盖不全解释。

3. 当前 `DataCopy(..., count)` 假定 row/col chunk 连续且对齐。

如果 `count` 不是硬件 DataCopy 友好的元素数，或者 `colStart` 对齐不满足要求，可能导致尾部数据不正确。当前 `VEC_CHUNK=256`，但实际 `count = colEnd - col`，尾块可能不是 8/16 的倍数。若后续仍有小范围误差，需要将 vector 读写改成 aligned block 加 mask 或手写 tail。

## 同步 API 与流水对应

当前同步设计：

```text
AIC:
  for g:
    compute all 16 inner tmp matrices
    CrossCoreSetFlag(readyFlag)
    CrossCoreWaitFlag(doneFlag)

AIV0/AIV1:
  for g:
    CrossCoreWaitFlag(readyFlag)
    process group
    CrossCoreSetFlag(doneFlag)
```

`CrossCoreSetFlag<0x2, ...>` 的 mode 2 语义适合 1 个 AIC 对 2 个 AIV 的同步。两个 AIV 都 set 同一个 done flag 后，AIC 的 wait 才能通过。

当前 ready/done flag 按 group 奇偶复用：

```cpp
readyFlag = 7 + (g & 1)
doneFlag  = 9 + (g & 1)
```

这个设计本身可以避免同一个 flag 在 17 个 group 中连续高频复用，但仍有前提：同一个 AIC/AIV pair 的 flag 计数在 wait 后被正确清除。

## 当前现象的最可能解释

### 现象 1：后面都是 0

最可能原因是：

```text
launch blockDim < ceil(M/singleCoreM) * ceil(N/singleCoreN)
```

也就是实际输出 tile 网格没有被所有 block 覆盖。

当前代码没有打印 tiling 结果，无法确认 `singleCoreM/singleCoreN` 与 `blockDim` 是否一致覆盖全矩阵。这个问题优先级最高。

### 现象 2：前面有值但不等于 Python golden

可能原因按优先级排序：

1. AIC Matmul 中间输出被 AIV 按错误 stride 读取。
2. `U=8n` 未做输入/中间矩阵 stride 对齐，Cube Matmul 实际读写和 Python 紧凑矩阵不一致。
3. `SetAType(..., true)`、`MatmulType<..., true>`、`SetTensorA(..., true)` 三者的转置语义在当前 CANN 版本下不是简单等价叠加，需要用最小 Matmul 单测确认。
4. vector `DataCopy` tail 未按硬件要求对齐。

## 建议的修复顺序

1. 在 host `GenerateTiling` 后打印 tiling 信息。

必须打印：

```cpp
std::cout << "U=" << u
          << " blockDim=" << blockDim
          << " M=" << tilingData.M
          << " N=" << tilingData.N
          << " singleCoreM=" << tilingData.singleCoreM
          << " singleCoreN=" << tilingData.singleCoreN
          << std::endl;
```

如果当前 CANN 版本字段名不可用，就用 `SaveToBuffer` 后在 kernel 临时输出 debug 小数组，或查本机 `TCubeTiling` 定义。

2. 保证 `blockDim` 覆盖所有输出 tile。

如果不能读取 `usedCoreNum`，建议不要依赖 `MultiCoreMatmulTiling` 自选 tile。改为显式设置较大的固定 tile，使：

```text
ceil(U / singleCoreM) * ceil(U / singleCoreN) <= blockDim
```

例如先调试用：

```text
singleCoreM = U
singleCoreN = U
blockDim = 1
```

这会牺牲并行度，但可以先验证数值公式。

3. 引入 `UAligned` 和 `tmpStride`。

建议新增参数：

```text
U       = 8n
UAlign  = AlignUp(U, 16)
tmpStride = UAlign
```

输入 `Ar/Ai` pack 到 `[17, 16, 256, UAlign]`，补零。Matmul 计算 `[UAlign, UAlign]`，AIV 只输出前 `U * U`。

4. 单独验证四个 Matmul。

先只输出 `tmpRR/tmpII/tmpRI/tmpIR` 中某个 group/inner 的一个 tile，和 Python 的：

```python
Ar.T @ Ar
Ai.T @ Ai
Ar.T @ Ai
Ai.T @ Ar
```

对比。若这一步不一致，问题在 AIC Matmul 输入布局/转置/offset；若一致，问题在 AIV 后处理。

5. 最后再恢复多 group 同步。

先调试：

```text
GROUP_NUM = 1
AVG_NUM = 1
blockDim = 1
```

验证通后，再逐步恢复 `AVG_NUM=16`、`GROUP_NUM=17`、多 tile。

## 最小判断

当前代码中，复数公式拆解是合理的：

```text
real = RR + II
imag = RI - IR
B = real^2 + imag^2
```

但当前实现还不能证明 API 输出布局与 Python 紧凑矩阵布局一致，也不能证明 launch 的 block 覆盖了所有输出 tile。这两个问题足以解释当前结果。因此下一步应先做 tiling 打印和单 Matmul 对比，而不是继续调 `B/BPlur/Bsum/Csum` 的公式。

## 参考

- 官方 BareMixInvocation 样例：https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/22_baremix_kernellaunch/BareMixInvocation
- AscendC `GetBlockIdx` 文档：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0185.html
- AscendC `CrossCoreSetFlag` 文档：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/ascendcopapi/atlasascendc_api_07_0273.html
- AscendC ND_ALIGN 说明：https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_10026.html
