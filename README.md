## 目录结构介绍
```
├── BareMixInvocation              // 通过AIC/AIV MIX方式实现复数Gram矩阵融合算子
│   ├── cmake                      // 编译工程文件
│   ├── scripts
│   │   ├── gen_data.py            // 输入数据和真值数据生成脚本
│   │   ├── verify_data.py         // 验证输出数据和真值数据是否一致
│   │   └── verify_result.py       // 与官方样例命名兼容的验证入口
│   ├── CMakeLists.txt             // 编译工程文件
│   ├── data_utils.h               // 数据读入写出函数
│   ├── main.cpp                   // 主函数，调用算子的应用程序，含CPU域及NPU域调用
│   ├── baremix_custom_tiling.cpp  // 算子tiling实现
│   ├── baremix_custom.cpp         // 算子kernel实现
│   └── run.sh                     // 编译运行算子的脚本
```

## 算子规格描述
输入：
- `Ar`: `float32 [272, 256, 8n]`
- `Ai`: `float32 [272, 256, 8n]`

输出：
- `B`: `float32 [17, 8n, 8n]`
- `BPlur`: `float32 [17, 8n, 8n]`
- `BPlui`: `float32 [17, 8n, 8n]`
- `Bsum`: `float32 [8n, 8n]`
- `Csum`: `float32 [n, n]`

核函数名：`baremix_custom`

## 代码实现介绍
`baremix_custom.cpp` 使用 `KERNEL_TYPE_MIX_AIC_1_2` 启用 AIC/AIV 混合核。AIC 侧按 group 调用 Matmul 高阶 API，连续算完该 group 下 16 个 inner 的四个实矩阵乘：
```
RR = Ar^T * Ar
II = Ai^T * Ai
RI = Ar^T * Ai
IR = Ai^T * Ar
```

AIV 侧通过 `CrossCoreWaitFlag` 等待 AIC 完成，再计算：
```
real = RR + II
imag = RI - IR
B     += (real * real + imag * imag) / 16
BPlur += real / 16
BPlui += imag / 16，上三角保留原值，下三角取共轭
Bsum  += B / 17
Csum  += B[::8, ::8] / 17
```

AIV 不再把每个 inner 的部分和写回 GM 后再读出继续累加；现在每个 row/col chunk 在 UB 中累加完 16 个 inner，再一次性写出 `B/BPlur/BPlui`，避免 MTE3 写回和下一轮 MTE2 读入之间的数据相关冒险。

AIC 每个 group 只发送一次 ready flag。AIV 等 ready 后处理该 group 的 16 个 inner，完成后两个 AIV sub block 对同一个 done flag 调用 `CrossCoreSetFlag<0x2, ...>`；mode 2 会在两个 AIV 都 set 后放行 AIC，AIC 再复用临时 workspace。ready/done flag 按 group 奇偶交替使用，避免同一 flagId 高频设置。

`baremix_custom_tiling.cpp` 不固定 `16x16` 分块，而是让 `MultiCoreMatmulTiling` 根据 `8n * 8n` 输出规模生成能覆盖全矩阵的 `singleCoreM/singleCoreN`。为兼容不同 CANN 版本，kernel 不直接读取 `TCubeTiling::usedCoreNum`，而是使用自定义 tiling 参数中的 `blockDim`。

## 运行样例算子
```bash
bash run.sh -r npu -v Ascend910B1 -n 1
```

参数：
- `-r, --run-mode`: `cpu / sim / npu`
- `-v, --soc-version`: `Ascend910B1 / Ascend910B2 / Ascend910B3 / Ascend910B4`
- `-n, --user-num`: 用户数 `n`

输出文件位于 `output/`，验证脚本会对比：
- `b.bin` vs `golden_b.bin`
- `bplur.bin` vs `golden_bplur.bin`
- `bplui.bin` vs `golden_bplui.bin`
- `bsum.bin` vs `golden_bsum.bin`
- `csum.bin` vs `golden_csum.bin`
