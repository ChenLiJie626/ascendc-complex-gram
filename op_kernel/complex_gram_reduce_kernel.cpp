/*
 * Pure AIV reduce kernel for Bsum/Csum after complex_gram_single_kernel.
 * Launch this kernel on the same stream after the single AIC/AIV kernel.
 */

#include "kernel_operator.h"
#include "complex_gram_tiling.h"

using namespace AscendC;
using OutT = float;
constexpr float REDUCE_INV_17 = 1.0f / 17.0f;

class ComplexGramReduceKernel {
public:
    __aicore__ inline void Init(GM_ADDR b, GM_ADDR bsum, GM_ADDR csum, GM_ADDR tiling)
    {
        t = *reinterpret_cast<__gm__ ComplexGramTilingData *>(tiling);
        bGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutT *>(b), static_cast<uint64_t>(t.groupNum) * t.k * t.k);
        bsumGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutT *>(bsum), static_cast<uint64_t>(t.k) * t.k);
        csumGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutT *>(csum), static_cast<uint64_t>(t.n) * t.n);
    }

    __aicore__ inline void Process()
    {
        const int32_t blockId = GetBlockIdx();
        const int32_t blockNum = GetBlockNum();
        const int64_t k2 = static_cast<int64_t>(t.k) * t.k;
        for (int64_t idx = blockId; idx < k2; idx += blockNum) {
            OutT acc = 0.0f;
            for (int32_t g = 0; g < t.groupNum; ++g) {
                acc += bGm.GetValue(static_cast<int64_t>(g) * k2 + idx);
            }
            bsumGm.SetValue(idx, acc * REDUCE_INV_17);
        }
        const int64_t n2 = static_cast<int64_t>(t.n) * t.n;
        for (int64_t idx = blockId; idx < n2; idx += blockNum) {
            const int32_t a = static_cast<int32_t>(idx / t.n);
            const int32_t b = static_cast<int32_t>(idx - static_cast<int64_t>(a) * t.n);
            const int64_t elem = static_cast<int64_t>(a * 8) * t.k + b * 8;
            OutT acc = 0.0f;
            for (int32_t g = 0; g < t.groupNum; ++g) {
                acc += bGm.GetValue(static_cast<int64_t>(g) * k2 + elem);
            }
            csumGm.SetValue(idx, acc * REDUCE_INV_17);
        }
    }

private:
    ComplexGramTilingData t{};
    GlobalTensor<OutT> bGm, bsumGm, csumGm;
};

extern "C" __global__ __aicore__ void complex_gram_reduce_kernel(GM_ADDR b,
                                                                   GM_ADDR bsum,
                                                                   GM_ADDR csum,
                                                                   GM_ADDR tiling)
{
    ComplexGramReduceKernel op;
    op.Init(b, bsum, csum, tiling);
    op.Process();
}
