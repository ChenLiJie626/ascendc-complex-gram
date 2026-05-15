
/*
 * AscendC kernel for complex Gram statistics.
 *
 * Inputs:
 *   Ar: float [256, 8*n], real part of A
 *   Ai: float [256, 8*n], imag part of A
 *   n : number of users, K = 8*n
 * Outputs:
 *   B    : float [17, K, K], averaged norm of per-row complex outer products
 *   BPlur: float [17, K, K], real part of averaged complex Gram/outer product
 *   BPlui: float [17, K, K], imag part of averaged complex Gram/outer product
 *   Bsum : float [K, K], average of B over 17 groups
 *   Csum : float [n, n], average over groups of B[g, a*8, b*8]
 *
 * Mathematical definition implemented:
 *   p(row,r,c) = conj(A[row,r]) * A[row,c]
 *               = (Ar[row,r]*Ar[row,c] + Ai[row,r]*Ai[row,c])
 *                 + j(Ar[row,r]*Ai[row,c] - Ai[row,r]*Ar[row,c])
 *   B[g,r,c]     = (1/16) * sum_{i=0..15, row=g*16+i<256} |p(row,r,c)|^2
 *   BPlu[g,r,c]  = (1/16) * sum_{i=0..15, row=g*16+i<256} p(row,r,c)
 *   Bsum[r,c]    = (1/17) * sum_g B[g,r,c]
 *   Csum[a,b]    = (1/17) * sum_g B[g,a*8,b*8]
 *
 * Important shape note:
 *   The user-provided loops use g in range(17) and i in range(16), which address
 *   272 rows, but A is declared [256, 8*n].  This kernel treats rows >= 256 as
 *   zero and still divides every group by 16, exactly preserving the written
 *   B/=16 and final /=17 behavior without reading out of bounds.
 *   If your real input is [272, 8*n], change A_ROWS to 272.
 */

#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t A_ROWS = 256;
constexpr int32_t GROUPS = 17;
constexpr int32_t ROWS_PER_GROUP = 16;
constexpr float INV_ROWS_PER_GROUP = 1.0f / 16.0f;
constexpr float INV_GROUPS = 1.0f / 17.0f;

class KernelComplexGram {
public:
    __aicore__ inline KernelComplexGram() {}

    __aicore__ inline void Init(GM_ADDR ar, GM_ADDR ai,
                                GM_ADDR b, GM_ADDR bPlur, GM_ADDR bPlui,
                                GM_ADDR bsum, GM_ADDR csum,
                                int32_t userNum)
    {
        n = userNum;
        k = userNum * 8;
        totalK2 = static_cast<int64_t>(k) * static_cast<int64_t>(k);
        arGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ar), static_cast<uint64_t>(A_ROWS) * k);
        aiGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ai), static_cast<uint64_t>(A_ROWS) * k);
        bGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(b), static_cast<uint64_t>(GROUPS) * totalK2);
        bPlurGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bPlur), static_cast<uint64_t>(GROUPS) * totalK2);
        bPluiGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bPlui), static_cast<uint64_t>(GROUPS) * totalK2);
        bsumGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bsum), totalK2);
        csumGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(csum), static_cast<uint64_t>(n) * n);
    }

    __aicore__ inline void Process()
    {
        const int32_t blockId = GetBlockIdx();
        const int32_t blockNum = GetBlockNum();

        // 1) Produce B/BPlu for all groups and matrix elements.
        const int64_t totalB = static_cast<int64_t>(GROUPS) * totalK2;
        for (int64_t idx = blockId; idx < totalB; idx += blockNum) {
            const int32_t g = static_cast<int32_t>(idx / totalK2);
            const int64_t rem = idx - static_cast<int64_t>(g) * totalK2;
            const int32_t r = static_cast<int32_t>(rem / k);
            const int32_t c = static_cast<int32_t>(rem - static_cast<int64_t>(r) * k);

            float sumNorm = 0.0f;
            float sumRe = 0.0f;
            float sumIm = 0.0f;
            AccumulateOneGroup(g, r, c, sumNorm, sumRe, sumIm);

            bGm.SetValue(idx, sumNorm * INV_ROWS_PER_GROUP);
            bPlurGm.SetValue(idx, sumRe * INV_ROWS_PER_GROUP);
            bPluiGm.SetValue(idx, sumIm * INV_ROWS_PER_GROUP);
        }

        // 2) Produce Bsum directly by recomputation; avoids cross-core atomics/global sync.
        for (int64_t idx = blockId; idx < totalK2; idx += blockNum) {
            const int32_t r = static_cast<int32_t>(idx / k);
            const int32_t c = static_cast<int32_t>(idx - static_cast<int64_t>(r) * k);
            float sumNormAllGroups = 0.0f;
            for (int32_t g = 0; g < GROUPS; ++g) {
                float sumNorm = 0.0f;
                float dummyRe = 0.0f;
                float dummyIm = 0.0f;
                AccumulateOneGroup(g, r, c, sumNorm, dummyRe, dummyIm);
                sumNormAllGroups += sumNorm * INV_ROWS_PER_GROUP;
            }
            bsumGm.SetValue(idx, sumNormAllGroups * INV_GROUPS);
        }

        // 3) Produce Csum[n,n] from B[g, a*8, b*8] definition, also by recomputation.
        const int64_t totalN2 = static_cast<int64_t>(n) * n;
        for (int64_t idx = blockId; idx < totalN2; idx += blockNum) {
            const int32_t a = static_cast<int32_t>(idx / n);
            const int32_t b = static_cast<int32_t>(idx - static_cast<int64_t>(a) * n);
            const int32_t r = a * 8;
            const int32_t c = b * 8;
            float sumNormAllGroups = 0.0f;
            for (int32_t g = 0; g < GROUPS; ++g) {
                float sumNorm = 0.0f;
                float dummyRe = 0.0f;
                float dummyIm = 0.0f;
                AccumulateOneGroup(g, r, c, sumNorm, dummyRe, dummyIm);
                sumNormAllGroups += sumNorm * INV_ROWS_PER_GROUP;
            }
            csumGm.SetValue(idx, sumNormAllGroups * INV_GROUPS);
        }
    }

private:
    __aicore__ inline void AccumulateOneGroup(int32_t g, int32_t r, int32_t c,
                                              float &sumNorm, float &sumRe, float &sumIm)
    {
        const int32_t baseRow = g * ROWS_PER_GROUP;
        for (int32_t i = 0; i < ROWS_PER_GROUP; ++i) {
            const int32_t row = baseRow + i;
            if (row >= A_ROWS) {
                continue;
            }
            const int64_t offR = static_cast<int64_t>(row) * k + r;
            const int64_t offC = static_cast<int64_t>(row) * k + c;

            const float arR = arGm.GetValue(offR);
            const float aiR = aiGm.GetValue(offR);
            const float arC = arGm.GetValue(offC);
            const float aiC = aiGm.GetValue(offC);

            // conj(arR + j*aiR) * (arC + j*aiC)
            const float re = arR * arC + aiR * aiC;
            const float im = arR * aiC - aiR * arC;
            sumRe += re;
            sumIm += im;
            sumNorm += re * re + im * im;
        }
    }

private:
    int32_t n = 0;
    int32_t k = 0;
    int64_t totalK2 = 0;
    GlobalTensor<float> arGm;
    GlobalTensor<float> aiGm;
    GlobalTensor<float> bGm;
    GlobalTensor<float> bPlurGm;
    GlobalTensor<float> bPluiGm;
    GlobalTensor<float> bsumGm;
    GlobalTensor<float> csumGm;
};

extern "C" __global__ __aicore__ void complex_gram_kernel(GM_ADDR ar, GM_ADDR ai,
                                                           GM_ADDR b,
                                                           GM_ADDR bPlur,
                                                           GM_ADDR bPlui,
                                                           GM_ADDR bsum,
                                                           GM_ADDR csum,
                                                           int32_t n)
{
    KernelComplexGram op;
    op.Init(ar, ai, b, bPlur, bPlui, bsum, csum, n);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
// Kernel launch helper for AscendC invocation from host-side generated stub.
void complex_gram_do(uint32_t blockDim, void *stream, uint8_t *ar, uint8_t *ai,
                     uint8_t *b, uint8_t *bPlur, uint8_t *bPlui,
                     uint8_t *bsum, uint8_t *csum, int32_t n)
{
    complex_gram_kernel<<<blockDim, nullptr, stream>>>(ar, ai, b, bPlur, bPlui, bsum, csum, n);
}
#endif
