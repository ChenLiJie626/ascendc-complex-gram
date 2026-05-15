/*
 * AscendC complex Gram fused operator, cube + vector pipeline.
 *
 * Requested tensors:
 *   Ar/Ai  : [272, 256, 8*n]
 *   B      : [17, 8*n, 8*n]
 *   BPlur  : [17, 8*n, 8*n]
 *   BPlui  : [17, 8*n, 8*n]
 *   Bsum   : [8*n, 8*n]
 *   Csum   : [n, n]
 *
 * Math:
 *   P_s = A_s^H @ A_s
 *       = (Ar_s^T Ar_s + Ai_s^T Ai_s)
 *         + j(Ar_s^T Ai_s - Ai_s^T Ar_s)
 *   B[g]     = (1/16) * sum_{i=0..15} (real(P)^2 + imag(P)^2)
 *   BPlur[g] = (1/16) * sum real(P)
 *   BPlui[g] = (1/16) * sum imag(P)
 *   Bsum     = (1/17) * sum_g B[g]
 *   Csum[a,b]= (1/17) * sum_g B[g,a*8,b*8]
 *
 * Implementation design:
 *   1. complex_gram_cube_kernel runs on cube/AIC and computes four real GEMMs
 *      for each slice s:
 *          rr = Ar^T @ Ar, ii = Ai^T @ Ai, ri = Ar^T @ Ai, ir = Ai^T @ Ar
 *      into a temporary workspace.
 *   2. complex_gram_vector_epilogue_kernel runs after cube kernel in the same stream.
 *      Stream order is the global wait: vector does not start until cube has finished.
 *      It fuses complex reconstruction, norm, /16, and B/BPlu stores.
 *   3. complex_gram_vector_reduce_kernel runs after epilogue and produces Bsum/Csum.
 *
 * Workspace layout, all float, contiguous:
 *   tmpRR[272,K,K] | tmpII[272,K,K] | tmpRI[272,K,K] | tmpIR[272,K,K]
 *   workspace bytes = 4 * 272 * K * K * sizeof(float), K=8*n.
 *
 * Notes for integration:
 *   - The Matmul high-level API needs normal CANN/AscendC tiling generation in the
 *     op host/tiling side. This file is written as a kernel-side template; wire the
 *     tiling object according to your CANN version's Matmul examples.
 *   - For best cube performance, store Ar/Ai as fp16/bf16 if accuracy allows. If
 *     your input is true fp32, use the platform-supported fp32/HF32 Matmul mode or
 *     cast in a preceding stage. The output and workspace here are float.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

constexpr int32_t OUTER = 272;
constexpr int32_t M_DIM = 256;
constexpr int32_t GROUPS = 17;
constexpr int32_t SLICES_PER_GROUP = 16;
constexpr float INV_16 = 1.0f / 16.0f;
constexpr float INV_17 = 1.0f / 17.0f;
constexpr int32_t VEC_TILE = 256;  // floats; adjust by UB budget if you add double buffering.

// Keep this alias in one place so you can switch half/bfloat16/float input easily.
// If Ar/Ai are fp32 in GM, change InT to float and enable the corresponding Matmul mode
// supported by your AscendC/CANN release.
using InT = half;
using OutT = float;

using AType = MatmulType<TPosition::GM, CubeFormat::ND, InT>;
using BType = MatmulType<TPosition::GM, CubeFormat::ND, InT>;
using CType = MatmulType<TPosition::GM, CubeFormat::ND, OutT>;
using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, OutT>;

class ComplexGramCubeKernel {
public:
    __aicore__ inline void Init(GM_ADDR ar, GM_ADDR ai, GM_ADDR workspace, int32_t userNum)
    {
        n = userNum;
        k = userNum * 8;
        k2 = static_cast<int64_t>(k) * k;
        sliceInSize = static_cast<int64_t>(M_DIM) * k;
        planeSize = static_cast<int64_t>(OUTER) * k2;

        arGm.SetGlobalBuffer(reinterpret_cast<__gm__ InT *>(ar), static_cast<uint64_t>(OUTER) * M_DIM * k);
        aiGm.SetGlobalBuffer(reinterpret_cast<__gm__ InT *>(ai), static_cast<uint64_t>(OUTER) * M_DIM * k);
        tmpRRGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutT *>(workspace), planeSize);
        tmpIIGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutT *>(workspace) + planeSize, planeSize);
        tmpRIGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutT *>(workspace) + 2 * planeSize, planeSize);
        tmpIRGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutT *>(workspace) + 3 * planeSize, planeSize);
    }

    __aicore__ inline void Process()
    {
        // One AIC task handles one or more 256xK slices. Matmul computes KxK = A^T @ A.
        const int32_t blockId = GetBlockIdx();
        const int32_t blockNum = GetBlockNum();
        for (int32_t s = blockId; s < OUTER; s += blockNum) {
            const int64_t inOff = static_cast<int64_t>(s) * sliceInSize;
            const int64_t outOff = static_cast<int64_t>(s) * k2;
            GemmATransBNoTrans(arGm[inOff], arGm[inOff], tmpRRGm[outOff]); // Ar^T Ar
            GemmATransBNoTrans(aiGm[inOff], aiGm[inOff], tmpIIGm[outOff]); // Ai^T Ai
            GemmATransBNoTrans(arGm[inOff], aiGm[inOff], tmpRIGm[outOff]); // Ar^T Ai
            GemmATransBNoTrans(aiGm[inOff], arGm[inOff], tmpIRGm[outOff]); // Ai^T Ar
        }
    }

private:
    __aicore__ inline void GemmATransBNoTrans(const GlobalTensor<InT> &a,
                                              const GlobalTensor<InT> &b,
                                              const GlobalTensor<OutT> &c)
    {
        // This is the standard AscendC high-level Matmul flow. In a full custom-op
        // project, bind the generated matmul tiling before IterateAll if your CANN
        // version requires it (for example via Init/SetTiling on the Matmul object).
        Matmul<AType, BType, CType, BiasType> mm;
        mm.SetOrgShape(M_DIM, k, k); // original A is [M_DIM,K], B is [M_DIM,K], output [K,K]
        mm.SetTensorA(a, true);      // transpose A: [K,M_DIM]
        mm.SetTensorB(b, false);     // B: [M_DIM,K]
        mm.IterateAll(c, false);     // false: no bias
        mm.End();
    }

private:
    int32_t n = 0;
    int32_t k = 0;
    int64_t k2 = 0;
    int64_t sliceInSize = 0;
    int64_t planeSize = 0;
    GlobalTensor<InT> arGm;
    GlobalTensor<InT> aiGm;
    GlobalTensor<OutT> tmpRRGm;
    GlobalTensor<OutT> tmpIIGm;
    GlobalTensor<OutT> tmpRIGm;
    GlobalTensor<OutT> tmpIRGm;
};

class ComplexGramVectorEpilogueKernel {
public:
    __aicore__ inline void Init(GM_ADDR workspace, GM_ADDR b, GM_ADDR bPlur, GM_ADDR bPlui, int32_t userNum)
    {
        n = userNum;
        k = userNum * 8;
        k2 = static_cast<int64_t>(k) * k;
        planeSize = static_cast<int64_t>(OUTER) * k2;

        tmpRRGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutT *>(workspace), planeSize);
        tmpIIGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutT *>(workspace) + planeSize, planeSize);
        tmpRIGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutT *>(workspace) + 2 * planeSize, planeSize);
        tmpIRGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutT *>(workspace) + 3 * planeSize, planeSize);
        bGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutT *>(b), static_cast<uint64_t>(GROUPS) * k2);
        bPlurGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutT *>(bPlur), static_cast<uint64_t>(GROUPS) * k2);
        bPluiGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutT *>(bPlui), static_cast<uint64_t>(GROUPS) * k2);

        pipe.InitBuffer(bufRR, VEC_TILE * sizeof(OutT));
        pipe.InitBuffer(bufII, VEC_TILE * sizeof(OutT));
        pipe.InitBuffer(bufRI, VEC_TILE * sizeof(OutT));
        pipe.InitBuffer(bufIR, VEC_TILE * sizeof(OutT));
        pipe.InitBuffer(bufRe, VEC_TILE * sizeof(OutT));
        pipe.InitBuffer(bufIm, VEC_TILE * sizeof(OutT));
        pipe.InitBuffer(bufNorm, VEC_TILE * sizeof(OutT));
        pipe.InitBuffer(bufTmp, VEC_TILE * sizeof(OutT));
    }

    __aicore__ inline void Process()
    {
        const int32_t blockId = GetBlockIdx();
        const int32_t blockNum = GetBlockNum();
        const int64_t total = static_cast<int64_t>(GROUPS) * k2;

        for (int64_t base = static_cast<int64_t>(blockId) * VEC_TILE; base < total; base += static_cast<int64_t>(blockNum) * VEC_TILE) {
            const int32_t len = static_cast<int32_t>((base + VEC_TILE <= total) ? VEC_TILE : (total - base));
            LocalTensor<OutT> rr = bufRR.Get<OutT>();
            LocalTensor<OutT> ii = bufII.Get<OutT>();
            LocalTensor<OutT> ri = bufRI.Get<OutT>();
            LocalTensor<OutT> ir = bufIR.Get<OutT>();
            LocalTensor<OutT> re = bufRe.Get<OutT>();
            LocalTensor<OutT> im = bufIm.Get<OutT>();
            LocalTensor<OutT> norm = bufNorm.Get<OutT>();
            LocalTensor<OutT> tmp = bufTmp.Get<OutT>();

            Duplicate(re, static_cast<OutT>(0.0f), len);
            Duplicate(im, static_cast<OutT>(0.0f), len);
            Duplicate(norm, static_cast<OutT>(0.0f), len);

            const int32_t g0 = static_cast<int32_t>(base / k2);
            const int64_t elem0 = base - static_cast<int64_t>(g0) * k2;

            // The loop is tile-friendly when the tile does not cross a group boundary.
            // For simplicity, VEC_TILE should be <= k2 and block partition should be chosen
            // so the common case is within one g. If a tile crosses g, scalar cleanup below
            // still preserves correctness.
            if (elem0 + len <= k2) {
                for (int32_t i = 0; i < SLICES_PER_GROUP; ++i) {
                    const int32_t s = g0 * SLICES_PER_GROUP + i;
                    const int64_t off = static_cast<int64_t>(s) * k2 + elem0;
                    DataCopy(rr, tmpRRGm[off], len);
                    DataCopy(ii, tmpIIGm[off], len);
                    DataCopy(ri, tmpRIGm[off], len);
                    DataCopy(ir, tmpIRGm[off], len);

                    Add(tmp, rr, ii, len);      // tmp = rr + ii = real(P)
                    Add(re, re, tmp, len);      // accumulate real
                    Sub(tmp, ri, ir, len);      // tmp = ri - ir = imag(P)
                    Add(im, im, tmp, len);      // accumulate imag
                }

                // Norm must be sum over slices of per-slice |P_s|^2, not |sum P_s|^2.
                // Compute it in a second pass with vector ops.
                Duplicate(norm, static_cast<OutT>(0.0f), len);
                for (int32_t i = 0; i < SLICES_PER_GROUP; ++i) {
                    const int32_t s = g0 * SLICES_PER_GROUP + i;
                    const int64_t off = static_cast<int64_t>(s) * k2 + elem0;
                    DataCopy(rr, tmpRRGm[off], len);
                    DataCopy(ii, tmpIIGm[off], len);
                    DataCopy(ri, tmpRIGm[off], len);
                    DataCopy(ir, tmpIRGm[off], len);
                    Add(rr, rr, ii, len);       // rr = real(P_s)
                    Sub(ri, ri, ir, len);       // ri = imag(P_s)
                    Mul(tmp, rr, rr, len);      // tmp = real^2
                    Mul(ii, ri, ri, len);       // ii = imag^2
                    Add(tmp, tmp, ii, len);     // tmp = norm
                    Add(norm, norm, tmp, len);
                }

                Muls(re, re, static_cast<OutT>(INV_16), len);
                Muls(im, im, static_cast<OutT>(INV_16), len);
                Muls(norm, norm, static_cast<OutT>(INV_16), len);
                DataCopy(bGm[base], norm, len);
                DataCopy(bPlurGm[base], re, len);
                DataCopy(bPluiGm[base], im, len);
            } else {
                // Rare boundary tile crossing group; scalar path keeps code robust.
                for (int32_t t = 0; t < len; ++t) {
                    const int64_t idx = base + t;
                    const int32_t g = static_cast<int32_t>(idx / k2);
                    const int64_t elem = idx - static_cast<int64_t>(g) * k2;
                    OutT sumRe = 0.0f, sumIm = 0.0f, sumNorm = 0.0f;
                    for (int32_t i = 0; i < SLICES_PER_GROUP; ++i) {
                        const int32_t s = g * SLICES_PER_GROUP + i;
                        const int64_t off = static_cast<int64_t>(s) * k2 + elem;
                        const OutT real = tmpRRGm.GetValue(off) + tmpIIGm.GetValue(off);
                        const OutT imag = tmpRIGm.GetValue(off) - tmpIRGm.GetValue(off);
                        sumRe += real;
                        sumIm += imag;
                        sumNorm += real * real + imag * imag;
                    }
                    bGm.SetValue(idx, sumNorm * INV_16);
                    bPlurGm.SetValue(idx, sumRe * INV_16);
                    bPluiGm.SetValue(idx, sumIm * INV_16);
                }
            }
        }
    }

private:
    int32_t n = 0;
    int32_t k = 0;
    int64_t k2 = 0;
    int64_t planeSize = 0;
    TPipe pipe;
    TBuf<TPosition::VECCALC> bufRR, bufII, bufRI, bufIR, bufRe, bufIm, bufNorm, bufTmp;
    GlobalTensor<OutT> tmpRRGm, tmpIIGm, tmpRIGm, tmpIRGm;
    GlobalTensor<OutT> bGm, bPlurGm, bPluiGm;
};

class ComplexGramVectorReduceKernel {
public:
    __aicore__ inline void Init(GM_ADDR b, GM_ADDR bsum, GM_ADDR csum, int32_t userNum)
    {
        n = userNum;
        k = userNum * 8;
        k2 = static_cast<int64_t>(k) * k;
        bGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutT *>(b), static_cast<uint64_t>(GROUPS) * k2);
        bsumGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutT *>(bsum), k2);
        csumGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutT *>(csum), static_cast<uint64_t>(n) * n);
    }

    __aicore__ inline void Process()
    {
        const int32_t blockId = GetBlockIdx();
        const int32_t blockNum = GetBlockNum();

        for (int64_t idx = blockId; idx < k2; idx += blockNum) {
            OutT acc = 0.0f;
            for (int32_t g = 0; g < GROUPS; ++g) {
                acc += bGm.GetValue(static_cast<int64_t>(g) * k2 + idx);
            }
            bsumGm.SetValue(idx, acc * INV_17);
        }

        const int64_t n2 = static_cast<int64_t>(n) * n;
        for (int64_t idx = blockId; idx < n2; idx += blockNum) {
            const int32_t a = static_cast<int32_t>(idx / n);
            const int32_t b = static_cast<int32_t>(idx - static_cast<int64_t>(a) * n);
            const int64_t elem = static_cast<int64_t>(a * 8) * k + b * 8;
            OutT acc = 0.0f;
            for (int32_t g = 0; g < GROUPS; ++g) {
                acc += bGm.GetValue(static_cast<int64_t>(g) * k2 + elem);
            }
            csumGm.SetValue(idx, acc * INV_17);
        }
    }

private:
    int32_t n = 0;
    int32_t k = 0;
    int64_t k2 = 0;
    GlobalTensor<OutT> bGm, bsumGm, csumGm;
};

extern "C" __global__ __aicore__ void complex_gram_cube_kernel(GM_ADDR ar, GM_ADDR ai,
                                                                 GM_ADDR workspace,
                                                                 int32_t n)
{
    ComplexGramCubeKernel op;
    op.Init(ar, ai, workspace, n);
    op.Process();
}

extern "C" __global__ __aicore__ void complex_gram_vector_epilogue_kernel(GM_ADDR workspace,
                                                                           GM_ADDR b,
                                                                           GM_ADDR bPlur,
                                                                           GM_ADDR bPlui,
                                                                           int32_t n)
{
    ComplexGramVectorEpilogueKernel op;
    op.Init(workspace, b, bPlur, bPlui, n);
    op.Process();
}

extern "C" __global__ __aicore__ void complex_gram_vector_reduce_kernel(GM_ADDR b,
                                                                         GM_ADDR bsum,
                                                                         GM_ADDR csum,
                                                                         int32_t n)
{
    ComplexGramVectorReduceKernel op;
    op.Init(b, bsum, csum, n);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
// Host-side launch helper. Launch all kernels on the same stream; this is the required
// cube -> vector wait/order. Allocate workspace before calling this function.
static inline uint64_t complex_gram_workspace_bytes(int32_t n)
{
    const int64_t k = static_cast<int64_t>(n) * 8;
    return static_cast<uint64_t>(4LL * OUTER * k * k * sizeof(OutT));
}

void complex_gram_fused_do(uint32_t cubeBlockDim,
                           uint32_t vectorBlockDim,
                           void *stream,
                           uint8_t *ar,
                           uint8_t *ai,
                           uint8_t *workspace,
                           uint8_t *b,
                           uint8_t *bPlur,
                           uint8_t *bPlui,
                           uint8_t *bsum,
                           uint8_t *csum,
                           int32_t n)
{
    complex_gram_cube_kernel<<<cubeBlockDim, nullptr, stream>>>(ar, ai, workspace, n);
    complex_gram_vector_epilogue_kernel<<<vectorBlockDim, nullptr, stream>>>(workspace, b, bPlur, bPlui, n);
    complex_gram_vector_reduce_kernel<<<vectorBlockDim, nullptr, stream>>>(b, bsum, csum, n);
}
#endif
