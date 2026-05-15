/*
 * ComplexGram fused AscendC kernel template, organized like official custom-op samples.
 *
 * Synchronization model used here:
 *   - cube kernel, vector epilogue kernel, and vector reduce kernel are launched on the
 *     same stream in this exact order.
 *   - Ascend runtime stream ordering is the global synchronization: all GM writes from
 *     complex_gram_cube_kernel are complete and visible before
 *     complex_gram_vector_epilogue_kernel starts; all B writes are complete before
 *     complex_gram_vector_reduce_kernel starts.
 *   - Therefore this version does NOT use in-kernel SetFlag/WaitFlag between AIC/AIV.
 *     That avoids unsafe cross-block global synchronization assumptions.
 *
 * Logical 20 cube : 40 vector mapping:
 *   cube c owns task ids c, c+20, c+40, ...
 *   vector v maps to pairCube=v/2 and lane=v%2.
 *   vectors 2*c and 2*c+1 consume the same task sequence as cube c, splitting each
 *   BM*BN output tile into first/second half.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "complex_gram_tiling.h"

using namespace AscendC;

using InT = half;     // Change to float if your platform/CANN Matmul supports fp32/HF32 path.
using OutT = float;

using AType = MatmulType<TPosition::GM, CubeFormat::ND, InT>;
using BType = MatmulType<TPosition::GM, CubeFormat::ND, InT>;
using CType = MatmulType<TPosition::GM, CubeFormat::ND, OutT>;
using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, OutT>;

constexpr float INV_16 = 1.0f / 16.0f;
constexpr float INV_17 = 1.0f / 17.0f;

__aicore__ inline void DecodeTask(int32_t taskId, const ComplexGramTilingData &t,
                                  int32_t &g, int32_t &tm, int32_t &tn)
{
    const int32_t tilePerGroup = t.tileMNum * t.tileNNum;
    g = taskId / tilePerGroup;
    const int32_t r = taskId - g * tilePerGroup;
    tm = r / t.tileNNum;
    tn = r - tm * t.tileNNum;
}

__aicore__ inline int64_t TaskWorkspaceBase(int32_t taskId, const ComplexGramTilingData &t)
{
    return static_cast<int64_t>(taskId) * static_cast<int64_t>(t.oneTaskTmpElems);
}

class ComplexGramCubeTiled {
public:
    __aicore__ inline void Init(GM_ADDR ar, GM_ADDR ai, GM_ADDR workspace, GM_ADDR tiling)
    {
        t = *reinterpret_cast<__gm__ ComplexGramTilingData *>(tiling);
        arGm.SetGlobalBuffer(reinterpret_cast<__gm__ InT *>(ar), static_cast<uint64_t>(272) * t.mDim * t.k);
        aiGm.SetGlobalBuffer(reinterpret_cast<__gm__ InT *>(ai), static_cast<uint64_t>(272) * t.mDim * t.k);
        wsGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutT *>(workspace), t.workspaceBytes / sizeof(OutT));
    }

    __aicore__ inline void Process()
    {
        const int32_t cubeId = GetBlockIdx();
        // Use t.cubeBlockNum rather than GetBlockNum() so vector can mirror this exact task mapping.
        for (int32_t task = cubeId; task < t.taskNum; task += t.cubeBlockNum) {
            int32_t g, tm, tn;
            DecodeTask(task, t, g, tm, tn);
            const int32_t row0 = tm * t.bm;
            const int32_t col0 = tn * t.bn;
            const int32_t actualBm = (row0 + t.bm <= t.k) ? t.bm : (t.k - row0);
            const int32_t actualBn = (col0 + t.bn <= t.k) ? t.bn : (t.k - col0);
            const int64_t taskBase = TaskWorkspaceBase(task, t);
            const int64_t onePlane = static_cast<int64_t>(t.slicesPerGroup) * t.bm * t.bn;

            for (int32_t i = 0; i < t.slicesPerGroup; ++i) {
                const int32_t s = g * t.slicesPerGroup + i;
                const int64_t arLeft = static_cast<int64_t>(s) * t.mDim * t.k + row0;
                const int64_t aiLeft = static_cast<int64_t>(s) * t.mDim * t.k + row0;
                const int64_t arRight = static_cast<int64_t>(s) * t.mDim * t.k + col0;
                const int64_t aiRight = static_cast<int64_t>(s) * t.mDim * t.k + col0;
                const int64_t tileOff = static_cast<int64_t>(i) * t.bm * t.bn;

                // Each output tile stores compact [BM,BN] with padding on boundary tiles.
                GemmTileATransBNoTrans(arGm[arLeft], arGm[arRight], wsGm[taskBase + tileOff], actualBm, actualBn);
                GemmTileATransBNoTrans(aiGm[aiLeft], aiGm[aiRight], wsGm[taskBase + onePlane + tileOff], actualBm, actualBn);
                GemmTileATransBNoTrans(arGm[arLeft], aiGm[aiRight], wsGm[taskBase + 2 * onePlane + tileOff], actualBm, actualBn);
                GemmTileATransBNoTrans(aiGm[aiLeft], arGm[arRight], wsGm[taskBase + 3 * onePlane + tileOff], actualBm, actualBn);
            }
        }
    }

private:
    __aicore__ inline void GemmTileATransBNoTrans(const GlobalTensor<InT> &a,
                                                  const GlobalTensor<InT> &b,
                                                  const GlobalTensor<OutT> &c,
                                                  int32_t actualBm,
                                                  int32_t actualBn)
    {
        Matmul<AType, BType, CType, BiasType> mm;
        // A original slice is [256,K]. Pointer a starts at column row0, pointer b at column col0.
        // The complete official integration should pass Matmul tiling generated for actualBm x actualBn x 256.
        mm.SetOrgShape(t.mDim, actualBm, actualBn);
        mm.SetTensorA(a, true);   // [256, actualBm] -> [actualBm, 256]
        mm.SetTensorB(b, false);  // [256, actualBn]
        mm.IterateAll(c, false);
        mm.End();
    }

private:
    ComplexGramTilingData t{};
    GlobalTensor<InT> arGm, aiGm;
    GlobalTensor<OutT> wsGm;
};

class ComplexGramVectorEpilogueTiled {
public:
    __aicore__ inline void Init(GM_ADDR workspace, GM_ADDR b, GM_ADDR bPlur, GM_ADDR bPlui, GM_ADDR tiling)
    {
        t = *reinterpret_cast<__gm__ ComplexGramTilingData *>(tiling);
        wsGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutT *>(workspace), t.workspaceBytes / sizeof(OutT));
        bGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutT *>(b), static_cast<uint64_t>(t.groupNum) * t.k * t.k);
        bPlurGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutT *>(bPlur), static_cast<uint64_t>(t.groupNum) * t.k * t.k);
        bPluiGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutT *>(bPlui), static_cast<uint64_t>(t.groupNum) * t.k * t.k);
    }

    __aicore__ inline void Process()
    {
        const int32_t v = GetBlockIdx();
        const int32_t pairCube = v / 2;
        const int32_t lane = v & 1;
        for (int32_t task = pairCube; task < t.taskNum; task += t.cubeBlockNum) {
            ConsumeTask(task, lane);
        }
    }

private:
    __aicore__ inline void ConsumeTask(int32_t task, int32_t lane)
    {
        int32_t g, tm, tn;
        DecodeTask(task, t, g, tm, tn);
        const int32_t row0 = tm * t.bm;
        const int32_t col0 = tn * t.bn;
        const int32_t actualBm = (row0 + t.bm <= t.k) ? t.bm : (t.k - row0);
        const int32_t actualBn = (col0 + t.bn <= t.k) ? t.bn : (t.k - col0);
        const int32_t totalElem = actualBm * actualBn;
        const int32_t laneBegin = (lane == 0) ? 0 : ((totalElem + 1) / 2);
        const int32_t laneEnd = (lane == 0) ? ((totalElem + 1) / 2) : totalElem;

        const int64_t taskBase = TaskWorkspaceBase(task, t);
        const int64_t onePlane = static_cast<int64_t>(t.slicesPerGroup) * t.bm * t.bn;

        for (int32_t e = laneBegin; e < laneEnd; ++e) {
            const int32_t tr = e / actualBn;
            const int32_t tc = e - tr * actualBn;
            const int64_t compactOff = static_cast<int64_t>(tr) * t.bm + tc;
            OutT sumRe = 0.0f;
            OutT sumIm = 0.0f;
            OutT sumNorm = 0.0f;
            for (int32_t i = 0; i < t.slicesPerGroup; ++i) {
                const int64_t sliceOff = static_cast<int64_t>(i) * t.bm * t.bn + compactOff;
                const OutT rr = wsGm.GetValue(taskBase + sliceOff);
                const OutT ii = wsGm.GetValue(taskBase + onePlane + sliceOff);
                const OutT ri = wsGm.GetValue(taskBase + 2 * onePlane + sliceOff);
                const OutT ir = wsGm.GetValue(taskBase + 3 * onePlane + sliceOff);
                const OutT re = rr + ii;
                const OutT im = ri - ir;
                sumRe += re;
                sumIm += im;
                sumNorm += re * re + im * im;
            }
            const int32_t globalRow = row0 + tr;
            const int32_t globalCol = col0 + tc;
            const int64_t out = static_cast<int64_t>(g) * t.k * t.k + static_cast<int64_t>(globalRow) * t.k + globalCol;
            bGm.SetValue(out, sumNorm * INV_16);
            bPlurGm.SetValue(out, sumRe * INV_16);
            bPluiGm.SetValue(out, sumIm * INV_16);
        }
    }

private:
    ComplexGramTilingData t{};
    GlobalTensor<OutT> wsGm, bGm, bPlurGm, bPluiGm;
};

class ComplexGramVectorReduce {
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
        const int32_t v = GetBlockIdx();
        const int32_t vNum = GetBlockNum();
        const int64_t k2 = static_cast<int64_t>(t.k) * t.k;
        for (int64_t idx = v; idx < k2; idx += vNum) {
            OutT acc = 0.0f;
            for (int32_t g = 0; g < t.groupNum; ++g) {
                acc += bGm.GetValue(static_cast<int64_t>(g) * k2 + idx);
            }
            bsumGm.SetValue(idx, acc * INV_17);
        }
        const int64_t n2 = static_cast<int64_t>(t.n) * t.n;
        for (int64_t idx = v; idx < n2; idx += vNum) {
            const int32_t a = static_cast<int32_t>(idx / t.n);
            const int32_t b = static_cast<int32_t>(idx - static_cast<int64_t>(a) * t.n);
            const int64_t elem = static_cast<int64_t>(a * 8) * t.k + b * 8;
            OutT acc = 0.0f;
            for (int32_t g = 0; g < t.groupNum; ++g) {
                acc += bGm.GetValue(static_cast<int64_t>(g) * k2 + elem);
            }
            csumGm.SetValue(idx, acc * INV_17);
        }
    }

private:
    ComplexGramTilingData t{};
    GlobalTensor<OutT> bGm, bsumGm, csumGm;
};

extern "C" __global__ __aicore__ void complex_gram_cube_kernel(GM_ADDR ar, GM_ADDR ai,
                                                                 GM_ADDR workspace,
                                                                 GM_ADDR tiling)
{
    ComplexGramCubeTiled op;
    op.Init(ar, ai, workspace, tiling);
    op.Process();
}

extern "C" __global__ __aicore__ void complex_gram_vector_epilogue_kernel(GM_ADDR workspace,
                                                                           GM_ADDR b,
                                                                           GM_ADDR bPlur,
                                                                           GM_ADDR bPlui,
                                                                           GM_ADDR tiling)
{
    ComplexGramVectorEpilogueTiled op;
    op.Init(workspace, b, bPlur, bPlui, tiling);
    op.Process();
}

extern "C" __global__ __aicore__ void complex_gram_vector_reduce_kernel(GM_ADDR b,
                                                                         GM_ADDR bsum,
                                                                         GM_ADDR csum,
                                                                         GM_ADDR tiling)
{
    ComplexGramVectorReduce op;
    op.Init(b, bsum, csum, tiling);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void complex_gram_fused_do(void *stream,
                           uint8_t *ar, uint8_t *ai,
                           uint8_t *workspace,
                           uint8_t *b, uint8_t *bPlur, uint8_t *bPlui,
                           uint8_t *bsum, uint8_t *csum,
                           uint8_t *tiling)
{
    auto *t = reinterpret_cast<ComplexGramTilingData *>(tiling);
    // Same stream launch order is the cube-vector synchronization point.
    complex_gram_cube_kernel<<<t->cubeBlockNum, nullptr, stream>>>(ar, ai, workspace, tiling);
    complex_gram_vector_epilogue_kernel<<<t->vectorBlockNum, nullptr, stream>>>(workspace, b, bPlur, bPlui, tiling);
    complex_gram_vector_reduce_kernel<<<t->vectorBlockNum, nullptr, stream>>>(b, bsum, csum, tiling);
}
#endif
