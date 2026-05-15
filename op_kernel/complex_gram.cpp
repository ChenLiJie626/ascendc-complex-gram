/**
 * @file complex_gram.cpp
 * Official msOpGen-style kernel file for ComplexGram.
 *
 * Entry points provided:
 *   extern "C" __global__ __aicore__ void complex_gram(...)
 *   void complex_gram_do(...)
 *
 * The kernel is organized as one AIC/AIV fused producer-consumer kernel:
 *   - AIC/cube side computes GEMM tiles into per-cube ping-pong workspace.
 *   - AIV/vector side waits on ready flags, consumes tile halves, writes B/BPlur/BPlui.
 *   - Bsum/Csum are recomputed directly from Ar/Ai on AIV side to avoid unsafe global
 *     barrier inside a single kernel.
 *
 * Adapt the four SetFlag/WaitFlag wrappers to your CANN version's official fusion sample.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "complex_gram_tiling.h"

using namespace AscendC;
using InT = half;
using OutT = float;

using AType = MatmulType<TPosition::GM, CubeFormat::ND, InT>;
using BType = MatmulType<TPosition::GM, CubeFormat::ND, InT>;
using CType = MatmulType<TPosition::GM, CubeFormat::ND, OutT>;
using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, OutT>;

constexpr float INV_16 = 1.0f / 16.0f;
constexpr float INV_17 = 1.0f / 17.0f;
constexpr float INV_272 = 1.0f / 272.0f;

__aicore__ inline void DecodeTask(uint32_t taskId, const ComplexGramTilingData &t,
                                  uint32_t &g, uint32_t &tm, uint32_t &tn)
{
    const uint32_t tilePerGroup = t.tileMNum * t.tileNNum;
    g = taskId / tilePerGroup;
    const uint32_t r = taskId - g * tilePerGroup;
    tm = r / t.tileNNum;
    tn = r - tm * t.tileNNum;
}

__aicore__ inline uint32_t ReadyFlagId(uint32_t cubeId, uint32_t lane, uint32_t buf, const ComplexGramTilingData &t)
{
    return cubeId * t.flagSlotsPerBuf * t.pingPong + buf * t.flagSlotsPerBuf + lane;
}

__aicore__ inline uint32_t DoneFlagId(uint32_t cubeId, uint32_t lane, uint32_t buf, const ComplexGramTilingData &t)
{
    return cubeId * t.flagSlotsPerBuf * t.pingPong + buf * t.flagSlotsPerBuf + 2 + lane;
}

__aicore__ inline int64_t PairBufferBase(uint32_t cubeId, uint32_t buf, const ComplexGramTilingData &t)
{
    return (static_cast<int64_t>(cubeId) * t.pingPong + buf) * static_cast<int64_t>(t.oneTaskTmpElems);
}

__aicore__ inline void NotifyVecLaneReady(uint32_t flagId)
{
#if defined(COMPLEX_GRAM_ENABLE_REAL_AIC_AIV_FLAGS)
    SetFlag<HardEvent::MTE3_MTE2>(flagId);
#else
    (void)flagId;
#endif
}

__aicore__ inline void WaitVecLaneReady(uint32_t flagId)
{
#if defined(COMPLEX_GRAM_ENABLE_REAL_AIC_AIV_FLAGS)
    WaitFlag<HardEvent::MTE3_MTE2>(flagId);
#else
    (void)flagId;
#endif
}

__aicore__ inline void NotifyCubeLaneDone(uint32_t flagId)
{
#if defined(COMPLEX_GRAM_ENABLE_REAL_AIC_AIV_FLAGS)
    SetFlag<HardEvent::MTE2_MTE3>(flagId);
#else
    (void)flagId;
#endif
}

__aicore__ inline void WaitCubeLaneDone(uint32_t flagId)
{
#if defined(COMPLEX_GRAM_ENABLE_REAL_AIC_AIV_FLAGS)
    WaitFlag<HardEvent::MTE2_MTE3>(flagId);
#else
    (void)flagId;
#endif
}

class KernelComplexGram {
public:
    __aicore__ inline KernelComplexGram() {}

    __aicore__ inline void Init(GM_ADDR ar, GM_ADDR ai,
                                GM_ADDR b, GM_ADDR bplur, GM_ADDR bplui,
                                GM_ADDR bsum, GM_ADDR csum,
                                GM_ADDR workspace,
                                const ComplexGramTilingData &tilingData)
    {
        t = tilingData;
        arGm.SetGlobalBuffer(reinterpret_cast<__gm__ InT *>(ar), static_cast<uint64_t>(272) * t.mDim * t.k);
        aiGm.SetGlobalBuffer(reinterpret_cast<__gm__ InT *>(ai), static_cast<uint64_t>(272) * t.mDim * t.k);
        bGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutT *>(b), static_cast<uint64_t>(t.groupNum) * t.k * t.k);
        bplurGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutT *>(bplur), static_cast<uint64_t>(t.groupNum) * t.k * t.k);
        bpluiGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutT *>(bplui), static_cast<uint64_t>(t.groupNum) * t.k * t.k);
        bsumGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutT *>(bsum), static_cast<uint64_t>(t.k) * t.k);
        csumGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutT *>(csum), static_cast<uint64_t>(t.n) * t.n);
        wsGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutT *>(workspace), t.singleKernelWorkspaceBytes / sizeof(OutT));
    }

    __aicore__ inline void Process()
    {
#if defined(__DAV_CUBE__)
        ProcessCube();
#else
        ProcessVector();
#endif
    }

private:
    __aicore__ inline void ProcessCube()
    {
        const uint32_t cubeId = GetBlockIdx();
        uint32_t localIdx = 0;
        for (uint32_t task = cubeId; task < t.taskNum; task += t.cubeBlockNum, ++localIdx) {
            const uint32_t buf = localIdx & 1;
            if (localIdx >= t.pingPong) {
                WaitCubeLaneDone(DoneFlagId(cubeId, 0, buf, t));
                WaitCubeLaneDone(DoneFlagId(cubeId, 1, buf, t));
            }
            ProduceTaskTile(task, cubeId, buf);
            NotifyVecLaneReady(ReadyFlagId(cubeId, 0, buf, t));
            NotifyVecLaneReady(ReadyFlagId(cubeId, 1, buf, t));
        }
        const uint32_t totalLocalTasks = (t.taskNum > cubeId) ? ((t.taskNum - 1 - cubeId) / t.cubeBlockNum + 1) : 0;
        const uint32_t drain = totalLocalTasks < t.pingPong ? totalLocalTasks : t.pingPong;
        for (uint32_t d = 0; d < drain; ++d) {
            const uint32_t lastLocalIdx = totalLocalTasks - 1 - d;
            const uint32_t buf = lastLocalIdx & 1;
            WaitCubeLaneDone(DoneFlagId(cubeId, 0, buf, t));
            WaitCubeLaneDone(DoneFlagId(cubeId, 1, buf, t));
        }
    }

    __aicore__ inline void ProcessVector()
    {
        const uint32_t vectorId = GetBlockIdx();
        const uint32_t pairCube = vectorId >> 1;
        const uint32_t lane = vectorId & 1;
        uint32_t localIdx = 0;
        for (uint32_t task = pairCube; task < t.taskNum; task += t.cubeBlockNum, ++localIdx) {
            const uint32_t buf = localIdx & 1;
            WaitVecLaneReady(ReadyFlagId(pairCube, lane, buf, t));
            ConsumeTaskTile(task, pairCube, lane, buf);
            NotifyCubeLaneDone(DoneFlagId(pairCube, lane, buf, t));
        }
        // Single-kernel complete output: recompute Bsum/Csum directly from input so
        // no global barrier over B is required.
        RecomputeBsumAndCsum(vectorId, GetBlockNum());
    }

    __aicore__ inline void ProduceTaskTile(uint32_t task, uint32_t cubeId, uint32_t buf)
    {
        uint32_t g, tm, tn;
        DecodeTask(task, t, g, tm, tn);
        const uint32_t row0 = tm * t.bm;
        const uint32_t col0 = tn * t.bn;
        const uint32_t actualBm = (row0 + t.bm <= t.k) ? t.bm : (t.k - row0);
        const uint32_t actualBn = (col0 + t.bn <= t.k) ? t.bn : (t.k - col0);
        const int64_t base = PairBufferBase(cubeId, buf, t);
        const int64_t onePlane = static_cast<int64_t>(t.slicesPerGroup) * t.bm * t.bn;

        for (uint32_t i = 0; i < t.slicesPerGroup; ++i) {
            const uint32_t s = g * t.slicesPerGroup + i;
            const int64_t arLeft = static_cast<int64_t>(s) * t.mDim * t.k + row0;
            const int64_t aiLeft = static_cast<int64_t>(s) * t.mDim * t.k + row0;
            const int64_t arRight = static_cast<int64_t>(s) * t.mDim * t.k + col0;
            const int64_t aiRight = static_cast<int64_t>(s) * t.mDim * t.k + col0;
            const int64_t tileOff = static_cast<int64_t>(i) * t.bm * t.bn;
            GemmTile(arGm[arLeft], arGm[arRight], wsGm[base + tileOff], actualBm, actualBn);
            GemmTile(aiGm[aiLeft], aiGm[aiRight], wsGm[base + onePlane + tileOff], actualBm, actualBn);
            GemmTile(arGm[arLeft], aiGm[aiRight], wsGm[base + 2 * onePlane + tileOff], actualBm, actualBn);
            GemmTile(aiGm[aiLeft], arGm[arRight], wsGm[base + 3 * onePlane + tileOff], actualBm, actualBn);
        }
    }

    __aicore__ inline void GemmTile(const GlobalTensor<InT> &a,
                                    const GlobalTensor<InT> &b,
                                    const GlobalTensor<OutT> &c,
                                    uint32_t actualBm,
                                    uint32_t actualBn)
    {
        Matmul<AType, BType, CType, BiasType> mm;
        mm.SetOrgShape(t.mDim, actualBm, actualBn);
        mm.SetTensorA(a, true);
        mm.SetTensorB(b, false);
        mm.IterateAll(c, false);
        mm.End();
    }

    __aicore__ inline void ConsumeTaskTile(uint32_t task, uint32_t cubeId, uint32_t lane, uint32_t buf)
    {
        uint32_t g, tm, tn;
        DecodeTask(task, t, g, tm, tn);
        const uint32_t row0 = tm * t.bm;
        const uint32_t col0 = tn * t.bn;
        const uint32_t actualBm = (row0 + t.bm <= t.k) ? t.bm : (t.k - row0);
        const uint32_t actualBn = (col0 + t.bn <= t.k) ? t.bn : (t.k - col0);
        const uint32_t totalElem = actualBm * actualBn;
        const uint32_t begin = (lane == 0) ? 0 : ((totalElem + 1) >> 1);
        const uint32_t end = (lane == 0) ? ((totalElem + 1) >> 1) : totalElem;
        const int64_t base = PairBufferBase(cubeId, buf, t);
        const int64_t onePlane = static_cast<int64_t>(t.slicesPerGroup) * t.bm * t.bn;

        for (uint32_t e = begin; e < end; ++e) {
            const uint32_t tr = e / actualBn;
            const uint32_t tc = e - tr * actualBn;
            const int64_t compactOff = static_cast<int64_t>(tr) * t.bm + tc;
            OutT sumRe = 0.0f;
            OutT sumIm = 0.0f;
            OutT sumNorm = 0.0f;
            for (uint32_t i = 0; i < t.slicesPerGroup; ++i) {
                const int64_t sliceOff = static_cast<int64_t>(i) * t.bm * t.bn + compactOff;
                const OutT rr = wsGm.GetValue(base + sliceOff);
                const OutT ii = wsGm.GetValue(base + onePlane + sliceOff);
                const OutT ri = wsGm.GetValue(base + 2 * onePlane + sliceOff);
                const OutT ir = wsGm.GetValue(base + 3 * onePlane + sliceOff);
                const OutT re = rr + ii;
                const OutT im = ri - ir;
                sumRe += re;
                sumIm += im;
                sumNorm += re * re + im * im;
            }
            const uint32_t globalRow = row0 + tr;
            const uint32_t globalCol = col0 + tc;
            const int64_t out = static_cast<int64_t>(g) * t.k * t.k + static_cast<int64_t>(globalRow) * t.k + globalCol;
            bGm.SetValue(out, sumNorm * INV_16);
            bplurGm.SetValue(out, sumRe * INV_16);
            bpluiGm.SetValue(out, sumIm * INV_16);
        }
    }

    __aicore__ inline void RecomputeBsumAndCsum(uint32_t vectorId, uint32_t vectorNum)
    {
        const int64_t k2 = static_cast<int64_t>(t.k) * t.k;
        for (int64_t idx = vectorId; idx < k2; idx += vectorNum) {
            const uint32_t r = static_cast<uint32_t>(idx / t.k);
            const uint32_t c = static_cast<uint32_t>(idx - static_cast<int64_t>(r) * t.k);
            bsumGm.SetValue(idx, ComputeAverageNorm(r, c));
        }
        const int64_t n2 = static_cast<int64_t>(t.n) * t.n;
        for (int64_t idx = vectorId; idx < n2; idx += vectorNum) {
            const uint32_t a = static_cast<uint32_t>(idx / t.n);
            const uint32_t b = static_cast<uint32_t>(idx - static_cast<int64_t>(a) * t.n);
            csumGm.SetValue(idx, ComputeAverageNorm(a * 8, b * 8));
        }
    }

    __aicore__ inline OutT ComputeAverageNorm(uint32_t r, uint32_t c)
    {
        OutT total = 0.0f;
        for (uint32_t s = 0; s < 272; ++s) {
            OutT re = 0.0f;
            OutT im = 0.0f;
            for (uint32_t m = 0; m < t.mDim; ++m) {
                const int64_t offR = static_cast<int64_t>(s) * t.mDim * t.k + static_cast<int64_t>(m) * t.k + r;
                const int64_t offC = static_cast<int64_t>(s) * t.mDim * t.k + static_cast<int64_t>(m) * t.k + c;
                const OutT arR = static_cast<OutT>(arGm.GetValue(offR));
                const OutT aiR = static_cast<OutT>(aiGm.GetValue(offR));
                const OutT arC = static_cast<OutT>(arGm.GetValue(offC));
                const OutT aiC = static_cast<OutT>(aiGm.GetValue(offC));
                re += arR * arC + aiR * aiC;
                im += arR * aiC - aiR * arC;
            }
            total += re * re + im * im;
        }
        return total * INV_272;
    }

private:
    ComplexGramTilingData t{};
    GlobalTensor<InT> arGm, aiGm;
    GlobalTensor<OutT> bGm, bplurGm, bpluiGm, bsumGm, csumGm, wsGm;
};

extern "C" __global__ __aicore__ void complex_gram(GM_ADDR ar, GM_ADDR ai,
                                                     GM_ADDR b, GM_ADDR bplur, GM_ADDR bplui,
                                                     GM_ADDR bsum, GM_ADDR csum,
                                                     GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelComplexGram op;
    op.Init(ar, ai, b, bplur, bplui, bsum, csum, workspace, tiling_data);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
// Official msOpGen-style host-call wrapper for the generated runtime stub.
void complex_gram_do(uint32_t blockDim, void *l2ctrl, void *stream,
                     uint8_t *ar, uint8_t *ai,
                     uint8_t *b, uint8_t *bplur, uint8_t *bplui,
                     uint8_t *bsum, uint8_t *csum,
                     uint8_t *workspace, uint8_t *tiling)
{
    complex_gram<<<blockDim, l2ctrl, stream>>>(ar, ai, b, bplur, bplui, bsum, csum, workspace, tiling);
}
#endif
