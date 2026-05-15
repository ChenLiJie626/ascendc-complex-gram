/*
 * Experimental single-kernel AIC/AIV synchronized ComplexGram template.
 *
 * Goal:
 *   Put cube and vector work in one kernel and use SetFlag/WaitFlag for fine-grained
 *   producer/consumer synchronization:
 *
 *     AIC cube c computes one group-tile into a per-cube ping-pong workspace buffer.
 *     AIC sets ready flags for its two AIV partners.
 *     AIV vector 2*c and 2*c+1 wait ready, consume first/second half of the tile,
 *     write B/BPlur/BPlui, then set done flags.
 *     Before reusing a ping-pong buffer, AIC waits for both done flags.
 *
 * Important:
 *   AscendC flag APIs and HardEvent names differ across CANN releases and between
 *   examples. This file isolates the synchronization calls in four wrappers below:
 *     NotifyVecLaneReady, WaitVecLaneReady, NotifyCubeLaneDone, WaitCubeLaneDone.
 *   Replace the wrapper bodies with the exact SetFlag/WaitFlag signatures from your
 *   installed CANN version's AIC/AIV fusion sample. The task schedule, buffer reuse,
 *   and flag-id design are complete and tested by test_tiling_plan.py.
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

constexpr float SINGLE_INV_16 = 1.0f / 16.0f;
constexpr float SINGLE_INV_17 = 1.0f / 17.0f;

__aicore__ inline void DecodeSingleTask(int32_t taskId, const ComplexGramTilingData &t,
                                        int32_t &g, int32_t &tm, int32_t &tn)
{
    const int32_t tilePerGroup = t.tileMNum * t.tileNNum;
    g = taskId / tilePerGroup;
    const int32_t r = taskId - g * tilePerGroup;
    tm = r / t.tileNNum;
    tn = r - tm * t.tileNNum;
}

__aicore__ inline int32_t ReadyFlagId(int32_t cubeId, int32_t lane, int32_t buf, const ComplexGramTilingData &t)
{
    return cubeId * t.flagSlotsPerBuf * t.pingPong + buf * t.flagSlotsPerBuf + lane;
}

__aicore__ inline int32_t DoneFlagId(int32_t cubeId, int32_t lane, int32_t buf, const ComplexGramTilingData &t)
{
    return cubeId * t.flagSlotsPerBuf * t.pingPong + buf * t.flagSlotsPerBuf + 2 + lane;
}

__aicore__ inline int64_t PairBufferBase(int32_t cubeId, int32_t buf, const ComplexGramTilingData &t)
{
    return (static_cast<int64_t>(cubeId) * t.pingPong + buf) * static_cast<int64_t>(t.oneTaskTmpElems);
}

// -----------------------------------------------------------------------------
// Synchronization wrappers.
// -----------------------------------------------------------------------------
// Replace these four wrappers with your exact CANN-version API. Typical official
// examples use SetFlag<HardEvent::...>(flagId) and WaitFlag<HardEvent::...>(flagId)
// or their cross-core event equivalents. Keeping wrappers makes the rest of the
// algorithm stable even if API names differ.

__aicore__ inline void NotifyVecLaneReady(int32_t flagId)
{
#if defined(COMPLEX_GRAM_ENABLE_REAL_AIC_AIV_FLAGS)
    // Example placeholder; adapt HardEvent according to CANN official fusion sample.
    SetFlag<HardEvent::MTE3_MTE2>(flagId);
#else
    (void)flagId;
#endif
}

__aicore__ inline void WaitVecLaneReady(int32_t flagId)
{
#if defined(COMPLEX_GRAM_ENABLE_REAL_AIC_AIV_FLAGS)
    WaitFlag<HardEvent::MTE3_MTE2>(flagId);
#else
    (void)flagId;
#endif
}

__aicore__ inline void NotifyCubeLaneDone(int32_t flagId)
{
#if defined(COMPLEX_GRAM_ENABLE_REAL_AIC_AIV_FLAGS)
    SetFlag<HardEvent::MTE2_MTE3>(flagId);
#else
    (void)flagId;
#endif
}

__aicore__ inline void WaitCubeLaneDone(int32_t flagId)
{
#if defined(COMPLEX_GRAM_ENABLE_REAL_AIC_AIV_FLAGS)
    WaitFlag<HardEvent::MTE2_MTE3>(flagId);
#else
    (void)flagId;
#endif
}

class ComplexGramSingleKernel {
public:
    __aicore__ inline void Init(GM_ADDR ar, GM_ADDR ai,
                                GM_ADDR workspace,
                                GM_ADDR b, GM_ADDR bPlur, GM_ADDR bPlui,
                                GM_ADDR bsum, GM_ADDR csum,
                                GM_ADDR tiling)
    {
        t = *reinterpret_cast<__gm__ ComplexGramTilingData *>(tiling);
        arGm.SetGlobalBuffer(reinterpret_cast<__gm__ InT *>(ar), static_cast<uint64_t>(272) * t.mDim * t.k);
        aiGm.SetGlobalBuffer(reinterpret_cast<__gm__ InT *>(ai), static_cast<uint64_t>(272) * t.mDim * t.k);
        wsGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutT *>(workspace), t.singleKernelWorkspaceBytes / sizeof(OutT));
        bGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutT *>(b), static_cast<uint64_t>(t.groupNum) * t.k * t.k);
        bPlurGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutT *>(bPlur), static_cast<uint64_t>(t.groupNum) * t.k * t.k);
        bPluiGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutT *>(bPlui), static_cast<uint64_t>(t.groupNum) * t.k * t.k);
        bsumGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutT *>(bsum), static_cast<uint64_t>(t.k) * t.k);
        csumGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutT *>(csum), static_cast<uint64_t>(t.n) * t.n);
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
        const int32_t cubeId = GetBlockIdx();
        int32_t localIdx = 0;
        for (int32_t task = cubeId; task < t.taskNum; task += t.cubeBlockNum, ++localIdx) {
            const int32_t buf = localIdx & 1;
            if (localIdx >= t.pingPong) {
                WaitCubeLaneDone(DoneFlagId(cubeId, 0, buf, t));
                WaitCubeLaneDone(DoneFlagId(cubeId, 1, buf, t));
            }
            ProduceTaskTile(task, cubeId, buf);
            NotifyVecLaneReady(ReadyFlagId(cubeId, 0, buf, t));
            NotifyVecLaneReady(ReadyFlagId(cubeId, 1, buf, t));
        }

        // Final drain: make sure vectors have consumed the last one/two buffers before kernel exits.
        const int32_t totalLocalTasks = (t.taskNum > cubeId) ? ((t.taskNum - 1 - cubeId) / t.cubeBlockNum + 1) : 0;
        const int32_t drain = totalLocalTasks < t.pingPong ? totalLocalTasks : t.pingPong;
        for (int32_t d = 0; d < drain; ++d) {
            const int32_t lastLocalIdx = totalLocalTasks - 1 - d;
            const int32_t buf = lastLocalIdx & 1;
            WaitCubeLaneDone(DoneFlagId(cubeId, 0, buf, t));
            WaitCubeLaneDone(DoneFlagId(cubeId, 1, buf, t));
        }
    }

    __aicore__ inline void ProcessVector()
    {
        const int32_t vectorId = GetBlockIdx();
        const int32_t pairCube = vectorId >> 1;
        const int32_t lane = vectorId & 1;
        int32_t localIdx = 0;
        for (int32_t task = pairCube; task < t.taskNum; task += t.cubeBlockNum, ++localIdx) {
            const int32_t buf = localIdx & 1;
            WaitVecLaneReady(ReadyFlagId(pairCube, lane, buf, t));
            ConsumeTaskTile(task, pairCube, lane, buf);
            NotifyCubeLaneDone(DoneFlagId(pairCube, lane, buf, t));
        }

        // Optional: only a subset of vectors compute Bsum/Csum after all epilogues.
        // True single-kernel Bsum/Csum needs a global barrier across all vector lanes,
        // which ordinary flags do not provide safely. Keep Bsum/Csum in a follow-up
        // vector reduce kernel, or add a rigorously designed global completion counter.
        (void)bsumGm;
        (void)csumGm;
    }

    __aicore__ inline void ProduceTaskTile(int32_t task, int32_t cubeId, int32_t buf)
    {
        int32_t g, tm, tn;
        DecodeSingleTask(task, t, g, tm, tn);
        const int32_t row0 = tm * t.bm;
        const int32_t col0 = tn * t.bn;
        const int32_t actualBm = (row0 + t.bm <= t.k) ? t.bm : (t.k - row0);
        const int32_t actualBn = (col0 + t.bn <= t.k) ? t.bn : (t.k - col0);
        const int64_t base = PairBufferBase(cubeId, buf, t);
        const int64_t onePlane = static_cast<int64_t>(t.slicesPerGroup) * t.bm * t.bn;

        for (int32_t i = 0; i < t.slicesPerGroup; ++i) {
            const int32_t s = g * t.slicesPerGroup + i;
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
                                    int32_t actualBm,
                                    int32_t actualBn)
    {
        Matmul<AType, BType, CType, BiasType> mm;
        mm.SetOrgShape(t.mDim, actualBm, actualBn);
        mm.SetTensorA(a, true);
        mm.SetTensorB(b, false);
        mm.IterateAll(c, false);
        mm.End();
    }

    __aicore__ inline void ConsumeTaskTile(int32_t task, int32_t cubeId, int32_t lane, int32_t buf)
    {
        int32_t g, tm, tn;
        DecodeSingleTask(task, t, g, tm, tn);
        const int32_t row0 = tm * t.bm;
        const int32_t col0 = tn * t.bn;
        const int32_t actualBm = (row0 + t.bm <= t.k) ? t.bm : (t.k - row0);
        const int32_t actualBn = (col0 + t.bn <= t.k) ? t.bn : (t.k - col0);
        const int32_t totalElem = actualBm * actualBn;
        const int32_t begin = (lane == 0) ? 0 : ((totalElem + 1) >> 1);
        const int32_t end = (lane == 0) ? ((totalElem + 1) >> 1) : totalElem;
        const int64_t base = PairBufferBase(cubeId, buf, t);
        const int64_t onePlane = static_cast<int64_t>(t.slicesPerGroup) * t.bm * t.bn;

        for (int32_t e = begin; e < end; ++e) {
            const int32_t tr = e / actualBn;
            const int32_t tc = e - tr * actualBn;
            const int64_t compactOff = static_cast<int64_t>(tr) * t.bm + tc;
            OutT sumRe = 0.0f;
            OutT sumIm = 0.0f;
            OutT sumNorm = 0.0f;
            for (int32_t i = 0; i < t.slicesPerGroup; ++i) {
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
            const int32_t globalRow = row0 + tr;
            const int32_t globalCol = col0 + tc;
            const int64_t out = static_cast<int64_t>(g) * t.k * t.k + static_cast<int64_t>(globalRow) * t.k + globalCol;
            bGm.SetValue(out, sumNorm * SINGLE_INV_16);
            bPlurGm.SetValue(out, sumRe * SINGLE_INV_16);
            bPluiGm.SetValue(out, sumIm * SINGLE_INV_16);
        }
    }

private:
    ComplexGramTilingData t{};
    GlobalTensor<InT> arGm, aiGm;
    GlobalTensor<OutT> wsGm, bGm, bPlurGm, bPluiGm, bsumGm, csumGm;
};

extern "C" __global__ __aicore__ void complex_gram_single_kernel(GM_ADDR ar, GM_ADDR ai,
                                                                   GM_ADDR workspace,
                                                                   GM_ADDR b,
                                                                   GM_ADDR bPlur,
                                                                   GM_ADDR bPlui,
                                                                   GM_ADDR bsum,
                                                                   GM_ADDR csum,
                                                                   GM_ADDR tiling)
{
    ComplexGramSingleKernel op;
    op.Init(ar, ai, workspace, b, bPlur, bPlui, bsum, csum, tiling);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void complex_gram_single_do(void *stream,
                            uint8_t *ar, uint8_t *ai,
                            uint8_t *workspace,
                            uint8_t *b, uint8_t *bPlur, uint8_t *bPlui,
                            uint8_t *bsum, uint8_t *csum,
                            uint8_t *tiling)
{
    auto *t = reinterpret_cast<ComplexGramTilingData *>(tiling);
    // In the real package, compile this kernel as an AIC/AIV fusion kernel and set
    // both AIC=20 and AIV=40 block dimensions according to the official sample's
    // launch macro. Some generated stubs expose separate cube/vector block dims.
    complex_gram_single_kernel<<<t->cubeBlockNum, nullptr, stream>>>(ar, ai, workspace, b, bPlur, bPlui, bsum, csum, tiling);
}
#endif
