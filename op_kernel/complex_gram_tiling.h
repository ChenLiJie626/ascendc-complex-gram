#pragma once

#include <cstdint>

// Host/kernel shared tiling contract for ComplexGram.
// Keep this struct POD and 32-byte aligned when wiring it to official CANN tiling data.
struct ComplexGramTilingData {
    int32_t n;
    int32_t k;                 // 8*n
    int32_t groupNum;          // 17
    int32_t slicesPerGroup;    // 16
    int32_t mDim;              // 256

    int32_t cubeBlockNum;      // 20
    int32_t vectorBlockNum;    // 40
    int32_t blockPerCube;      // vectorBlockNum / cubeBlockNum = 2

    int32_t bm;                // output row tile, default 32
    int32_t bn;                // output col tile, default 32
    int32_t tileMNum;          // ceil(k / bm)
    int32_t tileNNum;          // ceil(k / bn)
    int32_t taskNum;           // groupNum * tileMNum * tileNNum

    int32_t tileElem;          // bm * bn
    int32_t vectorTile0;       // first half element count for lane 0
    int32_t vectorTile1;       // second half element count for lane 1

    uint64_t oneTaskTmpElems;  // 4 * slicesPerGroup * bm * bn
    uint64_t workspaceBytes;   // taskNum * oneTaskTmpElems * sizeof(float), multi-kernel mode

    int32_t pingPong;          // 2, single-kernel AIC/AIV ring buffer depth
    int32_t flagSlotsPerBuf;   // ready0, ready1, done0, done1
    uint64_t onePairTmpElems;  // pingPong * oneTaskTmpElems
    uint64_t singleKernelWorkspaceBytes; // cubeBlockNum * onePairTmpElems * sizeof(float)
};

static inline ComplexGramTilingData MakeComplexGramTiling(int32_t n,
                                                          int32_t bm = 32,
                                                          int32_t bn = 32,
                                                          int32_t cubeBlockNum = 20,
                                                          int32_t vectorBlockNum = 40)
{
    ComplexGramTilingData t{};
    t.n = n;
    t.k = n * 8;
    t.groupNum = 17;
    t.slicesPerGroup = 16;
    t.mDim = 256;
    t.cubeBlockNum = cubeBlockNum;
    t.vectorBlockNum = vectorBlockNum;
    t.blockPerCube = vectorBlockNum / cubeBlockNum;
    t.bm = bm;
    t.bn = bn;
    t.tileMNum = (t.k + bm - 1) / bm;
    t.tileNNum = (t.k + bn - 1) / bn;
    t.taskNum = t.groupNum * t.tileMNum * t.tileNNum;
    t.tileElem = bm * bn;
    t.vectorTile0 = (t.tileElem + 1) / 2;
    t.vectorTile1 = t.tileElem - t.vectorTile0;
    t.oneTaskTmpElems = static_cast<uint64_t>(4) * t.slicesPerGroup * bm * bn;
    t.workspaceBytes = static_cast<uint64_t>(t.taskNum) * t.oneTaskTmpElems * sizeof(float);
    t.pingPong = 2;
    t.flagSlotsPerBuf = 4;
    t.onePairTmpElems = static_cast<uint64_t>(t.pingPong) * t.oneTaskTmpElems;
    t.singleKernelWorkspaceBytes = static_cast<uint64_t>(t.cubeBlockNum) * t.onePairTmpElems * sizeof(float);
    return t;
}
