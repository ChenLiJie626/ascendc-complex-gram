/**
 * @file complex_gram_tiling.h
 * Official msOpGen-style tiling data definition for ComplexGram.
 */
#ifndef COMPLEX_GRAM_TILING_H
#define COMPLEX_GRAM_TILING_H

#include <cstdint>

#ifdef COMPLEX_GRAM_HOST_STANDALONE
struct ComplexGramTilingData {
    uint32_t n;
    uint32_t k;
    uint32_t groupNum;
    uint32_t slicesPerGroup;
    uint32_t mDim;
    uint32_t cubeBlockNum;
    uint32_t vectorBlockNum;
    uint32_t blockPerCube;
    uint32_t bm;
    uint32_t bn;
    uint32_t tileMNum;
    uint32_t tileNNum;
    uint32_t taskNum;
    uint32_t tileElem;
    uint32_t vectorTile0;
    uint32_t vectorTile1;
    uint32_t pingPong;
    uint32_t flagSlotsPerBuf;
    uint64_t oneTaskTmpElems;
    uint64_t fullWorkspaceBytes;
    uint64_t singleKernelWorkspaceBytes;
};
#else
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(ComplexGramTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, n);
    TILING_DATA_FIELD_DEF(uint32_t, k);
    TILING_DATA_FIELD_DEF(uint32_t, groupNum);
    TILING_DATA_FIELD_DEF(uint32_t, slicesPerGroup);
    TILING_DATA_FIELD_DEF(uint32_t, mDim);
    TILING_DATA_FIELD_DEF(uint32_t, cubeBlockNum);
    TILING_DATA_FIELD_DEF(uint32_t, vectorBlockNum);
    TILING_DATA_FIELD_DEF(uint32_t, blockPerCube);
    TILING_DATA_FIELD_DEF(uint32_t, bm);
    TILING_DATA_FIELD_DEF(uint32_t, bn);
    TILING_DATA_FIELD_DEF(uint32_t, tileMNum);
    TILING_DATA_FIELD_DEF(uint32_t, tileNNum);
    TILING_DATA_FIELD_DEF(uint32_t, taskNum);
    TILING_DATA_FIELD_DEF(uint32_t, tileElem);
    TILING_DATA_FIELD_DEF(uint32_t, vectorTile0);
    TILING_DATA_FIELD_DEF(uint32_t, vectorTile1);
    TILING_DATA_FIELD_DEF(uint32_t, pingPong);
    TILING_DATA_FIELD_DEF(uint32_t, flagSlotsPerBuf);
    TILING_DATA_FIELD_DEF(uint64_t, oneTaskTmpElems);
    TILING_DATA_FIELD_DEF(uint64_t, fullWorkspaceBytes);
    TILING_DATA_FIELD_DEF(uint64_t, singleKernelWorkspaceBytes);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ComplexGram, ComplexGramTilingData)
} // namespace optiling
#endif

static inline ComplexGramTilingData MakeComplexGramTilingData(uint32_t n,
                                                              uint32_t bm = 32,
                                                              uint32_t bn = 32,
                                                              uint32_t cubeBlockNum = 20,
                                                              uint32_t vectorBlockNum = 40)
{
    ComplexGramTilingData t{};
#ifdef COMPLEX_GRAM_HOST_STANDALONE
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
    t.pingPong = 2;
    t.flagSlotsPerBuf = 4;
    t.oneTaskTmpElems = static_cast<uint64_t>(4) * t.slicesPerGroup * bm * bn;
    t.fullWorkspaceBytes = static_cast<uint64_t>(t.taskNum) * t.oneTaskTmpElems * sizeof(float);
    t.singleKernelWorkspaceBytes = static_cast<uint64_t>(t.cubeBlockNum) * t.pingPong * t.oneTaskTmpElems * sizeof(float);
#else
    t.set_n(n);
    t.set_k(n * 8);
    t.set_groupNum(17);
    t.set_slicesPerGroup(16);
    t.set_mDim(256);
    t.set_cubeBlockNum(cubeBlockNum);
    t.set_vectorBlockNum(vectorBlockNum);
    t.set_blockPerCube(vectorBlockNum / cubeBlockNum);
    t.set_bm(bm);
    t.set_bn(bn);
    const uint32_t k = n * 8;
    const uint32_t tileMNum = (k + bm - 1) / bm;
    const uint32_t tileNNum = (k + bn - 1) / bn;
    const uint32_t taskNum = 17 * tileMNum * tileNNum;
    const uint32_t tileElem = bm * bn;
    const uint64_t oneTaskTmpElems = static_cast<uint64_t>(4) * 16 * bm * bn;
    t.set_tileMNum(tileMNum);
    t.set_tileNNum(tileNNum);
    t.set_taskNum(taskNum);
    t.set_tileElem(tileElem);
    t.set_vectorTile0((tileElem + 1) / 2);
    t.set_vectorTile1(tileElem - ((tileElem + 1) / 2));
    t.set_pingPong(2);
    t.set_flagSlotsPerBuf(4);
    t.set_oneTaskTmpElems(oneTaskTmpElems);
    t.set_fullWorkspaceBytes(static_cast<uint64_t>(taskNum) * oneTaskTmpElems * sizeof(float));
    t.set_singleKernelWorkspaceBytes(static_cast<uint64_t>(cubeBlockNum) * 2 * oneTaskTmpElems * sizeof(float));
#endif
    return t;
}

#endif // COMPLEX_GRAM_TILING_H
