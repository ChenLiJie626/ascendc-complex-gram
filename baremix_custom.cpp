/**
 * @file baremix_custom.cpp
 */
#define ASCENDC_CUBE_ONLY

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

namespace complex_gram_fused {
constexpr uint32_t GROUP_NUM = 17;
constexpr uint32_t AVG_NUM = 16;
constexpr uint32_t K_DIM = 256;
constexpr uint32_t USER_VEC = 8;
constexpr uint32_t AIV_PER_AIC = 2;
constexpr uint32_t TMP_MATMUL_NUM = 4;

struct Params {
    uint32_t n;
    uint32_t u;
    uint32_t k;
    uint32_t blockDim;
    uint32_t sysWorkspaceSize;
    uint32_t reserved;
    uint64_t userWorkspaceSize;
};
}  // namespace complex_gram_fused

namespace {

constexpr uint16_t FLAG_CUBE_DONE_BASE = 7;
constexpr uint16_t FLAG_VEC_DONE_BASE = 9;
constexpr uint32_t VEC_CHUNK = 256;
constexpr float INV_AVG_NUM = 1.0f / static_cast<float>(complex_gram_fused::AVG_NUM);
constexpr float INV_GROUP_NUM = 1.0f / static_cast<float>(complex_gram_fused::GROUP_NUM);

using AType = half;
using BType = half;
using CType = float;
using BiasType = float;
using Mm = Matmul<MatmulType<TPosition::GM, CubeFormat::ND, AType, true>,
                  MatmulType<TPosition::GM, CubeFormat::ND, BType, false>,
                  MatmulType<TPosition::GM, CubeFormat::ND, CType>,
                  MatmulType<TPosition::GM, CubeFormat::ND, BiasType>>;

__aicore__ inline uint32_t MinU32(uint32_t lhs, uint32_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline uint32_t CeilDiv(uint32_t value, uint32_t div)
{
    return div == 0 ? 0 : (value + div - 1) / div;
}

__aicore__ inline void CopyCubeTiling(TCubeTiling *tiling, GM_ADDR tilingGm)
{
    auto src = reinterpret_cast<__gm__ uint32_t *>(tilingGm);
    auto dst = reinterpret_cast<uint32_t *>(tiling);
    constexpr uint32_t words = sizeof(TCubeTiling) / sizeof(uint32_t);
    for (uint32_t i = 0; i < words; ++i) {
        dst[i] = src[i];
    }
}

__aicore__ inline void CopyParams(complex_gram_fused::Params *params, GM_ADDR tilingGm)
{
    constexpr uint64_t paramOffset = ((sizeof(TCubeTiling) + 31) / 32) * 32;
    auto src = reinterpret_cast<__gm__ uint32_t *>(reinterpret_cast<__gm__ uint8_t *>(tilingGm) + paramOffset);
    auto dst = reinterpret_cast<uint32_t *>(params);
    constexpr uint32_t words = sizeof(complex_gram_fused::Params) / sizeof(uint32_t);
    for (uint32_t i = 0; i < words; ++i) {
        dst[i] = src[i];
    }
}

struct TileInfo {
    uint32_t rowStart;
    uint32_t colStart;
    uint32_t rowLen;
    uint32_t colLen;
    uint32_t offsetA;
    uint32_t offsetB;
    uint32_t offsetC;
    bool valid;
};

__aicore__ inline TileInfo CalcTileInfo(uint32_t blockIdx, const TCubeTiling &tiling)
{
    TileInfo info{0, 0, 0, 0, 0, 0, 0, false};
    const uint32_t mBlocks = CeilDiv(tiling.M, tiling.singleCoreM);
    const uint32_t nBlocks = CeilDiv(tiling.N, tiling.singleCoreN);
    if (blockIdx >= mBlocks * nBlocks) {
        return info;
    }

    const uint32_t mIdx = blockIdx % mBlocks;
    const uint32_t nIdx = blockIdx / mBlocks;
    info.rowStart = mIdx * tiling.singleCoreM;
    info.colStart = nIdx * tiling.singleCoreN;
    info.rowLen = MinU32(tiling.singleCoreM, tiling.M - info.rowStart);
    info.colLen = MinU32(tiling.singleCoreN, tiling.N - info.colStart);
    info.offsetA = info.rowStart;
    info.offsetB = info.colStart;
    info.offsetC = info.rowStart * tiling.N + info.colStart;
    info.valid = info.rowLen > 0 && info.colLen > 0;
    return info;
}

class ComplexGramCubeKernel {
public:
    __aicore__ inline void Init(GM_ADDR ar, GM_ADDR ai, GM_ADDR workspace, const TCubeTiling &tiling,
                                const complex_gram_fused::Params &params, TPipe *pipe)
    {
        ar_ = reinterpret_cast<__gm__ AType *>(ar);
        ai_ = reinterpret_cast<__gm__ AType *>(ai);
        tiling_ = tiling;
        params_ = params;
        pipe_ = pipe;

        auto userWorkspace = reinterpret_cast<__gm__ uint8_t *>(workspace) + params_.sysWorkspaceSize;
        const uint64_t tmpMatrixElems = static_cast<uint64_t>(complex_gram_fused::AVG_NUM) * params_.u * params_.u;
        tmpRR_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(userWorkspace), tmpMatrixElems);
        tmpII_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(userWorkspace) + tmpMatrixElems, tmpMatrixElems);
        tmpRI_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(userWorkspace) + tmpMatrixElems * 2, tmpMatrixElems);
        tmpIR_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(userWorkspace) + tmpMatrixElems * 3, tmpMatrixElems);
    }

    __aicore__ inline void Process()
    {
        const uint32_t aicBlockIdx = GetBlockIdx();
        tile_ = CalcTileInfo(aicBlockIdx, tiling_);
        if (!tile_.valid) {
            return;
        }

        REGIST_MATMUL_OBJ(pipe_, GetSysWorkSpacePtr(), mm_);
        for (uint32_t g = 0; g < complex_gram_fused::GROUP_NUM; ++g) {
            for (uint32_t inner = 0; inner < complex_gram_fused::AVG_NUM; ++inner) {
                const uint64_t slice = (static_cast<uint64_t>(g) * complex_gram_fused::AVG_NUM + inner) *
                                       params_.k * params_.u;
                const uint64_t tmpBase = static_cast<uint64_t>(inner) * params_.u * params_.u;
                RunMatmul(ar_ + slice, ar_ + slice, tmpRR_, tmpBase);
                RunMatmul(ai_ + slice, ai_ + slice, tmpII_, tmpBase);
                RunMatmul(ar_ + slice, ai_ + slice, tmpRI_, tmpBase);
                RunMatmul(ai_ + slice, ar_ + slice, tmpIR_, tmpBase);
            }
            const uint16_t readyFlag = FLAG_CUBE_DONE_BASE + (g & 1);
            const uint16_t doneFlag = FLAG_VEC_DONE_BASE + (g & 1);
            CrossCoreSetFlag<0x2, PIPE_FIX>(readyFlag);
            CrossCoreWaitFlag(doneFlag);
        }
    }

private:
    __aicore__ inline void RunMatmul(__gm__ AType *aBase, __gm__ BType *bBase, GlobalTensor<float> &cGm,
                                     uint64_t tmpBase)
    {
        GlobalTensor<AType> aGm;
        GlobalTensor<BType> bGm;
        aGm.SetGlobalBuffer(aBase, params_.k * params_.u);
        bGm.SetGlobalBuffer(bBase, params_.k * params_.u);

        mm_.Init(&tiling_);
        mm_.SetTensorA(aGm[tile_.offsetA], true);
        mm_.SetTensorB(bGm[tile_.offsetB], false);
        mm_.SetTail(tile_.rowLen, tile_.colLen);
        mm_.IterateAll(cGm[tmpBase + tile_.offsetC]);
        mm_.End();
    }

    __gm__ AType *ar_;
    __gm__ AType *ai_;
    TCubeTiling tiling_;
    complex_gram_fused::Params params_;
    TileInfo tile_;
    TPipe *pipe_;
    Mm mm_;
    GlobalTensor<float> tmpRR_;
    GlobalTensor<float> tmpII_;
    GlobalTensor<float> tmpRI_;
    GlobalTensor<float> tmpIR_;
};

class ComplexGramVectorKernel {
public:
    __aicore__ inline void Init(GM_ADDR b, GM_ADDR bPlur, GM_ADDR bPlui, GM_ADDR bsum, GM_ADDR csum,
                                GM_ADDR workspace, const TCubeTiling &tiling,
                                const complex_gram_fused::Params &params, TPipe *pipe)
    {
        tiling_ = tiling;
        params_ = params;
        pipe_ = pipe;

        const uint64_t matrixElems = static_cast<uint64_t>(params_.u) * params_.u;
        const uint64_t tmpMatrixElems = static_cast<uint64_t>(complex_gram_fused::AVG_NUM) * matrixElems;
        b_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(b), complex_gram_fused::GROUP_NUM * matrixElems);
        bPlur_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bPlur), complex_gram_fused::GROUP_NUM * matrixElems);
        bPlui_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bPlui), complex_gram_fused::GROUP_NUM * matrixElems);
        bsum_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bsum), matrixElems);
        csum_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(csum), static_cast<uint64_t>(params_.n) * params_.n);

        auto userWorkspace = reinterpret_cast<__gm__ uint8_t *>(workspace) + params_.sysWorkspaceSize;
        tmpRR_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(userWorkspace), tmpMatrixElems);
        tmpII_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(userWorkspace) + tmpMatrixElems, tmpMatrixElems);
        tmpRI_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(userWorkspace) + tmpMatrixElems * 2, tmpMatrixElems);
        tmpIR_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(userWorkspace) + tmpMatrixElems * 3, tmpMatrixElems);

        pipe_->InitBuffer(calcBuf_, 8 * VEC_CHUNK * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t aivBlockIdx = GetBlockIdx();
        const uint32_t aicBlockIdx = aivBlockIdx / complex_gram_fused::AIV_PER_AIC;
        tile_ = CalcTileInfo(aicBlockIdx, tiling_);
        if (!tile_.valid) {
            return;
        }

        const uint32_t subIdx = aivBlockIdx % complex_gram_fused::AIV_PER_AIC;
        const uint32_t rowsPerSub = CeilDiv(tile_.rowLen, complex_gram_fused::AIV_PER_AIC);
        rowBegin_ = tile_.rowStart + subIdx * rowsPerSub;
        rowEnd_ = MinU32(tile_.rowStart + tile_.rowLen, rowBegin_ + rowsPerSub);

        for (uint32_t g = 0; g < complex_gram_fused::GROUP_NUM; ++g) {
            const uint16_t readyFlag = FLAG_CUBE_DONE_BASE + (g & 1);
            const uint16_t doneFlag = FLAG_VEC_DONE_BASE + (g & 1);
            CrossCoreWaitFlag(readyFlag);
            for (uint32_t inner = 0; inner < complex_gram_fused::AVG_NUM; ++inner) {
                ProcessSlice(g, inner);
            }
            CrossCoreSetFlag<0x2, PIPE_MTE3>(doneFlag);
        }
    }

private:
    __aicore__ inline void ProcessSlice(uint32_t g, uint32_t inner)
    {
        const uint64_t matrixElems = static_cast<uint64_t>(params_.u) * params_.u;
        const uint64_t tmpBase = static_cast<uint64_t>(inner) * matrixElems;
        const uint64_t groupBase = static_cast<uint64_t>(g) * matrixElems;
        LocalTensor<float> rr = calcBuf_.GetWithOffset<float>(VEC_CHUNK, 0);
        LocalTensor<float> ii = calcBuf_.GetWithOffset<float>(VEC_CHUNK, VEC_CHUNK * sizeof(float));
        LocalTensor<float> ri = calcBuf_.GetWithOffset<float>(VEC_CHUNK, VEC_CHUNK * 2 * sizeof(float));
        LocalTensor<float> ir = calcBuf_.GetWithOffset<float>(VEC_CHUNK, VEC_CHUNK * 3 * sizeof(float));
        LocalTensor<float> bLocal = calcBuf_.GetWithOffset<float>(VEC_CHUNK, VEC_CHUNK * 4 * sizeof(float));
        LocalTensor<float> prLocal = calcBuf_.GetWithOffset<float>(VEC_CHUNK, VEC_CHUNK * 5 * sizeof(float));
        LocalTensor<float> piLocal = calcBuf_.GetWithOffset<float>(VEC_CHUNK, VEC_CHUNK * 6 * sizeof(float));
        LocalTensor<float> sumLocal = calcBuf_.GetWithOffset<float>(VEC_CHUNK, VEC_CHUNK * 7 * sizeof(float));

        for (uint32_t row = rowBegin_; row < rowEnd_; ++row) {
            uint32_t col = tile_.colStart;
            const uint32_t colEnd = tile_.colStart + tile_.colLen;
            while (col < colEnd) {
                const uint32_t count = MinU32(VEC_CHUNK, colEnd - col);
                const uint64_t offset = static_cast<uint64_t>(row) * params_.u + col;
                const uint64_t outOffset = groupBase + offset;

                DataCopy(rr, tmpRR_[tmpBase + offset], count);
                DataCopy(ii, tmpII_[tmpBase + offset], count);
                DataCopy(ri, tmpRI_[tmpBase + offset], count);
                DataCopy(ir, tmpIR_[tmpBase + offset], count);
                if (inner != 0) {
                    DataCopy(bLocal, b_[outOffset], count);
                    DataCopy(prLocal, bPlur_[outOffset], count);
                    DataCopy(piLocal, bPlui_[outOffset], count);
                }
                if (inner == complex_gram_fused::AVG_NUM - 1 && g != 0) {
                    DataCopy(sumLocal, bsum_[offset], count);
                }
                PipeBarrier<PIPE_ALL>();

                Add(rr, rr, ii, count);
                Sub(ri, ri, ir, count);
                Mul(ii, rr, rr, count);
                Mul(ir, ri, ri, count);
                Add(ii, ii, ir, count);
                Muls(ii, ii, INV_AVG_NUM, count);

                if (inner == 0) {
                    Duplicate(bLocal, 0.0f, count);
                    Duplicate(prLocal, 0.0f, count);
                    Duplicate(piLocal, 0.0f, count);
                }

                Add(bLocal, bLocal, ii, count);
                Muls(rr, rr, INV_AVG_NUM, count);
                ApplyTriangleConjugate(row, col, count, ri);
                Muls(ri, ri, INV_AVG_NUM, count);
                Add(prLocal, prLocal, rr, count);
                Add(piLocal, piLocal, ri, count);
                PipeBarrier<PIPE_V>();

                DataCopy(b_[outOffset], bLocal, count);
                DataCopy(bPlur_[outOffset], prLocal, count);
                DataCopy(bPlui_[outOffset], piLocal, count);

                if (inner == complex_gram_fused::AVG_NUM - 1) {
                    if (g == 0) {
                        Duplicate(sumLocal, 0.0f, count);
                    }
                    Muls(ii, bLocal, INV_GROUP_NUM, count);
                    Add(sumLocal, sumLocal, ii, count);
                    PipeBarrier<PIPE_V>();
                    DataCopy(bsum_[offset], sumLocal, count);
                    UpdateCsum(g, row, col, count, bLocal);
                }

                col += count;
            }
        }
    }

    __aicore__ inline void ApplyTriangleConjugate(uint32_t row, uint32_t col, uint32_t count,
                                                  LocalTensor<float> &imag)
    {
        if (row >= col + count) {
            Muls(imag, imag, -1.0f, count);
            return;
        }
        if (row < col) {
            return;
        }

        PipeBarrier<PIPE_V>();
        const uint32_t lowerCount = row - col;
        for (uint32_t idx = 0; idx < lowerCount; ++idx) {
            imag.SetValue(idx, -imag.GetValue(idx));
        }
    }

    __aicore__ inline void UpdateCsum(uint32_t g, uint32_t row, uint32_t col, uint32_t count,
                                      LocalTensor<float> &bLocal)
    {
        if ((row & (complex_gram_fused::USER_VEC - 1)) != 0) {
            return;
        }

        PipeBarrier<PIPE_V>();
        uint32_t c = col;
        const uint32_t end = col + count;
        const uint32_t rem = c & (complex_gram_fused::USER_VEC - 1);
        if (rem != 0) {
            c += complex_gram_fused::USER_VEC - rem;
        }

        const uint32_t csumRow = row / complex_gram_fused::USER_VEC;
        while (c < end) {
            const uint32_t localIdx = c - col;
            const uint32_t csumCol = c / complex_gram_fused::USER_VEC;
            const uint64_t csumOffset = static_cast<uint64_t>(csumRow) * params_.n + csumCol;
            const float addValue = bLocal.GetValue(localIdx) * INV_GROUP_NUM;
            const float oldValue = (g == 0) ? 0.0f : csum_.GetValue(csumOffset);
            csum_.SetValue(csumOffset, oldValue + addValue);
            c += complex_gram_fused::USER_VEC;
        }
    }

    TCubeTiling tiling_;
    complex_gram_fused::Params params_;
    TileInfo tile_;
    uint32_t rowBegin_;
    uint32_t rowEnd_;
    TPipe *pipe_;
    TBuf<TPosition::VECCALC> calcBuf_;
    GlobalTensor<float> tmpRR_;
    GlobalTensor<float> tmpII_;
    GlobalTensor<float> tmpRI_;
    GlobalTensor<float> tmpIR_;
    GlobalTensor<float> b_;
    GlobalTensor<float> bPlur_;
    GlobalTensor<float> bPlui_;
    GlobalTensor<float> bsum_;
    GlobalTensor<float> csum_;
};

}  // namespace

extern "C" __global__ __aicore__ void baremix_custom(GM_ADDR ar, GM_ADDR ai, GM_ADDR b, GM_ADDR bPlur,
                                                     GM_ADDR bPlui, GM_ADDR bsum, GM_ADDR csum,
                                                     GM_ADDR workspace, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    TPipe pipe;
    TCubeTiling cubeTiling;
    complex_gram_fused::Params params;
    CopyCubeTiling(&cubeTiling, tilingGm);
    CopyParams(&params, tilingGm);

    if ASCEND_IS_AIC {
        ComplexGramCubeKernel op;
        op.Init(ar, ai, workspace, cubeTiling, params, &pipe);
        op.Process();
    }

    if ASCEND_IS_AIV {
        ComplexGramVectorKernel op;
        op.Init(b, bPlur, bPlui, bsum, csum, workspace, cubeTiling, params, &pipe);
        op.Process();
    }
}
