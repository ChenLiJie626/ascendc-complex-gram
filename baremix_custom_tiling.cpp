/**
 * @file baremix_custom_tiling.cpp
 */
#include <cstring>
#include <cstdint>
#include <iostream>

#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

using namespace matmul_tiling;

namespace complex_gram_fused {
constexpr uint32_t K_DIM = 256;
constexpr uint32_t USER_VEC = 8;
constexpr uint32_t AVG_NUM = 16;
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

uint64_t AlignUp(uint64_t value, uint64_t align)
{
    return (value + align - 1) / align * align;
}

uint32_t CeilDiv(uint32_t value, uint32_t div)
{
    return div == 0 ? 0 : (value + div - 1) / div;
}

uint64_t MatrixElems(uint32_t n)
{
    const uint64_t u = static_cast<uint64_t>(n) * USER_VEC;
    return u * u;
}

uint64_t UserWorkspaceBytes(uint32_t n)
{
    return TMP_MATMUL_NUM * AVG_NUM * MatrixElems(n) * sizeof(float);
}
}  // namespace complex_gram_fused

size_t GetTilingBufferSize()
{
    return complex_gram_fused::AlignUp(sizeof(optiling::TCubeTiling), 32) +
           complex_gram_fused::AlignUp(sizeof(complex_gram_fused::Params), 32);
}

void GenerateTiling(const char *socVersion, uint32_t n, uint8_t *tilingBuf, size_t tilingBufSize,
                    uint32_t &blockDim, size_t &workspaceSize)
{
    if (n == 0 || tilingBuf == nullptr || tilingBufSize < GetTilingBufferSize()) {
        std::cout << "invalid complex gram tiling args" << std::endl;
        return;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    if (ascendcPlatform == nullptr) {
        std::cout << "failed to get AscendC platform" << std::endl;
        return;
    }

    const uint32_t u = n * complex_gram_fused::USER_VEC;
    const uint32_t mBlocksAtMinTile = complex_gram_fused::CeilDiv(u, 16);
    const uint32_t nBlocksAtMinTile = complex_gram_fused::CeilDiv(u, 16);
    const uint32_t maxAic = ascendcPlatform->GetCoreNumAic();
    blockDim = mBlocksAtMinTile * nBlocksAtMinTile;
    if (blockDim == 0) {
        blockDim = 1;
    }
    if (blockDim > maxAic) {
        blockDim = maxAic;
    }

    optiling::TCubeTiling tilingData;
    MultiCoreMatmulTiling tilingApi(*ascendcPlatform);
    tilingApi.SetDim(blockDim);
    tilingApi.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, true);
    tilingApi.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, false);
    tilingApi.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
    tilingApi.SetOrgShape(u, u, complex_gram_fused::K_DIM);
    tilingApi.SetShape(u, u, complex_gram_fused::K_DIM);
    tilingApi.SetBias(false);
    tilingApi.SetTraverse(MatrixTraverse::FIRSTM);
    tilingApi.SetBufferSpace(-1, -1, -1);

    int64_t res = tilingApi.GetTiling(tilingData);
    tilingData.set_stepM(1);
    tilingData.set_stepN(1);
    if (res == -1) {
        std::cout << "gen tiling failed" << std::endl;
        return;
    }
    const size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
    const size_t userWorkspaceSize = static_cast<size_t>(complex_gram_fused::UserWorkspaceBytes(n));
    workspaceSize = systemWorkspaceSize + userWorkspaceSize;

    complex_gram_fused::Params params{};
    params.n = n;
    params.u = u;
    params.k = complex_gram_fused::K_DIM;
    params.blockDim = blockDim;
    params.sysWorkspaceSize = static_cast<uint32_t>(systemWorkspaceSize);
    params.userWorkspaceSize = userWorkspaceSize;

    std::memset(tilingBuf, 0, tilingBufSize);
    tilingData.SaveToBuffer(tilingBuf, tilingData.GetDataSize());
    const size_t paramOffset = complex_gram_fused::AlignUp(sizeof(optiling::TCubeTiling), 32);
    std::memcpy(tilingBuf + paramOffset, &params, sizeof(params));
}
