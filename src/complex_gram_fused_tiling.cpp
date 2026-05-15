#include "complex_gram_fused_tiling.h"

#include <cstring>
#include <iostream>

#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace complex_gram_fused {

namespace {

uint32_t CeilDivHost(uint32_t value, uint32_t div)
{
    return div == 0 ? 0 : (value + div - 1) / div;
}

uint32_t PickBlockDim(platform_ascendc::PlatformAscendC *platform, uint32_t u)
{
    const uint32_t maxAic = platform->GetCoreNumAic();
    const uint32_t minBlock = CeilDivHost(u, 16) * CeilDivHost(u, 16);
    if (minBlock == 0) {
        return 1;
    }
    return minBlock < maxAic ? minBlock : maxAic;
}

}  // namespace

size_t GetTilingBufferSize()
{
    return AlignUp(sizeof(optiling::TCubeTiling), 32) + AlignUp(sizeof(ComplexGramFusedParams), 32);
}

bool GenerateTiling(const char *socVersion, uint32_t n, uint32_t requestedBlockDim, uint8_t *tilingBuffer,
                    size_t tilingBufferSize, uint32_t *actualBlockDim, size_t *workspaceSize)
{
    if (n == 0 || tilingBuffer == nullptr || tilingBufferSize < GetTilingBufferSize()) {
        std::cerr << "invalid complex_gram_fused tiling arguments" << std::endl;
        return false;
    }

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    if (platform == nullptr) {
        std::cerr << "failed to get AscendC platform" << std::endl;
        return false;
    }

    const uint32_t u = n * USER_VEC;
    uint32_t blockDim = requestedBlockDim == 0 ? PickBlockDim(platform, u) : requestedBlockDim;
    if (blockDim == 0) {
        blockDim = 1;
    }

    optiling::TCubeTiling cubeTilingData;
    matmul_tiling::MultiCoreMatmulTiling cubeTiling(*platform);
    cubeTiling.SetDim(blockDim);
    cubeTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_FLOAT16, true);
    cubeTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_FLOAT16, false);
    cubeTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetBias(false);
    cubeTiling.SetShape(u, u, K_DIM);
    cubeTiling.SetOrgShape(u, u, K_DIM);
    cubeTiling.SetBufferSpace(-1, -1, -1);

    if (cubeTiling.GetTiling(cubeTilingData) == -1) {
        std::cerr << "complex_gram_fused matmul tiling failed" << std::endl;
        return false;
    }

    const size_t sysWorkspace = static_cast<size_t>(platform->GetLibApiWorkSpaceSize());
    const size_t userWorkspace = static_cast<size_t>(UserWorkspaceBytes(n));
    ComplexGramFusedParams params{};
    params.n = n;
    params.u = u;
    params.k = K_DIM;
    params.blockDim = blockDim;
    params.sysWorkspaceSize = static_cast<uint32_t>(sysWorkspace);
    params.userWorkspaceSize = userWorkspace;

    std::memset(tilingBuffer, 0, tilingBufferSize);
    cubeTilingData.SaveToBuffer(tilingBuffer, cubeTilingData.GetDataSize());
    const size_t paramOffset = AlignUp(sizeof(optiling::TCubeTiling), 32);
    std::memcpy(tilingBuffer + paramOffset, &params, sizeof(params));

    if (actualBlockDim != nullptr) {
        *actualBlockDim = blockDim;
    }
    if (workspaceSize != nullptr) {
        *workspaceSize = sysWorkspace + userWorkspace;
    }
    return true;
}

}  // namespace complex_gram_fused
