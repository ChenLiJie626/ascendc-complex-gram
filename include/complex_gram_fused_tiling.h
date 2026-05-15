#ifndef COMPLEX_GRAM_FUSED_TILING_H
#define COMPLEX_GRAM_FUSED_TILING_H

#include <cstddef>
#include <cstdint>

namespace complex_gram_fused {

constexpr uint32_t GROUP_NUM = 17;
constexpr uint32_t AVG_NUM = 16;
constexpr uint32_t K_DIM = 256;
constexpr uint32_t USER_VEC = 8;
constexpr uint32_t TMP_MATMUL_NUM = 4;
constexpr uint32_t AIV_PER_AIC = 2;

struct ComplexGramFusedParams {
    uint32_t n;
    uint32_t u;
    uint32_t k;
    uint32_t blockDim;
    uint32_t sysWorkspaceSize;
    uint32_t reserved;
    uint64_t userWorkspaceSize;
};

inline uint64_t AlignUp(uint64_t value, uint64_t align)
{
    return (value + align - 1) / align * align;
}

inline uint64_t MatrixElems(uint32_t n)
{
    const uint64_t u = static_cast<uint64_t>(n) * USER_VEC;
    return u * u;
}

inline uint64_t UserWorkspaceBytes(uint32_t n)
{
    return TMP_MATMUL_NUM * MatrixElems(n) * sizeof(float);
}

#ifndef __CCE_AICORE__
size_t GetTilingBufferSize();

bool GenerateTiling(const char *socVersion, uint32_t n, uint32_t requestedBlockDim, uint8_t *tilingBuffer,
                    size_t tilingBufferSize, uint32_t *actualBlockDim, size_t *workspaceSize);
#endif

}  // namespace complex_gram_fused

#endif  // COMPLEX_GRAM_FUSED_TILING_H
