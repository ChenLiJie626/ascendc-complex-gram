/*
 * ComplexGram host tiling skeleton for msOpGen/CANN.
 *
 * Production responsibilities:
 *   1. Read Ar/Ai shapes and dtype from gert::TilingContext.
 *   2. Compute n, K, BM/BN, taskNum, workspace sizes.
 *   3. Generate/copy Matmul tiling data required by AscendC Matmul API.
 *   4. Write ComplexGramTilingData to raw tiling buffer.
 *   5. Set AIC/AIV block dimensions according to the official fusion-kernel API.
 *
 * This file is kept compilable as a standalone calculator when CANN headers are not
 * present by defining COMPLEX_GRAM_TILING_STANDALONE.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "../op_kernel/complex_gram_tiling.h"

struct ComplexGramHostTilingResult {
    ComplexGramTilingData tiling;
    uint32_t cubeBlockDim;
    uint32_t vectorBlockDim;
    uint64_t workspaceBytes;
};

ComplexGramHostTilingResult BuildComplexGramHostTiling(int32_t n, int32_t bm = 32, int32_t bn = 32)
{
    ComplexGramHostTilingResult r{};
    r.tiling = MakeComplexGramTiling(n, bm, bn, 20, 40);
    r.cubeBlockDim = static_cast<uint32_t>(r.tiling.cubeBlockNum);
    r.vectorBlockDim = static_cast<uint32_t>(r.tiling.vectorBlockNum);
    // Use single-kernel ping-pong workspace for the fused AIC/AIV epilogue.
    // The reduce kernel needs no extra workspace in this skeleton.
    r.workspaceBytes = r.tiling.singleKernelWorkspaceBytes;
    return r;
}

#ifdef COMPLEX_GRAM_TILING_STANDALONE
int main(int argc, char **argv)
{
    const int32_t n = argc > 1 ? std::atoi(argv[1]) : 16;
    const auto r = BuildComplexGramHostTiling(n);
    std::printf("n=%d K=%d BM=%d BN=%d\n", r.tiling.n, r.tiling.k, r.tiling.bm, r.tiling.bn);
    std::printf("tileM=%d tileN=%d tasks=%d\n", r.tiling.tileMNum, r.tiling.tileNNum, r.tiling.taskNum);
    std::printf("cubeBlockDim=%u vectorBlockDim=%u singleKernelWorkspace=%llu bytes fullWorkspace=%llu bytes\n",
                r.cubeBlockDim, r.vectorBlockDim,
                static_cast<unsigned long long>(r.tiling.singleKernelWorkspaceBytes),
                static_cast<unsigned long long>(r.tiling.workspaceBytes));
    return 0;
}
#endif

#ifndef COMPLEX_GRAM_TILING_STANDALONE
// Replace this block with the exact function signature and registration generated
// by your msOpGen version. The pseudo-code below is intentionally explicit.
/*
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "graph/utils/type_utils.h"
#include "complex_gram_tiling.h"

namespace optiling {
static ge::graphStatus TilingComplexGram(gert::TilingContext *context)
{
    const auto arShape = context->GetInputShape(0)->GetStorageShape();
    const int64_t k = arShape.GetDim(2);
    if (arShape.GetDim(0) != 272 || arShape.GetDim(1) != 256 || k % 8 != 0) {
        return ge::GRAPH_FAILED;
    }
    const int32_t n = static_cast<int32_t>(k / 8);
    auto r = BuildComplexGramHostTiling(n, 32, 32);

    // Workspace 0: AIC/AIV ping-pong workspace.
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = r.workspaceBytes;

    // In generated projects, setting AIC/AIV block dims may require a tiling key
    // and a fusion-specific launch config. Consult the official AIC/AIV fusion sample.
    context->SetBlockDim(r.cubeBlockDim);

    auto *raw = context->GetRawTilingData();
    auto *data = raw->GetData<ComplexGramTilingData>();
    *data = r.tiling;
    raw->SetDataSize(sizeof(ComplexGramTilingData));
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("ComplexGram", TilingComplexGram, 1);
}  // namespace optiling
*/
#endif
