/**
 * @file complex_gram.cpp
 * Official msOpGen-style host file for ComplexGram.
 *
 * This file intentionally follows the structure shown in Ascend/msopgen quick start:
 *   - include ../op_kernel/<op>_tiling.h
 *   - namespace optiling::TilingFunc
 *   - namespace ge::InferShape / InferDataType
 *   - namespace ops::ComplexGram : public OpDef
 *   - OP_ADD(ComplexGram)
 */

#include "../op_kernel/complex_gram_tiling.h"

#ifdef COMPLEX_GRAM_HOST_STANDALONE
#include <cstdint>
#include <cstdio>
#include <cstdlib>

struct ComplexGramHostTilingResult {
    ComplexGramTilingData tiling;
    uint32_t blockDim;
    uint64_t workspaceBytes;
};

static ComplexGramHostTilingResult BuildComplexGramHostTiling(uint32_t n, uint32_t bm = 32, uint32_t bn = 32)
{
    ComplexGramHostTilingResult r{};
    r.tiling = MakeComplexGramTilingData(n, bm, bn, 20, 40);
    // msOpGen exposes one blockDim. For an AIC/AIV fused kernel, use the generated
    // project's fusion launch configuration to provide AIC=20 and AIV=40. This
    // standalone value is kept as the AIC/cube dimension.
    r.blockDim = r.tiling.cubeBlockNum;
    r.workspaceBytes = r.tiling.singleKernelWorkspaceBytes;
    return r;
}

int main(int argc, char **argv)
{
    const uint32_t n = argc > 1 ? static_cast<uint32_t>(std::atoi(argv[1])) : 16;
    const auto r = BuildComplexGramHostTiling(n);
    std::printf("n=%u K=%u BM=%u BN=%u\n", r.tiling.n, r.tiling.k, r.tiling.bm, r.tiling.bn);
    std::printf("tileM=%u tileN=%u tasks=%u\n", r.tiling.tileMNum, r.tiling.tileNNum, r.tiling.taskNum);
    std::printf("blockDim=%u cubeBlockNum=%u vectorBlockNum=%u\n", r.blockDim,
                r.tiling.cubeBlockNum, r.tiling.vectorBlockNum);
    std::printf("singleKernelWorkspace=%llu bytes fullWorkspace=%llu bytes\n",
                static_cast<unsigned long long>(r.tiling.singleKernelWorkspaceBytes),
                static_cast<unsigned long long>(r.tiling.fullWorkspaceBytes));
    return 0;
}

#else

#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *arShape = context->GetInputShape(0);
    const auto &shape = arShape->GetStorageShape();
    if (shape.GetDimNum() != 3 || shape.GetDim(0) != 272 || shape.GetDim(1) != 256) {
        return ge::GRAPH_FAILED;
    }
    const int64_t k64 = shape.GetDim(2);
    if (k64 <= 0 || (k64 % 8) != 0) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t n = static_cast<uint32_t>(k64 / 8);
    ComplexGramTilingData tiling = MakeComplexGramTilingData(n, 32, 32, 20, 40);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // One msOpGen blockDim field is available here. Official AIC/AIV fusion samples
    // may provide an additional launch config for vectorBlockNum=40. Keep cube dim here.
    context->SetBlockDim(20);

    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = tiling.get_singleKernelWorkspaceBytes();
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *arShape = context->GetInputShape(0);
    const gert::Shape *aiShape = context->GetInputShape(1);
    if (arShape == nullptr || aiShape == nullptr || arShape->GetDimNum() != 3 || aiShape->GetDimNum() != 3) {
        return GRAPH_FAILED;
    }
    if (arShape->GetDim(0) != 272 || arShape->GetDim(1) != 256 ||
        aiShape->GetDim(0) != arShape->GetDim(0) || aiShape->GetDim(1) != arShape->GetDim(1) ||
        aiShape->GetDim(2) != arShape->GetDim(2)) {
        return GRAPH_FAILED;
    }
    const int64_t k = arShape->GetDim(2);
    if (k <= 0 || (k % 8) != 0) {
        return GRAPH_FAILED;
    }
    const int64_t n = k / 8;
    gert::Shape *bShape = context->GetOutputShape(0);
    gert::Shape *bplurShape = context->GetOutputShape(1);
    gert::Shape *bpluiShape = context->GetOutputShape(2);
    gert::Shape *bsumShape = context->GetOutputShape(3);
    gert::Shape *csumShape = context->GetOutputShape(4);
    *bShape = gert::Shape({17, k, k});
    *bplurShape = gert::Shape({17, k, k});
    *bpluiShape = gert::Shape({17, k, k});
    *bsumShape = gert::Shape({k, k});
    *csumShape = gert::Shape({n, n});
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    context->SetOutputDataType(2, ge::DT_FLOAT);
    context->SetOutputDataType(3, ge::DT_FLOAT);
    context->SetOutputDataType(4, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class ComplexGram : public OpDef {
public:
    explicit ComplexGram(const char *name) : OpDef(name)
    {
        this->Input("ar")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Input("ai")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Output("b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Output("bplur")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Output("bplui")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Output("bsum")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Output("csum")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        // Replace this config with the exact AddConfig generated by msOpGen for your SoC.
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(ComplexGram);
} // namespace ops
#endif
