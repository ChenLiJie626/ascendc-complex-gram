/*
 * ComplexGram op prototype skeleton for an msOpGen/CANN custom operator project.
 *
 * This file shows the real pieces that must exist in a production project:
 *   - input/output dtype and shape validation
 *   - output shape inference
 *   - operator registration
 *
 * The exact macro names vary by CANN/msOpGen version. Keep this file as the place
 * to paste the generated prototype from msOpGen, then fill the shape logic below.
 */

#include <cstdint>

// Typical CANN generated projects include headers similar to these. Uncomment and
// adjust to your generated project.
// #include "register/op_def_registry.h"
// #include "graph/operator_reg.h"

namespace complex_gram_proto {

struct Shape3D {
    int64_t d0;
    int64_t d1;
    int64_t d2;
};

struct ComplexGramShapes {
    Shape3D ar;
    Shape3D ai;
    int64_t n;
    int64_t k;
    int64_t b[3];      // [17,K,K]
    int64_t bplur[3];  // [17,K,K]
    int64_t bplui[3];  // [17,K,K]
    int64_t bsum[2];   // [K,K]
    int64_t csum[2];   // [n,n]
};

inline bool InferComplexGramShapes(const Shape3D &ar, const Shape3D &ai, ComplexGramShapes &out)
{
    if (ar.d0 != 272 || ar.d1 != 256) {
        return false;
    }
    if (ai.d0 != ar.d0 || ai.d1 != ar.d1 || ai.d2 != ar.d2) {
        return false;
    }
    if (ar.d2 <= 0 || (ar.d2 % 8) != 0) {
        return false;
    }
    const int64_t n = ar.d2 / 8;
    const int64_t k = ar.d2;
    out.ar = ar;
    out.ai = ai;
    out.n = n;
    out.k = k;
    out.b[0] = 17; out.b[1] = k; out.b[2] = k;
    out.bplur[0] = 17; out.bplur[1] = k; out.bplur[2] = k;
    out.bplui[0] = 17; out.bplui[1] = k; out.bplui[2] = k;
    out.bsum[0] = k; out.bsum[1] = k;
    out.csum[0] = n; out.csum[1] = n;
    return true;
}

}  // namespace complex_gram_proto

/*
Paste/replace with the concrete registration emitted by msOpGen, for example:

IMPLEMT_COMMON_INFERFUNC(ComplexGramInferShape) {
    auto arShape = op.GetInputDesc(0).GetShape();
    auto aiShape = op.GetInputDesc(1).GetShape();
    // validate [272,256,K], K%8==0
    // set output desc shapes:
    //   y0 B      [17,K,K]
    //   y1 BPlur  [17,K,K]
    //   y2 BPlui  [17,K,K]
    //   y3 Bsum   [K,K]
    //   y4 Csum   [K/8,K/8]
    return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(ComplexGram, ComplexGramInferShape);

REG_OP(ComplexGram)
    .INPUT(ar, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(ai, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(b, TensorType({DT_FLOAT}))
    .OUTPUT(b_plur, TensorType({DT_FLOAT}))
    .OUTPUT(b_plui, TensorType({DT_FLOAT}))
    .OUTPUT(bsum, TensorType({DT_FLOAT}))
    .OUTPUT(csum, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(ComplexGram);
*/
