/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */

#include "register/op_impl_registry.h"

namespace ops {

static ge::graphStatus InferShapeWMmadDemo(gert::InferShapeContext *context)
{
    const gert::Shape *b = context->GetInputShape(1);
    gert::Shape *c = context->GetOutputShape(0);
    if (b == nullptr || c == nullptr) {
        return ge::GRAPH_FAILED;
    }
    c->SetDimNum(b->GetDimNum());
    for (size_t i = 0; i < b->GetDimNum(); ++i) {
        c->SetDim(i, b->GetDim(i));
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeWMmadDemo(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(WMmadDemo)
    .InferShape(InferShapeWMmadDemo)
    .InferDataType(InferDataTypeWMmadDemo);

} // namespace ops
