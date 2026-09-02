/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */

#include "kernel_operator.h"
#include "w_mmad_demo.h"

using namespace AscendC;

// Dirty L0A with fp32 64x64 MMAD, then bf16 [64,64]@[64,128].
extern "C" __global__ __aicore__ void w_mmad_demo(
    GM_ADDR a, GM_ADDR b, GM_ADDR pre, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA(tilingData, tiling);
    (void)tilingData;
    (void)workspace;
    WMmadDemoOp::Kernel op;
    op.Init(a, b, pre, c);
    op.Process();
}
