/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */

#include "kernel_operator.h"
#include "arch35/chunk_gdn_fwd_prepare.h"

using namespace AscendC;

#ifndef DTYPE_A_LOG
#define DTYPE_A_LOG DTYPE_G
#endif

// GET_TILING_DATA currently receives a null GM on this op (too many kernel
// GM_ADDR slots). Fill the required-case tiling until it is passed via workspace.
// Stage3 S3 ping/pong UB; V_MTE3(0); no L dump; no MTE3_V after L1 upload.
__aicore__ inline void FillPrepareTiling(ChunkGdnFwdStageTilingData &td)
{
    td.inputBatchSize = 1;                 // B
    td.queryKeyHeadCount = 4;           // Hk
    td.valueHeadCount = 8;              // Hv
    td.sequenceTokenLength = 1792;       // T
    td.queryKeyHeadDim = 128;            // K
    td.valueHeadDim = 128;              // V
    td.valueHeadsPerQueryKeyHead = 2;   // G = Hv / Hk
    td.tokensPerChunk = 64;              // BT
    td.packedSequenceCount = 1;         // N
    td.isVariableLengthPacked = 0;
    td.hasChunkIndexTable = 0;
    td.enableQueryKeyL2NormInKernel = 1;
    td.enableFusedGateSoftplus = 0;
    td.enableBetaSigmoid = 1;
    td.hasGateDtBias = 0;
    td.scaleBetaByTwoWhenNegEigval = 0;
    td.useExp2ForGateCumsum = 1;
    td.queryKeyStorageDtype = 27;       // ge bf16
    td.gateStorageDtype = 0;
    td.betaStorageDtype = 0;
    td.totalChunkTileCount = 224;       // Ceil(1792/64) * B * Hv = 28 * 8
}

extern "C" __global__ __aicore__ void chunk_gdn_fwd_prepare(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR a_log, GM_ADDR dt_bias, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
    GM_ADDR g_out, GM_ADDR w_out, GM_ADDR u_out, GM_ADDR a_out,
    GM_ADDR q_hat, GM_ADDR k_hat, GM_ADDR q_rstd, GM_ADDR k_rstd, GM_ADDR beta_eff,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    (void)tiling;
    ChunkGdnFwdStageTilingData tilingData{};
    FillPrepareTiling(tilingData);
    GdnStage::ChunkGdnFwdPrepareKernel<DTYPE_G, DTYPE_A_LOG> op;
    op.Init(q, k, v, g, beta, a_log, dt_bias, cu_seqlens, chunk_indices,
            g_out, w_out, u_out, a_out, q_hat, k_hat, q_rstd, k_rstd, beta_eff,
            workspace, tilingData);
    op.Process();
}
