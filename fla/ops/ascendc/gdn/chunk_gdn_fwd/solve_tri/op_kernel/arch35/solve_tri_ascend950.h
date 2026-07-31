/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */

 #ifndef SOLVE_TRI_ASCEND950_H
 #define SOLVE_TRI_ASCEND950_H
 
 #include "kernel_operator.h"
 #include "solve_tri_ascend950_common.h"
 #include "mem.h"
 
 using namespace AscendC;
 
 template <typename T>
 __aicore__ inline void TransposeB16(LocalTensor<T> dst, LocalTensor<T> src, uint32_t curTileLen)
 {
     TransDataTo5HDParams params;
     LocalTensor<T> srcLocalList[DATA_BLOCK_COUNT];
     LocalTensor<T> dstLocalList[DATA_BLOCK_COUNT];
     uint32_t aRepeartTimes = curTileLen / 16;
     params.repeatTimes = aRepeartTimes;
     params.srcRepStride = aRepeartTimes == 1 ? 0 : 1;
     params.dstRepStride = aRepeartTimes == 1 ? 0 : 16;
     for (int32_t i = 0; i < DATA_BLOCK_COUNT; i++) {
         uint32_t offset = curTileLen * i;
         srcLocalList[i] = src[offset];
     }
     for (int32_t i = 0; i < DATA_BLOCK_COUNT; i++) {
         uint32_t offset = 16 * i;
         dstLocalList[i] = dst[offset];
     }
     AscendC::TransDataTo5HD<T>(dstLocalList, srcLocalList, params);
 }
 
 
 template<typename T, typename U>
 __simd_vf__ inline void MulReduceScatterVF(__ubuf__ T* dstAddr, __ubuf__ T* src0Addr, __ubuf__ T* src1Addr, __ubuf__ U* idxAddr, uint32_t scatterCount, uint32_t oneRepeatSize)
 {
     AscendC::Reg::RegTensor<T> srcReg0;
     AscendC::Reg::RegTensor<T> srcReg1;
     AscendC::Reg::RegTensor<T> dstMulReg;
     AscendC::Reg::RegTensor<T> dstReduceBlkReg;
     AscendC::Reg::RegTensor<T> dstReducePairReg;
     AscendC::Reg::RegTensor<U> scatterIdxReg;
     
     uint32_t maskCount = oneRepeatSize;
     AscendC::Reg::MaskReg inputMask;
     AscendC::Reg::MaskReg scatterMask;
     inputMask = AscendC::Reg::UpdateMask<T, AscendC::Reg::RegTraitNumOne>(maskCount);
     scatterMask = AscendC::Reg::UpdateMask<T, AscendC::Reg::RegTraitNumOne>(scatterCount);
     for(uint16_t iterIdx = 1; iterIdx < 16; iterIdx++){
         AscendC::Reg::LoadAlign(scatterIdxReg, idxAddr);
         AscendC::Reg::Adds(scatterIdxReg, scatterIdxReg, (uint32_t)iterIdx, scatterMask);
         for (uint16_t i = 0; i < iterIdx; i++) {
             AscendC::Reg::LoadAlign(srcReg0, src0Addr + iterIdx * oneRepeatSize);
             AscendC::Reg::LoadAlign(srcReg1, src1Addr + i * oneRepeatSize);
             AscendC::Reg::Mul(dstMulReg, srcReg0, srcReg1, inputMask);
             AscendC::Reg::ReduceDataBlock<AscendC::Reg::ReduceType::SUM>(dstReduceBlkReg, dstMulReg, inputMask);
             AscendC::Reg::PairReduceElem<AscendC::Reg::PairReduce::SUM>(dstReducePairReg, dstReduceBlkReg, inputMask);
             AscendC::Reg::Scatter(dstAddr + i * oneRepeatSize, dstReducePairReg, scatterIdxReg, scatterMask);
         }
         // AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_ALL, AscendC::Reg::MemType::VEC_ALL>();
         AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
     }
 }
 
 // template<typename T, typename U>
 // __simd_vf__ inline void MulReduceScatterVF(__ubuf__ T* dstAddr, __ubuf__ T* src0Addr, __ubuf__ T* src1Addr, __ubuf__ U* idxAddr, uint32_t scatterCount, uint32_t oneRepeatSize, uint32_t repeatTimes)
 // {
 //     AscendC::Reg::RegTensor<T> srcReg0;
 //     AscendC::Reg::RegTensor<T> srcReg1;
 //     AscendC::Reg::RegTensor<T> dstMulReg;
 //     AscendC::Reg::RegTensor<T> dstReduceBlkReg;
 //     AscendC::Reg::RegTensor<T> dstReducePairReg;
 //     AscendC::Reg::RegTensor<U> scatterIdxReg;
     
 //     uint32_t maskCount = oneRepeatSize;
 //     AscendC::Reg::MaskReg inputMask;
 //     AscendC::Reg::MaskReg scatterMask;
 //     inputMask = AscendC::Reg::UpdateMask<T, AscendC::Reg::RegTraitNumOne>(maskCount);
 //     scatterMask = AscendC::Reg::UpdateMask<T, AscendC::Reg::RegTraitNumOne>(scatterCount);
 
     
 //     AscendC::Reg::LoadAlign(scatterIdxReg, idxAddr);
 //     AscendC::Reg::Adds(scatterIdxReg, scatterIdxReg, repeatTimes, scatterMask);
 //     for (uint16_t i = 0; i < repeatTimes; i++) {
 //         AscendC::Reg::LoadAlign(srcReg0, src0Addr + repeatTimes * oneRepeatSize);
 //         AscendC::Reg::LoadAlign(srcReg1, src1Addr + i * oneRepeatSize);
 //         AscendC::Reg::Mul(dstMulReg, srcReg0, srcReg1, inputMask);
 //         AscendC::Reg::ReduceDataBlock<AscendC::Reg::ReduceType::SUM>(dstReduceBlkReg, dstMulReg, inputMask);
 //         AscendC::Reg::PairReduceElem<AscendC::Reg::PairReduce::SUM>(dstReducePairReg, dstReduceBlkReg, inputMask);
 //         AscendC::Reg::Scatter(dstAddr + i * oneRepeatSize, dstReducePairReg, scatterIdxReg, scatterMask);
 //     }
 //     // AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
 // }
 
 
 template <typename InDtype, typename OutDtype>
 class SolveTri {
 public:
     __aicore__ inline void Init(GM_ADDR aGm, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR outGm,
                                 GM_ADDR workspace, const SolveTriTilingData *tilingData)
     {
         // Tiling
         batch_size = tilingData->batchSize;
         seq_length = tilingData->seqLen;
         num_head = tilingData->numHeads;
         chunk_size = tilingData->chunkSize;
         chunk_num_in_seq = tilingData->numChunks;
         chunk_num_total = tilingData->totalTiles; // 主循环上界 = 全部 tile 数
         mode = tilingData->layoutMode;            // 0=bnsd, 1=bsnd, 2=tnd, 3=ntd
         is_lower = tilingData->isLower;
         tiles_per_core = tilingData->tilesPerCore;
         total_tokens = tilingData->totalTokens;   // NTD: total_T（head 维在外的偏移计算）
 
         // GM（INT64 索引）
         gm_a.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(aGm));
         gm_cu_seqlens.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cu_seqlens));
         gm_chunk_indices.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(chunk_indices));
         gm_out.SetGlobalBuffer(reinterpret_cast<__gm__ OutDtype *>(outGm));
         OnChipBuffer buf;
 
         ub_A = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(0);
         ub_A_fp32 = buf.template GetBuffer<BufferType::ASCEND_UB, float>(2 * 1024);
         ub_I = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(6 * 1024);
         ub_I_fp32 = buf.template GetBuffer<BufferType::ASCEND_UB, float>(8 * 1024);
         ub_res = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(12 * 1024);
         ub_idx_b32 = buf.template GetBuffer<BufferType::ASCEND_UB, uint32_t>(14 * 1024);
         
         // Core
         num_core = AscendC::GetBlockNum();
         core_idx = AscendC::GetBlockIdx();
         sub_block_idx = AscendC::GetSubBlockIdx();
         
         // 辅助矩阵缓存标记：0 为无效初值（cur 恒 >=16），首个 tile 必触发生成
         last_chunk_size = 0;
     }
 
     __aicore__ inline int64_t CeilDiv(int64_t a, int64_t b)
     {
         return (a + b - 1) / b;
     }
 
     __aicore__ inline void ub_to_l1(AscendC::LocalTensor<InDtype> l1Tensor,
                                     AscendC::LocalTensor<InDtype> ubTensor, uint32_t chunkSize)
     {
         AscendC::DataCopy(l1Tensor, ubTensor,
                           AscendC::DataCopyParams(1, chunkSize * chunkSize / 16, 0, 0));
     }
 
     __aicore__ inline void FixpipeL0cToL1(AscendC::LocalTensor<InDtype> l1Tensor,
                                           AscendC::LocalTensor<float> l0CTensor, uint32_t chunkSize)
     {
         AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::NZ> fixPipeParams;
         fixPipeParams.nSize = chunkSize;
         fixPipeParams.mSize = chunkSize;
         fixPipeParams.srcStride = chunkSize;
         fixPipeParams.dstStride = chunkSize * 16;
         if constexpr (std::is_same_v<InDtype, half>) {
             fixPipeParams.quantPre = QuantMode_t::F322F16;
         } else {
             fixPipeParams.quantPre = QuantMode_t::F322BF16;
         }
         AscendC::Fixpipe<InDtype, float, CFG_NZ_L1>(l1Tensor, l0CTensor, fixPipeParams);
     }
 
     __aicore__ inline void FixpipeL0cToUB(AscendC::LocalTensor<InDtype> ubTensor,
                                           AscendC::LocalTensor<float> l0CTensor, uint32_t chunkSize)
     {
         AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::NZ> fixPipeParams;
         fixPipeParams.nSize = chunkSize;
         fixPipeParams.mSize = chunkSize;
         fixPipeParams.srcStride = chunkSize;
         fixPipeParams.dstStride = chunkSize * 16;
         fixPipeParams.dualDstCtl = 0;
         fixPipeParams.subBlockId = 0;
         if constexpr (std::is_same_v<InDtype, half>) {
             fixPipeParams.quantPre = QuantMode_t::F322F16;
         } else {
             fixPipeParams.quantPre = QuantMode_t::F322BF16;
         }
         AscendC::Fixpipe<InDtype, float, CFG_NZ_UB>(ubTensor, l0CTensor, fixPipeParams);
     }
 
     __aicore__ inline int64_t ChunkAlign(int64_t cur_chunk)
     {
         if (cur_chunk <= 16)
             return 16;
         if (cur_chunk <= 32)
             return 32;
         if (cur_chunk <= 64)
             return 64;
         return 128;
     }
 
     // 结果写回：L0C(FP32, NZ) -> gm_out(ND, 行优先)。
     //   validRows = 有效行数（尾块 < curSize），只写 validRows 行避免跨序列覆写。
     //   curSize   = 对齐后 chunk 尺寸（= L0C/输出的列数与源行跨度）。
     //   dstStride = GM 物理行跨度。
     __aicore__ inline void FixpipeL0cToGM(AscendC::GlobalTensor<OutDtype> gmTensor,
                                           AscendC::LocalTensor<float> l0CTensor,
                                           uint32_t validRows, uint32_t curSize, uint32_t dstStride)
     {
         auto intriParams = AscendC::FixpipeParamsV220(curSize, validRows, curSize, dstStride, false);
         if constexpr (std::is_same_v<OutDtype, half>) {
             intriParams.quantPre = QuantMode_t::F322F16;
         } else {
             intriParams.quantPre = QuantMode_t::F322BF16;
         }
         AscendC::Fixpipe<OutDtype, float, AscendC::CFG_ROW_MAJOR>(gmTensor, l0CTensor, intriParams);
     }
 
 
     // 由 loop_idx 计算该 tile 的 GM 偏移、对齐后 chunk 尺寸 cur_size 与有效行数 actual_size。
     __aicore__ inline void ComputeTile(int64_t loop_idx, int64_t &x_gm_offset,
                                        int64_t &cur_size, int64_t &actual_size)
     {
         int64_t seq_idx = 0;
         int64_t chunk_in_seq_idx = 0;
         int64_t head_idx = 0;
         int64_t chunk_idx = 0;
         int64_t local_seq_length = seq_length;
         int64_t local_chunk_num_in_seq = chunk_num_in_seq;
 
         if (mode == 0) { // BNSD: [B, H, S, BT]  (chunks contiguous, row_stride = BT)
             seq_idx = loop_idx / (chunk_num_in_seq * num_head);
             head_idx = (loop_idx / chunk_num_in_seq) % num_head;
             chunk_in_seq_idx = loop_idx % chunk_num_in_seq;
             x_gm_offset = seq_idx * num_head * seq_length * chunk_size +
                           head_idx * seq_length * chunk_size +
                           chunk_in_seq_idx * chunk_size * chunk_size;
         } else if (mode == 1) { // BSND: [B, S, H, BT]  (non-contiguous, row_stride = H*BT)
             seq_idx = loop_idx / (chunk_num_in_seq * num_head);
             chunk_in_seq_idx = loop_idx % (chunk_num_in_seq * num_head) / num_head;
             head_idx = loop_idx % (chunk_num_in_seq * num_head) % num_head;
             x_gm_offset = seq_idx * seq_length * num_head * chunk_size +
                           chunk_in_seq_idx * chunk_size * num_head * chunk_size +
                           head_idx * chunk_size;
         } else if (mode == 2) { // TND varlen: [total_T, H, BT]; B = 1  (non-contiguous, row_stride = H*BT)
             chunk_idx = loop_idx / num_head;
             head_idx = loop_idx % num_head;
             seq_idx = gm_chunk_indices.GetValue(chunk_idx * 2);
             chunk_in_seq_idx = gm_chunk_indices.GetValue(chunk_idx * 2 + 1);
             local_seq_length = gm_cu_seqlens.GetValue(seq_idx + 1) - gm_cu_seqlens.GetValue(seq_idx);
             local_chunk_num_in_seq = CeilDiv(local_seq_length, chunk_size);
             int64_t bos = gm_cu_seqlens.GetValue(seq_idx);
             x_gm_offset = (bos + chunk_in_seq_idx * chunk_size) * num_head * chunk_size +
                           head_idx * chunk_size;
         } else { // NTD varlen: [H, total_T, BT]; B = 1  (contiguous, row_stride = BT)
             chunk_idx = loop_idx / num_head;
             head_idx = loop_idx % num_head;
             seq_idx = gm_chunk_indices.GetValue(chunk_idx * 2);
             chunk_in_seq_idx = gm_chunk_indices.GetValue(chunk_idx * 2 + 1);
             local_seq_length = gm_cu_seqlens.GetValue(seq_idx + 1) - gm_cu_seqlens.GetValue(seq_idx);
             local_chunk_num_in_seq = CeilDiv(local_seq_length, chunk_size);
             int64_t bos = gm_cu_seqlens.GetValue(seq_idx);
             // head 在最外维: head_idx * total_T * BT + (bos + chunk*BT) * BT
             x_gm_offset = head_idx * total_tokens * chunk_size +
                           (bos + chunk_in_seq_idx * chunk_size) * chunk_size;
         }
 
         bool is_last = (chunk_in_seq_idx == (local_chunk_num_in_seq - 1));
         actual_size = is_last ? (local_seq_length - chunk_in_seq_idx * chunk_size) : chunk_size;
         cur_size = is_last ? ChunkAlign(actual_size) : chunk_size;
     }
 
     // ---- Process：CrossCoreFlag + round-robin，MCH 与 MBH 每 tile 逐层握手 ----
     __aicore__ inline void Process()
     {
         int32_t drvStart = is_lower ? 1 : 0;
         int32_t othStart = is_lower ? 0 : 1;
         // BNSD(0)/NTD(3): 单 chunk 内数据连续, row_stride = BT
         // BSND(1)/TND(2): 单 chunk 内数据不连续, row_stride = H*BT
         int64_t row_stride = (mode == 0 || mode == 3) ? chunk_size : (num_head * chunk_size);
 
         if ASCEND_IS_AIV {
             if (sub_block_idx == 0) {
                 for (int64_t loop_idx = core_idx / 2; loop_idx < chunk_num_total; loop_idx += num_core) {
                     int64_t x_gm_offset = 0;
                     int64_t cur = 0;
                     int64_t actual_size = 0;
                     ComputeTile(loop_idx, x_gm_offset, cur, actual_size);
 
                     // 辅助矩阵生成 K方向上拼接的4个16*16单位矩阵 
                     // 这个计划常驻UB
                     AscendC::Duplicate(ub_I, (InDtype)0, 1024);
                     SetFlag<AscendC::HardEvent::V_S>(0); 
                     WaitFlag<AscendC::HardEvent::V_S>(0);
                     for (uint32_t i = 0; i < 16; i++) {
                         ub_I.SetValue((uint32_t)(i * 65), 1);
                     }
                     SetFlag<AscendC::HardEvent::S_V>(0); 
                     WaitFlag<AscendC::HardEvent::S_V>(0);
                     for(uint64_t i = 1; i < 4; i++){
                         AscendC::DataCopy(ub_I[i * 16], ub_I, AscendC::DataCopyParams(16, 1, 3, 3));
                     }
                     AscendC::Cast(ub_I_fp32, ub_I, AscendC::RoundMode::CAST_NONE, 1024);
 
                     // 把A的叶子节点搬入UB，搬入4个按照K方向上拼接
                     uint16_t src_blk_stride = static_cast<uint16_t>(row_stride / 16 - 1);
                     uint16_t des_blk_stride = static_cast<uint16_t>(actual_size / 16 - 1);
                     uint64_t num_valid_fracs = static_cast<uint64_t>(CeilDiv(actual_size, 16));
                     for (uint64_t i = 0; i < num_valid_fracs; i++) {
 
                         uint64_t srcOffset = i * (16 * (uint64_t)row_stride + 16);
                         uint64_t dstOffset = i * 16;
                         AscendC::DataCopy(ub_A[dstOffset], gm_a[x_gm_offset + srcOffset], AscendC::DataCopyParams(16, 1, src_blk_stride, des_blk_stride));
                     }
                     SetFlag<AscendC::HardEvent::MTE2_V>(0); 
                     WaitFlag<AscendC::HardEvent::MTE2_V>(0);
                     
                     // A -> -A
                     AscendC::Muls(ub_A, ub_A, -1, 1024);
                     // Cast A fp16 -> fp32
                     AscendC::Cast(ub_A_fp32, ub_A, AscendC::RoundMode::CAST_NONE, 1024);
                     
                     // 创建scatter indexTensor
                     AscendC::Duplicate(ub_idx_b32, (uint32_t)0, 4);
                     SetFlag<AscendC::HardEvent::V_S>(0); 
                     WaitFlag<AscendC::HardEvent::V_S>(0);
                     for (uint32_t i = 0; i < 4; i++) {
                         ub_idx_b32.SetValue((uint32_t)(i), (uint32_t)(16 * i));
                     }
                     SetFlag<AscendC::HardEvent::S_V>(0); 
                     WaitFlag<AscendC::HardEvent::S_V>(0);
                     // AscendC::DumpTensor(ub_idx_b32, 0, 4);
                     
                     // 调用 VF 计算 mul + reduceSum + scatter
                     __ubuf__ float* src0Addr = reinterpret_cast<__ubuf__ float*>(ub_A_fp32.GetPhyAddr());
                     __ubuf__ float* src1Addr = reinterpret_cast<__ubuf__ float*>(ub_I_fp32.GetPhyAddr());
                     __ubuf__ float* dstAddr = reinterpret_cast<__ubuf__ float*>(ub_I_fp32.GetPhyAddr());
                     __ubuf__ uint32_t* idxAddr = reinterpret_cast<__ubuf__ uint32_t*>(ub_idx_b32.GetPhyAddr());
                     MulReduceScatterVF(dstAddr, src0Addr, src1Addr, idxAddr, 4, 64);
                     // for(uint32_t iterIdx = 1; iterIdx < 16; iterIdx++){
                     //     MulReduceScatterVF(dstAddr, src0Addr, src1Addr, idxAddr, 4, 64, iterIdx);
                     // }
 
                     
                     AscendC::DumpTensor(ub_I_fp32, 0, 16 * 64);
                     AscendC::PipeBarrier<PIPE_ALL>();
                     AscendC::Cast(ub_I, ub_I_fp32, AscendC::RoundMode::CAST_RINT, 1024);
                     AscendC::DumpTensor(ub_I, 1, 16 * 64);
                     AscendC::PipeBarrier<PIPE_ALL>();
                     TransposeB16(ub_res, ub_I, 64);
 
                     // asc_time_stamp(1);
                     // for(uint32_t iterIdx = 1; iterIdx < 16; iterIdx++){
                     //     MulReduceScatterVF(dstAddr, src0Addr, src1Addr, idxAddr, 64, 4, 64, iterIdx);
                     // }
                     
                     AscendC::DumpTensor(ub_res, 2, 16 * 64);
                     
                 }
             }
         }
 
         if ASCEND_IS_AIC {
 
         }
     }
 
 private:
     // Gm
     AscendC::GlobalTensor<InDtype> gm_a;
     AscendC::GlobalTensor<int64_t> gm_cu_seqlens;
     AscendC::GlobalTensor<int64_t> gm_chunk_indices;
     AscendC::GlobalTensor<OutDtype> gm_out;
 
     // UB
     AscendC::LocalTensor<InDtype> ub_A;
     AscendC::LocalTensor<float> ub_A_fp32;
     AscendC::LocalTensor<InDtype> ub_I;
     AscendC::LocalTensor<float> ub_I_fp32;
     AscendC::LocalTensor<InDtype> ub_res;
     AscendC::LocalTensor<uint32_t> ub_idx_b32;
     
     // Tiling
     int64_t batch_size;
     int64_t seq_length;
     int64_t num_head;
     int64_t chunk_size;
     int64_t chunk_num_in_seq;
     int64_t chunk_num_total;
     int64_t mode;
     int64_t is_lower;
     int64_t tiles_per_core;
     int64_t total_tokens;   // NTD: total_T，用于 head 维在外的偏移计算
 
     // Core
     int64_t num_core;
     int64_t core_idx;
     int64_t sub_block_idx;
 
     // 辅助矩阵当前缓存对应的 chunk 尺寸（仅尺寸变化时重建 I/Zero/l1_I）
     int64_t last_chunk_size;
 };
 
 
 #endif  // SOLVE_TRI_ASCEND950_H
 