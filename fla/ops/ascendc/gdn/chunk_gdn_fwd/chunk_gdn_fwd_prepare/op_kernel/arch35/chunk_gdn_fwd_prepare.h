/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * Fused GDN prepare (arch35). Helpers: mem.h, common.h, offset.h,
 * vf.h, copy.h, matmul.h. Reference VF: l2norm_regbase_vf.h
 * and MulReduceScatterVF32 in vf.h (hoist + inner unroll 2).
 * Memory map: design/chunk_gdn_fwd_prepare_ascendc-design.md.
 * Init binds every tile with OnChipBuffer::GetBuffer.
 *
 * kkt is Cube k' @ k'^T from L1 NZ k' (Stage1 (64,1,7,0) upload).
 * Cube MBH for strictly-lower L (solve_tri is_lower, LeafLeft driving):
 *   Y = I + LeafLeft @ (-L)
 *   A = LeafLeft + Y @ LeafRight
 *     tmp = I @ LeafLeft            (init_flag=True)
 *     A   = Y @ LeafRight + tmp     (init_flag=False)
 * LeafRight = blkdiag((I+L00)^{-1}, 0), LeafLeft = blkdiag(0, (I+L11)^{-1}).
 * Leaves / -L / I go L1 as 8-col ND->NZ (no UB 5HD). Leaf LoadData flips
 * ifTranspose vs I/-L/Y (w_mmad_demo).
 * A = (I+L)^{-1} = [XR, 0; -XL L10 XR, XL].
 * Stage4 Fixpipe uses Arch3510 isChannelSplit (16x16 -> 16x8) into this
 * tile's u_out slot, then MTE2 back to L1[0, 64). Stage5 Fixpipe L0C ->
 * gmA bf16 ND, then AIC MTE2 GM ND -> L1[256, 288) cube NZ. GET_TILING_DATA
 * / workspace GM are unbound (too many GM_ADDR).
 *
 * Scheduling: total work = Ceil(T/BT)*B*Hv. Each AIC + two AIV process up
 * to 4 tiles per pack (last pack may be shorter). AIV0: task 0,2; AIV1:
 * task 1,3. Per pack each AIV finishes Stage1 ping then pong (Set taskIdx
 * after k' L1). AIC Wait that flag, kkt, then Set the same id (PIPE_FIX)
 * so the matching AIV can Stage3. After Stage3 L1 writes, AIV Set
 * taskIdx+4; AIC Wait that (4, 21, 6, 23) and Stage4, same 0,1,2,3 order
 * and even/odd L0 banks as Stage2. Stage4 is pure AIC: the pack's 4 Stage4
 * tasks finish before any Stage5 task starts (same 0,1,2,3 / even-odd L0).
 * Stage6 is AIV after Stage3 (same 0,2 / 1,3 ping-pong). It reads k'/v from
 * GM and resident g'/β, so it does not wait Stage4/5. vb/kbg stay bf16 ND,
 * then DataCopy (64,1,srcGap,0) into L1 NZ with C0=16 (not fp32 C0=8).
 * After the pack's Stage5 and Stage6, AIC PipeBarrier ALL then Stage7
 * (serial: PIPE_ALL around each tile). W = A @ kbg then u = A @ vb, one
 * MMAD each [64,64]@[64,128]. L0A/L0B ping [32,48) pong [48,64), SetSize
 * 16 KiB. L0C even [0,64) odd [64,128). LoadData A 64x64, B 64x128. One
 * Fixpipe n=128 dstStride=K/V. Even/odd FIX_M, never both.

 */

#ifndef CHUNK_GDN_FWD_PREPARE_H
#define CHUNK_GDN_FWD_PREPARE_H

#include "kernel_operator.h"
#include "mem.h"
#include "common.h"
#include "offset.h"
#include "vf.h"
#include "copy.h"
#include "matmul.h"
#include "l2norm_regbase_vf.h"

namespace GdnStage {
using namespace AscendC;
using namespace AscendC::MicroAPI;

template <typename GateDtype, typename ALogDtype>
class ChunkGdnFwdPrepareKernel : public PrepareState {
public:
    using InDtype = bfloat16_t;

    template <typename Td>
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
                                GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR cu, GM_ADDR idx,
                                GM_ADDR gOut, GM_ADDR wOut, GM_ADDR uOut, GM_ADDR aOut,
                                GM_ADDR qHat, GM_ADDR kHat, GM_ADDR qRstd, GM_ADDR kRstd,
                                GM_ADDR betaEff, GM_ADDR workspace, const Td &td)
    {
        LoadTiling(td);
        InitOnChip();
        // Stage4 Y NZ scratch: this tile's u_out bytes (16 KiB). Per-tile,
        // Stage7 overwrites u after Y is on L1. Workspace GM is unbound.
        gmWsY.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(uOut));
        (void)workspace;
        gmQ.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(q));
        gmK.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(k));
        gmV.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(v));
        gmG.SetGlobalBuffer(reinterpret_cast<__gm__ GateDtype *>(g));
        gmBeta.SetGlobalBuffer(reinterpret_cast<__gm__ GateDtype *>(beta));
        if (aLog != nullptr) {
            gmALog.SetGlobalBuffer(reinterpret_cast<__gm__ ALogDtype *>(aLog));
        }
        if (dtBias != nullptr) {
            gmDt.SetGlobalBuffer(reinterpret_cast<__gm__ ALogDtype *>(dtBias));
        }
        if (cu != nullptr) {
            gmCu.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cu));
        }
        if (idx != nullptr) {
            gmIdx.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(idx));
        }
        gmGOut.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gOut));
        gmW.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(wOut));
        // Same bytes as gmW [B,HV,T,128] bf16 == [B,HV,T,64] fp32. Temporary
        // Stage3 dump of ND -L until Stage7 owns w. Host audit uses L = -dump.
        gmLDump.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(wOut));
        gmU.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(uOut));
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(aOut));
        gmQHat.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(qHat));
        gmKHat.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(kHat));
        gmQRstd.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(qRstd));
        gmKRstd.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(kRstd));
        hasBetaOut = 0;
        if (betaEff != nullptr) {
            gmBetaEff.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(betaEff));
            hasBetaOut = 1;
        }
    }

    // Explicit on-chip map. Offsets match design Stage 0-3; GetBuffer style
    // matches solve_tri_ascend950_64.h Init.
    __aicore__ inline void InitOnChip()
    {
        OnChipBuffer buf;

        // =====================================================================
        // UB map (KiB). Offsets are from offset.h. Live ranges overlap
        // by design: S1 input [10, 75) is released after k' is on L1, then
        // S3 reuses [10, 82). Bit-mask at [0, 0.5) stays until kernel exit.
        //
        //   [0.00,  0.50)  ubMaskBits     512 B  uint8  64x64 strict-lower bits
        //   [0.50,  1.00)  ubVcsIdx      512 B  u32    VCS scatter idx {0,32}
        //   [1.00,  9.00)  ubIVcs        8 KiB  fp32   I_vcs = [I32 | I32]
        //   [9.00,  9.25)  ubGPrime[0]  256 B  fp32   g' ping, [64]
        //   [9.25,  9.50)  ubBetaEff[0] 256 B  fp32   beta_eff ping, [64]
        //   [9.50,  9.75)  ubGPrime[1]  256 B  fp32   g' pong
        //   [9.75, 10.00)  ubBetaEff[1] 256 B  fp32   beta_eff pong
        //   [10.0, 26.00)  ubQ[0]      16 KiB  bf16   S1 q ping
        //   [26.0, 42.00)  ubK[0]      16 KiB  bf16   S1 k ping
        //   [42.0, 58.00)  ubQ[1]      16 KiB  bf16   S1 q pong
        //   [58.0, 74.00)  ubK[1]      16 KiB  bf16   S1 k pong
        //   [74.0, 74.50)  rstd pong    512 B  fp32   k/q rstd of 2nd task
        //   [75.0, 91.00)  ubQHat[0]   16 KiB  bf16   S1 q' ping
        //   [91.0, 107.0)  ubKHat[0]   16 KiB  bf16   S1 k' ping
        //   [107,  107.25) ubKRstd[0]  256 B  fp32   K rstd ping
        //   [107.25,107.50) ubQRstd[0] 256 B  fp32   Q rstd ping
        //   [107.50,107.75) ubGateTmp  256 B  fp32   fused-gate scratch [64]
        //   [108,  124.0)  ubQHat[1]   16 KiB  bf16   S1 q' pong
        //   [124,  140.0)  ubKHat[1]   16 KiB  bf16   S1 k' pong
        //   [140,  156.0)  ubKkt[0]    16 KiB  fp32   Cube kkt ping ND (Fixpipe NZ2ND)
        //   [156,  172.0)  ubKkt[1]    16 KiB  fp32   Cube kkt pong ND
        //   [172,  188.0)  ubMaskFp32  16 KiB  fp32   64x64 mask for VF
        // S0 only, before g' is stored:
        //   [9, 25) unused (was NZ I)   [25, 41) ubS0Zero
        // S3 (after S1 q/k released):
        //   [10, 18) ubLPacked  [18, 34) ubResVcs   [34, 50) ubLFull
        //   [50, 66) ubS3Nz16 [66, 82) ubS3Nz8    [82, ~106) ubS3LeafTmp
        // S6 (after S3; overlaps S1/S3, g'/β at [9,10) stay):
        //   [32, 48) ubS6K[0] 16 KiB k' ping   [48, 64) ubS6K[1] k' pong
        //   [64, 80) ubS6V[0] 16 KiB v ping    [80, 96) ubS6V[1] v pong
        //   vb/kbg ND → L1 NZ C0=16 via DataCopy (64,1,srcGap,0)
        // =====================================================================

        // UB[0.00, 0.50) KiB = 512 B, uint8.
        // Strict-lower triangular bit-mask of the 64x64 chunk.
        // Packed as 64*64/8 bytes: row i, bit j is 1 iff token-row i > col j.
        // VF cannot consume bits; Stage0 expands this once into ubMaskFp32.
        ubMaskBits = buf.template GetBuffer<BufferType::ASCEND_UB, uint8_t>(kUbMaskBits);

        // UB[0.50, 1.00) KiB = 512 B, uint32. Only the first 8 values are used.
        // VCS scatter index for packing two 32x32 leaves into I_vcs 32x64:
        // values are {0, 32} (left leaf at col 0, right leaf at col 32).
        ubVcsIdx = buf.template GetBuffer<BufferType::ASCEND_UB, uint32_t>(kUbVcsIdx);

        // UB[1.00, 9.00) KiB = 8 KiB, fp32 ND [32, 64].
        // I_vcs = concat along K of two I_32 identity leaves. Stage3 VCS
        // copies this to ubResVcs and overwrites with (I+Lii)^{-1}.
        ubIVcs = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbIVcs);

        // UB[9.00, 25.00) KiB unused (old Cube NZ I). Overlaps g'/beta.

        // UB[25.00, 41.00) KiB = 16 KiB, fp32.
        // Stage0 ND I then zeros for leaf L1. Released before S1.
        ubS0Zero = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbS0Zero);

        // UB[9.00, 10.00) resident g'/beta, then S1 q/k/g/beta and S2 kkt ping/pong.
        // db=0: AIV0 task 0 / AIV1 task 1; db=1: AIV0 task 2 / AIV1 task 3.
        for (int32_t db = 0; db < 2; ++db) {
            ubGPrime[db] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbGPrime[db]);
            ubBetaEff[db] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbBetaEff[db]);
            ubQ[db] = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(kUbS1Q[db]);
            ubK[db] = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(kUbS1K[db]);
            ubG[db] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbS1G[db]);
            ubBeta[db] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbS1Beta[db]);
            ubKkt[db] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbS2Kkt[db]);
            ubS6K[db] = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(kUbS6K[db]);
            ubS6V[db] = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(kUbS6V[db]);
            ubQHat[db] = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(kUbS1QHat[db]);
            ubKHat[db] = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(kUbS1KHat[db]);
            ubKRstd[db] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbS1KRstd[db]);
            ubQRstd[db] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbS1QRstd[db]);
        }

        // UB[172.0, 188.0) KiB = 16 KiB, fp32 ND [64, 64].
        // Expanded 0/1 mask from ubMaskBits. ConstructLowerLExp2VF reads this
        // to build L = tril(exp2(g_i-g_j))*beta*kkt. Lives past S1/S2.
        ubMaskFp32 = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbMaskFp32);

        // UB[10.00, 18.00) KiB = 8 KiB, fp32 ND [32, 64]. Overlaps ubQ[0].
        // Packed diagonal leaves -L00|-L11 for VCS. Valid only after S1 q is done.
        ubLPacked = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbS3LPacked);

        // UB[18.00, 34.00) KiB = 16 KiB, fp32 ND [32, 64]. Overlaps ubKPing start.
        // VCS working copy of I_vcs, then (I+Lii)^{-1} for both 32x32 leaves. Overlaps ubK[0].
        ubResVcs = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbS3ResVcs);

        // UB[34.00, 50.00) KiB = 16 KiB, fp32 ND [64, 64].
        // -L = -tril(exp2(Δg))*β*kkt from ConstructLowerLExp2VF. Pack and L1 both read this.
        ubLFull = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbS3LFull);

        // UB[50.00, 66.00) KiB = 16 KiB, fp32 64x64.
        // -L ND->NZ with C0=16. Stage5 also uses this as A NZ->ND destination.
        ubS3Nz16 = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbS3Nz16);

        // UB[66.00, 82.00) KiB = 16 KiB, fp32 64x64.
        // Cube NZ C0=8 of -L, or a zeroed 64x64 before scattering leaf inverses.
        ubS3Nz8 = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbS3Nz8);

        // UB[82.00, ~106) KiB, fp32.
        // TransposeB32 workspace plus two 32x32 NZ leaves before L1 scatter
        // into LeafRight / LeafLeft.
        ubS3LeafTmp = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbS3LeafTmp);

        // HardEvent ids (do not reuse a live id without Wait):
        //   0  S1 Q/K GM copy
        //   1  S1 gate/beta GM copy
        //   2  S1 L2Norm store / gate scalar aLog
        //   3  S1 g' / beta_eff GM store
        //   4  S3 dump L / leaf
        //   5  S1 k' ND -> L1 NZ / S3 I_nz / S6 vb,kbg UB -> L1
        //   6  S3 dump PRINTF (V_S)
        //   7  S3 -L / leaves UB -> L1
        // Stage4/5 reuse bank 0/1 for FIX_M / M_FIX / FIX_MTE2 / M_MTE1.
        // Stage6 reuses 0 for GM k'/v copy after S3 Wait.

        // UB[107.50, 107.75) KiB. Fused-gate tmp, [64]. Ping rstd is at 107.
        ubGateTmp = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbS1GateTmp);

        // L1 512 KiB. Four task slots (taskIdx 0..3).
        for (uint32_t t = 0; t < static_cast<uint32_t>(kTasksPerRound); ++t) {
            l1KHat[t] = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(L1KHat(t));
            l1NegL[t] = buf.template GetBuffer<BufferType::ASCEND_CB, float>(L1NegL(t));
            l1LeafRight[t] = buf.template GetBuffer<BufferType::ASCEND_CB, float>(L1LeafRight(t));
            l1LeafLeft[t] = buf.template GetBuffer<BufferType::ASCEND_CB, float>(L1LeafLeft(t));
            l1A[t] = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(L1ResidentA(t));
            l1Kbg[t] = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(L1ResidentKbg(t));
            l1Kbg1[t] = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(
                L1ResidentKbg(t) + kSlotBf16_64);
            l1Vb[t] = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(L1ResidentVb(t));
            l1Vb1[t] = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(
                L1ResidentVb(t) + kSlotBf16_64);
            // Y aliases k' at L1[0,64). Stage4 overwrites after Stage2.
            l1Y[t] = buf.template GetBuffer<BufferType::ASCEND_CB, float>(L1Y(t));
        }

        // L1[480, 496) KiB = 16 KiB, fp32 cube-NZ I_64, C0=8.
        l1I = buf.template GetBuffer<BufferType::ASCEND_CB, float>(kL1ResidentI);

        // ---- L0A/L0B 64 KiB each, L0C 256 KiB. bf16 and fp32 views alias banks. ----

        // L0A/L0B [0, 16) KiB, bf16. Stage2 AIV0 tasks 0 and 2.
        l0A = buf.template GetBuffer<BufferType::ASCEND_L0A, InDtype>(0);
        l0B = buf.template GetBuffer<BufferType::ASCEND_L0B, InDtype>(0);

        // L0A/L0B [16, 32) KiB, bf16. Stage2 AIV1 tasks 1 and 3.
        l0A1 = buf.template GetBuffer<BufferType::ASCEND_L0A, InDtype>(kL0Bf16Pair);
        l0B1 = buf.template GetBuffer<BufferType::ASCEND_L0B, InDtype>(kL0Bf16Pair);

        // L0A/L0B [0, 16) KiB, fp32 view of the same banks as l0A/l0B.
        // Stage4/5 MBH: I, -L, LeafRight, LeafLeft.
        l0Af = buf.template GetBuffer<BufferType::ASCEND_L0A, float>(0);
        l0Bf = buf.template GetBuffer<BufferType::ASCEND_L0B, float>(0);

        // L0A/L0B [16, 32) KiB, fp32. Second fp32 pair for overlapped MBH.
        l0Af1 = buf.template GetBuffer<BufferType::ASCEND_L0A, float>(kL0Fp32Pair);
        l0Bf1 = buf.template GetBuffer<BufferType::ASCEND_L0B, float>(kL0Fp32Pair);

        // L0C [0, 64) KiB, fp32. Stage2 kkt / Stage4 Y / Stage5 A (even).
        l0C = buf.template GetBuffer<BufferType::ASCEND_L0C, float>(0);

        // L0C [64, 128) KiB, fp32. Stage2/4/5 odd tasks.
        l0C1 = buf.template GetBuffer<BufferType::ASCEND_L0C, float>(kL0C1);

        // Stage7 L0A/L0B: [32, 48) even, [48, 64) odd. SetSize(16 KiB) so
        // ping does not keep the remaining 32 KiB and overlap pong.
        l0AS7 = buf.template GetBuffer<BufferType::ASCEND_L0A, InDtype>(kL0S7Ping);
        l0BS7 = buf.template GetBuffer<BufferType::ASCEND_L0B, InDtype>(kL0S7Ping);
        l0AS71 = buf.template GetBuffer<BufferType::ASCEND_L0A, InDtype>(kL0S7Pong);
        l0BS71 = buf.template GetBuffer<BufferType::ASCEND_L0B, InDtype>(kL0S7Pong);
        l0AS7.SetSize(kL0S7Slot / sizeof(InDtype));
        l0BS7.SetSize(kL0S7Slot / sizeof(InDtype));
        l0AS71.SetSize(kL0S7Slot / sizeof(InDtype));
        l0BS71.SetSize(kL0S7Slot / sizeof(InDtype));
        // L0C 64x128 fp32 NZ is 32 KiB: even [0, 64), odd [64, 128).
        l0CS7 = buf.template GetBuffer<BufferType::ASCEND_L0C, float>(0);
        l0CS71 = buf.template GetBuffer<BufferType::ASCEND_L0C, float>(kL0C1);

        subBlock = AscendC::GetSubBlockIdx();
        if ASCEND_IS_AIV {
            // MIX 1:2: GetBlockIdx is 0,1 for cube 0's two Vectors. GetBlockNum
            // on AIV already equals the cube count (same as AIC, 28 here). The
            // runtime PRINTF "Block 0/56" is 2*cubes; do not divide again.
            coreIdx = AscendC::GetBlockIdx() / 2;
            numCore = AscendC::GetBlockNum();
        } else {
            coreIdx = AscendC::GetBlockIdx();
            numCore = AscendC::GetBlockNum();
        }
        auxReady = 0;
    }

    // AIV0: task 0 ping, 2 pong. AIV1: task 1 ping, 3 pong.
    __aicore__ inline int32_t PingPongSlot(int64_t taskIdx) const
    {
        return static_cast<int32_t>((taskIdx >> 1) & 1);
    }

    // ========================= Stage 0 =========================
    __aicore__ inline void Stage0_GenerateResidentAux()
    {
        if (auxReady != 0) {
            return;
        }
        Prepare::Stage0_GenIdentity(ubIVcs, ubVcsIdx);
        Prepare::Stage0_ExpandBitMaskToFp32(ubMaskFp32);
        // Leaf zeros first so both AIVs copy in parallel. ubS0Zero is then
        // free for AIV0 to paint I. AIV1 can leave for Stage1 while AIV0
        // uploads I. L1 leaf slots are disjoint by taskIdx (AIV0: 0,2;
        // AIV1: 1,3). Scatter only rewrites the 32x32 quadrant; TR/BL stay
        // zero if primed once.
        Duplicate(ubS0Zero, 0.0f, static_cast<int32_t>(kChunk64 * kChunk64));
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        for (uint32_t t = static_cast<uint32_t>(subBlock);
             t < static_cast<uint32_t>(kTasksPerRound); t += 2) {
            UbToL1Fp32(l1LeafLeft[t], ubS0Zero, kChunk64);
            UbToL1Fp32(l1LeafRight[t], ubS0Zero, kChunk64);
        }
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
        // Cube I_64 is shared L1: only AIV0 paints ND I and 8-col uploads.
        if (subBlock == 0) {
            Prepare::Stage0_PaintIdentity64(ubS0Zero);
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            UbNd64ToL1Nz8(l1I, ubS0Zero);
            SetFlag<HardEvent::MTE3_V>(0);
            WaitFlag<HardEvent::MTE3_V>(0);
        }
        auxReady = 1;
    }

    // ========================= Stage 1 =========================
    // K then Q: each is GM copy, L2Norm, hat/rstd store. K also ND→L1 NZ
    // (64,1,7,0). NotifyAic after k' is on L1 so Cube kkt overlaps Q/gate.
    // Hats/rstd are ping-pong (db), so Q L2Norm can overlap K rstd MTE3
    // and the next task can overlap this task's hat MTE3.
    __aicore__ inline void Stage1_OneTask(const ChunkRange &chunk, int64_t hv, int64_t taskIdx)
    {
        const int64_t hk = hv / HRatio;
        const int32_t db = PingPongSlot(taskIdx);
        const int32_t nElem = static_cast<int32_t>(chunk.M * K);
        const int32_t nValid = static_cast<int32_t>(chunk.M);
        const int32_t nPad = static_cast<int32_t>(chunkSize);
        const int64_t offQk = OffsetBHTD(chunk.batch, hk, chunk.tokenStart, HK, T, K);
        const int64_t offG = OffsetBHT(chunk.batch, hv, chunk.tokenStart, HV, T);
        const int64_t offRstd = OffsetBHT(chunk.batch, hk, chunk.tokenStart, HK, T);
        LocalTensor<InDtype> l1K = l1KHat[static_cast<uint32_t>(taskIdx)];
        LocalTensor<float> ubGfp = ubGPrime[db];
        LocalTensor<float> ubBfp = ubBetaEff[db];
        LocalTensor<GateDtype> ubGRaw = ubGfp.template ReinterpretCast<GateDtype>();
        LocalTensor<GateDtype> ubBRaw = ubBfp.template ReinterpretCast<GateDtype>();

        // Ping only: previous pack Stage3 (or Stage0) overlaps ubK[0].
        if (db == 0) {
            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
        }

        
        Duplicate(ubGfp, 0.0f, nPad);
        Duplicate(ubBfp, 0.0f, nPad);
        SetFlag<HardEvent::V_MTE2>(1);
        
        // Process K
        DataCopy(ubK[db], gmK[offQk], nElem);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        GdnL2Norm::L2NormFwdK128VF<InDtype>(ubK[db], ubKHat[db], ubKRstd[db], static_cast<uint32_t>(nValid), kGdnL2NormEps);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        for (uint16_t frac = 0; frac < 8; ++frac) {
            DataCopy(l1K[static_cast<int32_t>(frac) * 16 * 64], ubKHat[db][static_cast<int32_t>(frac) * 16], DataCopyParams(64, 1, 7, 0));
        }
        CrossCoreSetFlag<0x4, PIPE_MTE3>(static_cast<uint16_t>(taskIdx));
        DataCopy(gmKHat[offQk], ubKHat[db], nElem);
        DataCopy(gmKRstd[offRstd], ubKRstd[db], nValid);

        // Process Q   
        DataCopy(ubQ[db], gmQ[offQk], nElem);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        GdnL2Norm::L2NormFwdK128VF<InDtype>(ubQ[db], ubQHat[db], ubQRstd[db], static_cast<uint32_t>(nValid), kGdnL2NormEps);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopy(gmQHat[offQk], ubQHat[db], nElem);
        DataCopy(gmQRstd[offRstd], ubQRstd[db], nValid);

        // g / beta share the resident 256 B slots. GateDtype view of the same
        // address: fp32 copy is identity; b16/fp16 fills the first 128 B, then
        // one VL LoadCast stores 64 fp32. n<=64, load-then-store, no ubQHat.
        WaitFlag<HardEvent::V_MTE2>(1);
        DataCopy(ubGRaw, gmG[offG], nValid);
        DataCopy(ubBRaw, gmBeta[offG], nValid);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);

        if (useGateInKernel != 0) {
            SetFlag<HardEvent::V_S>(0);
            WaitFlag<HardEvent::V_S>(2);
            float aLog = ScalarToFp32(gmALog.GetValue(hv));
            float dt = (hasDtBias != 0) ? ScalarToFp32(gmDt.GetValue(hv)) : 0.0f;
            SetFlag<HardEvent::S_V>(2);
            WaitFlag<HardEvent::S_V>(2);
            GateSoftplusVF<GateDtype>(ubGRaw, ubGfp, aLog, dt, static_cast<uint32_t>(nValid));
        } else if constexpr (!IsSameType<GateDtype, float>::value) {
            CopyToFp32VF<GateDtype>(ubGRaw, ubGfp, static_cast<uint32_t>(nValid));
        }
        float scale = (useExp2 != 0) ? kGdnRcpLn2 : 1.0f;
        ChunkCumsumScaleVF(ubGfp, scale, static_cast<uint32_t>(nValid));
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopy(gmGOut[offG], ubGfp, nValid);

        if (useBetaSigmoid != 0) {
            float bscale = (allowNegEigval != 0) ? 2.0f : 1.0f;
            BetaSigmoidVF<GateDtype>(ubBRaw, ubBfp, bscale, static_cast<uint32_t>(nValid));
        } else if constexpr (!IsSameType<GateDtype, float>::value) {
            CopyToFp32VF<GateDtype>(ubBRaw, ubBfp, static_cast<uint32_t>(nValid));
        }

        if (hasBetaOut != 0) {
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            DataCopy(gmBetaEff[offG], ubBfp, nValid);
        }
    }

    __aicore__ inline void UploadBf16NdToL1(LocalTensor<InDtype> l1, LocalTensor<InDtype> ubNd,
                                            LocalTensor<InDtype> ubNz, uint32_t cols)
    {
        (void)ubNz;
        UploadBf16ColsToL1(l1, ubNd, 0, cols, cols);
    }

    // Stage1 k' fractal: dest N-frac fc at fc*16*64, src C0 at colOff+fc*16,
    // src row gap = ndCols/16-1. nCols=64 is one WuMatmul panel.
    __aicore__ inline void UploadBf16ColsToL1(LocalTensor<InDtype> l1,
                                              LocalTensor<InDtype> ubNd,
                                              uint32_t colOff, uint32_t ndCols, uint32_t nCols)
    {
        const uint32_t nFrac = nCols / 16;
        const uint16_t srcGap = static_cast<uint16_t>(ndCols / 16 - 1);
        SetFlag<HardEvent::V_MTE3>(5);
        WaitFlag<HardEvent::V_MTE3>(5);
        for (uint32_t fc = 0; fc < nFrac; ++fc) {
            DataCopy(l1[static_cast<int32_t>(fc) * 16 * static_cast<int32_t>(kChunk64)],
                     ubNd[static_cast<int32_t>(colOff + fc * 16)],
                     DataCopyParams(kChunk64, 1, srcGap, 0));
        }
        SetFlag<HardEvent::MTE3_V>(5);
        WaitFlag<HardEvent::MTE3_V>(5);
    }

    // Independent s7: UB Nd2Nz then UbToL1. Dest must not be a Fixpipe UB
    // (that 507015'd on this MIX kernel). ubQ[0] at 10KB is 16 KiB, free
    // after Stage3, and is not a Fixpipe target (kkt lands on ubKkt).
    __aicore__ inline void CopyNdToL1ViaUbNd2Nz(LocalTensor<InDtype> l1,
                                                LocalTensor<InDtype> ubNd,
                                                uint32_t cols)
    {
        Nd2NzParams nd;
        nd.ndNum = 1;
        nd.nValue = static_cast<uint32_t>(chunkSize);
        nd.dValue = cols;
        nd.srcDValue = cols;
        nd.srcNdMatrixStride = 0;
        nd.dstNzC0Stride = static_cast<uint32_t>(chunkSize);
        nd.dstNzNStride = 1;
        nd.dstNzMatrixStride = 0;
        LocalTensor<InDtype> ubNz = ubQ[0];
        SetFlag<HardEvent::V_MTE3>(5);
        WaitFlag<HardEvent::V_MTE3>(5);
        DataCopy(ubNz, ubNd, nd);
        UbToL1Elems<InDtype>(l1, ubNz, static_cast<uint32_t>(chunkSize * cols));
        SetFlag<HardEvent::MTE3_V>(5);
        WaitFlag<HardEvent::MTE3_V>(5);
    }

    // Cube wait id for Vector taskIdx. Odd tasks are AIV1 → +16.
    __aicore__ inline uint16_t CubeWaitFlagForTask(int64_t taskIdx) const
    {
        uint16_t flag = static_cast<uint16_t>(taskIdx);
        if ((taskIdx & 1) != 0) {
            flag += kAiv1IntraFlagOff;
        }
        return flag;
    }

    // Same ids as Stage1 (0,1,2,3). AIC already Waited this id for k' ready,
    // then Sets it again after kkt Fixpipe so that AIV can Stage3.
    // AIV waits the raw id; AIC Sets id+16 for AIV1.
    __aicore__ inline void NotifyAivKktDone(int64_t taskIdx)
    {
        CrossCoreSetFlag<0x4, PIPE_FIX>(CubeWaitFlagForTask(taskIdx));
    }

    __aicore__ inline void WaitCubeKktDone(int64_t taskIdx)
    {
        CrossCoreWaitFlag<0x4>(static_cast<uint16_t>(taskIdx));
    }

    // Stage3 L1 ready → Stage4. AIV Sets raw taskIdx+4; AIC Waits +16 on AIV1.
    __aicore__ inline void NotifyAicStage3Done(int64_t taskIdx)
    {
        CrossCoreSetFlag<0x4, PIPE_MTE3>(static_cast<uint16_t>(taskIdx + kFlagS3DoneBase));
    }

    __aicore__ inline void WaitAivStage3Done(int64_t taskIdx)
    {
        CrossCoreWaitFlag<0x4, PIPE_MTE1>(CubeWaitFlagForTask(taskIdx + kFlagS3DoneBase));
    }

    __aicore__ inline void NotifyAivStage5Done(int64_t taskIdx)
    {
        CrossCoreSetFlag<0x4, PIPE_FIX>(CubeWaitFlagForTask(taskIdx + kFlagS4DumpBase));
    }

    __aicore__ inline void WaitAicStage5Done(int64_t taskIdx)
    {
        CrossCoreWaitFlag<0x4>(static_cast<uint16_t>(taskIdx + kFlagS4DumpBase));
    }

    // Stage6 L1 vb/kbg ready → Stage7. Reuse Stage3 ids 4..7 (consumed
    // after Stage4). AIV Sets raw taskIdx+4; AIC Waits +16 on AIV1.
    // Intra-block 12..15 is ignored on this SoC (Wait returns immediately).
    __aicore__ inline void NotifyAicStage6Done(int64_t taskIdx)
    {
        CrossCoreSetFlag<0x4, PIPE_MTE3>(static_cast<uint16_t>(taskIdx + kFlagS6DoneBase));
    }

    __aicore__ inline void WaitAivStage6Done(int64_t taskIdx)
    {
        CrossCoreWaitFlag<0x4, PIPE_MTE1>(CubeWaitFlagForTask(taskIdx + kFlagS6DoneBase));
    }

    // ========================= Stage 3 =========================
    // Packed -L00 | -L11 from ubLFull (VF already wrote -L). VCS consumes this as-is.
    __aicore__ inline void PackDiagLeavesFromUb(LocalTensor<float> packed, LocalTensor<float> L)
    {
        const uint16_t burst = 4;
        const uint16_t gap = 4;
        DataCopy(packed, L, DataCopyParams(32, burst, gap, gap));
        DataCopy(packed[32], L[32 * 64 + 32], DataCopyParams(32, burst, gap, gap));
    }

    // Packed VCS ND -> L1 NZ quadrants (Right=TL L00, Left=BR L11).
    // 8-col DataCopy, no UB TransDataTo5HD. Off-diagonal stays Stage0 zero.
    __aicore__ inline void UploadDiagLeavesToL1(LocalTensor<float> packedNd32x64, int64_t taskIdx)
    {
        UbPackedLeafToL1(l1LeafRight[taskIdx], packedNd32x64, 0, 0, 0);
        UbPackedLeafToL1(l1LeafLeft[taskIdx], packedNd32x64, 1, 1,
                         static_cast<int32_t>(kVcs32));
    }

    __aicore__ inline void Stage3_ConstructLAndVcs(int32_t localSlot, LocalTensor<float> kkt, int64_t taskIdx)
    {
        (void)taskIdx;
        LocalTensor<float> ubL = ubLFull;
        LocalTensor<float> g = ubGPrime[localSlot];
        LocalTensor<float> beta = ubBetaEff[localSlot];
        ConstructLowerLExp2VF(kkt, g, beta, ubMaskFp32, ubL, static_cast<uint32_t>(chunkSize));
        PackDiagLeavesFromUb(ubLPacked, ubL);
        DataCopy(ubResVcs, ubIVcs, static_cast<int32_t>(kVcsPackedElems32));
        MulReduceScatterVF32(ubResVcs, ubLPacked, ubResVcs, ubVcsIdx);
    }

    __aicore__ inline void Stage3_AivOne(const ChunkRange &chunk, int64_t hv, int64_t taskIdx)
    {
        WaitCubeKktDone(taskIdx);
        const int32_t db = PingPongSlot(taskIdx);
        Stage3_ConstructLAndVcs(db, ubKkt[db], taskIdx);
        // Host audit until Stage7 owns w: ND -L on w. u is Stage4 Y ND.
        SetFlag<HardEvent::V_MTE3>(4);
        WaitFlag<HardEvent::V_MTE3>(4);
        const int64_t offL = OffsetBHTD(chunk.batch, hv, chunk.tokenStart, HV, T, 64);
        DataCopy(gmLDump[offL], ubLFull, static_cast<int32_t>(chunkSize * chunkSize));
        SetFlag<HardEvent::MTE3_V>(4);
        WaitFlag<HardEvent::MTE3_V>(4);
        SetFlag<HardEvent::V_MTE3>(7);
        WaitFlag<HardEvent::V_MTE3>(7);
        UploadDiagLeavesToL1(ubResVcs, taskIdx);
        UbNd64ToL1Nz8(l1NegL[taskIdx], ubLFull);
        SetFlag<HardEvent::MTE3_V>(7);
        WaitFlag<HardEvent::MTE3_V>(7);
        NotifyAicStage3Done(taskIdx);
    }


    // ========================= Stage 2 =========================
    // kkt = k' @ k'^T. AIV0 tasks 0/2 use l0 (event 0); AIV1 tasks 1/3 use l01 (event 1).
    // Wait FIX_M(bank) before reuse of that L0 (0 vs 2, 1 vs 3, and across packs).
    // First Wait on each bank is primed by SetFlag FIX_M in ProcessAic, so
    // task 0 does not wait task 1 and vice versa. After Fixpipe only Set
    // FIX_M, no Wait: Notify is
    // PIPE_FIX and the other bank can Matmul while this Fixpipe runs.
    // ubKkt ping/pong is PingPongSlot. subBlockId: 0=AIV0, 1=AIV1.
    __aicore__ inline void Stage2_AicOne(int64_t taskIdx)
    {
        const int32_t bt = static_cast<int32_t>(chunkSize);
        const int32_t kk = static_cast<int32_t>(K);
        const uint32_t n = static_cast<uint32_t>(chunkSize);
        const uint8_t subBlk = static_cast<uint8_t>(taskIdx & 1);
        const int32_t db = PingPongSlot(taskIdx);
        if ((taskIdx & 1) == 0) {
            WaitFlag<HardEvent::FIX_M>(0);
            MatmulToL0C<InDtype>(l1KHat[taskIdx], l1KHat[taskIdx], l0A, l0B, l0C,
                                 bt, bt, kk, true, false);
            SetFlag<HardEvent::M_FIX>(0);
            WaitFlag<HardEvent::M_FIX>(0);
            FixpipeL0cToUbFp32Nd(ubKkt[db], l0C, n, subBlk);
            SetFlag<HardEvent::FIX_M>(0);
        } else {
            WaitFlag<HardEvent::FIX_M>(1);
            MatmulToL0C<InDtype>(l1KHat[taskIdx], l1KHat[taskIdx], l0A1, l0B1, l0C1,
                                 bt, bt, kk, true, false);
            SetFlag<HardEvent::M_FIX>(1);
            WaitFlag<HardEvent::M_FIX>(1);
            FixpipeL0cToUbFp32Nd(ubKkt[db], l0C1, n, subBlk);
            SetFlag<HardEvent::FIX_M>(1);
        }
        NotifyAivKktDone(taskIdx);
    }

    // ========================= Stage 4 =========================
    // Y = I + LeafLeft @ (-L)  (driving = LeafLeft / X_L).
    //   B: L0C = I @ I                init_flag=True
    //   C: L0C = LeafLeft @ (-L) + I  init_flag=False
    // Fixpipe Arch3510 isChannelSplit (16x16 -> 16x8) into this tile's u
    // GM (16 KiB), then MTE2 -> L1 Y. Audit: second Fixpipe ND overwrites
    // that NZ scratch so host can read Y. Stage7 overwrites u. Even/odd L0
    // as Stage2.
    __aicore__ inline void Stage4_AicOne(const ChunkRange &chunk, int64_t hv, int64_t taskIdx)
    {
        const int32_t bt = static_cast<int32_t>(chunkSize);
        const uint8_t bank = static_cast<uint8_t>(taskIdx & 1);
        const int64_t offU = OffsetBHTD(chunk.batch, hv, chunk.tokenStart, HV, T, V);
        const uint32_t wsOff = static_cast<uint32_t>(offU / (sizeof(float) / sizeof(InDtype)));
        WaitFlag<HardEvent::FIX_M>(bank);
        if (bank == 0) {
            MatmulToL0C<float>(l1I, l1I, l0Af, l0Bf, l0C, bt, bt, bt, true, true);
            SetFlag<HardEvent::M_MTE1>(0);
            WaitFlag<HardEvent::M_MTE1>(0);
            MatmulToL0C<float>(l1LeafLeft[taskIdx], l1NegL[taskIdx], l0Af, l0Bf, l0C,
                               bt, bt, bt, false, true, true);
            SetFlag<HardEvent::M_FIX>(0);
            WaitFlag<HardEvent::M_FIX>(0);
            FixpipeL0cToGmNzCs(gmWsY[wsOff], l0C, kChunk64);
            SetFlag<HardEvent::FIX_MTE2>(0);
            WaitFlag<HardEvent::FIX_MTE2>(0);
            CopyGmNzToL1Fp32(l1Y[taskIdx], gmWsY[wsOff], kChunk64);
            SetFlag<HardEvent::MTE2_MTE1>(0);
            WaitFlag<HardEvent::MTE2_MTE1>(0);
            FixpipeL0cToGmNd<float>(gmWsY[wsOff], l0C, kChunk64, kChunk64, kChunk64);
            SetFlag<HardEvent::FIX_M>(0);
        } else {
            MatmulToL0C<float>(l1I, l1I, l0Af1, l0Bf1, l0C1, bt, bt, bt, true, true);
            SetFlag<HardEvent::M_MTE1>(1);
            WaitFlag<HardEvent::M_MTE1>(1);
            MatmulToL0C<float>(l1LeafLeft[taskIdx], l1NegL[taskIdx], l0Af1, l0Bf1, l0C1,
                               bt, bt, bt, false, true, true);
            SetFlag<HardEvent::M_FIX>(1);
            WaitFlag<HardEvent::M_FIX>(1);
            FixpipeL0cToGmNzCs(gmWsY[wsOff], l0C1, kChunk64);
            SetFlag<HardEvent::FIX_MTE2>(1);
            WaitFlag<HardEvent::FIX_MTE2>(1);
            CopyGmNzToL1Fp32(l1Y[taskIdx], gmWsY[wsOff], kChunk64);
            SetFlag<HardEvent::MTE2_MTE1>(1);
            WaitFlag<HardEvent::MTE2_MTE1>(1);
            FixpipeL0cToGmNd<float>(gmWsY[wsOff], l0C1, kChunk64, kChunk64, kChunk64);
            SetFlag<HardEvent::FIX_M>(1);
        }
    }

    // ========================= Stage 5 =========================
    // A = LeafLeft + Y @ LeafRight  (other = LeafRight / X_R).
    //   tmp = I @ LeafLeft               init_flag=True
    //   A   = Y @ LeafRight + tmp        init_flag=False
    // L0C -> gmA bf16 ND, then GM ND -> L1 A cube NZ (DataCopy Nd2NzParams
    // on AIC, TPosition A1). Wait FIX_M after the Fixpipe so MTE2 does not
    // race the GM write (FIX_MTE2 after ROW_MAJOR GM Fixpipe hung). Live
    // tiles always have M in 1..BT. Even/odd L0 banks match Stage2/4.
    __aicore__ inline void Stage5_AicOne(const ChunkRange &chunk, int64_t hv, int64_t taskIdx)
    {
        const int32_t bt = static_cast<int32_t>(chunkSize);
        const uint8_t bank = static_cast<uint8_t>(taskIdx & 1);
        const uint32_t n = static_cast<uint32_t>(chunkSize);
        const uint32_t m = static_cast<uint32_t>(chunk.M);
        const int64_t offA = OffsetBHTD(chunk.batch, hv, chunk.tokenStart, HV, T, chunkSize);
        WaitFlag<HardEvent::FIX_M>(bank);
        if (bank == 0) {
            MatmulToL0C<float>(l1I, l1LeafLeft[taskIdx], l0Af, l0Bf, l0C, bt, bt, bt, true, false);
            SetFlag<HardEvent::M_MTE1>(0);
            WaitFlag<HardEvent::M_MTE1>(0);
            MatmulToL0C<float>(l1Y[taskIdx], l1LeafRight[taskIdx], l0Af, l0Bf, l0C,
                               bt, bt, bt, false, false);
            SetFlag<HardEvent::M_FIX>(0);
            WaitFlag<HardEvent::M_FIX>(0);
            FixpipeL0cToGmNd<InDtype>(gmA[offA], l0C, m, n, n);
            SetFlag<HardEvent::FIX_M>(0);
            WaitFlag<HardEvent::FIX_M>(0);
            CopyGmNdToL1Nz<InDtype>(l1A[taskIdx], gmA[offA], n, n);
            SetFlag<HardEvent::MTE2_MTE1>(0);
            WaitFlag<HardEvent::MTE2_MTE1>(0);
            SetFlag<HardEvent::FIX_M>(0);
        } else {
            MatmulToL0C<float>(l1I, l1LeafLeft[taskIdx], l0Af1, l0Bf1, l0C1, bt, bt, bt, true, false);
            SetFlag<HardEvent::M_MTE1>(1);
            WaitFlag<HardEvent::M_MTE1>(1);
            MatmulToL0C<float>(l1Y[taskIdx], l1LeafRight[taskIdx], l0Af1, l0Bf1, l0C1,
                               bt, bt, bt, false, false);
            SetFlag<HardEvent::M_FIX>(1);
            WaitFlag<HardEvent::M_FIX>(1);
            FixpipeL0cToGmNd<InDtype>(gmA[offA], l0C1, m, n, n);
            SetFlag<HardEvent::FIX_M>(1);
            WaitFlag<HardEvent::FIX_M>(1);
            CopyGmNdToL1Nz<InDtype>(l1A[taskIdx], gmA[offA], n, n);
            SetFlag<HardEvent::MTE2_MTE1>(1);
            WaitFlag<HardEvent::MTE2_MTE1>(1);
            SetFlag<HardEvent::FIX_M>(1);
        }
        NotifyAivStage5Done(taskIdx);
    }

    // ========================= Stage 6 =========================
    // vb = v * β, kbg = k' * β * exp2(g'). BF16 ND in UB, then L1 NZ C0=16
    // with DataCopy (64, 1, srcGap, 0) — same fractal as Stage1 k'.
    // AIV0 tasks 0,2; AIV1 tasks 1,3. k' from GM. g'/β from Stage1 resident.
    __aicore__ inline void Stage6_AivOne(const ChunkRange &chunk, int64_t hv, int64_t taskIdx)
    {
        const int64_t hk = hv / HRatio;
        const int64_t offK = OffsetBHTD(chunk.batch, hk, chunk.tokenStart, HK, T, K);
        const int64_t offV = OffsetBHTD(chunk.batch, hv, chunk.tokenStart, HV, T, V);
        const int32_t nK = static_cast<int32_t>(chunkSize * K);
        const int32_t nV = static_cast<int32_t>(chunkSize * V);
        const int32_t db = PingPongSlot(taskIdx);
        LocalTensor<InDtype> ubKnd = ubS6K[db];
        LocalTensor<InDtype> ubVnd = ubS6V[db];
        Duplicate(ubKnd, (InDtype)0, nK);
        Duplicate(ubVnd, (InDtype)0, nV);
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
        if (chunk.M > 0) {
            DataCopy(ubKnd, gmKHat[offK], static_cast<int32_t>(chunk.M * K));
            DataCopy(ubVnd, gmV[offV], static_cast<int32_t>(chunk.M * V));
        }
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);

        LocalTensor<float> beta = ubBetaEff[db];
        LocalTensor<float> g = ubGPrime[db];
        ScaleRowsAlignedVF<InDtype>(ubVnd, ubVnd, beta, static_cast<uint32_t>(chunkSize),
                                    static_cast<uint32_t>(V));
        BetaExp2gVF(beta, g, ubGateTmp, static_cast<uint32_t>(chunkSize));
        ScaleRowsK128VF<InDtype>(ubKnd, ubKnd, ubGateTmp, static_cast<uint32_t>(chunkSize));

        const int32_t nz = 1 - db;
        UploadBf16NdToL1(l1Vb[taskIdx], ubVnd, ubS6V[nz], static_cast<uint32_t>(V));
        UploadBf16NdToL1(l1Kbg[taskIdx], ubKnd, ubS6K[nz], static_cast<uint32_t>(K));
        NotifyAicStage6Done(taskIdx);
    }

    // ========================= Stage 7 =========================
    // W = A @ kbg then u = A @ vb. One MMAD each [64,64]@[64,128]. L0A/L0B
    // ping [32,48) even, pong [48,64) odd, 16 KiB each. Serial PIPE_ALL
    // around copy / MMAD / Fixpipe. A is still 64x64 (kStep=4); refresh
    // from gmA. B isTranspose kStep=8 dstStride=8. One Fixpipe n=128,
    // dstStride=K then V. Even FIX_M(0) / odd FIX_M(1), never both.
    __aicore__ inline void Stage7_MmadFix(LocalTensor<InDtype> l1A,
                                          LocalTensor<InDtype> l1B,
                                          LocalTensor<InDtype> l0A,
                                          LocalTensor<InDtype> l0B,
                                          LocalTensor<float> l0C,
                                          GlobalTensor<InDtype> gmOut,
                                          int64_t off, int32_t n, int32_t rows,
                                          int32_t bt, uint8_t evt)
    {
        WuMatmulToL0C<InDtype>(l1A, l1B, l0A, l0B, l0C, bt, n, bt, evt);
        PipeBarrier<PIPE_ALL>();
        if (rows > 0) {
            FixpipeL0cToGmNd<InDtype>(gmOut[off], l0C,
                                      static_cast<uint32_t>(rows),
                                      static_cast<uint32_t>(n),
                                      static_cast<uint32_t>(n));
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Stage7_AicOne(const ChunkRange &chunk, int64_t hv, int64_t taskIdx)
    {
        const int64_t offW = OffsetBHTD(chunk.batch, hv, chunk.tokenStart, HV, T, K);
        const int64_t offU = OffsetBHTD(chunk.batch, hv, chunk.tokenStart, HV, T, V);
        const int64_t offA = OffsetBHTD(chunk.batch, hv, chunk.tokenStart, HV, T, chunkSize);
        const int32_t rows = static_cast<int32_t>(chunk.M);
        const int32_t nK = static_cast<int32_t>(K);
        const int32_t nV = static_cast<int32_t>(V);
        const int32_t bt = static_cast<int32_t>(chunkSize);
        const uint8_t bank = static_cast<uint8_t>(taskIdx & 1);
        PipeBarrier<PIPE_ALL>();
        WaitFlag<HardEvent::FIX_M>(bank);
        CopyGmNdToL1Nz<InDtype>(l1A[taskIdx], gmA[offA], bt, bt);
        PipeBarrier<PIPE_ALL>();

        if (bank == 0) {
            Stage7_MmadFix(l1A[taskIdx], l1Kbg[taskIdx], l0AS7, l0BS7, l0CS7,
                           gmW, offW, nK, rows, bt, 0);
            Stage7_MmadFix(l1A[taskIdx], l1Vb[taskIdx], l0AS7, l0BS7, l0CS7,
                           gmU, offU, nV, rows, bt, 0);
            SetFlag<HardEvent::FIX_M>(0);
        } else {
            Stage7_MmadFix(l1A[taskIdx], l1Kbg[taskIdx], l0AS71, l0BS71, l0CS71,
                           gmW, offW, nK, rows, bt, 1);
            Stage7_MmadFix(l1A[taskIdx], l1Vb[taskIdx], l0AS71, l0BS71, l0CS71,
                           gmU, offU, nV, rows, bt, 1);
            SetFlag<HardEvent::FIX_M>(1);
        }
    }

    __aicore__ inline void ProcessAiv()
    {
        Stage0_GenerateResidentAux();
        const int64_t nPacks = CeilDiv(totalChunks, kTasksPerRound);
        for (int64_t pack = coreIdx; pack < nPacks; pack += numCore) {
            const int64_t base = pack * kTasksPerRound;
            const int64_t nThis = (totalChunks - base) < kTasksPerRound ? (totalChunks - base) : kTasksPerRound;
            for (int64_t t = subBlock; t < nThis; t += 2) {
                const int64_t workId = base + t;
                Stage1_OneTask(GetChunkRange(*this, gmCu, gmIdx, workId / HV),
                               workId % HV, t);
            }
            for (int64_t t = subBlock; t < nThis; t += 2) {
                const int64_t workId = base + t;
                Stage3_AivOne(GetChunkRange(*this, gmCu, gmIdx, workId / HV),
                              workId % HV, t);
            }
            // l1Y aliases l1KHat; Stage5 still reads Y. Wait Stage5 before
            // the next pack's Stage1 k'.
            for (int64_t t = subBlock; t < nThis; t += 2) {
                WaitAicStage5Done(t);
            }
            // // Wait Stage5 so Y/A L1 traffic is done, then vb/kbg NZ upload.
            // for (int64_t t = subBlock; t < nThis; t += 2) {
            //     WaitAicStage5Done(t);
            // }
            // for (int64_t t = subBlock; t < nThis; t += 2) {
            //     const int64_t workId = base + t;
            //     Stage6_AivOne(GetChunkRange(*this, gmCu, gmIdx, workId / HV),
            //                   workId % HV, t);
            // }
        }
    }

    __aicore__ inline void ProcessAic()
    {
        // Stage2 Wait FIX_M(bank) before the first MMAD. Prime both banks here
        // (no prior Fixpipe). kkt zeros via mmad.cmatrixInitVal, not a dummy C.
        SetFlag<HardEvent::FIX_M>(0);
        SetFlag<HardEvent::FIX_M>(1);
        const int64_t nPacks = CeilDiv(totalChunks, kTasksPerRound);
        for (int64_t pack = coreIdx; pack < nPacks; pack += numCore) {
            const int64_t base = pack * kTasksPerRound;
            const int64_t nThis = (totalChunks - base) < kTasksPerRound ? (totalChunks - base) : kTasksPerRound;
            for (int64_t t = 0; t < nThis; ++t) {
                CrossCoreWaitFlag<0x4, PIPE_MTE1>(CubeWaitFlagForTask(t));
                Stage2_AicOne(t);
            }
            for (int64_t t = 0; t < nThis; ++t) {
                WaitAivStage3Done(t);
                const int64_t workId = base + t;
                Stage4_AicOne(GetChunkRange(*this, gmCu, gmIdx, workId / HV),
                              workId % HV, t);
            }
            PipeBarrier<PIPE_ALL>();
            for (int64_t t = 0; t < nThis; ++t) {
                const int64_t workId = base + t;
                Stage5_AicOne(GetChunkRange(*this, gmCu, gmIdx, workId / HV),
                              workId % HV, t);
            }
            // for (int64_t t = 0; t < nThis; ++t) {
            //     WaitAivStage6Done(t);
            // }
            // PipeBarrier<PIPE_ALL>();
            // for (int64_t t = 0; t < nThis; ++t) {
            //     const int64_t workId = base + t;
            //     Stage7_AicOne(GetChunkRange(*this, gmCu, gmIdx, workId / HV),
            //                   workId % HV, t);
            //     PipeBarrier<PIPE_ALL>();
            // }
            // PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            ProcessAiv();
        }
        if ASCEND_IS_AIC {
            ProcessAic();
        }
    }

private:
    // GM
    // Input 
    GlobalTensor<InDtype> gmQ;
    GlobalTensor<InDtype> gmK;
    GlobalTensor<InDtype> gmV;
    GlobalTensor<GateDtype> gmG;
    GlobalTensor<GateDtype> gmBeta;
    GlobalTensor<ALogDtype> gmALog;
    GlobalTensor<ALogDtype> gmDt;
    // Output
    GlobalTensor<InDtype> gmQHat;
    GlobalTensor<InDtype> gmKHat;
    GlobalTensor<InDtype> gmW;
    GlobalTensor<float> gmLDump;
    GlobalTensor<InDtype> gmU;
    GlobalTensor<InDtype> gmA;
    GlobalTensor<float> gmQRstd;
    GlobalTensor<float> gmKRstd;
    GlobalTensor<float> gmGOut;
    GlobalTensor<float> gmBetaEff;
    // Workspace
    GlobalTensor<float> gmWsY;

    // Local
    // UB
    // S0
    LocalTensor<uint8_t> ubMaskBits;
    LocalTensor<uint32_t> ubVcsIdx;
    LocalTensor<float> ubIVcs, ubS0Zero, ubMaskFp32;

    // S1
    LocalTensor<float> ubGPrime[2],ubBetaEff[2];
    LocalTensor<InDtype> ubQ[2], ubK[2], ubQHat[2], ubKHat[2];
    LocalTensor<float> ubG[2], ubBeta[2], ubKRstd[2], ubQRstd[2];

    // S2
    LocalTensor<float> ubKkt[2];

    
    LocalTensor<float> ubLPacked, ubResVcs, ubLFull, ubS3Nz16, ubS3Nz8, ubS3LeafTmp;

    LocalTensor<float> ubGateTmp;
    LocalTensor<InDtype> ubS6K[2], ubS6V[2];

    // L1
    LocalTensor<InDtype> l1KHat[4], l1A[4], l1Kbg[4], l1Kbg1[4], l1Vb[4], l1Vb1[4];
    LocalTensor<float> l1NegL[4], l1LeafRight[4], l1LeafLeft[4], l1Y[4];

    // L0
    LocalTensor<float> l1I;
    LocalTensor<InDtype> l0A, l0B, l0A1, l0B1, l0AS7, l0BS7, l0AS71, l0BS71;
    LocalTensor<float> l0Af, l0Bf, l0Af1, l0Bf1, l0C, l0C1, l0CS7, l0CS71;

    int64_t hasBetaOut;
};

} // namespace GdnStage

#endif
