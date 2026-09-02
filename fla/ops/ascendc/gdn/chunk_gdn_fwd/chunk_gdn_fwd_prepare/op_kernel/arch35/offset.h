/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * GM BNSD offsets, chunk walk, and on-chip byte offsets.
 */

#ifndef CHUNK_GDN_FWD_PREPARE_OFFSET_H
#define CHUNK_GDN_FWD_PREPARE_OFFSET_H

#include "common.h"
#include "mem.h"

namespace GdnStage {

constexpr uint32_t kPrepareKb = 1024;
constexpr uint32_t kSlotBf16_64 = 64 * 64 * static_cast<uint32_t>(sizeof(bfloat16_t));
constexpr uint32_t kSlotFp32_64 = 64 * 64 * static_cast<uint32_t>(sizeof(float));
constexpr uint32_t kVecFp32 = 64 * static_cast<uint32_t>(sizeof(float));
constexpr uint32_t kMaskBitsBytes = 64 * 64 / 8;

// UB resident
constexpr uint32_t kUbMaskBits = 0;
constexpr uint32_t kUbVcsIdx = 512;
constexpr uint32_t kUbIVcs = 1 * kPrepareKb;
constexpr uint32_t kUbGPrime0 = 9 * kPrepareKb;
constexpr uint32_t kUbBetaEff0 = 9 * kPrepareKb + kVecFp32;
constexpr uint32_t kUbGPrime1 = 9 * kPrepareKb + 2 * kVecFp32;
constexpr uint32_t kUbBetaEff1 = 9 * kPrepareKb + 3 * kVecFp32;

// UB S0 temps (released before S1 writes g')
constexpr uint32_t kUbS0INz8 = 9 * kPrepareKb;
constexpr uint32_t kUbS0Zero = 25 * kPrepareKb;

// UB S1 input db [10, 75). One AIV, two tasks: task 0/1 → ping, task 2/3 → pong.
constexpr uint32_t kUbS1QPing = 10 * kPrepareKb;
constexpr uint32_t kUbS1KPing = 26 * kPrepareKb;
constexpr uint32_t kUbS1QPong = 42 * kPrepareKb;
constexpr uint32_t kUbS1KPong = 58 * kPrepareKb;
constexpr uint32_t kUbS1GPing = 74 * kPrepareKb;
constexpr uint32_t kUbS1BetaPing = 74 * kPrepareKb + kVecFp32;
constexpr uint32_t kUbS1GPong = 74 * kPrepareKb + 2 * kVecFp32;
constexpr uint32_t kUbS1BetaPong = 74 * kPrepareKb + 3 * kVecFp32;
constexpr uint32_t kUbS1QHat = 75 * kPrepareKb;
constexpr uint32_t kUbS1KHat = 91 * kPrepareKb;
constexpr uint32_t kUbS1KRstd = 107 * kPrepareKb;
constexpr uint32_t kUbS1QRstd = 107 * kPrepareKb + kVecFp32;
constexpr uint32_t kUbS1GateTmp = 107 * kPrepareKb + 2 * kVecFp32;

constexpr uint32_t kUbGPrime[2] = {kUbGPrime0, kUbGPrime1};
constexpr uint32_t kUbBetaEff[2] = {kUbBetaEff0, kUbBetaEff1};
constexpr uint32_t kUbS1Q[2] = {kUbS1QPing, kUbS1QPong};
constexpr uint32_t kUbS1K[2] = {kUbS1KPing, kUbS1KPong};
constexpr uint32_t kUbS1G[2] = {kUbS1GPing, kUbS1GPong};
constexpr uint32_t kUbS1Beta[2] = {kUbS1BetaPing, kUbS1BetaPong};

// UB S2
constexpr uint32_t kUbS2KktPing = 140 * kPrepareKb;
constexpr uint32_t kUbS2KktPong = 156 * kPrepareKb;
constexpr uint32_t kUbS2Kkt[2] = {kUbS2KktPing, kUbS2KktPong};

// UB S3 (overlaps S1 input after k' is on L1/GM)
constexpr uint32_t kUbS3LPacked = 10 * kPrepareKb;
constexpr uint32_t kUbS3ResVcs = 18 * kPrepareKb;
constexpr uint32_t kUbS3LFull = 34 * kPrepareKb;
constexpr uint32_t kUbS3Nz16 = 50 * kPrepareKb;
constexpr uint32_t kUbS3Nz8 = 66 * kPrepareKb;
constexpr uint32_t kUbS3LeafTmp = 82 * kPrepareKb;
constexpr uint32_t kUbMaskFp32 = 172 * kPrepareKb;

// UB S6 (after S3; overlaps S1/S3). K' 16 KiB. V=128 tile is 16 KiB.
// Do not place V pong at UB[96,128): that covers rstd/gate at 107 used by
// BetaExp2gVF and stomps the pong vb ND (task 2/3). Both V slots stay
// below 107: ping [64,80), pong [80,96).
constexpr uint32_t kUbS6KPing = 32 * kPrepareKb;
constexpr uint32_t kUbS6KPong = 48 * kPrepareKb;
constexpr uint32_t kUbS6VPing = 64 * kPrepareKb;
constexpr uint32_t kUbS6VPong = 80 * kPrepareKb;
constexpr uint32_t kUbS6K[2] = {kUbS6KPing, kUbS6KPong};
constexpr uint32_t kUbS6V[2] = {kUbS6VPing, kUbS6VPong};

// L1
constexpr uint32_t kBytesA64 = 64 * 64 * 2;
constexpr uint32_t kBytesK128 = 64 * 128 * 2;
constexpr uint32_t kBytesVb256 = 64 * 256 * 2;
constexpr uint32_t kBytesFp32Nz64 = 64 * 64 * 4;
constexpr uint32_t kWsPerCoreBytes = 128 * kPrepareKb;
constexpr uint32_t kWsYElems = kChunk64 * kChunk64;
constexpr uint32_t kWsYPerCoreBytes = 64 * kPrepareKb;

constexpr uint32_t kL1KHat0 = 0;
constexpr uint32_t kL1NegL0 = 64 * kPrepareKb;
constexpr uint32_t kL1LeafRight0 = 128 * kPrepareKb;
constexpr uint32_t kL1LeafLeft0 = 192 * kPrepareKb;
constexpr uint32_t kL1ResidentA0 = 256 * kPrepareKb;
constexpr uint32_t kL1ResidentKbg0 = 288 * kPrepareKb;
constexpr uint32_t kL1ResidentVb0 = 352 * kPrepareKb;
constexpr uint32_t kL1ResidentI = 480 * kPrepareKb;
constexpr uint32_t kL1ResidentZero = 496 * kPrepareKb;

// L0 (same physical banks, typed views)
constexpr uint32_t kL0Bf16Pair = 16 * kPrepareKb;
constexpr uint32_t kL0Fp32Pair = 16 * kPrepareKb;
// Stage7 L0A/L0B: even [32, 48), odd [48, 64), 16 KiB each via SetSize.
// S2/S5 stay in [0, 32). L0C is 64x128 fp32 NZ = 32 KiB and stays at
// [0, 64) / [64, 128); do not share these A/B slots.
constexpr uint32_t kL0S7Slot = 16 * kPrepareKb;
constexpr uint32_t kL0S7Ping = 32 * kPrepareKb;
constexpr uint32_t kL0S7Pong = 48 * kPrepareKb;
constexpr uint32_t kL0C1 = 64 * kPrepareKb;
constexpr uint32_t kL0CZero = 128 * kPrepareKb;

__aicore__ inline uint32_t L1KHat(uint32_t taskIdx)
{
    return kL1KHat0 + taskIdx * kBytesK128;
}

// Y overwrites k' at L1[0, 64). Same 16 KiB slots: 64x128 bf16 == 64x64 fp32 NZ.
__aicore__ inline uint32_t L1Y(uint32_t taskIdx)
{
    return kL1KHat0 + taskIdx * kBytesFp32Nz64;
}

__aicore__ inline uint32_t L1NegL(uint32_t taskIdx)
{
    return kL1NegL0 + taskIdx * kBytesFp32Nz64;
}

__aicore__ inline uint32_t L1LeafRight(uint32_t taskIdx)
{
    return kL1LeafRight0 + taskIdx * kBytesFp32Nz64;
}

__aicore__ inline uint32_t L1LeafLeft(uint32_t taskIdx)
{
    return kL1LeafLeft0 + taskIdx * kBytesFp32Nz64;
}

__aicore__ inline uint32_t L1ResidentA(uint32_t taskIdx)
{
    return kL1ResidentA0 + taskIdx * kBytesA64;
}

__aicore__ inline uint32_t L1ResidentKbg(uint32_t taskIdx)
{
    return kL1ResidentKbg0 + taskIdx * kBytesK128;
}

__aicore__ inline uint32_t L1ResidentVb(uint32_t taskIdx)
{
    return kL1ResidentVb0 + taskIdx * kBytesVb256;
}

// One BT-token window on a single sequence (not including hv).
// chunkIdx in GetChunkRange is workId / HV.
struct ChunkRange {
    // BNSD dim-0. Varlen packed layout is always 0: sequences sit on T.
    int64_t batch;
    // Which sequence this window belongs to. Dense: same as batch.
    // Varlen: chunk_indices[2 * chunkIdx].
    int64_t seqId;
    // First token on T (or packed T). Dense: localChunk * BT.
    // Varlen: cu_seqlens[seqId] + localChunk * BT.
    int64_t tokenStart;
    // Valid tokens in this window, in 1..BT. The last window of a sequence
    // may be shorter than BT.
    int64_t M;
    // Window index within the sequence. Dense: chunkIdx % Ceil(T / BT).
    // Varlen: chunk_indices[2 * chunkIdx + 1].
    int64_t localChunk;
};

// Maps sequence-chunk index (workId / HV) to a BT-token window. No hv.
template <typename St>
__aicore__ inline ChunkRange GetChunkRange(const St &st,
                                           AscendC::GlobalTensor<int64_t> gmCuSeqlens,
                                           AscendC::GlobalTensor<int64_t> gmChunkIndices,
                                           int64_t chunkIdx)
{
    ChunkRange chunk{};
    const int64_t bt = st.chunkSize;
    if (st.isVariable != 0 && st.hasChunkIndices != 0) {
        chunk.seqId = gmChunkIndices.GetValue(chunkIdx * 2);
        chunk.localChunk = gmChunkIndices.GetValue(chunkIdx * 2 + 1);
        const int64_t seqStart = gmCuSeqlens.GetValue(chunk.seqId);
        const int64_t seqEnd = gmCuSeqlens.GetValue(chunk.seqId + 1);
        chunk.tokenStart = seqStart + chunk.localChunk * bt;
        int64_t remain = seqEnd - chunk.tokenStart;
        chunk.M = remain > bt ? bt : remain;
        chunk.batch = 0;
        return chunk;
    }
    const int64_t nt = CeilDiv(st.T, bt);
    chunk.batch = chunkIdx / nt;
    chunk.localChunk = chunkIdx % nt;
    chunk.seqId = chunk.batch;
    chunk.tokenStart = chunk.localChunk * bt;
    int64_t remain = st.T - chunk.tokenStart;
    chunk.M = remain > bt ? bt : remain;
    if (chunk.M < 0) {
        chunk.M = 0;
    }
    return chunk;
}

// Linear index of [b, h, t, 0] in GM [B, H, T, D].
__aicore__ inline int64_t OffsetBHTD(int64_t b, int64_t h, int64_t t, int64_t heads, int64_t seq, int64_t dim)
{
    return ((b * heads + h) * seq + t) * dim;
}

// Linear index of [b, h, t] in GM [B, H, T].
__aicore__ inline int64_t OffsetBHT(int64_t b, int64_t h, int64_t t, int64_t heads, int64_t seq)
{
    return (b * heads + h) * seq + t;
}

} // namespace GdnStage

#endif
