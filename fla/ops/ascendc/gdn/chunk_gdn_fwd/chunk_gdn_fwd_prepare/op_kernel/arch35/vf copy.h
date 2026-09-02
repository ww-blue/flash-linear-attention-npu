/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * Vector VF used by fused chunk_gdn_fwd_prepare. Do not edit VF bodies.
 */

#ifndef CHUNK_GDN_FWD_PREPARE_VF_H
#define CHUNK_GDN_FWD_PREPARE_VF_H

#include "kernel_operator.h"
#include "common.h"

namespace GdnStage {
using namespace AscendC;
using namespace AscendC::MicroAPI;

namespace {
constexpr int32_t VL = 64;

constexpr CastTrait kCastB162B32 = {
    RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN,
};
constexpr CastTrait kCastB322B16 = {
    RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT,
};
} // namespace

template <typename T>
__aicore__ inline void LoadCastB16(__ubuf__ T *src, RegTensor<float> &dst, MaskReg &mask)
{
    if constexpr (IsSameType<T, float>::value) {
        LoadAlign(dst, src);
    } else {
        RegTensor<T> xB16;
        LoadAlign<T, LoadDist::DIST_UNPACK_B16>(xB16, src);
        Cast<float, T, kCastB162B32>(dst, xB16, mask);
    }
}

template <typename T>
__aicore__ inline float ScalarToFp32(T x)
{
    if constexpr (IsSameType<T, float>::value) {
        return x;
    } else if constexpr (IsSameType<T, half>::value) {
        return static_cast<float>(x);
    } else {
        return ToFloat(x);
    }
}

template <typename T>
__aicore__ inline void StoreCastB16(__ubuf__ T *dst, RegTensor<float> &src, MaskReg &mask)
{
    if constexpr (IsSameType<T, float>::value) {
        StoreAlign(dst, src, mask);
    } else {
        RegTensor<T> yB16;
        Cast<T, float, kCastB322B16>(yB16, src, mask);
        StoreAlign<T, StoreDist::DIST_PACK_B32>(dst, yB16, mask);
    }
}

// Inclusive prefix sum of 64 fp32 values, then * scale (RCP_LN2).
__aicore__ inline void ChunkCumsumScaleVF(LocalTensor<float> &x, float scale, uint32_t n)
{
    __ubuf__ float *addr = (__ubuf__ float *)x.GetPhyAddr();
    __VEC_SCOPE__
    {
        MaskReg preg1 = CreateMask<float, MaskPattern::VL1>();
        RegTensor<float> acc, v, s;
        Duplicate(acc, 0.0f, preg1);
        Duplicate(s, scale, preg1);
        for (uint16_t i = 0; i < static_cast<uint16_t>(n); i++) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(v, addr + i);
            Add(acc, acc, v, preg1);
            Mul(v, acc, s, preg1);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(addr + i, v, preg1);
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        }
    }
}

// fused gate: g_raw = -exp(a_log) * softplus(g + dt_bias)
// Stable softplus: relu(x) + ln(1 + exp(-|x|)). a_log is broadcast per head.
template <typename T>
__aicore__ inline void GateSoftplusVF(LocalTensor<T> &gIn, LocalTensor<float> &gRaw,
                                      float aLog, float dtBias, uint32_t n)
{
    __ubuf__ T *src = (__ubuf__ T *)gIn.GetPhyAddr();
    __ubuf__ float *dst = (__ubuf__ float *)gRaw.GetPhyAddr();
    __VEC_SCOPE__
    {
        uint32_t sreg = n;
        MaskReg preg = UpdateMask<float>(sreg);
        RegTensor<float> x, ax, nx, e, sp, reluX, negA, out, zero;
        LoadCastB16<T>(src, x, preg);
        Adds(x, x, dtBias, preg);
        Abs(ax, x, preg);
        Muls(nx, ax, -1.0f, preg);
        Exp(e, nx, preg);
        Adds(e, e, 1.0f, preg);
        Ln(sp, e, preg);
        Duplicate(zero, 0.0f, preg);
        Max(reluX, x, zero, preg);
        Add(sp, sp, reluX, preg);
        Duplicate(negA, aLog, preg);
        Exp(negA, negA, preg);
        Muls(negA, negA, -1.0f, preg);
        Mul(out, sp, negA, preg);
        StoreAlign(dst, out, preg);
    }
}

template <typename T>
__aicore__ inline void CopyToFp32VF(LocalTensor<T> &src, LocalTensor<float> &dst, uint32_t n)
{
    __ubuf__ T *s = (__ubuf__ T *)src.GetPhyAddr();
    __ubuf__ float *d = (__ubuf__ float *)dst.GetPhyAddr();
    __VEC_SCOPE__
    {
        uint32_t sreg = n;
        MaskReg preg = UpdateMask<float>(sreg);
        RegTensor<float> x;
        LoadCastB16<T>(s, x, preg);
        StoreAlign(d, x, preg);
    }
}

// sigmoid or 2*sigmoid
template <typename T>
__aicore__ inline void BetaSigmoidVF(LocalTensor<T> &betaIn, LocalTensor<float> &betaOut,
                                     float scale, uint32_t n)
{
    __ubuf__ T *src = (__ubuf__ T *)betaIn.GetPhyAddr();
    __ubuf__ float *dst = (__ubuf__ float *)betaOut.GetPhyAddr();
    __VEC_SCOPE__
    {
        uint32_t sreg = n;
        MaskReg preg = UpdateMask<float>(sreg);
        RegTensor<float> x, e, one, den, s;
        LoadCastB16<T>(src, x, preg);
        Muls(x, x, -1.0f, preg);
        Exp(e, x, preg);
        Duplicate(one, 1.0f, preg);
        Add(den, one, e, preg);
        Div(s, one, den, preg);
        Muls(s, s, scale, preg);
        StoreAlign(dst, s, preg);
    }
}

// L[i,j] = mask[i,j] * beta[i] * kkt[i,j] * exp2(clip(g[i]-g[j], -50, 50))
// mask is 64x64 fp32 strict-lower (1 if i>j else 0)
__aicore__ inline void ConstructLowerLExp2VF(LocalTensor<float> &kkt, LocalTensor<float> &g,
                                             LocalTensor<float> &beta, LocalTensor<float> &mask,
                                             LocalTensor<float> &L, uint32_t n)
{
    constexpr float kLn2 = 0.6931471825f;
    __ubuf__ float *kktAddr = (__ubuf__ float *)kkt.GetPhyAddr();
    __ubuf__ float *gAddr = (__ubuf__ float *)g.GetPhyAddr();
    __ubuf__ float *betaAddr = (__ubuf__ float *)beta.GetPhyAddr();
    __ubuf__ float *maskAddr = (__ubuf__ float *)mask.GetPhyAddr();
    __ubuf__ float *lAddr = (__ubuf__ float *)L.GetPhyAddr();
    __VEC_SCOPE__
    {
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> gJ, gI, kktRow, gDiff, gate, betaI, out, mRow, lo, hi;
        Duplicate(lo, -kGdnGateClip, pregAll);
        Duplicate(hi, kGdnGateClip, pregAll);
        LoadAlign(gJ, gAddr);
        for (uint16_t i = 0; i < 64; i++) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(gI, gAddr + i);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(betaI, betaAddr + i);
            LoadAlign(kktRow, kktAddr + static_cast<uint32_t>(i) * 64);
            LoadAlign(mRow, maskAddr + static_cast<uint32_t>(i) * 64);
            Sub(gDiff, gI, gJ, pregAll);
            Max(gDiff, gDiff, lo, pregAll);
            Min(gDiff, gDiff, hi, pregAll);
            Muls(gDiff, gDiff, kLn2, pregAll);
            Exp(gate, gDiff, pregAll);
            Mul(out, kktRow, gate, pregAll);
            Mul(out, out, betaI, pregAll);
            Mul(out, out, mRow, pregAll);
            StoreAlign(lAddr + static_cast<uint32_t>(i) * 64, out, pregAll);
        }
        (void)n;
    }
}

// y[t, :] = x[t, :] * scale[t]   last dim 128 (2 VL)
template <typename T>
__aicore__ inline void ScaleRowsK128VF(LocalTensor<T> &x, LocalTensor<T> &y,
                                       LocalTensor<float> &scale, uint32_t rows)
{
    __ubuf__ T *xAddr = (__ubuf__ T *)x.GetPhyAddr();
    __ubuf__ T *yAddr = (__ubuf__ T *)y.GetPhyAddr();
    __ubuf__ float *sAddr = (__ubuf__ float *)scale.GetPhyAddr();
    __VEC_SCOPE__
    {
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        for (uint16_t r = 0; r < static_cast<uint16_t>(rows); r++) {
            RegTensor<float> s, a, b, ya, yb;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(s, sAddr + r);
            LoadCastB16<T>(xAddr, a, pregAll);
            LoadCastB16<T>(xAddr + VL, b, pregAll);
            Mul(ya, a, s, pregAll);
            Mul(yb, b, s, pregAll);
            StoreCastB16<T>(yAddr, ya, pregAll);
            StoreCastB16<T>(yAddr + VL, yb, pregAll);
            xAddr += 2 * VL;
            yAddr += 2 * VL;
        }
    }
}

// y[t, :] = x[t, :] * scale[t]   last dim 256 (4 VL) or 128
template <typename T>
__aicore__ inline void ScaleRowsAlignedVF(LocalTensor<T> &x, LocalTensor<T> &y,
                                          LocalTensor<float> &scale, uint32_t rows, uint32_t dim)
{
    const uint16_t tiles = static_cast<uint16_t>(dim / VL);
    __ubuf__ T *xAddr = (__ubuf__ T *)x.GetPhyAddr();
    __ubuf__ T *yAddr = (__ubuf__ T *)y.GetPhyAddr();
    __ubuf__ float *sAddr = (__ubuf__ float *)scale.GetPhyAddr();
    __VEC_SCOPE__
    {
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        for (uint16_t r = 0; r < static_cast<uint16_t>(rows); r++) {
            RegTensor<float> s, a, o;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(s, sAddr + r);
            for (uint16_t t = 0; t < tiles; t++) {
                LoadCastB16<T>(xAddr + t * VL, a, pregAll);
                Mul(o, a, s, pregAll);
                StoreCastB16<T>(yAddr + t * VL, o, pregAll);
            }
            xAddr += dim;
            yAddr += dim;
        }
    }
}

// scale[t] = beta[t] * exp2(g[t])
__aicore__ inline void BetaExp2gVF(LocalTensor<float> &beta, LocalTensor<float> &g,
                                   LocalTensor<float> &out, uint32_t n)
{
    constexpr float kLn2 = 0.6931471825f;
    __ubuf__ float *bAddr = (__ubuf__ float *)beta.GetPhyAddr();
    __ubuf__ float *gAddr = (__ubuf__ float *)g.GetPhyAddr();
    __ubuf__ float *oAddr = (__ubuf__ float *)out.GetPhyAddr();
    __VEC_SCOPE__
    {
        uint32_t sreg = n;
        MaskReg preg = UpdateMask<float>(sreg);
        RegTensor<float> gv, bv, e;
        LoadAlign(gv, gAddr);
        LoadAlign(bv, bAddr);
        Muls(gv, gv, kLn2, preg);
        Exp(e, gv, preg);
        Mul(e, e, bv, preg);
        StoreAlign(oAddr, e, preg);
    }
}

namespace Prepare {

// Strict-lower 64x64 fp32 mask: 1 if col < row, else 0.
// One VL is 64 fp32 on 950, so each row is one RegTensor.
// ConstructLowerLExp2VF multiplies L by this; bit packing is unused.
__aicore__ inline void Stage0_ExpandBitMaskToFp32(AscendC::LocalTensor<float> ubMaskFp32)
{
    using namespace AscendC::MicroAPI;
    __ubuf__ float *dst = (__ubuf__ float *)ubMaskFp32.GetPhyAddr();
    __VEC_SCOPE__
    {
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> col;
        RegTensor<float> rowIdx;
        RegTensor<float> ones;
        RegTensor<float> zeros;
        RegTensor<float> row;
        Arange(col, 0.0f);
        Duplicate(ones, 1.0f, pregAll);
        Duplicate(zeros, 0.0f, pregAll);
        Duplicate(rowIdx, 0.0f, pregAll);
        for (uint16_t i = 0; i < static_cast<uint16_t>(kChunk64); ++i) {
            MaskReg pregLower;
            Compare<float, AscendC::CMPMODE::LT>(pregLower, col, rowIdx, pregAll);
            Select(row, ones, zeros, pregLower);
            StoreAlign(dst + static_cast<uint32_t>(i) * kChunk64, row, pregAll);
            Adds(rowIdx, rowIdx, 1.0f, pregAll);
        }
    }
}

} // namespace Prepare
} // namespace GdnStage

using namespace AscendC;

// 32×32 叶子 VF。两叶打包 32×64：oneRepeatSize=64，scatterCount=2。
template <typename T, typename U>
__simd_vf__ inline void MulReduceScatterVF32(__ubuf__ T *dstAddr, __ubuf__ T *src0Addr, __ubuf__ T *src1Addr,
                                             __ubuf__ U *idxAddr, uint32_t scatterCount, uint32_t oneRepeatSize)
{
    AscendC::Reg::RegTensor<T> srcReg0;
    AscendC::Reg::RegTensor<T> srcReg1;
    AscendC::Reg::RegTensor<T> dstMulReg;
    AscendC::Reg::RegTensor<T> dstReduceBlkReg;
    AscendC::Reg::RegTensor<T> dstReducePairReg;
    AscendC::Reg::RegTensor<T> dstReducePairReg2;
    AscendC::Reg::RegTensor<U> scatterIdxReg;

    uint32_t maskCount = oneRepeatSize;
    uint32_t pairMaskCount = scatterCount * 2;
    AscendC::Reg::MaskReg inputMask;
    AscendC::Reg::MaskReg pairMask;
    AscendC::Reg::MaskReg scatterMask;
    inputMask = AscendC::Reg::UpdateMask<T, AscendC::Reg::RegTraitNumOne>(maskCount);
    pairMask = AscendC::Reg::UpdateMask<T, AscendC::Reg::RegTraitNumOne>(pairMaskCount);
    scatterMask = AscendC::Reg::UpdateMask<T, AscendC::Reg::RegTraitNumOne>(scatterCount);
    for (uint16_t iterIdx = 1; iterIdx < 32; iterIdx++) {
        AscendC::Reg::LoadAlign(scatterIdxReg, idxAddr);
        AscendC::Reg::Adds(scatterIdxReg, scatterIdxReg, (uint32_t)iterIdx, scatterMask);
        for (uint16_t i = 0; i < iterIdx; i++) {
            AscendC::Reg::LoadAlign(srcReg0, src0Addr + iterIdx * oneRepeatSize);
            AscendC::Reg::LoadAlign(srcReg1, src1Addr + i * oneRepeatSize);
            AscendC::Reg::Mul(dstMulReg, srcReg0, srcReg1, inputMask);
            AscendC::Reg::ReduceDataBlock<AscendC::Reg::ReduceType::SUM>(dstReduceBlkReg, dstMulReg, inputMask);
            AscendC::Reg::PairReduceElem<AscendC::Reg::PairReduce::SUM>(dstReducePairReg, dstReduceBlkReg, inputMask);
            AscendC::Reg::PairReduceElem<AscendC::Reg::PairReduce::SUM>(dstReducePairReg2, dstReducePairReg, pairMask);
            AscendC::Reg::Scatter(dstAddr + i * oneRepeatSize, dstReducePairReg2, scatterIdxReg, scatterMask);
        }
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    }
}

#endif
