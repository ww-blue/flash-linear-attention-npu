/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * Isolated MIX demo: C = A @ B, bf16 [64,64] @ [64,128] -> [64,128].
 * First a fused-Stage5-like fp32 64x64 MMAD (LoadData 32 L0A rows / 16 KiB),
 * PipeBarrier ALL, then Stage7 bf16 LoadData (16 rows / 8 KiB) + MMAD.
 */

#ifndef W_MMAD_DEMO_H
#define W_MMAD_DEMO_H

#include "kernel_operator.h"

namespace WMmadDemoOp {
using namespace AscendC;

constexpr int32_t kM = 64;
constexpr int32_t kK = 64;
constexpr int32_t kN = 128;
constexpr uint16_t kNumMFracs64 = 4;
constexpr uint32_t kL1Fp32Bytes = 64 * 64 * sizeof(float);
constexpr uint32_t kL1AOff = kL1Fp32Bytes;
constexpr uint32_t kL1BOff = kL1Fp32Bytes + 64 * 64 * sizeof(bfloat16_t);

struct OnChipBuffer {
    template <typename T>
    using Tensor = LocalTensor<T>;

    __aicore__ inline OnChipBuffer()
    {
        buffer_[0] = Tensor<uint8_t>(TPosition::A1, 0, 512 * 1024);
        buffer_[1] = Tensor<uint8_t>(TPosition::A2, 0, 64 * 1024);
        buffer_[2] = Tensor<uint8_t>(TPosition::B2, 0, 64 * 1024);
        buffer_[3] = Tensor<uint8_t>(TPosition::CO1, 0, 256 * 1024);
    }

    template <typename Dtype>
    __aicore__ inline Tensor<Dtype> L1(uint32_t off) const
    {
        return buffer_[0][off].template ReinterpretCast<Dtype>();
    }
    template <typename Dtype>
    __aicore__ inline Tensor<Dtype> L0A(uint32_t off) const
    {
        return buffer_[1][off].template ReinterpretCast<Dtype>();
    }
    template <typename Dtype>
    __aicore__ inline Tensor<Dtype> L0B(uint32_t off) const
    {
        return buffer_[2][off].template ReinterpretCast<Dtype>();
    }
    template <typename Dtype>
    __aicore__ inline Tensor<Dtype> L0C(uint32_t off) const
    {
        return buffer_[3][off].template ReinterpretCast<Dtype>();
    }

    LocalTensor<uint8_t> buffer_[4];
};

__aicore__ inline void FillLoad2D(LoadData2DParamsV2 &p, uint16_t mStep, uint16_t kStep,
                                  uint16_t srcStride, uint16_t dstStride, bool transpose)
{
    p.mStartPosition = 0;
    p.kStartPosition = 0;
    p.mStep = mStep;
    p.kStep = kStep;
    p.srcStride = srcStride;
    p.dstStride = dstStride;
    p.ifTranspose = transpose;
    p.sid = 0;
}

template <typename T>
__aicore__ inline void CopyGmNdToL1Nz(LocalTensor<T> l1, GlobalTensor<T> gm,
                                      uint32_t rows, uint32_t cols)
{
    Nd2NzParams p;
    p.ndNum = 1;
    p.nValue = rows;
    p.dValue = cols;
    p.srcDValue = cols;
    p.srcNdMatrixStride = 0;
    p.dstNzC0Stride = static_cast<uint16_t>(rows);
    p.dstNzNStride = 1;
    p.dstNzMatrixStride = 0;
    DataCopy(l1, gm, p);
}

template <typename OutDtype>
__aicore__ inline void FixpipeL0cToGmNd(GlobalTensor<OutDtype> gm, LocalTensor<float> l0C,
                                        uint32_t validRows, uint32_t nSize, uint32_t dstStride)
{
    FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> p;
    p.nSize = static_cast<uint16_t>(nSize);
    p.mSize = static_cast<uint16_t>(validRows);
    p.srcStride = static_cast<uint16_t>(kM);
    p.dstStride = dstStride;
    p.quantPre = QuantMode_t::F322BF16;
    p.dualDstCtl = 0;
    p.subBlockId = 0;
    p.isChannelSplit = false;
    p.params.ndNum = 1;
    p.params.srcNdStride = 0;
    p.params.dstNdStride = 0;
    Fixpipe<OutDtype, float, CFG_ROW_MAJOR>(gm, l0C, p);
}

class Kernel {
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR pre, GM_ADDR c)
    {
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(a));
        gmB.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(b));
        gmS5.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pre));
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(c));
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            return;
        }
        ProcessAic();
    }

    __aicore__ inline void ProcessAic()
    {
        OnChipBuffer buf;
        LocalTensor<float> l1S5 = buf.L1<float>(0);
        LocalTensor<bfloat16_t> l1A = buf.L1<bfloat16_t>(kL1AOff);
        LocalTensor<bfloat16_t> l1B = buf.L1<bfloat16_t>(kL1BOff);
        LocalTensor<float> l0Af = buf.L0A<float>(0);
        LocalTensor<float> l0Bf = buf.L0B<float>(0);
        LocalTensor<bfloat16_t> l0A = buf.L0A<bfloat16_t>(0);
        LocalTensor<bfloat16_t> l0B = buf.L0B<bfloat16_t>(0);
        LocalTensor<float> l0C = buf.L0C<float>(0);

        SetFlag<HardEvent::FIX_M>(0);
        WaitFlag<HardEvent::FIX_M>(0);

        // ---- fused Stage5: fp32 64x64, LoadData kStep=8, 32 L0A rows ----
        CopyGmNdToL1Nz<float>(l1S5, gmS5, kM, kK);
        SetFlag<HardEvent::MTE2_MTE1>(0);
        WaitFlag<HardEvent::MTE2_MTE1>(0);

        const uint16_t mFracs = kM / 16;
        const uint16_t fp32KFracs = kK / 8;
        LoadData2DParamsV2 loadS5A;
        FillLoad2D(loadS5A, mFracs, fp32KFracs, kNumMFracs64, mFracs, false);
        LoadData(l0Af, l1S5, loadS5A);
        LoadData2DParamsV2 loadS5B;
        FillLoad2D(loadS5B, mFracs, fp32KFracs, kNumMFracs64, mFracs, true);
        LoadData(l0Bf, l1S5, loadS5B);
        SetFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::MTE1_M>(0);

        MmadParams mmadS5;
        mmadS5.m = kM;
        mmadS5.n = kK;
        mmadS5.k = kK;
        mmadS5.cmatrixInitVal = true;
        mmadS5.cmatrixSource = false;
        mmadS5.unitFlag = 0;
        Mmad(l0C, l0Af, l0Bf, mmadS5);
        SetFlag<HardEvent::M_FIX>(0);
        WaitFlag<HardEvent::M_FIX>(0);
        PipeBarrier<PIPE_ALL>();

        // ---- fused Stage7: bf16 [64,64]@[64,128], A LoadData only 16 rows ----
        CopyGmNdToL1Nz<bfloat16_t>(l1A, gmA, kM, kK);
        CopyGmNdToL1Nz<bfloat16_t>(l1B, gmB, kK, kN);
        SetFlag<HardEvent::MTE2_MTE1>(0);
        WaitFlag<HardEvent::MTE2_MTE1>(0);

        const uint16_t kFracs = kK / 16;
        const uint16_t nFracs = kN / 16;
        LoadData2DParamsV2 loadA;
        FillLoad2D(loadA, mFracs, kFracs, mFracs, mFracs, false);
        LoadData(l0A, l1A, loadA);
        LoadData2DParamsV2 loadB;
        FillLoad2D(loadB, kFracs, nFracs, kFracs, nFracs, true);
        LoadData(l0B, l1B, loadB);
        SetFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::MTE1_M>(0);

        MmadParams mmad;
        mmad.m = kM;
        mmad.n = kN;
        mmad.k = kK;
        mmad.cmatrixInitVal = true;
        mmad.cmatrixSource = false;
        mmad.unitFlag = 0;
        Mmad(l0C, l0A, l0B, mmad);

        SetFlag<HardEvent::M_FIX>(0);
        WaitFlag<HardEvent::M_FIX>(0);
        FixpipeL0cToGmNd<bfloat16_t>(gmC, l0C, kM, kM, kN);
        SetFlag<HardEvent::FIX_M>(0);
        WaitFlag<HardEvent::FIX_M>(0);
        FixpipeL0cToGmNd<bfloat16_t>(gmC[kM], l0C[kM * kM], kM, kM, kN);
        SetFlag<HardEvent::FIX_M>(0);
    }

private:
    GlobalTensor<bfloat16_t> gmA, gmB, gmC;
    GlobalTensor<float> gmS5;
};

} // namespace WMmadDemoOp

#endif
