"""Demo: fp32 64x64 MMAD (dirty L0A) then bf16 [64,64]@[64,128], fused Stage5+7."""
from __future__ import annotations

import os
import traceback

import torch
from fla_npu.ops.ascendc._aclnn_ctypes import _call_aclnn, _empty_like


def w_mmad_demo(a: torch.Tensor, b: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    c = _empty_like(b)
    return _call_aclnn(
        "aclnnWMmadDemo",
        lambda ctx: [
            ctx.tensor(a, "a"),
            ctx.tensor(b, "b"),
            ctx.tensor(pre, "pre"),
            ctx.tensor(c, "c"),
        ],
        c,
    )


def main():
    dev = int(os.environ.get("TEST_DEVICE_ID", "0"))
    torch.npu.set_device(dev)
    torch.manual_seed(0)
    a = torch.randn(64, 64, dtype=torch.bfloat16, device="npu")
    b = torch.randn(64, 128, dtype=torch.bfloat16, device="npu")
    # Stage5-like leftover: large fp32 64x64 (kernel A absmax can be 1e9).
    pre = torch.randn(64, 64, dtype=torch.float32, device="npu") * 1e3
    ref = torch.matmul(a.float(), b.float())
    print(f"device={dev} pre absmax={pre.abs().max().item():.4g}")
    c = w_mmad_demo(a, b, pre)
    torch.npu.synchronize()
    got = c.float().cpu()
    r = ref.cpu()
    d = (got - r).abs()
    close = torch.isclose(got, r, atol=8e-2, rtol=8e-2)
    nfail = int((~close).sum().item())
    print(f"device={dev} shape A={tuple(a.shape)} B={tuple(b.shape)} C={tuple(c.shape)}")
    print(f"vs A@B fail={nfail}/{got.numel()} max={d.max().item():.4g} mean={d.mean().item():.4g}")
    print(f"C absmax={got.abs().max().item():.4g} zeros={int((got == 0).sum().item())}/{got.numel()}")
    print("C[:2,:8]=\n", got[:2, :8])
    print("ref[:2,:8]=\n", r[:2, :8])
    print("C[:2,64:72]=\n", got[:2, 64:72])
    print("ref[:2,64:72]=\n", r[:2, 64:72])
    if nfail:
        raise SystemExit(f"FAIL w_mmad_demo mismatch max={d.max().item():.4g}")
    print("PASS w_mmad_demo")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL w_mmad_demo: {exc}")
        traceback.print_exc()
        raise
