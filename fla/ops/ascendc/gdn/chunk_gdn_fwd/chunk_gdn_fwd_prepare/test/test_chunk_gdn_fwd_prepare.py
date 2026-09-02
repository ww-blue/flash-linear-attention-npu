"""Numerics test for fused chunk_gdn_fwd_prepare vs Gate_of_Babylon gdn golden.

Golden is exactly ``cpu_gdn_fwd_l2norm_to_recompute`` in
``/data/w00933206/ops/Gate_of_Babylon/tests/gdn/cpu_golden.py``.
That file is not modified. Intermediate dtypes stay as written there:

  l2norm:  fp32 rstd, hat cast back to input dtype (bf16 here)
  g/beta:  fp32; cumsum is fp32 * RCP_LN2
  A/w/u:   fp32 (exp2 + linalg.inv + matmul)

Required case: B=1, HK=4, HV=8, T=1792, K=V=128, BT=64, BF16.
  use_qk_l2norm=True, use_exp2=True, use_beta_sigmoid=True, use_gate=False.
  q_hat / k_hat                         elementwise atol=rtol=2e-2
  q_rstd / k_rstd / beta_out / g_cumsum elementwise atol=rtol=5e-3
  A                                     elementwise atol=rtol=8e-2
  w / u                                 per-chunk tile-max rel <= 8e-2
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import torch
from fla_npu.ops import ascendc as ascendc_ops

sys.path.insert(0, "/data/w00933206/ops/Gate_of_Babylon/tests/gdn")
from cpu_golden import cpu_gdn_fwd_l2norm_to_recompute  # noqa: E402

B, HK, HV, T, K, V, BT = 1, 4, 8, 1792, 128, 128, 64
GDN_DIR = Path("/data/w00933206/ops/Gate_of_Babylon/tests/gdn/cpu_golden.py")


def setup_npu():
    os.environ["TBE_PARALLEL_COMPILE_ENABLE"] = "0"
    os.environ["PARALLEL_COMPILE"] = "0"
    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))


def assert_close(name, got, ref, atol, rtol):
    g = got.detach().float()
    r = ref.detach().float()
    d = (g - r).abs()
    print(f"  {name:12s} max={d.max().item():.4g} mean={d.mean().item():.4g} "
          f"got_absmax={g.abs().max().item():.4g} ref_absmax={r.abs().max().item():.4g}")
    if not torch.allclose(g, r, atol=atol, rtol=rtol, equal_nan=False):
        raise AssertionError(
            f"{name} mismatch max={d.max().item():.4g} mean={d.mean().item():.4g}"
        )


def assert_tile_rel(name, got, ref, n_head, dim, limit=8e-2):
    g = got.detach().float().reshape(1, n_head, T // BT, BT, dim)
    r = ref.detach().float().reshape(1, n_head, T // BT, BT, dim)
    trel = (g - r).abs().amax(dim=(-1, -2)) / r.abs().amax(dim=(-1, -2)).clamp(min=1e-12)
    n_bad = int((trel > limit).sum().item())
    print(f"  {name:12s} tile-rel max={trel.max().item():.4g} mean={trel.mean().item():.4g} "
          f"gt{limit}={n_bad}/{trel.numel()} got_absmax={g.abs().max().item():.4g} "
          f"ref_absmax={r.abs().max().item():.4g}")
    if n_bad > 0 or (not torch.isfinite(g).all()):
        raise AssertionError(f"{name} tile-rel max={trel.max().item():.4g} (need <= {limit})")


def run_required_case():
    setup_npu()
    torch.manual_seed(0)
    dtype = torch.bfloat16
    # Inputs, golden, and checks stay on CPU. Only the fused kernel runs on NPU.
    q = torch.randn(B, HK, T, K, dtype=dtype)
    k = torch.randn(B, HK, T, K, dtype=dtype)
    v = torch.randn(B, HV, T, V, dtype=dtype)
    g = torch.randn(B, HV, T, dtype=torch.float32)
    beta = torch.randn(B, HV, T, dtype=torch.float32)

    outs = ascendc_ops.chunk_gdn_fwd_prepare(
        q.contiguous().npu(),
        k.contiguous().npu(),
        v.contiguous().npu(),
        g.contiguous().npu(),
        beta.contiguous().npu(),
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        use_exp2=True,
    )
    torch.npu.synchronize()
    q_hat, k_hat, q_rstd, k_rstd, beta_out, g_cumsum, w, u, A = [t.cpu() for t in outs]

    ref = cpu_gdn_fwd_l2norm_to_recompute(
        q, k, v, g, beta,
        chunk_size=BT,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=True,
        layout="bnsd",
    )

    print(f"golden={GDN_DIR}")
    print("fused prepare vs gdn cpu_gdn_fwd_l2norm_to_recompute  "
          f"B={B} HK={HK} HV={HV} T={T} K={K} V={V} BT={BT} BF16")
    assert_close("q_hat", q_hat, ref.q, 2e-2, 2e-2)
    assert_close("k_hat", k_hat, ref.k, 2e-2, 2e-2)
    assert_close("q_rstd", q_rstd, ref.q_rstd, 5e-3, 5e-3)
    assert_close("k_rstd", k_rstd, ref.k_rstd, 5e-3, 5e-3)
    assert_close("beta_out", beta_out, ref.beta, 5e-3, 5e-3)
    assert_close("g_cumsum", g_cumsum, ref.g, 5e-3, 5e-3)
    assert_close("A", A, ref.a, 8e-2, 8e-2)
    assert_tile_rel("w", w, ref.w, HV, K)
    assert_tile_rel("u", u, ref.u, HV, V)
    print("PASS")


def main():
    os.environ["TBE_PARALLEL_COMPILE_ENABLE"] = "0"
    os.environ["PARALLEL_COMPILE"] = "0"
    try:
        run_required_case()
    except Exception as exc:
        print(f"FAIL: {exc}")
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
