"""Stage0-5 audit vs host golden.

Kernel dumps (until Stage7 owns u):
  u  <- fp32 ND Y = I + LeafLeft @ (-L), 64x64 per chunk (Stage4)
  A  <- bf16 ND (I+L)^{-1} 64x64 per chunk (Stage5)

L is no longer dumped on w. Host L for Y reconstruction is from
kernel k_hat / g_cumsum / beta_out (clip=50).

Required case, same flags as test_chunk_gdn_fwd_prepare.py.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
from fla_npu.ops import ascendc as ascendc_ops

sys.path.insert(0, "/data/w00933206/ops/Gate_of_Babylon/tests/gdn")
from cpu_golden import cpu_gdn_fwd_l2norm_to_recompute  # noqa: E402

B, HK, HV, T, K, V, BT = 1, 4, 8, 1792, 128, 128, 64
CLIP = 50.0
EPS = 1e-7


def setup_npu():
    os.environ["TBE_PARALLEL_COMPILE_ENABLE"] = "0"
    os.environ["PARALLEL_COMPILE"] = "0"
    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))


def bf16_storage_as_f32(t: torch.Tensor, shape):
    bits = t.detach().cpu().contiguous().view(torch.int16).numpy()
    return torch.from_numpy(np.array(bits).view(np.float32).reshape(shape).copy())


def metrics(actual, golden, mask=None):
    a = actual.detach().cpu().double().numpy()
    g = golden.detach().cpu().double().numpy()
    if mask is not None:
        m = mask.detach().cpu().numpy().astype(bool)
        if m.ndim == 2:
            a, g = a[..., m], g[..., m]
        else:
            a, g = a[m], g[m]
    else:
        a, g = a.reshape(-1), g.reshape(-1)
    fin = np.isfinite(a) & np.isfinite(g)
    a, g = a[fin], g[fin]
    abs_err = np.abs(a - g)
    rel = abs_err / (np.abs(g) + EPS)
    return {
        "n": int(a.size),
        "max_abs": float(abs_err.max()) if a.size else 0.0,
        "mean_abs": float(abs_err.mean()) if a.size else 0.0,
        "mare": float(rel.max()) if a.size else 0.0,
        "mere": float(rel.mean()) if a.size else 0.0,
        "got_absmax": float(np.abs(a).max()) if a.size else 0.0,
        "ref_absmax": float(np.abs(g).max()) if a.size else 0.0,
        "nan": int((~np.isfinite(actual.detach().cpu().numpy().reshape(-1))).sum()),
    }


def fmt(m):
    return (f"max|err|={m['max_abs']:.4g} mean={m['mean_abs']:.4g} "
            f"MARE={100 * m['mare']:.4g}% MERE={100 * m['mere']:.4g}% "
            f"got_absmax={m['got_absmax']:.4g} ref_absmax={m['ref_absmax']:.4g} n={m['n']}")


def golden_L_tiles(k_bnsd, g_bnsd, beta_bnsd, clip=True):
    """k [B,HK,T,K], g/beta [B,HV,T] -> L [B,HV,NT,BT,BT] fp32."""
    g_ratio = HV // k_bnsd.shape[1]
    k_hv = k_bnsd.float().repeat_interleave(g_ratio, dim=1)
    nt = T // BT
    k_c = k_hv.reshape(B, HV, nt, BT, -1)
    g_c = g_bnsd.float().reshape(B, HV, nt, BT)
    b_c = beta_bnsd.float().reshape(B, HV, nt, BT)
    kkt = torch.matmul(k_c, k_c.transpose(-1, -2))
    tril = torch.tril(torch.ones(BT, BT, dtype=torch.bool), diagonal=-1)
    gdiff = g_c.unsqueeze(-1) - g_c.unsqueeze(-2)
    if clip:
        gdiff = gdiff.clamp(-CLIP, CLIP)
    gate = torch.exp2(gdiff)
    return kkt * gate * b_c.unsqueeze(-1) * tril.to(kkt.dtype)


def main():
    setup_npu()
    torch.manual_seed(0)
    dtype = torch.bfloat16
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

    print("=== Stage1 (sanity; Stage3-5 must not stomp hats/g/beta) ===")
    for name, got, r, atol in (
        ("q_hat", q_hat, ref.q, 2e-2),
        ("k_hat", k_hat, ref.k, 2e-2),
        ("q_rstd", q_rstd, ref.q_rstd, 5e-3),
        ("k_rstd", k_rstd, ref.k_rstd, 5e-3),
        ("beta_out", beta_out, ref.beta, 5e-3),
        ("g_cumsum", g_cumsum, ref.g, 5e-3),
    ):
        d = (got.float() - r.float()).abs()
        ok = torch.allclose(got.float(), r.float(), atol=atol, rtol=atol, equal_nan=False)
        print(f"  {name:10s} max={d.max().item():.4g} mean={d.mean().item():.4g} "
              f"got_absmax={got.float().abs().max().item():.4g} PASS={ok}")

    nt = T // BT
    Y_got = bf16_storage_as_f32(u, (B, HV, T, BT)).reshape(B, HV, nt, BT, BT)
    L_s1 = golden_L_tiles(k_hat, g_cumsum, beta_out, clip=True)
    eye32 = torch.eye(32, dtype=torch.float64)
    eye64 = torch.eye(BT, dtype=torch.float64)
    L_d = L_s1.double()
    XL = torch.linalg.inv(eye32 + L_d[..., 32:, 32:])

    print("\n=== Stage4 Y = I + LeafLeft @ (-L) (host L from k_hat/g/beta) ===")
    Y_ref = eye64.expand_as(L_d).clone()
    Y_ref[..., 32:, :] = XL @ (-L_d[..., 32:, :]) + eye64[32:]
    m_y = metrics(Y_got, Y_ref)
    print("  all        ", fmt(m_y), f"nan={m_y['nan']}")
    I_top = eye64[:32].expand(B, HV, nt, 32, BT)
    m_tl = metrics(Y_got[..., :32, :], I_top)
    print("  Y[:32] vs I", fmt(m_tl))
    print(f"  Y absmax={Y_got.abs().max().item():.4g}")

    print("\n=== Stage5 A = (I+L)^{-1} (bf16 ND) ===")
    A_got = A.float().reshape(B, HV, nt, BT, BT)
    A_inv = torch.linalg.inv(eye64 + L_d)
    m_a = metrics(A_got, A_inv)
    print("  vs inv(I+L_s1) ", fmt(m_a), f"nan={m_a['nan']}")
    m_ag = metrics(A.float(), ref.a.float())
    print("  vs golden A    ", fmt(m_ag))
    res = (eye64 + L_d) @ A_got.double() - eye64
    print(f"  |(I+L)@A-I| max={res.abs().max().item():.4g} mean={res.abs().mean().item():.4g}")
    XR = torch.linalg.inv(eye32 + L_d[..., :32, :32])
    print("  A[:32,:32] vs XR", fmt(metrics(A_got[..., :32, :32], XR)))
    print("  A[:32,32:] ~ 0  ", fmt(metrics(A_got[..., :32, 32:],
                                           torch.zeros(B, HV, nt, 32, 32))))
    print("  A[32:,32:] vs XL", fmt(metrics(A_got[..., 32:, 32:], XL)))
    print(f"  A absmax={A_got.abs().max().item():.4g}")
    print(f"  w absmax={w.float().abs().max().item():.4g} (Stage7 off, not L dump)")


if __name__ == "__main__":
    main()
