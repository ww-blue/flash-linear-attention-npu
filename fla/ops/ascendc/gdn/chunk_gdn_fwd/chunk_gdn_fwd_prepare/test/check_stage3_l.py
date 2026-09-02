"""Stage3 L / VCS leaf audit vs host golden.

Kernel dumps (until Stage7 owns these buffers):
  w  <- fp32 ND -L 64x64 per chunk (VF writes -L; host reports L = -dump)
  u  <- packed (I+Lii)^{-1} at row (tokenStart/BT)*32, 32x64
         left 32 cols = (I+L00)^{-1}, right 32 cols = (I+L11)^{-1}

Required case, same flags as test_chunk_gdn_fwd_prepare.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

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

    print("=== Stage1 (sanity; Stage3 must not stomp hats/g/beta) ===")
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
    L_got = -bf16_storage_as_f32(w, (B, HV, T, BT)).reshape(B, HV, nt, BT, BT)
    leaf_got = bf16_storage_as_f32(u, (B, HV, T, BT))

    L_s1 = golden_L_tiles(k_hat, g_cumsum, beta_out, clip=True)
    L_s1_noclip = golden_L_tiles(k_hat, g_cumsum, beta_out, clip=False)
    L_ref = golden_L_tiles(ref.k, ref.g, ref.beta, clip=True)

    tril = torch.tril(torch.ones(BT, BT, dtype=torch.bool), diagonal=-1)
    upper = ~tril

    print("\n=== dumped L vs host L from kernel k_hat/g/beta (clip=50, Stage2+3) ===")
    m = metrics(L_got, L_s1)
    m_lo = metrics(L_got, L_s1, tril)
    print("  all        ", fmt(m), f"nan={m['nan']}")
    print("  strict-lo  ", fmt(m_lo))
    print("  vs no-clip ", fmt(metrics(L_got, L_s1_noclip)))
    print("  vs Stage1 golden k/g/beta", fmt(metrics(L_got, L_ref)))
    print(f"  upper-tri |L_got| max={L_got[..., upper].abs().max().item():.4g} "
          f"diag |L_got| max={L_got.diagonal(dim1=-2, dim2=-1).abs().max().item():.4g}")

    # Worst tile.
    abs_e = (L_got - L_s1).abs()
    flat = abs_e.reshape(-1)
    idx = int(flat.argmax())
    loc = np.unravel_index(idx, tuple(abs_e.shape))
    print(f"  worst @ {loc}: got={L_got[loc].item():.6g} ref={L_s1[loc].item():.6g} "
          f"abs={abs_e[loc].item():.6g}")

    print("\n=== VCS packed leaves (ND is inv.T, same as solve_tri pre-NZ) ===")
    eye = torch.eye(32, dtype=torch.float64)
    inv00_err, inv11_err, res00, res11 = [], [], [], []
    for hv in range(HV):
        for c in range(nt):
            sl = slice(c * 32, (c + 1) * 32)
            packed = leaf_got[0, hv, sl].double()
            x00, x11 = packed[:, :32].T, packed[:, 32:].T
            L00 = L_got[0, hv, c, :32, :32].double()
            L11 = L_got[0, hv, c, 32:, 32:].double()
            g00 = torch.linalg.inv(eye + L00)
            g11 = torch.linalg.inv(eye + L11)
            inv00_err.append(float((x00 - g00).abs().max()))
            inv11_err.append(float((x11 - g11).abs().max()))
            res00.append(float(((eye + L00) @ x00 - eye).abs().max()))
            res11.append(float(((eye + L11) @ x11 - eye).abs().max()))
    print(f"  (I+L00)^-1 max|err|={max(inv00_err):.4g} mean={float(np.mean(inv00_err)):.4g}")
    print(f"  (I+L11)^-1 max|err|={max(inv11_err):.4g} mean={float(np.mean(inv11_err)):.4g}")
    print(f"  residual |(I+L00)@inv-I| max={max(res00):.4g} mean={float(np.mean(res00)):.4g}")
    print(f"  residual |(I+L11)@inv-I| max={max(res11):.4g} mean={float(np.mean(res11)):.4g}")
    print(f"  leaf absmax={leaf_got.abs().max().item():.4g} "
          f"nan={int((~torch.isfinite(leaf_got)).sum())}")

    # Official A is not produced (Stage5 off).
    print(f"\nA got_absmax={A.float().abs().max().item():.4g} (Stage5 off, expect 0)")


if __name__ == "__main__":
    main()
