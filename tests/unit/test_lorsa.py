from types import SimpleNamespace

import torch

from llamascopium import LorsaConfig, LowRankSparseAttention


def test_w_v_active_subspace_init_supports_non_square_ov_dimension():
    torch.manual_seed(0)

    cfg = LorsaConfig(
        d_model=8,
        expansion_factor=1.5,
        hook_point_in="hook_in",
        hook_point_out="hook_out",
        n_qk_heads=3,
        d_qk_head=2,
        positional_embedding_type="none",
        rotary_dim=2,
        n_ctx=4,
        dtype=torch.float32,
        device="cpu",
    )
    lorsa = LowRankSparseAttention(cfg)
    lorsa.init_parameters()

    mhsa = SimpleNamespace(
        cfg=SimpleNamespace(n_heads=3, d_head=2, d_model=8),
        W_V=torch.randn(3, 8, 2),
        W_O=torch.randn(3, 2, 8),
    )
    assert mhsa.cfg.n_heads * mhsa.cfg.d_head != mhsa.cfg.d_model

    x = torch.randn(2, 4, 8)
    initial_w_v = lorsa.W_V.detach().clone()

    v_per_head = torch.einsum("bd,ndh->bnh", x.reshape(-1, cfg.d_model), mhsa.W_V)
    captured_v = torch.einsum("bnh,nhd->bnd", v_per_head, mhsa.W_V.permute(0, 2, 1))

    expected_w_v = initial_w_v.clone()
    expected_w_o = torch.empty_like(lorsa.W_O)
    n_ov_per_orig_head = cfg.n_ov_heads // mhsa.cfg.n_heads
    for orig_head_index in range(mhsa.cfg.n_heads):
        v = captured_v[:, orig_head_index]
        demeaned_v = v - v.mean(dim=0)
        U, _, _ = torch.svd(demeaned_v.T.to(torch.float32))
        proj_weight = U[:, : cfg.d_qk_head]
        head_slice = slice(
            orig_head_index * n_ov_per_orig_head,
            (orig_head_index + 1) * n_ov_per_orig_head,
        )
        expected_w_v[head_slice] = initial_w_v[head_slice, : cfg.d_qk_head] @ proj_weight.T
        expected_w_o[head_slice] = expected_w_v[head_slice] @ mhsa.W_V[orig_head_index] @ mhsa.W_O[orig_head_index]

    expected_w_v = expected_w_v / expected_w_v.norm(dim=1, keepdim=True)
    expected_w_o = expected_w_o / expected_w_o.norm(dim=1, keepdim=True)

    lorsa.init_W_V_with_active_subspace_per_head({"hook_in": x}, mhsa)

    assert torch.allclose(lorsa.W_V, expected_w_v)
    assert torch.allclose(lorsa.W_O, expected_w_o)
