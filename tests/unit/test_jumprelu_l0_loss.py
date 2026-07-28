import math

import torch

from llamascopium import SAEConfig
from llamascopium.activation_functions import JumpReLU
from llamascopium.models.sae import SparseAutoEncoder


def test_jumprelu_l0_quad_loss_updates_threshold():
    cfg = SAEConfig(
        hook_point_in="in",
        hook_point_out="out",
        d_model=2,
        expansion_factor=2,
        device="cpu",
        dtype=torch.float32,
        act_fn="jumprelu",
        jumprelu_threshold_window=2.0,
    )
    sae = SparseAutoEncoder(cfg)
    sae.init_parameters(
        encoder_uniform_bound=1.0,
        decoder_uniform_bound=1.0,
        init_log_jumprelu_threshold_value=math.log(0.5),
    )
    assert isinstance(sae.activation_function, JumpReLU)

    with torch.no_grad():
        sae.W_E.fill_(0.0)
        sae.W_E[0, 0] = 1.0
        sae.W_E[1, 1] = 1.0
        sae.b_E.zero_()
        sae.W_D.fill_(0.1)
        sae.b_D.zero_()

    batch = {
        "in": torch.tensor([[0.4, 0.6]], dtype=torch.float32),
        "out": torch.zeros(1, 2, dtype=torch.float32),
    }
    loss = sae.compute_loss(
        batch,
        sparsity_loss_type="jumprelu-l0-quad",
        target_l0=2.0,
        l1_coefficient=1.0,
        return_aux_data=False,
    )
    loss.backward()

    grad = sae.activation_function.log_jumprelu_threshold.grad
    assert grad is not None
    assert grad.abs().sum() > 0


def test_jumprelu_lp_loss_only_penalizes_dead_features():
    cfg = SAEConfig(
        hook_point_in="in",
        hook_point_out="out",
        d_model=2,
        expansion_factor=2,
        device="cpu",
        dtype=torch.float32,
        act_fn="jumprelu",
        jumprelu_threshold_window=2.0,
    )
    sae = SparseAutoEncoder(cfg)
    sae.init_parameters(
        encoder_uniform_bound=1.0,
        decoder_uniform_bound=1.0,
        init_log_jumprelu_threshold_value=math.log(0.5),
    )
    assert isinstance(sae.activation_function, JumpReLU)

    with torch.no_grad():
        sae.W_E.zero_()
        sae.b_E.copy_(torch.tensor([0.3, 0.2, 0.7, 0.8]))
        sae.W_D.zero_()
        sae.W_D[:, 0] = 1.0
        sae.W_D[0, 0] = 2.0
        sae.b_D.zero_()

    dead_mask = torch.tensor([True, True, False, False])
    update_calls = 0

    def update_dead_statistics(*args):
        nonlocal update_calls
        update_calls += 1
        return dead_mask

    batch = {
        "in": torch.zeros(1, 2),
        "out": torch.zeros(1, 2),
        "tokens": torch.zeros(1, dtype=torch.long),
    }
    ctx = sae.compute_loss(
        batch,
        lp_coefficient=2.0,
        update_dead_statistics=update_dead_statistics,
        return_aux_data=True,
    )

    # Feature 0 is above threshold in gate space: 0.3 * ||W_D[0]|| = 0.6.
    # Only feature 1 contributes: 2.0 * (0.5 - 0.2) = 0.6.
    torch.testing.assert_close(ctx["l_p"], torch.tensor(0.6))
    assert update_calls == 1

    ctx["l_p"].backward()
    assert sae.b_E.grad is not None
    torch.testing.assert_close(sae.b_E.grad, torch.tensor([0.0, -2.0, 0.0, 0.0]))
    assert sae.activation_function.log_jumprelu_threshold.grad is None
