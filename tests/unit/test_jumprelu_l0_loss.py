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
