import hashlib

import safetensors.torch as safe

from llamascopium.models.sae import SAEConfig, SparseAutoEncoder
from llamascopium.trainer import Trainer, TrainerConfig


def test_save_checkpoint_syncs_latest_weights_to_root(tmp_path) -> None:
    sae = SparseAutoEncoder(
        SAEConfig(
            hook_point_in="layer_0",
            hook_point_out="layer_0",
            d_model=8,
            expansion_factor=2,
            device="cpu",
            dtype="float32",
        )
    )
    trainer = Trainer(
        TrainerConfig(
            exp_result_path=str(tmp_path),
            n_checkpoints=0,
        )
    )
    trainer.cur_step = 876

    trainer.save_checkpoint(sae=sae, checkpoint_path=tmp_path)

    latest_weights = tmp_path / "sae_weights.safetensors"
    step_weights = tmp_path / "checkpoints" / "step_876" / "sae_weights.safetensors"

    assert latest_weights.exists()
    assert step_weights.exists()

    latest_state = safe.load_file(str(latest_weights))
    step_state = safe.load_file(str(step_weights))

    assert latest_state.keys() == step_state.keys()
    assert hashlib.sha256(latest_weights.read_bytes()).hexdigest() == hashlib.sha256(
        step_weights.read_bytes()
    ).hexdigest()
