import math
import os

import torch

from lm_saes import (
    ActivationFactoryConfig,
    ActivationFactoryDatasetSource,
    ActivationFactoryTarget,
    BufferShuffleConfig,
    DatasetConfig,
    InitializerConfig,
    LanguageModelConfig,
    SAEConfig,
    TrainerConfig,
    TrainSAESettings,
    WandbConfig,
    train_sae,
)

if __name__ == "__main__":
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    settings = TrainSAESettings(
        sae=SAEConfig(
            hook_point_in="blocks.6.hook_resid_post",
            hook_point_out="blocks.6.hook_resid_post",
            d_model=768,
            expansion_factor=8,
            act_fn="jumprelu",
            jumprelu_threshold_window=4.0,
            dtype=torch.float32,
            device="cuda",
        ),
        initializer=InitializerConfig(
            grid_search_init_norm=True,
            init_log_jumprelu_threshold_value=math.log(0.1),
        ),
        trainer=TrainerConfig(
            lr=5e-5,
            l1_coefficient=0.3,
            total_training_tokens=800_000_000,
            log_frequency=2000,
            eval_frequency=1000000,
            n_checkpoints=0,
            sparsity_loss_type="tanh-quad",
            use_batch_norm_mse=False,
            check_point_save_mode="linear",
            exp_result_path="results",
            jumprelu_lr_factor=0.1,
        ),
        model=LanguageModelConfig(
            model_name="EleutherAI/pythia-160m",
            device="cuda",
            dtype="torch.float16",
        ),
        model_name="pythia-160m",
        datasets={
            "SlimPajama-3B": DatasetConfig(
                dataset_name_or_path="Hzfinfdu/SlimPajama-3B",
            )
        },
        wandb=WandbConfig(
            wandb_project="lm-saes",
            exp_name="pythia-160m-sae",
        ),
        activation_factory=ActivationFactoryConfig(
            sources=[
                ActivationFactoryDatasetSource(
                    name="SlimPajama-3B",
                )
            ],
            target=ActivationFactoryTarget.ACTIVATIONS_1D,
            hook_points=["blocks.6.hook_resid_post"],
            batch_size=4096,
            buffer_size=4096 * 4,
            buffer_shuffle=BufferShuffleConfig(
                perm_seed=42,
                generator_device="cuda",
            ),
        ),
        sae_name="pythia-160m-sae",
        sae_series="pythia-sae",
    )
    train_sae(settings)
