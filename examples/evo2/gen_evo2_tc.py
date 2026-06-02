from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from llamascopium import (
    ActivationFactoryTarget,
    BufferShuffleConfig,
    DatasetConfig,
    GenerateActivationsSettings,
    LanguageModelConfig,
    generate_activations,
)

from llamascopium.utils.evo2_hooks import (
    default_activation_dir,
    get_evo2_arch,
    tc_hook_points_for_all_layers,
    tc_hook_points_for_layer,
    validate_layer,
)

DEFAULT_DATASET_PATH = Path("/inspire/hdd/global_user/hezhengfu-240208120186/data/rlin_data/Evo2/evo2_gtdb_v220_stitched_sae")


def parse_layers(layer_specs: list[str] | None, num_layers: int) -> list[int]:
    if not layer_specs:
        return list(range(num_layers))

    layers: set[int] = set()
    for spec_group in layer_specs:
        for spec in spec_group.split(","):
            spec = spec.strip()
            if not spec:
                continue
            if "-" in spec:
                start_str, end_str = spec.split("-", maxsplit=1)
                start = int(start_str)
                end = int(end_str)
                if start > end:
                    raise ValueError(f"Invalid layer range `{spec}`: start must be <= end.")
                for layer in range(start, end + 1):
                    validate_layer(layer, num_layers)
                    layers.add(layer)
            else:
                layer = int(spec)
                validate_layer(layer, num_layers)
                layers.add(layer)
    return sorted(layers)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Evo2 transcoder activations (1D).")
    parser.add_argument("--model-name", type=str, default="evo2_7b")
    parser.add_argument(
        "--layers",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional layer selection. Supports single layers, comma-separated lists, "
            "and inclusive ranges like `0-14`. Example: `--layers 0-14 17 20,22`."
        ),
    )
    parser.add_argument("--dataset-path", type=str, default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--dataset-name", type=str, default="dna")
    parser.add_argument("--output-dir", type=str, default=str(default_activation_dir("tc_1d")))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--backend", type=str, choices=["transformer_lens", "evo2"], default="transformer_lens")
    parser.add_argument("--activation-device", type=str, choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--total-tokens", type=int, default=1_000_000)
    parser.add_argument("--context-size", type=int, default=64)
    parser.add_argument("--crop-mode", type=str, choices=["prefix", "random"], default="random")
    parser.add_argument("--crop-seed", type=int, default=42)
    parser.add_argument("--model-batch-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--buffer-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="torch.bfloat16")
    return parser


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    args = build_parser().parse_args()
    arch = get_evo2_arch(args.model_name)
    selected_layers = parse_layers(args.layers, int(arch["num_layers"]))
    hook_points = (
        tc_hook_points_for_all_layers(args.model_name)
        if len(selected_layers) == int(arch["num_layers"])
        else [hook for layer in selected_layers for hook in tc_hook_points_for_layer(layer)]
    )
    print(f"[INFO] selected_layers={selected_layers}")
    print(f"[INFO] hook_points={hook_points}")

    settings = GenerateActivationsSettings(
        model=LanguageModelConfig(
            model_name=args.model_name,
            device=args.device,
            dtype=args.dtype,
            backend=args.backend,
            prepend_bos=False,
            max_length=args.context_size,
            activation_device=args.activation_device,
            random_crop_to_max_length=args.crop_mode == "random",
            random_crop_seed=args.crop_seed,
        ),
        model_name=args.model_name,
        dataset=DatasetConfig(
            dataset_name_or_path=args.dataset_path,
            is_dataset_on_disk=True,
        ),
        dataset_name=args.dataset_name,
        hook_points=hook_points,
        output_dir=args.output_dir,
        total_tokens=args.total_tokens,
        context_size=args.context_size,
        n_samples_per_chunk=None,
        model_batch_size=args.model_batch_size,
        target=ActivationFactoryTarget.ACTIVATIONS_1D,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        buffer_shuffle=BufferShuffleConfig(
            perm_seed=42,
            generator_device=args.activation_device,
        ),
        num_workers=args.num_workers,
        device_type="cpu" if args.device == "cpu" else "cuda",
    )
    generate_activations(settings)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
