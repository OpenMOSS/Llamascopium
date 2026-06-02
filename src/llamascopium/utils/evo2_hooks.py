from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from evo2.utils import CONFIG_MAP, HF_MODEL_NAME_MAP

Evo2HookKind = Literal["lorsa", "sconv"]

REPO_ROOT = Path(__file__).resolve().parents[3]

HF_TO_INTERNAL_MODEL_NAME = {hf_name: internal_name for internal_name, hf_name in HF_MODEL_NAME_MAP.items()}


def normalize_evo2_model_name(model_name: str) -> str:
    if model_name in CONFIG_MAP:
        return model_name
    if model_name in HF_TO_INTERNAL_MODEL_NAME:
        return HF_TO_INTERNAL_MODEL_NAME[model_name]
    raise KeyError(model_name)


def load_evo2_config(model_name: str = "evo2_7b") -> dict:
    config_relpath = CONFIG_MAP[normalize_evo2_model_name(model_name)]
    config_path = REPO_ROOT / "third_party" / "evo2" / "evo2" / config_relpath
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def get_evo2_arch(model_name: str = "evo2_7b") -> dict[str, object]:
    cfg = load_evo2_config(model_name)
    hidden_size = int(cfg["hidden_size"])
    num_layers = int(cfg["num_layers"])
    num_attention_heads = int(cfg["num_attention_heads"])
    attn_layers = {int(layer) for layer in cfg["attn_layer_idxs"]}
    conv_layers = set(range(num_layers)) - attn_layers
    return {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_attention_heads": num_attention_heads,
        "d_qk_head": hidden_size // num_attention_heads,
        "rotary_dim": hidden_size // num_attention_heads,
        "attn_layers": attn_layers,
        "conv_layers": conv_layers,
    }


def validate_layer(layer: int, num_layers: int) -> None:
    if not 0 <= layer < num_layers:
        raise ValueError(f"Invalid layer {layer}. Expected 0 <= layer < {num_layers}.")


def tc_hook_points_for_layer(layer: int) -> list[str]:
    return [
        f"blocks.{layer}.hook_mlp_in",
        f"blocks.{layer}.hook_mlp_out",
    ]


def tc_hook_points_for_all_layers(model_name: str = "evo2_7b") -> list[str]:
    arch = get_evo2_arch(model_name)
    return [hook for layer in range(int(arch["num_layers"])) for hook in tc_hook_points_for_layer(layer)]


def is_lorsa_layer(layer: int, model_name: str = "evo2_7b") -> bool:
    return layer in get_evo2_arch(model_name)["attn_layers"]


def gen_lorsa_hook_points_for_layer(layer: int) -> list[str]:
    return [
        f"blocks.{layer}.hook_attn_in",
        f"blocks.{layer}.hook_attn_out",
    ]


def gen_sconv_hook_points_for_layer(layer: int) -> list[str]:
    return [
        f"blocks.{layer}.hook_conv_in",
        f"blocks.{layer}.hook_fir_out",
        f"blocks.{layer}.hook_filter_out",
        f"blocks.{layer}.hook_conv_out",
    ]


def gen_evo2_hook_points_for_layer(layer: int, model_name: str = "evo2_7b") -> tuple[list[str], Evo2HookKind]:
    if is_lorsa_layer(layer, model_name):
        return gen_lorsa_hook_points_for_layer(layer), "lorsa"
    return gen_sconv_hook_points_for_layer(layer), "sconv"


def gen_evo2_layer_hook_points(layer: int, model_name: str = "evo2_7b") -> tuple[list[str], Evo2HookKind, str]:
    hook_points, kind = gen_evo2_hook_points_for_layer(layer, model_name)
    return hook_points, kind, ("attn" if kind == "lorsa" else "conv")


def gen_tc_hook_points_for_layer(layer: int) -> list[str]:
    return tc_hook_points_for_layer(layer)


def gen_tc_hook_points(model_name: str = "evo2_7b") -> list[str]:
    arch = get_evo2_arch(model_name)
    return [hook for layer in range(int(arch["num_layers"])) for hook in gen_tc_hook_points_for_layer(layer)]


def gen_lorsa_hook_points(model_name: str = "evo2_7b") -> list[str]:
    arch = get_evo2_arch(model_name)
    return [hook for layer in sorted(arch["attn_layers"]) for hook in gen_lorsa_hook_points_for_layer(layer)]


def gen_sconv_hook_points(model_name: str = "evo2_7b") -> list[str]:
    arch = get_evo2_arch(model_name)
    return [hook for layer in sorted(arch["conv_layers"]) for hook in gen_sconv_hook_points_for_layer(layer)]


def default_activation_dir(kind: str) -> Path:
    return REPO_ROOT / "artifacts" / "evo2_7b" / "activations" / kind


def default_result_dir(kind: str) -> Path:
    return REPO_ROOT / "artifacts" / "evo2_7b" / "results" / kind
