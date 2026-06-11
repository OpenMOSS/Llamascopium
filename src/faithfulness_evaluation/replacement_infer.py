from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections.abc import Iterator, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformer_lens import HookedTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from ..chess_utils import (
    get_move_from_policy_output,
    get_move_from_policy_output_with_prob,
)
from ..lm_saes import LowRankSparseAttention, SparseAutoEncoder
from ..lm_saes.config import LorsaConfig, SAEConfig

ComponentKind = Literal["mlp", "attn"]

DEFAULT_MODEL_NAME = "lc0/BT4-1024x15x32h"
DEFAULT_TC_ROOT = PROJECT_ROOT / "result_BT4" / "tc" / "k_30_e_16"
DEFAULT_LORSA_ROOT = PROJECT_ROOT / "result_BT4" / "lorsa" / "k_30_e_16"
DEFAULT_PROMPT_DATASET_PATH = Path(
    "/inspire/hdd/global_user/hezhengfu-240208120186/data/rlin_data/Chess/chess_master_data"
)
DEFAULT_MEAN_ACTIVATION_ROOT = PROJECT_ROOT / "activations" / "BT4"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "exp" / "53ICMLnew" / "12replacement_model_infer" / "results"
DEFAULT_NUM_LAYERS = 15

COMPONENT_SPECS: dict[ComponentKind, dict[str, str]] = {
    "mlp": {
        "input_hook_suffix": "resid_mid_after_ln",
        "output_hook_suffix": "hook_mlp_out",
        "mean_subdir": "mlp_out_mean",
    },
    "attn": {
        "input_hook_suffix": "hook_attn_in",
        "output_hook_suffix": "hook_attn_out",
        "mean_subdir": "attn_out_mean",
    },
}

__all__ = [
    "DEFAULT_LORSA_ROOT",
    "DEFAULT_MEAN_ACTIVATION_ROOT",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_NUM_LAYERS",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_PROMPT_DATASET_PATH",
    "DEFAULT_TC_ROOT",
    "COMPONENT_SPECS",
    "aggregate_metric",
    "build_mean_ablation_hooks",
    "build_single_layer_replacement_hooks",
    "compute_batch_distribution_metrics",
    "compute_legal_distribution_metrics",
    "evaluate_replacement_vs_mean_ablation",
    "extract_policy_logits",
    "load_bt4_model",
    "load_lorsa_for_layer",
    "load_mean_activation",
    "load_transcoder_for_layer",
    "parse_dtype",
    "predict_top_legal_move",
    "resolve_device",
    "run_policy_logits",
    "sample_fens_from_dataset",
    "write_summary_text",
    "chunked",
]


def resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def parse_dtype(name: str) -> torch.dtype:
    name = name.lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def chunked(items: Sequence[str], chunk_size: int) -> Iterator[list[str]]:
    for start in range(0, len(items), chunk_size):
        yield list(items[start : start + chunk_size])


def load_bt4_model(
    model_name: str = DEFAULT_MODEL_NAME,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> HookedTransformer:
    model = HookedTransformer.from_pretrained_no_processing(
        model_name,
        dtype=dtype,
    ).eval()
    return model.to(resolve_device(device))


def load_transcoder_for_layer(
    layer: int,
    *,
    tc_root: str | Path = DEFAULT_TC_ROOT,
    checkpoint_subpath: str | Path | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    strict_loading: bool = True,
) -> SparseAutoEncoder:
    layer_dir = Path(tc_root) / f"L{layer}"
    if checkpoint_subpath is None:
        transcoder = SparseAutoEncoder.from_pretrained(
            str(layer_dir),
            device=resolve_device(device),
            dtype=dtype,
            strict_loading=strict_loading,
        )
        return transcoder.eval()

    pretrained_path = _resolve_layer_checkpoint_path(layer_dir, checkpoint_subpath)
    cfg = SAEConfig.from_pretrained(
        str(layer_dir),
        device=resolve_device(device),
        dtype=dtype,
        strict_loading=strict_loading,
    )
    cfg.sae_pretrained_name_or_path = str(pretrained_path)
    transcoder = SparseAutoEncoder.from_config(cfg)
    return transcoder.eval()


def load_lorsa_for_layer(
    layer: int,
    *,
    lorsa_root: str | Path = DEFAULT_LORSA_ROOT,
    checkpoint_subpath: str | Path | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    strict_loading: bool = True,
) -> LowRankSparseAttention:
    layer_dir = Path(lorsa_root) / f"L{layer}"
    if checkpoint_subpath is None:
        lorsa = LowRankSparseAttention.from_pretrained(
            str(layer_dir),
            device=resolve_device(device),
            dtype=dtype,
            strict_loading=strict_loading,
        )
        return lorsa.eval()

    pretrained_path = _resolve_layer_checkpoint_path(layer_dir, checkpoint_subpath)
    cfg = LorsaConfig.from_pretrained(
        str(layer_dir),
        device=resolve_device(device),
        dtype=dtype,
        strict_loading=strict_loading,
    )
    cfg.sae_pretrained_name_or_path = str(pretrained_path)
    lorsa = LowRankSparseAttention.from_config(cfg)
    return lorsa.eval()


def _resolve_layer_checkpoint_path(layer_dir: Path, checkpoint_subpath: str | Path) -> Path:
    checkpoint_subpath = Path(checkpoint_subpath)
    candidates: list[Path] = []
    if checkpoint_subpath.is_absolute():
        candidates.append(checkpoint_subpath)
    else:
        candidates.append(layer_dir / checkpoint_subpath)
        if checkpoint_subpath.parts[0] != "checkpoints":
            candidates.append(layer_dir / "checkpoints" / checkpoint_subpath)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Checkpoint path `{checkpoint_subpath}` not found for layer dir {layer_dir}. Tried: {tried}"
    )


def load_mean_activation(
    layer: int,
    component: ComponentKind,
    *,
    mean_activation_root: str | Path = DEFAULT_MEAN_ACTIVATION_ROOT,
) -> torch.Tensor:
    spec = COMPONENT_SPECS[component]
    hook_name = f"blocks.{layer}.{spec['output_hook_suffix']}"
    safe_name = hook_name.replace(".", "_")
    file_path = Path(mean_activation_root) / spec["mean_subdir"] / f"{safe_name}_mean.safetensors"
    if not file_path.exists():
        raise FileNotFoundError(f"Mean activation file not found: {file_path}")
    data = load_file(str(file_path))
    return data["mean"]


def build_single_layer_replacement_hooks(
    *,
    layer: int,
    component: ComponentKind,
    transcoder: SparseAutoEncoder | None = None,
    lorsa: LowRankSparseAttention | None = None,
    zero_bos: bool = False,
    use_hidden_pre: bool = False,
) -> list[tuple[str, Any]]:
    spec = COMPONENT_SPECS[component]
    input_hook = f"blocks.{layer}.{spec['input_hook_suffix']}"
    output_hook = f"blocks.{layer}.{spec['output_hook_suffix']}"
    captured: dict[str, torch.Tensor] = {}

    def _capture(activation: torch.Tensor, hook: Any) -> torch.Tensor:
        captured["input"] = activation
        return activation

    def _replace(_activation: torch.Tensor, hook: Any) -> torch.Tensor:
        source = captured["input"]
        if component == "mlp":
            if transcoder is None:
                raise ValueError("transcoder is required for mlp replacement")
            encoded = (
                transcoder.encode(source, return_hidden_pre=True)[1]
                if use_hidden_pre
                else transcoder.encode(source)
            )
            if zero_bos and encoded.ndim >= 2:
                encoded[:, 0] = 0
            return transcoder.decode(encoded)

        if lorsa is None:
            raise ValueError("lorsa is required for attn replacement")
        encoded = lorsa.encode(source, return_hidden_pre=True)[1] if use_hidden_pre else lorsa.encode(source)
        if zero_bos and encoded.ndim >= 2:
            encoded[:, 0] = 0
        return lorsa.decode(encoded)

    return [(input_hook, _capture), (output_hook, _replace)]


def build_mean_ablation_hooks(
    *,
    layer: int,
    component: ComponentKind,
    mean_activation: torch.Tensor,
) -> list[tuple[str, Any]]:
    output_hook = f"blocks.{layer}.{COMPONENT_SPECS[component]['output_hook_suffix']}"

    def _replace(activation: torch.Tensor, hook: Any) -> torch.Tensor:
        mean = mean_activation.to(device=activation.device, dtype=activation.dtype)
        view_shape = (1,) * (activation.ndim - mean.ndim) + tuple(mean.shape)
        return mean.view(*view_shape).expand_as(activation)

    return [(output_hook, _replace)]


def extract_policy_logits(model_output: Any) -> torch.Tensor:
    output = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
    if output.ndim == 1:
        return output.unsqueeze(0)
    if output.ndim == 2:
        return output
    if output.ndim == 3:
        return output[:, -1, :]
    raise RuntimeError(f"Unexpected output shape: {tuple(output.shape)}")


def _forward_policy_logits(model: HookedTransformer, prompts: Sequence[str]) -> torch.Tensor:
    output = model(list(prompts), prepend_bos=False)
    return extract_policy_logits(output)


def run_policy_logits(
    model: HookedTransformer,
    prompts: Sequence[str],
    *,
    fwd_hooks: list[tuple[str, Any]] | None = None,
) -> torch.Tensor:
    prompts = list(prompts)
    hook_context = model.hooks(fwd_hooks=fwd_hooks) if fwd_hooks else nullcontext()
    with torch.inference_mode():
        with hook_context:
            try:
                return _forward_policy_logits(model, prompts)
            except Exception:
                rows: list[torch.Tensor] = []
                for prompt in prompts:
                    output = model(prompt, prepend_bos=False)
                    rows.append(extract_policy_logits(output)[0])
                return torch.stack(rows, dim=0)


def _sample_fens_with_datasets(dataset_path: Path, sample_size: int, seed: int) -> list[str]:
    from datasets import DatasetDict, load_from_disk

    dataset = load_from_disk(str(dataset_path))
    if isinstance(dataset, DatasetDict):
        dataset = dataset["train"] if "train" in dataset else dataset[next(iter(dataset.keys()))]

    total = len(dataset)
    if total <= sample_size:
        selected = dataset["fen"]
    else:
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(total), sample_size))
        selected = dataset.select(indices)["fen"]
    return [fen.strip() for fen in selected if isinstance(fen, str) and fen.strip()]


def _iter_arrow_column(file_path: Path, column_name: str) -> Iterator[str]:
    import pyarrow as pa
    import pyarrow.ipc as ipc

    with pa.memory_map(str(file_path), "r") as source:
        try:
            reader = ipc.open_file(source)
            batches = (reader.get_batch(i) for i in range(reader.num_record_batches))
        except pa.ArrowInvalid:
            reader = ipc.open_stream(source)
            batches = reader
        for batch in batches:
            col_idx = batch.schema.get_field_index(column_name)
            if col_idx < 0:
                raise KeyError(f"Column `{column_name}` not found in {file_path}")
            for value in batch.column(col_idx).to_pylist():
                if isinstance(value, str) and value.strip():
                    yield value.strip()


def _sample_fens_with_arrow(dataset_path: Path, sample_size: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    reservoir: list[str] = []
    seen = 0
    for file_path in sorted(dataset_path.glob("data-*.arrow")):
        for fen in _iter_arrow_column(file_path, "fen"):
            seen += 1
            if len(reservoir) < sample_size:
                reservoir.append(fen)
            else:
                idx = rng.randrange(seen)
                if idx < sample_size:
                    reservoir[idx] = fen
    return reservoir


def sample_fens_from_dataset(
    dataset_path: str | Path = DEFAULT_PROMPT_DATASET_PATH,
    *,
    sample_size: int = 10_000,
    seed: int = 42,
) -> list[str]:
    dataset_path = Path(dataset_path)
    try:
        return _sample_fens_with_datasets(dataset_path, sample_size, seed)
    except Exception:
        return _sample_fens_with_arrow(dataset_path, sample_size, seed)


def _kl_divergence_from_log_probs(
    p_probs: torch.Tensor,
    p_log_probs: torch.Tensor,
    q_log_probs: torch.Tensor,
) -> torch.Tensor:
    return torch.sum(p_probs * (p_log_probs - q_log_probs), dim=-1)


def compute_batch_distribution_metrics(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
) -> dict[str, torch.Tensor]:
    ref_log_probs = F.log_softmax(reference_logits.to(torch.float64), dim=-1)
    cand_log_probs = F.log_softmax(candidate_logits.to(torch.float64), dim=-1)
    ref_probs = ref_log_probs.exp()
    cand_probs = cand_log_probs.exp()

    kl_ref_to_cand = _kl_divergence_from_log_probs(ref_probs, ref_log_probs, cand_log_probs)
    kl_cand_to_ref = _kl_divergence_from_log_probs(cand_probs, cand_log_probs, ref_log_probs)

    mixed_probs = (0.5 * (ref_probs + cand_probs)).clamp_min(1e-12)
    mixed_log_probs = mixed_probs.log()
    jsd = 0.5 * (
        _kl_divergence_from_log_probs(ref_probs, ref_log_probs, mixed_log_probs)
        + _kl_divergence_from_log_probs(cand_probs, cand_log_probs, mixed_log_probs)
    )

    return {
        "kl_ref_to_candidate": kl_ref_to_cand,
        "kl_candidate_to_ref": kl_cand_to_ref,
        "jsd": jsd,
    }


def _kl_divergence_dict(p_dict: dict[str, float], q_dict: dict[str, float]) -> float | None:
    eps = 1e-12
    support = set(p_dict) | set(q_dict)
    if not support:
        return None
    value = 0.0
    for move in support:
        p_val = max(p_dict.get(move, eps), eps)
        q_val = max(q_dict.get(move, eps), eps)
        value += p_val * math.log(p_val / q_val)
    return float(value)


def _js_divergence_dict(p_dict: dict[str, float], q_dict: dict[str, float]) -> float | None:
    eps = 1e-12
    support = set(p_dict) | set(q_dict)
    if not support:
        return None
    mixed = {
        move: 0.5 * (max(p_dict.get(move, eps), eps) + max(q_dict.get(move, eps), eps))
        for move in support
    }
    kl_pm = _kl_divergence_dict(p_dict, mixed)
    kl_qm = _kl_divergence_dict(q_dict, mixed)
    if kl_pm is None or kl_qm is None:
        return None
    return float(0.5 * (kl_pm + kl_qm))


def _legal_move_entries(
    logits_row: torch.Tensor,
    fen: str,
) -> tuple[list[tuple[str, float, float]], str | None]:
    result = get_move_from_policy_output_with_prob(
        logits_row.detach().cpu().unsqueeze(0),
        fen,
        return_list=True,
    )
    if not isinstance(result, list) or not result:
        return [], None
    return [(uci, float(logit), float(prob)) for uci, logit, prob in result], result[0][0]


def _legal_move_prob_dict(logits_row: torch.Tensor, fen: str) -> tuple[dict[str, float], str | None]:
    entries, top_move = _legal_move_entries(logits_row, fen)
    return {uci: prob for uci, _logit, prob in entries}, top_move


def _legal_move_logit_dict(logits_row: torch.Tensor, fen: str) -> tuple[dict[str, float], str | None]:
    entries, top_move = _legal_move_entries(logits_row, fen)
    return {uci: logit for uci, logit, _prob in entries}, top_move


def _top1_top2_margin_dict(score_dict: dict[str, float]) -> float | None:
    if len(score_dict) < 2:
        return None
    top_two = sorted(score_dict.values(), reverse=True)[:2]
    return float(top_two[0] - top_two[1])


def _rank_desc(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1], reverse=True)
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(indexed):
        end = pos + 1
        while end < len(indexed) and indexed[end][1] == indexed[pos][1]:
            end += 1
        avg_rank = float((pos + 1 + end) / 2.0)
        for idx, _ in indexed[pos:end]:
            ranks[idx] = avg_rank
        pos = end
    return ranks


def _pearson_corr(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    centered_left = [value - mean_left for value in left]
    centered_right = [value - mean_right for value in right]
    denom_left = math.sqrt(sum(value * value for value in centered_left))
    denom_right = math.sqrt(sum(value * value for value in centered_right))
    if denom_left == 0.0 or denom_right == 0.0:
        return None
    numer = sum(l * r for l, r in zip(centered_left, centered_right, strict=False))
    return float(numer / (denom_left * denom_right))


def _spearman_rank_corr_dict(left_dict: dict[str, float], right_dict: dict[str, float]) -> float | None:
    support = sorted(set(left_dict) & set(right_dict))
    if len(support) < 2:
        return None
    left_values = [left_dict[key] for key in support]
    right_values = [right_dict[key] for key in support]
    return _pearson_corr(_rank_desc(left_values), _rank_desc(right_values))


def compute_legal_distribution_metrics(
    reference_logits_row: torch.Tensor,
    candidate_logits_row: torch.Tensor,
    fen: str,
) -> dict[str, float | None]:
    ref_dict, _ = _legal_move_prob_dict(reference_logits_row, fen)
    cand_dict, _ = _legal_move_prob_dict(candidate_logits_row, fen)
    ref_logit_dict, _ = _legal_move_logit_dict(reference_logits_row, fen)
    cand_logit_dict, _ = _legal_move_logit_dict(candidate_logits_row, fen)
    ref_margin = _top1_top2_margin_dict(ref_logit_dict)
    cand_margin = _top1_top2_margin_dict(cand_logit_dict)
    return {
        "kl_ref_to_candidate": _kl_divergence_dict(ref_dict, cand_dict),
        "kl_candidate_to_ref": _kl_divergence_dict(cand_dict, ref_dict),
        "jsd": _js_divergence_dict(ref_dict, cand_dict),
        "reference_top1_logit_margin": ref_margin,
        "candidate_top1_logit_margin": cand_margin,
        "top1_logit_margin_delta": None
        if ref_margin is None or cand_margin is None
        else float(cand_margin - ref_margin),
        "top1_logit_margin_abs_delta": None
        if ref_margin is None or cand_margin is None
        else float(abs(cand_margin - ref_margin)),
        "logit_rank_spearman": _spearman_rank_corr_dict(ref_logit_dict, cand_logit_dict),
    }


def predict_top_legal_move(logits_row: torch.Tensor, fen: str) -> str | None:
    try:
        return str(get_move_from_policy_output(logits_row.detach().cpu().unsqueeze(0), fen))
    except Exception:
        return None


def aggregate_metric(values: Sequence[float | None]) -> dict[str, float | int | None]:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return {"count": 0, "mean": None, "std": None, "median": None, "min": None, "max": None}
    return {
        "count": len(filtered),
        "mean": float(statistics.fmean(filtered)),
        "std": float(statistics.pstdev(filtered)) if len(filtered) > 1 else 0.0,
        "median": float(statistics.median(filtered)),
        "min": float(min(filtered)),
        "max": float(max(filtered)),
    }


def _paired_better_rate(left: Sequence[float | None], right: Sequence[float | None]) -> float | None:
    valid = [(l, r) for l, r in zip(left, right, strict=False) if l is not None and r is not None]
    if not valid:
        return None
    better = sum(1 for l, r in valid if l < r)
    return float(better / len(valid))


def _paired_higher_rate(left: Sequence[float | None], right: Sequence[float | None]) -> float | None:
    valid = [(l, r) for l, r in zip(left, right, strict=False) if l is not None and r is not None]
    if not valid:
        return None
    better = sum(1 for l, r in valid if l > r)
    return float(better / len(valid))


def _mean_bool(values: Sequence[bool]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def evaluate_replacement_vs_mean_ablation(
    *,
    model: HookedTransformer,
    transcoder: SparseAutoEncoder,
    lorsa: LowRankSparseAttention,
    prompts: Sequence[str],
    layer: int,
    component: ComponentKind,
    mean_activation_root: str | Path = DEFAULT_MEAN_ACTIVATION_ROOT,
    batch_size: int = 64,
    progress_every: int = 20,
    zero_bos: bool = False,
    use_hidden_pre: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    replacement_hooks = build_single_layer_replacement_hooks(
        layer=layer,
        component=component,
        transcoder=transcoder,
        lorsa=lorsa,
        zero_bos=zero_bos,
        use_hidden_pre=use_hidden_pre,
    )
    mean_activation = load_mean_activation(
        layer,
        component,
        mean_activation_root=mean_activation_root,
    ).to(model.cfg.device, model.cfg.dtype)
    mean_hooks = build_mean_ablation_hooks(
        layer=layer,
        component=component,
        mean_activation=mean_activation,
    )

    rows: list[dict[str, Any]] = []
    replacement_full_kl: list[float] = []
    replacement_full_jsd: list[float] = []
    mean_full_kl: list[float] = []
    mean_full_jsd: list[float] = []
    replacement_legal_kl: list[float | None] = []
    replacement_legal_jsd: list[float | None] = []
    replacement_legal_ref_margin: list[float | None] = []
    replacement_legal_candidate_margin: list[float | None] = []
    replacement_legal_margin_abs_delta: list[float | None] = []
    replacement_legal_rank_corr: list[float | None] = []
    mean_legal_kl: list[float | None] = []
    mean_legal_jsd: list[float | None] = []
    mean_legal_ref_margin: list[float | None] = []
    mean_legal_candidate_margin: list[float | None] = []
    mean_legal_margin_abs_delta: list[float | None] = []
    mean_legal_rank_corr: list[float | None] = []
    replacement_top_preserved: list[bool] = []
    mean_top_preserved: list[bool] = []

    batches = list(chunked(list(prompts), batch_size))
    for batch_idx, batch_prompts in enumerate(batches, start=1):
        original_logits = run_policy_logits(model, batch_prompts)
        replacement_logits = run_policy_logits(model, batch_prompts, fwd_hooks=replacement_hooks)
        mean_logits = run_policy_logits(model, batch_prompts, fwd_hooks=mean_hooks)

        replacement_full = compute_batch_distribution_metrics(original_logits, replacement_logits)
        mean_full = compute_batch_distribution_metrics(original_logits, mean_logits)

        for row_idx, fen in enumerate(batch_prompts):
            original_row = original_logits[row_idx]
            replacement_row = replacement_logits[row_idx]
            mean_row = mean_logits[row_idx]

            replacement_full_kl_value = float(replacement_full["kl_ref_to_candidate"][row_idx].item())
            replacement_full_jsd_value = float(replacement_full["jsd"][row_idx].item())
            mean_full_kl_value = float(mean_full["kl_ref_to_candidate"][row_idx].item())
            mean_full_jsd_value = float(mean_full["jsd"][row_idx].item())

            replacement_legal = compute_legal_distribution_metrics(original_row, replacement_row, fen)
            mean_legal = compute_legal_distribution_metrics(original_row, mean_row, fen)

            original_move = predict_top_legal_move(original_row, fen)
            replacement_move = predict_top_legal_move(replacement_row, fen)
            mean_move = predict_top_legal_move(mean_row, fen)
            replacement_agree = original_move is not None and replacement_move == original_move
            mean_agree = original_move is not None and mean_move == original_move

            replacement_full_kl.append(replacement_full_kl_value)
            replacement_full_jsd.append(replacement_full_jsd_value)
            mean_full_kl.append(mean_full_kl_value)
            mean_full_jsd.append(mean_full_jsd_value)
            replacement_legal_kl.append(replacement_legal["kl_ref_to_candidate"])
            replacement_legal_jsd.append(replacement_legal["jsd"])
            replacement_legal_ref_margin.append(replacement_legal["reference_top1_logit_margin"])
            replacement_legal_candidate_margin.append(replacement_legal["candidate_top1_logit_margin"])
            replacement_legal_margin_abs_delta.append(replacement_legal["top1_logit_margin_abs_delta"])
            replacement_legal_rank_corr.append(replacement_legal["logit_rank_spearman"])
            mean_legal_kl.append(mean_legal["kl_ref_to_candidate"])
            mean_legal_jsd.append(mean_legal["jsd"])
            mean_legal_ref_margin.append(mean_legal["reference_top1_logit_margin"])
            mean_legal_candidate_margin.append(mean_legal["candidate_top1_logit_margin"])
            mean_legal_margin_abs_delta.append(mean_legal["top1_logit_margin_abs_delta"])
            mean_legal_rank_corr.append(mean_legal["logit_rank_spearman"])
            replacement_top_preserved.append(replacement_agree)
            mean_top_preserved.append(mean_agree)

            rows.append(
                {
                    "sample_idx": len(rows),
                    "fen": fen,
                    "original_move": original_move,
                    "replacement_move": replacement_move,
                    "mean_ablation_move": mean_move,
                    "replacement_agrees_with_original": replacement_agree,
                    "mean_ablation_agrees_with_original": mean_agree,
                    "replacement_full_kl_ref_to_candidate": replacement_full_kl_value,
                    "replacement_full_jsd": replacement_full_jsd_value,
                    "mean_ablation_full_kl_ref_to_candidate": mean_full_kl_value,
                    "mean_ablation_full_jsd": mean_full_jsd_value,
                    "replacement_legal_kl_ref_to_candidate": replacement_legal["kl_ref_to_candidate"],
                    "replacement_legal_jsd": replacement_legal["jsd"],
                    "original_legal_top1_logit_margin": replacement_legal["reference_top1_logit_margin"],
                    "replacement_legal_top1_logit_margin": replacement_legal["candidate_top1_logit_margin"],
                    "replacement_legal_top1_logit_margin_delta": replacement_legal["top1_logit_margin_delta"],
                    "replacement_legal_top1_logit_margin_abs_delta": replacement_legal[
                        "top1_logit_margin_abs_delta"
                    ],
                    "replacement_legal_logit_rank_spearman": replacement_legal["logit_rank_spearman"],
                    "mean_ablation_legal_kl_ref_to_candidate": mean_legal["kl_ref_to_candidate"],
                    "mean_ablation_legal_jsd": mean_legal["jsd"],
                    "mean_ablation_legal_top1_logit_margin": mean_legal["candidate_top1_logit_margin"],
                    "mean_ablation_legal_top1_logit_margin_delta": mean_legal["top1_logit_margin_delta"],
                    "mean_ablation_legal_top1_logit_margin_abs_delta": mean_legal[
                        "top1_logit_margin_abs_delta"
                    ],
                    "mean_ablation_legal_logit_rank_spearman": mean_legal["logit_rank_spearman"],
                }
            )

        if progress_every > 0 and (
            batch_idx % progress_every == 0 or batch_idx == len(batches)
        ):
            print(
                f"[replacement_infer] processed {batch_idx}/{len(batches)} batches "
                f"({len(rows)}/{len(prompts)} prompts)"
            )

    summary = {
        "layer": layer,
        "component": component,
        "num_prompts": len(prompts),
        "replacement_vs_original": {
            "full_distribution": {
                "kl_ref_to_candidate": aggregate_metric(replacement_full_kl),
                "jsd": aggregate_metric(replacement_full_jsd),
            },
            "legal_distribution": {
                "kl_ref_to_candidate": aggregate_metric(replacement_legal_kl),
                "jsd": aggregate_metric(replacement_legal_jsd),
                "reference_top1_logit_margin": aggregate_metric(replacement_legal_ref_margin),
                "candidate_top1_logit_margin": aggregate_metric(replacement_legal_candidate_margin),
                "top1_logit_margin_abs_delta": aggregate_metric(replacement_legal_margin_abs_delta),
                "logit_rank_spearman": aggregate_metric(replacement_legal_rank_corr),
            },
            "top1_agreement_rate": _mean_bool(replacement_top_preserved),
        },
        "mean_ablation_vs_original": {
            "full_distribution": {
                "kl_ref_to_candidate": aggregate_metric(mean_full_kl),
                "jsd": aggregate_metric(mean_full_jsd),
            },
            "legal_distribution": {
                "kl_ref_to_candidate": aggregate_metric(mean_legal_kl),
                "jsd": aggregate_metric(mean_legal_jsd),
                "reference_top1_logit_margin": aggregate_metric(mean_legal_ref_margin),
                "candidate_top1_logit_margin": aggregate_metric(mean_legal_candidate_margin),
                "top1_logit_margin_abs_delta": aggregate_metric(mean_legal_margin_abs_delta),
                "logit_rank_spearman": aggregate_metric(mean_legal_rank_corr),
            },
            "top1_agreement_rate": _mean_bool(mean_top_preserved),
        },
        "replacement_better_than_mean_ablation": {
            "full_kl_rate": _paired_better_rate(replacement_full_kl, mean_full_kl),
            "full_jsd_rate": _paired_better_rate(replacement_full_jsd, mean_full_jsd),
            "legal_kl_rate": _paired_better_rate(replacement_legal_kl, mean_legal_kl),
            "legal_jsd_rate": _paired_better_rate(replacement_legal_jsd, mean_legal_jsd),
            "legal_top1_logit_margin_abs_delta_rate": _paired_better_rate(
                replacement_legal_margin_abs_delta, mean_legal_margin_abs_delta
            ),
            "legal_logit_rank_spearman_rate": _paired_higher_rate(
                replacement_legal_rank_corr, mean_legal_rank_corr
            ),
        },
    }
    return rows, summary


def write_summary_text(summary: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)

    def _fmt(value: Any) -> str:
        if value is None:
            return "None"
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    lines = [
        f"layer: {summary['layer']}",
        f"component: {summary['component']}",
        f"num_prompts: {summary['num_prompts']}",
        "",
        "[replacement_vs_original.full_distribution]",
    ]
    for metric_name, values in summary["replacement_vs_original"]["full_distribution"].items():
        lines.append(f"{metric_name}: mean={_fmt(values['mean'])} std={_fmt(values['std'])}")
    lines.append(
        f"top1_agreement_rate: {_fmt(summary['replacement_vs_original']['top1_agreement_rate'])}"
    )
    lines.extend(["", "[mean_ablation_vs_original.full_distribution]"])
    for metric_name, values in summary["mean_ablation_vs_original"]["full_distribution"].items():
        lines.append(f"{metric_name}: mean={_fmt(values['mean'])} std={_fmt(values['std'])}")
    lines.append(
        f"top1_agreement_rate: {_fmt(summary['mean_ablation_vs_original']['top1_agreement_rate'])}"
    )
    lines.extend(["", "[replacement_vs_original.legal_distribution]"])
    for metric_name, values in summary["replacement_vs_original"]["legal_distribution"].items():
        lines.append(f"{metric_name}: mean={_fmt(values['mean'])} std={_fmt(values['std'])}")
    lines.append(
        f"top1_agreement_rate: {_fmt(summary['replacement_vs_original']['top1_agreement_rate'])}"
    )
    lines.extend(["", "[mean_ablation_vs_original.legal_distribution]"])
    for metric_name, values in summary["mean_ablation_vs_original"]["legal_distribution"].items():
        lines.append(f"{metric_name}: mean={_fmt(values['mean'])} std={_fmt(values['std'])}")
    lines.append(
        f"top1_agreement_rate: {_fmt(summary['mean_ablation_vs_original']['top1_agreement_rate'])}"
    )
    lines.extend(["", "[replacement_better_than_mean_ablation]"])
    for metric_name, value in summary["replacement_better_than_mean_ablation"].items():
        lines.append(f"{metric_name}: {_fmt(value)}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate single-layer replacement fidelity vs mean ablation on a BT4 FEN dataset."
    )
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--component", choices=("mlp", "attn"), required=True)
    parser.add_argument("--sample-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--tc-root", type=Path, default=DEFAULT_TC_ROOT)
    parser.add_argument("--tc-checkpoint-subpath", type=str, default=None)
    parser.add_argument("--lorsa-root", type=Path, default=DEFAULT_LORSA_ROOT)
    parser.add_argument("--lorsa-checkpoint-subpath", type=str, default=None)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_PROMPT_DATASET_PATH)
    parser.add_argument("--mean-activation-root", type=Path, default=DEFAULT_MEAN_ACTIVATION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--zero-bos", action="store_true")
    parser.add_argument("--use-hidden-pre", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.layer < 0 or args.layer >= DEFAULT_NUM_LAYERS:
        raise ValueError(f"layer must be in [0, {DEFAULT_NUM_LAYERS - 1}]")

    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype)
    output_dir = args.output_root / f"layer{args.layer}_{args.component}_n{args.sample_size}_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[replacement_infer] loading model + layer modules for layer={args.layer} "
        f"component={args.component} on {device}"
    )
    model = load_bt4_model(args.model_name, device=device, dtype=dtype)
    transcoder = load_transcoder_for_layer(
        args.layer,
        tc_root=args.tc_root,
        checkpoint_subpath=args.tc_checkpoint_subpath,
        device=device,
        dtype=dtype,
    )
    lorsa = load_lorsa_for_layer(
        args.layer,
        lorsa_root=args.lorsa_root,
        checkpoint_subpath=args.lorsa_checkpoint_subpath,
        device=device,
        dtype=dtype,
    )

    prompts = sample_fens_from_dataset(
        args.dataset_path,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    if len(prompts) < args.sample_size:
        print(
            f"[replacement_infer] warning: sampled {len(prompts)} prompts, "
            f"less than requested {args.sample_size}"
        )

    rows, summary = evaluate_replacement_vs_mean_ablation(
        model=model,
        transcoder=transcoder,
        lorsa=lorsa,
        prompts=prompts,
        layer=args.layer,
        component=args.component,
        mean_activation_root=args.mean_activation_root,
        batch_size=args.batch_size,
        progress_every=args.progress_every,
        zero_bos=args.zero_bos,
        use_hidden_pre=args.use_hidden_pre,
    )
    summary["config"] = {
        "model_name": args.model_name,
        "layer": args.layer,
        "component": args.component,
        "sample_size": args.sample_size,
        "actual_sample_size": len(prompts),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": device,
        "dtype": args.dtype,
        "dataset_path": str(args.dataset_path),
        "tc_root": str(args.tc_root),
        "tc_checkpoint_subpath": args.tc_checkpoint_subpath,
        "lorsa_root": str(args.lorsa_root),
        "lorsa_checkpoint_subpath": args.lorsa_checkpoint_subpath,
        "mean_activation_root": str(args.mean_activation_root),
        "zero_bos": args.zero_bos,
        "use_hidden_pre": args.use_hidden_pre,
    }

    (output_dir / "sampled_fens.txt").write_text("\n".join(prompts) + "\n", encoding="utf-8")
    _write_csv(rows, output_dir / "per_prompt_metrics.csv")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary_text(summary, output_dir / "summary.txt")
    print(f"[replacement_infer] finished. results saved to {output_dir}")


if __name__ == "__main__":
    main()
