from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import chess
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "server"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from server.constants import BT4_MODEL_NAME, get_bt4_sae_combo
from server.circuit_trace_defaults import DEFAULT_VJP_BATCH_SIZE
from server.circuits_service import create_graph_from_attribution, load_model_and_transcoders, run_attribution
from lm_saes.circuit.graph_lc0 import Graph, compute_graph_scores
from lm_saes.circuit.leela_board import LeelaBoard
from src.chess_utils import get_feature_encoder_vector, get_feature_vector, get_move_from_model


DEFAULT_SAE_SERIES = "BT4-exp128"


@dataclass(frozen=True)
class EvalCase:
    fen: str
    move_uci: str | None = None
    negative_move_uci: str | None = None
    label: str | None = None


@dataclass(frozen=True)
class GraphFeature:
    node_index: int
    global_id: int
    feature_type: str
    layer: int
    position: int
    feature_idx: int
    activation_value: float


@dataclass
class BT4ModelBundle:
    model: Any
    transcoders: dict[int, Any]
    lorsas: list[Any]
    combo_id: str
    sae_series: str
    device: str


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify_case(case: EvalCase, index: int) -> str:
    prefix = case.label or f"case_{index:04d}"
    fen_hash = hashlib.md5(case.fen.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{fen_hash}"


def _clean_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_feature_type(value: str) -> str:
    value = value.lower()
    if value == "tc":
        return "transcoder"
    if value == "transcoder":
        return "transcoder"
    if value == "lorsa":
        return "lorsa"
    raise ValueError(f"Unsupported feature type: {value}")


def load_cases(path: str | Path, max_cases: int | None = None) -> list[EvalCase]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    cases: list[EvalCase] = []
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("cases", payload.get("items", []))
        if not isinstance(payload, list):
            raise ValueError(f"Unsupported JSON payload in {path}")
        rows = payload
    elif suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    elif suffix in {".txt", ".fen"}:
        rows = []
        for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split("\t")
            row: dict[str, Any] = {"fen": parts[0], "label": f"line_{idx:04d}"}
            if len(parts) > 1 and parts[1]:
                row["move_uci"] = parts[1]
            if len(parts) > 2 and parts[2]:
                row["negative_move_uci"] = parts[2]
            rows.append(row)
    else:
        raise ValueError(f"Unsupported case file format: {path}")

    for idx, row in enumerate(rows):
        fen = row.get("fen") or row.get("prompt")
        if not fen:
            continue
        cases.append(
            EvalCase(
                fen=str(fen).strip(),
                move_uci=_clean_optional_str(row.get("move_uci") or row.get("target_move") or row.get("move")),
                negative_move_uci=_clean_optional_str(
                    row.get("negative_move_uci") or row.get("negative_move")
                ),
                label=_clean_optional_str(row.get("label") or row.get("id") or row.get("name")) or f"case_{idx:04d}",
            )
        )
        if max_cases is not None and len(cases) >= max_cases:
            break
    return cases


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_jsonl(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_bt4_bundle(
    *,
    device: str,
    combo_id: str,
    sae_series: str = DEFAULT_SAE_SERIES,
    model_name: str = BT4_MODEL_NAME,
    n_layers: int = 15,
) -> BT4ModelBundle:
    combo = get_bt4_sae_combo(combo_id)
    replacement_model, transcoders, lorsas = load_model_and_transcoders(
        model_name=model_name,
        device=device,
        tc_base_path=combo["tc_base_path"],
        lorsa_base_path=combo["lorsa_base_path"],
        n_layers=n_layers,
        cache_key=f"{model_name}::{combo['id']}::{device}",
    )
    return BT4ModelBundle(
        model=replacement_model,
        transcoders=transcoders,
        lorsas=lorsas,
        combo_id=combo["id"],
        sae_series=sae_series,
        device=device,
    )


def resolve_case_move(bundle: BT4ModelBundle, case: EvalCase) -> EvalCase:
    if case.move_uci is not None:
        return case
    move_uci = get_move_from_model(bundle.model, case.fen)
    return EvalCase(
        fen=case.fen,
        move_uci=move_uci,
        negative_move_uci=case.negative_move_uci,
        label=case.label,
    )


def build_graph_for_case(
    *,
    bundle: BT4ModelBundle,
    case: EvalCase,
    side: str,
    slug: str,
    max_feature_nodes: int,
    max_n_logits: int = 1,
    desired_logit_prob: float = 0.95,
    batch_size: int = 1,
    vjp_batch_size: int = DEFAULT_VJP_BATCH_SIZE,
    order_mode: str = "positive",
    save_activation_info: bool = True,
) -> tuple[Graph, dict[str, Any], EvalCase]:
    resolved_case = resolve_case_move(bundle, case)
    attribution_result = run_attribution(
        model=bundle.model,
        prompt=resolved_case.fen,
        fen=resolved_case.fen,
        move_uci=resolved_case.move_uci,
        negative_move_uci=resolved_case.negative_move_uci,
        side=side,
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
        max_feature_nodes=max_feature_nodes,
        batch_size=batch_size,
        vjp_batch_size=vjp_batch_size,
        order_mode=order_mode,
        mongo_client=None,
        sae_series=bundle.sae_series,
        save_activation_info=save_activation_info,
    )
    graph = create_graph_from_attribution(
        model=bundle.model,
        attribution_result=attribution_result,
        prompt=resolved_case.fen,
        side=side,
        slug=slug,
        sae_series=bundle.sae_series,
    )
    return graph, attribution_result, resolved_case


def selected_graph_features(graph: Graph) -> list[GraphFeature]:
    selected = graph.selected_features.detach().cpu().tolist()
    activation_info = graph.activation_info or {}
    feature_entries = {int(item["featureId"]): item for item in activation_info.get("features", [])}
    lorsa_count = int(graph.lorsa_active_features.shape[0])

    features: list[GraphFeature] = []
    for node_index, global_id in enumerate(selected):
        entry = feature_entries.get(int(global_id))
        if entry is not None:
            features.append(
                GraphFeature(
                    node_index=node_index,
                    global_id=int(global_id),
                    feature_type=_normalize_feature_type(str(entry["type"])),
                    layer=int(entry["layer"]),
                    position=int(entry["position"]),
                    feature_idx=int(entry.get("feature_idx", entry.get("head_idx"))),
                    activation_value=float(entry["activation_value"]),
                )
            )
            continue

        if global_id < lorsa_count:
            layer, position, feature_idx = graph.lorsa_active_features[global_id].tolist()
            activation_value = float(graph.lorsa_activation_values[global_id].item())
            feature_type = "lorsa"
        else:
            local_idx = global_id - lorsa_count
            layer, position, feature_idx = graph.tc_active_features[local_idx].tolist()
            activation_value = float(graph.tc_activation_values[local_idx].item())
            feature_type = "transcoder"
        features.append(
            GraphFeature(
                node_index=node_index,
                global_id=int(global_id),
                feature_type=feature_type,
                layer=int(layer),
                position=int(position),
                feature_idx=int(feature_idx),
                activation_value=activation_value,
            )
        )
    return features


def build_logit_weights(graph: Graph) -> torch.Tensor:
    adjacency = graph.adjacency_matrix.detach().to(torch.float64).cpu()
    weights = torch.zeros(adjacency.shape[0], dtype=torch.float64)
    n_logits = int(len(graph.logit_tokens))
    if n_logits > 0:
        weights[-n_logits:] = graph.logit_probabilities.detach().to(torch.float64).cpu().view(-1)
    return weights


def normalize_matrix_abs(matrix: torch.Tensor) -> torch.Tensor:
    matrix = matrix.to(torch.float64).cpu()
    return matrix.abs() / matrix.abs().sum(dim=1, keepdim=True).clamp(min=1e-12)


def normalize_matrix_signed(matrix: torch.Tensor) -> torch.Tensor:
    matrix = matrix.to(torch.float64).cpu()
    return matrix / matrix.abs().sum(dim=1, keepdim=True).clamp(min=1e-12)


def compute_backward_influence(
    normalized_matrix: torch.Tensor,
    seed: torch.Tensor,
    *,
    max_iter: int = 2048,
    tol: float = 1e-12,
) -> torch.Tensor:
    current = seed @ normalized_matrix
    total = current.clone()
    for _ in range(max_iter):
        if float(current.abs().max().item()) <= tol:
            break
        current = current @ normalized_matrix
        total = total + current
    return total


def feature_direct_and_indirect_scores(
    graph: Graph,
    *,
    signed: bool,
) -> tuple[np.ndarray, np.ndarray]:
    adjacency = graph.adjacency_matrix.detach().cpu()
    normalized = normalize_matrix_signed(adjacency) if signed else normalize_matrix_abs(adjacency)
    logit_weights = build_logit_weights(graph)
    direct = (logit_weights @ normalized).detach().cpu().numpy()
    indirect = compute_backward_influence(normalized, logit_weights).detach().cpu().numpy()
    n_features = len(graph.selected_features)
    return direct[:n_features], indirect[:n_features]


def pairwise_backward_scores(
    graph: Graph,
    *,
    target_node_indices: Sequence[int],
    signed: bool,
) -> dict[int, np.ndarray]:
    adjacency = graph.adjacency_matrix.detach().cpu()
    normalized = normalize_matrix_signed(adjacency) if signed else normalize_matrix_abs(adjacency)
    n_features = len(graph.selected_features)
    out: dict[int, np.ndarray] = {}
    for target_idx in target_node_indices:
        seed = torch.zeros(normalized.shape[0], dtype=torch.float64)
        seed[target_idx] = 1.0
        scores = compute_backward_influence(normalized, seed)[:n_features]
        out[int(target_idx)] = scores.detach().cpu().numpy()
    return out


def feature_key(feature: GraphFeature) -> tuple[str, int, int, int]:
    return (feature.feature_type, feature.layer, feature.position, feature.feature_idx)


def build_sparse_activation_lookup(
    lorsa_sparse: torch.Tensor,
    tc_sparse: torch.Tensor,
) -> dict[tuple[str, int, int, int], float]:
    lookup: dict[tuple[str, int, int, int], float] = {}
    for feature_type, sparse_tensor in (("lorsa", lorsa_sparse), ("transcoder", tc_sparse)):
        sparse_tensor = sparse_tensor.coalesce().cpu()
        if sparse_tensor._nnz() == 0:
            continue
        indices = sparse_tensor.indices().T.tolist()
        values = sparse_tensor.values().tolist()
        for (layer, position, feature_idx), value in zip(indices, values):
            lookup[(feature_type, int(layer), int(position), int(feature_idx))] = float(value)
    return lookup


def get_sparse_activation_value(
    lookup: dict[tuple[str, int, int, int], float],
    feature: GraphFeature,
) -> float:
    return float(lookup.get(feature_key(feature), 0.0))


def feature_output_hook_name(feature: GraphFeature) -> str:
    if feature.feature_type == "lorsa":
        return f"blocks.{feature.layer}.hook_attn_out"
    if feature.feature_type == "transcoder":
        return f"blocks.{feature.layer}.hook_mlp_out"
    raise ValueError(feature.feature_type)


def build_feature_delta_hook(
    bundle: BT4ModelBundle,
    feature: GraphFeature,
    delta_activation: float,
) -> tuple[str, Any]:
    decoder = get_feature_vector(
        bundle.lorsas,
        bundle.transcoders,
        feature.feature_type,
        feature.layer,
        feature.feature_idx,
    ).detach()
    hook_name = feature_output_hook_name(feature)

    def _hook(act: torch.Tensor, hook) -> torch.Tensor:
        out = act.clone()
        delta = (delta_activation * decoder).to(device=out.device, dtype=out.dtype)
        if out.dim() == 3:
            out[:, feature.position, :] = out[:, feature.position, :] + delta
        elif out.dim() == 2:
            out[feature.position, :] = out[feature.position, :] + delta
        else:
            raise ValueError(f"Unexpected activation rank at {hook_name}: {tuple(out.shape)}")
        return out

    return hook_name, _hook


def build_encoder_direction(feature: GraphFeature, bundle: BT4ModelBundle, delta_activation: float) -> torch.Tensor:
    encoder = get_feature_encoder_vector(
        bundle.lorsas,
        bundle.transcoders,
        feature.feature_type,
        feature.layer,
        feature.feature_idx,
    ).detach().to(torch.float64).cpu()
    scale = float(delta_activation) / float(torch.dot(encoder, encoder).item() + 1e-12)
    return (encoder * scale).to(torch.float32)


def build_random_direction(feature: GraphFeature, bundle: BT4ModelBundle, delta_activation: float, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    base_direction = build_encoder_direction(feature, bundle, delta_activation).to(torch.float64)
    random_direction = torch.randn(base_direction.shape, generator=generator, dtype=torch.float64)
    projection = torch.dot(random_direction, base_direction) / (torch.dot(base_direction, base_direction) + 1e-12)
    random_direction = random_direction - projection * base_direction
    norm = float(torch.linalg.norm(random_direction).item())
    if norm <= 1e-12:
        return base_direction.to(torch.float32)
    target_norm = float(torch.linalg.norm(base_direction).item())
    return (random_direction / norm * target_norm).to(torch.float32)


def build_position_vector_hook(hook_name: str, position: int, vector: torch.Tensor) -> tuple[str, Any]:
    vector = vector.detach().cpu()

    def _hook(act: torch.Tensor, hook) -> torch.Tensor:
        out = act.clone()
        delta = vector.to(device=out.device, dtype=out.dtype)
        if out.dim() == 3:
            out[:, position, :] = out[:, position, :] + delta
        elif out.dim() == 2:
            out[position, :] = out[position, :] + delta
        else:
            raise ValueError(f"Unexpected activation rank at {hook_name}: {tuple(out.shape)}")
        return out

    return hook_name, _hook


def build_full_tensor_hook(hook_name: str, delta: torch.Tensor) -> tuple[str, Any]:
    delta = delta.detach().cpu()

    def _hook(act: torch.Tensor, hook) -> torch.Tensor:
        out = act.clone()
        delta_local = delta.to(device=out.device, dtype=out.dtype)
        if out.dim() == 3 and delta_local.dim() == 2:
            out = out + delta_local.unsqueeze(0)
        elif out.dim() == delta_local.dim():
            out = out + delta_local
        else:
            raise ValueError(
                f"Delta rank mismatch at {hook_name}: act={tuple(out.shape)} delta={tuple(delta_local.shape)}"
            )
        return out

    return hook_name, _hook


def run_setup_attribution_sparse(
    model: Any,
    fen: str,
    *,
    extra_hooks: Sequence[tuple[str, Any]] | None = None,
) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if extra_hooks:
        with model.hooks(fwd_hooks=list(extra_hooks)):
            return model.setup_attribution(fen, sparse=True)
    return model.setup_attribution(fen, sparse=True)


def run_with_cache_hooks(
    model: Any,
    fen: str,
    *,
    hook_names: Sequence[str],
    extra_hooks: Sequence[tuple[str, Any]] | None = None,
) -> tuple[Any, dict[str, torch.Tensor]]:
    names = set(hook_names)
    names_filter = lambda name: name in names
    if extra_hooks:
        with model.hooks(fwd_hooks=list(extra_hooks)), torch.no_grad():
            return model.run_with_cache(fen, prepend_bos=False, names_filter=names_filter)
    with torch.no_grad():
        return model.run_with_cache(fen, prepend_bos=False, names_filter=names_filter)


def build_frozen_error_crm_hooks(
    model: Any,
    fen: str,
) -> tuple[list[tuple[str, Any]], torch.Tensor]:
    return build_frozen_error_partial_crm_hooks(
        model,
        fen,
        replace_attention=True,
        replace_mlp=True,
    )


def build_frozen_error_partial_crm_hooks(
    model: Any,
    fen: str,
    *,
    replace_attention: bool,
    replace_mlp: bool,
) -> tuple[list[tuple[str, Any]], torch.Tensor]:
    """Build hooks for a partially replaced CRM.

    This helper supports three ablation settings with one implementation:

    - full CRM: ``replace_attention=True, replace_mlp=True``
    - only lorsa: ``replace_attention=True, replace_mlp=False``
    - only transcoder: ``replace_attention=False, replace_mlp=True``

    When one branch is disabled, the underlying model path is left untouched.
    This is exactly what we need for feature-perturbation faithfulness
    experiments that isolate the contribution of attention replacement or MLP
    replacement.
    """

    _, _, _, _, error_vectors, _ = run_setup_attribution_sparse(model, fen)
    attn_in_cache: dict[int, torch.Tensor] = {}
    mlp_in_cache: dict[int, torch.Tensor] = {}
    n_layers = model.cfg.n_layers

    def capture_attn_in(act: torch.Tensor, hook, layer: int) -> torch.Tensor:
        attn_in_cache[layer] = act
        return act

    def replace_attn_out(_act: torch.Tensor, hook, layer: int) -> torch.Tensor:
        encoded = model.lorsas[layer].encode(attn_in_cache[layer])
        decoded = model.lorsas[layer].decode(encoded)
        if decoded.ndim == 4:
            decoded = decoded.sum(dim=1)
        delta = error_vectors[layer].to(device=decoded.device, dtype=decoded.dtype)
        if decoded.dim() == 3:
            delta = delta.unsqueeze(0)
        return decoded + delta

    def capture_mlp_in(act: torch.Tensor, hook, layer: int) -> torch.Tensor:
        mlp_in_cache[layer] = act
        return act

    def replace_mlp_out(_act: torch.Tensor, hook, layer: int) -> torch.Tensor:
        encoded = model.transcoders[str(layer)].encode(mlp_in_cache[layer])
        decoded = model.transcoders[str(layer)].decode(encoded)
        delta = error_vectors[n_layers + layer].to(device=decoded.device, dtype=decoded.dtype)
        if decoded.dim() == 3:
            delta = delta.unsqueeze(0)
        return decoded + delta

    hooks: list[tuple[str, Any]] = []
    for layer in range(n_layers):
        if replace_attention:
            hooks.append((f"blocks.{layer}.attn.hook_in", lambda act, hook, layer=layer: capture_attn_in(act, hook, layer)))
            hooks.append((f"blocks.{layer}.attn.hook_out", lambda act, hook, layer=layer: replace_attn_out(act, hook, layer)))
        if replace_mlp:
            hooks.append((f"blocks.{layer}.mlp.hook_in", lambda act, hook, layer=layer: capture_mlp_in(act, hook, layer)))
            hooks.append((f"blocks.{layer}.mlp.hook_out", lambda act, hook, layer=layer: replace_mlp_out(act, hook, layer)))
    return hooks, error_vectors


def _lookup_move_idx(lboard: LeelaBoard, move_uci: str) -> int | None:
    try:
        return int(lboard.uci2idx(move_uci))
    except Exception:
        if len(move_uci) == 5 and move_uci[4] in {"q", "r", "b", "n"}:
            try:
                return int(lboard.uci2idx(move_uci[:4]))
            except Exception:
                return None
        return None


def legal_move_distribution(policy_logits: torch.Tensor, fen: str) -> tuple[list[str], torch.Tensor]:
    logits = policy_logits.detach()
    if logits.ndim == 2:
        logits = logits[0]
    lboard = LeelaBoard.from_fen(fen, history_synthesis=True)
    board = chess.Board(fen)
    moves: list[str] = []
    legal_logits: list[float] = []
    for move in board.legal_moves:
        uci = move.uci()
        idx = _lookup_move_idx(lboard, uci)
        if idx is None:
            continue
        moves.append(uci)
        legal_logits.append(float(logits[idx].item()))
    if not legal_logits:
        return [], torch.empty(0, dtype=torch.float64)
    logits_tensor = torch.tensor(legal_logits, dtype=torch.float64)
    probs = torch.softmax(logits_tensor - logits_tensor.max(), dim=0)
    return moves, probs


def kl_divergence_from_policy_logits(
    reference_policy_logits: torch.Tensor,
    ablated_policy_logits: torch.Tensor,
    fen: str,
    *,
    legal_only: bool,
) -> float:
    if legal_only:
        ref_moves, ref_probs = legal_move_distribution(reference_policy_logits, fen)
        abl_moves, abl_probs = legal_move_distribution(ablated_policy_logits, fen)
        if ref_moves != abl_moves:
            move_to_prob = {move: abl_probs[idx] for idx, move in enumerate(abl_moves)}
            aligned = [float(move_to_prob.get(move, 1e-12)) for move in ref_moves]
            abl_probs = torch.tensor(aligned, dtype=torch.float64)
        p = ref_probs.clamp(min=1e-12)
        q = abl_probs.clamp(min=1e-12)
    else:
        p = torch.softmax(reference_policy_logits.detach().view(-1).to(torch.float64), dim=0).clamp(min=1e-12)
        q = torch.softmax(ablated_policy_logits.detach().view(-1).to(torch.float64), dim=0).clamp(min=1e-12)
    return float((p * (p.log() - q.log())).sum().item())


def rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def _filter_finite_pair(x: Iterable[float], y: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(list(x), dtype=np.float64)
    y_arr = np.asarray(list(y), dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    return x_arr[mask], y_arr[mask]


def pearson_corr(x: Iterable[float], y: Iterable[float]) -> float:
    x_arr, y_arr = _filter_finite_pair(x, y)
    if x_arr.size < 2:
        return float("nan")
    x_centered = x_arr - x_arr.mean()
    y_centered = y_arr - y_arr.mean()
    denom = math.sqrt(float((x_centered**2).sum() * (y_centered**2).sum()))
    if denom <= 1e-12:
        return float("nan")
    return float((x_centered * y_centered).sum() / denom)


def spearman_corr(x: Iterable[float], y: Iterable[float]) -> float:
    x_arr, y_arr = _filter_finite_pair(x, y)
    if x_arr.size < 2:
        return float("nan")
    return pearson_corr(rankdata_average(x_arr), rankdata_average(y_arr))


def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.detach().to(torch.float64).reshape(-1)
    y = y.detach().to(torch.float64).reshape(-1)
    x_norm = float(torch.linalg.norm(x).item())
    y_norm = float(torch.linalg.norm(y).item())
    if x_norm <= 1e-12 and y_norm <= 1e-12:
        return 1.0
    if x_norm <= 1e-12 or y_norm <= 1e-12:
        return 0.0
    return float(torch.dot(x, y).item() / (x_norm * y_norm))


def normalized_mse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = prediction.detach().to(torch.float64)
    target = target.detach().to(torch.float64)
    mse = float(torch.mean((prediction - target) ** 2).item())
    denom = float(torch.mean(target**2).item())
    return float(mse / max(denom, 1e-12))


def summarize_metric(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan")}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=0)),
    }


def graph_score_payload(graph: Graph) -> dict[str, float]:
    replacement_score, completeness_score = compute_graph_scores(graph, use_lorsa=True)
    return {
        "replacement_score": float(replacement_score),
        "completeness_score": float(completeness_score),
    }
