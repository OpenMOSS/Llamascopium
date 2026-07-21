import functools
import re
import threading
import traceback
from datetime import datetime
from typing import Any, Optional, cast

import torch
from fastapi import APIRouter, BackgroundTasks, Response
from pydantic import BaseModel
from torch.distributed.device_mesh import DeviceMesh

from llamascopium import (
    CircuitConfig,
    CircuitFeatureTarget,
    CircuitInput,
    CircuitStatus,
    LorsaConfig,
    MatryoshkaSAEConfig,
    MatryoshkaSparseAutoEncoder,
    MOLTConfig,
    NodeDimension,
    NodeIndexed,
    NodeInfo,
    SAEConfig,
    TransformerLensLanguageModel,
    prune_attribution,
)
from llamascopium.utils.distributed import is_primary_rank
from llamascopium.utils.logging import get_distributed_logger
from llamascopium.utils.timer import timer
from server.config import LRU_CACHE_SIZE_CIRCUITS, client, device, sae_series
from server.logic.loaders import get_model, get_sae, get_sae_cfg
from server.logic.samples import list_feature_data
from server.logic.workers import distributed
from server.utils.common import make_serializable, synchronized

logger = get_distributed_logger(__name__)

router = APIRouter(tags=["circuits"])

# Lock for thread-safe circuit generation
_generation_lock = threading.Lock()


class PreviewRequest(BaseModel):
    input: CircuitInput
    sae_set_name: str


@router.post("/preview")
@distributed
def preview(request: PreviewRequest, device_mesh: DeviceMesh | None = None):
    """Preview the prompt and predicted next tokens."""
    sae_set = client.get_sae_set(name=request.sae_set_name)
    assert sae_set is not None, f"SAE set {request.sae_set_name} not found"
    sae_names = sae_set.sae_names
    model_name = client.get_sae_model_name(sae_names[0], sae_set.sae_series)
    model = get_model(name=model_name, device_mesh=device_mesh)
    assert isinstance(model, TransformerLensLanguageModel), (
        f"Preview only supports TransformerLens backend, got {type(model)}"
    )

    assert model.tokenizer is not None, "Preview requires a tokenizer"
    if request.input.input_type == "plain_text":
        prompt = request.input.text
    elif request.input.input_type == "chat_template":
        prompt = cast(
            str,
            model.tokenizer.apply_chat_template(
                request.input.messages, tokenize=False, add_generation_prompt=False, continue_final_message=True
            ),
        )
    else:
        raise ValueError(f"Invalid input type: {request.input.input_type}")

    # Generate next predicted tokens
    with torch.no_grad():
        logits = model(prompt)[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        topk = torch.topk(probs, 5)

        next_tokens = []
        for i, prob in zip(topk.indices.tolist(), topk.values.tolist()):
            token_str = model.tokenizer.decode([i])
            next_tokens.append({"token": token_str, "prob": prob})

    return {"prompt": prompt, "next_tokens": next_tokens}


class CreateSaeSetRequest(BaseModel):
    name: str
    sae_names: list[str]


@router.get("/sae-sets")
def list_sae_sets():
    """List all available SAE sets for the current series."""
    return [sae_set.name for sae_set in client.list_sae_sets() if sae_set.sae_series == sae_series]


@router.post("/sae-sets")
def create_sae_set(request: CreateSaeSetRequest):
    """Create a new SAE set in the current series."""
    try:
        client.add_sae_set(
            name=request.name,
            sae_series=sae_series,
            sae_names=request.sae_names,
        )
        return {"message": f"SAE set '{request.name}' created successfully"}
    except Exception as e:
        if "duplicate key" in str(e).lower():
            return Response(
                content=f"SAE set with name '{request.name}' already exists",
                status_code=409,
            )
        raise


class GenerateCircuitRequest(BaseModel):
    """Request to generate a new circuit graph.

    New feature targets use explicit SAE names. Legacy
    (layer, feature_idx, pos, is_lorsa) tuples remain supported.
    """

    input: CircuitInput
    name: Optional[str] = None
    group: Optional[str] = None
    desired_logit_prob: float = 0.98
    max_feature_nodes: int = 256
    qk_tracing_topk: int = 10
    max_n_logits: int = 1
    list_of_features: Optional[list[tuple[int, int, int, bool] | CircuitFeatureTarget]] = None
    matryoshka_feature_range: Optional[tuple[int, int]] = None
    parent_id: Optional[str] = None


def get_matryoshka_range_options(sae_set_name: str) -> dict[str, Any]:
    sae_set = client.get_sae_set(name=sae_set_name)
    if sae_set is None:
        raise ValueError(f"SAE set {sae_set_name} not found")

    configs = [
        (sae_name, cfg)
        for sae_name in sae_set.sae_names
        if isinstance((cfg := get_sae_cfg(name=sae_name)), MatryoshkaSAEConfig)
    ]
    if not configs:
        return {"sae_names": [], "prefixes": [], "segments": [], "atomic_segments": []}

    boundary_sets: list[set[int]] = []
    for _, cfg in configs:
        boundary_sets.append({0, *cfg.matryoshka_widths})

    common_boundaries = sorted(set.intersection(*boundary_sets))
    prefixes = [(0, end) for end in common_boundaries[1:]]
    segments = [
        (start, end)
        for start_idx, start in enumerate(common_boundaries[1:], start=1)
        for end in common_boundaries[start_idx + 1 :]
    ]
    atomic_segments = list(zip(common_boundaries[:-1], common_boundaries[1:]))
    return {
        "sae_names": [name for name, _ in configs],
        "prefixes": prefixes,
        "segments": segments,
        "atomic_segments": atomic_segments,
    }


@router.get("/sae-sets/{sae_set_name}/matryoshka-ranges")
def list_matryoshka_ranges(sae_set_name: str):
    """List selectable Matryoshka ranges and their atomic subsegments."""
    try:
        return get_matryoshka_range_options(sae_set_name)
    except ValueError as exc:
        return Response(content=str(exc), status_code=404)


@timer.time("concretize_graph_data")
def concretize_graph_data(graph_data: dict[str, Any], circuit_sae_series: str):
    """Concretize a graph data by adding feature data. This will modify the graph data in place."""
    logger.info("Retrieving feature records for circuit")

    indices_by_sae: dict[str, set[int]] = {}
    for node in graph_data["nodes"]:
        sae_name = node.get("sae_name")
        if sae_name is not None:
            indices_by_sae.setdefault(sae_name, set()).add(int(node["feature"]))

    features: dict[tuple[str, int], dict[str, Any]] = {}
    for sae_name, indices in indices_by_sae.items():
        features.update(
            list_feature_data(
                sae_name=sae_name,
                indices=sorted(indices),
                series=circuit_sae_series,
                with_samplings=False,
            )
        )

    missing: set[tuple[str, int]] = set()
    for node in graph_data["nodes"]:
        sae_name = node.get("sae_name")
        if sae_name is None:
            continue
        feature_index = int(node["feature"])
        key = (sae_name, feature_index)
        feature = features.get(key)
        if feature is None:
            missing.add(key)
            feature = {
                "feature_index": feature_index,
                "analysis_name": "unavailable",
                "interpretation": None,
                "dictionary_name": sae_name,
                "act_times": 0,
                "max_feature_act": 0.0,
                "n_analyzed_tokens": None,
                "logits": None,
            }
        node["feature"] = feature

    if missing:
        missing_examples = sorted(missing)[:20]
        logger.warning(
            "Circuit feature lookup unresolved: circuit_series=%s count=%d examples=%s",
            circuit_sae_series,
            len(missing),
            missing_examples,
        )


@synchronized
@functools.lru_cache(maxsize=LRU_CACHE_SIZE_CIRCUITS)
def load_circuit_graph(*, circuit_id: str, node_threshold: float, edge_threshold: float) -> dict[str, Any]:
    """Load, prune, and concretize a circuit graph. Cached for repeated access."""
    circuit = client.get_circuit(circuit_id)
    if circuit is None:
        raise ValueError(f"Circuit {circuit_id} not found")
    if circuit.status != CircuitStatus.COMPLETED:
        raise ValueError(f"Circuit {circuit_id} is not completed (status: {circuit.status})")

    ar = client.load_attribution(circuit_id, device=device)
    if ar is None:
        raise ValueError(f"Attribution data not found for circuit {circuit_id}")

    sae_set = client.get_sae_set(name=circuit.sae_set_name)
    if sae_set is None:
        raise ValueError(f"SAE set {circuit.sae_set_name} not found")
    sae_configs = [(sae_name, get_sae_cfg(name=sae_name)) for sae_name in sae_set.sae_names]

    required_matryoshka_nodes = NodeDimension.empty(device=ar.attribution.data.device)
    if circuit.config.matryoshka_feature_range is not None and ar.prompt_token_ids:
        final_position = len(ar.prompt_token_ids) - 1
        matryoshka_keys = {
            cfg.hook_point_out + ".sae.hook_feature_acts"
            for _, cfg in sae_configs
            if isinstance(cfg, MatryoshkaSAEConfig)
        }
        required_infos = []
        for key in matryoshka_keys:
            node = ar.attribution.dimensions[1].node_mappings.get(key)
            if node is None:
                continue
            indices = node.indices[node.indices[:, 0] == final_position]
            if len(indices) > 0:
                required_infos.append(NodeInfo(key=key, indices=indices))
        required_matryoshka_nodes = NodeDimension.from_node_infos(
            required_infos,
            device=ar.attribution.data.device,
        )

    if ar.probs is not None:
        reduction_weight = ar.probs
    else:
        assert ar.targets is not None
        reduction_weight = torch.ones(
            len(ar.targets), device=ar.attribution.data.device, dtype=ar.attribution.data.dtype
        )
    attribution = prune_attribution(
        ar.attribution,
        reduction_weight,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
        targets=ar.targets,
        required_nodes=required_matryoshka_nodes,
    )

    sae_metadata: dict[str, dict[str, Any]] = {}
    for sae_name, cfg in sae_configs:
        assert isinstance(cfg, SAEConfig | LorsaConfig | MOLTConfig)
        is_matryoshka = isinstance(cfg, MatryoshkaSAEConfig)
        if is_matryoshka and circuit.config.matryoshka_feature_range is None:
            continue
        layer_match = re.search(r"blocks\.(\d+)\.", cfg.hook_point_out)
        layer_idx = int(layer_match.group(1)) if layer_match else 0
        sae_metadata[cfg.hook_point_out] = {
            "sae_name": sae_name,
            "is_lorsa": isinstance(cfg, LorsaConfig),
            "is_matryoshka": is_matryoshka,
            "layer_idx": layer_idx,
        }

    def parse_hook_layer(hook_key: str) -> int:
        match = re.search(r"blocks\.(\d+)\.", hook_key)
        return int(match.group(1)) if match else 0

    edge_weights, (targets, sources) = attribution.nonzero()
    ov_nodes = (sources + targets).unique()

    qk_nodes = (
        (
            sum(
                [results.dimensions[0] + results.dimensions[1] for results, _ in ar.qk_trace_results.pairs]
                + [results.dimensions[0] for results, _ in ar.qk_trace_results.q_marginal]
                + [results.dimensions[0] for results, _ in ar.qk_trace_results.k_marginal],
                NodeDimension.empty(device=device),
            ).unique()
            - ov_nodes
        )
        if ar.qk_trace_results is not None
        else NodeDimension.empty(device=device)
    )
    is_from_qk_tracings = torch.zeros(len(ov_nodes) + len(qk_nodes), device=device, dtype=torch.bool)
    is_from_qk_tracings[len(ov_nodes) :] = True
    node_activations = torch.zeros(len(ov_nodes) + len(qk_nodes), device=device, dtype=torch.float32)
    node_activations[: len(ov_nodes)] = ar.activations[ov_nodes].data

    nodes = ov_nodes + qk_nodes

    logit_token_map = {token_id: token for token_id, token in zip(ar.logit_token_ids or [], ar.logit_tokens or [])}
    logit_prob_map = {
        token_id: float(prob.item())
        for token_id, prob in zip(ar.logit_token_ids or [], ar.probs if ar.probs is not None else [])
    }
    target_vocab_index = (
        int(targets.node_mappings["logits"].indices[0][0].item()) if "logits" in targets.node_mappings else -1
    )
    max_feature_layer = max(
        (
            2 * int(item["layer_idx"]) + (2 if item["is_matryoshka"] else int(not item["is_lorsa"]))
            for item in sae_metadata.values()
        ),
        default=-1,
    )

    @timer.time("make_node")
    def make_node(
        node_key: str, indices_tuple: tuple[int, ...], activation: float, is_from_qk_tracing: bool
    ) -> dict[str, Any]:
        indices = list(indices_tuple)

        if node_key == "hook_embed":
            pos = indices[0]
            token_id = int(ar.prompt_token_ids[pos]) if pos < len(ar.prompt_token_ids) else -1
            token = ar.prompt_tokens[pos] if pos < len(ar.prompt_tokens) else ""
            return {
                "feature_type": "embedding",
                "node_id": f"E_{token_id}_{pos}",
                "layer": -1,
                "ctx_idx": pos,
                "token": token,
                "is_target_logit": False,
                "is_from_qk_tracing": is_from_qk_tracing,
            }

        if node_key == "logits":
            vocab_idx = indices[0]
            ctx_idx = max(len(ar.prompt_token_ids) - 1, 0)
            layer = max_feature_layer + 1
            return {
                "feature_type": "logit",
                "node_id": f"{layer}_{vocab_idx}_{ctx_idx}",
                "layer": layer,
                "ctx_idx": ctx_idx,
                "token_prob": logit_prob_map[vocab_idx],
                "token": logit_token_map[vocab_idx],
                "is_target_logit": vocab_idx == target_vocab_index,
                "is_from_qk_tracing": is_from_qk_tracing,
            }

        if node_key.endswith(".error"):
            hook_point_out = node_key.removesuffix(".error")
            metadata = sae_metadata.get(hook_point_out, None)
            is_lorsa = bool(metadata["is_lorsa"]) if metadata is not None else False
            is_matryoshka = bool(metadata["is_matryoshka"]) if metadata is not None else False
            layer_base = int(metadata["layer_idx"]) if metadata is not None else parse_hook_layer(hook_point_out)
            layer = 2 * layer_base + (2 if is_matryoshka else int(not is_lorsa))
            pos = indices[0]
            return {
                "feature_type": (
                    "residual reconstruction error"
                    if is_matryoshka
                    else "lorsa error"
                    if is_lorsa
                    else "mlp reconstruction error"
                ),
                "node_id": f"R_{layer_base}_error_{pos}" if is_matryoshka else f"{layer}_error_{pos}",
                "layer": layer,
                "ctx_idx": pos,
                "is_target_logit": False,
                "is_from_qk_tracing": is_from_qk_tracing,
            }

        if node_key.endswith(".sae.hook_feature_acts"):
            hook_point_out = node_key.removesuffix(".sae.hook_feature_acts")
            metadata = sae_metadata.get(hook_point_out, None)
            is_lorsa = bool(metadata["is_lorsa"]) if metadata is not None else False
            is_matryoshka = bool(metadata["is_matryoshka"]) if metadata is not None else False
            layer_base = int(metadata["layer_idx"]) if metadata is not None else parse_hook_layer(hook_point_out)
            layer = 2 * layer_base + (2 if is_matryoshka else int(not is_lorsa))
            pos = indices[0] if len(indices) > 0 else 0
            feature_idx = indices[1] if len(indices) > 1 else 0
            return {
                "feature_type": (
                    "matryoshka sae" if is_matryoshka else "lorsa" if is_lorsa else "cross layer transcoder"
                ),
                "node_id": (f"R_{layer_base}_{feature_idx}_{pos}" if is_matryoshka else f"{layer}_{feature_idx}_{pos}"),
                "layer": layer,
                "ctx_idx": pos,
                "feature": feature_idx,
                "sae_name": metadata["sae_name"] if metadata is not None else None,
                "matryoshka_feature_range": (circuit.config.matryoshka_feature_range if is_matryoshka else None),
                "activation": activation,
                "qk_tracing_results": None,
                "is_target_logit": False,
                "is_from_qk_tracing": is_from_qk_tracing,
            }
        return {
            "feature_type": "bias",
            "node_id": f"{node_key}:{'_'.join(str(v) for v in indices)}",
            "layer": 0,
            "ctx_idx": indices[0] if len(indices) > 0 else 0,
            "is_target_logit": False,
            "is_from_qk_tracing": is_from_qk_tracing,
        }

    node_entries = [
        make_node(
            str(ni.key),
            tuple(ni.indices[0].tolist()),
            float(activation),
            bool(is_from_qk_tracing),
        )
        for ni, activation, is_from_qk_tracing in zip(nodes, node_activations, is_from_qk_tracings)
    ]
    node_ids = [node_data["node_id"] for node_data in node_entries]

    target_node_offsets = nodes.nodes_to_offsets(targets).tolist()
    source_node_offsets = nodes.nodes_to_offsets(sources).tolist()
    edge_weight_values = edge_weights.tolist()
    links = [
        {
            "source": node_ids[source_node_offset],
            "target": node_ids[target_node_offset],
            "weight": float(weight),
        }
        for weight, source_node_offset, target_node_offset in zip(
            edge_weight_values, source_node_offsets, target_node_offsets
        )
    ]

    if ar.qk_trace_results is not None:
        qk_targets_node_offsets = nodes.nodes_to_offsets(ar.qk_trace_results.pairs.dimensions[0]).tolist()

        def make_qk_pair_contributors(pair_slot: NodeIndexed[torch.Tensor]) -> list[tuple[str, str, float]]:
            q_node_offsets = nodes.nodes_to_offsets(pair_slot.dimensions[0]).tolist()
            k_node_offsets = nodes.nodes_to_offsets(pair_slot.dimensions[1]).tolist()
            return [
                (node_ids[q_node_offset], node_ids[k_node_offset], float(result))
                for q_node_offset, k_node_offset, result in zip(q_node_offsets, k_node_offsets, pair_slot.value)
            ]

        def make_qk_marginal_contributors(marginal_slot: NodeIndexed[torch.Tensor]) -> list[tuple[str, float]]:
            node_offsets = nodes.nodes_to_offsets(marginal_slot.dimensions[0]).tolist()
            return [
                (node_ids[node_offset], float(result)) for node_offset, result in zip(node_offsets, marginal_slot.value)
            ]

        for qk_target_node_offset, (pair_slot, _), (q_marginal_slot, _), (k_marginal_slot, _) in zip(
            qk_targets_node_offsets,
            ar.qk_trace_results.pairs,
            ar.qk_trace_results.q_marginal,
            ar.qk_trace_results.k_marginal,
        ):
            node_entries[qk_target_node_offset]["qk_tracing_results"] = {
                "pair_wise_contributors": make_qk_pair_contributors(pair_slot),
                "top_q_marginal_contributors": make_qk_marginal_contributors(q_marginal_slot),
                "top_k_marginal_contributors": make_qk_marginal_contributors(k_marginal_slot),
            }

    graph_data: dict[str, Any] = {
        "metadata": {
            "prompt_tokens": ar.prompt_tokens,
            "prompt": circuit.prompt,
            "schema_version": 1,
        },
        "nodes": node_entries,
        "links": links,
    }

    concretize_graph_data(graph_data, circuit.sae_series)

    return graph_data


@distributed
def run_circuit_attribution(
    circuit_id: str,
    sae_set_name: str,
    prompt: str,
    request: GenerateCircuitRequest,
    device_mesh: DeviceMesh | None = None,
):
    """Background task to run circuit attribution and store the result."""
    try:
        with _generation_lock:
            # Update status to running
            if is_primary_rank(device_mesh):
                client.update_circuit_status(circuit_id, CircuitStatus.RUNNING)
                client.update_circuit_progress(circuit_id, 0.0, "Loading sparse dictionaries...")

            # Load SAE set and models
            sae_set = client.get_sae_set(name=sae_set_name)
            assert sae_set is not None, f"SAE set {sae_set_name} not found"
            sae_names = sae_set.sae_names
            saes = {sae_name: get_sae(name=sae_name, device_mesh=device_mesh) for sae_name in sae_names}
            replacement_saes = {
                sae_name: sae
                for sae_name, sae in saes.items()
                if request.matryoshka_feature_range is not None or not isinstance(sae, MatryoshkaSparseAutoEncoder)
            }
            assert replacement_saes, "The selected SAE set has no enabled sparse dictionaries"

            model_name = client.get_sae_model_name(sae_names[0], sae_set.sae_series)
            model = get_model(name=model_name, device_mesh=device_mesh)
            assert isinstance(model, TransformerLensLanguageModel), (
                "Circuit tracing only supports exact model of TransformerLens backend"
            )

            if is_primary_rank(device_mesh):
                client.update_circuit_progress(circuit_id, 15.0, "Computing attribution...")

            progress_checkpoints = [(15.0, 95.0, "Feature influence computation")]

            def progress_callback(current: float, total: float, phase: str):
                phase_start, phase_end = next(
                    (p_start, p_end) for p_start, p_end, p_phase in progress_checkpoints if p_phase == phase
                )
                overall_progress = phase_start + (current / total) * (phase_end - phase_start)
                if is_primary_rank(device_mesh):
                    client.update_circuit_progress(circuit_id, overall_progress, phase)

            feature_targets: list[tuple[str, int, int]] = []
            if request.list_of_features:
                layer_key_map: dict[tuple[int, bool], str] = {}
                for sae in replacement_saes.values():
                    if isinstance(sae, MatryoshkaSparseAutoEncoder):
                        continue
                    cfg = sae.cfg
                    assert isinstance(cfg, SAEConfig | LorsaConfig | MOLTConfig)
                    m = re.search(r"blocks\.(\d+)\.", cfg.hook_point_out)
                    if m:
                        layer_key_map[(int(m.group(1)), isinstance(cfg, LorsaConfig))] = (
                            cfg.hook_point_out + ".sae.hook_feature_acts"
                        )
                legacy_targets = [target for target in request.list_of_features if isinstance(target, tuple)]
                missing = [
                    (layer, is_lorsa)
                    for layer, _, _, is_lorsa in legacy_targets
                    if (layer, is_lorsa) not in layer_key_map
                ]
                assert not missing, f"No SAE in set {sae_set_name} covers (layer, is_lorsa)={missing}"
                for target in request.list_of_features:
                    if isinstance(target, CircuitFeatureTarget):
                        sae = replacement_saes.get(target.sae_name)
                        assert sae is not None, f"SAE {target.sae_name} is not enabled in set {sae_set_name}"
                        cfg = sae.cfg
                        assert isinstance(cfg, SAEConfig | LorsaConfig | MOLTConfig)
                        if isinstance(sae, MatryoshkaSparseAutoEncoder):
                            assert request.matryoshka_feature_range is not None
                            start, end = request.matryoshka_feature_range
                            assert start <= target.feature_index < end, (
                                f"Matryoshka feature {target.feature_index} is outside enabled range [{start}, {end})"
                            )
                        feature_targets.append(
                            (
                                cfg.hook_point_out + ".sae.hook_feature_acts",
                                target.position,
                                target.feature_index,
                            )
                        )
                    else:
                        layer, feat_idx, pos, is_lorsa = target
                        feature_targets.append((layer_key_map[(layer, is_lorsa)], pos, feat_idx))

            attribution = model.attribute(
                inputs=prompt,
                replacement_modules=list(replacement_saes.values()),
                target_type="features" if feature_targets else "logits",
                features=feature_targets,
                max_n_logits=request.max_n_logits,
                desired_logit_prob=request.desired_logit_prob,
                batch_size=max(10, request.qk_tracing_topk),
                max_features=request.max_feature_nodes,
                enable_qk_tracing=request.qk_tracing_topk > 0,
                qk_topk=request.qk_tracing_topk,
                matryoshka_feature_range=request.matryoshka_feature_range,
            ).full_tensor()

            if is_primary_rank(device_mesh):
                client.update_circuit_progress(circuit_id, 90.0, "Storing attribution data...")
                client.store_attribution(circuit_id, attribution)

                client.update_circuit_progress(circuit_id, 100.0, "Completed")
                client.update_circuit_status(circuit_id, CircuitStatus.COMPLETED)

            logger.info(f"Circuit {circuit_id} attribution completed successfully")

    except Exception as e:
        error_msg = f"Attribution failed: {str(e)}\n{traceback.format_exc()}"
        logger.error(f"Circuit {circuit_id} attribution failed: {error_msg}")
        client.update_circuit_status(circuit_id, CircuitStatus.FAILED, error_message=str(e))


@router.post("/circuits")
def create_circuit(sae_set_name: str, request: GenerateCircuitRequest, background_tasks: BackgroundTasks):
    """Start circuit graph generation as a background task.

    Returns immediately with the circuit_id and status='pending'.
    Use GET /circuits/{id}/status to check progress.
    """
    sae_set = client.get_sae_set(name=sae_set_name)
    if sae_set is None:
        return Response(content=f"SAE set {sae_set_name} not found", status_code=404)

    range_options = get_matryoshka_range_options(sae_set_name)
    allowed_ranges = {tuple(item) for item in range_options["prefixes"] + range_options["segments"]}
    if request.matryoshka_feature_range is not None and request.matryoshka_feature_range not in allowed_ranges:
        return Response(
            content=(
                f"Invalid Matryoshka feature range {request.matryoshka_feature_range}; "
                f"allowed ranges are {sorted(allowed_ranges)}"
            ),
            status_code=400,
        )

    sae_names = sae_set.sae_names

    model_name = client.get_sae_model_name(sae_names[0], sae_set.sae_series)
    model = get_model(name=model_name, device_mesh=None)
    # if not isinstance(model, TransformerLensLanguageModel):
    #     return Response(
    #         content="Circuit tracing only supports TransformerLens backend",
    #         status_code=400,
    #     )

    # Determine prompt
    assert model.tokenizer is not None, "Circuit generation requires a tokenizer"
    if request.input.input_type == "plain_text":
        prompt = request.input.text
    elif request.input.input_type == "chat_template":
        prompt = cast(
            str,
            model.tokenizer.apply_chat_template(
                request.input.messages, tokenize=False, add_generation_prompt=False, continue_final_message=True
            ),
        )
    else:
        return Response(content=f"Invalid input type: {request.input.input_type}", status_code=400)

    # Create circuit config (without threshold params)
    config = CircuitConfig(
        desired_logit_prob=request.desired_logit_prob,
        max_feature_nodes=request.max_feature_nodes,
        qk_tracing_topk=request.qk_tracing_topk,
        max_n_logits=request.max_n_logits,
        list_of_features=request.list_of_features,
        matryoshka_feature_range=request.matryoshka_feature_range,
    )

    # Create circuit record with pending status
    circuit_id = client.create_circuit(
        sae_set_name=sae_set_name,
        sae_series=sae_series,
        prompt=prompt,
        input=request.input,
        config=config,
        name=request.name,
        group=request.group,
        parent_id=request.parent_id,
    )

    # Start background task
    background_tasks.add_task(
        run_circuit_attribution,
        circuit_id=circuit_id,
        sae_set_name=sae_set_name,
        prompt=prompt,
        request=request,
    )

    return {
        "circuit_id": circuit_id,
        "status": CircuitStatus.PENDING,
        "name": request.name,
        "group": request.group,
        "sae_set_name": sae_set_name,
        "prompt": prompt,
        "config": config.model_dump(),
        "input": request.input.model_dump(),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "parent_id": request.parent_id,
    }


@router.get("/circuits")
def list_circuits(limit: int = 100, skip: int = 0, group: Optional[str] = None):
    """List all circuits for the current SAE series."""
    circuits = client.list_circuits(sae_series=sae_series, limit=limit, skip=skip, group=group)

    results = []
    for circuit in circuits:
        circuit["created_at"] = circuit["created_at"].isoformat() + "Z"
        results.append(circuit)

    return results


@router.get("/circuits/{circuit_id}/status")
def get_circuit_status(circuit_id: str):
    """Get the current status of a circuit generation."""
    status = client.get_circuit_status(circuit_id)
    if status is None:
        return Response(content=f"Circuit {circuit_id} not found", status_code=404)

    return status


@router.get("/circuits/{circuit_id}")
def get_circuit(circuit_id: str, node_threshold: float = 0.6, edge_threshold: float = 0.8):
    """Get a circuit by its ID with dynamic pruning.

    The raw graph is pruned on-the-fly using the provided threshold parameters.

    Args:
        circuit_id: The circuit ID.
        node_threshold: Keep nodes contributing to this fraction of total influence (0-1).
        edge_threshold: Keep edges contributing to this fraction of total influence (0-1).
    """
    circuit = client.get_circuit(circuit_id)
    if circuit is None:
        return Response(content=f"Circuit {circuit_id} not found", status_code=404)

    # Check if circuit is completed
    if circuit.status != CircuitStatus.COMPLETED:
        return Response(
            content=f"Circuit {circuit_id} is not ready (status: {circuit.status})",
            status_code=202,
            headers={"X-Circuit-Status": circuit.status},
        )

    try:
        graph_data = load_circuit_graph(
            circuit_id=circuit_id, node_threshold=node_threshold, edge_threshold=edge_threshold
        )
    except ValueError as e:
        return Response(content=str(e), status_code=404)

    result = {
        "circuit_id": circuit.id,
        "name": circuit.name,
        "group": circuit.group,
        "input": circuit.input.model_dump(),
        "sae_set_name": circuit.sae_set_name,
        "prompt": circuit.prompt,
        "config": circuit.config.model_dump(),
        "graph_data": graph_data,
        "created_at": circuit.created_at.isoformat() + "Z",
        "parent_id": circuit.parent_id,
        "status": circuit.status,
        "node_threshold": node_threshold,
        "edge_threshold": edge_threshold,
    }

    return make_serializable(result)


@router.get("/circuits/{circuit_id}/qk/{node_id}")
def get_circuit_qk_node(circuit_id: str, node_id: str, node_threshold: float = 0.6, edge_threshold: float = 0.8):
    """Get QK tracing data for a specific node, returning only the target node and referenced nodes."""
    circuit = client.get_circuit(circuit_id)
    if circuit is None:
        return Response(content=f"Circuit {circuit_id} not found", status_code=404)

    if circuit.status != CircuitStatus.COMPLETED:
        return Response(
            content=f"Circuit {circuit_id} is not ready (status: {circuit.status})",
            status_code=202,
            headers={"X-Circuit-Status": circuit.status},
        )

    try:
        graph_data = load_circuit_graph(
            circuit_id=circuit_id, node_threshold=node_threshold, edge_threshold=edge_threshold
        )
    except ValueError as e:
        return Response(content=str(e), status_code=404)

    nodes_by_id = {node["node_id"]: node for node in graph_data["nodes"]}

    target_node = nodes_by_id.get(node_id)
    if target_node is None:
        return Response(content=f"Node {node_id} not found in circuit {circuit_id}", status_code=404)

    qk_results = target_node.get("qk_tracing_results")
    if qk_results is None:
        return Response(content=f"Node {node_id} has no QK tracing results", status_code=404)

    # Collect all node IDs referenced in QK tracing results
    referenced_ids: set[str] = set()
    for ref_id, _ in qk_results.get("top_q_marginal_contributors", []):
        referenced_ids.add(ref_id)
    for ref_id, _ in qk_results.get("top_k_marginal_contributors", []):
        referenced_ids.add(ref_id)
    for q_id, k_id, _ in qk_results.get("pair_wise_contributors", []):
        referenced_ids.add(q_id)
        referenced_ids.add(k_id)
    referenced_ids.discard(node_id)

    referenced_nodes = [nodes_by_id[nid] for nid in referenced_ids if nid in nodes_by_id]

    return make_serializable(
        {
            "target_node": target_node,
            "referenced_nodes": referenced_nodes,
        }
    )


@router.delete("/circuits/{circuit_id}")
def delete_circuit(circuit_id: str):
    """Delete a circuit by its ID."""
    success = client.delete_circuit(circuit_id)
    if success:
        return {"message": "Circuit deleted successfully"}
    else:
        return Response(content=f"Circuit {circuit_id} not found", status_code=404)
