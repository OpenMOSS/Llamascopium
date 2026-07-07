import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote

import numpy as np
import torch
from fastapi import APIRouter, HTTPException, Response


CIRCUIT_TAXONOMY_LABELS = [
    "Det",
    "Src",
    "Tgt",
    "Val",
    "Reg",
    "Cap",
    "Pro",
    "Mov",
    "Tac",
    "Spa",
    "Uninterpretable",
]

_CIRCUIT_TAXONOMY_PREFIX_RE = re.compile(
    r"^\[(%s)\]\s*" % "|".join(re.escape(label) for label in CIRCUIT_TAXONOMY_LABELS)
)
_CIRCUIT_TAXONOMY_FEN_RE = re.compile(
    r"\b(?:[pnbrqkPNBRQK1-8]+/){7}[pnbrqkPNBRQK1-8]+\s+[wb]\s+(?:K?Q?k?q?|-)\s+(?:[a-h][36]|-)\s+\d+\s+\d+\b"
)


def get_circuit_taxonomy_router(
    *,
    client: Any,
    sae_series: str,
    repo_root: Path,
    roots: list[dict[str, Any]],
    get_dataset: Callable[[str, int, int], Any],
    get_hooked_model: Callable[[str], Any],
    hooked_transformer_available: bool,
    make_serializable: Callable[[Any], Any],
) -> APIRouter:
    router = APIRouter()
    evidence_output_dir = repo_root / "outputs" / "circuit_taxonomy_evidence"
    review_state_path = repo_root / "outputs" / "circuit_taxonomy_reviews" / "review-state.json"
    review_state_snapshot_dir = review_state_path.parent / "saved"

    def list_directory_options() -> list[dict[str, str]]:
        options: list[dict[str, str]] = []
        for root_cfg in roots:
            root_path = Path(root_cfg["path"]).resolve()
            if not root_path.exists():
                continue
            candidate_directories = sorted(
                directory
                for directory in root_path.rglob("*")
                if directory.is_dir() and any(path.is_file() for path in directory.glob("*.json"))
            )
            for directory in candidate_directories:
                try:
                    relative_path = directory.resolve().relative_to(repo_root.resolve())
                except ValueError:
                    continue
                file_count = len([path for path in directory.glob("*.json") if path.is_file()])
                relative_to_root = directory.resolve().relative_to(root_path)
                relative_to_root_str = str(relative_to_root).replace("\\", "/")
                options.append(
                    {
                        "id": str(relative_path).replace("\\", "/"),
                        "label": f"{root_cfg['label']} / {relative_to_root_str}",
                        "combo_id": directory.name,
                        "root_id": str(root_cfg["id"]),
                        "file_count": str(file_count),
                    }
                )
        return options

    def resolve_directory(directory_id: str) -> Path:
        normalized = unquote(directory_id).strip().replace("\\", "/")
        for option in list_directory_options():
            if option["id"] == normalized:
                return (repo_root / option["id"]).resolve()

        candidate = (repo_root / normalized).resolve()
        if candidate.is_dir() and any(path.is_file() for path in candidate.glob("*.json")):
            for root_cfg in roots:
                root_path = Path(root_cfg["path"]).resolve()
                try:
                    candidate.relative_to(root_path)
                except ValueError:
                    continue
                return candidate
        raise HTTPException(status_code=404, detail=f"Unknown circuit taxonomy directory: {directory_id}")

    def build_dictionary_name(analysis_name: str | None, layer_idx: int, feature_type: str) -> str:
        normalized_type = "lorsa" if feature_type == "lorsa" else "tc"
        component_suffix = "A" if normalized_type == "lorsa" else "M"
        default_combo_suffix = "k30_e16"

        match = re.match(r"^BT4_(tc|lorsa)(?:_(k\d+_e\d+))?$", analysis_name or "")
        if match and match.group(1) == normalized_type:
            combo_suffix = match.group(2) or default_combo_suffix
            return f"BT4_{normalized_type}_L{layer_idx}{component_suffix}_{combo_suffix}"

        return f"BT4_{normalized_type}_L{layer_idx}{component_suffix}_{default_combo_suffix}"

    def parse_features(payload: dict[str, Any]) -> list[dict[str, Any]]:
        metadata = payload.get("metadata", {}) or {}
        raw_features: dict[tuple[int, str, int], dict[str, Any]] = {}

        for node in payload.get("nodes", []):
            feature_type = str(node.get("feature_type", "")).strip().lower()
            if feature_type not in {"lorsa", "cross layer transcoder"}:
                continue

            node_id = str(node.get("node_id", ""))
            parts = node_id.split("_")
            if len(parts) < 2:
                continue

            try:
                raw_layer = int(parts[0])
                feature_index = int(parts[1])
            except ValueError:
                continue

            layer_idx = raw_layer // 2
            dictionary_name = build_dictionary_name(
                metadata.get("lorsa_analysis_name")
                if feature_type == "lorsa"
                else (metadata.get("tc_analysis_name") or metadata.get("clt_analysis_name")),
                layer_idx,
                feature_type,
            )
            normalized_type = "lorsa" if feature_type == "lorsa" else "transcoder"
            key = (layer_idx, normalized_type, feature_index)

            if key in raw_features:
                continue

            raw_features[key] = {
                "node_id": node_id,
                "layer": layer_idx,
                "feature_index": feature_index,
                "feature_type": normalized_type,
                "dictionary_name": dictionary_name,
                "ctx_idx": node.get("ctx_idx"),
                "label": f"L{layer_idx} {normalized_type} #{feature_index}",
            }

        return sorted(
            raw_features.values(),
            key=lambda item: (
                int(item["layer"]),
                0 if item["feature_type"] == "lorsa" else 1,
                int(item["feature_index"]),
            ),
        )

    def extract_prefix(text: str | None) -> str | None:
        if not text:
            return None
        match = _CIRCUIT_TAXONOMY_PREFIX_RE.match(text)
        return match.group(1) if match else None

    def apply_prefix(text: str | None, taxonomy: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return f"[{taxonomy}]"
        if _CIRCUIT_TAXONOMY_PREFIX_RE.match(cleaned):
            return _CIRCUIT_TAXONOMY_PREFIX_RE.sub(f"[{taxonomy}] ", cleaned, count=1).strip()
        return f"[{taxonomy}] {cleaned}".strip()

    def get_feature_text(dictionary_name: str, feature_index: int) -> str:
        feature_doc = client.feature_collection.find_one(
            {
                "sae_name": dictionary_name,
                "sae_series": sae_series,
                "index": feature_index,
            },
            {
                "interpretation.text": 1,
            },
        )
        if not feature_doc:
            return ""

        interpretation = feature_doc.get("interpretation")
        if not isinstance(interpretation, dict):
            return ""

        return str(interpretation.get("text", "") or "")

    def find_first_unannotated_feature_index(
        features: list[dict[str, Any]],
        start_index: int = 0,
    ) -> int | None:
        for feature_index in range(max(start_index, 0), len(features)):
            feature = features[feature_index]
            interpretation_text = get_feature_text(
                str(feature.get("dictionary_name", "")).strip(),
                int(feature.get("feature_index", -1)),
            )
            if not extract_prefix(interpretation_text):
                return feature_index
        return None

    def list_files(directory: Path) -> list[Path]:
        return sorted(path for path in directory.glob("*.json") if path.is_file())

    def board_index_to_square(index: int) -> str:
        if index < 0 or index > 63:
            return f"idx{index}"
        return f"{'abcdefgh'[index % 8]}{8 - (index // 8)}"

    def extract_fen(value: Any) -> str | None:
        if isinstance(value, dict):
            fen = value.get("fen")
            if isinstance(fen, str) and fen.strip():
                return fen.strip()
            text = value.get("text")
            if isinstance(text, str):
                match = _CIRCUIT_TAXONOMY_FEN_RE.search(text)
                if match:
                    return match.group(0)
        if isinstance(value, str):
            match = _CIRCUIT_TAXONOMY_FEN_RE.search(value)
            if match:
                return match.group(0)
        return None

    def compact_signals(value: Any, prefix: str = "", depth: int = 0) -> dict[str, Any]:
        if depth > 3:
            return {}
        signals: dict[str, Any] = {}
        if isinstance(value, dict):
            for key, child in value.items():
                key_text = str(key)
                path = f"{prefix}.{key_text}" if prefix else key_text
                lower = key_text.lower()
                if any(token in lower for token in ("wdl", "value", "move", "policy", "prob", "logit")):
                    if isinstance(child, (str, int, float, bool)) or child is None:
                        signals[path] = child
                    elif isinstance(child, list):
                        signals[path] = child[:5]
                    elif isinstance(child, dict):
                        signals[path] = {
                            str(k): v
                            for k, v in list(child.items())[:8]
                            if isinstance(v, (str, int, float, bool)) or v is None
                        }
                signals.update(compact_signals(child, path, depth + 1))
        elif isinstance(value, list) and depth < 2:
            for index, child in enumerate(value[:5]):
                signals.update(compact_signals(child, f"{prefix}[{index}]", depth + 1))
        return signals

    def get_board_model_summary(fen: str, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
        if fen in cache:
            return cache[fen]

        model_name = "lc0/BT4-1024x15x32h"
        summary: dict[str, Any] = {}
        try:
            if not hooked_transformer_available:
                raise RuntimeError("HookedTransformer is not available")

            model = get_hooked_model(model_name)
            with torch.no_grad():
                output, _ = model.run_with_cache(fen, prepend_bos=False)

            if isinstance(output, (list, tuple)) and len(output) >= 2:
                wdl_tensor = output[1]
                if wdl_tensor.shape == torch.Size([1, 3]):
                    wdl = [
                        float(wdl_tensor[0][0].item()),
                        float(wdl_tensor[0][1].item()),
                        float(wdl_tensor[0][2].item()),
                    ]
                    summary["wdl"] = {
                        "current_player_win": wdl[0],
                        "draw": wdl[1],
                        "current_player_loss": wdl[2],
                        "raw": wdl,
                    }
                    summary["wdl_value"] = float(wdl[0] - wdl[2])
                    summary["value"] = summary["wdl_value"]

            from src.lm_saes.circuit.leela_board import LeelaBoard
            import chess

            policy_output = output[0] if isinstance(output, (list, tuple)) else output
            if policy_output.dim() == 3:
                policy_output = policy_output[:, -1, :]
            if policy_output.dim() == 2:
                policy_output = policy_output[0]
            elif policy_output.dim() != 1:
                raise RuntimeError(f"Unexpected policy output shape: {tuple(policy_output.shape)}")

            policy_output = policy_output.detach().cpu()
            leela_board = LeelaBoard.from_fen(fen, history_synthesis=True)
            legal_entries: list[tuple[str, int, float]] = []
            for move in chess.Board(fen).legal_moves:
                uci = move.uci()
                try:
                    idx = int(leela_board.uci2idx(uci))
                except (KeyError, IndexError, ValueError):
                    if len(uci) == 5 and uci[4] in "qrbn":
                        try:
                            idx = int(leela_board.uci2idx(uci[:4]))
                        except (KeyError, IndexError, ValueError):
                            continue
                    else:
                        continue
                if 0 <= idx < int(policy_output.numel()):
                    legal_entries.append((uci, idx, float(policy_output[idx].item())))

            if legal_entries:
                legal_logits = torch.tensor([entry[2] for entry in legal_entries], dtype=torch.float32)
                legal_probs = torch.softmax(legal_logits - legal_logits.max(), dim=0).tolist()
                ranked = sorted(
                    (
                        {
                            "uci": uci,
                            "logit": round(logit, 6),
                            "prob": round(float(prob), 6),
                            "idx": idx,
                        }
                        for (uci, idx, logit), prob in zip(legal_entries, legal_probs)
                    ),
                    key=lambda item: item["logit"],
                    reverse=True,
                )
                summary["top_moves"] = ranked[:5]
            else:
                summary["top_moves"] = []

        except Exception as error:
            summary = {
                **summary,
                "top_moves": summary.get("top_moves", []),
                "model_analysis_error": str(error),
            }

        cache[fen] = summary
        return summary

    def top_square_entries(indices: Any, values: Any, limit: int) -> list[dict[str, Any]]:
        if indices is None or values is None:
            return []
        index_values: list[tuple[int, float]] = []
        for raw_index, raw_value in zip(np.asarray(indices).reshape(-1), np.asarray(values).reshape(-1)):
            try:
                index = int(raw_index)
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if 0 <= index < 64:
                index_values.append((index, value))
        index_values.sort(key=lambda item: abs(item[1]), reverse=True)
        return [
            {
                "index": index,
                "square": board_index_to_square(index),
                "value": round(value, 6),
            }
            for index, value in index_values[:limit]
        ]

    def top_z_pairs(z_pattern_indices: Any, z_pattern_values: Any, limit: int) -> list[dict[str, Any]]:
        if z_pattern_indices is None or z_pattern_values is None:
            return []
        indices = np.asarray(z_pattern_indices)
        values = np.asarray(z_pattern_values).reshape(-1)
        if indices.size == 0 or values.size == 0:
            return []

        pairs: list[tuple[int, int, float]] = []
        if indices.ndim == 2 and indices.shape[0] >= 2:
            sources = indices[0].reshape(-1)
            targets = indices[1].reshape(-1)
            iterator = zip(sources, targets, values)
        elif indices.ndim == 2 and indices.shape[1] == 2:
            iterator = zip(indices[:, 0].reshape(-1), indices[:, 1].reshape(-1), values)
        else:
            return []

        for raw_source, raw_target, raw_value in iterator:
            try:
                source = int(raw_source)
                target = int(raw_target)
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if 0 <= source < 64 and 0 <= target < 64:
                pairs.append((source, target, value))
        pairs.sort(key=lambda item: abs(item[2]), reverse=True)
        return [
            {
                "source_index": source,
                "source_square": board_index_to_square(source),
                "target_index": target,
                "target_square": board_index_to_square(target),
                "value": round(value, 6),
            }
            for source, target, value in pairs[:limit]
        ]

    def iter_sparse_samples(sampling: Any):
        feature_acts_indices = np.asarray(sampling.feature_acts_indices)
        feature_acts_values = np.asarray(sampling.feature_acts_values)
        if feature_acts_indices.size == 0 or feature_acts_indices.ndim < 2 or feature_acts_indices.shape[1] == 0:
            return

        sample_ids = feature_acts_indices[0]
        sample_order = list(dict.fromkeys(int(item) for item in sample_ids.tolist()))
        z_indices = np.asarray(sampling.z_pattern_indices) if sampling.z_pattern_indices is not None else None
        z_values = np.asarray(sampling.z_pattern_values) if sampling.z_pattern_values is not None else None

        for sample_id in sample_order:
            act_mask = sample_ids == sample_id
            act_indices = feature_acts_indices[1, act_mask]
            act_values = feature_acts_values[act_mask]

            sample_z_indices = None
            sample_z_values = None
            if z_indices is not None and z_values is not None and z_indices.size > 0 and z_indices.ndim >= 2:
                z_mask = z_indices[0] == sample_id
                sample_z_indices = z_indices[1:, z_mask]
                sample_z_values = z_values[z_mask]

            yield sample_id, act_indices, act_values, sample_z_indices, sample_z_values

    def build_feature_evidence(
        directory_id: str,
        circuit_detail: dict[str, Any],
        feature_ref: dict[str, Any],
        feature_index_in_circuit: int,
        max_samples: int,
        top_squares: int,
        top_z: int,
        board_model_summary_cache: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        dictionary_name = str(feature_ref.get("dictionary_name", "")).strip()
        feature_index = int(feature_ref.get("feature_index", -1))
        feature = client.get_feature(sae_name=dictionary_name, sae_series=sae_series, index=feature_index)
        if feature is None:
            raise HTTPException(status_code=404, detail=f"Feature {feature_index} not found in SAE {dictionary_name}")

        analysis = next((item for item in feature.analyses if item.name == "top_activations"), None)
        if analysis is None:
            analysis = feature.analyses[0] if feature.analyses else None

        sample_summaries: list[dict[str, Any]] = []
        if analysis is not None:
            for sampling in analysis.samplings:
                for sample_id, act_indices, act_values, z_indices, z_values in iter_sparse_samples(sampling):
                    if len(sample_summaries) >= max_samples:
                        break

                    dataset_index = int(sample_id)
                    context_idx = sampling.context_idx[dataset_index] if dataset_index < len(sampling.context_idx) else None
                    dataset_name = sampling.dataset_name[dataset_index] if dataset_index < len(sampling.dataset_name) else None
                    model_name = sampling.model_name[dataset_index] if dataset_index < len(sampling.model_name) else None
                    shard_idx = (
                        int(sampling.shard_idx[dataset_index])
                        if sampling.shard_idx is not None and dataset_index < len(sampling.shard_idx)
                        else 0
                    )
                    n_shards = (
                        int(sampling.n_shards[dataset_index])
                        if sampling.n_shards is not None and dataset_index < len(sampling.n_shards)
                        else 1
                    )

                    dataset_row: Any = {}
                    fen = None
                    if dataset_name is not None and context_idx is not None:
                        try:
                            row = get_dataset(str(dataset_name), shard_idx, n_shards)[int(context_idx)]
                            dataset_row = make_serializable(row)
                            fen = extract_fen(dataset_row)
                        except Exception as dataset_error:
                            dataset_row = {"error": f"Failed to load dataset row: {str(dataset_error)}"}

                    model_summary = get_board_model_summary(fen, board_model_summary_cache) if fen else {}
                    sample_summaries.append(
                        {
                            "sample_index": dataset_index,
                            "context_idx": int(context_idx) if context_idx is not None else None,
                            "dataset_name": dataset_name,
                            "model_name": model_name,
                            "fen": fen,
                            "side_to_move": fen.split()[1] if fen and len(fen.split()) >= 2 else None,
                            "wdl": model_summary.get("wdl"),
                            "wdl_value": model_summary.get("wdl_value"),
                            "value": model_summary.get("value"),
                            "top_moves": model_summary.get("top_moves", []),
                            "model_analysis_error": model_summary.get("model_analysis_error"),
                            "top_activated_squares": top_square_entries(act_indices, act_values, top_squares),
                            "top_z_pairs": top_z_pairs(z_indices, z_values, top_z),
                            "signals": compact_signals(dataset_row),
                        }
                    )
                if len(sample_summaries) >= max_samples:
                    break

        interpretation = feature.interpretation if isinstance(feature.interpretation, dict) else {}
        metadata = circuit_detail.get("metadata", {}) if isinstance(circuit_detail.get("metadata"), dict) else {}
        return {
            "directory_id": directory_id,
            "file_name": circuit_detail.get("file_name"),
            "circuit_index": circuit_detail.get("circuit_index"),
            "feature_index_in_circuit": feature_index_in_circuit,
            "dictionary_name": dictionary_name,
            "feature_index": feature_index,
            "layer": feature_ref.get("layer"),
            "feature_type": feature_ref.get("feature_type"),
            "node_id": feature_ref.get("node_id"),
            "label": feature_ref.get("label"),
            "existing_interpretation": str(interpretation.get("text", "") or ""),
            "circuit_metadata": {
                "prompt": metadata.get("prompt"),
                "target_move": metadata.get("target_move"),
                "predicted_move_uci": metadata.get("predicted_move_uci"),
                "logit_moves": metadata.get("logit_moves"),
                "lorsa_analysis_name": metadata.get("lorsa_analysis_name"),
                "tc_analysis_name": metadata.get("tc_analysis_name") or metadata.get("clt_analysis_name"),
            },
            "feature_stats": {
                "analysis_name": analysis.name if analysis is not None else None,
                "max_feature_act": analysis.max_feature_acts if analysis is not None else None,
                "act_times": analysis.act_times if analysis is not None else None,
                "n_analyzed_tokens": analysis.n_analyzed_tokens if analysis is not None else None,
            },
            "top_activation_samples": sample_summaries,
        }

    def collect_evidence_items(
        directory_id: str,
        file_name: str | None,
        start_feature_index: int,
        limit: int | None,
        max_samples: int,
        top_squares: int,
        top_z: int,
        include_annotated: bool = False,
        single_file_only: bool = False,
    ) -> list[dict[str, Any]]:
        directory = resolve_directory(directory_id)
        files = list_files(directory)
        if not files:
            return []

        file_names = [path.name for path in files]
        start_circuit_index = 0
        if file_name:
            if file_name not in file_names:
                raise HTTPException(status_code=404, detail=f"Circuit file not found: {file_name}")
            start_circuit_index = file_names.index(file_name)

        safe_limit = None if limit is None or int(limit) <= 0 else max(1, int(limit))
        safe_max_samples = max(1, min(int(max_samples), 20))
        safe_top_squares = max(1, min(int(top_squares), 64))
        safe_top_z = max(1, min(int(top_z), 64))
        evidence_items: list[dict[str, Any]] = []
        board_model_summary_cache: dict[str, dict[str, Any]] = {}

        for circuit_index in range(start_circuit_index, len(files)):
            path = files[circuit_index]
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            features = parse_features(payload)
            feature_start = max(int(start_feature_index), 0) if circuit_index == start_circuit_index else 0

            circuit_detail = {
                "directory_id": directory_id,
                "file_name": path.name,
                "circuit_index": circuit_index,
                "total_circuits": len(files),
                "total_features": len(features),
                "features": features,
                "metadata": payload.get("metadata", {}) or {},
            }

            while safe_limit is None or len(evidence_items) < safe_limit:
                if include_annotated:
                    if feature_start >= len(features):
                        break
                    feature_index_in_circuit = feature_start
                else:
                    feature_index_in_circuit = find_first_unannotated_feature_index(features, feature_start)
                    if feature_index_in_circuit is None:
                        break

                feature_ref = features[feature_index_in_circuit]
                evidence_items.append(
                    build_feature_evidence(
                        directory_id=directory_id,
                        circuit_detail=circuit_detail,
                        feature_ref=feature_ref,
                        feature_index_in_circuit=feature_index_in_circuit,
                        max_samples=safe_max_samples,
                        top_squares=safe_top_squares,
                        top_z=safe_top_z,
                        board_model_summary_cache=board_model_summary_cache,
                    )
                )
                feature_start = feature_index_in_circuit + 1

            if safe_limit is not None and len(evidence_items) >= safe_limit:
                break
            if single_file_only and file_name:
                break

        return evidence_items

    def evidence_items_to_jsonl(evidence_items: list[dict[str, Any]]) -> str:
        content = "\n".join(json.dumps(make_serializable(item), ensure_ascii=False) for item in evidence_items)
        if content:
            content += "\n"
        return content

    def coerce_review_state(request: dict[str, Any] | None = None) -> dict[str, Any]:
        request = request or {}
        proposals = request.get("proposals", [])
        if not isinstance(proposals, list):
            raise HTTPException(status_code=400, detail="proposals must be a list")

        active_review_index = request.get("active_review_index", 0)
        try:
            active_review_index = max(0, int(active_review_index))
        except (TypeError, ValueError):
            active_review_index = 0

        return {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "active_review_index": active_review_index,
            "proposals": proposals,
        }

    def write_review_state(path: Path, state: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(f"{path.suffix}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2)
        tmp_path.replace(path)

    @router.get("/circuit_taxonomy/directories")
    def list_directories():
        return {
            "directories": [
                {
                    **option,
                    "file_count": int(option["file_count"]),
                }
                for option in list_directory_options()
            ],
            "taxonomy_labels": CIRCUIT_TAXONOMY_LABELS,
        }

    @router.get("/circuit_taxonomy/circuits")
    def list_circuits(directory_id: str):
        directory = resolve_directory(directory_id)
        circuits: list[dict[str, Any]] = []
        for index, path in enumerate(list_files(directory)):
            circuits.append(
                {
                    "file_name": path.name,
                    "index": index,
                    "prompt": None,
                    "target_move": None,
                    "predicted_move_uci": None,
                    "slug": None,
                    "feature_count": 0,
                }
            )
        return {"directory_id": directory_id, "circuits": circuits, "total_circuits": len(circuits)}

    @router.get("/circuit_taxonomy/circuit")
    def get_circuit(directory_id: str, file_name: str):
        directory = resolve_directory(directory_id)
        path = (directory / file_name).resolve()
        if path.parent != directory or not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail=f"Circuit file not found: {file_name}")

        files = [file_path.name for file_path in list_files(directory)]
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        features = parse_features(payload)
        first_unannotated_feature_index = find_first_unannotated_feature_index(features)
        return {
            "directory_id": directory_id,
            "file_name": file_name,
            "circuit_index": files.index(file_name),
            "total_circuits": len(files),
            "total_features": len(features),
            "first_unannotated_feature_index": first_unannotated_feature_index,
            "features": features,
            "graph_data": payload,
            "metadata": payload.get("metadata", {}) or {},
        }

    @router.get("/circuit_taxonomy/resume")
    def get_resume_target(
        directory_id: str,
        file_name: str | None = None,
        start_feature_index: int = 0,
    ):
        directory = resolve_directory(directory_id)
        files = list_files(directory)

        if not files:
            return {
                "directory_id": directory_id,
                "completed": True,
                "file_name": None,
                "circuit_index": None,
                "feature_index": None,
            }

        file_names = [path.name for path in files]
        start_circuit_index = 0
        if file_name:
            if file_name not in file_names:
                raise HTTPException(status_code=404, detail=f"Circuit file not found: {file_name}")
            start_circuit_index = file_names.index(file_name)

        for circuit_index in range(start_circuit_index, len(files)):
            path = files[circuit_index]
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            features = parse_features(payload)
            feature_index = find_first_unannotated_feature_index(
                features,
                start_feature_index if circuit_index == start_circuit_index else 0,
            )
            if feature_index is not None:
                return {
                    "directory_id": directory_id,
                    "completed": False,
                    "file_name": path.name,
                    "circuit_index": circuit_index,
                    "feature_index": feature_index,
                }

        return {
            "directory_id": directory_id,
            "completed": True,
            "file_name": file_names[-1],
            "circuit_index": len(file_names) - 1,
            "feature_index": None,
        }

    @router.get("/circuit_taxonomy/export_evidence")
    def export_evidence(
        directory_id: str,
        file_name: str | None = None,
        start_feature_index: int = 0,
        limit: int = 100,
        max_samples: int = 6,
        top_squares: int = 8,
        top_z: int = 12,
        include_annotated: bool = False,
        single_file_only: bool = False,
    ):
        content = evidence_items_to_jsonl(
            collect_evidence_items(
                directory_id=directory_id,
                file_name=file_name,
                start_feature_index=start_feature_index,
                limit=limit,
                max_samples=max_samples,
                top_squares=top_squares,
                top_z=top_z,
                include_annotated=include_annotated,
                single_file_only=single_file_only,
            )
        )
        return Response(
            content=content,
            media_type="application/x-ndjson",
            headers={"Content-Disposition": 'attachment; filename="circuit-taxonomy-evidence.jsonl"'},
        )

    @router.post("/circuit_taxonomy/save_directory_evidence")
    def save_directory_evidence(request: dict[str, Any]):
        directory_id = str(request.get("directory_id", "")).strip()
        if not directory_id:
            raise HTTPException(status_code=400, detail="directory_id is required")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_directory_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", directory_id).strip("_") or "directory"
        file_path = evidence_output_dir / safe_directory_id / f"taxonomy-evidence-{timestamp}.jsonl"

        evidence_items = collect_evidence_items(
            directory_id=directory_id,
            file_name=None,
            start_feature_index=0,
            limit=int(request["limit"]) if request.get("limit") is not None else None,
            max_samples=int(request.get("max_samples", 6)),
            top_squares=int(request.get("top_squares", 8)),
            top_z=int(request.get("top_z", 12)),
            include_annotated=bool(request.get("include_annotated", False)),
            single_file_only=False,
        )
        content = evidence_items_to_jsonl(evidence_items)

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
        except OSError as error:
            raise HTTPException(status_code=500, detail=f"Failed to save taxonomy evidence: {error}")

        return {
            "status": "saved",
            "directory_id": directory_id,
            "path": str(file_path),
            "relative_path": str(file_path.relative_to(repo_root)),
            "item_count": len(evidence_items),
        }

    @router.post("/circuit_taxonomy/save_all_circuit_evidence")
    def save_all_circuit_evidence(request: dict[str, Any]):
        directory_id = str(request.get("directory_id", "")).strip()
        if not directory_id:
            raise HTTPException(status_code=400, detail="directory_id is required")

        directory = resolve_directory(directory_id)
        files = list_files(directory)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_directory_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", directory_id).strip("_") or "directory"
        output_dir = evidence_output_dir / safe_directory_id / f"per-circuit-{timestamp}"
        max_samples = int(request.get("max_samples", 6))
        top_squares = int(request.get("top_squares", 8))
        top_z = int(request.get("top_z", 12))
        include_annotated = bool(request.get("include_annotated", True))

        saved_files: list[dict[str, Any]] = []
        total_items = 0
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            for path in files:
                evidence_items = collect_evidence_items(
                    directory_id=directory_id,
                    file_name=path.name,
                    start_feature_index=0,
                    limit=None,
                    max_samples=max_samples,
                    top_squares=top_squares,
                    top_z=top_z,
                    include_annotated=include_annotated,
                    single_file_only=True,
                )
                output_path = output_dir / f"{path.stem}.evidence.jsonl"
                output_path.write_text(evidence_items_to_jsonl(evidence_items), encoding="utf-8")
                total_items += len(evidence_items)
                saved_files.append(
                    {
                        "file_name": path.name,
                        "item_count": len(evidence_items),
                        "path": str(output_path),
                        "relative_path": str(output_path.relative_to(repo_root)),
                    }
                )
        except OSError as error:
            raise HTTPException(status_code=500, detail=f"Failed to save per-circuit taxonomy evidence: {error}")

        return {
            "status": "saved",
            "directory_id": directory_id,
            "path": str(output_dir),
            "relative_path": str(output_dir.relative_to(repo_root)),
            "file_count": len(saved_files),
            "item_count": total_items,
            "files": saved_files,
        }

    @router.get("/circuit_taxonomy/review_state")
    def get_review_state():
        if not review_state_path.exists():
            return {
                "status": "missing",
                "proposals": [],
                "active_review_index": 0,
                "updated_at": None,
            }

        try:
            with review_state_path.open("r", encoding="utf-8") as handle:
                state = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise HTTPException(status_code=500, detail=f"Failed to load taxonomy review state: {error}")

        proposals = state.get("proposals", [])
        if not isinstance(proposals, list):
            proposals = []

        return {
            "status": "loaded",
            "proposals": proposals,
            "active_review_index": int(state.get("active_review_index", 0) or 0),
            "updated_at": state.get("updated_at"),
        }

    @router.put("/circuit_taxonomy/review_state")
    def save_review_state(request: dict[str, Any]):
        try:
            state = coerce_review_state(request)
            write_review_state(review_state_path, state)
        except OSError as error:
            raise HTTPException(status_code=500, detail=f"Failed to save taxonomy review state: {error}")

        return {
            "status": "saved",
            "path": str(review_state_path),
            "relative_path": str(review_state_path.relative_to(repo_root)),
            "proposal_count": len(state["proposals"]),
            "active_review_index": state["active_review_index"],
            "updated_at": state["updated_at"],
        }

    @router.post("/circuit_taxonomy/review_state/snapshot")
    def snapshot_review_state(request: dict[str, Any] | None = None):
        try:
            state = coerce_review_state(request)
            write_review_state(review_state_path, state)

            saved_at = datetime.now(timezone.utc)
            timestamp = saved_at.strftime("%Y%m%dT%H%M%SZ")
            snapshot_state = {
                **state,
                "saved_at": saved_at.isoformat(),
                "source_path": str(review_state_path.relative_to(repo_root)),
            }
            snapshot_path = review_state_snapshot_dir / f"review-state-{timestamp}.json"
            suffix = 1
            while snapshot_path.exists():
                snapshot_path = review_state_snapshot_dir / f"review-state-{timestamp}-{suffix}.json"
                suffix += 1
            write_review_state(snapshot_path, snapshot_state)
        except OSError as error:
            raise HTTPException(status_code=500, detail=f"Failed to snapshot taxonomy review state: {error}")

        return {
            "status": "snapshot_saved",
            "path": str(review_state_path),
            "relative_path": str(review_state_path.relative_to(repo_root)),
            "snapshot_path": str(snapshot_path),
            "snapshot_relative_path": str(snapshot_path.relative_to(repo_root)),
            "proposal_count": len(state["proposals"]),
            "active_review_index": state["active_review_index"],
            "updated_at": state["updated_at"],
            "saved_at": snapshot_state["saved_at"],
        }

    @router.post("/circuit_taxonomy/review_state/reset")
    def reset_review_state():
        try:
            state = coerce_review_state({"proposals": [], "active_review_index": 0})
            write_review_state(review_state_path, state)
        except OSError as error:
            raise HTTPException(status_code=500, detail=f"Failed to reset taxonomy review state: {error}")

        return {
            "status": "reset",
            "path": str(review_state_path),
            "relative_path": str(review_state_path.relative_to(repo_root)),
            "proposal_count": 0,
            "active_review_index": 0,
            "updated_at": state["updated_at"],
        }

    def annotate_feature_item(dictionary_name: str, feature_index: int, taxonomy: str, overwrite: bool) -> dict[str, Any]:
        try:
            feature_index = int(feature_index)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="feature_index must be an integer")

        if taxonomy not in CIRCUIT_TAXONOMY_LABELS:
            raise HTTPException(status_code=400, detail=f"Unsupported taxonomy label: {taxonomy}")
        if not dictionary_name:
            raise HTTPException(status_code=400, detail="dictionary_name is required")

        feature = client.get_feature(
            sae_name=dictionary_name,
            sae_series=sae_series,
            index=feature_index,
        )
        if feature is None:
            raise HTTPException(
                status_code=404,
                detail=f"Feature {feature_index} not found in SAE {dictionary_name}",
            )

        existing_interpretation = feature.interpretation if isinstance(feature.interpretation, dict) else None
        existing_text = ""
        if existing_interpretation is not None:
            existing_text = str(existing_interpretation.get("text", "") or "")
        existing_prefix = extract_prefix(existing_text)

        if existing_prefix == taxonomy:
            return {
                "status": "unchanged",
                "taxonomy": taxonomy,
                "existing_taxonomy": existing_prefix,
                "interpretation": existing_interpretation,
            }

        if existing_prefix and existing_prefix != taxonomy and not overwrite:
            return {
                "status": "conflict",
                "taxonomy": taxonomy,
                "existing_taxonomy": existing_prefix,
                "existing_text": existing_text,
                "proposed_text": apply_prefix(existing_text, taxonomy),
            }

        new_text = apply_prefix(existing_text, taxonomy)
        updated_interpretation = dict(existing_interpretation or {})
        updated_interpretation["text"] = new_text
        updated_interpretation["method"] = updated_interpretation.get("method") or "taxonomy_label"
        updated_interpretation["validation"] = updated_interpretation.get("validation") or []

        try:
            client.update_feature(
                sae_name=dictionary_name,
                sae_series=sae_series,
                feature_index=feature_index,
                update_data={"interpretation": updated_interpretation},
            )
        except Exception as update_error:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save taxonomy interpretation: {str(update_error)}",
            )

        return {
            "status": "updated",
            "taxonomy": taxonomy,
            "existing_taxonomy": existing_prefix,
            "interpretation": updated_interpretation,
            "overwritten": existing_prefix is not None and existing_prefix != taxonomy,
        }

    @router.post("/circuit_taxonomy/annotate")
    def annotate_feature(request: dict[str, Any]):
        dictionary_name = str(request.get("dictionary_name", "")).strip()
        taxonomy = str(request.get("taxonomy", "")).strip()
        overwrite = bool(request.get("overwrite", False))
        try:
            feature_index = int(request.get("feature_index"))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="feature_index must be an integer")

        return annotate_feature_item(
            dictionary_name=dictionary_name,
            feature_index=feature_index,
            taxonomy=taxonomy,
            overwrite=overwrite,
        )

    @router.post("/circuit_taxonomy/annotate_batch")
    def annotate_batch(request: dict[str, Any]):
        raw_items = request.get("items", [])
        if not isinstance(raw_items, list):
            raise HTTPException(status_code=400, detail="items must be a list")

        overwrite = bool(request.get("overwrite", True))
        results: list[dict[str, Any]] = []
        counts: dict[str, int] = {
            "updated": 0,
            "unchanged": 0,
            "conflict": 0,
            "error": 0,
        }

        for raw_index, raw_item in enumerate(raw_items):
            if not isinstance(raw_item, dict):
                counts["error"] += 1
                results.append(
                    {
                        "index": raw_index,
                        "status": "error",
                        "error": "item must be an object",
                    }
                )
                continue

            dictionary_name = str(raw_item.get("dictionary_name") or raw_item.get("dictionaryName") or "").strip()
            taxonomy = str(raw_item.get("taxonomy", "")).strip()
            try:
                feature_index = int(raw_item.get("feature_index", raw_item.get("featureIndex")))
                result = annotate_feature_item(
                    dictionary_name=dictionary_name,
                    feature_index=feature_index,
                    taxonomy=taxonomy,
                    overwrite=overwrite,
                )
                status = str(result.get("status", "updated"))
                counts[status] = counts.get(status, 0) + 1
                results.append(
                    {
                        "index": raw_index,
                        "dictionary_name": dictionary_name,
                        "feature_index": feature_index,
                        **result,
                    }
                )
            except HTTPException as error:
                counts["error"] += 1
                results.append(
                    {
                        "index": raw_index,
                        "dictionary_name": dictionary_name,
                        "feature_index": raw_item.get("feature_index", raw_item.get("featureIndex")),
                        "taxonomy": taxonomy,
                        "status": "error",
                        "error": error.detail,
                    }
                )
            except Exception as error:
                counts["error"] += 1
                results.append(
                    {
                        "index": raw_index,
                        "dictionary_name": dictionary_name,
                        "feature_index": raw_item.get("feature_index", raw_item.get("featureIndex")),
                        "taxonomy": taxonomy,
                        "status": "error",
                        "error": str(error),
                    }
                )

        return {
            "status": "completed",
            "item_count": len(raw_items),
            "counts": counts,
            "results": results,
        }

    @router.post("/circuit_taxonomy/import_proposals")
    def import_proposals(request: dict[str, Any]):
        """Apply proposal taxonomies only to features with empty interpretations.

        This endpoint intentionally has stricter preservation semantics than
        ``annotate_batch``: any non-empty interpretation text is immutable,
        regardless of whether it already has a taxonomy prefix.
        """
        raw_items = request.get("items", [])
        if not isinstance(raw_items, list):
            raise HTTPException(status_code=400, detail="items must be a list")

        parsed_items: dict[tuple[str, int], dict[str, Any]] = {}
        conflicting_keys: set[tuple[str, int]] = set()
        results: list[dict[str, Any]] = []
        counts: dict[str, int] = {
            "updated": 0,
            "skipped_existing": 0,
            "duplicate": 0,
            "conflict": 0,
            "error": 0,
        }

        for raw_index, raw_item in enumerate(raw_items):
            if not isinstance(raw_item, dict):
                counts["error"] += 1
                results.append({"index": raw_index, "status": "error", "error": "item must be an object"})
                continue

            dictionary_name = str(raw_item.get("dictionary_name") or raw_item.get("dictionaryName") or "").strip()
            taxonomy = str(raw_item.get("taxonomy", "")).strip().strip("[]")
            try:
                feature_index = int(raw_item.get("feature_index", raw_item.get("featureIndex")))
            except (TypeError, ValueError):
                counts["error"] += 1
                results.append({
                    "index": raw_index,
                    "status": "error",
                    "dictionary_name": dictionary_name,
                    "error": "feature_index must be an integer",
                })
                continue

            if not dictionary_name:
                counts["error"] += 1
                results.append({"index": raw_index, "status": "error", "error": "dictionary_name is required"})
                continue
            if taxonomy not in CIRCUIT_TAXONOMY_LABELS:
                counts["error"] += 1
                results.append({
                    "index": raw_index,
                    "status": "error",
                    "dictionary_name": dictionary_name,
                    "feature_index": feature_index,
                    "taxonomy": taxonomy,
                    "error": f"Unsupported taxonomy label: {taxonomy}",
                })
                continue

            item_key = (dictionary_name, feature_index)
            previous = parsed_items.get(item_key)
            if previous is not None:
                if previous["taxonomy"] != taxonomy:
                    conflicting_keys.add(item_key)
                else:
                    counts["duplicate"] += 1
                continue
            parsed_items[item_key] = {
                "index": raw_index,
                "dictionary_name": dictionary_name,
                "feature_index": feature_index,
                "taxonomy": taxonomy,
            }

        for item_key, item in parsed_items.items():
            if item_key in conflicting_keys:
                counts["conflict"] += 1
                results.append({
                    **item,
                    "status": "conflict",
                    "error": "The uploaded proposals contain different taxonomy labels for this feature",
                })
                continue

            feature = client.get_feature(
                sae_name=item["dictionary_name"],
                sae_series=sae_series,
                index=item["feature_index"],
            )
            if feature is None:
                counts["error"] += 1
                results.append({**item, "status": "error", "error": "Feature not found"})
                continue

            raw_interpretation = feature.interpretation
            if isinstance(raw_interpretation, dict):
                existing_text = str(raw_interpretation.get("text", "") or "").strip()
            elif isinstance(raw_interpretation, str):
                existing_text = raw_interpretation.strip()
            else:
                existing_text = ""

            if existing_text:
                counts["skipped_existing"] += 1
                results.append({
                    **item,
                    "status": "skipped_existing",
                    "existing_text": existing_text,
                })
                continue

            updated_interpretation = dict(raw_interpretation) if isinstance(raw_interpretation, dict) else {}
            updated_interpretation["text"] = f"[{item['taxonomy']}]"
            updated_interpretation["method"] = updated_interpretation.get("method") or "taxonomy_label"
            updated_interpretation["validation"] = updated_interpretation.get("validation") or []
            try:
                client.update_feature(
                    sae_name=item["dictionary_name"],
                    sae_series=sae_series,
                    feature_index=item["feature_index"],
                    update_data={"interpretation": updated_interpretation},
                )
            except Exception as update_error:
                counts["error"] += 1
                results.append({**item, "status": "error", "error": str(update_error)})
                continue

            counts["updated"] += 1
            results.append({
                **item,
                "status": "updated",
                "interpretation": updated_interpretation,
            })

        return {
            "status": "completed",
            "item_count": len(raw_items),
            "unique_feature_count": len(parsed_items),
            "counts": counts,
            "results": results,
        }

    return router
