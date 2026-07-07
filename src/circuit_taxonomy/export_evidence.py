from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pymongo
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lm_saes.config import MongoDBConfig  # noqa: E402
from lm_saes.database import MongoClient  # noqa: E402
from lm_saes.resource_loaders import load_dataset_shard  # noqa: E402
from chess_utils import get_move_from_model  # noqa: E402


TAXONOMY_LABELS = [
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

TAXONOMY_PREFIX_RE = re.compile(r"^\[(%s)\]\s*" % "|".join(re.escape(label) for label in TAXONOMY_LABELS))
FEN_RE = re.compile(
    r"\b(?:[pnbrqkPNBRQK1-8]+/){7}[pnbrqkPNBRQK1-8]+\s+[wb]\s+(?:K?Q?k?q?|-)\s+(?:[a-h][36]|-)\s+\d+\s+\d+\b"
)


def make_serializable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "detach") and hasattr(obj, "cpu"):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def board_index_to_square(index: int, side_to_move: str | None = None) -> str:
    if index < 0 or index > 63:
        return f"idx{index}"
    file_name = "abcdefgh"[index % 8]
    if side_to_move == "w":
        rank = 1 + (index // 8)
    else:
        rank = 8 - (index // 8)
    return f"{file_name}{rank}"


def extract_fen(value: Any) -> str | None:
    if isinstance(value, dict):
        fen = value.get("fen")
        if isinstance(fen, str) and fen.strip():
            return fen.strip()
        text = value.get("text")
        if isinstance(text, str):
            match = FEN_RE.search(text)
            if match:
                return match.group(0)
    if isinstance(value, str):
        match = FEN_RE.search(value)
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


def top_square_entries(indices: Any, values: Any, limit: int, side_to_move: str | None = None) -> list[dict[str, Any]]:
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
            "square": board_index_to_square(index, side_to_move),
            "value": round(value, 6),
        }
        for index, value in index_values[:limit]
    ]


def top_z_pairs(
    z_pattern_indices: Any,
    z_pattern_values: Any,
    limit: int,
    side_to_move: str | None = None,
) -> list[dict[str, Any]]:
    if z_pattern_indices is None or z_pattern_values is None:
        return []
    indices = np.asarray(z_pattern_indices)
    values = np.asarray(z_pattern_values).reshape(-1)
    if indices.size == 0 or values.size == 0:
        return []

    pairs: list[tuple[int, int, float]] = []
    if indices.ndim == 2 and indices.shape[0] >= 2:
        iterator = zip(indices[0].reshape(-1), indices[1].reshape(-1), values)
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
            "source_square": board_index_to_square(source, side_to_move),
            "target_square": board_index_to_square(target, side_to_move),
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


class EvidenceExporter:
    def __init__(
        self,
        *,
        client: MongoClient,
        sae_series: str,
        include_model_summary: bool,
        model_name: str,
        model_device: str | None,
        max_dataset_cache: int = 16,
    ):
        self.client = client
        self.sae_series = sae_series
        self.include_model_summary = include_model_summary
        self.model_name = model_name
        self.model_device = model_device
        self._model: Any = None
        self._model_load_error: str | None = None
        self._board_model_summary_cache: dict[str, dict[str, Any]] = {}
        self._get_dataset = lru_cache(maxsize=max_dataset_cache)(self._load_dataset)

    def _load_dataset(self, name: str, shard_idx: int, n_shards: int):
        cfg = self.client.get_dataset_cfg(name)
        if cfg is None:
            raise ValueError(f"Dataset {name} not found")
        return load_dataset_shard(cfg, shard_idx, n_shards)

    def get_feature_with_series(self, dictionary_name: str, feature_index: int):
        feature = self.client.get_feature(sae_name=dictionary_name, sae_series=self.sae_series, index=feature_index)
        return feature, self.sae_series

    def get_model(self):
        if self._model is not None:
            return self._model
        if self._model_load_error is not None:
            raise RuntimeError(self._model_load_error)

        try:
            from transformer_lens import HookedTransformer

            print(f"Loading BT4 model for WDL/top_moves: {self.model_name}", flush=True)
            model = HookedTransformer.from_pretrained_no_processing(
                self.model_name,
                dtype=torch.float32,
            ).eval()
            if self.model_device:
                model = model.to(self.model_device)
            self._model = model
            return self._model
        except Exception as error:
            self._model_load_error = str(error)
            raise

    def get_board_model_summary(self, fen: str) -> dict[str, Any]:
        if not self.include_model_summary:
            return {
                "top_moves": [],
                "model_analysis_error": "Model summary disabled.",
            }
        if fen in self._board_model_summary_cache:
            return self._board_model_summary_cache[fen]

        summary: dict[str, Any] = {}
        try:
            model = self.get_model()
            with torch.no_grad():
                output, _ = model.run_with_cache(fen, prepend_bos=False)

            if isinstance(output, (list, tuple)) and len(output) >= 2:
                wdl_tensor = output[1]
                if tuple(wdl_tensor.shape) == (1, 3):
                    wdl = [
                        float(wdl_tensor[0][0].detach().cpu().item()),
                        float(wdl_tensor[0][1].detach().cpu().item()),
                        float(wdl_tensor[0][2].detach().cpu().item()),
                    ]
                    summary["wdl"] = {
                        "current_player_win": wdl[0],
                        "draw": wdl[1],
                        "current_player_loss": wdl[2],
                    }
                    summary["value"] = float(wdl[0] - wdl[2])

            legal_moves = get_move_from_model(model, fen, return_list=True)
            if isinstance(legal_moves, list) and legal_moves:
                summary["top_moves"] = [
                    {
                        "uci": uci,
                    }
                    for uci, _ in legal_moves[:5]
                ]
            else:
                summary["top_moves"] = []

        except Exception as error:
            summary = {
                **summary,
                "top_moves": summary.get("top_moves", []),
                "model_analysis_error": str(error),
            }

        self._board_model_summary_cache[fen] = summary
        return summary

    def build_feature_evidence(
        self,
        *,
        directory_id: str,
        circuit_detail: dict[str, Any],
        feature_ref: dict[str, Any],
        feature_index_in_circuit: int,
        max_samples: int,
        top_squares: int,
        top_z: int,
        include_dataset: bool,
    ) -> dict[str, Any]:
        dictionary_name = str(feature_ref.get("dictionary_name", "")).strip()
        feature_index = int(feature_ref.get("feature_index", -1))
        feature, _resolved_series = self.get_feature_with_series(dictionary_name, feature_index)
        if feature is None:
            raise ValueError(f"Feature {feature_index} not found in SAE {dictionary_name}/{self.sae_series}")

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
                    if include_dataset and dataset_name is not None and context_idx is not None:
                        try:
                            row = self._get_dataset(str(dataset_name), shard_idx, n_shards)[int(context_idx)]
                            dataset_row = make_serializable(row)
                            fen = extract_fen(dataset_row)
                        except Exception as dataset_error:
                            dataset_row = {"error": f"Failed to load dataset row: {dataset_error}"}

                    model_summary = self.get_board_model_summary(fen) if fen else {}
                    side_to_move = fen.split()[1] if fen and len(fen.split()) >= 2 else None
                    sample_summaries.append(
                        {
                            "fen": fen,
                            "wdl": model_summary.get("wdl"),
                            "value": model_summary.get("value"),
                            "top_moves": model_summary.get("top_moves", []),
                            "top_activated_squares": top_square_entries(
                                act_indices,
                                act_values,
                                top_squares,
                                side_to_move,
                            ),
                            "top_z_pairs": top_z_pairs(
                                z_indices,
                                z_values,
                                top_z,
                                side_to_move,
                            ),
                        }
                    )
                if len(sample_summaries) >= max_samples:
                    break

        interpretation = feature.interpretation if isinstance(feature.interpretation, dict) else {}
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
            "feature_stats": {
                "max_feature_act": analysis.max_feature_acts if analysis is not None else None,
                "act_times": analysis.act_times if analysis is not None else None,
                "n_analyzed_tokens": analysis.n_analyzed_tokens if analysis is not None else None,
            },
            "top_activation_samples": sample_summaries,
        }


def iter_circuit_files(directory: Path, file_glob: str) -> list[Path]:
    return sorted(path for path in directory.glob(file_glob) if path.is_file())


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(make_serializable(item), ensure_ascii=False) + "\n")


def evidence_file_has_model_summary(path: Path, max_items: int = 20) -> bool:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                if line_index >= max_items:
                    break
                if not line.strip():
                    continue
                item = json.loads(line)
                samples = item.get("top_activation_samples")
                if not isinstance(samples, list):
                    continue
                for sample in samples:
                    if not isinstance(sample, dict):
                        continue
                    if sample.get("wdl") is not None:
                        return True
                    top_moves = sample.get("top_moves")
                    if isinstance(top_moves, list) and len(top_moves) > 0:
                        return True
    except (OSError, json.JSONDecodeError):
        return False
    return False


def export_circuit_file(
    *,
    exporter: EvidenceExporter,
    directory_id: str,
    file_path: Path,
    circuit_index: int,
    total_circuits: int,
    output_path: Path,
    max_samples: int,
    top_squares: int,
    top_z: int,
    include_dataset: bool,
    limit_features: int | None,
    show_progress: bool,
) -> int:
    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    features = parse_features(payload)
    if limit_features is not None:
        features = features[:limit_features]

    circuit_detail = {
        "directory_id": directory_id,
        "file_name": file_path.name,
        "circuit_index": circuit_index,
        "total_circuits": total_circuits,
        "total_features": len(features),
        "features": features,
        "metadata": payload.get("metadata", {}) or {},
    }

    items = []
    feature_iter = enumerate(features)
    if show_progress:
        feature_iter = tqdm(
            feature_iter,
            total=len(features),
            desc=file_path.stem,
            unit="feature",
            leave=False,
        )
    for feature_index_in_circuit, feature_ref in feature_iter:
        try:
            items.append(
                exporter.build_feature_evidence(
                    directory_id=directory_id,
                    circuit_detail=circuit_detail,
                    feature_ref=feature_ref,
                    feature_index_in_circuit=feature_index_in_circuit,
                    max_samples=max_samples,
                    top_squares=top_squares,
                    top_z=top_z,
                    include_dataset=include_dataset,
                )
            )
        except Exception as error:
            items.append(
                {
                    "directory_id": directory_id,
                    "file_name": circuit_detail.get("file_name"),
                    "circuit_index": circuit_index,
                    "feature_index_in_circuit": feature_index_in_circuit,
                    "dictionary_name": feature_ref.get("dictionary_name"),
                    "feature_index": feature_ref.get("feature_index"),
                    "layer": feature_ref.get("layer"),
                    "feature_type": feature_ref.get("feature_type"),
                    "node_id": feature_ref.get("node_id"),
                    "label": feature_ref.get("label"),
                    "error": str(error),
                    "top_activation_samples": [],
                }
            )

    write_jsonl(output_path, items)
    return len(items)


def sample_feature_refs(files: list[Path], max_files: int = 3, max_features: int = 50) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for file_path in files[:max_files]:
        with file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        refs.extend(parse_features(payload)[:max_features])
        if len(refs) >= max_features:
            return refs[:max_features]
    return refs


def preflight_mongo(
    *,
    mongo_uri: str,
    mongo_db: str,
    requested_sae_series: str | None,
    feature_refs: list[dict[str, Any]],
    timeout_ms: int,
) -> str:
    if not feature_refs:
        raise SystemExit("Mongo preflight failed: no feature refs were parsed from circuit files.")

    client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=timeout_ms)
    try:
        client.admin.command("ping")
        collection = client[mongo_db]["features"]
        series_counts: dict[str, int] = {}
        missing: list[str] = []
        for feature_ref in feature_refs:
            dictionary_name = str(feature_ref.get("dictionary_name", "")).strip()
            feature_index = int(feature_ref.get("feature_index", -1))
            query: dict[str, Any] = {
                "sae_name": dictionary_name,
                "index": feature_index,
            }
            if requested_sae_series:
                query["sae_series"] = requested_sae_series
            docs = list(collection.find(query, {"sae_series": 1}).limit(5))
            if not docs:
                missing.append(f"{dictionary_name}#{feature_index}")
                continue
            for doc in docs:
                series = doc.get("sae_series")
                if isinstance(series, str) and series:
                    series_counts[series] = series_counts.get(series, 0) + 1
        if not series_counts:
            examples = ", ".join(missing[:5])
            raise SystemExit(
                "Mongo preflight failed: none of the sampled circuit features were found in "
                f"{mongo_db}.features. Examples: {examples}"
            )

        selected_series = requested_sae_series
        if requested_sae_series and requested_sae_series not in series_counts:
            counts_text = ", ".join(f"{series}:{count}" for series, count in sorted(series_counts.items()))
            raise SystemExit(
                f"Mongo preflight failed: requested --sae-series {requested_sae_series!r} was not found for sampled features. "
                f"Available sampled series: {counts_text}"
            )
        if not selected_series:
            selected_series = max(series_counts.items(), key=lambda item: item[1])[0]

        counts_text = ", ".join(f"{series}:{count}" for series, count in sorted(series_counts.items()))
        print(f"Mongo preflight OK: {mongo_db}.features reachable; sampled sae_series counts: {counts_text}")
        print(f"Using sae_series={selected_series!r}")
        if missing:
            raise SystemExit(
                f"Mongo preflight failed: {len(missing)} sampled feature(s) were not found under "
                f"sae_series={selected_series!r}. First examples: {', '.join(missing[:5])}"
            )
        return selected_series
    except pymongo.errors.PyMongoError as error:
        raise SystemExit(
            "Mongo preflight failed: could not connect/query MongoDB. "
            f"uri={mongo_uri!r}, db={mongo_db!r}, error={error}"
        ) from error
    finally:
        client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export circuit taxonomy evidence without using the backend API.")
    parser.add_argument(
        "--directory",
        default="exp/60ICLR/circuits/random_data/results/results_4096/k_30_e_16",
        help="Circuit JSON directory, absolute or relative to repo root.",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to outputs/circuit_taxonomy_evidence/<directory>/per-circuit-<timestamp>.")
    parser.add_argument("--sae-series", default=os.environ.get("SAE_SERIES"))
    parser.add_argument("--mongo-uri", default=os.environ.get("MONGO_URI", "mongodb://10.244.36.152:27017"))
    parser.add_argument("--mongo-db", default=os.environ.get("MONGO_DB", "mechinterp"))
    parser.add_argument("--mongo-timeout-ms", type=int, default=5000)
    parser.add_argument("--no-mongo-preflight", action="store_true", help="Skip Mongo connectivity and sae_series preflight.")
    parser.add_argument("--file-glob", default="*.json")
    parser.add_argument("--max-samples", type=int, default=6)
    parser.add_argument("--top-squares", type=int, default=8)
    parser.add_argument("--top-z", type=int, default=12)
    parser.add_argument("--model-name", default="lc0/BT4-1024x15x32h")
    parser.add_argument("--model-device", default=None, help="Optional torch device for the BT4 model, e.g. cuda or cpu.")
    parser.add_argument("--no-model-summary", action="store_true", help="Do not compute WDL/value/top_moves from the BT4 model.")
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--limit-features", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true", help="Skip output files that already exist.")
    parser.add_argument("--no-dataset", action="store_true", help="Do not load dataset rows/FENs; fastest mode.")
    parser.add_argument("--no-progress", action="store_true", help="Disable per-circuit tqdm feature progress bars.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    directory = Path(args.directory)
    if not directory.is_absolute():
        directory = (REPO_ROOT / directory).resolve()
    if not directory.is_dir():
        raise SystemExit(f"Circuit directory does not exist: {directory}")

    directory_id = str(directory.relative_to(REPO_ROOT)).replace("\\", "/")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = (REPO_ROOT / output_dir).resolve()
    else:
        safe_directory_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", directory_id).strip("_") or "directory"
        output_dir = REPO_ROOT / "outputs" / "circuit_taxonomy_evidence" / safe_directory_id / f"per-circuit-{timestamp}"

    files = iter_circuit_files(directory, args.file_glob)
    if args.limit_files is not None:
        files = files[: args.limit_files]
    if not files:
        raise SystemExit(f"No circuit files matched {args.file_glob} in {directory}")

    sae_series = args.sae_series
    if not args.no_mongo_preflight:
        sae_series = preflight_mongo(
            mongo_uri=args.mongo_uri,
            mongo_db=args.mongo_db,
            requested_sae_series=sae_series,
            feature_refs=sample_feature_refs(files),
            timeout_ms=args.mongo_timeout_ms,
        )
    if not sae_series:
        raise SystemExit("No sae_series selected. Set SAE_SERIES, pass --sae-series, or keep Mongo preflight enabled.")

    client = MongoClient(MongoDBConfig(mongo_uri=args.mongo_uri, mongo_db=args.mongo_db))
    exporter = EvidenceExporter(
        client=client,
        sae_series=sae_series,
        include_model_summary=not args.no_model_summary,
        model_name=args.model_name,
        model_device=args.model_device,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    total_items = 0
    print(f"Exporting {len(files)} circuit files from {directory}")
    print(f"Output directory: {output_dir}")
    print(
        f"SAE series: {sae_series}; include_dataset={not args.no_dataset}; "
        f"include_model_summary={not args.no_model_summary}"
    )

    for circuit_index, file_path in enumerate(files):
        output_path = output_dir / f"{file_path.stem}.evidence.jsonl"
        if args.skip_existing and output_path.exists() and (args.no_model_summary or evidence_file_has_model_summary(output_path)):
            print(f"[{circuit_index + 1}/{len(files)}] skip existing {output_path.name}")
            continue
        if args.skip_existing and output_path.exists() and not args.no_model_summary:
            print(
                f"[{circuit_index + 1}/{len(files)}] regenerating {output_path.name} because existing file lacks WDL/top_moves",
                flush=True,
            )

        print(f"[{circuit_index + 1}/{len(files)}] exporting {file_path.name}", flush=True)
        try:
            item_count = export_circuit_file(
                exporter=exporter,
                directory_id=directory_id,
                file_path=file_path,
                circuit_index=circuit_index,
                total_circuits=len(files),
                output_path=output_path,
                max_samples=max(1, min(args.max_samples, 20)),
                top_squares=max(1, min(args.top_squares, 64)),
                top_z=max(1, min(args.top_z, 64)),
                include_dataset=not args.no_dataset,
                limit_features=args.limit_features,
                show_progress=not args.no_progress,
            )
        except Exception as error:
            error_path = output_dir / f"{file_path.stem}.error.txt"
            error_path.write_text(str(error), encoding="utf-8")
            print(f"  ERROR: {error}; wrote {error_path}", flush=True)
            manifest.append({"file_name": file_path.name, "status": "error", "error": str(error)})
            continue

        total_items += item_count
        manifest.append(
            {
                "file_name": file_path.name,
                "status": "saved",
                "item_count": item_count,
                "relative_path": str(output_path.relative_to(REPO_ROOT)),
            }
        )
        print(f"  wrote {item_count} items -> {output_path.name}", flush=True)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "directory_id": directory_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "file_count": len(manifest),
                "item_count": total_items,
                "files": manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Done. Wrote {total_items} evidence items. Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
