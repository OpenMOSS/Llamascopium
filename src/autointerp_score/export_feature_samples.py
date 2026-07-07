"""Export top-activation evidence for LoRSA, Transcoder, or MLP features.

Each JSONL row describes one feature and is suitable for manual or LLM-based
autointerp consistency/complexity scoring.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lm_saes.config import MongoDBConfig  # noqa: E402
from lm_saes.database import MongoClient  # noqa: E402
from lm_saes.resource_loaders import load_dataset_shard  # noqa: E402

FEATURE_PATTERNS = {
    "lorsa": re.compile(r"^BT4_lorsa_L(?P<layer>\d+)A(?:_(?P<suffix>.+))?$"),
    "transcoder": re.compile(r"^BT4_tc_L(?P<layer>\d+)M(?:_(?P<suffix>.+))?$"),
    "mlp": re.compile(r"^BT4_mlp_L(?P<layer>\d+)$"),
}
FEN_RE = re.compile(
    r"\b(?:[pnbrqkPNBRQK1-8]+/){7}[pnbrqkPNBRQK1-8]+\s+[wb]\s+(?:K?Q?k?q?|-)\s+(?:[a-h][36]|-)\s+\d+\s+\d+\b"
)


def normalize_feature_type(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    aliases = {"tc": "transcoder", "cross_layer_transcoder": "transcoder", "sae": "mlp"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in FEATURE_PATTERNS:
        raise ValueError(f"Unsupported feature type: {value}")
    return normalized


def parse_sae_name(sae_name: str) -> tuple[str, int] | None:
    for feature_type, pattern in FEATURE_PATTERNS.items():
        match = pattern.fullmatch(sae_name)
        if match:
            return feature_type, int(match.group("layer"))
    return None


def make_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): make_serializable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_serializable(child) for child in value]
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().tolist()
    return value


def extract_fen(row: Any) -> str | None:
    if isinstance(row, dict):
        fen = row.get("fen")
        if isinstance(fen, str) and fen.strip():
            return fen.strip()
        text = row.get("text")
        if isinstance(text, str):
            match = FEN_RE.search(text)
            if match:
                return match.group(0)
    return None


def board_index_to_square(index: int, side_to_move: str | None) -> str:
    if not 0 <= index < 64:
        return f"idx{index}"
    file_name = "abcdefgh"[index % 8]
    rank = 1 + index // 8 if side_to_move == "w" else 8 - index // 8
    return f"{file_name}{rank}"


def top_square_entries(indices: Any, values: Any, limit: int, side_to_move: str | None) -> list[dict[str, Any]]:
    entries = []
    for raw_index, raw_value in zip(np.asarray(indices).reshape(-1), np.asarray(values).reshape(-1)):
        index, value = int(raw_index), float(raw_value)
        if 0 <= index < 64:
            entries.append((index, value))
    entries.sort(key=lambda item: abs(item[1]), reverse=True)
    return [
        {"index": index, "square": board_index_to_square(index, side_to_move), "value": round(value, 6)}
        for index, value in entries[:limit]
    ]


def top_z_pairs(indices: Any, values: Any, limit: int, side_to_move: str | None) -> list[dict[str, Any]]:
    if indices is None or values is None:
        return []
    index_array = np.asarray(indices)
    value_array = np.asarray(values).reshape(-1)
    if index_array.ndim != 2 or index_array.shape[0] < 2:
        return []
    pairs = []
    for raw_source, raw_target, raw_value in zip(index_array[0], index_array[1], value_array):
        source, target, value = int(raw_source), int(raw_target), float(raw_value)
        if 0 <= source < 64 and 0 <= target < 64:
            pairs.append((source, target, value))
    pairs.sort(key=lambda item: abs(item[2]), reverse=True)
    return [
        {
            "source_index": source,
            "target_index": target,
            "source_square": board_index_to_square(source, side_to_move),
            "target_square": board_index_to_square(target, side_to_move),
            "value": round(value, 6),
        }
        for source, target, value in pairs[:limit]
    ]


def iter_sparse_samples(sampling: Any) -> Iterable[tuple[int, Any, Any, Any, Any]]:
    act_indices = np.asarray(sampling.feature_acts_indices)
    act_values = np.asarray(sampling.feature_acts_values)
    if act_indices.ndim < 2 or act_indices.shape[0] < 2 or act_indices.shape[1] == 0:
        return
    sample_ids = act_indices[0]
    z_indices = np.asarray(sampling.z_pattern_indices) if sampling.z_pattern_indices is not None else None
    z_values = np.asarray(sampling.z_pattern_values) if sampling.z_pattern_values is not None else None
    for sample_id in dict.fromkeys(int(item) for item in sample_ids.tolist()):
        mask = sample_ids == sample_id
        sample_z_indices = sample_z_values = None
        if z_indices is not None and z_values is not None and z_indices.ndim >= 2 and z_indices.size:
            z_mask = z_indices[0] == sample_id
            sample_z_indices = z_indices[1:, z_mask]
            sample_z_values = z_values[z_mask]
        yield sample_id, act_indices[1, mask], act_values[mask], sample_z_indices, sample_z_values


class FeatureSampleExporter:
    def __init__(self, client: MongoClient, sae_series: str):
        self.client = client
        self.sae_series = sae_series
        self._get_dataset = lru_cache(maxsize=16)(self._load_dataset)

    def _load_dataset(self, name: str, shard_idx: int, n_shards: int):
        cfg = self.client.get_dataset_cfg(name)
        if cfg is None:
            raise ValueError(f"Dataset {name} not found")
        return load_dataset_shard(cfg, shard_idx, n_shards)

    def discover_saes(
        self,
        feature_type: str,
        analysis_name: str,
        sae_name: str | None,
        layer: int | None,
    ) -> list[str]:
        query = {"name": analysis_name, "sae_series": self.sae_series}
        if sae_name:
            query["sae_name"] = sae_name
        names = sorted({str(item["sae_name"]) for item in self.client.analysis_collection.find(query, {"sae_name": 1})})
        selected = []
        for name in names:
            parsed = parse_sae_name(name)
            if parsed is None:
                continue
            parsed_type, parsed_layer = parsed
            if parsed_type == feature_type and (layer is None or layer == parsed_layer):
                selected.append(name)
        if sae_name and not selected:
            raise ValueError(
                f"Analysis {analysis_name!r} for {sae_name!r}/{self.sae_series!r} was not found or does not match {feature_type}"
            )
        return selected

    def build_sample(
        self,
        sampling: Any,
        sample_id: int,
        act_indices: Any,
        act_values: Any,
        z_indices: Any,
        z_values: Any,
        top_squares: int,
        top_z: int,
    ) -> dict[str, Any]:
        context_idx = int(sampling.context_idx[sample_id]) if sample_id < len(sampling.context_idx) else None
        dataset_name = str(sampling.dataset_name[sample_id]) if sample_id < len(sampling.dataset_name) else None
        shard_idx = (
            int(sampling.shard_idx[sample_id])
            if sampling.shard_idx is not None and sample_id < len(sampling.shard_idx)
            else 0
        )
        n_shards = (
            int(sampling.n_shards[sample_id])
            if sampling.n_shards is not None and sample_id < len(sampling.n_shards)
            else 1
        )
        row: Any = {}
        dataset_error = None
        if dataset_name is not None and context_idx is not None:
            try:
                row = self._get_dataset(dataset_name, shard_idx, n_shards)[context_idx]
            except Exception as error:
                dataset_error = str(error)
        row = make_serializable(row)
        fen = extract_fen(row)
        side_to_move = fen.split()[1] if fen and len(fen.split()) > 1 else None
        sample = {
            "fen": fen,
            "context_idx": context_idx,
            "dataset_name": dataset_name,
            "shard_idx": shard_idx,
            "n_shards": n_shards,
            "top_activated_squares": top_square_entries(act_indices, act_values, top_squares, side_to_move),
            "top_z_pairs": top_z_pairs(z_indices, z_values, top_z, side_to_move),
            "dataset_row": row,
        }
        if dataset_error:
            sample["dataset_error"] = dataset_error
        return sample

    def export_feature(
        self,
        sae_name: str,
        feature_index: int,
        feature_type: str,
        analysis_name: str,
        max_samples: int,
        top_squares: int,
        top_z: int,
    ) -> dict[str, Any]:
        feature = self.client.get_feature(sae_name=sae_name, sae_series=self.sae_series, index=feature_index)
        if feature is None:
            raise ValueError(f"Feature {sae_name}#{feature_index} not found")
        analysis = next((item for item in feature.analyses if item.name == analysis_name), None)
        if analysis is None:
            raise ValueError(f"Analysis {analysis_name!r} missing for {sae_name}#{feature_index}")
        samples = []
        for sampling in analysis.samplings:
            for sparse_sample in iter_sparse_samples(sampling):
                samples.append(self.build_sample(sampling, *sparse_sample, top_squares, top_z))
                if len(samples) >= max_samples:
                    break
            if len(samples) >= max_samples:
                break
        parsed = parse_sae_name(sae_name)
        interpretation = feature.interpretation if isinstance(feature.interpretation, dict) else {}
        return {
            "dictionary_name": sae_name,
            "sae_series": self.sae_series,
            "analysis_name": analysis_name,
            "feature_type": feature_type,
            "layer": parsed[1] if parsed else None,
            "feature_index": feature_index,
            "interpretation": str(interpretation.get("text", "") or ""),
            "feature_stats": {
                "act_times": analysis.act_times,
                "max_feature_acts": analysis.max_feature_acts,
                "n_analyzed_tokens": analysis.n_analyzed_tokens,
            },
            "top_activation_samples": samples,
        }

    def export(
        self,
        feature_type: str,
        analysis_name: str,
        sae_name: str | None,
        layer: int | None,
        limit: int,
        max_samples: int,
        top_squares: int,
        top_z: int,
    ) -> list[dict[str, Any]]:
        sae_names = self.discover_saes(feature_type, analysis_name, sae_name, layer)
        rows = []
        for selected_sae in sae_names:
            remaining = limit - len(rows)
            if remaining <= 0:
                break
            query = {
                "sae_name": selected_sae,
                "sae_series": self.sae_series,
                "analyses": {"$elemMatch": {"name": analysis_name, "max_feature_acts": {"$gt": 0}}},
            }
            cursor = self.client.feature_collection.find(query, {"index": 1}).sort("index", 1).limit(remaining)
            for item in cursor:
                rows.append(
                    self.export_feature(
                        selected_sae,
                        int(item["index"]),
                        feature_type,
                        analysis_name,
                        max_samples,
                        top_squares,
                        top_z,
                    )
                )
        return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-type", required=True, choices=["lorsa", "transcoder", "mlp", "tc"])
    parser.add_argument("--sae-name", help="Exact SAE name, e.g. BT4_lorsa_L9A_k30_e16")
    parser.add_argument("--layer", type=int, help="Optional layer filter when discovering analyses")
    parser.add_argument("--sae-series", default=os.environ.get("SAE_SERIES", "BT4-exp128"))
    parser.add_argument("--analysis-name", default="default")
    parser.add_argument("--limit", type=int, default=100, help="Maximum features to export (default: 100)")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--top-squares", type=int, default=12)
    parser.add_argument("--top-z", type=int, default=16)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mongo-uri", default=os.environ.get("MONGO_URI", "mongodb://localhost:27017/"))
    parser.add_argument("--mongo-db", default=os.environ.get("MONGO_DB", "mechinterp"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit <= 0 or args.max_samples <= 0:
        raise SystemExit("--limit and --max-samples must be positive")
    feature_type = normalize_feature_type(args.feature_type)
    output = args.output or REPO_ROOT / "outputs" / "autointerp_score" / f"{feature_type}.jsonl"
    client = MongoClient(MongoDBConfig(mongo_uri=args.mongo_uri, mongo_db=args.mongo_db))
    exporter = FeatureSampleExporter(client, args.sae_series)
    rows = exporter.export(
        feature_type,
        args.analysis_name,
        args.sae_name,
        args.layer,
        args.limit,
        args.max_samples,
        args.top_squares,
        args.top_z,
    )
    if not rows:
        raise SystemExit("No matching analyzed features were found")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(make_serializable(row), ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "feature_type": feature_type, "features": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
