"""Select a reproducible, layer-balanced sample of circuit feature evidence."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SUPPORTED_TYPES = {"lorsa", "transcoder"}


def interpretation_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return str(value.get("text", "") or "").strip()
    return ""


def evidence_quality(row: dict[str, Any]) -> tuple[int, int]:
    samples = row.get("top_activation_samples") or []
    usable = sum(bool(sample.get("fen") and sample.get("top_activated_squares")) for sample in samples)
    return usable, len(samples)


def load_unique_evidence(input_dir: Path) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, int]]:
    rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    occurrences: Counter[tuple[str, int]] = Counter()
    interpretation_variants: defaultdict[tuple[str, int], set[str]] = defaultdict(set)

    files = sorted(input_dir.glob("trace_*.evidence.jsonl"))
    if not files:
        raise FileNotFoundError(f"No trace evidence JSONL files found under {input_dir}")

    for path in files:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
                feature_type = str(row.get("feature_type", "")).strip().lower()
                if feature_type not in SUPPORTED_TYPES:
                    continue
                key = (str(row["dictionary_name"]), int(row["feature_index"]))
                occurrences[key] += 1
                text = interpretation_text(row.get("existing_interpretation"))
                if text:
                    interpretation_variants[key].add(text)

                previous = rows_by_key.get(key)
                if previous is None or evidence_quality(row) > evidence_quality(previous):
                    rows_by_key[key] = row

    for key, row in rows_by_key.items():
        variants = interpretation_variants[key]
        row["interpretation"] = max(variants, key=len) if variants else ""
        row["circuit_occurrences"] = occurrences[key]
        row["interpretation_variants"] = sorted(variants)

    stats = {
        "evidence_files": len(files),
        "unique_features": len(rows_by_key),
        "features_with_interpretation_conflicts": sum(len(values) > 1 for values in interpretation_variants.values()),
    }
    return rows_by_key, stats


def select_rows(
    rows_by_key: dict[tuple[str, int], dict[str, Any]],
    per_layer: int,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, int]]]:
    grouped: defaultdict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows_by_key.values():
        feature_type = str(row["feature_type"]).lower()
        layer = int(row["layer"])
        if evidence_quality(row)[0] == 6:
            grouped[(feature_type, layer)].append(row)

    selected_by_type: dict[str, list[dict[str, Any]]] = {feature_type: [] for feature_type in SUPPORTED_TYPES}
    counts: dict[str, dict[str, int]] = {feature_type: {} for feature_type in SUPPORTED_TYPES}
    for feature_type in sorted(SUPPORTED_TYPES):
        for layer in range(15):
            candidates = sorted(
                grouped[(feature_type, layer)],
                key=lambda row: (str(row["dictionary_name"]), int(row["feature_index"])),
            )
            if len(candidates) < per_layer:
                raise ValueError(
                    f"Only {len(candidates)} complete {feature_type} features are available at layer {layer}; "
                    f"requested {per_layer}"
                )
            rng = random.Random(f"{seed}:{feature_type}:{layer}")
            chosen = rng.sample(candidates, per_layer)
            chosen.sort(key=lambda row: int(row["feature_index"]))
            selected_by_type[feature_type].extend(chosen)
            counts[feature_type][str(layer)] = len(chosen)
    return selected_by_type, counts


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--features-per-layer", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.features_per_layer <= 0:
        raise SystemExit("--features-per-layer must be positive")

    rows_by_key, input_stats = load_unique_evidence(args.input_dir)
    selected_by_type, counts = select_rows(rows_by_key, args.features_per_layer, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}
    for feature_type, rows in selected_by_type.items():
        output = args.output_dir / f"circuit_{feature_type}_selected_evidence.jsonl"
        write_jsonl(output, rows)
        outputs[feature_type] = str(output)

    manifest = {
        "source": str(args.input_dir),
        "selection": "uniform_without_replacement_from_unique_features_with_6_usable_samples",
        "seed": args.seed,
        "features_per_layer": args.features_per_layer,
        "layers": list(range(15)),
        "input_stats": input_stats,
        "selected_by_layer": counts,
        "outputs": outputs,
    }
    manifest_path = args.output_dir / "circuit_autointerp_selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
