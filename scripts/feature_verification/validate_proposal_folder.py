#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from feature_varification import (  # noqa: E402
    VerificationCase,
    build_rule,
    feature_spec_from_proposal,
    run_proposal_validation,
)

DEFAULT_MODEL_NAME = "lc0/BT4-1024x15x32h"
DEFAULT_TC_ROOT = PROJECT_ROOT / "result_BT4/tc/k_30_e_16"
DEFAULT_LORSA_ROOT = PROJECT_ROOT / "result_BT4/lorsa/k_30_e_16"
EXCLUDED_TAXONOMIES = {"Spa", "Reg", "Uninterpretable"}


def _jsonl_files(path: Path, pattern: str) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob(pattern))


def _feature_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["dictionary_name"]), int(row["feature_index"])


def _validation_signature(row: Mapping[str, Any]) -> str:
    return json.dumps(
        {
            "interpretation": row.get("interpretation"),
            "taxonomy": row.get("taxonomy"),
            "validation": row.get("validation"),
        },
        sort_keys=True,
    )


def load_proposals(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    files = _jsonl_files(path, "*.proposals.jsonl")
    if not files:
        raise FileNotFoundError(f"No *.proposals.jsonl files found under {path}")

    proposals: dict[tuple[str, int], dict[str, Any]] = {}
    signatures: dict[tuple[str, int], str] = {}
    for source_path in files:
        with source_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                key = _feature_key(row)
                signature = _validation_signature(row)
                previous = signatures.get(key)
                if previous is not None and previous != signature:
                    raise ValueError(
                        f"Conflicting duplicate proposal for {key[0]}:{key[1]} "
                        f"at {source_path}:{line_no}"
                    )
                signatures[key] = signature
                proposals.setdefault(key, row)
    return proposals


def _sample_move_uci(sample: Mapping[str, Any]) -> str | None:
    moves = sample.get("top_moves") or []
    if not moves:
        return None
    first = moves[0]
    if isinstance(first, Mapping):
        value = first.get("uci")
    else:
        value = first
    return str(value) if value else None


def _case_from_sample(sample: Mapping[str, Any], label: str) -> VerificationCase:
    metadata = {
        "wdl": sample.get("wdl") or {},
        "value": sample.get("value"),
        "context_idx": sample.get("context_idx"),
        "dataset_name": sample.get("dataset_name"),
    }
    return VerificationCase(
        fen=str(sample["fen"]),
        move_uci=_sample_move_uci(sample),
        label=label,
        metadata=metadata,
    )


def load_cases(
    evidence_path: Path,
    proposal_keys: set[tuple[str, int]],
) -> tuple[dict[tuple[str, int], list[VerificationCase]], dict[tuple[str, int], float]]:
    files = _jsonl_files(evidence_path, "*.jsonl")
    if not files:
        raise FileNotFoundError(f"No evidence JSONL files found under {evidence_path}")

    cases: dict[tuple[str, int], list[VerificationCase]] = defaultdict(list)
    seen_fens: dict[tuple[str, int], set[str]] = defaultdict(set)
    max_activations: dict[tuple[str, int], float] = {}
    for source_path in files:
        with source_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                try:
                    key = _feature_key(row)
                except (KeyError, TypeError, ValueError):
                    continue
                if key not in proposal_keys:
                    continue
                feature_stats = row.get("feature_stats") or {}
                raw_max_activation = feature_stats.get("max_feature_act")
                if raw_max_activation is None:
                    raw_max_activation = feature_stats.get("max_feature_acts")
                if raw_max_activation is not None:
                    max_activations[key] = max(max_activations.get(key, float("-inf")), float(raw_max_activation))
                for sample_index, sample in enumerate(row.get("top_activation_samples") or []):
                    fen = str(sample.get("fen") or "")
                    if not fen or fen in seen_fens[key]:
                        continue
                    seen_fens[key].add(fen)
                    cases[key].append(
                        _case_from_sample(
                            sample,
                            label=f"{source_path.stem}:sample_{sample_index:02d}",
                        )
                    )
    return cases, max_activations


def _safe_filename(dictionary_name: str, feature_index: int) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", dictionary_name)
    return f"{safe_name}__{feature_index}.json"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _base_output(proposal: Mapping[str, Any], case_count: int) -> dict[str, Any]:
    output = {
        "feature": {
            "dictionary_name": proposal["dictionary_name"],
            "feature_index": int(proposal["feature_index"]),
            "feature_type": proposal["feature_type"],
            "layer": int(proposal["layer"]),
        },
        "taxonomy": proposal.get("taxonomy"),
        "interpretation": proposal.get("interpretation"),
        "rationale": proposal.get("rationale"),
        "validation": (proposal.get("validation") or [None])[0],
        "n_cases_selected": case_count,
    }
    if proposal.get("activation_threshold") is not None:
        output["activation_threshold"] = proposal["activation_threshold"]
    return output


def _case_limit(proposal: Mapping[str, Any], override: int | None) -> int:
    if override is not None:
        return override
    validations = proposal.get("validation") or []
    if validations:
        return int((validations[0].get("cases") or {}).get("max_cases", 24))
    return 24


def _with_activation_threshold(
    proposal: Mapping[str, Any],
    *,
    max_activation: float,
    activation_ratio: float,
) -> dict[str, Any]:
    resolved = copy.deepcopy(dict(proposal))
    resolved["validation"][0]["threshold"] = {
        "mode": "absolute",
        "value": activation_ratio * max_activation,
        "scope": "dataset",
    }
    resolved["activation_threshold"] = {
        "ratio": activation_ratio,
        "max_activation": max_activation,
        "absolute_value": activation_ratio * max_activation,
    }
    return resolved


def _prepare_jobs(
    proposals: Mapping[tuple[str, int], dict[str, Any]],
    cases_by_feature: Mapping[tuple[str, int], Sequence[VerificationCase]],
    max_activations: Mapping[tuple[str, int], float],
    output_dir: Path,
    *,
    activation_ratio: float,
    max_cases: int | None,
    overwrite: bool,
    limit: int | None,
) -> tuple[dict[tuple[str, int], list[tuple[dict[str, Any], list[VerificationCase], Path]]], int, int, int]:
    jobs: dict[tuple[str, int], list[tuple[dict[str, Any], list[VerificationCase], Path]]] = defaultdict(list)
    skipped = 0
    excluded = 0
    preparation_errors = 0
    selected_items = sorted(proposals.items())
    if limit is not None:
        selected_items = selected_items[:limit]

    for key, proposal in selected_items:
        if str(proposal.get("taxonomy")) in EXCLUDED_TAXONOMIES:
            excluded += 1
            continue
        output_path = output_dir / _safe_filename(*key)
        if output_path.exists() and not overwrite:
            try:
                existing = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
            if existing.get("status") == "completed":
                skipped += 1
                continue

        selected_cases = list(cases_by_feature.get(key, []))[: _case_limit(proposal, max_cases)]
        base = _base_output(proposal, len(selected_cases))
        validations = proposal.get("validation") or []
        if not validations:
            _write_json(output_path, {**base, "status": "error", "error": "Proposal has no validation rule"})
            preparation_errors += 1
            continue
        max_activation = max_activations.get(key)
        if max_activation is None:
            _write_json(output_path, {**base, "status": "error", "error": "Evidence has no max_feature_act"})
            preparation_errors += 1
            continue
        proposal = _with_activation_threshold(
            proposal,
            max_activation=max_activation,
            activation_ratio=activation_ratio,
        )
        base = _base_output(proposal, len(selected_cases))
        try:
            rule = build_rule(validations[0]["rule"])
            feature = feature_spec_from_proposal(proposal)
        except Exception as error:
            _write_json(
                output_path,
                {**base, "status": "error", "error": f"{type(error).__name__}: {error}"},
            )
            preparation_errors += 1
            continue

        if getattr(rule, "requires_move_uci", False):
            selected_cases = [case for case in selected_cases if case.move_uci]
            base["n_cases_selected"] = len(selected_cases)
        if not selected_cases:
            _write_json(output_path, {**base, "status": "error", "error": "No usable evidence cases"})
            preparation_errors += 1
            continue
        jobs[(feature.feature_type, feature.layer)].append((proposal, selected_cases, output_path))

    return jobs, skipped, excluded, preparation_errors


def _load_model(model_name: str, device: str) -> Any:
    import torch
    from transformer_lens import HookedTransformer

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    return HookedTransformer.from_pretrained_no_processing(
        model_name,
        device=device,
        dtype=torch.float32,
    ).eval()


def _load_sae(feature_type: str, layer: int, root: Path, device: str) -> Any:
    import torch

    from lm_saes import LowRankSparseAttention, SparseAutoEncoder

    checkpoint = str(root / f"L{layer}")
    if feature_type == "transcoder":
        return SparseAutoEncoder.from_pretrained(checkpoint, dtype=torch.float32, device=device)
    return LowRankSparseAttention.from_pretrained(checkpoint, dtype=torch.float32, device=device)


def _model_inputs(feature_type: str, layer: int, sae: Any) -> tuple[list[Any], dict[int, Any]]:
    if feature_type == "transcoder":
        return [], {layer: sae}
    lorsas: list[Any] = [None] * (layer + 1)
    lorsas[layer] = sae
    return lorsas, {}


def run_jobs(
    jobs: Mapping[tuple[str, int], Sequence[tuple[dict[str, Any], list[VerificationCase], Path]]],
    *,
    model_name: str,
    device: str,
    tc_root: Path,
    lorsa_root: Path,
    max_examples: int,
) -> tuple[int, int]:
    import torch

    if not jobs:
        return 0, 0
    model = _load_model(model_name, device)
    completed = 0
    errors = 0
    for (feature_type, layer), group_jobs in sorted(jobs.items()):
        root = tc_root if feature_type == "transcoder" else lorsa_root
        print(f"Loading {feature_type} layer {layer} for {len(group_jobs)} features", flush=True)
        sae = _load_sae(feature_type, layer, root, device)
        lorsas, transcoders = _model_inputs(feature_type, layer, sae)
        for proposal, cases, output_path in group_jobs:
            base = _base_output(proposal, len(cases))
            key = _feature_key(proposal)
            try:
                run_kwargs: dict[str, Any] = {"show_progress": False}
                rule_type = proposal["validation"][0]["rule"]["type"]
                if rule_type != "value_outcome":
                    run_kwargs["max_examples"] = max_examples
                result = run_proposal_validation(
                    proposal=proposal,
                    model=model,
                    lorsas=lorsas,
                    transcoders=transcoders,
                    cases=cases,
                    **run_kwargs,
                )
                result_payload = result.to_dict() if hasattr(result, "to_dict") else result
                _write_json(output_path, {**base, "status": "completed", "result": result_payload})
                completed += 1
                print(f"Completed {key[0]}:{key[1]}", flush=True)
            except KeyboardInterrupt:
                raise
            except Exception as error:
                _write_json(
                    output_path,
                    {**base, "status": "error", "error": f"{type(error).__name__}: {error}"},
                )
                errors += 1
                print(f"Failed {key[0]}:{key[1]}: {error}", file=sys.stderr, flush=True)
        del sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return completed, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate every unique feature in proposal JSONL files and write one JSON result per feature."
    )
    parser.add_argument("--proposal-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default=os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME))
    parser.add_argument("--tc-root", type=Path, default=Path(os.environ.get("TC_ROOT", DEFAULT_TC_ROOT)))
    parser.add_argument("--lorsa-root", type=Path, default=Path(os.environ.get("LORSA_ROOT", DEFAULT_LORSA_ROOT)))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument(
        "--activation-ratio",
        type=float,
        default=0.3,
        help="A square is active only above this fraction of the feature's global max activation.",
    )
    parser.add_argument("--max-cases", type=int, help="Override each proposal's cases.max_cases value.")
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--limit", type=int, help="Validate only the first N unique features for a smoke test.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_cases is not None and args.max_cases <= 0:
        raise ValueError("--max-cases must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if not 0.0 < args.activation_ratio <= 1.0:
        raise ValueError("--activation-ratio must be in (0, 1]")

    proposals = load_proposals(args.proposal_dir)
    cases_by_feature, max_activations = load_cases(args.evidence_dir, set(proposals))
    jobs, skipped, excluded, preparation_errors = _prepare_jobs(
        proposals,
        cases_by_feature,
        max_activations,
        args.output_dir,
        activation_ratio=args.activation_ratio,
        max_cases=args.max_cases,
        overwrite=args.overwrite,
        limit=args.limit,
    )
    completed, execution_errors = run_jobs(
        jobs,
        model_name=args.model_name,
        device=args.device,
        tc_root=args.tc_root,
        lorsa_root=args.lorsa_root,
        max_examples=args.max_examples,
    )
    print(
        f"Finished: completed={completed}, errors={preparation_errors + execution_errors}, skipped={skipped}, "
        f"excluded={excluded}, "
        f"output_dir={args.output_dir}"
    )
    return 1 if preparation_errors + execution_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
