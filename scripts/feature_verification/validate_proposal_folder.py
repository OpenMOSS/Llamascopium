#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from feature_varification import build_rule, feature_spec_from_proposal  # noqa: E402
from feature_varification.value import ValueOutcomeRule  # noqa: E402

DEFAULT_MODEL_NAME = "lc0/BT4-1024x15x32h"
DEFAULT_DATASET_PATH = Path(
    "/inspire/hdd/global_user/hezhengfu-240208120186/data/rlin_data/Chess/chess_master_data"
)
DEFAULT_TC_ROOT = PROJECT_ROOT / "result_BT4/tc/k_30_e_16"
DEFAULT_LORSA_ROOT = PROJECT_ROOT / "result_BT4/lorsa/k_30_e_16"
EXCLUDED_TAXONOMIES = {"Spa", "Reg", "Uninterpretable"}


@dataclass(frozen=True)
class ValidationJob:
    proposal: dict[str, Any]
    rule: Any
    rule_signature: str
    feature_id: int
    threshold: float
    max_activation: float
    output_path: Path


@dataclass
class ValidationGroup:
    feature_type: str
    layer: int
    jobs: list[ValidationJob]
    feature_ids: torch.Tensor | None = None
    thresholds: torch.Tensor | None = None
    counts: torch.Tensor | None = None
    total_activations: torch.Tensor | None = None
    processed_fens: torch.Tensor | None = None
    completed: torch.Tensor | None = None

    def initialize(self, device: str) -> None:
        size = len(self.jobs)
        self.feature_ids = torch.tensor([job.feature_id for job in self.jobs], dtype=torch.long, device=device)
        self.thresholds = torch.tensor([job.threshold for job in self.jobs], dtype=torch.float32, device=device)
        self.counts = torch.zeros((size, 4), dtype=torch.long, device=device)
        self.total_activations = torch.zeros(size, dtype=torch.long, device=device)
        self.processed_fens = torch.zeros(size, dtype=torch.long, device=device)
        self.completed = torch.zeros(size, dtype=torch.bool, device=device)


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


def load_max_activations(
    feature_stats_path: Path,
    proposal_keys: set[tuple[str, int]],
) -> dict[tuple[str, int], float]:
    """Read feature maxima only; top-activation FENs are deliberately ignored."""

    files = _jsonl_files(feature_stats_path, "*.jsonl")
    if not files:
        raise FileNotFoundError(f"No feature-stat JSONL files found under {feature_stats_path}")

    maxima: dict[tuple[str, int], float] = {}
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
                stats = row.get("feature_stats") or {}
                value = stats.get("max_feature_act")
                if value is None:
                    value = stats.get("max_feature_acts")
                if value is not None:
                    maxima[key] = max(maxima.get(key, float("-inf")), float(value))
    return maxima


def _safe_filename(dictionary_name: str, feature_index: int) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", dictionary_name)
    return f"{safe_name}__{feature_index}.json"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _is_completed_output(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "completed"


def _resolved_proposal(proposal: Mapping[str, Any], threshold: float) -> dict[str, Any]:
    resolved = copy.deepcopy(dict(proposal))
    resolved["validation"][0]["threshold"] = {
        "mode": "absolute",
        "value": threshold,
        "scope": "dataset",
    }
    return resolved


def prepare_groups(
    proposals: Mapping[tuple[str, int], dict[str, Any]],
    maxima: Mapping[tuple[str, int], float],
    output_dir: Path,
    *,
    activation_ratio: float,
    overwrite: bool,
    limit: int | None,
) -> tuple[dict[tuple[str, int], ValidationGroup], dict[str, int]]:
    grouped_jobs: dict[tuple[str, int], list[ValidationJob]] = {}
    stats = {"excluded": 0, "skipped": 0, "errors": 0, "queued": 0}

    selected = [
        (key, proposal)
        for key, proposal in sorted(proposals.items())
        if str(proposal.get("taxonomy")) not in EXCLUDED_TAXONOMIES
    ]
    stats["excluded"] = len(proposals) - len(selected)
    if limit is not None:
        selected = selected[:limit]

    for key, proposal in selected:
        output_path = output_dir / _safe_filename(*key)
        if not overwrite and _is_completed_output(output_path):
            stats["skipped"] += 1
            continue
        base = {
            "feature": {"dictionary_name": key[0], "feature_index": key[1]},
            "taxonomy": proposal.get("taxonomy"),
            "interpretation": proposal.get("interpretation"),
        }
        validations = proposal.get("validation") or []
        if not validations:
            _write_json(output_path, {**base, "status": "error", "error": "Proposal has no validation rule"})
            stats["errors"] += 1
            continue
        max_activation = maxima.get(key)
        if max_activation is None:
            _write_json(output_path, {**base, "status": "error", "error": "Feature stats have no max activation"})
            stats["errors"] += 1
            continue
        try:
            rule_config = validations[0]["rule"]
            rule = build_rule(rule_config)
            feature = feature_spec_from_proposal(proposal)
        except Exception as error:
            _write_json(
                output_path,
                {**base, "status": "error", "error": f"{type(error).__name__}: {error}"},
            )
            stats["errors"] += 1
            continue

        threshold = activation_ratio * max_activation
        resolved = _resolved_proposal(proposal, threshold)
        rule_signature = json.dumps(rule_config, sort_keys=True)
        job = ValidationJob(
            proposal=resolved,
            rule=rule,
            rule_signature=rule_signature,
            feature_id=feature.feature_id,
            threshold=threshold,
            max_activation=max_activation,
            output_path=output_path,
        )
        group_key = (feature.feature_type, feature.layer)
        grouped_jobs.setdefault(group_key, []).append(job)
        stats["queued"] += 1

    groups = {
        key: ValidationGroup(feature_type=key[0], layer=key[1], jobs=jobs)
        for key, jobs in grouped_jobs.items()
    }
    return groups, stats


def _load_dataset(path: Path, split: str | None) -> Any:
    from datasets import DatasetDict, load_from_disk

    dataset = load_from_disk(str(path))
    if isinstance(dataset, DatasetDict):
        if split is not None:
            return dataset[split]
        if "train" in dataset:
            return dataset["train"]
        return dataset[next(iter(dataset))]
    return dataset


def _load_model(model_name: str, device: str) -> Any:
    from transformer_lens import HookedTransformer

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    return HookedTransformer.from_pretrained_no_processing(
        model_name,
        device=device,
        dtype=torch.float32,
    ).eval()


def _load_saes(
    group_keys: list[tuple[str, int]],
    *,
    tc_root: Path,
    lorsa_root: Path,
    device: str,
) -> dict[tuple[str, int], Any]:
    from lm_saes import LowRankSparseAttention, SparseAutoEncoder

    modules: dict[tuple[str, int], Any] = {}
    for feature_type, layer in sorted(group_keys):
        if feature_type == "transcoder":
            checkpoint = tc_root / f"L{layer}"
            module = SparseAutoEncoder.from_pretrained(
                str(checkpoint),
                dtype=torch.float32,
                device=device,
            )
        else:
            checkpoint = lorsa_root / f"L{layer}"
            module = LowRankSparseAttention.from_pretrained(
                str(checkpoint),
                dtype=torch.float32,
                device=device,
            )
        modules[(feature_type, layer)] = module
        print(f"Loaded {feature_type} layer {layer}: {checkpoint}", flush=True)
    return modules


def _hook_name(feature_type: str, layer: int) -> str:
    if feature_type == "transcoder":
        return f"blocks.{layer}.resid_mid_after_ln"
    return f"blocks.{layer}.hook_attn_in"


def _model_metadata(output: Any) -> dict[str, Any]:
    value_output = output[1][0]
    values = value_output.detach().to(torch.float32).cpu().tolist()
    return {
        "wdl": {
            "current_player_win": values[0],
            "draw": values[1],
            "current_player_loss": values[2],
        }
    }


def _top_move_uci(output: Any, fen: str) -> str | None:
    from chess_utils.move import get_move_from_policy_output

    try:
        move = get_move_from_policy_output(output[0], fen, return_list=False)
    except Exception:
        return None
    return str(move) if move else None


def _rule_targets(
    groups: Mapping[tuple[str, int], ValidationGroup],
    *,
    fen: str,
    move_uci: str | None,
    metadata: Mapping[str, Any],
    device: str,
) -> tuple[dict[str, torch.Tensor], dict[str, bool], set[str]]:
    spatial: dict[str, torch.Tensor] = {}
    value: dict[str, bool] = {}
    unavailable: set[str] = set()
    seen: set[str] = set()
    for group in groups.values():
        if group.completed is not None and bool(group.completed.all().item()):
            continue
        for job in group.jobs:
            if job.rule_signature in seen:
                continue
            seen.add(job.rule_signature)
            if isinstance(job.rule, ValueOutcomeRule):
                value[job.rule_signature] = bool(job.rule.matches(metadata))
                continue
            if getattr(job.rule, "requires_move_uci", False) and move_uci is None:
                unavailable.add(job.rule_signature)
                continue
            try:
                result = job.rule.evaluate(fen, move_uci)
            except (ValueError, KeyError):
                unavailable.add(job.rule_signature)
                continue
            spatial[job.rule_signature] = torch.tensor(result.mask, dtype=torch.bool, device=device)
    return spatial, value, unavailable


def _update_group(
    group: ValidationGroup,
    encoded: torch.Tensor,
    *,
    spatial_targets: Mapping[str, torch.Tensor],
    value_targets: Mapping[str, bool],
    unavailable_rules: set[str],
    min_activations: int,
) -> list[int]:
    assert group.feature_ids is not None
    assert group.thresholds is not None
    assert group.counts is not None
    assert group.total_activations is not None
    assert group.processed_fens is not None
    assert group.completed is not None

    acts_all = encoded[0] if encoded.dim() == 3 else encoded
    feature_acts = acts_all[:64].index_select(1, group.feature_ids)
    active = feature_acts > group.thresholds.unsqueeze(0)
    available = torch.tensor(
        [job.rule_signature not in unavailable_rules for job in group.jobs],
        dtype=torch.bool,
        device=feature_acts.device,
    )
    unfinished = ~group.completed
    eligible = unfinished & available
    if not bool(unfinished.any().item()):
        return []

    spatial_indices = [
        index
        for index, job in enumerate(group.jobs)
        if not isinstance(job.rule, ValueOutcomeRule) and job.rule_signature not in unavailable_rules
    ]
    if spatial_indices:
        index_tensor = torch.tensor(spatial_indices, dtype=torch.long, device=feature_acts.device)
        active_spatial = active.index_select(1, index_tensor)
        target_spatial = torch.stack(
            [spatial_targets[group.jobs[index].rule_signature] for index in spatial_indices],
            dim=1,
        )
        valid = eligible.index_select(0, index_tensor).to(torch.long)
        updates = torch.stack(
            [
                (active_spatial & target_spatial).sum(dim=0),
                (active_spatial & ~target_spatial).sum(dim=0),
                (~active_spatial & ~target_spatial).sum(dim=0),
                (~active_spatial & target_spatial).sum(dim=0),
            ],
            dim=1,
        )
        group.counts[index_tensor] += updates * valid.unsqueeze(1)

    value_indices = [index for index, job in enumerate(group.jobs) if isinstance(job.rule, ValueOutcomeRule)]
    if value_indices:
        index_tensor = torch.tensor(value_indices, dtype=torch.long, device=feature_acts.device)
        active_case = active.index_select(1, index_tensor).any(dim=0)
        target_case = torch.tensor(
            [value_targets[group.jobs[index].rule_signature] for index in value_indices],
            dtype=torch.bool,
            device=feature_acts.device,
        )
        valid = eligible.index_select(0, index_tensor).to(torch.long)
        updates = torch.stack(
            [
                active_case & target_case,
                active_case & ~target_case,
                ~active_case & ~target_case,
                ~active_case & target_case,
            ],
            dim=1,
        ).to(torch.long)
        group.counts[index_tensor] += updates * valid.unsqueeze(1)

    active_counts = active.sum(dim=0).to(torch.long)
    group.total_activations += active_counts * eligible.to(torch.long)
    group.processed_fens += eligible.to(torch.long)
    newly_completed_mask = unfinished & (group.total_activations >= min_activations)
    group.completed |= newly_completed_mask
    return torch.nonzero(newly_completed_mask, as_tuple=False).reshape(-1).cpu().tolist()


def _counts_dict(values: list[int]) -> dict[str, Any]:
    tp, fp, tn, fn = values
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total": total,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def _write_job_result(
    group: ValidationGroup,
    index: int,
    *,
    status: str,
    min_activations: int,
    scanned_fens: int,
    dataset_path: Path,
    seed: int,
) -> None:
    assert group.counts is not None
    assert group.total_activations is not None
    assert group.processed_fens is not None

    job = group.jobs[index]
    counts = group.counts[index].detach().cpu().tolist()
    proposal = job.proposal
    payload = {
        "status": status,
        "feature": {
            "dictionary_name": proposal["dictionary_name"],
            "feature_index": int(proposal["feature_index"]),
            "feature_type": group.feature_type,
            "layer": group.layer,
        },
        "taxonomy": proposal.get("taxonomy"),
        "interpretation": proposal.get("interpretation"),
        "rationale": proposal.get("rationale"),
        "validation": proposal["validation"][0],
        "activation_threshold": {
            "ratio": job.threshold / job.max_activation,
            "max_activation": job.max_activation,
            "absolute_value": job.threshold,
        },
        "dataset_scan": {
            "dataset_path": str(dataset_path),
            "seed": seed,
            "global_fens_scanned": scanned_fens,
            "feature_fens_evaluated": int(group.processed_fens[index].item()),
            "target_activations": min_activations,
            "observed_activations": int(group.total_activations[index].item()),
        },
        "metric_level": "position" if not isinstance(job.rule, ValueOutcomeRule) else "case",
        "result": {"counts": _counts_dict(counts)},
    }
    _write_json(job.output_path, payload)


def run_dataset_scan(
    groups: dict[tuple[str, int], ValidationGroup],
    *,
    dataset: Any,
    dataset_path: Path,
    fen_column: str,
    model_name: str,
    device: str,
    tc_root: Path,
    lorsa_root: Path,
    min_activations: int,
    max_fens: int,
    seed: int,
    progress_every: int,
) -> tuple[int, int]:
    if not groups:
        return 0, 0

    model = _load_model(model_name, device)
    modules = _load_saes(list(groups), tc_root=tc_root, lorsa_root=lorsa_root, device=device)
    for group in groups.values():
        group.initialize(device)

    needs_move = any(getattr(job.rule, "requires_move_uci", False) for group in groups.values() for job in group.jobs)
    rng = random.Random(seed)
    completed = 0
    scanned = 0
    dataset_size = len(dataset)

    while scanned < max_fens and completed < sum(len(group.jobs) for group in groups.values()):
        row = dataset[rng.randrange(dataset_size)]
        fen = str(row.get(fen_column) or "")
        if not fen:
            continue
        try:
            with torch.no_grad():
                output, cache = model.run_with_cache(fen, prepend_bos=False)
                metadata = _model_metadata(output)
                move_uci = _top_move_uci(output, fen) if needs_move else None
                spatial_targets, value_targets, unavailable_rules = _rule_targets(
                    groups,
                    fen=fen,
                    move_uci=move_uci,
                    metadata=metadata,
                    device=device,
                )
                for key, group in groups.items():
                    assert group.completed is not None
                    if bool(group.completed.all().item()):
                        continue
                    encoded = modules[key].encode(cache[_hook_name(*key)])
                    newly_completed = _update_group(
                        group,
                        encoded,
                        spatial_targets=spatial_targets,
                        value_targets=value_targets,
                        unavailable_rules=unavailable_rules,
                        min_activations=min_activations,
                    )
                    for index in newly_completed:
                        completed += 1
                        _write_job_result(
                            group,
                            index,
                            status="completed",
                            min_activations=min_activations,
                            scanned_fens=scanned + 1,
                            dataset_path=dataset_path,
                            seed=seed,
                        )
                del cache, output
        except KeyboardInterrupt:
            raise
        except Exception as error:
            print(f"Skipped FEN because {type(error).__name__}: {error}", file=sys.stderr, flush=True)
            continue

        scanned += 1
        if scanned % progress_every == 0:
            print(f"Scanned {scanned} FENs; completed {completed} features", flush=True)

    unfinished = 0
    for group in groups.values():
        assert group.completed is not None
        for index in torch.nonzero(~group.completed, as_tuple=False).reshape(-1).cpu().tolist():
            unfinished += 1
            _write_job_result(
                group,
                index,
                status="max_fens_reached",
                min_activations=min_activations,
                scanned_fens=scanned,
                dataset_path=dataset_path,
                seed=seed,
            )
    return completed, unfinished


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate proposal rules by streaming random positions from a chess dataset. "
            "Each FEN is forwarded once and all features in a layer/type group are evaluated together."
        )
    )
    parser.add_argument("--proposal-dir", type=Path, required=True)
    parser.add_argument(
        "--feature-stats-dir",
        type=Path,
        required=True,
        help="JSONL source for max_feature_act only; its top-activation FENs are never used.",
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--dataset-split")
    parser.add_argument("--fen-column", default="fen")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default=os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME))
    parser.add_argument("--tc-root", type=Path, default=Path(os.environ.get("TC_ROOT", DEFAULT_TC_ROOT)))
    parser.add_argument("--lorsa-root", type=Path, default=Path(os.environ.get("LORSA_ROOT", DEFAULT_LORSA_ROOT)))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument("--activation-ratio", type=float, default=0.3)
    parser.add_argument("--min-activations", type=int, default=1000)
    parser.add_argument("--max-fens", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--limit", type=int, help="Validate only the first N eligible unique features.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 < args.activation_ratio <= 1.0:
        raise ValueError("--activation-ratio must be in (0, 1]")
    if args.min_activations <= 0:
        raise ValueError("--min-activations must be positive")
    if args.max_fens <= 0:
        raise ValueError("--max-fens must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    proposals = load_proposals(args.proposal_dir)
    maxima = load_max_activations(args.feature_stats_dir, set(proposals))
    groups, preparation = prepare_groups(
        proposals,
        maxima,
        args.output_dir,
        activation_ratio=args.activation_ratio,
        overwrite=args.overwrite,
        limit=args.limit,
    )
    print(f"Prepared groups: {preparation}", flush=True)
    dataset = _load_dataset(args.dataset_path, args.dataset_split)
    completed, unfinished = run_dataset_scan(
        groups,
        dataset=dataset,
        dataset_path=args.dataset_path,
        fen_column=args.fen_column,
        model_name=args.model_name,
        device=args.device,
        tc_root=args.tc_root,
        lorsa_root=args.lorsa_root,
        min_activations=args.min_activations,
        max_fens=args.max_fens,
        seed=args.seed,
        progress_every=args.progress_every,
    )
    print(
        f"Finished: completed={completed}, max_fens_reached={unfinished}, "
        f"preparation_errors={preparation['errors']}, skipped={preparation['skipped']}, "
        f"excluded={preparation['excluded']}, output_dir={args.output_dir}"
    )
    return 1 if preparation["errors"] or unfinished else 0


if __name__ == "__main__":
    raise SystemExit(main())
