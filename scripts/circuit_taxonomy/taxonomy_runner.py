#!/usr/bin/env python3
"""Extract circuit taxonomy evidence and optionally ask Codex to draft labels.

The runner never writes final annotations. It produces JSONL proposals that can
be imported into the Circuit Taxonomy Annotation page's LLM Review Queue.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

try:
    import msgpack
except ImportError:  # pragma: no cover - exercised in environments missing deps
    msgpack = None


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_NAME = "annotate-circuit-taxonomy"
SKILL_RUBRIC_PATH = Path.home() / ".codex" / "skills" / SKILL_NAME / "references" / "taxonomy-rubric.md"
DEFAULT_BASE_URL = os.environ.get("VITE_BACKEND_URL") or os.environ.get("BACKEND_URL") or "http://127.0.0.1:3000"
TAXONOMY_LABELS = {
    "Det",
    "Src",
    "Tgt",
    "Val",
    "Cap",
    "Pro",
    "Mov",
    "Tac",
    "Reg",
    "Spa",
    "Uninterpretable",
}


def request_json(base_url: str, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.get(f"{base_url.rstrip('/')}{path}", params=params, timeout=120)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not return a JSON object")
    return data


def request_feature(base_url: str, dictionary_name: str, feature_index: int) -> dict[str, Any]:
    response = requests.get(
        f"{base_url.rstrip('/')}/dictionaries/{quote(dictionary_name, safe='')}/features/{feature_index}",
        headers={"Accept": "application/x-msgpack"},
        timeout=180,
    )
    response.raise_for_status()
    data = unpack_msgpack(response.content)
    if not isinstance(data, dict):
        raise ValueError("Feature API did not return a msgpack object")
    return data


def unpack_msgpack(content: bytes) -> Any:
    if msgpack is not None:
        return msgpack.unpackb(content, raw=False)

    ui_dir = REPO_ROOT / "ui"
    if not (ui_dir / "node_modules" / "@msgpack" / "msgpack").exists():
        raise RuntimeError(
            "Python package 'msgpack' is not installed, and ui/node_modules/@msgpack/msgpack was not found. "
            "Install repo dependencies or run: pip install msgpack"
        )

    with tempfile.NamedTemporaryFile("wb", suffix=".msgpack", delete=False) as handle:
        handle.write(content)
        packed_path = Path(handle.name)
    try:
        script = (
            "import fs from 'node:fs';"
            "import { decode } from '@msgpack/msgpack';"
            "const data = fs.readFileSync(process.argv[1]);"
            "console.log(JSON.stringify(decode(data)));"
        )
        result = subprocess.run(
            ["node", "--input-type=module", "-e", script, str(packed_path)],
            cwd=ui_dir,
            text=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return json.loads(result.stdout)
    finally:
        packed_path.unlink(missing_ok=True)


def board_index_to_square(index: int) -> str:
    if index < 0 or index > 63:
        return f"idx{index}"
    file_name = "abcdefgh"[index % 8]
    rank = 8 - (index // 8)
    return f"{file_name}{rank}"


def extract_fen(text: str | None) -> str | None:
    if not text:
        return None
    pattern = re.compile(
        r"\b(?:[pnbrqkPNBRQK1-8]+/){7}[pnbrqkPNBRQK1-8]+\s+[wb]\s+(?:K?Q?k?q?|-)\s+(?:[a-h][36]|-)\s+\d+\s+\d+\b"
    )
    match = pattern.search(text)
    return match.group(0) if match else None


def top_index_values(indices: Any, values: Any, limit: int) -> list[dict[str, Any]]:
    if not isinstance(indices, list) or not isinstance(values, list):
        return []
    pairs: list[tuple[int, float]] = []
    for raw_index, raw_value in zip(indices, values):
        try:
            index = int(raw_index)
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if 0 <= index < 64:
            pairs.append((index, value))
    pairs.sort(key=lambda item: abs(item[1]), reverse=True)
    return [
        {
            "index": index,
            "square": board_index_to_square(index),
            "value": round(value, 6),
        }
        for index, value in pairs[:limit]
    ]


def normalize_z_pairs(indices: Any, values: Any, limit: int) -> list[dict[str, Any]]:
    if not isinstance(indices, list) or not isinstance(values, list) or not indices:
        return []

    raw_pairs: list[tuple[int, int, float]] = []
    first = indices[0]
    if isinstance(first, list) and len(first) == 2 and all(not isinstance(x, list) for x in first):
        for pair, raw_value in zip(indices, values):
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            try:
                source = int(pair[0])
                target = int(pair[1])
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            raw_pairs.append((source, target, value))
    elif len(indices) >= 2 and isinstance(indices[0], list) and isinstance(indices[1], list):
        for raw_source, raw_target, raw_value in zip(indices[0], indices[1], values):
            try:
                source = int(raw_source)
                target = int(raw_target)
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            raw_pairs.append((source, target, value))

    raw_pairs.sort(key=lambda item: abs(item[2]), reverse=True)
    return [
        {
            "source_index": source,
            "source_square": board_index_to_square(source),
            "target_index": target,
            "target_square": board_index_to_square(target),
            "value": round(value, 6),
        }
        for source, target, value in raw_pairs[:limit]
        if 0 <= source < 64 and 0 <= target < 64
    ]


def compact_signal_fields(value: Any, prefix: str = "", depth: int = 0) -> dict[str, Any]:
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
                        str(k): v for k, v in list(child.items())[:8] if isinstance(v, (str, int, float, bool)) or v is None
                    }
            signals.update(compact_signal_fields(child, path, depth + 1))
    elif isinstance(value, list) and depth < 2:
        for index, child in enumerate(value[:5]):
            signals.update(compact_signal_fields(child, f"{prefix}[{index}]", depth + 1))
    return signals


def summarize_feature_samples(feature: dict[str, Any], max_samples: int, top_squares: int, top_z: int) -> list[dict[str, Any]]:
    groups = feature.get("sample_groups")
    if not isinstance(groups, list) or not groups:
        return []

    selected_group = None
    for group in groups:
        if isinstance(group, dict) and group.get("analysis_name") == "top_activations":
            selected_group = group
            break
    if selected_group is None:
        selected_group = groups[0] if isinstance(groups[0], dict) else None
    if not selected_group:
        return []

    samples = selected_group.get("samples")
    if not isinstance(samples, list):
        return []

    summaries: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            continue
        fen = str(sample.get("fen") or extract_fen(sample.get("text")) or "")
        active = top_index_values(sample.get("feature_acts_indices"), sample.get("feature_acts_values"), top_squares)
        z_pairs = normalize_z_pairs(sample.get("z_pattern_indices"), sample.get("z_pattern_values"), top_z)
        if not fen and not active and not z_pairs:
            continue
        summaries.append(
            {
                "sample_index": sample_index,
                "fen": fen,
                "side_to_move": fen.split()[1] if len(fen.split()) >= 2 else None,
                "top_activated_squares": active,
                "top_z_pairs": z_pairs,
                "signals": compact_signal_fields(sample),
            }
        )
        if len(summaries) >= max_samples:
            break
    return summaries


def make_evidence_item(
    *,
    directory_id: str,
    circuit: dict[str, Any],
    feature_ref: dict[str, Any],
    feature_index_in_circuit: int,
    feature: dict[str, Any],
    max_samples: int,
    top_squares: int,
    top_z: int,
) -> dict[str, Any]:
    metadata = circuit.get("metadata") if isinstance(circuit.get("metadata"), dict) else {}
    interpretation = feature.get("interpretation") if isinstance(feature.get("interpretation"), dict) else {}
    interpretation_text = str(interpretation.get("text", "") if isinstance(interpretation, dict) else "")

    return {
        "directory_id": directory_id,
        "file_name": circuit.get("file_name"),
        "circuit_index": circuit.get("circuit_index"),
        "feature_index_in_circuit": feature_index_in_circuit,
        "dictionary_name": feature_ref.get("dictionary_name"),
        "feature_index": feature_ref.get("feature_index"),
        "layer": feature_ref.get("layer"),
        "feature_type": feature_ref.get("feature_type"),
        "node_id": feature_ref.get("node_id"),
        "label": feature_ref.get("label"),
        "existing_interpretation": interpretation_text,
        "circuit_metadata": {
            "prompt": metadata.get("prompt"),
            "target_move": metadata.get("target_move"),
            "predicted_move_uci": metadata.get("predicted_move_uci"),
            "logit_moves": metadata.get("logit_moves"),
            "lorsa_analysis_name": metadata.get("lorsa_analysis_name"),
            "tc_analysis_name": metadata.get("tc_analysis_name") or metadata.get("clt_analysis_name"),
        },
        "feature_stats": {
            "analysis_name": feature.get("analysis_name"),
            "max_feature_act": feature.get("max_feature_act"),
            "act_times": feature.get("act_times"),
            "n_analyzed_tokens": feature.get("n_analyzed_tokens"),
        },
        "top_activation_samples": summarize_feature_samples(feature, max_samples, top_squares, top_z),
    }


def resolve_directory(base_url: str, directory_id: str | None) -> str:
    if directory_id:
        return directory_id
    response = request_json(base_url, "/circuit_taxonomy/directories")
    directories = response.get("directories")
    if not isinstance(directories, list) or not directories:
        raise RuntimeError("No circuit taxonomy directories returned by backend")
    first = directories[0]
    if not isinstance(first, dict) or not first.get("id"):
        raise RuntimeError("First circuit taxonomy directory is malformed")
    return str(first["id"])


def extract_evidence(args: argparse.Namespace) -> list[dict[str, Any]]:
    directory_id = resolve_directory(args.base_url, args.directory_id)
    resume = request_json(
        args.base_url,
        "/circuit_taxonomy/resume",
        {
            "directory_id": directory_id,
            **({"file_name": args.file_name} if args.file_name else {}),
            "start_feature_index": args.start_feature_index,
        },
    )

    if resume.get("completed"):
        return []

    current_file = str(resume.get("file_name") or "")
    current_feature_index = int(resume.get("feature_index") or 0)
    circuit_cache: dict[str, dict[str, Any]] = {}
    evidence: list[dict[str, Any]] = []

    while current_file and len(evidence) < args.limit:
        if current_file not in circuit_cache:
            circuit_cache[current_file] = request_json(
                args.base_url,
                "/circuit_taxonomy/circuit",
                {"directory_id": directory_id, "file_name": current_file},
            )
        circuit = circuit_cache[current_file]
        features = circuit.get("features")
        if not isinstance(features, list):
            raise RuntimeError(f"Circuit {current_file} has no feature list")
        if current_feature_index < 0 or current_feature_index >= len(features):
            break

        feature_ref = features[current_feature_index]
        if not isinstance(feature_ref, dict):
            current_feature_index += 1
            continue

        dictionary_name = str(feature_ref.get("dictionary_name", ""))
        feature_index = int(feature_ref.get("feature_index", -1))
        feature = request_feature(args.base_url, dictionary_name, feature_index)
        evidence.append(
            make_evidence_item(
                directory_id=directory_id,
                circuit=circuit,
                feature_ref=feature_ref,
                feature_index_in_circuit=current_feature_index,
                feature=feature,
                max_samples=args.max_samples,
                top_squares=args.top_squares,
                top_z=args.top_z,
            )
        )

        next_resume = request_json(
            args.base_url,
            "/circuit_taxonomy/resume",
            {
                "directory_id": directory_id,
                "file_name": current_file,
                "start_feature_index": current_feature_index + 1,
            },
        )
        if next_resume.get("completed") or not next_resume.get("file_name") or next_resume.get("feature_index") is None:
            break
        current_file = str(next_resume["file_name"])
        current_feature_index = int(next_resume["feature_index"])

    return evidence


def export_evidence_from_backend(args: argparse.Namespace) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "limit": args.limit,
        "max_samples": args.max_samples,
        "top_squares": args.top_squares,
        "top_z": args.top_z,
        "start_feature_index": args.start_feature_index,
    }
    if args.directory_id:
        params["directory_id"] = args.directory_id
    else:
        params["directory_id"] = resolve_directory(args.base_url, None)
    if args.file_name:
        params["file_name"] = args.file_name

    response = requests.get(
        f"{args.base_url.rstrip('/')}/circuit_taxonomy/export_evidence",
        params=params,
        timeout=600,
    )
    response.raise_for_status()

    items: list[dict[str, Any]] = []
    for line_no, line in enumerate(response.text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        item = json.loads(stripped)
        if not isinstance(item, dict):
            raise ValueError(f"Export line {line_no} is not a JSON object")
        items.append(item)
    return items


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            items.append(item)
    return items


def build_label_prompt(evidence_items: list[dict[str, Any]]) -> str:
    rubric = SKILL_RUBRIC_PATH.read_text(encoding="utf-8") if SKILL_RUBRIC_PATH.exists() else ""
    evidence_lines = "\n".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) for item in evidence_items)
    return f"""Use ${SKILL_NAME} to draft circuit feature taxonomy proposals.

You are labeling candidates for a human review queue, not committing annotations.
Apply this rubric conservatively:

{rubric}

Return JSONL only. Do not include markdown fences or prose. Each line must be one JSON object with:
directory_id, file_name, circuit_index, feature_index_in_circuit, dictionary_name, feature_index, layer, feature_type, node_id, taxonomy, confidence, rationale, evidence_summary.

Allowed taxonomy values: {", ".join(sorted(TAXONOMY_LABELS))}
Use Uninterpretable when the evidence is weak or does not clearly satisfy a category.

Evidence JSONL:
{evidence_lines}
"""


def extract_json_objects(text: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return []
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json|jsonl)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass

    objects: list[dict[str, Any]] = []
    for line in stripped.splitlines():
        candidate = line.strip().removeprefix("- ").strip()
        if not candidate or not candidate.startswith("{"):
            continue
        try:
            item = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            objects.append(item)
    return objects


def validate_proposals(proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    for item in proposals:
        taxonomy = str(item.get("taxonomy", "")).strip().strip("[]")
        if taxonomy not in TAXONOMY_LABELS:
            taxonomy = "Uninterpretable"
        try:
            confidence = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        item = {**item, "taxonomy": taxonomy, "confidence": max(0.0, min(1.0, confidence))}
        if item.get("dictionary_name") and item.get("feature_index") is not None:
            valid.append(item)
    return valid


def label_with_codex(args: argparse.Namespace, evidence_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    all_proposals: list[dict[str, Any]] = []
    for start in range(0, len(evidence_items), args.chunk_size):
        chunk = evidence_items[start : start + args.chunk_size]
        prompt = build_label_prompt(chunk)
        with tempfile.NamedTemporaryFile("w+", encoding="utf-8", suffix=".jsonl", delete=False) as output_file:
            output_path = Path(output_file.name)
        command = [
            args.codex_bin,
            "exec",
            "--cd",
            str(REPO_ROOT),
            "--sandbox",
            "read-only",
            "--output-last-message",
            str(output_path),
            "-",
        ]
        if args.model:
            command.extend(["--model", args.model])
        subprocess.run(command, input=prompt, text=True, check=True)
        text = output_path.read_text(encoding="utf-8")
        output_path.unlink(missing_ok=True)
        proposals = validate_proposals(extract_json_objects(text))
        all_proposals.extend(proposals)
    return all_proposals


def command_extract(args: argparse.Namespace) -> int:
    evidence = export_evidence_from_backend(args)
    write_jsonl(args.output, evidence)
    print(f"Wrote {len(evidence)} evidence items to {args.output}")
    return 0


def command_prompt(args: argparse.Namespace) -> int:
    evidence = read_jsonl(args.evidence)
    prompt = build_label_prompt(evidence[: args.limit] if args.limit else evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(prompt, encoding="utf-8")
    print(f"Wrote Codex labeling prompt to {args.output}")
    return 0


def command_label(args: argparse.Namespace) -> int:
    evidence = read_jsonl(args.evidence)
    if args.limit:
        evidence = evidence[: args.limit]
    proposals = label_with_codex(args, evidence)
    write_jsonl(args.output, proposals)
    print(f"Wrote {len(proposals)} proposals to {args.output}")
    return 0


def command_run(args: argparse.Namespace) -> int:
    evidence = export_evidence_from_backend(args)
    write_jsonl(args.evidence_output, evidence)
    print(f"Wrote {len(evidence)} evidence items to {args.evidence_output}")
    if args.no_label:
        return 0
    proposals = label_with_codex(args, evidence)
    write_jsonl(args.proposals_output, proposals)
    print(f"Wrote {len(proposals)} proposals to {args.proposals_output}")
    return 0


def add_common_extract_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--directory-id", default=None)
    parser.add_argument("--file-name", default=None)
    parser.add_argument("--start-feature-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=6)
    parser.add_argument("--top-squares", type=int, default=8)
    parser.add_argument("--top-z", type=int, default=12)


def add_common_codex_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--model", default=None)
    parser.add_argument("--chunk-size", type=int, default=5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract", help="Extract review evidence JSONL from backend APIs")
    add_common_extract_args(extract_parser)
    extract_parser.add_argument("--output", type=Path, default=Path("outputs/circuit_taxonomy/evidence.jsonl"))
    extract_parser.set_defaults(func=command_extract)

    prompt_parser = subparsers.add_parser("prompt", help="Build a Codex labeling prompt from evidence JSONL")
    prompt_parser.add_argument("--evidence", type=Path, required=True)
    prompt_parser.add_argument("--output", type=Path, default=Path("outputs/circuit_taxonomy/label_prompt.md"))
    prompt_parser.add_argument("--limit", type=int, default=None)
    prompt_parser.set_defaults(func=command_prompt)

    label_parser = subparsers.add_parser("label", help="Ask codex exec to label evidence JSONL")
    label_parser.add_argument("--evidence", type=Path, required=True)
    label_parser.add_argument("--output", type=Path, default=Path("outputs/circuit_taxonomy/proposals.jsonl"))
    label_parser.add_argument("--limit", type=int, default=None)
    add_common_codex_args(label_parser)
    label_parser.set_defaults(func=command_label)

    run_parser = subparsers.add_parser("run", help="Extract evidence and ask codex exec to draft labels")
    add_common_extract_args(run_parser)
    add_common_codex_args(run_parser)
    run_parser.add_argument("--evidence-output", type=Path, default=Path("outputs/circuit_taxonomy/evidence.jsonl"))
    run_parser.add_argument("--proposals-output", type=Path, default=Path("outputs/circuit_taxonomy/proposals.jsonl"))
    run_parser.add_argument("--no-label", action="store_true")
    run_parser.set_defaults(func=command_run)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except requests.HTTPError as error:
        response = error.response
        print(f"HTTP error {response.status_code}: {response.text}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
