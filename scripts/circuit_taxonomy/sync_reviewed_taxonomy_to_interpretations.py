#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

LABELS = {
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
}
PREFIX_RE = re.compile(r"^\[(%s)\]\s*" % "|".join(sorted(LABELS)))


def compact_text(text: str, *, max_len: int = 180) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    text = text.strip(" ,.;")
    if len(text) <= max_len:
        return text
    cut = text[:max_len].rsplit(" ", 1)[0] or text[:max_len]
    return cut.rstrip(" ,.;") + "..."


def interpretation_text(label: str, rationale: str) -> str:
    rationale = compact_text(rationale)
    return f"[{label}] {rationale}" if rationale else f"[{label}]"


def build_rationale(raw: dict[str, Any], label: str) -> str:
    if raw.get("reviewed_taxonomy"):
        evidence = compact_text(str(raw.get("evidence_summary") or raw.get("rationale") or ""), max_len=210)
        if evidence:
            return f"Human review set taxonomy to {label}. Evidence: {evidence}"
        return f"Human review set taxonomy to {label}."
    return str(raw.get("rationale") or raw.get("evidence_summary") or "")


def input_files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.glob("*.proposals.jsonl"))
    return [path]


def load_items(path: Path, *, approved_only: bool) -> list[dict[str, Any]]:
    items: dict[tuple[str, int], dict[str, Any]] = {}
    for source_path in input_files(path):
        with source_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                status = str(raw.get("status", "")).strip().lower()
                if approved_only and status and status != "approved":
                    continue

                dictionary_name = str(raw.get("dictionary_name", "")).strip()
                feature_index = raw.get("feature_index")
                label = str(raw.get("reviewed_taxonomy") or raw.get("taxonomy") or "").strip()
                if not dictionary_name or label not in LABELS:
                    continue
                try:
                    feature_index = int(feature_index)
                except (TypeError, ValueError):
                    continue

                rationale = build_rationale(raw, label)
                proposed_text = raw.get("interpretation")
                if not isinstance(proposed_text, str):
                    proposed_text = interpretation_text(label, str(rationale))
                key = (dictionary_name, feature_index)
                item = {
                    "dictionary_name": dictionary_name,
                    "feature_index": feature_index,
                    "taxonomy": label,
                    "text": compact_text(proposed_text, max_len=260),
                    "rationale": compact_text(str(rationale), max_len=260),
                    "confidence": raw.get("confidence"),
                    "method": "taxonomy_review",
                    "validation": raw.get("validation") or [],
                    "source": {
                        "path": str(source_path),
                        "line": line_no,
                        "id": raw.get("id"),
                        "file_name": raw.get("file_name"),
                        "node_id": raw.get("node_id"),
                        "original_taxonomy": raw.get("original_taxonomy") or raw.get("taxonomy"),
                        "status": raw.get("status"),
                    },
                }
                previous = items.get(key)
                if previous and any(previous[field] != item[field] for field in ("taxonomy", "text", "rationale", "validation")):
                    raise ValueError(f"Conflicting duplicate interpretation for {dictionary_name}:{feature_index}")
                items[key] = item
    return list(items.values())


def write_preview(items: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def sync_to_mongo(items: list[dict[str, Any]], *, mongo_uri: str, db_name: str, sae_series: str) -> int:
    from pymongo import MongoClient, UpdateOne

    client = MongoClient(mongo_uri)
    collection = client[db_name]["features"]
    operations = []
    for item in items:
        interpretation = {
            "text": item["text"],
            "taxonomy": item["taxonomy"],
            "rationale": item["rationale"],
            "method": item["method"],
            "validation": item["validation"],
            "source": item["source"],
        }
        if item.get("confidence") is not None:
            interpretation["confidence"] = item["confidence"]
        operations.append(
            UpdateOne(
                {
                    "sae_name": item["dictionary_name"],
                    "sae_series": sae_series,
                    "index": item["feature_index"],
                },
                {"$set": {"interpretation": interpretation}},
            )
        )
    if not operations:
        return 0
    result = collection.bulk_write(operations, ordered=False)
    return int(result.modified_count + result.upserted_count)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync reviewed taxonomy labels/rationales into feature interpretations.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("scripts/circuit_taxonomy/circuit-taxonomy-review-edits (17).jsonl"),
    )
    parser.add_argument("--preview-output", type=Path, default=Path("outputs/circuit_taxonomy_reviews/compact-interpretations.jsonl"))
    parser.add_argument("--sae-series", default=os.environ.get("SAE_SERIES", "default"))
    parser.add_argument("--mongo-uri", default=os.environ.get("MONGO_URI", "mongodb://localhost:27017"))
    parser.add_argument("--db-name", default=os.environ.get("MONGO_DB", "mechinterp"))
    parser.add_argument("--include-non-approved", action="store_true")
    parser.add_argument("--write", action="store_true", help="Write to MongoDB. Without this flag only a preview JSONL is generated.")
    args = parser.parse_args()

    items = load_items(args.input, approved_only=not args.include_non_approved)
    write_preview(items, args.preview_output)
    print(f"Prepared {len(items)} unique feature interpretations")
    print(f"Preview: {args.preview_output}")

    if not args.write:
        print("Dry run only. Re-run with --write to update MongoDB.")
        return 0

    try:
        changed = sync_to_mongo(items, mongo_uri=args.mongo_uri, db_name=args.db_name, sae_series=args.sae_series)
    except Exception as error:
        print(f"Mongo sync failed: {error}", file=sys.stderr)
        return 1
    print(f"MongoDB updated documents: {changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
