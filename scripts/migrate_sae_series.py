#!/usr/bin/env python3
"""Move one SAE and its related records to another SAE series."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

from pymongo import MongoClient
from pymongo.errors import PyMongoError


@dataclass(frozen=True)
class MigrationCounts:
    saes: int
    features: int
    analyses: int
    bookmarks: int
    sae_sets: int
    circuits: int


def count_records(db, name: str, series: str) -> MigrationCounts:
    return MigrationCounts(
        saes=db.saes.count_documents({"name": name, "series": series}),
        features=db.features.count_documents({"sae_name": name, "sae_series": series}),
        analyses=db.analyses.count_documents({"sae_name": name, "sae_series": series}),
        bookmarks=db.bookmarks.count_documents({"sae_name": name, "sae_series": series}),
        sae_sets=db.sae_sets.count_documents({"sae_names": name, "sae_series": series}),
        circuits=db.circuits.count_documents(
            {
                "sae_series": series,
                "$or": [
                    {"sae_names": name},
                    {"clt_names": name},
                    {"lorsa_names": name},
                ],
            }
        ),
    )


def print_counts(label: str, counts: MigrationCounts) -> None:
    print(f"[{label}]")
    for collection, count in vars(counts).items():
        print(f"  {collection}: {count}")


def validate_plan(source: MigrationCounts, target: MigrationCounts) -> list[str]:
    errors = []
    if source.features == 0 and source.analyses == 0:
        errors.append("source series has neither feature nor analysis documents")
    if source.saes > 1:
        errors.append(f"source series has {source.saes} SAE documents; expected at most one")
    if target.saes and source.saes:
        errors.append("both source and target series contain an SAE document")
    for collection in ("features", "analyses", "bookmarks"):
        if getattr(target, collection):
            errors.append(f"target series already contains {collection} documents")
    if source.sae_sets or source.circuits:
        errors.append("source SAE is referenced by SAE sets or circuits; migrate those relationships explicitly")
    return errors


def migrate(db, name: str, source_series: str, target_series: str, session) -> dict[str, int]:
    session_kwargs = {"session": session} if session is not None else {}
    results = {
        "saes": db.saes.update_many(
            {"name": name, "series": source_series},
            {"$set": {"series": target_series}},
            **session_kwargs,
        ).modified_count,
        "features": db.features.update_many(
            {"sae_name": name, "sae_series": source_series},
            {"$set": {"sae_series": target_series}},
            **session_kwargs,
        ).modified_count,
        "analyses": db.analyses.update_many(
            {"sae_name": name, "sae_series": source_series},
            {"$set": {"sae_series": target_series}},
            **session_kwargs,
        ).modified_count,
        "bookmarks": db.bookmarks.update_many(
            {"sae_name": name, "sae_series": source_series},
            {"$set": {"sae_series": target_series}},
            **session_kwargs,
        ).modified_count,
    }
    return results


def rollback_non_transactional(
    db,
    name: str,
    source_series: str,
    target_series: str,
    original_source: MigrationCounts,
) -> None:
    """Move only originally present source records back after a failed migration."""
    if original_source.saes:
        db.saes.update_many(
            {"name": name, "series": target_series},
            {"$set": {"series": source_series}},
        )
    if original_source.features:
        db.features.update_many(
            {"sae_name": name, "sae_series": target_series},
            {"$set": {"sae_series": source_series}},
        )
    if original_source.analyses:
        db.analyses.update_many(
            {"sae_name": name, "sae_series": target_series},
            {"$set": {"sae_series": source_series}},
        )
    if original_source.bookmarks:
        db.bookmarks.update_many(
            {"sae_name": name, "sae_series": target_series},
            {"$set": {"sae_series": source_series}},
        )


def verify_migration(
    db,
    name: str,
    source_series: str,
    target_series: str,
    original_source: MigrationCounts,
    original_target: MigrationCounts,
) -> tuple[MigrationCounts, MigrationCounts]:
    source_after = count_records(db, name, source_series)
    target_after = count_records(db, name, target_series)
    moved_collections = ("saes", "features", "analyses", "bookmarks")
    for collection in moved_collections:
        if getattr(source_after, collection) != 0:
            raise RuntimeError(f"source still contains {collection} documents")
        expected = getattr(original_source, collection) + getattr(original_target, collection)
        actual = getattr(target_after, collection)
        if actual != expected:
            raise RuntimeError(f"target {collection} count is {actual}; expected {expected}")
    return source_after, target_after


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Exact SAE name to migrate.")
    parser.add_argument("--from-series", required=True, help="Current SAE series.")
    parser.add_argument("--to-series", required=True, help="Destination SAE series.")
    parser.add_argument("--mongo-uri", default=os.environ.get("MONGO_URI", "mongodb://localhost:27017/"))
    parser.add_argument("--mongo-db", default=os.environ.get("MONGO_DB", "llamascope2"))
    parser.add_argument("--apply", action="store_true", help="Execute the migration. Without this flag, only inspect.")
    parser.add_argument(
        "--allow-non-transactional",
        action="store_true",
        help="Allow guarded migration on a standalone MongoDB server that does not support transactions.",
    )
    args = parser.parse_args()

    if args.from_series == args.to_series:
        parser.error("--from-series and --to-series must differ")

    client = MongoClient(args.mongo_uri)
    db = client[args.mongo_db]
    client.admin.command("ping")

    source = count_records(db, args.name, args.from_series)
    target = count_records(db, args.name, args.to_series)
    print(f"SAE: {args.name}")
    print(f"database: {args.mongo_db}")
    print(f"series: {args.from_series} -> {args.to_series}")
    print_counts("source", source)
    print_counts("target", target)

    errors = validate_plan(source, target)
    if errors:
        print("[refused] migration preflight failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 2

    if not args.apply:
        print("[dry-run] preflight passed; rerun with --apply to execute the transaction")
        return 0

    if args.allow_non_transactional:
        print("[warning] running without a MongoDB transaction")
        try:
            modified = migrate(db, args.name, args.from_series, args.to_series, session=None)
            remaining, destination = verify_migration(
                db,
                args.name,
                args.from_series,
                args.to_series,
                source,
                target,
            )
        except (PyMongoError, RuntimeError) as exc:
            print(f"[failed] {exc}; attempting rollback", file=sys.stderr)
            rollback_non_transactional(db, args.name, args.from_series, args.to_series, source)
            print("[rollback] completed; run a dry-run to verify counts", file=sys.stderr)
            return 1
    else:
        try:
            with client.start_session() as session:
                with session.start_transaction():
                    modified = migrate(db, args.name, args.from_series, args.to_series, session)
            remaining, destination = verify_migration(
                db,
                args.name,
                args.from_series,
                args.to_series,
                source,
                target,
            )
        except PyMongoError as exc:
            print(f"[failed] transaction aborted: {exc}", file=sys.stderr)
            print(
                "If this is a standalone MongoDB server, rerun with --allow-non-transactional.",
                file=sys.stderr,
            )
            return 1

    print("[applied]")
    for collection, count in modified.items():
        print(f"  {collection}: {count} documents moved")

    print_counts("source after migration", remaining)
    print_counts("target after migration", destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
