import importlib.util
import sys
from pathlib import Path

import mongomock

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "migrate_sae_series.py"
SPEC = importlib.util.spec_from_file_location("migrate_sae_series", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

MigrationCounts = MODULE.MigrationCounts
migrate = MODULE.migrate
rollback_non_transactional = MODULE.rollback_non_transactional
validate_plan = MODULE.validate_plan
verify_migration = MODULE.verify_migration


def test_migrate_changes_only_series_fields():
    db = mongomock.MongoClient().test
    name = "matryoshka"
    source = "source-series"
    target = "target-series"
    db.saes.insert_one({"name": name, "series": source, "path": "sae-path"})
    db.features.insert_one(
        {
            "sae_name": name,
            "sae_series": source,
            "index": 7,
            "analyses": [{"name": "default", "payload": "preserved"}],
        }
    )
    db.analyses.insert_one({"name": "default", "sae_name": name, "sae_series": source})
    db.bookmarks.insert_one({"sae_name": name, "sae_series": source, "feature_index": 7})

    modified = migrate(db, name, source, target, session=None)

    assert modified == {"saes": 1, "features": 1, "analyses": 1, "bookmarks": 1}
    feature = db.features.find_one({"sae_name": name, "sae_series": target})
    assert feature is not None
    assert feature["index"] == 7
    assert feature["analyses"] == [{"name": "default", "payload": "preserved"}]
    assert db.features.count_documents({"sae_name": name, "sae_series": source}) == 0


def test_validate_plan_rejects_target_conflicts():
    source = MigrationCounts(saes=1, features=10, analyses=1, bookmarks=0, sae_sets=0, circuits=0)
    target = MigrationCounts(saes=0, features=1, analyses=0, bookmarks=0, sae_sets=0, circuits=0)

    assert validate_plan(source, target) == ["target series already contains features documents"]


def test_rollback_does_not_move_preexisting_target_sae():
    db = mongomock.MongoClient().test
    name = "matryoshka"
    source = "source-series"
    target = "target-series"
    db.saes.insert_one({"name": name, "series": target, "path": "target-metadata"})
    db.features.insert_one({"sae_name": name, "sae_series": target, "index": 0})
    db.analyses.insert_one({"name": "default", "sae_name": name, "sae_series": target})
    original_source = MigrationCounts(saes=0, features=1, analyses=1, bookmarks=0, sae_sets=0, circuits=0)

    rollback_non_transactional(db, name, source, target, original_source)

    assert db.saes.count_documents({"name": name, "series": target}) == 1
    assert db.features.count_documents({"sae_name": name, "sae_series": source}) == 1
    assert db.analyses.count_documents({"sae_name": name, "sae_series": source}) == 1


def test_verify_migration_accepts_preexisting_target_sae():
    db = mongomock.MongoClient().test
    name = "matryoshka"
    source = "source-series"
    target = "target-series"
    db.saes.insert_one({"name": name, "series": target})
    db.features.insert_one({"sae_name": name, "sae_series": target, "index": 0})
    db.analyses.insert_one({"name": "default", "sae_name": name, "sae_series": target})
    original_source = MigrationCounts(saes=0, features=1, analyses=1, bookmarks=0, sae_sets=0, circuits=0)
    original_target = MigrationCounts(saes=1, features=0, analyses=0, bookmarks=0, sae_sets=0, circuits=0)

    source_after, target_after = verify_migration(db, name, source, target, original_source, original_target)

    assert source_after.features == 0
    assert target_after.saes == 1
    assert target_after.features == 1
