import argparse
import json
import os
import re
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "google/gemma-scope-2-270m-pt"
PATH_IN_REPO = "transcoder_all"
DIR_PATTERN = re.compile(
    rf"{PATH_IN_REPO}/layer_(\d+)_width_16k_l0_big/(.+)$"
)
DOWNLOAD_ORDER = {
    "config.json": 0,
    "params.safetensors": 1,
    "examples.safetensors": 2,
}
DEFAULT_LOCAL_DIR = (
    Path(__file__).resolve().parent
    / "downloads"
    / "gemma-scope-2-270m-pt"
    / "transcoder_all_width_16k_l0_big"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Gemma Scope 2 270M transcoder_all width_16k l0_big dictionaries."
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path(os.environ.get("HF_LOCAL_DIR", DEFAULT_LOCAL_DIR)),
        help="Download destination. Can also be set with HF_LOCAL_DIR.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list matched files and per-layer l0 values.",
    )
    return parser.parse_args()


def collect_target_files(api: HfApi) -> list[tuple[int, int, str]]:
    items = api.list_repo_tree(
        repo_id=REPO_ID,
        repo_type="model",
        path_in_repo=PATH_IN_REPO,
        recursive=True,
    )

    target_files = []
    for item in items:
        path = getattr(item, "path", "")
        match = DIR_PATTERN.match(path)
        if not match:
            continue

        layer = int(match.group(1))
        filename = match.group(2)
        order = DOWNLOAD_ORDER.get(filename, 9)
        target_files.append((layer, order, path))

    return sorted(target_files)


def load_l0_by_layer(target_files: list[tuple[int, int, str]]) -> dict[int, int]:
    l0_by_layer = {}
    for layer, _, path in target_files:
        if not path.endswith("/config.json"):
            continue

        config_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="model",
            filename=path,
        )
        with open(config_path) as f:
            config = json.load(f)

        l0_by_layer[layer] = config["l0"]

    return dict(sorted(l0_by_layer.items()))


def main() -> None:
    args = parse_args()
    api = HfApi()
    target_files = collect_target_files(api)
    l0_by_layer = load_l0_by_layer(target_files)

    print(f"Repo: {REPO_ID}")
    print(f"Pattern: {PATH_IN_REPO}/layer_*_width_16k_l0_big")
    print(f"Local dir: {args.local_dir}")
    print(f"Found {len(target_files)} files to download")
    print("\nPer-layer l0:")
    for layer, l0 in l0_by_layer.items():
        print(f"  layer {layer}: l0={l0}")

    if args.dry_run:
        print("\nDry run only; no model files downloaded.")
        return

    for i, (layer, _, path) in enumerate(target_files, 1):
        print(f"\n[{i}/{len(target_files)}] layer {layer}: {path}", flush=True)
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="model",
            filename=path,
            local_dir=str(args.local_dir),
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
