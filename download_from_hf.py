import re
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

repo_id = "google/gemma-scope-2-12b-pt"
local_dir = Path(
    "/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/llamascope2/train/gemma_scope_12b"
)

api = HfApi()

items = api.list_repo_tree(
    repo_id=repo_id,
    repo_type="model",
    path_in_repo="transcoder_all",
    recursive=True,
)

target_files = []

for item in items:
    path = getattr(item, "path", "")

    m = re.match(r"transcoder_all/layer_(\d+)_width_16k_l0_small/(.+)$", path)
    if not m:
        continue

    layer = int(m.group(1))
    filename = m.group(2)

    if filename == "config.json":
        order = 0
    elif filename == "params.safetensors":
        order = 1
    elif filename == "examples.safetensors":
        order = 2
    else:
        order = 9

    target_files.append((layer, order, path))

target_files.sort()

print(f"Found {len(target_files)} files to download")

for i, (layer, order, path) in enumerate(target_files, 1):
    print(f"\n[{i}/{len(target_files)}] layer {layer}: {path}", flush=True)

    hf_hub_download(
        repo_id=repo_id,
        repo_type="model",
        filename=path,
        local_dir=str(local_dir),
        resume_download=True,
    )

print("\nDone.")