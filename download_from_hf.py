from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="google/gemma-scope-2-270m-pt",
    repo_type="model",
    allow_patterns="transcoder_all/*",
    local_dir="./gemma_scope"
)