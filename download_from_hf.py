from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="google/gemma-scope-2-270m-pt",
    repo_type="model",
    allow_patterns="transcoder_all/*",
    local_dir="/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/llamascope2/gemma-scope-2-270m-pt"
)