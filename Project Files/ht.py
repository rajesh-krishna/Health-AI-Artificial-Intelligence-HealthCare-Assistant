from huggingface_hub import snapshot_download
# Downloads model to local directory
snapshot_download(
    repo_id="ibm-granite/granite-3.3-2b-instruct",
    local_dir="./granite-3.3-2b-instruct",
    local_dir_use_symlinks=False  # Use this to avoid symlinks on Windows
)
