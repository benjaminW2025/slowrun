# Redirect every cache / temp / log dir that defaults to $HOME into /jumbo (the home quota is tiny).
# Sourced by every mg_study runner script; for direct `python ...` calls run `source mg_study/env.sh` first.
# Override the location with SLOWRUN_CACHE=/some/other/jumbo/dir.
SLOWRUN_CACHE=${SLOWRUN_CACHE:-/jumbo/lisp/f007twf/slowrun_cache}
export WANDB_DIR="$SLOWRUN_CACHE/wandb"                # run files
export WANDB_DATA_DIR="$SLOWRUN_CACHE/wandb_data"      # artifact staging (default ~/.local/share/wandb: the crash)
export WANDB_CACHE_DIR="$SLOWRUN_CACHE/wandb_cache"    # default ~/.cache/wandb
export WANDB_CONFIG_DIR="$SLOWRUN_CACHE/wandb_config"  # default ~/.config/wandb
export WANDB_ARTIFACT_DIR="$SLOWRUN_CACHE/wandb_artifacts"
export TRITON_CACHE_DIR="$SLOWRUN_CACHE/triton"         # default ~/.triton/cache
export TORCHINDUCTOR_CACHE_DIR="$SLOWRUN_CACHE/inductor"
export TORCH_HOME="$SLOWRUN_CACHE/torch"                # default ~/.cache/torch
export CUDA_CACHE_PATH="$SLOWRUN_CACHE/nv_compute"      # default ~/.nv/ComputeCache
export HF_HOME="$SLOWRUN_CACHE/huggingface"             # default ~/.cache/huggingface (prepare_data.py)
export XDG_CACHE_HOME="$SLOWRUN_CACHE/xdg_cache"        # anything else using ~/.cache
export PIP_CACHE_DIR="$SLOWRUN_CACHE/pip"
export MPLCONFIGDIR="$SLOWRUN_CACHE/matplotlib"
export PYTHONUNBUFFERED=1                               # nohup logs show prints immediately
export TMPDIR="$SLOWRUN_CACHE/tmp"                      # tempfile (test_proxy.py synthetic data)
mkdir -p "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_ARTIFACT_DIR" \
         "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_HOME" "$CUDA_CACHE_PATH" "$HF_HOME" \
         "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$MPLCONFIGDIR" "$TMPDIR"
