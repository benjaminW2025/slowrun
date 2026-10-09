# RTX 8000: serial eight-rank meta-gradient experiment

This directory starts from the [Tiny entry 12 submission](https://github.com/qlabs-eng/slowrun/blob/a2c443e18f99d0f6e35c6013c752e636fe402213/tiny/train.py), commit `a2c443e18f99d0f6e35c6013c752e636fe402213`. It predates the FP8 multi-token loss, so it is a feasible pre-meta **mechanism baseline** for a Quadro RTX 8000. It is **not** the later Tiny entry 14 recipe used by [meta-gradient PR #99](https://github.com/qlabs-eng/slowrun/pull/99); numbers from this experiment must not be presented as a reproduction of PR #99's leaderboard result.

## Fixed parts of the comparison

- The source submission's model, data loader, next-token loss, learning-rate and weight-decay schedules, checkpoint/EMA logic, and parameter grouping are copied into `train.py`.
- Its `DistMuonAdamW`, `adamw_step_fused`, and `muon_step_fused` implementations are identical in syntax to the later Tiny PR before the RTX precision change. Muon still updates the matrix groups; AdamW still updates embeddings, the output head, and scalar/small groups. The temporary trial is a separate normalized raw-gradient step, **not** a Muon step.
- All arms use one optimizer update per global batch of 524,288 tokens. `off`, `mlp`, `qk`, and `v` differ only in which matrices receive a temporary step before B; the final Muon+AdamW optimizer always updates every parameter from the accumulated gradient.
- `mlp` means all `*.mlp.*` Muon matrices; `qk` means attention `c_q` and `c_k`; `v` means attention `c_v`. It does not include value-embedding projections or attention output projections.

## One serial optimizer step

The 8×H100 PR gives each GPU 32 sequences and splits those into 16 A / 16 B. Here eight *virtual ranks* receive those groups sequentially. For virtual rank `r`, A's gradient is calculated at the common starting weights `θ`, and the selected matrices are moved by `−0.5 g_{A,r}/||g_{A,r}||₂`. B's gradient is calculated at that rank's temporary weights. All weights are restored exactly before the next virtual rank. The two half-gradients from each rank are included at their sample-correct weights, then averaged across all eight ranks; Muon+AdamW runs once. `off` uses the identical batches and accumulation, without a temporary step. A smaller `--device-batch-size` subdivides each virtual rank's A and B half-batches without changing the global token count.

The port uses FP32 parameters and optimizer state, FP16 autocast with gradient scaling, and FP32 Muon polar iterations by default. This preserves the optimizer *formula* while changing its numerical precision from the H100 recipe's BF16. `--precision fp32` disables autocast for a slower correctness reference. The source submission already falls back from Hopper-only FlashAttention 3 to PyTorch scaled-dot-product attention. A fixed-seed serial simulation will not be bitwise identical to eight concurrently seeded GPUs, but the gradient formula is the same up to floating-point/RNG differences.

## Run

From the repository root, install `requirements.txt`, prepare `fineweb_data/fineweb_train.pt` and `fineweb_val.pt` using the repository's `prepare_data.py`, and use a CUDA-enabled PyTorch build that supports the RTX 8000. Launch with `torchrun --standalone --nproc_per_node=1`; the copied optimizer expects an initialized process group even with one GPU. Set `WANDB_MODE=offline` if you do not want online experiment logging.

First run a *short smoke test* with `--mg-mode off` and then `--mg-mode mlp`, and inspect finite loss, peak VRAM, step time, and raw final validation loss. For example:

```bash
export WANDB_MODE=offline
torchrun --standalone --nproc_per_node=1 research/rtx8000_meta/train.py \
  --mg-mode off --seed 42 --device-batch-size 4 \
  --max-train-steps 4 --schedule-steps 4 --eval-tokens 131072 --swa-last-epochs 0 \
  --run-name rtx-smoke-off
```

Then benchmark 10–20 warmed-up full-size steps to choose a feasible pilot length. Use **identical arguments** for `off`, `mlp`, `qk`, and `v`, except `--mg-mode` and unique run names. For a short pilot, set `--schedule-steps` equal to `--max-train-steps` to compress LR/WD/recurrence schedules consistently in every arm; set `--swa-last-epochs 0` because a short run does not traverse the source submission's actual epoch/checkpoint schedule. For example, after timing shows it is feasible:

```bash
torchrun --standalone --nproc_per_node=1 research/rtx8000_meta/train.py \
  --mg-mode qk --seed 42 --device-batch-size 4 \
  --max-train-steps 100 --schedule-steps 100 --eval-tokens 1048576 --swa-last-epochs 0 \
  --run-name rtx-qk-s42
```

For a memory error, reduce `--device-batch-size` to 2 or 1; the global batch and virtual-rank token count remain fixed, but each optimizer step takes more microbatches. For an FP16 overflow, lower `--loss-scale`, or use `--precision fp32` for diagnosis. The run fails on an overflow rather than silently counting a skipped optimizer step.

Each run writes `runs/<run-name>/result.json`, its exact training script and virtual-rank helper, and a terminal log. Compare `raw_final_val_loss` first; EMA and checkpoint-averaged scores are additional outcomes. Do not choose an arm based on a single small difference: repeat the best comparison with matched seeds and equal effective tokens. No GPU training has been run in this repository yet.

## Local correctness check

`python3 research/rtx8000_meta/test_virtual_rank.py` tests the serial gradient against independently computed A and B gradients, including multiple microbatches, loss scaling, exact weight restoration, and matrix selection. It runs on CPU without FineWeb data.
