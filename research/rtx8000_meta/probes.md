# Cross-batch optimizer probes

Diagnostics are optional and do not modify the training algorithm. Output is `runs/<run-name>/probes.jsonl`, one JSON record per probe event. `--probe-every N` probes before update 1, then N+1, 2N+1, etc.; default 0 disables them. During training, A/B are captured from the first `--probe-pairs` virtual ranks (default one, maximum eight). Every matrix group uses the same pairs. `--probe-only --probe-alignment-only` instead reads just the requested pairs and exits without an optimizer update or an unnecessary training backward. These are training-corpus sequences, not a validation set; distinct sequences need not come from independent documents.

## What is measured

- Selected weight Frobenius norm, per matrix and jointly across the selected group.
- A and B gradient norms and cosine alignment, both at the original unadapted weights.
- Per-matrix Muon A update norm, update/weight ratio, cautious decay norm and fraction of coordinates receiving decay. A zero weight norm gives a null relative ratio, not infinity.
- B loss after raw A descent and Muon-transformed A descent at matched Euclidean radii (default 0.1 and 0.5). The radius is across the entire selected group, not per matrix.
- B loss after native Muon A descent, cautious decay alone, and their sum, using the current scheduled LR/WD and the current live momentum/variance buffers.
- The same native decomposition for the actual averaged training gradient, including the meta-gradient algorithm's B contribution.
- For each perturbation: its absolute and relative norm, actual B-loss change, first-order prediction `g_B dot displacement`, and nonlinear residual. Negative B-loss change means improvement. A negative cosine between B gradient and the Muon displacement is locally favorable.

The A-only preview asks what happens if the next optimizer update uses this episode's A gradient with the live optimizer state. It is not a cold-start Muon direction. The actual-training-gradient preview describes a restricted part of the real upcoming update. Because that gradient already includes B, its improvement on B is not independent transfer evidence. Both previews modify only the selected matrices; embeddings, scalars, and all other matrices remain fixed. The full real optimizer updates more parameters.

Muon preview executes the production kernel body with copied momentum and variance state. It applies the original shape-dependent LR factor. Cautious decay masks are computed from the corresponding transformed update at the original weights. All native components are evaluated starting from the same original weights; combined decay is not applied as a second sequential step with a recomputed mask.

Probe gradients use the same FP16 autocast and loss scale as training, with normalization after backward to avoid underflow. Dropout modules are disabled during probes to reduce measurement noise, while the model's training recurrence/backprop policy remains enabled. Probe forward calls use the original eager model to avoid recompiling the training model for dropout changes. Parameters, training gradient tensors, optimizer buffers, module modes and Torch RNG are preserved. Weights restore exactly even on a probe exception. Checkpoints do not include temporary perturbations.

## Adapted B gradient measurements

For each raw normalized A perturbation and radius, compute `g_B_prime` on the same B samples at `theta + d_A`. The five main measurements are:

1. `cosine_a_b`: cosine between A and B gradients at original theta.
2. `b_loss_change`: B loss after the A perturbation minus B loss at original theta.
3. `cosine_a_b_prime`: A gradient versus adapted B gradient.
4. `relative_b_gradient_change`: `||g_B_prime - g_B|| / ||g_B||`.
5. `cosine_a_b_correction`: A gradient versus `g_B_prime - g_B`.

Direct B comparisons add `cosine_b_b_prime`, `b_prime_norm`, `b_prime_norm_ratio` and `b_correction_norm`. Both original gradient norms are also recorded. These measurements exist jointly across the selected group and individually per matrix, except the B-loss change, which measures the joint perturbation. The per-matrix correction does not represent a separate intervention on that matrix alone. Raw gradient tensors are not saved: retaining all matrices for every pair/radius would require large files. Zero-norm cosine/ratio measurements are null, not fabricated zeros. A zero A gradient produces a logged no-op perturbation.

`--probe-alignment-only` skips the Muon/decay scenarios to focus GPU time and memory on these measurements. It does not require populated optimizer state. Loading `--init-model` restores parameters only, not optimizer buffers. This is sufficient for raw-gradient alignment profiling; live-state Muon diagnostics still require an online training run with populated state.

## Profile an existing checkpoint

Use the actual path to your saved checkpoint. `model.pt` may contain EMA or checkpoint-averaged parameters: check its run's `result.json` `model_variant`. It need not be the exact raw model associated with `raw_final_val_loss`.

```bash
CHECKPOINT=runs/audit-v-500/model.pt
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled python research/rtx8000_meta/train.py \
  --init-model "$CHECKPOINT" --mg-mode off --precision fp16 --loss-scale 128 \
  --device-batch-size 1 --virtual-ranks 8 --total-batch-size 524288 \
  --schedule-steps 500 --max-train-steps 1 --eval-tokens 32768 \
  --iteration-schedule constant --num-iterations 2 --seed 42 --no-compile \
  --probe-every 1 --probe-pairs 8 --probe-modes qk,v,mlp \
  --probe-radii 0.1,0.5 --probe-alignment-only --probe-only \
  --run-name alignment-v-checkpoint
```

This uses the final two-pass recurrence with the inherited one-pass backprop policy. Eight pairs contain 16 sequences in A and 16 in B each, for 256 sequences / 524,288 target-token positions total. Both radii and all three groups reuse those sequences. No optimizer update occurs; the profiling run writes `probes.jsonl` and exits, without saving a replacement model or a new training result.

```bash
python research/rtx8000_meta/summarize_probes.py \
  runs/alignment-v-checkpoint/probes.jsonl --alignment
python research/rtx8000_meta/summarize_probes.py \
  runs/alignment-v-checkpoint/probes.jsonl --alignment --aggregate
```

Aggregate output groups by file, update, selected group and radius. It reports means and sample standard deviations across pairs, counts defined values, and reports how many pairs suffered a B-loss increase. Eight pairs are an initial descriptive profile, not a definitive significance test. Repeat with `--probe-pair-offset 8`, then 16, etc., and fresh run names to sample additional pairs at the same checkpoint. Changing `--seed` alone does not change the loader’s deterministic first-epoch permutation. The offset is available only for standalone alignment profiling so it cannot silently skip training data.

## Interpretation

Compare raw and transformed directions at the same radius before interpreting the native Muon result: native update sizes vary across parameter groups. Matching Euclidean radius does not match functional movement or remove parameterization effects. Inspect weight-relative step size, multiple radii and multiple A/B pairs before concluding that a group intrinsically transfers better.

The nonlinear residual includes finite-step curvature and numerical effects; it is not an exact Hessian measurement. With recurrent training, the gradient can be a truncated-backprop gradient, so its dot product need not equal the derivative of the full recurrent forward-loss function. These measurements are optimization diagnostics, not proof of validation generalization.

Probes add forward/backward work and temporary memory. Wall-clock comparisons should use probe time separately (`probe_seconds`) or disable probes. Begin on a 48-GB RTX 8000. Full-size CUDA memory use and numerical results remain unverified locally; the CPU suite checks equation equivalence and preservation of training state.

## Commands

After activating the environment and pulling `codex/experiments`, run a short smoke test with probes before updates 1 and 3:

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled python research/rtx8000_meta/train.py \
  --mg-mode mlp --precision fp16 --loss-scale 128 --device-batch-size 1 \
  --virtual-ranks 8 --total-batch-size 524288 --schedule-steps 500 \
  --max-train-steps 3 --eval-tokens 32768 --seed 42 --no-compile \
  --probe-every 2 --probe-modes mlp --probe-radii 0.1,0.5 \
  --run-name probe-smoke-mlp
```

For a full MLP training run, probe MLP, QK and V at the same checkpoints:

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled python research/rtx8000_meta/train.py \
  --mg-mode mlp --precision fp16 --loss-scale 128 --device-batch-size 1 \
  --virtual-ranks 8 --total-batch-size 524288 --schedule-steps 500 \
  --max-train-steps 500 --eval-tokens 262144 --seed 42 --no-compile \
  --probe-every 50 --probe-modes mlp,qk,v --probe-radii 0.1,0.5 \
  --run-name probe-mlp-500
```

Summarize the main measurements:

```bash
python research/rtx8000_meta/summarize_probes.py runs/probe-mlp-500/probes.jsonl
```

To compare training trajectories, repeat the same diagnostic command with `--mg-mode off` and a fresh run name. Existing runs started without diagnostics continue unchanged and cannot emit these probes retroactively.
