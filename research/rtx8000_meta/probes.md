# Cross-batch optimizer probes

Diagnostics are optional and do not modify the training algorithm. Output is `runs/<run-name>/probes.jsonl`, one JSON record per probe event. `--probe-every N` probes before update 1, then N+1, 2N+1, etc.; default 0 disables them. A and B are the actual first virtual rank's two halves, reused without advancing the data loader. This samples one episode per event, not all eight ranks or a validation set.

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
