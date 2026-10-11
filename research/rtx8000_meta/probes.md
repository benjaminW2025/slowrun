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

## Muon B-prime and A/B/C transfer (schema 3)

`--probe-muon-bprime` extends the raw alignment profile. It requires
`--probe-alignment-only --probe-every N`. The flag name means the old native
A/decay scenarios are skipped; the explicitly requested Muon B-prime comparisons
still run. Standalone `--probe-only` works without a preliminary training backward.
Online probes use the current optimizer buffers. All previews use private copies
of the same buffers and never commit a candidate step.

For each selected group and A radius, record:

- `cosine_muon_a_b_prime`: native Muon descent directions from A and B-prime.
- `cosine_raw_a_muon_b_prime`: raw A gradient versus the negative Muon B-prime
  displacement (a gradient-like direction, including per-shape LR factors).
- `cosine_muon_b_b_prime`, `relative_muon_b_change`, and
  `cosine_muon_a_correction`: what survives or changes through Muon.
- `cosine_muon_paired_meta`, `relative_muon_paired_meta_change`: Muon applied
  directly to `(A+B)/2` versus directly to `(A+Bprime)/2`. Muon is nonlinear;
  averaging individually transformed A/B directions is not this computation.
- Native B-prime displacement norm and its ratio to B's. Metrics are also stored
  per matrix; these refer to a joint group perturbation, not independent matrix trials.

The transformed correction compares `Muon(Bprime) - Muon(B)`, not
`Muon(Bprime - B)`. The separate C correction trial transforms `(Bprime-B)/2`
on its own. Neither substitutes for comparing the complete paired and meta
updates, because the transform is nonlinear.

B-prime is computed at the temporarily perturbed weights. Original weights are
restored BEFORE any candidate optimizer preview or C displacement. Every candidate
starts at original theta. Muon previews include momentum, Polar Express, variance
adjustment and shape-dependent LR. Weight decay is excluded to isolate the gradient
transformation.

`--init-model` loads weights only. Schema 3 records `optimizer_state=fresh` when
there is no history, or `live` for online state. A checkpoint profile with fresh
state is not a reconstruction of that checkpoint's original optimizer update.
Use online probes in a corrected training run to inspect real accumulated history.

`--probe-abc` additionally reads C from a separate validation loader. It uses the
same number of sequences as each A/B half. C neither enters candidate construction
nor advances the training loader; all groups/candidates reuse the same C examples.
C's gradient is used only to report directional alignment and a linearized loss
change. A and B remain training examples. The same input file for training/C is
rejected, but separate filenames alone cannot prove corpus-level deduplication.

At original theta, compare raw descent and Muon descent from:

1. A alone; B alone; their arithmetic mean (`paired`).
2. `agreement`: per coordinate, keep the shared sign with the smaller absolute
   gradient magnitude; otherwise zero. This is an explicit, basis-dependent
   heuristic, not a unique shared component or the AND-mask algorithm.
3. `remainder = paired - agreement`. Together they reconstruct the paired gradient.
4. `zero_input`: control for the effect of optimizer history without a current
   gradient. Fresh-state Muon gives a no-op; live momentum can give a nonzero step.
5. For each A adaptation radius: B-prime alone (`adapted_b`), the meta-gradient
   `(A+Bprime)/2` (`meta`), and `(Bprime-B)/2` alone (`meta_correction`).

Every C trial has the same group-wide Euclidean displacement norm, set separately
by `--probe-transfer-radius` (default 0.1). This is independent of `--probe-radii`,
which sets the temporary A perturbation. Zero directions are recorded as no-ops.
A normalized correction-only trial is a direction test: it does not represent its
actual fractional contribution to the production update. These are restricted
selected-group interventions using one A/B pair's gradient, not full-model or
full-global-batch optimizer steps.

Record `c_loss_change` (negative improves C), `cosine_c_displacement` (negative is
locally favorable), input gradient norm, native displacement norm, applied norm,
linearized loss change and residual. With truncated recurrent backprop, gradient
projections are not exact derivatives of the full recurrent forward objective;
actual C-loss changes are the primary endpoint. Repeatedly selecting hypotheses
using C turns it into development data; confirm a chosen intervention separately.

Start with one pair to measure runtime/memory before requesting eight. Use a fresh
run name; `probes.jsonl` appends, so reusing names can mix old and corrected events.
Extended summaries keep each appended event separate.

```bash
# First: Muon B-prime only, on the same existing checkpoint.
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled python research/rtx8000_meta/train.py \
  --init-model runs/audit-v-500/model.pt \
  --mg-mode off --precision fp16 --loss-scale 128 --no-compile \
  --device-batch-size 1 --virtual-ranks 8 --total-batch-size 524288 \
  --schedule-steps 500 --max-train-steps 1 --eval-tokens 32768 \
  --iteration-schedule constant --num-iterations 2 --seed 42 \
  --probe-every 1 --probe-pairs 1 --probe-modes qk,v,mlp \
  --probe-radii 0.1,0.5 --probe-alignment-only --probe-muon-bprime --probe-only \
  --run-name muon-bprime-v-one-pair

# Then: same settings, adding --probe-abc --probe-transfer-radius 0.1
# and changing --run-name to abc-v-one-pair.

python research/rtx8000_meta/summarize_probes.py \
  runs/muon-bprime-v-one-pair/probes.jsonl --muon-bprime --aggregate
python research/rtx8000_meta/summarize_probes.py \
  runs/abc-v-one-pair/probes.jsonl --transfer --aggregate
```

The transfer summary includes `vs_paired_c_change`, a paired comparison against
the ordinary A/B mean with the same transform, C examples and displacement norm.
Negative means the candidate beat the ordinary paired update on C. It also reports
sample standard deviations and C-hurt counts. This does not establish statistical
significance or long-run generalization. For more pairs, raise `--probe-pairs` to 8.
For another block, use `--probe-pair-offset 8` and a fresh run name; both A/B and C
offsets advance consistently. Online C batches advance across probe events and
may eventually wrap the validation loader; the starting epoch/batch is recorded.

### Short launch commands

From the cluster repo, pull the branch and run:

```bash
git pull --ff-only
bash research/rtx8000_meta/run_profile.sh muon 0 1
bash research/rtx8000_meta/run_profile.sh abc 0 1
```

Arguments are profile kind, visible GPU index, pair count, and optionally checkpoint
path. Defaults use `runs/audit-v-500/model.pt`. The launcher uses `.venv/bin/python`
when present, generates a unique run name, performs no optimizer update, and prints
the aggregate summaries and output path. Once the one-pair runtime/memory is known,
replace the final `1` with `8` to reproduce the original eight-pair sampling budget.
Each A, B and C half still has 16 sequences with these batch settings. On a shared
GPU node, use the device assigned to your session in the GPU argument.


### Joint QKV and larger A/B/C samples

`--mg-mode qkv` perturbs Q, K, and V projection matrices jointly, using one
normalized A-gradient direction with the requested **total** step norm across
all selected matrices. It does not include attention output projections, and
it does not change Muon/Adam parameter groups or stop any other weights training.
The profile supports the same `qkv` group alongside `qk`, `v`, and `mlp`.

The launcher now defaults to eight pairs and all four groups. First replicate
at the existing radii, then sweep temporary A-step radii on those same pairs:

```bash
bash research/rtx8000_meta/run_profile.sh abc 0 8
PROBE_RADII=0.025,0.05,0.1,0.25,0.5 bash research/rtx8000_meta/run_profile.sh abc 0 8
```

Confirm any promising setting on a different data block:

```bash
PROBE_PAIR_OFFSET=8 PROBE_RADII=0.025,0.05,0.1,0.25,0.5 bash research/rtx8000_meta/run_profile.sh abc 0 8
```

Environment overrides: `PROBE_MODES` (default `qk,v,qkv,mlp`), `PROBE_RADII`
(default `0.1,0.5`), `PROBE_PAIR_OFFSET` (default `0`), and
`PROBE_TRANSFER_RADIUS` (default `0.1`). These are distinct knobs: `PROBE_RADII`
sets temporary A adaptation, while `PROBE_TRANSFER_RADIUS` sets the candidate
update norm for C evaluation. Zero A adaptation is already represented by the
ordinary paired baseline, so it does not require a zero-radius trial. Checkpoint
profiles still start with fresh optimizer state. These runs measure transfer;
they are not training runs and do not establish a long-run improvement.
