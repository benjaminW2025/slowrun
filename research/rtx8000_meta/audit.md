# Upstream implementation audit — 2026-10-09

## Sources and scope

- Model baseline: Tiny entry 12, [`a2c443e18f99d0f6e35c6013c752e636fe402213`](https://github.com/qlabs-eng/slowrun/blob/a2c443e18f99d0f6e35c6013c752e636fe402213/tiny/train.py).
- Meta-gradient reference: merged [PR #99](https://github.com/qlabs-eng/slowrun/pull/99), commit `45bf3fd`.
- PR #99's model baseline was entry 14 (`d1fbdb7`), which includes MTP, FP8 and full-layer XSA changes. This experiment deliberately uses entry 12 for portability. It tests the meta-gradient rule on that model; it does not reproduce the complete PR #99 recipe or its leaderboard score.
- Audit tests read upstream source directly from the local Git object database. A shallow clone must fetch the historical commits before running these tests.

## Verified against original source

AST comparison, ignoring comments and formatting, finds only three changed definitions: `GPT` (embedding/rotary precision), `muon_step_fused` (BF16 to FP32 polar iteration), and `DistMuonAdamW` (local execution path).

Unchanged definitions include attention, MLP, blocks, recurrence helpers, data loader, validation, AdamW math, LR and momentum schedule helpers. Within GPT, initialization equations, forward computation, recurrence, optimizer setup and all methods other than the two precision adaptations are unchanged. Existing CLI defaults are unchanged except physical device batch size. Polar Express coefficients match both upstream sources. Optimizer grouping in entry 12 also matches PR #99.

Numerical checks:

1. Execute original distributed optimizer with world-size-one identity collectives and compare to the local path. Use common FP32 polar precision to isolate routing from the documented dtype adaptation. Four updates, square/tall/wide matrices, large and small Adam tensors, intermittent missing gradients: parameters and optimizer buffers match exactly.
2. Execute original and adapted small transformers in common FP32 precision, dropout disabled. Compare losses, gradients and updated parameters over one- and two-pass recurrence: exact equality.
3. Execute PR #99's actual `mg_temporary_step` in the reference A/B loop. Compare serial virtual-rank gradients with one/two selected matrices, no selected matrices, and multiple microbatches per half: agree within FP32 rounding tolerance.
4. Existing tests check matrix selection, scale cancellation, exact restoration, zero gradients, nonfinite gradients, and FP16 large-vocabulary gradient retention.

## Meta-gradient semantics

Each virtual rank starts from the same original parameters. Compute A's gradient, temporarily move the selected matrices along its raw normalized direction with total Euclidean radius 0.5, evaluate B there, restore parameters, and accumulate A plus B gradients. Average equally across halves and ranks, then apply one Muon/AdamW optimizer step. No differentiation through A's temporary step occurs: this is first-order, matching the PR.

MLP selection includes gate, expansion and output matrices. QK selects query and key projections together; V selects value projections. The final optimizer grouping is unchanged in every mode. Other parameters still receive A/B gradients; they are not frozen.

## Confirmed problems and fixes

- **FP16 underflow (fixed in preceding commit `0b0d1ee`):** dividing each loss by 256 before backward erased non-target softmax gradients at the initial scale. Backpropagate scaled microbatch mean losses, accumulate in FP32, and divide the accumulated gradients afterward. The regression reproduces the failure in the old code and passes with the correction. This is a demonstrated numerical defect; the corrected full GPU learning curve remains unverified.
- **EMA update count:** original final correction used `step // update_interval`, although updates occurred at step indices 0, 10, 20, etc. Count actual updates. Test run lengths 1, 5, 10, 11, 20, 21 and 500 against an analytical constant-parameter reference.
- **Checkpoint averaging metrics:** averaging overwrote the model without updating `val_loss`, which could retain the EMA score. Update the metrics to describe the saved weights, and record `model_variant` and `model_path` in results. Preserve `raw_final_val_loss` separately for experimental comparisons.
- **Short-run checkpoint eligibility:** under a compressed 500-step schedule, averaging previously included a checkpoint around step 190 even though SWA started near 367. Collect averaging checkpoints only inside the scheduled SWA window. A short run with fewer than two eligible checkpoints skips checkpoint averaging.
- Earlier process-group and zero-initialized QKV-gradient fixes remain covered by tests.

## Recurrent mixed-precision correction — 2026-10-10

The outer autocast context previously cached FP16 weight casts created during
an early recurrent pass under `no_grad`. Reusing those detached casts in the
subsequent differentiable pass disconnected QK, V and MLP matrices from the
loss. The profiler silently replaced `None` gradients with zeros. A small
version of the actual model reproduces missing gradients for every selected
matrix with CPU BF16 autocast; disabling the cache restores nonzero A, B and
adapted-B gradients. Production CUDA FP16 autocast now sets
`cache_enabled=False` for both training and profiling. Diagnostics reject
missing gradients instead of reporting them as zeros. Legitimate tensor-valued
zero gradients at initialization remain permitted.

This affects mixed-precision training whenever recurrence uses a no-grad pass
followed by a differentiable pass (e.g. the last 150 steps of the default
500-step, 1x-to-2x schedule). Earlier single-pass updates are unaffected by this
specific bug. The old 500-step arm scores must be rerun before interpreting
them as evidence about correctly trained two-pass models. Existing checkpoints
can still be profiled with the fix, but the fix cannot repair their past updates.
CPU regressions cover both `.backward()` and the profiler's `autograd.grad`;
actual CUDA FP16 execution still needs a GPU smoke test.

## Additional training-path audit — 2026-10-10

- Every optimizer update now checks all optimizer-managed tensors for missing or
  nonfinite gradients before changing parameters. Disabled XSA parameters are
  explicitly outside that set. Tensor-valued zero gradients remain allowed;
  disconnected (`None`) gradients do not. The normalized A trial also rejects
  missing gradients, rather than substituting zeros.
- Added complete-script CPU fixtures for all four modes across a one-pass to
  two-pass transition, including loader consumption, schedules, EMA, exported
  metrics and saved finite weights. Fixtures reduce sequence length and vocabulary
  and disable compilation; they execute the actual training loop. A standalone
  two-pair, three-group profiling fixture verifies no optimizer update/model save.
- Added CPU FP16 tests with half-precision rotary buffers and a real GradScaler
  for all four modes across one- and two-pass updates. These complement the
  BF16 cache regression; they do not establish CUDA kernel correctness.
- A B-forward exception must restore temporarily perturbed A weights exactly.
- Fixed two reporting defects: `--save-result`/`--output_json` previously had no
  effect, and checkpoint averaging selected the best metric by loss rather than
  BPB, unlike the other evaluations. These changes do not change training updates.

## Intentional differences and remaining limits

- RTX path uses FP16 autocast, FP32 parameters/optimizer state, FP32 Muon polar iteration, and FP16 rotary buffers. Original uses BF16 and stores embeddings in BF16. Equality tests at common precision do not establish equality across these actual precision regimes. RMS normalization's default epsilon also depends on input dtype.
- Serial microbatching changes dropout random-number consumption. Loader source is identical, but a physical batch of one changes discarded epoch-tail sequences; with `--no-doc-shuffle`, it also changes the unit of epoch batch shuffling. This is not a bitwise replica of eight-GPU data order. All experimental arms must use the same configuration.
- The PR clamps gradient norm at 1e-12; the adaptation skips exactly zero directions and normalizes nonzero directions. Extremely tiny gradients below that floor are an edge-case difference.
- `--schedule-steps 500` compresses LR, WD, recurrence and SWA schedule positions. It does not create 15 actual dataset epochs. Muon momentum retains the original 300-step ramp. This is a screening configuration, not the original full-length training recipe.
- `--no-compile` disables model compilation; optimizer functions still use `torch.compile`. CPU audit tests remove those decorators to isolate their math. CUDA compilation, CUDA FP16 attention/backward, and full-size GPU training are not validated by these tests.
- Compare `raw_final_val_loss` across arms. `val_loss` describes `model.pt`, which may be raw, EMA or checkpoint average as recorded by `model_variant`. `best_val_loss` may refer to a different evaluated model.

## Run the checks

```bash
python -m unittest discover -s research/rtx8000_meta -p 'test_*.py' -v
```

Before another long sweep, run the corrected `off` control for 20 steps with the same 500-step schedule and inspect raw training loss and final validation. A decreasing small CPU toy loss and passing algebraic tests cannot substitute for that GPU check. Existing runs made before the underflow correction should not be used as evidence for or against meta-gradients.
