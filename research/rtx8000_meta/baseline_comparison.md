# Matched baseline versus MLP meta-gradient

Goal: establish whether the corrected local trainer runs reliably and whether
MLP adaptation improves held-out loss over an otherwise identical control.
No new tuning or diagnostic interventions in this comparison.

## Scope

`off` is ordinary gradient accumulation across both A/B halves at the original
weights. `mlp` temporarily adapts the Muon-managed MLP matrices by normalized
raw-gradient descent with radius 0.5 before evaluating B. Both use one-way
episodes, the same batches, initialization seed, dropout, decay, optimizer,
schedule, validation budget, and averaging procedure. Each starts a fresh process.
The primary metric is `raw_final_val_loss`; the trainer's final averaged-model
loss and model variant are reported separately. Keep averaging results: the
upstream PR reports that averaging contributes to its benefit.

This is the RTX 8000 adaptation of tiny entry 12, not an exact reproduction of
PR #99 on entry 14. Differences include the model baseline, precision, serial
virtual ranks, physical microbatch size and screening schedule. Passing local
CPU tests does not establish CUDA correctness or reproduction of the PR score.

## Run on the GPU machine

From the repository root, activate the existing training environment, then:

```bash
git pull --ff-only fork codex/experiments
python -m unittest discover -s research/rtx8000_meta -p 'test_*.py'
python research/rtx8000_meta/compare_baseline.py --stage smoke --gpu 0 --execute
python research/rtx8000_meta/compare_baseline.py --stage pilot --gpu 0 --seed 42 --execute
```

Use the remote name pointing to `benjaminW2025/slowrun` (often `origin` on
the GPU machine) instead of `fork` if needed. Dependencies are those of the
existing trainer; `fineweb_data/fineweb_train.pt` and `fineweb_val.pt` must exist.
The launcher preflights imports, CUDA and data paths before training.
Omit `--execute` to inspect commands without training. The interpreter used
to launch the runner is used for both training processes.

The smoke runs three updates in **constant two-pass recurrence** to exercise
the corrected recurrent backward immediately. Stop on missing/nonfinite gradients,
exceptions or OOM; smoke losses are not evidence about generalization.

The pilot runs 500 updates with the same compressed 500-step schedule and
262,144 requested validation tokens per arm. Probes are disabled. Both arms
run sequentially on the selected GPU to avoid competing for its resources.
Inspect each run's `terminal.log`, including its recurrence transition and loss
trajectory. A completed run alone does not establish numerical correctness.

## Interpretation and next decision

- Compare raw final loss first, and averaged loss only alongside its model variant.
- Check matching step counts and configurations in the saved comparison JSON.
- Record process wall time as end-to-end cost; it includes initialization,
  compilation, evaluation and saving, not just training.
- A single-seed pilot is screening evidence, not confirmation. If stable and
  promising, repeat paired seeds 43 and 44 before selecting a new variant.
  Alternate arm execution order in subsequent controlled timing work if needed.
- The pilot processes about 262M training-token exposures (~2.6 passes), not
  the original 15-pass run. Do not reject the full method on a short-run result.
  A full-length matched comparison is required before a final efficacy decision;
  matching the original PR also requires its newer model and hardware recipe.
- Do not select hyperparameters using these validation outcomes. Set any
  practical improvement threshold before the confirmation runs, based on pilot
  variability and affordable compute, rather than inventing one afterwards.

Each comparison saves commands, Git revision, tracked source diff, dataset
SHA-256 hashes, per-arm results and process times under `runs/baseline-*`.
The trainer saves its usual logs, configuration and checkpoint under the
corresponding `-off` and `-mlp` run directories. Existing runs are not overwritten.
