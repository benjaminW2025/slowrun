# Default: 101M parameters on one L40

The default is now `--profile medium`: **101,059,795 parameters**, verified
by instantiating the actual GPT architecture with its padded GPT-2 vocabulary.
Settings: 14 layers, width 512, 8 heads, context 2,048, physical batch four,
one rank-local A/B episode, and 32,768 tokens per optimizer update.
This is four forward/backward microbatches per update (two A, two B), rather
than the original 256. MLP adaptation norm is 0.5, with matching settings
between the control and meta-gradient arms. This radius is not known to be
optimal at this scale.

The full existing 100M-token training file is retained; no subset is generated
for this profile. Pilot: 500 updates per arm and 65,536 validation tokens.
Each arm sees 16,384,000 training-token exposures, less than one dataset pass.
This pilot compares early learning, not repeated-data generalization. A
longer repeated-data comparison must follow before making claims about that
regime, and success at 101M parameters does not establish larger-scale gains.

```bash
git pull --ff-only origin codex/experiments
python research/rtx8000_meta/compare_baseline.py --profile medium --stage benchmark --gpu 1 --execute
```

The benchmark runs eight updates per arm in two-pass recurrence, prints
measured timings and a 500-update training-time estimate, then stops.
Only after inspecting the timings, launch:

```bash
python research/rtx8000_meta/compare_baseline.py --profile medium --stage pilot --gpu 1 --execute
```

Runtime on the L40 is unmeasured. Initialization, optimizer compilation,
evaluation, and shared-filesystem writes can add substantial overhead.
Benchmark and smoke exercise two-pass recurrence immediately; the pilot
retains the original late 1x-to-2x transition. Both arms are fresh processes
and execute sequentially on GPU 1. Output: `runs/baseline-medium-*`.

---

# Small comparison on one L40

The former small profile is available with `--profile small`. Full-size commands
remain available only with explicit `--profile original`.

From your GPU checkout after activating the training environment:

```bash
git pull --ff-only origin codex/experiments
python research/rtx8000_meta/compare_baseline.py --profile small --stage benchmark --gpu 1 --execute
# Inspect the measured timing and estimate before running:
python research/rtx8000_meta/compare_baseline.py --profile small --stage pilot --gpu 1 --execute
```

The benchmark runs 8 updates per arm with two recurrent passes throughout,
reports median step times after the first step, and estimates training time.
It never starts the pilot automatically. Compilation, evaluation, checkpoint
writes and startup add time; runtime on the L40 has not been measured locally.
A 10–20 minute pair is a target, not a measured promise.

The small profile uses four layers, width 256, four heads, context 512,
physical batch four, one A/B episode, and 8,192 tokens per optimizer update:
**four microbatches instead of 256**. It retains GPT-2 vocabulary/tokenization.
The pilot runs 200 updates per arm on a fixed 262,144-token training prefix
(about six passes), and evaluates 8,192 held-out tokens. The final training
document may be truncated, and chunk/tail exclusions slightly reduce the
actual corpus used. The prefix and held-out file are identical across arms.

Temporary MLP adaptation radius is fixed at 0.1 for this smaller model; this
is an unvalidated starting setting, not a rescaled equivalent of PR #99's 0.5.
Use this comparison to check training and study a specific small-scale setting.
A null result cannot reject meta-gradient at other radii or larger scales.
The very small validation sample and one seed make results exploratory.
All other settings match between arms. The pilot includes a 1x-to-2x recurrence
transition; benchmark and smoke exercise two-pass recurrence immediately.

Results and source/data provenance are saved under `runs/baseline-small-*`.
`comparison.json` contains both arm results; `terminal.log` is in each arm's
own directory. Raw final validation loss is primary; averaged loss is separate.

---

## Original full-size profile reference

The remainder describes `--profile original` explicitly, not the new default.

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
python research/rtx8000_meta/compare_baseline.py --profile original --stage smoke --gpu 0 --execute
python research/rtx8000_meta/compare_baseline.py --profile original --stage pilot --gpu 0 --seed 42 --execute
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
