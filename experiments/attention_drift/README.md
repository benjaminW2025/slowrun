# Attention drift during pretraining

Does attention routing settle early while held-out language modeling loss continues to improve? This standalone pilot measures how attention probability distributions change on fixed held-out inputs. See the [experiment protocol](https://chatgpt.com/library?file_id=libfile_bf281aa6eb7c81919dcbd7e54ab39f18&account_id=personal) for the interpretation and follow-up decisions.

Use Python 3.11+ with PyTorch and NumPy. Install Matplotlib to generate the plot. The trainer defaults to the existing Slowrun GPT-2-tokenized splits at `fineweb_data/fineweb_train.pt` and `fineweb_data/fineweb_val.pt` relative to the repo root. If those files live elsewhere, pass their actual paths with `--train` and `--val`.

From this directory, run:

```sh
python3 experiment.py train --out runs/pilot --batch-size 4
python3 experiment.py analyze --run runs/pilot
python3 plot.py --run runs/pilot
```

Use separate training and validation splits. Plain text files are read as byte IDs. For pretokenized one-dimensional integer NumPy `.npy` arrays, set `--vocab-size` to match the tokenizer. The default validation setup needs at least 16,385 tokens.

Training and analysis use CUDA when available, otherwise CPU. Set `--device` explicitly to override. The pilot defaults are a six-layer Transformer, context 256, 12,000 steps, and a 200-step warmup followed by constant learning rate. Training samples windows with replacement from a fixed data RNG independent of model initialization. The run directory must be new. Keep the original validation file for analysis, or pass its new location with `--val`. Checkpoints contain weights and metrics for analysis.

Analysis compares matching heads over 250 and 2,000 steps on identical held-out windows. It uses deterministic query positions across the latter three quarters of each window, including all valid keys for each query. `drift.csv` records mean and 95th-percentile total variation per head, plus attention mass on the first token and the eight-position window ending at the query. The latter two statistics can overlap. `training.csv` records the validation loss and learning rate at every saved checkpoint. `probe.json` records the exact windows and positions used. The plot shows individual heads colored by layer.

For a short smoke run, set `--steps 8 --save-every 2 --context 16 --width 32 --heads 4 --layers 2 --batch-size 2 --val-windows 4` on training and `--lags 2 4 --windows 4` on analysis. The pilot should inform the repeat runs' budget and plateau criterion. Keep all data and training arguments fixed when varying `--seed`.

## Layer 0 Q/K freeze

Run one unfrozen control through step 6,000 and save full training state at
steps 2,000 and 4,000:

```sh
python experiment.py train --out runs/freeze-control --batch-size 4 \
  --steps 6000 --fork-at 2000 4000 --device cuda
```

Once their checkpoints appear, the two freeze branches can run independently
on separate GPUs. Run the following commands in separate shells with your
environment active. Each branch consumes 2,000 more batches.

```sh
python experiment.py train --out runs/freeze-2k --batch-size 4 \
  --resume runs/freeze-control/step-00002000.pt --freeze-layer0-qk \
  --steps 4000 --device cuda

python experiment.py train --out runs/freeze-4k --batch-size 4 \
  --resume runs/freeze-control/step-00004000.pt --freeze-layer0-qk \
  --steps 6000 --device cuda
```

Each branch restores the model, AdamW state and batch sampler from the
control checkpoint. Use the same model and training arguments as the control,
including any explicit data paths. Only layer 0's query and key projection
weights are held fixed. Its value projection and the rest of the model still
train. This measures the effect on loss at matched token counts. The packed
projection still runs its backward pass, so this setup does not benchmark
backward-pass savings.

Compare the logged validation losses at step 4,000 for the early freeze and
step 6,000 for the later freeze, each against the control at the same step.
A positive frozen-minus-control difference is the loss penalty for freezing.

To compare the entire loss trajectory, run `python compare_freeze.py` after
the freeze runs. The output is CSV and includes the shared starting checkpoint
and every later step present in both the branch and the control. Use
`python compare_freeze.py > runs/freeze-comparison.csv` to save a copy.

## MLP depth at fixed linear compute

From the same initial seed, compare one 4× GELU MLP against two sequential 2×
GELU MLPs per block. Each narrow MLP has its own pre-norm and residual. The two
variants have identical initial attention and embedding weights and the same
total MLP linear weight count. The deep variant starts from the two halves of
the shallow variant's initialized linear weights. It adds a LayerNorm, a
residual addition, and an output bias vector per block, so exact FLOPs and
wall time can differ.

```sh
python experiment.py train --out runs/mlp-1 --mlp-depth 1 --batch-size 4 \
  --steps 6000 --device cuda
python experiment.py train --out runs/mlp-2 --mlp-depth 2 --batch-size 4 \
  --steps 6000 --device cuda
```

Compare the printed validation losses at step 6,000 and at earlier matched
steps. Both runs see the same data in the same order. Attention capacity is
fixed for this experiment.

## Fixed-norm classifier

Use `--classifier-norm 1` to normalize each final token vector after the
final LayerNorm, and each output classifier row, to unit L2 norm. Logits are
their raw dot products, with no learned scale. The classifier still shares
its underlying weights with the input embedding table. The input lookup
uses those weights directly, and row normalization applies on the classifier
side. With a shared norm of r, logits are r² times cosine similarity.
Omit the flag for the original head.

```sh
python experiment.py train --out runs/unit-head --classifier-norm 1 \
  --batch-size 4 --steps 6000 --device cuda
```

## Target cosine alignment

Compare two cosine heads with the same fixed classifier norm and identical
training settings. The second run adds 0.1 × (1 − target cosine) to the
training objective. The target cosine comes from the correct token's logit
divided by the squared classifier norm.

```sh
python -u experiment.py train --out runs/cos-ce --classifier-norm 1 \
  --batch-size 4 --steps 6000 --device cuda
python -u experiment.py train --out runs/cos-align --classifier-norm 1 \
  --alignment-weight 0.1 --batch-size 4 --steps 6000 --device cuda
```

Both runs report plain cross-entropy as `validation_loss`. For cosine heads,
check `validation_top1`, `validation_target_cosine`, and
`validation_within_token_cosine` at matched checkpoints. The last metric
averages the cosine of distinct pairs of validation states with the same
target token, giving each repeated target type equal weight. Its coverage
is reported as `validation_repeated_token_types`.

## Prefix freeze and time to validation loss

Resume all three arms from the same original, unfrozen step-4,000 checkpoint.
Use the same GPU model and comparable load for timing. If using one GPU, run
the commands sequentially. Keep any data path overrides from the original run.

```sh
resume=runs/freeze-control/step-00004000.pt
python -u experiment.py train --resume "$resume" --out runs/prefix/control \
  --batch-size 4 --steps 8000 --device cuda
python -u experiment.py train --resume "$resume" --out runs/prefix/embeddings \
  --freeze-prefix 0 --batch-size 4 --steps 8000 --device cuda
python -u experiment.py train --resume "$resume" --out runs/prefix/block0 \
  --freeze-prefix 1 --batch-size 4 --steps 8000 --device cuda
python compare_prefix.py --target 5.90
```

Both freeze arms freeze token and positional embeddings. In this pilot the
token matrix is tied to the output head, so both arms also freeze the
classifier weights. The classifier still backpropagates into its inputs.
The second freeze arm additionally freezes block 0. Its output has no
autograd history, so backward starts at the first trainable block.

The target of 5.90 nats is based on the earlier control crossing it between
steps 6,750 and 7,000. The comparison uses the first saved checkpoint at or
below that target. Training timings include batch loading and optimizer work,
and exclude validation, checkpoint writes, and setup. The final loss and
mean milliseconds per update use the full matched continuation.

## Prefix freeze with untied embeddings

First train a fresh unfrozen model with `--untie-embeddings` so all three
timed continuations start with the same independent input and output
weights and their own optimizer state. Both freeze arms leave the output
head trainable. Older tied checkpoints still work with the original commands.

```sh
python -u experiment.py train --untie-embeddings --out runs/untied-prefix/base \
  --batch-size 4 --steps 4000 --fork-at 4000 --device cuda
resume=runs/untied-prefix/base/step-00004000.pt
python -u experiment.py train --untie-embeddings --resume "$resume" \
  --out runs/untied-prefix/control --batch-size 4 --steps 8000 --device cuda
python -u experiment.py train --untie-embeddings --resume "$resume" \
  --out runs/untied-prefix/embeddings --freeze-prefix 0 \
  --batch-size 4 --steps 8000 --device cuda
python -u experiment.py train --untie-embeddings --resume "$resume" \
  --out runs/untied-prefix/block0 --freeze-prefix 1 \
  --batch-size 4 --steps 8000 --device cuda
python compare_prefix.py --runs runs/untied-prefix --target 5.90
```

### Freeze Q/K in the second block

Use the same untied step-4,000 checkpoint as the untied prefix control.
Compare block 0 and block 1 separately at matched steps against that control.

```sh
for block in 0 1; do
  python -u experiment.py train --untie-embeddings \
    --resume runs/untied-prefix/base/step-00004000.pt \
    --out "runs/untied-qk/block${block}" --freeze-qk-layers "$block" \
    --batch-size 4 --steps 8000 --device cuda || break
done
```

If both individual freezes have small penalties, use `--freeze-qk-layers 0 1`
with a new output directory to test the joint freeze. These runs measure loss.
The packed QKV projection still computes its full gradient.

### Benchmark Q/K freezing with weight-gradient savings

Run these sequentially on the same GPU. Both arms reuse the untied 4k
checkpoint and keep packed QKV weights. The frozen arm computes the input
gradient through Q/K/V and only the V rows of the weight-gradient matmul.
The packed AdamW update still runs and frozen rows are restored afterward.

```sh
python -u experiment.py train --untie-embeddings \
  --resume runs/untied-prefix/base/step-00004000.pt \
  --out runs/qk-speed/control --batch-size 4 --steps 4500 --device cuda
python -u experiment.py train --untie-embeddings \
  --resume runs/untied-prefix/base/step-00004000.pt \
  --out runs/qk-speed/both --freeze-qk-layers 0 1 --skip-frozen-qk-grad \
  --batch-size 4 --steps 4500 --device cuda
```

Subtract each arm's cumulative `training_seconds` at step 4,250 from its
value at 4,500 to exclude the first 250 steps from the timing. Repeat in
reverse order with new output directories if the timing difference is small.
