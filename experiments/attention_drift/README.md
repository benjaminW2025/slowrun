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
