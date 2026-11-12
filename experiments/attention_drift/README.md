# Attention drift during pretraining

Does attention routing settle early while held-out language modeling loss continues to improve? This standalone pilot measures how attention probability distributions change on fixed held-out inputs. See the [experiment protocol](https://chatgpt.com/library?file_id=libfile_bf281aa6eb7c81919dcbd7e54ab39f18&account_id=personal) for the interpretation and follow-up decisions.

Use Python 3.11+ with PyTorch and NumPy. Install Matplotlib to generate the plot. Supply separate, non-overlapping train and validation corpora. Plain files are read as byte IDs for a self-contained pilot. For a subword experiment, supply two one-dimensional integer NumPy `.npy` arrays from the same tokenizer and pass `--vocab-size` to training. The default validation setup needs at least 16,385 tokens.

From this directory, run:

```sh
python3 experiment.py train --train /path/to/train.txt --val /path/to/val.txt \
  --out runs/pilot --seed 0
python3 experiment.py analyze --run runs/pilot
python3 plot.py --run runs/pilot
```

Training and analysis use CUDA when available, otherwise CPU. Set `--device` explicitly to override. The pilot defaults are a six-layer Transformer, context 256, 12,000 steps, and a 200-step warmup followed by constant learning rate. Training samples windows with replacement from a fixed data RNG independent of model initialization. The run directory must be new. Keep the original validation file for analysis, or pass its new location with `--val`. Checkpoints contain weights and metrics for analysis.

Analysis compares matching heads over 250 and 2,000 steps on identical held-out windows. It uses deterministic query positions across the latter three quarters of each window, including all valid keys for each query. `drift.csv` records mean and 95th-percentile total variation per head, plus attention mass on the first token and the eight-position window ending at the query. The latter two statistics can overlap. `training.csv` records the validation loss and learning rate at every saved checkpoint. `probe.json` records the exact windows and positions used. The plot shows individual heads colored by layer.

For a short smoke run, set `--steps 8 --save-every 2 --context 16 --width 32 --heads 4 --layers 2 --batch-size 2 --val-windows 4` on training and `--lags 2 4 --windows 4` on analysis. The pilot should inform the repeat runs' budget and plateau criterion. Keep all data and training arguments fixed when varying `--seed`.
