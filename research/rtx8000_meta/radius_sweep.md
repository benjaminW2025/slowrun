# Checkpoint perturbation-radius sweep

Question: does a fixed Euclidean adaptation radius produce consistent behavioral
changes across A/B pairs? Measure how adaptation affects A loss, B loss, B
prediction KL, and B's gradient on the selected MLP matrices.

## Run

Use a completed training checkpoint. The launcher reads its `result.json` to
recover architecture, precision, batch size and episode size. No model-size
assumptions are hardcoded. Start with 10 pairs to measure runtime:

```bash
python research/rtx8000_meta/run_radius_sweep.py \
  --checkpoint runs/YOUR-RUN/model.pt --pairs 10 --gpu 1 --execute
```

Then run the complete, same-pair sweep:

```bash
python research/rtx8000_meta/run_radius_sweep.py \
  --checkpoint runs/YOUR-RUN/model.pt --pairs 100 --gpu 1 --execute
```

The 100-pair run starts at pair zero, so it includes the initial 10 pairs.
Use `--offset 10 --pairs 90` instead for disjoint remaining pairs. Don't treat
repeated pairs from separate runs as additional independent samples.

For an intermediate epoch checkpoint, supply `--config path/to/result.json`
from its source run if automatic discovery fails, and choose `--iterations 1`
or `--iterations 2` to match the intended stage. The default is the source
run's maximum recurrence. The saved final `model.pt` may contain EMA or averaged
weights; the manifest includes the source result's variant. This does not imply
intermediate checkpoints share that variant.

Omit `--execute` to preview the command. Override radii with
`--radii 0,0.03,0.1,0.3,0.5,1.0` (the default). Output lives in a new
`runs/radius-sweep-*` directory. No optimizer update or replacement model save
occurs. FP16 runs retain their source loss scaling; BF16 runs don't use scaling.

## Measurement definition

- The training loader supplies deterministic shuffled sequences, grouped into
  disjoint consecutive A/B pairs. A and B have equal token counts. Same config,
  data, seed and offset reuse the same pairs across checkpoints. Pair hashes
  allow this to be checked. Different sequences can share source documents.
- One pair at a time is retained on the GPU. Every radius reuses that pair.
- Model parameters are restored exactly before each perturbation and on failure.
  RNG, dropout modes and existing `.grad` tensors are preserved. Dropout is off
  for all measurements; the training recurrence/truncated-backprop policy stays
  on. Therefore these gradients need not be derivatives of the full recurrent
  inference loss. The sweep is not full-backprop analysis.
- The A gradient is normalized jointly across selected Muon-managed MLP matrices.
  Zero is the baseline. A zero gradient produces a recorded no-op, not a fabricated
  direction. Nonfinite gradients or metrics abort the measurement.
- KL is full-vocabulary `KL(p_original || p_adapted)` on B, averaged over tokens,
  in nats. It is not the probability change of the observed target alone.
  Original logits are recomputed for each radius to avoid large CPU caches.
  At most one original/adapted microbatch-logit pair is retained; softmax/KL work
  is chunked in blocks of 128 tokens. Tiny negative KL values from roundoff
  are retained rather than silently clipped.
- Relative B-gradient change is `||g_B' - g_B|| / ||g_B||`, restricted to selected
  MLP parameters. Undefined zero-denominator ratios are null.
- Each pair is flushed to JSONL after all its radii complete. Partial runs retain
  completed pairs. The manifest records checkpoint/data hashes and configuration.

## Results

- `radius_pairs.jsonl` and `radius_pairs.csv`: per-pair, per-radius measurements.
- `radius_summary.csv`: means, medians, 10th/90th percentiles, defined counts,
  and fraction of pairs improving B.
- `radius_report.html`: scatter plots including ΔB loss versus B prediction KL,
  colored by radius, with hover labels; opens in any browser, no dependencies.
- `radius_manifest.json`: configuration, source-run result and provenance.

For a partial run, regenerate reports from completed pairs:

```bash
python research/rtx8000_meta/summarize_radius_sweep.py runs/YOUR-SWEEP/radius_pairs.jsonl
```

Look for whether comparable KL changes yield comparable B-loss responses across
pairs/radii, and whether the relationship changes at early versus late checkpoints.
The scatter is descriptive: correlation does not establish that KL determines
transfer, and this probe does not evaluate the final Muon update or generalization
on independent C examples. No tuning or significance claim follows from this alone.
