# MG mechanism study on one GPU (proxy)

The tiny-track record (`tiny/train.py`) needs 8xH100: FlashAttention-3, fp8 kernels and 8 ranks.
`mg_study/proxy.py` is a plain-PyTorch reimplementation for one GPU (written for a 48GB L40) that keeps
the parts the mechanism question depends on:

| | record (tiny track) | proxy |
|---|---|---|
| model | 300M GPT, FA3, fp8 MTP, recurrence | ~50M GPT (8 layers, d=512), torch SDPA, bf16 |
| data | 100M tokens, 15 epochs | first 20M tokens of the same file, 15 epochs (data-limited) |
| optimizer | Muon + AdamW | Muon (Newton-Schulz) on block matrices + AdamW on embedding/head |
| MG step | per rank: half A adapts (L2 step 0.5 on MLP matrices), half B evaluated there, undo | the same, per emulated rank (`--mg-groups 8`) |

## Commands (from the repo root, on the GPU machine)

```bash
bash mg_study/run_proxy.sh test      # 1. correctness tests on the GPU (must print ALL TESTS PASSED)
bash mg_study/run_proxy.sh smoke     # 2. 60 steps: tok/s, memory, ETA
bash mg_study/run_proxy.sh step0     # 3. off vs descent, seed 42: does MG help in the proxy?
bash mg_study/run_proxy.sh day1      # 4. off / descent / random x 3 seeds
bash mg_study/run_proxy.sh summary
```

Data: `fineweb_data/fineweb_train.pt` and `fineweb_val.pt` from `python prepare_data.py`.

Outputs: `runs_proxy/<arm>_s<seed>/{log.txt,result.json}`. `result.json` has final/best val loss, a
clean train loss (eval mode, unperturbed weights, first 2M training tokens), the eval history and
every logged cosine.

## What the arms mean

- `off`: ordinary gradient (same grouped loop, no temporary step; tested equal to the full-batch gradient)
- `descent`: the record: step theta - eta * g_A/|g_A|
- `ascent`: step theta + eta * g_A/|g_A| (SAM-like)
- `random`: fair coin per (step, group): cancels the O(eta) lookahead term in expectation

`[mg-snr]` lines log cos(g_A(theta), g_B(theta')). Averaging the descent-step and ascent-step cosines of
the random arm estimates the gradient SNR; half their difference estimates the curvature along v.

## Known differences from the record

No dropout by default (`--dropout`), no SWA/EMA averaging, no recurrence, MTP or value embeddings, a simple
LR schedule (warmup, constant, linear warmdown). The step norm 0.5 is absolute, as in the record; the
log prints it relative to |theta_P| for comparison.
