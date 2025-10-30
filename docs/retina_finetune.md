# Retina Surgery Fine-tuning Guide

This guide summarizes the steps required to fine-tune MoGE on the retina surgery stereo dataset and evaluate the requested depth metrics.

## Dataset Layout

The data loader expects the following directory structure (identical to the provided dataset):

```
../final_version_processed/
  s1_processed/
    instrincs.json
    left/
      imgs/*.png
      metric_depth/*.npy
      valid_region_mask/*.png
      instrument_mask/*.png
    right/ ... (ignored)
  s2_processed/
  ...
```

Only the left camera stream is used. Depth supervision comes from the `metric_depth/*.npy` files. Pixels outside `valid_region_mask` are discarded automatically, and the `instrument_mask` is kept for evaluation.

## Training

A dedicated training config is available at `configs/train/retina_surgery.json`. It wires the new retina-aware loader introduced in `moge.train.dataloader`. Example command:

```bash
python -m moge.scripts.train \
  --config configs/train/retina_surgery.json \
  --workspace workspace/retina_run \
  --batch_size_forward 4 \
  --gradient_accumulation_steps 2 \
  --num_iterations 100000 \
  --checkpoint path/to/pretrained.pt  # optional warm start
```

Adjust the batch size, accumulation and iteration count to fit your GPU budget. The loader assumes the dataset root lives at `../final_version_processed`; tweak the config if the path differs on your server.

## Evaluation & Metrics

Use the new script `moge/scripts/eval_retina.py` to report the requested metrics with mask handling:

```bash
python -m moge.scripts.eval_retina \
  --config configs/train/retina_surgery.json \
  --checkpoint workspace/retina_run/checkpoint/00010000.pt \
  --output retina_metrics.json
```

The script computes the following statistics over all valid pixels (white area in `valid_region_mask`):

- `d0.5`, `d1`, `d2`, `d3` (δ-threshold accuracies with δ ∈ {1.25^{0.5}, 1.25, 1.25², 1.25³})
- `abs_rel`, `sq_rel`, `rmse`, `rmse_log10`, `log10`, `silog`

Each metric is also reported on the intersection between `valid_region_mask` and `instrument_mask`, enabling targeted assessment of instrument regions. Results are printed to stdout and optionally serialized to the JSON file supplied via `--output`.

Use `--pretrained` instead of `--checkpoint` if you want to evaluate an unmodified pretrained model from Hugging Face. Additional options (`--limit`, `--num_tokens`, `--resolution_level`, `--device`) are documented by `python -m moge.scripts.eval_retina --help`.
