#!/usr/bin/env python3
"""Metric evaluation for the retina surgery dataset using MoGe outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

from moge.model import import_model_class_by_version
from moge.test.retina_metrics import compute_depth_metrics, average_metrics
from moge.utils.io import read_image
from moge.utils.retina_dataset import scan_retina_records, load_depth_map, read_binary_mask


def _load_model(
    config: Dict,
    checkpoint: Optional[Path],
    pretrained: Optional[str],
    device: torch.device,
    fp16: bool,
) -> torch.nn.Module:
    MoGeModel = import_model_class_by_version(config["model_version"])
    if checkpoint is not None:
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        state_dict = state.get("model", state)
        model = MoGeModel(**config["model"])
        model.load_state_dict(state_dict, strict=False)
    else:
        repo_or_path = pretrained or "Ruicheng/moge-2-vitl-normal"
        model = MoGeModel.from_pretrained(repo_or_path)
    model.to(device)
    model.eval()
    if fp16:
        model.half()
    return model


def _prepare_tensor(image: np.ndarray, device: torch.device, fp16: bool) -> torch.Tensor:
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).to(device)
    if fp16:
        tensor = tensor.half()
    return tensor


def _to_bool_mask(array: Optional[np.ndarray], reference_shape) -> torch.Tensor:
    if array is None:
        return torch.ones(reference_shape, dtype=torch.bool)
    if array.ndim == 3:
        array = array[..., 0]
    return torch.from_numpy((array > 0.5).astype(np.bool_))


def run_evaluation(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())

    retina_datasets = [ds for ds in config["data"]["datasets"] if ds.get("type") == "retina_surgery"]
    if not retina_datasets:
        raise RuntimeError("No retina_surgery dataset entry found in config")
    dataset_cfg = dict(retina_datasets[0])
    dataset_cfg["path"] = str(Path(args.dataset_root))
    if args.split:
        dataset_cfg["split"] = args.split

    device = torch.device(args.device)
    model = _load_model(config, args.checkpoint, args.pretrained, device, args.fp16)

    filenames, records = scan_retina_records(dataset_cfg)
    if args.limit is not None:
        filenames = filenames[: args.limit]

    per_sample = []
    torch.set_grad_enabled(False)

    for key in tqdm(filenames, desc="Retina evaluation"):
        record = records[key]
        image = read_image(record["image_path"])
        depth_gt = load_depth_map(record["depth_path"])
        valid_mask_np = read_binary_mask(record.get("valid_mask_path"))
        instrument_mask_np = read_binary_mask(record.get("instrument_mask_path"))

        if valid_mask_np is None:
            valid_mask_np = np.ones_like(depth_gt, dtype=np.float32)
        depth_gt = np.where(valid_mask_np > 0.5, depth_gt, np.nan)

        image_tensor = _prepare_tensor(image, device, args.fp16)
        output = model.infer(
            image_tensor,
            num_tokens=args.num_tokens,
            resolution_level=args.resolution_level,
            use_fp16=args.fp16,
        )
        pred_depth = output.get("depth")
        if pred_depth is None:
            raise RuntimeError("Model inference did not return depth predictions.")
        pred_depth = pred_depth.detach().cpu()
        if pred_depth.ndim == 3:
            pred_depth = pred_depth.squeeze(0)

        depth_gt_tensor = torch.from_numpy(depth_gt.astype(np.float32))
        valid_mask = _to_bool_mask(valid_mask_np, depth_gt_tensor.shape)
        instrument_mask = _to_bool_mask(instrument_mask_np, depth_gt_tensor.shape) if instrument_mask_np is not None else None

        overall_metrics = compute_depth_metrics(pred_depth, depth_gt_tensor, valid_mask)
        instrument_metrics = None
        if instrument_mask is not None:
            combined = valid_mask & instrument_mask
            if combined.any():
                instrument_metrics = compute_depth_metrics(pred_depth, depth_gt_tensor, combined)

        per_sample.append({
            "key": key,
            "overall": overall_metrics,
            "instrument": instrument_metrics,
        })

    aggregated = {
        "overall": average_metrics([item["overall"] for item in per_sample]),
        "instrument": average_metrics([
            item["instrument"] for item in per_sample if item["instrument"] is not None
        ]) if any(item["instrument"] is not None for item in per_sample) else {},
    }

    print("Overall metrics:")
    for metric, value in sorted(aggregated["overall"].items()):
        print(f"  {metric:>10s}: {value:.6f}")
    if aggregated["instrument"]:
        print("Instrument-region metrics:")
        for metric, value in sorted(aggregated["instrument"].items()):
            print(f"  {metric:>10s}: {value:.6f}")
    else:
        print("Instrument-region metrics: N/A")

    if args.output:
        payload = {
            "aggregated": aggregated,
            "per_sample": per_sample,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=4))
        print(f"Saved metrics to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate MoGe on the retina dataset")
    parser.add_argument("--config", default="configs/train/retina_surgery.json", help="Finetune config path")
    parser.add_argument("--dataset-root", default="../final_version_processed", help="Root directory of ../final_version_processed")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint (.pt) to evaluate")
    parser.add_argument("--pretrained", default=None, help="Optional pretrained repo/path if no checkpoint")
    parser.add_argument("--device", default="cuda", help="Evaluation device")
    parser.add_argument("--fp16", action="store_true", help="Run inference in half precision")
    parser.add_argument("--num-tokens", type=int, default=None, help="Override ViT tokens")
    parser.add_argument("--resolution-level", type=int, default=9, help="Resolution level used by model.infer")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for quick checks")
    parser.add_argument("--output", default=None, help="Optional JSON file to store metrics")
    parser.add_argument("--split", default=None, help="Optional split file (e.g. test.txt) relative to dataset root")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
