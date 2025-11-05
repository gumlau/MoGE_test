#!/usr/bin/env python3
"""Batch inference helper tailored for the retina surgery dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

from moge.model import import_model_class_by_version
from moge.utils.io import read_image
from moge.utils.vis import colorize_depth
from moge.utils.retina_dataset import scan_retina_records


def _find_default_split(dataset_root: Path) -> Optional[str]:
    candidates = [
        'val.txt',
        'split/val.txt',
        'validation.txt',
        'validation_split.txt',
        'test.txt',
        'split/test.txt',
        'train.txt',
    ]
    for candidate in candidates:
        if (dataset_root / candidate).exists():
            return candidate
    return None


def _load_model(
    config_path: Path,
    checkpoint: Optional[Path],
    pretrained: Optional[str],
    device: torch.device,
    fp16: bool,
) -> torch.nn.Module:
    config = json.loads(config_path.read_text())
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


def run_inference(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    config = json.loads(config_path.read_text())
    retina_datasets = [ds for ds in config["data"]["datasets"] if ds.get("type") == "retina_surgery"]
    if not retina_datasets:
        raise RuntimeError("No retina_surgery dataset entry found in config")
    dataset_cfg = dict(retina_datasets[0])
    dataset_cfg["path"] = str(dataset_root)
    split_file = args.split or _find_default_split(dataset_root)
    if split_file:
        dataset_cfg["split"] = split_file

    device = torch.device(args.device)
    model = _load_model(config_path, args.checkpoint, args.pretrained, device, args.fp16)

    filenames, records = scan_retina_records(dataset_cfg)
    if args.limit is not None:
        filenames = filenames[: args.limit]

    torch.set_grad_enabled(False)
    for key in tqdm(filenames, desc="Retina inference"):
        record = records[key]
        image = read_image(record["image_path"])
        image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).to(device)
        if args.fp16:
            image_tensor = image_tensor.half()

        output = model.infer(
            image_tensor,
            num_tokens=args.num_tokens,
            resolution_level=args.resolution_level,
            use_fp16=args.fp16,
        )

        depth = output["depth"].detach().cpu().numpy()
        if depth.ndim == 3:
            depth = depth.squeeze(0)
        mask = output.get("mask")
        if mask is not None:
            mask = mask.detach().cpu().numpy()
            if mask.ndim == 3:
                mask = mask.squeeze(0)

        save_dir = output_root / record["sequence"] / Path(record["frame_name"]).stem
        save_dir.mkdir(parents=True, exist_ok=True)

        np.save(save_dir / "depth.npy", depth.astype(np.float32))
        depth_vis = colorize_depth(depth)
        cv2.imwrite(str(save_dir / "depth_vis.png"), cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR))
        if mask is not None:
            cv2.imwrite(str(save_dir / "mask.png"), (mask > 0.5).astype(np.uint8) * 255)

    print(f"Saved predictions to {output_root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MoGe inference on the retina dataset")
    parser.add_argument("--config", default="configs/train/retina_surgery.json", help="Training config file used for retina setup")
    parser.add_argument("--dataset-root", default="../final_version_processed", help="Root directory of ../final_version_processed on the server")
    parser.add_argument("--output-dir", default="retina_predictions", help="Folder to write predictions")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to fine-tuned checkpoint (.pt)")
    parser.add_argument("--pretrained", default=None, help="Optional HuggingFace repo or local dir for MoGe.from_pretrained")
    parser.add_argument("--device", default="cuda", help="Inference device, e.g. cuda or cuda:0")
    parser.add_argument("--fp16", action="store_true", help="Enable half precision inference")
    parser.add_argument("--num-tokens", type=int, default=None, help="Override ViT tokens for inference")
    parser.add_argument("--resolution-level", type=int, default=9, help="Model resolution level (ignored if num_tokens set)")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of frames to process")
    parser.add_argument("--split", default=None, help="Optional split file (e.g. val.txt) relative to dataset root")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
