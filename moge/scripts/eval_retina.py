import os
from pathlib import Path
import sys
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)
import json
from typing import Dict, Optional, List

import click
import numpy as np
import torch
from tqdm import tqdm

from moge.utils.io import read_image
from moge.test.retina_metrics import compute_depth_metrics, average_metrics
from moge.model import import_model_class_by_version
from moge.utils.retina_dataset import scan_retina_records, read_binary_mask, load_depth_map


def _resolve_split(dataset_cfg: Dict, split_name: Optional[str]) -> Optional[str]:
    dataset_root = Path(dataset_cfg['path'])

    def _candidate_exists(candidate: str) -> bool:
        candidate_path = Path(candidate)
        if not candidate_path.is_absolute():
            candidate_path = dataset_root / candidate_path
        return candidate_path.exists()

    if split_name:
        return split_name

    candidates = [
        'test.txt',
        'split/test.txt',
        'val.txt',
        'split/val.txt',
        'validation.txt',
        dataset_cfg.get('split'),
        'train.txt',
    ]
    for candidate in candidates:
        if candidate and _candidate_exists(candidate):
            return candidate
    return None


def _load_retina_records(config: Dict, split_name: Optional[str]) -> Dict[str, Dict]:
    retina_datasets = [dataset for dataset in config['data']['datasets'] if dataset.get('type') == 'retina_surgery']
    if not retina_datasets:
        raise ValueError('No retina_surgery dataset configuration found in config file.')
    if len(retina_datasets) > 1:
        click.echo('Warning: multiple retina_surgery datasets configured, using the first entry.', err=True)
    dataset_cfg = dict(retina_datasets[0])
    resolved_split = _resolve_split(dataset_cfg, split_name)
    if resolved_split is not None:
        dataset_cfg['split'] = resolved_split
    filenames, records = scan_retina_records(dataset_cfg)
    return {key: records[key] for key in filenames}


def _load_masks(record: Dict) -> Dict[str, Optional[np.ndarray]]:
    valid = read_binary_mask(record.get('valid_mask_path'))
    if valid is None:
        valid = None
    instrument = read_binary_mask(record.get('instrument_mask_path'))
    if instrument is None:
        instrument = None
    return {'valid': valid, 'instrument': instrument}


def _prepare_image_tensor(image: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
    return tensor


def _to_bool_mask(array: Optional[np.ndarray], shape: np.ndarray) -> torch.Tensor:
    if array is None:
        return torch.ones(shape, dtype=torch.bool)
    if array.ndim == 3:
        array = array[..., 0]
    return torch.from_numpy((array > 0.5).astype(np.bool_))


def _merge_results(per_sample: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    overall_metrics = [item['overall'] for item in per_sample]
    instrument_metrics = [item['instrument'] for item in per_sample if item['instrument'] is not None]
    results = {
        'overall': average_metrics(overall_metrics),
    }
    if instrument_metrics:
        results['instrument'] = average_metrics(instrument_metrics)
    else:
        results['instrument'] = {}
    return results


@click.command()
@click.option('--config', 'config_path', type=click.Path(exists=True), default='configs/train/retina_surgery.json', help='Path to finetuning config containing retina dataset description.')
@click.option('--checkpoint', 'checkpoint_path', type=click.Path(exists=True), default=None, help='Path to a finetuned checkpoint (.pt) containing model weights.')
@click.option('--pretrained', 'pretrained_repo', type=str, default=None, help='Optional HuggingFace hub repo or local directory for loading pretrained weights via from_pretrained.')
@click.option('--device', type=str, default='cuda', help='Device for inference (e.g., "cuda", "cuda:0", "cpu").')
@click.option('--output', 'output_path', type=click.Path(), default=None, help='Optional path to write aggregated metrics JSON.')
@click.option('--limit', type=int, default=None, help='Maximum number of samples to evaluate.')
@click.option('--num_tokens', type=int, default=None, help='Override number of ViT tokens during inference.')
@click.option('--resolution_level', type=int, default=9, help='Resolution level passed to model.infer.')
@click.option('--split', type=str, default=None, help='Optional split file (e.g., val.txt) relative to the dataset root.')
def main(
    config_path: str,
    checkpoint_path: Optional[str],
    pretrained_repo: Optional[str],
    device: str,
    output_path: Optional[str],
    limit: Optional[int],
    num_tokens: Optional[int],
    resolution_level: int,
    split: Optional[str],
):
    torch.set_grad_enabled(False)
    device = torch.device(device)

    config = json.loads(Path(config_path).read_text())
    records = _load_retina_records(config, split)
    keys = list(records.keys())
    if limit is not None:
        keys = keys[:limit]

    if checkpoint_path is None and pretrained_repo is None:
        raise click.BadParameter('Either --checkpoint or --pretrained must be provided to load model weights.')

    MoGeModel = import_model_class_by_version(config['model_version'])
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        model = MoGeModel(**config['model'])
        state_dict = checkpoint.get('model', checkpoint)
        model.load_state_dict(state_dict, strict=False)
    else:
        model = MoGeModel.from_pretrained(pretrained_repo)
    model.to(device)
    model.eval()

    per_sample_metrics = []

    for key in tqdm(keys, desc='Evaluating retina dataset'):
        record = records[key]
        image = read_image(record['image_path'])
        depth = load_depth_map(record['depth_path'])
        masks = _load_masks(record)
        valid_mask_np = masks['valid']
        if valid_mask_np is None:
            valid_mask_np = np.ones_like(depth, dtype=np.float32)
        depth = np.where(valid_mask_np > 0.5, depth, np.nan)

        instrument_mask_np = masks['instrument']
        image_tensor = _prepare_image_tensor(image).to(device)

        output = model.infer(image_tensor, num_tokens=num_tokens, resolution_level=resolution_level)
        pred_depth = output.get('depth')
        if pred_depth is None:
            raise RuntimeError('Model inference did not return depth predictions.')
        pred_depth = pred_depth.detach().cpu()
        if pred_depth.ndim == 3:
            pred_depth = pred_depth.squeeze(0)

        gt_depth_tensor = torch.from_numpy(depth.astype(np.float32))
        valid_mask = _to_bool_mask(valid_mask_np, gt_depth_tensor.shape)
        instrument_mask = _to_bool_mask(instrument_mask_np, gt_depth_tensor.shape) if instrument_mask_np is not None else None

        overall_metrics = compute_depth_metrics(pred_depth, gt_depth_tensor, valid_mask)
        instrument_metrics = None
        if instrument_mask is not None:
            combined_mask = valid_mask & instrument_mask
            instrument_metrics = compute_depth_metrics(pred_depth, gt_depth_tensor, combined_mask)

        per_sample_metrics.append({
            'key': key,
            'overall': overall_metrics,
            'instrument': instrument_metrics,
        })

    aggregated = _merge_results(per_sample_metrics)

    click.echo('Overall metrics:')
    for metric, value in sorted(aggregated['overall'].items()):
        click.echo(f'  {metric:>10s}: {value:.6f}')
    if aggregated.get('instrument'):
        click.echo('Instrument-region metrics:')
        for metric, value in sorted(aggregated['instrument'].items()):
            click.echo(f'  {metric:>10s}: {value:.6f}')
    else:
        click.echo('Instrument-region metrics: N/A (no valid instrument pixels).')

    if output_path is not None:
        output = {
            'aggregated': aggregated,
            'per_sample': per_sample_metrics,
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(output, indent=4))


if __name__ == '__main__':
    main()
