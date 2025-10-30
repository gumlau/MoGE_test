from typing import Dict, List
import math

import torch


def _masked_values(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6):
    if mask is None:
        mask = torch.ones_like(gt, dtype=torch.bool)
    mask = mask & torch.isfinite(pred) & torch.isfinite(gt)
    if mask.ndim != pred.ndim:
        mask = mask.reshape(pred.shape)
    if not torch.any(mask):
        return None, None
    pred_vals = pred[mask]
    gt_vals = gt[mask]
    gt_vals = torch.clamp(gt_vals, min=eps)
    pred_vals = torch.clamp(pred_vals, min=eps)
    return pred_vals, gt_vals


def compute_depth_metrics(pred_depth: torch.Tensor, gt_depth: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> Dict[str, float]:
    """Compute masked depth metrics for retina dataset."""
    pred_vals, gt_vals = _masked_values(pred_depth, gt_depth, mask, eps=eps)
    metrics = {
        'd0.5': float('nan'),
        'd1': float('nan'),
        'd2': float('nan'),
        'd3': float('nan'),
        'abs_rel': float('nan'),
        'sq_rel': float('nan'),
        'rmse': float('nan'),
        'rmse_log10': float('nan'),
        'log10': float('nan'),
        'silog': float('nan'),
    }
    if pred_vals is None or gt_vals is None:
        return metrics

    diff = pred_vals - gt_vals
    metrics['abs_rel'] = (diff.abs() / gt_vals).mean().item()
    metrics['sq_rel'] = ((diff ** 2) / gt_vals).mean().item()
    metrics['rmse'] = torch.sqrt((diff ** 2).mean()).item()

    log10_pred = torch.log10(pred_vals)
    log10_gt = torch.log10(gt_vals)
    metrics['rmse_log10'] = torch.sqrt(((log10_pred - log10_gt) ** 2).mean()).item()
    metrics['log10'] = (log10_pred - log10_gt).abs().mean().item()

    log_pred = torch.log(pred_vals)
    log_gt = torch.log(gt_vals)
    log_diff = log_pred - log_gt
    silog = torch.sqrt(torch.clamp(log_diff.pow(2).mean() - log_diff.mean() ** 2, min=0.0))
    metrics['silog'] = (silog * 100).item()

    ratios = torch.maximum(gt_vals / pred_vals, pred_vals / gt_vals)
    thresholds = {
        'd0.5': 1.25 ** 0.5,
        'd1': 1.25,
        'd2': 1.25 ** 2,
        'd3': 1.25 ** 3,
    }
    for key, thr in thresholds.items():
        metrics[key] = (ratios < thr).float().mean().item()

    return metrics


def average_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics_list:
        return {}
    sums = {}
    counts = {}
    for metrics in metrics_list:
        for key, value in metrics.items():
            if value is None or math.isnan(value):
                continue
            sums[key] = sums.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / counts[key] for key in sums if counts[key] > 0}
