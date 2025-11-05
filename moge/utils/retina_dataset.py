from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def camera_to_intrinsics(camera_params: Dict) -> np.ndarray:
    lens = float(camera_params.get('lens', 0.0))
    sensor_width = float(camera_params.get('sensor_width', 0.0))
    sensor_height = float(camera_params.get('sensor_height', 0.0))
    if lens <= 0 or sensor_width <= 0 or sensor_height <= 0:
        raise ValueError('Invalid camera parameters for retina dataset conversion')
    fov_x = 2 * np.arctan(sensor_width / (2 * lens))
    fov_y = 2 * np.arctan(sensor_height / (2 * lens))
    fx = 0.5 / np.tan(fov_x / 2)
    fy = 0.5 / np.tan(fov_y / 2)
    intrinsics = np.array([
        [fx, 0.0, 0.5],
        [0.0, fy, 0.5],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return intrinsics


def read_binary_mask(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None or not path.exists():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return (mask > 127).astype(np.float32)


def load_depth_map(path: Path) -> np.ndarray:
    if path.suffix == '.npy':
        return np.load(path).astype(np.float32)
    depth = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise FileNotFoundError(f'Unable to read depth map at {path}')
    return depth.astype(np.float32)


def _normalize_split_entry(entry: str) -> Optional[str]:
    entry = entry.strip()
    if not entry or entry.startswith('#'):
        return None
    image_spec = entry.split(',')[0].strip()
    if not image_spec:
        return None
    image_spec = image_spec.replace('\\', '/')
    path = Path(image_spec)
    parts = path.parts
    if len(parts) >= 4 and parts[1] == 'left' and parts[2] == 'imgs':
        sequence = parts[0]
        frame = parts[-1]
        return f"{sequence}/left/{frame}"
    if len(parts) >= 3 and parts[1] == 'left':
        sequence = parts[0]
        frame = parts[-1]
        return f"{sequence}/left/{frame}"
    return image_spec


def _load_split_list(dataset_config: Dict) -> Optional[List[str]]:
    split_name = dataset_config.get('split')
    if not split_name:
        return None
    split_path = Path(split_name)
    search_paths: List[Path] = []
    if split_path.is_absolute():
        search_paths.append(split_path)
    else:
        dataset_root = Path(dataset_config['path'])
        search_paths.append(dataset_root / split_path)
        project_root = Path(__file__).resolve().parents[2]
        search_paths.append(project_root / split_path)
        search_paths.append(split_path)

    resolved_split_path: Optional[Path] = None
    for candidate in search_paths:
        if candidate.exists():
            resolved_split_path = candidate
            break

    if resolved_split_path is None:
        checked = ', '.join(str(p) for p in search_paths)
        raise FileNotFoundError(f'Split file not found. Checked: {checked}')

    if resolved_split_path.is_dir():
        raise ValueError(f'Split path {resolved_split_path} should be a file, not a directory.')

    entries: List[str] = []
    for raw in resolved_split_path.read_text().splitlines():
        key = _normalize_split_entry(raw)
        if key is not None:
            entries.append(key)
    return entries or None


def scan_retina_records(dataset_config: Dict) -> Tuple[List[str], Dict[str, Dict]]:
    root = Path(dataset_config['path'])
    records: Dict[str, Dict] = {}
    filenames: List[str] = []

    split_entries = _load_split_list(dataset_config)

    sequences = sorted([p for p in root.iterdir() if p.is_dir() and p.name.endswith('_processed')])
    if not sequences:
        raise RuntimeError(f'No sequences found under {root}')

    for seq_dir in sequences:
        intrinsics_path = seq_dir / 'instrincs.json'
        if intrinsics_path.exists():
            try:
                intrinsics_data = json.loads(intrinsics_path.read_text())
            except json.JSONDecodeError:
                print(f'Warning: failed to parse intrinsics file: {intrinsics_path}')
                intrinsics_data = []
        else:
            intrinsics_data = []

        camera_map = {entry['name']: entry.get('camera_l') for entry in intrinsics_data if 'camera_l' in entry}

        left_dir = seq_dir / 'left'
        image_dir = left_dir / 'imgs'
        metric_depth_dir = left_dir / 'metric_depth'
        depth_dir = left_dir / 'depth'
        valid_mask_dir = left_dir / 'valid_region_mask'
        instrument_mask_dir = left_dir / 'instrument_mask'

        for image_path in sorted(image_dir.glob('*.png')):
            frame_name = image_path.name
            frame_key = f"{seq_dir.name}/left/{frame_name}"
            depth_path_npy = metric_depth_dir / f"{image_path.stem}.npy"
            depth_path_png = depth_dir / frame_name
            depth_path = depth_path_npy if depth_path_npy.exists() else depth_path_png
            if not depth_path.exists():
                print(f'Warning: depth file missing for frame {frame_key}, skipping.')
                continue

            camera_params = camera_map.get(frame_name)
            if camera_params is None:
                print(f'Warning: intrinsics missing for frame {frame_key}, skipping.')
                continue

            intrinsics = camera_to_intrinsics(camera_params)
            records[frame_key] = {
                'image_path': image_path,
                'depth_path': depth_path,
                'valid_mask_path': valid_mask_dir / frame_name if valid_mask_dir.exists() else None,
                'instrument_mask_path': instrument_mask_dir / frame_name if instrument_mask_dir.exists() else None,
                'intrinsics': intrinsics,
                'sequence': seq_dir.name,
                'frame_name': frame_name,
            }
            filenames.append(frame_key)

    if not filenames:
        raise RuntimeError(f'No valid samples found in retina dataset at {root}.')

    if split_entries is not None:
        filtered_records: Dict[str, Dict] = {}
        missing_entries: List[str] = []
        for key in split_entries:
            if key in records:
                filtered_records[key] = records[key]
            else:
                missing_entries.append(key)
        if not filtered_records:
            missing_preview = ', '.join(missing_entries[:5])
            raise RuntimeError(
                f'No samples from split "{dataset_config["split"]}" matched the retina dataset. '
                f'Sample missing keys: {missing_preview}'
            )
        if missing_entries:
            print(
                f'Warning: {len(missing_entries)} entries listed in split file '
                f'"{dataset_config["split"]}" were not found in the dataset.'
            )
        filenames = list(filtered_records.keys())
        records = filtered_records

    return filenames, records
