from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from skimage.transform import resize

MODALITY_SUFFIXES: Dict[str, str] = {
	"flair": "flair",
	"t1": "t1",
	"t1ce": "t1ce",
	"t2": "t2",
	"seg": "seg",
}


@dataclass
class SliceSelectionConfig:
	tumor_slice_fraction: float = 1.0  # Use all tumor slices
	nontumor_slice_fraction: float = 0.1  # Sample 10% of non-tumor slices


def load_nifti(path: Path) -> np.ndarray:
	"""Load a NIfTI file into a numpy array (float32)."""

	image = nib.load(str(path))
	data = image.get_fdata().astype(np.float32)
	return data


def normalize_volume(volume: np.ndarray) -> np.ndarray:
	"""Z-score normalization over non-zero voxels."""

	mask = volume > 0
	if mask.sum() == 0:
		return np.zeros_like(volume, dtype=np.float32)

	mean = volume[mask].mean()
	std = volume[mask].std()
	if std == 0:
		std = 1.0
	normalized = (volume - mean) / std
	normalized[~mask] = 0.0
	return normalized.astype(np.float32)


def get_modality_paths(patient_dir: Path, modalities: Sequence[str]) -> Dict[str, Path]:
	paths: Dict[str, Path] = {}
	for modality in modalities:
		suffix = MODALITY_SUFFIXES[modality]
		file_name = f"{patient_dir.name}_{suffix}.nii.gz"
		paths[modality] = patient_dir / file_name
	return paths


def get_tumor_slice_indices(segmentation: np.ndarray) -> List[int]:
	"""Return all slice indices that contain any tumor pixels."""
	tumor_mask = segmentation > 0
	tumor_counts = tumor_mask.sum(axis=(0, 1))
	indices = [idx for idx, count in enumerate(tumor_counts) if count > 0]
	return indices


def get_best_tumor_slice(segmentation: np.ndarray, min_pixels: int = 10) -> int:
	"""Return slice index with maximum tumor area (fallback to middle slice)."""

	tumor_mask = segmentation > 0
	tumor_counts = tumor_mask.sum(axis=(0, 1))
	valid_indices = [idx for idx, count in enumerate(tumor_counts) if count >= min_pixels]
	if valid_indices:
		best_idx = max(valid_indices, key=lambda idx: tumor_counts[idx])
		return int(best_idx)

	depth = segmentation.shape[-1]
	return int(depth // 2)


def get_slice_window(center_idx: int, window_size: int, depth: int) -> List[int]:
	"""Return a symmetric window of slice indices around ``center_idx``."""

	if window_size <= 1:
		return [int(max(0, min(center_idx, depth - 1)))]

	radius = max(window_size // 2, 0)
	start = max(center_idx - radius, 0)
	end = min(center_idx + radius + 1, depth)
	indices = list(range(start, end))
	if len(indices) < window_size and indices:
		# Extend window if near boundaries
		while len(indices) < window_size and indices[0] > 0:
			indices.insert(0, indices[0] - 1)
		while len(indices) < window_size and indices[-1] < depth - 1:
			indices.append(indices[-1] + 1)
	return indices


def get_background_slice_indices(
	segmentation: np.ndarray,
	exclude: Sequence[int],
	max_slices: int,
	margin: int = 2,
) -> List[int]:
	"""Return slice indices without tumor signal near the tumor region."""

	exclude_set = set(exclude)
	candidate_indices = []
	depth = segmentation.shape[-1]
	tumor_present = bool(exclude)
	if tumor_present:
		min_idx, max_idx = min(exclude), max(exclude)
		lower = max(min_idx - margin, 0)
		upper = min(max_idx + margin, depth - 1)
	else:
		lower = 0
		upper = depth - 1

	for idx in range(lower, upper + 1):
		if idx in exclude_set:
			continue
		slice_has_tumor = np.any(segmentation[:, :, idx] > 0)
		if not slice_has_tumor:
			candidate_indices.append(idx)

	if not candidate_indices or max_slices <= 0:
		return []

	if len(candidate_indices) <= max_slices:
		return candidate_indices

	step = len(candidate_indices) / max_slices
	return [candidate_indices[int(i * step)] for i in range(max_slices)]


def select_slice_indices(segmentation: np.ndarray, config: SliceSelectionConfig) -> List[int]:
	"""Return selected slice indices based on tumor presence and fractions."""
	tumor_indices = get_tumor_slice_indices(segmentation)
	max_non_tumor = int(segmentation.shape[-1] * config.nontumor_slice_fraction)
	non_tumor_indices = get_background_slice_indices(segmentation, tumor_indices, max_non_tumor)
	return tumor_indices + non_tumor_indices


def resize_slice(slice_2d: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
	resized = resize(slice_2d, size, order=1, mode="constant", anti_aliasing=True)
	return resized.astype(np.float32)


def stack_modalities(modalities: Dict[str, np.ndarray], slice_idx: int, size: Tuple[int, int]) -> np.ndarray:
	channels = []
	for modality in modalities.values():
		slice_2d = modality[:, :, slice_idx]
		channels.append(resize_slice(slice_2d, size))
	stacked = np.stack(channels, axis=0)
	return stacked
