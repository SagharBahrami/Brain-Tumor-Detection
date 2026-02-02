from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .preprocessing import (
    get_modality_paths,
    load_nifti,
    normalize_volume,
    stack_modalities,
)
from functools import lru_cache


# Module-level cache for loaded modality volumes per patient_dir
# Keyed by (patient_dir_str, modalities_tuple)
@lru_cache(maxsize=8)
def _load_modalities_cached(patient_dir_str: str, modalities_tuple: Tuple[str, ...]) -> Dict[str, np.ndarray]:
	"""Load and normalize all modalities for a patient directory and cache the result.

	Returns a dict mapping modality name -> numpy array (normalized volume).
	The cache stores numpy arrays; be mindful of memory usage when adjusting maxsize.
	"""
	patient_dir = Path(patient_dir_str)
	paths = get_modality_paths(patient_dir, list(modalities_tuple))
	volumes: Dict[str, np.ndarray] = {}
	for modality, path in paths.items():
		data = load_nifti(path)
		data = normalize_volume(data)
		volumes[modality] = data
	return volumes


@lru_cache(maxsize=8)
def _load_segmentation_cached(patient_dir_str: str) -> np.ndarray:
	"""Load segmentation volume for a patient and cache the result.
	
	Returns normalized segmentation volume.
	The cache stores numpy arrays; maxsize=8 means we cache up to 8 patients in memory.
	"""
	patient_dir = Path(patient_dir_str)
	seg_paths = get_modality_paths(patient_dir, ["seg"])
	seg_volume = load_nifti(seg_paths["seg"])
	return seg_volume


class AdditiveGaussianNoise:
	"""Simple Gaussian noise transform for tensor images."""

	def __init__(self, std: float = 0.05):
		self.std = float(std)

	def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
		if not tensor.is_floating_point():
			tensor = tensor.float()
		noise = torch.randn_like(tensor) * self.std
		return tensor + noise


@dataclass
class DatasetConfig:
	modalities: Sequence[str]
	image_size: Tuple[int, int] = (224, 224)




class SliceTumorPatientDataset(Dataset):
	"""Dataset for slice-level tumor detection - PATIENT-CENTRIC.
	
	Each item is ONE PATIENT. __getitem__ returns ALL slices from that patient.
	- __len__ returns number of patients
	- __getitem__(idx) loads patient idx, extracts all slices with labels, returns batch of slices
	
	Usage: batch_size=1, num_workers=0
	One training step = one patient = ~155 slices
	"""
	
	def __init__(
		self,
		patient_dirs: List[Path],
		config: DatasetConfig,
		augment: bool = False,
	) -> None:
		self.patient_dirs = sorted([Path(p) for p in patient_dirs])
		self.config = config
		self.modalities = config.modalities
		self.image_size = config.image_size
		self.augment = augment
		self.transforms = self._build_transforms(augment)
	
	def _build_transforms(self, augment: bool) -> transforms.Compose:
		transform_list: List[transforms.Transform] = []
		if augment:
			# Minimal augmentation - avoid expensive operations
			transform_list.extend([
				transforms.RandomHorizontalFlip(p=0.5),
			])
		transform_list.append(transforms.ConvertImageDtype(torch.float32))
		return transforms.Compose(transform_list)
	
	def __len__(self) -> int:
		"""Returns number of patients (not slices)."""
		return len(self.patient_dirs)
	
	def __getitem__(self, idx: int) -> Dict[str, Any]:
		"""Load one patient and return ALL slices with labels.
		
		Returns:
		  {
		    'images': Tensor of shape (N_slices, C, H, W),
		    'labels': Tensor of shape (N_slices,),
		    'patient_id': str,
		    'slice_count': int
		  }
		"""
		patient_dir = self.patient_dirs[idx]
		patient_id = patient_dir.name
		
		# Load segmentation to determine slice count
		seg_volume = _load_segmentation_cached(str(patient_dir))
		num_slices = seg_volume.shape[2]
		
		# Load all modalities for this patient
		modality_volumes = _load_modalities_cached(str(patient_dir), tuple(self.modalities))
		
		# Extract all slices with labels
		images_list = []
		labels_list = []
		
		for slice_idx in range(num_slices):
			# Check if this slice contains tumor
			slice_seg = seg_volume[:, :, slice_idx]
			has_tumor = np.any(slice_seg > 0)
			label = 1.0 if has_tumor else 0.0
			
			# Stack modalities for this slice
			stacked_slice = stack_modalities(
				modality_volumes,
				slice_idx,
				self.image_size
			)
			
			# Convert to tensor and apply transforms
			image_tensor = torch.from_numpy(stacked_slice)
			image_tensor = self.transforms(image_tensor)
			
			images_list.append(image_tensor)
			labels_list.append(label)
		
		# Stack all slices into tensor
		images_batch = torch.stack(images_list)  # (N_slices, C, H, W)
		labels_batch = torch.tensor(labels_list, dtype=torch.float32)  # (N_slices,)
		
		return {
			"images": images_batch,
			"labels": labels_batch,
			"patient_id": patient_id,
			"slice_count": num_slices,
		}



def create_patient_centric_data_loaders(
	patient_dirs: List[Path],
	config: DatasetConfig,
	val_frac: float = 0.15,
	test_frac: float = 0.15,
	seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
	"""Create data loaders using patient-centric dataset.
	
	Each training step processes one full patient (~155 slices).
	Enforces batch_size=1 and num_workers=0 to avoid shared memory issues.
	
	Returns (train_loader, val_loader, test_loader)
	"""
	
	# Split patient directories
	np.random.seed(seed)
	patient_dirs_arr = np.array(patient_dirs)
	indices = np.random.permutation(len(patient_dirs_arr))
	
	n_val = int(len(patient_dirs_arr) * val_frac)
	n_test = int(len(patient_dirs_arr) * test_frac)
	n_train = len(patient_dirs_arr) - n_val - n_test
	
	train_dirs = patient_dirs_arr[indices[:n_train]]
	val_dirs = patient_dirs_arr[indices[n_train:n_train+n_val]]
	test_dirs = patient_dirs_arr[indices[n_train+n_val:]]
	
	# Create patient-centric datasets
	train_dataset = SliceTumorPatientDataset(train_dirs, config, augment=True)
	val_dataset = SliceTumorPatientDataset(val_dirs, config, augment=False)
	test_dataset = SliceTumorPatientDataset(test_dirs, config, augment=False)
	
	# Create data loaders with batch_size=1 and num_workers=4
	# Multiple workers speed up I/O loading with 24GB VRAM available.
	train_loader = DataLoader(
		train_dataset,
		batch_size=1,
		shuffle=True,
		num_workers=4,
		pin_memory=True,
	)
	
	val_loader = DataLoader(
		val_dataset,
		batch_size=1,
		shuffle=False,
		num_workers=4,
		pin_memory=True,
	)
	
	test_loader = DataLoader(
		test_dataset,
		batch_size=1,
		shuffle=False,
		num_workers=4,
		pin_memory=True,
	)
	
	print(f"Created patient-centric datasets (batch_size=1, num_workers=4):")
	print(f"  Train: {len(train_dataset)} patients")
	print(f"  Val: {len(val_dataset)} patients") 
	print(f"  Test: {len(test_dataset)} patients")
	
	return train_loader, val_loader, test_loader
