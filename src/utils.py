import random
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
	"""Set random seed for reproducibility."""

	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path) -> Path:
	"""Create directory if it does not exist."""

	path.mkdir(parents=True, exist_ok=True)
	return path


def count_parameters(model: torch.nn.Module) -> int:
	"""Return the number of trainable parameters."""

	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def move_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
	"""Move tensors in batch to device."""

	return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def compute_epoch_steps(num_samples: int, batch_size: int) -> int:
	"""Return number of steps per epoch (ceiling division)."""

	return (num_samples + batch_size - 1) // batch_size


def load_patient_manifest(manifest_path: Optional[Path]) -> Optional[Iterable[str]]:
	"""Load a manifest file with patient identifiers (one per line)."""

	if manifest_path is None:
		return None

	manifest_path = Path(manifest_path)
	if not manifest_path.exists():
		raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

	with manifest_path.open("r", encoding="utf-8") as f:
		return [line.strip() for line in f if line.strip()]
