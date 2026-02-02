"""PyTorch Lightning training module for slice-level tumor detection."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score

from .utils import set_seed


@dataclass
class TrainingConfig:
	"""Minimal config for training."""
	max_epochs: int = 20
	lr: float = 1e-3
	weight_decay: float = 1e-4
	seed: int = 42
	freeze_backbone_epochs: int = 5


class TumorDetectionModule(L.LightningModule):
	"""Lightning module for binary tumor detection."""
	
	def __init__(self, model: nn.Module, config: TrainingConfig):
		super().__init__()
		self.model = model
		self.config = config
		self.criterion = nn.BCEWithLogitsLoss()
		
		# Metrics for training
		self.train_accuracy = BinaryAccuracy()
		self.train_f1_score = BinaryF1Score()
		
		# Metrics for validation
		self.validation_accuracy = BinaryAccuracy()
		self.validation_f1_score = BinaryF1Score()
		
		# Metrics for testing
		self.test_accuracy = BinaryAccuracy()
		self.test_f1_score = BinaryF1Score()
		
		self.freeze_until = config.freeze_backbone_epochs
		self._freeze_backbone()
	
	def _freeze_backbone(self):
		"""Freeze all layers except classifier head."""
		for name, param in self.model.named_parameters():
			if not self._is_classifier_layer(name):
				param.requires_grad = False
	
	def _unfreeze_backbone(self):
		"""Unfreeze all parameters for fine-tuning."""
		for param in self.model.parameters():
			param.requires_grad = True
	
	@staticmethod
	def _is_classifier_layer(name: str) -> bool:
		"""Check if parameter belongs to classifier head."""
		classifier_names = ("fc", "classifier", "head", "linear")
		return any(x in name.lower() for x in classifier_names)
	
	def on_train_epoch_start(self):
		"""Unfreeze backbone after N epochs."""
		if self.current_epoch == self.freeze_until:
			self._unfreeze_backbone()
	
	def forward(self, x):
		return self.model(x)
	
	def training_step(self, batch, batch_idx):
		images = batch["images"]  # (batch_size=1, num_slices, C, H, W)
		labels = batch["labels"]  # (batch_size=1, num_slices)
		
		# Remove batch dimension
		images = images.squeeze(0)  # (num_slices, C, H, W)
		labels = labels.squeeze(0)  # (num_slices,)
		
		logits = self(images).squeeze(-1)
		loss = self.criterion(logits, labels)
		
		predictions = (torch.sigmoid(logits) > 0.5).float()
		self.train_accuracy(predictions, labels)
		self.train_f1_score(predictions, labels)
		
		self.log("train_loss", loss, prog_bar=True)
		self.log("train_accuracy", self.train_accuracy, prog_bar=True)
		self.log("train_f1_score", self.train_f1_score, prog_bar=False)
		return loss
	
	def validation_step(self, batch, batch_idx):
		images = batch["images"]  # (batch_size=1, num_slices, C, H, W)
		labels = batch["labels"]  # (batch_size=1, num_slices)
		
		# Remove batch dimension
		images = images.squeeze(0)  # (num_slices, C, H, W)
		labels = labels.squeeze(0)  # (num_slices,)
		
		logits = self(images).squeeze(-1)
		loss = self.criterion(logits, labels)
		
		predictions = (torch.sigmoid(logits) > 0.5).float()
		self.validation_accuracy(predictions, labels)
		self.validation_f1_score(predictions, labels)
		
		self.log("validation_loss", loss, prog_bar=True)
		self.log("validation_accuracy", self.validation_accuracy, prog_bar=True)
		self.log("validation_f1_score", self.validation_f1_score, prog_bar=False)
	
	def test_step(self, batch, batch_idx):
		images = batch["images"]  # (batch_size=1, num_slices, C, H, W)
		labels = batch["labels"]  # (batch_size=1, num_slices)
		
		# Remove batch dimension
		images = images.squeeze(0)  # (num_slices, C, H, W)
		labels = labels.squeeze(0)  # (num_slices,)
		
		logits = self(images).squeeze(-1)
		loss = self.criterion(logits, labels)
		
		predictions = (torch.sigmoid(logits) > 0.5).float()
		self.test_accuracy(predictions, labels)
		self.test_f1_score(predictions, labels)
		
		self.log("test_loss", loss, prog_bar=True)
		self.log("test_accuracy", self.test_accuracy, prog_bar=True)
		self.log("test_f1_score", self.test_f1_score, prog_bar=True)
	
	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.max_epochs, eta_min=1e-5)
		return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


def train_model(
	model: nn.Module,
	train_loader,
	val_loader,
	config: TrainingConfig,
	output_dir: Path = Path("experiments_logs"),
	experiment_name: str = "tumor_detection",
	model_name: str = "unknown",
	fast_dev_run: bool = False,
):
	"""Train model with clear checkpoint and logging organization.
	
	Checkpoint structure:
	  experiments_logs/
	    tumor_detection/
	      {model_name}_{YYYYMMDD_HHMMSS}/
	        checkpoints/
	          best-epoch=XX-val_acc=X.XXX-val_loss=X.XXX.ckpt
	          last-epoch=XX-val_acc=X.XXX-val_loss=X.XXX.ckpt
	        lightning_logs/
	          version_0/
	            hparams.yaml
	            metrics.csv
	
	Args:
	    fast_dev_run: If True, runs 1 batch for train/val/test to validate setup
	"""
	set_seed(config.seed)
	
	# Create timestamped directory for this model run
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	model_run_dir = output_dir / experiment_name / f"{model_name}_{timestamp}"
	model_run_dir.mkdir(parents=True, exist_ok=True)
	
	checkpoint_dir = model_run_dir / "checkpoints"
	checkpoint_dir.mkdir(parents=True, exist_ok=True)
	
	pl_module = TumorDetectionModule(model, config)
	
	trainer = L.Trainer(
		max_epochs=config.max_epochs,
		accelerator="auto",
		devices="auto",
		fast_dev_run=fast_dev_run,  # Run 1 batch for train/val/test if enabled
		callbacks=[
			ModelCheckpoint(
				dirpath=str(checkpoint_dir),
				monitor="validation_accuracy",
				mode="max",
				save_top_k=1,
				filename=f"{model_name}_best-{{epoch:02d}}-{{validation_accuracy:.4f}}-{{validation_loss:.4f}}",
			),
			ModelCheckpoint(
				dirpath=str(checkpoint_dir),
				save_last=True,
				filename=f"{model_name}_last-{{epoch:02d}}-{{validation_accuracy:.4f}}-{{validation_loss:.4f}}",
			),
			LearningRateMonitor(logging_interval="epoch"),
		],
		logger=CSVLogger(save_dir=str(model_run_dir), name="lightning_logs"),
		enable_progress_bar=True,
	)
	
	trainer.fit(pl_module, train_loader, val_loader)
	return trainer, model_run_dir
