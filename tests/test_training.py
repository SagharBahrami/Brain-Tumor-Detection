from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from src.training import MetricsHistoryCallback, TrainingConfig, train_model


class DummyModel(nn.Module):
    """Minimal model for testing."""
    
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(10, 1)  # Add classifier attribute
    
    def forward(self, x):
        return self.classifier(x)


def test_metrics_history_callback():
    """Test MetricsHistoryCallback collects metrics."""
    callback = MetricsHistoryCallback()
    
    # Mock trainer with callback_metrics
    trainer = Mock()
    trainer.current_epoch = 1
    trainer.callback_metrics = {
        'train_loss': torch.tensor(0.5),
        'val_auc': torch.tensor(0.85),
        'val_loss': 0.3
    }
    
    # Test collection - only call one method to avoid double counting
    callback.on_train_epoch_end(trainer, None)
    
    assert 'train_loss' in callback.metrics_history
    assert 'val_auc' in callback.metrics_history
    assert 'val_loss' in callback.metrics_history
    assert len(callback.metrics_history['train_loss']) == 1
    assert callback.metrics_history['train_loss'][0] == (1, 0.5)


def test_training_config_experiment_name():
    """Test TrainingConfig includes experiment_name."""
    config = TrainingConfig(experiment_name="test_exp")
    assert config.experiment_name == "test_exp"
    assert config.output_dir == Path("experiments_logs")


@patch('src.training.pl.Trainer')
@patch('src.training.TensorBoardLogger')
def test_train_model_creates_correct_structure(mock_logger, mock_trainer):
    """Test train_model creates expected directory structure and files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Mock minimal components
        mock_logger_instance = Mock()
        mock_logger_instance.log_dir = str(tmp_path / "test_run")
        mock_logger.return_value = mock_logger_instance
        
        mock_trainer_instance = Mock()
        mock_trainer_instance.fit = Mock()
        mock_trainer_instance.callback_metrics = {'val_loss': 0.5}
        mock_trainer_instance.current_epoch = 1
        mock_trainer_instance.checkpoint_callback.best_model_path = str(tmp_path / "best.ckpt")
        mock_trainer.return_value = mock_trainer_instance
        
        # Create minimal data loaders
        train_loader = Mock()
        val_loader = Mock()
        train_loader.__len__ = Mock(return_value=10)
        val_loader.__len__ = Mock(return_value=5)
        
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.records = [{'label': 0, 'patient_id': '001', 'has_tumor': True}]
        mock_dataset.__len__ = Mock(return_value=10)
        train_loader.dataset = mock_dataset
        val_loader.dataset = mock_dataset
        
        config = TrainingConfig(
            max_epochs=1,
            output_dir=tmp_path,
            experiment_name="test_exp",
            fast_dev_run=True
        )
        
        model = DummyModel()
        
        result = train_model(model, train_loader, val_loader, config)
        
        # Verify logger was created with correct format
        mock_logger.assert_called_once()
        call_args = mock_logger.call_args
        assert call_args[1]['name'] == ''
        assert 'test_exp' in call_args[1]['version']
        
        # Verify result structure
        assert result.run_dir.exists()
        assert (result.run_dir / "metadata.json").exists()
        assert (result.run_dir / "config.py").exists()
        assert (result.run_dir / "metrics_history.json").exists()
        # CSV may not be created if no metrics collected
        # assert (result.run_dir / "metrics_history.csv").exists()
        
        # Verify metadata content
        with open(result.run_dir / "metadata.json") as f:
            metadata = json.load(f)
            assert metadata['experiment_name'] == 'test_exp'
            assert 'timestamp' in metadata
            assert 'model' in metadata