"""
Comprehensive logging system for training
Logs to file, console, TensorBoard, and CSV
"""

import logging
import os
import csv
import json
from datetime import datetime
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np


class TrainingLogger:
    """Comprehensive logger for model training"""

    def __init__(
        self,
        model_name,
        log_dir='logs',
        save_dir='checkpoints',
        use_tensorboard=True,
        log_level=logging.INFO
    ):
        self.model_name = model_name
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_name = f"{model_name}_{self.timestamp}"

        # Create directories
        self.log_dir = Path(log_dir) / self.run_name
        self.save_dir = Path(save_dir) / self.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Setup Python logger
        self.logger = self._setup_logger(log_level)

        # Setup TensorBoard
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            self.writer = SummaryWriter(log_dir=str(self.log_dir / 'tensorboard'))
        else:
            self.writer = None

        # Metric tracking
        self.metrics_history = {}
        self.epoch_metrics = []
        self.step_metrics = []

        # CSV files
        self.epoch_csv = self.log_dir / 'epoch_metrics.csv'
        self.step_csv = self.log_dir / 'step_metrics.csv'
        self.config_json = self.log_dir / 'config.json'

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        self.best_metric_name = 'loss'

        self.logger.info("="*80)
        self.logger.info(f"Training Logger Initialized: {self.run_name}")
        self.logger.info(f"Log directory: {self.log_dir}")
        self.logger.info(f"Save directory: {self.save_dir}")
        self.logger.info("="*80)

    def _setup_logger(self, log_level):
        """Setup Python logger with file and console handlers"""
        logger = logging.getLogger(self.run_name)
        logger.setLevel(log_level)
        logger.handlers = []  # Clear existing handlers

        # File handler
        log_file = self.log_dir / 'training.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        # Formatter
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def log_config(self, config):
        """Log training configuration"""
        self.logger.info("Training Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")

        # Save to JSON
        with open(self.config_json, 'w') as f:
            json.dump(config, f, indent=4)

        self.logger.info(f"Configuration saved to: {self.config_json}")

    def log_model_info(self, model):
        """Log model architecture information"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.logger.info("Model Information:")
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")
        self.logger.info(f"  Parameters (M): {total_params / 1e6:.2f}M")

        # Save model architecture
        model_arch_file = self.log_dir / 'model_architecture.txt'
        with open(model_arch_file, 'w') as f:
            f.write(str(model))

        self.logger.info(f"Model architecture saved to: {model_arch_file}")

    def log_step(self, step, metrics, prefix='train'):
        """Log metrics for a single step"""
        self.global_step = step

        # Format metrics
        metric_str = f"[{prefix.upper()}] Step {step}"
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metric_str += f" | {key}: {value:.4f}"

        self.logger.debug(metric_str)

        # TensorBoard
        if self.writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'{prefix}/{key}', value, step)

        # Store step metrics
        step_data = {'step': step, 'prefix': prefix}
        step_data.update(metrics)
        self.step_metrics.append(step_data)

    def log_epoch(self, epoch, metrics, prefix='train'):
        """Log metrics for an epoch"""
        self.current_epoch = epoch

        # Format metrics
        metric_str = f"\n{'='*80}\n"
        metric_str += f"Epoch {epoch} [{prefix.upper()}]\n"
        metric_str += f"{'-'*80}\n"

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metric_str += f"  {key:20s}: {value:.6f}\n"

        metric_str += f"{'='*80}"

        self.logger.info(metric_str)

        # TensorBoard
        if self.writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'epoch_{prefix}/{key}', value, epoch)

        # Store epoch metrics
        epoch_data = {'epoch': epoch, 'prefix': prefix}
        epoch_data.update(metrics)
        self.epoch_metrics.append(epoch_data)

        # Track history
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                full_key = f'{prefix}_{key}'
                if full_key not in self.metrics_history:
                    self.metrics_history[full_key] = []
                self.metrics_history[full_key].append(value)

        # Save to CSV
        self._save_epoch_csv()

    def log_images(self, tag, images, step=None, nrow=8):
        """Log images to TensorBoard"""
        if self.writer and images is not None:
            step = step if step is not None else self.global_step

            # Handle different image formats
            if isinstance(images, torch.Tensor):
                from torchvision.utils import make_grid
                grid = make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
                self.writer.add_image(tag, grid, step)

            self.logger.debug(f"Logged images: {tag} at step {step}")

    def log_histogram(self, tag, values, step=None):
        """Log histogram to TensorBoard"""
        if self.writer and values is not None:
            step = step if step is not None else self.global_step
            self.writer.add_histogram(tag, values, step)

    def save_checkpoint(self, model, optimizer=None, scheduler=None,
                       discriminator=None, disc_optimizer=None,
                       metrics=None, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': model.state_dict(),
            'metrics': metrics or {}
        }

        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        if discriminator:
            checkpoint['discriminator_state_dict'] = discriminator.state_dict()

        if disc_optimizer:
            checkpoint['disc_optimizer_state_dict'] = disc_optimizer.state_dict()

        # Save latest checkpoint
        latest_path = self.save_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, latest_path)
        self.logger.info(f"Saved checkpoint: {latest_path}")

        # Save epoch checkpoint
        epoch_path = self.save_dir / f'checkpoint_epoch_{self.current_epoch}.pt'
        torch.save(checkpoint, epoch_path)

        # Save best checkpoint
        if is_best:
            best_path = self.save_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"✓ New best model saved! (epoch {self.current_epoch})")

        return latest_path

    def check_best_metric(self, metric_value, metric_name='loss', mode='min'):
        """Check if current metric is the best"""
        self.best_metric_name = metric_name

        is_best = False
        if mode == 'min':
            if metric_value < self.best_metric:
                self.best_metric = metric_value
                is_best = True
        else:  # mode == 'max'
            if metric_value > self.best_metric:
                self.best_metric = metric_value
                is_best = True

        return is_best

    def _save_epoch_csv(self):
        """Save epoch metrics to CSV"""
        if not self.epoch_metrics:
            return

        # Get all unique keys
        all_keys = set()
        for metrics in self.epoch_metrics:
            all_keys.update(metrics.keys())

        all_keys = sorted(all_keys)

        # Write CSV
        with open(self.epoch_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(self.epoch_metrics)

        self.logger.debug(f"Saved epoch metrics to: {self.epoch_csv}")

    def save_step_csv(self):
        """Save step metrics to CSV"""
        if not self.step_metrics:
            return

        # Get all unique keys
        all_keys = set()
        for metrics in self.step_metrics:
            all_keys.update(metrics.keys())

        all_keys = sorted(all_keys)

        # Write CSV
        with open(self.step_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(self.step_metrics)

        self.logger.info(f"Saved step metrics to: {self.step_csv}")

    def plot_training_curves(self, save_path=None):
        """Plot training curves from logged metrics"""
        if not self.metrics_history:
            self.logger.warning("No metrics to plot")
            return

        # Separate train and val metrics
        train_metrics = {k: v for k, v in self.metrics_history.items() if k.startswith('train_')}
        val_metrics = {k: v for k, v in self.metrics_history.items() if k.startswith('val_')}

        # Determine grid size
        num_metrics = len(set([k.split('_', 1)[1] for k in self.metrics_history.keys()]))
        ncols = 2
        nrows = (num_metrics + 1) // 2

        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
        if nrows == 1:
            axes = axes.reshape(1, -1)

        plot_idx = 0
        metric_names = set()

        # Get unique metric names
        for key in self.metrics_history.keys():
            metric_name = key.split('_', 1)[1]
            metric_names.add(metric_name)

        for metric_name in sorted(metric_names):
            if plot_idx >= nrows * ncols:
                break

            row = plot_idx // ncols
            col = plot_idx % ncols
            ax = axes[row, col]

            # Plot train
            train_key = f'train_{metric_name}'
            if train_key in self.metrics_history:
                epochs = list(range(len(self.metrics_history[train_key])))
                ax.plot(epochs, self.metrics_history[train_key],
                       label='Train', linewidth=2, marker='o', markersize=4)

            # Plot val
            val_key = f'val_{metric_name}'
            if val_key in self.metrics_history:
                epochs = list(range(len(self.metrics_history[val_key])))
                ax.plot(epochs, self.metrics_history[val_key],
                       label='Val', linewidth=2, marker='s', markersize=4)

            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=11)
            ax.set_title(f'{metric_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            plot_idx += 1

        # Hide unused subplots
        for idx in range(plot_idx, nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            axes[row, col].axis('off')

        plt.tight_layout()

        if save_path is None:
            save_path = self.log_dir / 'training_curves.png'

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Saved training curves to: {save_path}")
        plt.close()

    def get_summary(self):
        """Get training summary"""
        summary = {
            'model_name': self.model_name,
            'run_name': self.run_name,
            'total_epochs': self.current_epoch,
            'total_steps': self.global_step,
            'best_metric': self.best_metric,
            'best_metric_name': self.best_metric_name,
            'log_dir': str(self.log_dir),
            'save_dir': str(self.save_dir)
        }

        return summary

    def close(self):
        """Close logger and save final state"""
        self.logger.info("\n" + "="*80)
        self.logger.info("Training Completed!")
        self.logger.info("="*80)

        # Save final CSVs
        self._save_epoch_csv()
        self.save_step_csv()

        # Plot training curves
        self.plot_training_curves()

        # Save summary
        summary = self.get_summary()
        summary_file = self.log_dir / 'training_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)

        self.logger.info(f"\nTraining Summary:")
        for key, value in summary.items():
            self.logger.info(f"  {key}: {value}")

        # Close TensorBoard
        if self.writer:
            self.writer.close()

        self.logger.info("\nAll logs saved successfully!")
        self.logger.info("="*80 + "\n")


def get_logger(model_name, log_dir='logs', **kwargs):
    """Convenience function to create a logger"""
    return TrainingLogger(model_name, log_dir=log_dir, **kwargs)
