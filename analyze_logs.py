"""
Log analysis tool to compare and visualize training runs
"""

import json
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np

class LogAnalyzer:
    """Analyze training logs from multiple runs"""

    def __init__(self, log_dirs):
        self.log_dirs = [Path(d) for d in log_dirs]
        self.runs = {}
        self.load_logs()

    def load_logs(self):
        """Load logs from all directories"""
        print("Loading logs from all runs...")

        for log_dir in self.log_dirs:
            if not log_dir.exists():
                print(f"Warning: Directory not found: {log_dir}")
                continue

            # Find the actual run directory
            run_dirs = list(log_dir.glob('*'))
            for run_dir in run_dirs:
                if not run_dir.is_dir():
                    continue

                run_name = run_dir.name

                # Load epoch metrics
                epoch_csv = run_dir / 'epoch_metrics.csv'
                if epoch_csv.exists():
                    df_epoch = pd.read_csv(epoch_csv)
                else:
                    df_epoch = None

                # Load step metrics
                step_csv = run_dir / 'step_metrics.csv'
                if step_csv.exists():
                    df_step = pd.read_csv(step_csv)
                else:
                    df_step = None

                # Load summary
                summary_json = run_dir / 'training_summary.json'
                if summary_json.exists():
                    with open(summary_json, 'r') as f:
                        summary = json.load(f)
                else:
                    summary = {}

                # Load config
                config_json = run_dir / 'config.json'
                if config_json.exists():
                    with open(config_json, 'r') as f:
                        config = json.load(f)
                else:
                    config = {}

                self.runs[run_name] = {
                    'epoch_metrics': df_epoch,
                    'step_metrics': df_step,
                    'summary': summary,
                    'config': config,
                    'log_dir': run_dir
                }

                print(f"  Loaded: {run_name}")

        print(f"Total runs loaded: {len(self.runs)}\n")

    def compare_training_curves(self, metric='loss', save_path='comparison_training_curves.png'):
        """Compare training curves across all runs"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for run_name, data in self.runs.items():
            df = data['epoch_metrics']
            if df is None:
                continue

            # Train metrics
            train_col = f'train_{metric}' if f'train_{metric}' in df.columns else None
            if train_col and 'epoch' in df.columns:
                axes[0].plot(df['epoch'], df[train_col], label=run_name, linewidth=2, marker='o', markersize=4)

            # Val metrics
            val_col = f'val_{metric}' if f'val_{metric}' in df.columns else None
            if val_col and 'epoch' in df.columns:
                axes[1].plot(df['epoch'], df[val_col], label=run_name, linewidth=2, marker='s', markersize=4)

        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel(f'Train {metric.upper()}', fontsize=12)
        axes[0].set_title(f'Training {metric.upper()} Comparison', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel(f'Val {metric.upper()}', fontsize=12)
        axes[1].set_title(f'Validation {metric.upper()} Comparison', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison curves to: {save_path}")
        plt.close()

    def compare_all_metrics(self, save_dir='log_analysis'):
        """Compare all available metrics"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

        # Get all available metrics
        all_metrics = set()
        for run_name, data in self.runs.items():
            df = data['epoch_metrics']
            if df is not None:
                for col in df.columns:
                    if col not in ['epoch', 'prefix']:
                        # Extract metric name (remove train_/val_ prefix)
                        metric = col.replace('train_', '').replace('val_', '')
                        all_metrics.add(metric)

        print(f"Found {len(all_metrics)} unique metrics")

        # Plot each metric
        for metric in all_metrics:
            try:
                save_path = save_dir / f'comparison_{metric}.png'
                self.compare_training_curves(metric, save_path)
            except Exception as e:
                print(f"Warning: Could not plot {metric}: {e}")

    def generate_summary_table(self, save_path='log_summary.csv'):
        """Generate summary table comparing all runs"""
        rows = []

        for run_name, data in self.runs.items():
            summary = data['summary']
            config = data['config']
            df_epoch = data['epoch_metrics']

            row = {
                'Run Name': run_name,
                'Model': summary.get('model_name', 'Unknown'),
                'Total Epochs': summary.get('total_epochs', 0),
                'Total Steps': summary.get('total_steps', 0),
                'Batch Size': config.get('batch_size', 'N/A'),
                'Learning Rate': config.get('lr', 'N/A'),
                'Dataset': config.get('dataset', 'N/A')
            }

            # Get final metrics
            if df_epoch is not None and len(df_epoch) > 0:
                last_row = df_epoch.iloc[-1]

                for col in df_epoch.columns:
                    if col not in ['epoch', 'prefix']:
                        row[f'Final {col}'] = f"{last_row[col]:.4f}"

                # Get best metrics
                for col in df_epoch.columns:
                    if 'val_' in col:
                        best_val = df_epoch[col].min()
                        row[f'Best {col}'] = f"{best_val:.4f}"

            rows.append(row)

        df = pd.DataFrame(rows)

        # Save to CSV
        df.to_csv(save_path, index=False)
        print(f"\nSummary table saved to: {save_path}")

        # Print table
        print("\n" + "="*100)
        print("TRAINING RUNS SUMMARY")
        print("="*100)
        print(df.to_string(index=False))
        print("="*100 + "\n")

        return df

    def plot_step_metrics(self, metric='loss_gen', max_steps=None, save_path='step_metrics.png'):
        """Plot step-level metrics"""
        plt.figure(figsize=(14, 6))

        for run_name, data in self.runs.items():
            df = data['step_metrics']
            if df is None or metric not in df.columns:
                continue

            steps = df['step'].values
            values = df[metric].values

            if max_steps:
                mask = steps <= max_steps
                steps = steps[mask]
                values = values[mask]

            # Smooth the curve
            window_size = min(100, len(values) // 10)
            if window_size > 1:
                smoothed = pd.Series(values).rolling(window=window_size, min_periods=1).mean()
                plt.plot(steps, smoothed, label=run_name, linewidth=2, alpha=0.8)
            else:
                plt.plot(steps, values, label=run_name, linewidth=2, alpha=0.8)

        plt.xlabel('Training Step', fontsize=12)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
        plt.title(f'{metric.replace("_", " ").title()} - Step Level', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved step metrics to: {save_path}")
        plt.close()

    def compare_final_performance(self, save_path='final_performance.png'):
        """Compare final performance across runs"""
        metrics_to_compare = ['val_loss', 'val_recon_loss', 'val_vq_loss']
        run_names = []
        metric_values = {m: [] for m in metrics_to_compare}

        for run_name, data in self.runs.items():
            df = data['epoch_metrics']
            if df is None or len(df) == 0:
                continue

            # Get val metrics from last epoch
            val_rows = df[df['prefix'] == 'val']
            if len(val_rows) == 0:
                continue

            last_row = val_rows.iloc[-1]
            run_names.append(run_name)

            for metric in metrics_to_compare:
                if metric in last_row:
                    metric_values[metric].append(last_row[metric])
                else:
                    metric_values[metric].append(np.nan)

        if not run_names:
            print("No validation metrics found")
            return

        # Plot
        x = np.arange(len(run_names))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, metric in enumerate(metrics_to_compare):
            values = metric_values[metric]
            if not any(np.isnan(values)):
                ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())

        ax.set_xlabel('Model Run', fontsize=12)
        ax.set_ylabel('Loss Value', fontsize=12)
        ax.set_title('Final Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(run_names, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved final performance comparison to: {save_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze training logs')
    parser.add_argument('log_dirs', nargs='+', help='Log directories to analyze')
    parser.add_argument('--output-dir', default='log_analysis', help='Output directory')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("Training Log Analysis")
    print("="*80 + "\n")

    # Create analyzer
    analyzer = LogAnalyzer(args.log_dirs)

    if not analyzer.runs:
        print("No valid logs found!")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Generate summary table
    analyzer.generate_summary_table(output_dir / 'summary_table.csv')

    # Compare all metrics
    print("\nGenerating comparison plots...")
    analyzer.compare_all_metrics(output_dir)

    # Plot step metrics
    analyzer.plot_step_metrics('loss_gen', save_path=output_dir / 'step_loss_gen.png')

    # Final performance
    analyzer.compare_final_performance(output_dir / 'final_performance.png')

    print("\n" + "="*80)
    print(f"Analysis complete! Results saved to: {output_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
