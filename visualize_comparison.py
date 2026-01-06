"""
Visualization script to compare model performances
Creates comprehensive visualizations from benchmark results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_benchmark_results(path='results/benchmark_results.json'):
    """Load benchmark results from JSON file"""
    with open(path, 'r') as f:
        return json.load(f)


def create_comparison_table(results):
    """Create a comprehensive comparison table"""
    data = []

    for model_name, metrics in results.items():
        row = {
            'Model': model_name,
            'Parameters (M)': f"{metrics['parameters_M']:.2f}",
            'Inference (ms)': f"{metrics['inference_time_ms']:.2f}",
            'FPS': f"{metrics['throughput_fps']:.2f}",
            'Memory (MB)': f"{metrics['memory_allocated_MB']:.2f}",
            'PSNR (dB)': f"{metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f}",
            'SSIM': f"{metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}",
            'FID': f"{metrics.get('fid_score', -1):.2f}",
            'Recon Loss': f"{metrics['recon_loss_mean']:.4f}",
            'VQ Loss': f"{metrics['vq_loss_mean']:.4f}",
            'Codebook Usage': f"{metrics['codebook_usage_rate']*100:.1f}%"
        }
        data.append(row)

    df = pd.DataFrame(data)
    return df


def plot_parameter_comparison(results, save_path='results/parameters_comparison.png'):
    """Plot parameter count comparison"""
    models = list(results.keys())
    params = [results[m]['parameters_M'] for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, params, color=['#3498db', '#2ecc71', '#e74c3c'][:len(models)])

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}M',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Parameters (Millions)', fontsize=12)
    ax.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_inference_speed_comparison(results, save_path='results/inference_speed_comparison.png'):
    """Plot inference speed comparison"""
    models = list(results.keys())
    times = [results[m]['inference_time_ms'] for m in models]
    fps = [results[m]['throughput_fps'] for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Inference time
    bars1 = ax1.bar(models, times, color=['#3498db', '#2ecc71', '#e74c3c'][:len(models)])
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}ms',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_ylabel('Inference Time (ms)', fontsize=12)
    ax1.set_title('Inference Time (Lower is Better)', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Throughput (FPS)
    bars2 = ax2.bar(models, fps, color=['#3498db', '#2ecc71', '#e74c3c'][:len(models)])
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2.set_ylabel('Throughput (FPS)', fontsize=12)
    ax2.set_title('Throughput (Higher is Better)', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_quality_metrics(results, save_path='results/quality_metrics_comparison.png'):
    """Plot quality metrics comparison"""
    models = list(results.keys())
    psnr = [results[m]['psnr_mean'] for m in models]
    ssim = [results[m]['ssim_mean'] for m in models]
    fid = [results[m].get('fid_score', 0) for m in models]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    # PSNR
    bars1 = ax1.bar(models, psnr, color=['#3498db', '#2ecc71', '#e74c3c'][:len(models)])
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_ylabel('PSNR (dB)', fontsize=12)
    ax1.set_title('Peak Signal-to-Noise Ratio (Higher is Better)', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # SSIM
    bars2 = ax2.bar(models, ssim, color=['#3498db', '#2ecc71', '#e74c3c'][:len(models)])
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2.set_ylabel('SSIM', fontsize=12)
    ax2.set_title('Structural Similarity Index (Higher is Better)', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # FID (if available)
    if any(f > 0 for f in fid):
        bars3 = ax3.bar(models, fid, color=['#3498db', '#2ecc71', '#e74c3c'][:len(models)])
        for bar in bars3:
            height = bar.get_height()
            if height > 0:
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax3.set_ylabel('FID Score', fontsize=12)
        ax3.set_title('Frechet Inception Distance (Lower is Better)', fontsize=12, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'FID Not Available', ha='center', va='center', fontsize=14)
        ax3.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_memory_usage(results, save_path='results/memory_usage_comparison.png'):
    """Plot memory usage comparison"""
    models = list(results.keys())
    memory = [results[m]['memory_allocated_MB'] for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, memory, color=['#3498db', '#2ecc71', '#e74c3c'][:len(models)])

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f} MB',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Memory Usage (MB)', fontsize=12)
    ax.set_title('GPU Memory Usage Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_codebook_usage(results, save_path='results/codebook_usage_comparison.png'):
    """Plot codebook usage comparison"""
    models = list(results.keys())
    usage = [results[m]['codebook_usage_rate'] * 100 for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, usage, color=['#3498db', '#2ecc71', '#e74c3c'][:len(models)])

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Codebook Usage (%)', fontsize=12)
    ax.set_title('Codebook Usage Rate Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_radar_chart(results, save_path='results/radar_comparison.png'):
    """Create radar chart for overall comparison"""
    models = list(results.keys())

    # Normalize metrics (0-1 scale, higher is better)
    def normalize(values, inverse=False):
        vmin, vmax = min(values), max(values)
        if vmax == vmin:
            return [0.5] * len(values)
        if inverse:
            return [(vmax - v) / (vmax - vmin) for v in values]
        return [(v - vmin) / (vmax - vmin) for v in values]

    # Metrics (all normalized to 0-1, higher is better)
    metrics = {
        'Speed': normalize([results[m]['throughput_fps'] for m in models], inverse=False),
        'PSNR': normalize([results[m]['psnr_mean'] for m in models], inverse=False),
        'SSIM': normalize([results[m]['ssim_mean'] for m in models], inverse=False),
        'Efficiency': normalize([results[m]['parameters_M'] for m in models], inverse=True),
        'Memory': normalize([results[m]['memory_allocated_MB'] for m in models], inverse=True),
        'Codebook': [results[m]['codebook_usage_rate'] for m in models]
    }

    categories = list(metrics.keys())
    N = len(categories)

    # Compute angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    for i, model in enumerate(models):
        values = [metrics[cat][i] for cat in categories]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    plt.title('Overall Model Comparison\n(Normalized Metrics)',
              fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_efficiency_vs_quality(results, save_path='results/efficiency_vs_quality.png'):
    """Plot efficiency vs quality trade-off"""
    models = list(results.keys())
    params = [results[m]['parameters_M'] for m in models]
    psnr = [results[m]['psnr_mean'] for m in models]
    inference = [results[m]['inference_time_ms'] for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Parameters vs PSNR
    colors = ['#3498db', '#2ecc71', '#e74c3c'][:len(models)]
    for i, model in enumerate(models):
        ax1.scatter(params[i], psnr[i], s=300, alpha=0.6,
                   color=colors[i], label=model, edgecolors='black', linewidth=2)
        ax1.annotate(model, (params[i], psnr[i]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=10, fontweight='bold')

    ax1.set_xlabel('Model Size (M Parameters)', fontsize=12)
    ax1.set_ylabel('PSNR (dB)', fontsize=12)
    ax1.set_title('Model Size vs Quality', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Inference time vs PSNR
    for i, model in enumerate(models):
        ax2.scatter(inference[i], psnr[i], s=300, alpha=0.6,
                   color=colors[i], label=model, edgecolors='black', linewidth=2)
        ax2.annotate(model, (inference[i], psnr[i]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=10, fontweight='bold')

    ax2.set_xlabel('Inference Time (ms)', fontsize=12)
    ax2.set_ylabel('PSNR (dB)', fontsize=12)
    ax2.set_title('Speed vs Quality', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_loss_comparison(results, save_path='results/loss_comparison.png'):
    """Plot reconstruction and VQ loss comparison"""
    models = list(results.keys())
    recon_loss = [results[m]['recon_loss_mean'] for m in models]
    vq_loss = [results[m]['vq_loss_mean'] for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Reconstruction loss
    bars1 = ax1.bar(models, recon_loss, color=['#3498db', '#2ecc71', '#e74c3c'][:len(models)])
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_ylabel('Reconstruction Loss', fontsize=12)
    ax1.set_title('Reconstruction Loss (Lower is Better)', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # VQ loss
    bars2 = ax2.bar(models, vq_loss, color=['#3498db', '#2ecc71', '#e74c3c'][:len(models)])
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2.set_ylabel('VQ Loss', fontsize=12)
    ax2.set_title('Vector Quantization Loss (Lower is Better)', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def create_all_visualizations(results_path='results/benchmark_results.json'):
    """Create all visualization plots"""
    print("\n" + "="*80)
    print("Creating Comparison Visualizations")
    print("="*80 + "\n")

    # Load results
    results = load_benchmark_results(results_path)

    # Create results directory
    Path('results').mkdir(exist_ok=True)

    # Create comparison table
    print("Creating comparison table...")
    df = create_comparison_table(results)
    print("\n" + df.to_string(index=False))
    df.to_csv('results/comparison_table.csv', index=False)
    print("\nSaved: results/comparison_table.csv")

    # Create visualizations
    print("\nGenerating visualizations...")
    plot_parameter_comparison(results)
    plot_inference_speed_comparison(results)
    plot_quality_metrics(results)
    plot_memory_usage(results)
    plot_codebook_usage(results)
    plot_radar_chart(results)
    plot_efficiency_vs_quality(results)
    plot_loss_comparison(results)

    print("\n" + "="*80)
    print("All visualizations created successfully!")
    print("Check the 'results/' directory for all plots and tables.")
    print("="*80 + "\n")


if __name__ == '__main__':
    create_all_visualizations()
