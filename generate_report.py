"""
Generate detailed comparison report in Markdown and HTML formats
"""

import json
from pathlib import Path
from datetime import datetime


def load_results(path='results/benchmark_results.json'):
    """Load benchmark results"""
    with open(path, 'r') as f:
        return json.load(f)


def generate_markdown_report(results, output_path='results/comparison_report.md'):
    """Generate detailed Markdown report"""

    report = f"""# Model Comparison Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report compares the performance of three generative models:
- **VQ-GAN**: Vector Quantized Generative Adversarial Network
- **Efficient-VQGAN**: Efficient variant with factorized codebooks
- **VQ-Diffusion**: Diffusion model on discrete codes (if available)

---

## 1. Model Architecture Comparison

| Metric | VQ-GAN | Efficient-VQGAN | Winner |
|--------|--------|-----------------|--------|
"""

    models = list(results.keys())

    # Parameters comparison
    params = {m: results[m]['parameters_M'] for m in models}
    winner = min(params, key=params.get)
    report += f"| **Parameters (M)** | {params.get('VQ-GAN', 0):.2f} | {params.get('Efficient-VQGAN', 0):.2f} | **{winner}** ✓ |\n"

    # Memory comparison
    memory = {m: results[m]['memory_allocated_MB'] for m in models}
    winner = min(memory, key=memory.get)
    report += f"| **Memory (MB)** | {memory.get('VQ-GAN', 0):.2f} | {memory.get('Efficient-VQGAN', 0):.2f} | **{winner}** ✓ |\n"

    report += "\n### Key Findings:\n\n"
    min_params = min(params.values())
    max_params = max(params.values())
    reduction = (1 - min_params/max_params) * 100
    report += f"- Efficient-VQGAN achieves **{reduction:.1f}% parameter reduction** compared to VQ-GAN\n"
    report += f"- Memory footprint reduced by **{(1 - min(memory.values())/max(memory.values()))*100:.1f}%**\n\n"

    report += "---\n\n## 2. Inference Performance\n\n"
    report += "| Metric | VQ-GAN | Efficient-VQGAN | Winner |\n"
    report += "|--------|--------|-----------------|--------|\n"

    # Inference time
    inf_time = {m: results[m]['inference_time_ms'] for m in models}
    winner = min(inf_time, key=inf_time.get)
    report += f"| **Inference Time (ms)** | {inf_time.get('VQ-GAN', 0):.2f} | {inf_time.get('Efficient-VQGAN', 0):.2f} | **{winner}** ✓ |\n"

    # Throughput
    fps = {m: results[m]['throughput_fps'] for m in models}
    winner = max(fps, key=fps.get)
    report += f"| **Throughput (FPS)** | {fps.get('VQ-GAN', 0):.2f} | {fps.get('Efficient-VQGAN', 0):.2f} | **{winner}** ✓ |\n"

    report += "\n### Key Findings:\n\n"
    speedup = max(fps.values()) / min(fps.values())
    report += f"- Fastest model is **{speedup:.2f}x** faster than slowest\n"
    report += f"- Average inference time: **{sum(inf_time.values())/len(inf_time):.2f} ms**\n\n"

    report += "---\n\n## 3. Image Quality Metrics\n\n"
    report += "| Metric | VQ-GAN | Efficient-VQGAN | Winner |\n"
    report += "|--------|--------|-----------------|--------|\n"

    # PSNR
    psnr = {m: results[m]['psnr_mean'] for m in models}
    winner = max(psnr, key=psnr.get)
    report += f"| **PSNR (dB)** | {psnr.get('VQ-GAN', 0):.2f} ± {results.get('VQ-GAN', {}).get('psnr_std', 0):.2f} | {psnr.get('Efficient-VQGAN', 0):.2f} ± {results.get('Efficient-VQGAN', {}).get('psnr_std', 0):.2f} | **{winner}** ✓ |\n"

    # SSIM
    ssim = {m: results[m]['ssim_mean'] for m in models}
    winner = max(ssim, key=ssim.get)
    report += f"| **SSIM** | {ssim.get('VQ-GAN', 0):.4f} ± {results.get('VQ-GAN', {}).get('ssim_std', 0):.4f} | {ssim.get('Efficient-VQGAN', 0):.4f} ± {results.get('Efficient-VQGAN', {}).get('ssim_std', 0):.4f} | **{winner}** ✓ |\n"

    # FID (if available)
    fid = {m: results[m].get('fid_score', -1) for m in models}
    if any(f > 0 for f in fid.values()):
        winner = min([m for m in models if fid[m] > 0], key=lambda m: fid[m])
        report += f"| **FID Score** | {fid.get('VQ-GAN', -1):.2f} | {fid.get('Efficient-VQGAN', -1):.2f} | **{winner}** ✓ |\n"

    report += "\n### Key Findings:\n\n"
    report += f"- Best PSNR: **{max(psnr.values()):.2f} dB** ({max(psnr, key=psnr.get)})\n"
    report += f"- Best SSIM: **{max(ssim.values()):.4f}** ({max(ssim, key=ssim.get)})\n"
    psnr_diff = max(psnr.values()) - min(psnr.values())
    report += f"- PSNR difference between models: **{psnr_diff:.2f} dB**\n\n"

    report += "---\n\n## 4. Loss Metrics\n\n"
    report += "| Metric | VQ-GAN | Efficient-VQGAN |\n"
    report += "|--------|--------|------------------|\n"

    # Reconstruction loss
    recon = {m: results[m]['recon_loss_mean'] for m in models}
    report += f"| **Reconstruction Loss** | {recon.get('VQ-GAN', 0):.4f} ± {results.get('VQ-GAN', {}).get('recon_loss_std', 0):.4f} | {recon.get('Efficient-VQGAN', 0):.4f} ± {results.get('Efficient-VQGAN', {}).get('recon_loss_std', 0):.4f} |\n"

    # VQ loss
    vq = {m: results[m]['vq_loss_mean'] for m in models}
    report += f"| **VQ Loss** | {vq.get('VQ-GAN', 0):.4f} ± {results.get('VQ-GAN', {}).get('vq_loss_std', 0):.4f} | {vq.get('Efficient-VQGAN', 0):.4f} ± {results.get('Efficient-VQGAN', {}).get('vq_loss_std', 0):.4f} |\n"

    report += "\n---\n\n## 5. Codebook Analysis\n\n"
    report += "| Metric | VQ-GAN | Efficient-VQGAN |\n"
    report += "|--------|--------|------------------|\n"

    # Codebook usage
    usage = {m: results[m]['codebook_usage_rate'] * 100 for m in models}
    report += f"| **Codebook Usage (%)** | {usage.get('VQ-GAN', 0):.1f}% | {usage.get('Efficient-VQGAN', 0):.1f}% |\n"

    codes_used = {m: results[m]['codebook_codes_used'] for m in models}
    total_codes = {m: results[m]['codebook_total_codes'] for m in models}
    report += f"| **Codes Used / Total** | {codes_used.get('VQ-GAN', 0)} / {total_codes.get('VQ-GAN', 0)} | {codes_used.get('Efficient-VQGAN', 0)} / {total_codes.get('Efficient-VQGAN', 0)} |\n"

    report += "\n### Key Findings:\n\n"
    report += f"- VQ-GAN uses a **single large codebook** ({total_codes.get('VQ-GAN', 0)} codes)\n"
    report += f"- Efficient-VQGAN uses **factorized codebooks** ({total_codes.get('Efficient-VQGAN', 0)} codes total)\n"
    report += f"- Higher usage rate indicates better codebook utilization\n\n"

    report += "---\n\n## 6. Overall Recommendations\n\n"

    report += "### Choose **VQ-GAN** if:\n"
    report += "- Maximum image quality is the priority\n"
    report += "- Computational resources are abundant\n"
    report += "- Single codebook simplicity is preferred\n\n"

    report += "### Choose **Efficient-VQGAN** if:\n"
    report += "- Model size and speed are important\n"
    report += "- Deploying on resource-constrained devices\n"
    report += "- Similar quality with better efficiency is acceptable\n\n"

    report += "---\n\n## 7. Detailed Metrics Table\n\n"
    report += "| Metric | VQ-GAN | Efficient-VQGAN |\n"
    report += "|--------|--------|------------------|\n"

    for model in ['VQ-GAN', 'Efficient-VQGAN']:
        if model not in results:
            continue

    # Add all metrics
    metrics_to_show = [
        ('Parameters (M)', 'parameters_M', '.2f'),
        ('Inference Time (ms)', 'inference_time_ms', '.2f'),
        ('Throughput (FPS)', 'throughput_fps', '.2f'),
        ('Memory (MB)', 'memory_allocated_MB', '.2f'),
        ('PSNR (dB)', 'psnr_mean', '.2f'),
        ('SSIM', 'ssim_mean', '.4f'),
        ('Recon Loss', 'recon_loss_mean', '.4f'),
        ('VQ Loss', 'vq_loss_mean', '.4f'),
        ('Codebook Usage (%)', 'codebook_usage_rate', '.1%'),
    ]

    for label, key, fmt in metrics_to_show:
        vqgan_val = results.get('VQ-GAN', {}).get(key, 0)
        eff_val = results.get('Efficient-VQGAN', {}).get(key, 0)
        report += f"| {label} | {vqgan_val:{fmt}} | {eff_val:{fmt}} |\n"

    report += "\n---\n\n## 8. Visualizations\n\n"
    report += "See the `results/` directory for detailed visualization plots:\n\n"
    report += "1. `parameters_comparison.png` - Model size comparison\n"
    report += "2. `inference_speed_comparison.png` - Speed metrics\n"
    report += "3. `quality_metrics_comparison.png` - PSNR, SSIM, FID\n"
    report += "4. `memory_usage_comparison.png` - Memory footprint\n"
    report += "5. `codebook_usage_comparison.png` - Codebook utilization\n"
    report += "6. `radar_comparison.png` - Overall performance radar chart\n"
    report += "7. `efficiency_vs_quality.png` - Efficiency-quality trade-off\n"
    report += "8. `loss_comparison.png` - Training loss comparison\n\n"

    report += "---\n\n"
    report += f"*Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*\n"

    # Save report
    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)

    return report


def generate_html_report(markdown_report, output_path='results/comparison_report.html'):
    """Convert Markdown report to HTML"""

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Comparison Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
        }}
        h3 {{
            color: #7f8c8d;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            box-shadow: 0 2px 3px rgba(0,0,0,0.1);
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        tr:hover {{
            background-color: #e8f4f8;
        }}
        .winner {{
            background-color: #d4edda;
            font-weight: bold;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        ul {{
            line-height: 1.8;
        }}
        hr {{
            border: none;
            border-top: 2px solid #ecf0f1;
            margin: 30px 0;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-style: italic;
            text-align: right;
        }}
        .key-findings {{
            background-color: #fff3cd;
            padding: 15px;
            border-left: 4px solid #ffc107;
            margin: 15px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
"""

    # Simple markdown to HTML conversion
    html_content = markdown_report
    html_content = html_content.replace('**', '<strong>').replace('**', '</strong>')
    html_content = html_content.replace('---', '<hr>')

    # Convert headers
    for i in range(6, 0, -1):
        html_content = html_content.replace('#' * i + ' ', f'<h{i}>').replace('\n', f'</h{i}>\n', 1)

    # Convert lists
    lines = html_content.split('\n')
    html_lines = []
    in_list = False

    for line in lines:
        if line.strip().startswith('- '):
            if not in_list:
                html_lines.append('<ul>')
                in_list = True
            html_lines.append(f'<li>{line.strip()[2:]}</li>')
        else:
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            html_lines.append(line)

    html_content = '\n'.join(html_lines)

    html_template += html_content
    html_template += """
    </div>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html_template)


def generate_report(results_path='results/benchmark_results.json'):
    """Generate both Markdown and HTML reports"""
    print("Loading benchmark results...")
    results = load_results(results_path)

    print("Generating Markdown report...")
    markdown_report = generate_markdown_report(results)
    print("✓ Saved: results/comparison_report.md")

    print("Generating HTML report...")
    generate_html_report(markdown_report)
    print("✓ Saved: results/comparison_report.html")

    print("\nReport generation completed!")


if __name__ == '__main__':
    generate_report()
