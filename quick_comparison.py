"""
Quick comparison demo - runs a lightweight benchmark for testing
Uses fewer batches and smaller data for faster results
"""

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import time
import json
from pathlib import Path

from vqgan import VQGAN
from efficient_vqgan import EfficientVQGAN
from utils.metrics import calculate_psnr, calculate_ssim


def quick_benchmark():
    """Run a quick benchmark with minimal data"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*70}")
    print(f"Quick Model Comparison Demo")
    print(f"Device: {device}")
    print(f"{'='*70}\n")

    # Prepare minimal dataset
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)

    # Initialize models
    print("Initializing models...\n")

    vqgan = VQGAN(
        in_channels=3,
        hidden_dims=[128, 256, 512],
        latent_dim=256,
        num_embeddings=1024
    ).to(device)

    efficient_vqgan = EfficientVQGAN(
        in_channels=3,
        hidden_dims=[64, 128, 256],
        latent_dim=256,
        num_codebooks=4,
        codebook_size=256
    ).to(device)

    models = {
        'VQ-GAN': vqgan,
        'Efficient-VQGAN': efficient_vqgan
    }

    results = {}

    for model_name, model in models.items():
        print(f"\n{'-'*70}")
        print(f"Testing: {model_name}")
        print(f"{'-'*70}")

        model.eval()

        # Count parameters
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Parameters: {params:.2f}M")

        # Measure inference time (warmup + timing)
        dummy = torch.randn(1, 3, 256, 256).to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy)

        # Timing
        torch.cuda.synchronize() if device == 'cuda' else None
        times = []
        with torch.no_grad():
            for _ in range(20):
                start = time.time()
                _ = model(dummy)
                torch.cuda.synchronize() if device == 'cuda' else None
                times.append(time.time() - start)

        avg_time = sum(times) / len(times) * 1000  # ms
        fps = 1000 / avg_time
        print(f"Inference: {avg_time:.2f} ms ({fps:.2f} FPS)")

        # Test on real data (5 batches)
        psnr_vals = []
        ssim_vals = []

        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= 5:
                    break

                images = batch[0].to(device)
                recon, vq_loss, _ = model(images)

                psnr = calculate_psnr(images, recon)
                ssim = calculate_ssim(images, recon)

                psnr_vals.append(psnr.item())
                ssim_vals.append(ssim.item())

        avg_psnr = sum(psnr_vals) / len(psnr_vals)
        avg_ssim = sum(ssim_vals) / len(ssim_vals)

        print(f"PSNR: {avg_psnr:.2f} dB")
        print(f"SSIM: {avg_ssim:.4f}")

        # Memory
        if device == 'cuda':
            memory = torch.cuda.max_memory_allocated() / 1024**2
            print(f"Memory: {memory:.2f} MB")
        else:
            memory = 0

        results[model_name] = {
            'parameters_M': params,
            'inference_ms': avg_time,
            'fps': fps,
            'psnr': avg_psnr,
            'ssim': avg_ssim,
            'memory_MB': memory
        }

    # Print comparison
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}\n")

    print(f"{'Metric':<20} {'VQ-GAN':<20} {'Efficient-VQGAN':<20} {'Winner'}")
    print(f"{'-'*70}")

    # Parameters
    vq_params = results['VQ-GAN']['parameters_M']
    eff_params = results['Efficient-VQGAN']['parameters_M']
    winner = 'Efficient-VQGAN ✓' if eff_params < vq_params else 'VQ-GAN ✓'
    print(f"{'Parameters (M)':<20} {vq_params:<20.2f} {eff_params:<20.2f} {winner}")

    # Inference
    vq_inf = results['VQ-GAN']['inference_ms']
    eff_inf = results['Efficient-VQGAN']['inference_ms']
    winner = 'Efficient-VQGAN ✓' if eff_inf < vq_inf else 'VQ-GAN ✓'
    print(f"{'Inference (ms)':<20} {vq_inf:<20.2f} {eff_inf:<20.2f} {winner}")

    # FPS
    vq_fps = results['VQ-GAN']['fps']
    eff_fps = results['Efficient-VQGAN']['fps']
    winner = 'Efficient-VQGAN ✓' if eff_fps > vq_fps else 'VQ-GAN ✓'
    print(f"{'Throughput (FPS)':<20} {vq_fps:<20.2f} {eff_fps:<20.2f} {winner}")

    # PSNR
    vq_psnr = results['VQ-GAN']['psnr']
    eff_psnr = results['Efficient-VQGAN']['psnr']
    winner = 'Efficient-VQGAN ✓' if eff_psnr > vq_psnr else 'VQ-GAN ✓'
    print(f"{'PSNR (dB)':<20} {vq_psnr:<20.2f} {eff_psnr:<20.2f} {winner}")

    # SSIM
    vq_ssim = results['VQ-GAN']['ssim']
    eff_ssim = results['Efficient-VQGAN']['ssim']
    winner = 'Efficient-VQGAN ✓' if eff_ssim > vq_ssim else 'VQ-GAN ✓'
    print(f"{'SSIM':<20} {vq_ssim:<20.4f} {eff_ssim:<20.4f} {winner}")

    print(f"{'-'*70}\n")

    # Key insights
    print("KEY INSIGHTS:")
    param_reduction = (1 - eff_params / vq_params) * 100
    print(f"  • Efficient-VQGAN has {param_reduction:.1f}% fewer parameters")

    speed_improvement = (eff_fps / vq_fps - 1) * 100
    if speed_improvement > 0:
        print(f"  • Efficient-VQGAN is {speed_improvement:.1f}% faster")
    else:
        print(f"  • VQ-GAN is {-speed_improvement:.1f}% faster")

    quality_diff = eff_psnr - vq_psnr
    if abs(quality_diff) < 1.0:
        print(f"  • Quality is comparable (PSNR difference: {abs(quality_diff):.2f} dB)")
    elif quality_diff > 0:
        print(f"  • Efficient-VQGAN has {quality_diff:.2f} dB better PSNR")
    else:
        print(f"  • VQ-GAN has {abs(quality_diff):.2f} dB better PSNR")

    print(f"\n{'='*70}\n")

    # Save results
    Path('results').mkdir(exist_ok=True)
    with open('results/quick_comparison.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Results saved to: results/quick_comparison.json\n")

    return results


if __name__ == '__main__':
    quick_benchmark()
