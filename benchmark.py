"""
Benchmark script to compare VQ-GAN, VQ-Diffusion, and Efficient-VQGAN
Measures objective metrics: PSNR, SSIM, FID, inference time, parameters, etc.
"""

import torch
import torch.nn as nn
import time
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import psutil
import os

from vqgan import VQGAN
from efficient_vqgan import EfficientVQGAN
from utils.metrics import (
    calculate_psnr,
    calculate_ssim,
    InceptionV3FeatureExtractor,
    calculate_fid
)


class ModelBenchmark:
    def __init__(self, model, model_name, device='cuda'):
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.metrics = {}

    def count_parameters(self):
        """Count total and trainable parameters"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.metrics['total_parameters'] = total_params
        self.metrics['trainable_parameters'] = trainable_params
        self.metrics['parameters_M'] = total_params / 1e6

        return total_params, trainable_params

    def measure_inference_time(self, input_size=(1, 3, 256, 256), num_runs=100, warmup=10):
        """Measure inference time"""
        self.model.eval()

        dummy_input = torch.randn(input_size).to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(dummy_input)

        # Measure
        torch.cuda.synchronize() if self.device == 'cuda' else None
        times = []

        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                _ = self.model(dummy_input)
                torch.cuda.synchronize() if self.device == 'cuda' else None
                times.append(time.time() - start)

        avg_time = np.mean(times) * 1000  # Convert to ms
        std_time = np.std(times) * 1000

        self.metrics['inference_time_ms'] = avg_time
        self.metrics['inference_time_std_ms'] = std_time
        self.metrics['throughput_fps'] = 1000 / avg_time

        return avg_time, std_time

    def measure_memory_usage(self, input_size=(1, 3, 256, 256)):
        """Measure GPU memory usage"""
        if self.device != 'cuda':
            self.metrics['memory_allocated_MB'] = 0
            self.metrics['memory_reserved_MB'] = 0
            return 0, 0

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        self.model.eval()
        dummy_input = torch.randn(input_size).to(self.device)

        with torch.no_grad():
            _ = self.model(dummy_input)

        allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
        reserved = torch.cuda.max_memory_reserved() / 1024**2  # MB

        self.metrics['memory_allocated_MB'] = allocated
        self.metrics['memory_reserved_MB'] = reserved

        return allocated, reserved

    def evaluate_reconstruction_quality(self, dataloader, num_batches=50):
        """Evaluate reconstruction quality with PSNR and SSIM"""
        self.model.eval()

        psnr_values = []
        ssim_values = []
        recon_losses = []
        vq_losses = []

        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc=f'Evaluating {self.model_name}', total=num_batches)):
                if i >= num_batches:
                    break

                images = batch[0].to(self.device)
                recon, vq_loss, _ = self.model(images)

                # PSNR
                psnr = calculate_psnr(images, recon)
                psnr_values.append(psnr.item())

                # SSIM
                ssim = calculate_ssim(images, recon)
                ssim_values.append(ssim.item())

                # Reconstruction loss
                recon_loss = torch.abs(images - recon).mean()
                recon_losses.append(recon_loss.item())

                vq_losses.append(vq_loss.item())

        self.metrics['psnr_mean'] = np.mean(psnr_values)
        self.metrics['psnr_std'] = np.std(psnr_values)
        self.metrics['ssim_mean'] = np.mean(ssim_values)
        self.metrics['ssim_std'] = np.std(ssim_values)
        self.metrics['recon_loss_mean'] = np.mean(recon_losses)
        self.metrics['recon_loss_std'] = np.std(recon_losses)
        self.metrics['vq_loss_mean'] = np.mean(vq_losses)
        self.metrics['vq_loss_std'] = np.std(vq_losses)

        return self.metrics

    def calculate_fid_score(self, real_loader, num_batches=50):
        """Calculate FID score"""
        feature_extractor = InceptionV3FeatureExtractor().to(self.device)
        if feature_extractor.model is None:
            print(f"Skipping FID calculation for {self.model_name} (InceptionV3 not available)")
            self.metrics['fid_score'] = -1
            return -1

        feature_extractor.eval()
        self.model.eval()

        real_features = []
        fake_features = []

        with torch.no_grad():
            for i, batch in enumerate(tqdm(real_loader, desc=f'Calculating FID for {self.model_name}', total=num_batches)):
                if i >= num_batches:
                    break

                images = batch[0].to(self.device)

                # Real features
                real_feat = feature_extractor(images)
                real_features.append(real_feat.cpu().numpy())

                # Generate fake images
                recon, _, _ = self.model(images)
                fake_feat = feature_extractor(recon)
                fake_features.append(fake_feat.cpu().numpy())

        real_features = np.concatenate(real_features, axis=0)
        fake_features = np.concatenate(fake_features, axis=0)

        fid = calculate_fid(real_features, fake_features)
        self.metrics['fid_score'] = fid

        return fid

    def evaluate_codebook_usage(self, dataloader, num_batches=50):
        """Evaluate codebook usage"""
        self.model.eval()

        all_indices = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break

                images = batch[0].to(self.device)
                _, _, indices = self.model.encode(images)

                if isinstance(indices, list):
                    # Factorized codebooks
                    all_indices.append([idx.cpu().flatten() for idx in indices])
                else:
                    # Single codebook
                    all_indices.append(indices.cpu().flatten())

        if isinstance(all_indices[0], list):
            # Factorized codebooks
            num_codebooks = len(all_indices[0])
            total_codes_used = 0
            total_possible = 0

            for i in range(num_codebooks):
                indices = torch.cat([batch[i] for batch in all_indices])
                unique = torch.unique(indices)
                total_codes_used += len(unique)

                # Get codebook size
                if hasattr(self.model.quantizer, 'codebook_size'):
                    total_possible += self.model.quantizer.codebook_size

            usage_rate = total_codes_used / total_possible if total_possible > 0 else 0
            self.metrics['codebook_usage_rate'] = usage_rate
            self.metrics['codebook_codes_used'] = total_codes_used
            self.metrics['codebook_total_codes'] = total_possible
        else:
            # Single codebook
            indices = torch.cat(all_indices)
            unique = torch.unique(indices)

            total_codes = self.model.quantizer.num_embeddings
            usage_rate = len(unique) / total_codes

            self.metrics['codebook_usage_rate'] = usage_rate
            self.metrics['codebook_codes_used'] = len(unique)
            self.metrics['codebook_total_codes'] = total_codes

        return self.metrics

    def run_full_benchmark(self, dataloader, input_size=(4, 3, 256, 256)):
        """Run all benchmarks"""
        print(f"\n{'='*60}")
        print(f"Benchmarking {self.model_name}")
        print(f"{'='*60}")

        # Model size
        print("1. Counting parameters...")
        self.count_parameters()
        print(f"   Parameters: {self.metrics['parameters_M']:.2f}M")

        # Inference time
        print("2. Measuring inference time...")
        avg_time, std_time = self.measure_inference_time(input_size)
        print(f"   Inference time: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"   Throughput: {self.metrics['throughput_fps']:.2f} FPS")

        # Memory usage
        print("3. Measuring memory usage...")
        allocated, reserved = self.measure_memory_usage(input_size[:1])
        print(f"   Memory allocated: {allocated:.2f} MB")

        # Reconstruction quality
        print("4. Evaluating reconstruction quality...")
        self.evaluate_reconstruction_quality(dataloader)
        print(f"   PSNR: {self.metrics['psnr_mean']:.2f} ± {self.metrics['psnr_std']:.2f} dB")
        print(f"   SSIM: {self.metrics['ssim_mean']:.4f} ± {self.metrics['ssim_std']:.4f}")

        # Codebook usage
        print("5. Evaluating codebook usage...")
        self.evaluate_codebook_usage(dataloader)
        print(f"   Codebook usage: {self.metrics['codebook_usage_rate']*100:.1f}%")

        # FID score
        print("6. Calculating FID score...")
        fid = self.calculate_fid_score(dataloader, num_batches=30)
        if fid >= 0:
            print(f"   FID score: {fid:.2f}")

        print(f"{'='*60}\n")

        return self.metrics


def compare_models(save_results=True):
    """Compare all models"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Prepare data
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4
    )

    # Initialize models
    print("\nInitializing models...")

    vqgan = VQGAN(
        in_channels=3,
        hidden_dims=[128, 256, 512],
        latent_dim=256,
        num_embeddings=1024
    )

    efficient_vqgan = EfficientVQGAN(
        in_channels=3,
        hidden_dims=[64, 128, 256],
        latent_dim=256,
        num_codebooks=4,
        codebook_size=256
    )

    # Benchmarks
    benchmarks = {}

    # VQ-GAN
    vqgan_benchmark = ModelBenchmark(vqgan, 'VQ-GAN', device)
    benchmarks['VQ-GAN'] = vqgan_benchmark.run_full_benchmark(test_loader)

    # Efficient-VQGAN
    efficient_benchmark = ModelBenchmark(efficient_vqgan, 'Efficient-VQGAN', device)
    benchmarks['Efficient-VQGAN'] = efficient_benchmark.run_full_benchmark(test_loader)

    # Save results
    if save_results:
        os.makedirs('results', exist_ok=True)
        with open('results/benchmark_results.json', 'w') as f:
            json.dump(benchmarks, f, indent=4)
        print("\nResults saved to results/benchmark_results.json")

    return benchmarks


if __name__ == '__main__':
    results = compare_models(save_results=True)

    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)

    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Parameters: {metrics['parameters_M']:.2f}M")
        print(f"  Inference: {metrics['inference_time_ms']:.2f} ms ({metrics['throughput_fps']:.2f} FPS)")
        print(f"  Memory: {metrics['memory_allocated_MB']:.2f} MB")
        print(f"  PSNR: {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f} dB")
        print(f"  SSIM: {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")
        print(f"  FID: {metrics.get('fid_score', -1):.2f}")
        print(f"  Codebook usage: {metrics['codebook_usage_rate']*100:.1f}%")
