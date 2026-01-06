"""
빠른 데모: VQ-GAN vs Efficient-VQGAN 비교
짧은 학습으로 빠르게 결과를 확인합니다 (2-3분 소요)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 모델 import
from vqgan.model import VQGAN, Discriminator as VQGANDiscriminator
from efficient_vqgan.model import EfficientVQGAN, MultiScaleDiscriminator
from utils.metrics import calculate_psnr, calculate_ssim

print("\n" + "="*80)
print("🚀 VQ-GAN vs Efficient-VQGAN 빠른 데모")
print("="*80 + "\n")

# 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"디바이스: {device}")

CONFIG = {
    'device': device,
    'num_epochs': 2,  # 데모용으로 2 epoch만
    'batch_size': 8,
    'image_size': 64,  # 빠른 학습을 위해 작은 이미지
    'num_workers': 0,  # macOS multiprocessing 이슈 방지
    'lr': 1e-4,
    'disc_start': 100,
}

print(f"설정: {CONFIG['num_epochs']} epochs, 이미지 크기: {CONFIG['image_size']}x{CONFIG['image_size']}")
print(f"배치 크기: {CONFIG['batch_size']}\n")

# 데이터 준비
print("📥 데이터 로딩 중...")
transform = transforms.Compose([
    transforms.Resize(CONFIG['image_size']),
    transforms.CenterCrop(CONFIG['image_size']),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 빠른 데모를 위해 작은 subset만 사용
train_subset = torch.utils.data.Subset(train_dataset, range(1000))
val_subset = torch.utils.data.Subset(val_dataset, range(200))

train_loader = DataLoader(train_subset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'])
val_loader = DataLoader(val_subset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])

print(f"✓ 학습 데이터: {len(train_subset)} 샘플")
print(f"✓ 검증 데이터: {len(val_subset)} 샘플\n")

# 헬퍼 함수
class SimpleLPIPS(nn.Module):
    def forward(self, x, y):
        return torch.mean((x - y) ** 2)

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(torch.relu(1. - logits_real))
    loss_fake = torch.mean(torch.relu(1. + logits_fake))
    return 0.5 * (loss_real + loss_fake)

def train_model(model, discriminator, model_name, train_loader, val_loader, config):
    """모델 학습 함수"""
    print(f"\n{'='*80}")
    print(f"🔥 {model_name} 학습 시작")
    print(f"{'='*80}\n")

    # 옵티마이저
    if hasattr(model, 'encoder'):
        opt_gen = optim.Adam(
            list(model.encoder.parameters()) +
            list(model.decoder.parameters()) +
            list(model.quantizer.parameters()),
            lr=config['lr']
        )
    else:
        opt_gen = optim.Adam(model.parameters(), lr=config['lr'])

    opt_disc = optim.Adam(discriminator.parameters(), lr=config['lr'])

    perceptual_loss = SimpleLPIPS().to(config['device'])

    history = {'train_loss': [], 'val_loss': [], 'val_psnr': [], 'val_ssim': []}
    global_step = 0

    # 학습 루프
    for epoch in range(config['num_epochs']):
        model.train()
        discriminator.train()

        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{config['num_epochs']}")

        for batch in pbar:
            images = batch[0].to(config['device'])

            # Generator
            opt_gen.zero_grad()
            recon, vq_loss, _ = model(images)

            recon_loss = torch.abs(images - recon).mean()
            p_loss = perceptual_loss(images, recon)

            if global_step > config['disc_start']:
                if isinstance(discriminator, MultiScaleDiscriminator):
                    logits_fake = discriminator(recon)
                    g_loss = -sum([torch.mean(logit) for logit in logits_fake]) / len(logits_fake)
                else:
                    logits_fake = discriminator(recon)
                    g_loss = -torch.mean(logits_fake)
            else:
                g_loss = torch.tensor(0.0, device=config['device'])

            loss_gen = recon_loss + p_loss + 0.1 * g_loss + vq_loss
            loss_gen.backward()
            opt_gen.step()

            # Discriminator
            if global_step > config['disc_start']:
                opt_disc.zero_grad()

                if isinstance(discriminator, MultiScaleDiscriminator):
                    logits_real = discriminator(images.detach())
                    logits_fake = discriminator(recon.detach())
                    disc_loss = sum([hinge_d_loss(lr, lf) for lr, lf in zip(logits_real, logits_fake)]) / len(logits_real)
                else:
                    logits_real = discriminator(images.detach())
                    logits_fake = discriminator(recon.detach())
                    disc_loss = hinge_d_loss(logits_real, logits_fake)

                disc_loss.backward()
                opt_disc.step()

            epoch_loss += loss_gen.item()
            global_step += 1

            pbar.set_postfix({'loss': f'{loss_gen.item():.4f}'})

        avg_loss = epoch_loss / len(train_loader)
        history['train_loss'].append(avg_loss)

        # Validation
        model.eval()
        val_loss, val_psnr, val_ssim = 0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch[0].to(config['device'])
                recon, vq_loss, _ = model(images)

                val_loss += (torch.abs(images - recon).mean() + vq_loss).item()
                val_psnr += calculate_psnr(images, recon).item()
                val_ssim += calculate_ssim(images, recon).item()

        val_loss /= len(val_loader)
        val_psnr /= len(val_loader)
        val_ssim /= len(val_loader)

        history['val_loss'].append(val_loss)
        history['val_psnr'].append(val_psnr)
        history['val_ssim'].append(val_ssim)

        print(f"  Epoch {epoch+1}: Loss={val_loss:.4f}, PSNR={val_psnr:.2f} dB, SSIM={val_ssim:.4f}")

    print(f"\n✓ {model_name} 학습 완료!\n")
    return history

# VQ-GAN 학습
print("\n" + "="*80)
print("1/2: VQ-GAN")
print("="*80)

vqgan = VQGAN(
    in_channels=3,
    hidden_dims=[64, 128],  # 간단한 구조
    latent_dim=128,
    num_embeddings=512
).to(device)

disc_vq = VQGANDiscriminator(in_channels=3).to(device)

vqgan_params = sum(p.numel() for p in vqgan.parameters()) / 1e6
print(f"파라미터: {vqgan_params:.2f}M")

vqgan_history = train_model(vqgan, disc_vq, "VQ-GAN", train_loader, val_loader, CONFIG)

# Efficient-VQGAN 학습
print("\n" + "="*80)
print("2/2: Efficient-VQGAN")
print("="*80)

eff_vqgan = EfficientVQGAN(
    in_channels=3,
    hidden_dims=[32, 64],  # 더 작은 구조
    latent_dim=128,
    num_codebooks=4,
    codebook_size=128
).to(device)

disc_eff = MultiScaleDiscriminator(in_channels=3, num_scales=2).to(device)

eff_params = sum(p.numel() for p in eff_vqgan.parameters()) / 1e6
print(f"파라미터: {eff_params:.2f}M")

eff_history = train_model(eff_vqgan, disc_eff, "Efficient-VQGAN", train_loader, val_loader, CONFIG)

# 성능 측정
print("\n" + "="*80)
print("📊 성능 벤치마크")
print("="*80 + "\n")

def measure_speed(model, num_runs=50):
    model.eval()
    dummy = torch.randn(1, 3, CONFIG['image_size'], CONFIG['image_size']).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)

    if device == 'cuda':
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(dummy)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - start)

    return np.mean(times) * 1000

vqgan_time = measure_speed(vqgan)
eff_time = measure_speed(eff_vqgan)

# 결과 정리
results = {
    'VQ-GAN': {
        'params_M': vqgan_params,
        'inference_ms': vqgan_time,
        'fps': 1000 / vqgan_time,
        'final_psnr': vqgan_history['val_psnr'][-1],
        'final_ssim': vqgan_history['val_ssim'][-1],
        'final_loss': vqgan_history['val_loss'][-1]
    },
    'Efficient-VQGAN': {
        'params_M': eff_params,
        'inference_ms': eff_time,
        'fps': 1000 / eff_time,
        'final_psnr': eff_history['val_psnr'][-1],
        'final_ssim': eff_history['val_ssim'][-1],
        'final_loss': eff_history['val_loss'][-1]
    }
}

# 결과 테이블
df = pd.DataFrame(results).T
print(df.to_string())
print()

# 시각화
print("\n📈 결과 시각화 중...")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
epochs = list(range(1, CONFIG['num_epochs'] + 1))

# Loss
axes[0, 0].plot(epochs, vqgan_history['train_loss'], 'o-', label='VQ-GAN', linewidth=2)
axes[0, 0].plot(epochs, eff_history['train_loss'], 's-', label='Efficient-VQGAN', linewidth=2)
axes[0, 0].set_title('Training Loss', fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# PSNR
axes[0, 1].plot(epochs, vqgan_history['val_psnr'], 'o-', label='VQ-GAN', linewidth=2)
axes[0, 1].plot(epochs, eff_history['val_psnr'], 's-', label='Efficient-VQGAN', linewidth=2)
axes[0, 1].set_title('Validation PSNR', fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# SSIM
axes[0, 2].plot(epochs, vqgan_history['val_ssim'], 'o-', label='VQ-GAN', linewidth=2)
axes[0, 2].plot(epochs, eff_history['val_ssim'], 's-', label='Efficient-VQGAN', linewidth=2)
axes[0, 2].set_title('Validation SSIM', fontweight='bold')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Bar charts
models = ['VQ-GAN', 'Efficient-VQGAN']
colors = ['#3498db', '#2ecc71']

# Parameters
params = [results[m]['params_M'] for m in models]
axes[1, 0].bar(models, params, color=colors)
axes[1, 0].set_title('Model Size', fontweight='bold')
axes[1, 0].set_ylabel('Parameters (M)')
for i, v in enumerate(params):
    axes[1, 0].text(i, v, f'{v:.2f}M', ha='center', va='bottom', fontweight='bold')
axes[1, 0].grid(axis='y', alpha=0.3)

# Speed
fps = [results[m]['fps'] for m in models]
axes[1, 1].bar(models, fps, color=colors)
axes[1, 1].set_title('Throughput', fontweight='bold')
axes[1, 1].set_ylabel('FPS')
for i, v in enumerate(fps):
    axes[1, 1].text(i, v, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
axes[1, 1].grid(axis='y', alpha=0.3)

# Quality
psnr = [results[m]['final_psnr'] for m in models]
axes[1, 2].bar(models, psnr, color=colors)
axes[1, 2].set_title('Final PSNR', fontweight='bold')
axes[1, 2].set_ylabel('PSNR (dB)')
for i, v in enumerate(psnr):
    axes[1, 2].text(i, v, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
axes[1, 2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('demo_results.png', dpi=200, bbox_inches='tight')
print("✓ 그래프 저장: demo_results.png")

# 최종 요약
print("\n" + "="*80)
print("🎉 데모 완료!")
print("="*80)
print(f"\n주요 결과:")
print(f"  • VQ-GAN 파라미터:          {vqgan_params:.2f}M")
print(f"  • Efficient-VQGAN 파라미터: {eff_params:.2f}M")
print(f"  • 파라미터 감소:            {(1 - eff_params/vqgan_params)*100:.1f}%")
print(f"  • VQ-GAN FPS:               {results['VQ-GAN']['fps']:.2f}")
print(f"  • Efficient-VQGAN FPS:      {results['Efficient-VQGAN']['fps']:.2f}")
print(f"  • 속도 향상:                {(results['Efficient-VQGAN']['fps']/results['VQ-GAN']['fps']-1)*100:.1f}%")
print(f"  • VQ-GAN PSNR:              {results['VQ-GAN']['final_psnr']:.2f} dB")
print(f"  • Efficient-VQGAN PSNR:     {results['Efficient-VQGAN']['final_psnr']:.2f} dB")
print("\n" + "="*80)
print("\n💡 전체 학습을 위해서는:")
print("   1. complete_pipeline.ipynb 사용 (권장)")
print("   2. 또는 train_with_logging.py --model both --epochs 50")
print("="*80 + "\n")
