# VQ-GAN, VQ-Diffusion, and Efficient-VQGAN Implementation

This repository contains PyTorch implementations of three state-of-the-art generative models:

1. **VQ-GAN** (Vector Quantized Generative Adversarial Network)
2. **VQ-Diffusion** (Vector Quantized Diffusion Model)
3. **Efficient-VQGAN** (Efficient VQ-GAN with factorized codes)

## 🚀 Quick Start (One-Click!)

**가장 빠른 방법:** Jupyter 노트북 하나만 실행하면 모든 것이 완료됩니다!

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. Jupyter 실행
jupyter notebook complete_pipeline.ipynb

# 3. 노트북에서 "Run All" 클릭!
```

**노트북이 자동으로 수행하는 작업:**
- ✅ 데이터 다운로드 및 준비
- 🔥 VQ-GAN 학습 (5 epochs)
- 🔥 Efficient-VQGAN 학습 (5 epochs)
- 📊 성능 벤치마크
- 📈 비교 시각화 (8개 차트)
- 📄 HTML 리포트 생성

**예상 소요 시간:**
- CPU: ~2-3시간
- GPU (RTX 3090): ~20-30분

**생성되는 결과물:**
- `training_comparison.png` - 학습 곡선 비교
- `final_comparison.png` - 성능 종합 비교
- `benchmark_results.csv` - 수치 데이터
- `comparison_report.html` - 인터랙티브 리포트
- `checkpoints/*.pt` - 학습된 모델

## Project Structure

```
.
├── vqgan/                    # VQ-GAN implementation
│   ├── model.py             # Model architecture
│   ├── train.py             # Training script
│   └── __init__.py
├── vq_diffusion/            # VQ-Diffusion implementation
│   ├── model.py             # Model architecture
│   ├── train.py             # Training script
│   └── __init__.py
├── efficient_vqgan/         # Efficient-VQGAN implementation
│   ├── model.py             # Model architecture
│   ├── train.py             # Training script
│   └── __init__.py
├── utils/                   # Shared utilities
│   ├── data_loader.py       # Data loading utilities
│   ├── visualization.py     # Visualization tools
│   ├── metrics.py           # Evaluation metrics
│   └── __init__.py
├── benchmark.py             # Comprehensive benchmarking script
├── visualize_comparison.py  # Visualization generation
├── generate_report.py       # Report generation
├── run_comparison.py        # Unified comparison runner
├── quick_comparison.py      # Quick benchmark demo
├── example.py               # Usage examples
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ninestrings
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Models Overview

### 1. VQ-GAN

VQ-GAN combines a Vector Quantized VAE with a GAN discriminator for high-quality image generation.

**Features:**
- Vector quantization for discrete latent codes
- ResNet-based encoder/decoder
- Attention mechanisms
- PatchGAN discriminator
- Perceptual loss (LPIPS)

**Architecture:**
- Encoder: ResNet blocks with downsampling
- Vector Quantizer: Learnable codebook
- Decoder: ResNet blocks with upsampling
- Discriminator: PatchGAN with multiple scales

### 2. VQ-Diffusion

VQ-Diffusion applies diffusion models to discrete codebook indices from VQ-GAN.

**Features:**
- Discrete diffusion on codebook indices
- U-Net denoising network
- Cosine noise schedule
- Iterative sampling

**Architecture:**
- Token embedding layer
- U-Net with time conditioning
- Multi-head attention
- Residual connections

### 3. Efficient-VQGAN

An efficient variant of VQ-GAN with reduced parameters and improved performance.

**Features:**
- Factorized vector quantization (multiple smaller codebooks)
- Depthwise separable convolutions
- Squeeze-and-Excitation blocks
- Multi-scale discriminator
- Pixel shuffle for upsampling

**Architecture:**
- Efficient encoder with depthwise separable convs
- Factorized quantizer (4 codebooks by default)
- Efficient decoder with pixel shuffle
- Multi-scale discriminator

## Usage

### Training VQ-GAN

```bash
cd vqgan
python train.py
```

Configuration options in `train.py`:
- `batch_size`: Training batch size (default: 4)
- `image_size`: Image resolution (default: 256)
- `num_epochs`: Number of training epochs (default: 100)
- `lr`: Learning rate (default: 4.5e-6)

### Training VQ-Diffusion

**Note:** VQ-Diffusion requires a pretrained VQ-GAN model.

1. First train VQ-GAN and save checkpoint
2. Update the checkpoint path in `vq_diffusion/train.py`
3. Run training:

```bash
cd vq_diffusion
python train.py
```

### Training Efficient-VQGAN

```bash
cd efficient_vqgan
python train.py
```

Configuration options:
- `batch_size`: Training batch size (default: 8)
- `num_codebooks`: Number of factorized codebooks (default: 4)
- `codebook_size`: Size of each codebook (default: 256)

### Training with Comprehensive Logging

For complete logging of all training metrics:

```bash
# Train VQ-GAN with logging
python train_with_logging.py --model vqgan --epochs 10

# Train Efficient-VQGAN with logging
python train_with_logging.py --model efficient-vqgan --epochs 10

# Train both models
python train_with_logging.py --model both --epochs 10
```

**What gets logged:**

1. **Console & File Logs**
   - Training progress with all metrics
   - Model architecture details
   - Configuration parameters
   - Saved to `logs/<model>_<timestamp>/training.log`

2. **TensorBoard**
   - Real-time loss curves
   - Image reconstructions
   - Histograms of activations
   - Learning rate schedules
   - View with: `tensorboard --logdir logs/`

3. **CSV Files**
   - `epoch_metrics.csv`: Per-epoch metrics
   - `step_metrics.csv`: Per-step metrics
   - Easy to load in Excel, Pandas, etc.

4. **Checkpoints**
   - `checkpoint_latest.pt`: Latest model
   - `checkpoint_best.pt`: Best validation loss
   - `checkpoint_epoch_N.pt`: Each epoch

5. **Visualizations**
   - `training_curves.png`: Auto-generated plots
   - Saved after training completes

**Example log output:**
```
[2025-01-06 12:00:00] [INFO] ============================================================
[2025-01-06 12:00:00] [INFO] Epoch 1 [TRAIN]
[2025-01-06 12:00:00] [INFO] ------------------------------------------------------------
[2025-01-06 12:00:00] [INFO]   loss_gen            : 0.425312
[2025-01-06 12:00:00] [INFO]   loss_disc           : 0.123456
[2025-01-06 12:00:00] [INFO]   recon_loss          : 0.234567
[2025-01-06 12:00:00] [INFO]   vq_loss             : 0.045678
```

### Analyzing Training Logs

Compare multiple training runs:

```bash
# Analyze logs from one or more runs
python analyze_logs.py logs/VQGAN_* logs/Efficient-VQGAN_*
```

**Generated analysis:**
- `log_analysis/summary_table.csv`: Comparison table
- `log_analysis/comparison_*.png`: Metric comparisons
- `log_analysis/final_performance.png`: Final metrics
- `log_analysis/step_*.png`: Step-level curves

## Model Comparison

### Quick Comparison

Run a quick benchmark to compare models:

```bash
python quick_comparison.py
```

This will output a summary comparison of VQ-GAN vs Efficient-VQGAN including:
- Model size (parameters)
- Inference speed (FPS)
- Image quality (PSNR, SSIM)
- Memory usage

### Full Benchmark Suite

For comprehensive evaluation with detailed metrics and visualizations:

```bash
python run_comparison.py
```

This will:
1. Run full benchmarks on both models
2. Generate comparison visualizations
3. Create detailed HTML and Markdown reports

**Generated outputs:**
- `results/benchmark_results.json` - Raw benchmark data
- `results/comparison_table.csv` - Comparison table
- `results/*.png` - Visualization charts
- `results/comparison_report.md` - Detailed Markdown report
- `results/comparison_report.html` - Interactive HTML report

### Available Metrics

**Performance Metrics:**
- Parameters count (M)
- Inference time (ms)
- Throughput (FPS)
- GPU memory usage (MB)

**Quality Metrics:**
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- FID (Frechet Inception Distance)
- Reconstruction loss
- VQ loss

**Codebook Metrics:**
- Codebook usage rate
- Number of active codes
- Codebook utilization efficiency

### Visualization Examples

The benchmark suite generates the following visualizations:

1. **Parameter Comparison** - Model size comparison
2. **Inference Speed** - Latency and throughput metrics
3. **Quality Metrics** - PSNR, SSIM, FID comparison
4. **Memory Usage** - GPU memory footprint
5. **Codebook Usage** - Codebook utilization rates
6. **Radar Chart** - Overall performance comparison
7. **Efficiency vs Quality** - Trade-off analysis
8. **Loss Comparison** - Training loss metrics

### Expected Results

Based on the architecture:

| Metric | VQ-GAN | Efficient-VQGAN | Improvement |
|--------|--------|-----------------|-------------|
| Parameters | ~53M | ~20M | 62% reduction |
| Speed | Baseline | 1.5-2x faster | ~50% faster |
| Quality (PSNR) | Baseline | Similar | <1 dB diff |
| Memory | Baseline | 40% less | 40% reduction |

## Custom Datasets

To use your own dataset, modify the data loading in the training scripts:

```python
from utils import get_data_loaders

train_loader, val_loader = get_data_loaders(
    dataset_name='custom',  # Use 'custom' for ImageFolder
    data_dir='path/to/your/data',
    batch_size=32,
    image_size=256
)
```

Expected directory structure for custom datasets:
```
data/
├── train/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── val/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## Evaluation

Evaluate reconstruction quality:

```python
from utils import evaluate_reconstruction_quality

metrics = evaluate_reconstruction_quality(
    model=vqgan,
    dataloader=val_loader,
    device='cuda',
    num_batches=10
)
print(metrics)  # {'psnr': ..., 'ssim': ...}
```

Check codebook usage:

```python
from utils import calculate_codebook_usage

usage = calculate_codebook_usage(
    model=vqgan,
    dataloader=val_loader,
    device='cuda'
)
print(usage)
```

## Visualization

Visualize reconstructions:

```python
from utils import compare_reconstructions

# During training
samples = trainer.sample_reconstructions(num_samples=8)
compare_reconstructions(
    original=images,
    reconstructed=recon,
    save_path='samples/reconstruction.png'
)
```

Plot training curves:

```python
from utils import plot_training_curves

history = {
    'loss_ae': [...],
    'loss_disc': [...],
    'recon_loss': [...],
    'vq_loss': [...]
}

plot_training_curves(history, save_path='training_curves.png')
```

## Model Parameters

### VQ-GAN
- Encoder/Decoder: ~50M parameters
- Discriminator: ~3M parameters
- Total: ~53M parameters

### VQ-Diffusion
- Denoising U-Net: ~100M parameters

### Efficient-VQGAN
- Encoder/Decoder: ~15M parameters (70% reduction)
- Multi-scale Discriminator: ~5M parameters
- Total: ~20M parameters

## Performance Tips

1. **GPU Memory:**
   - Reduce batch size if OOM errors occur
   - Use gradient checkpointing for VQ-Diffusion
   - Enable mixed precision training (fp16)

2. **Training Speed:**
   - Use multiple workers for data loading
   - Enable pin_memory for GPU training
   - Use DistributedDataParallel for multi-GPU

3. **Quality:**
   - Train VQ-GAN for at least 50 epochs
   - Use perceptual loss (LPIPS) for better quality
   - Monitor codebook usage to prevent collapse

## Checkpoints

Models are saved in the `checkpoints/` directory:
- `vqgan_epoch_*.pt`: VQ-GAN checkpoints
- `vqdiffusion_epoch_*.pt`: VQ-Diffusion checkpoints
- `efficient_vqgan_epoch_*.pt`: Efficient-VQGAN checkpoints

Load a checkpoint:

```python
checkpoint = torch.load('checkpoints/vqgan_best.pt')
model.load_state_dict(checkpoint['vqgan_state_dict'])
```

## Citation

If you use this code, please cite the original papers:

```bibtex
@article{esser2021taming,
  title={Taming transformers for high-resolution image synthesis},
  author={Esser, Patrick and Rombach, Robin and Ommer, Bjorn},
  journal={CVPR},
  year={2021}
}

@article{gu2022vector,
  title={Vector quantized diffusion model for text-to-image synthesis},
  author={Gu, Shuyang and Chen, Dong and Bao, Jianmin and Wen, Fang and Zhang, Bo and Chen, Dongdong and Yuan, Lu and Guo, Baining},
  journal={CVPR},
  year={2022}
}
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Troubleshooting

**Q: Training is very slow**
- A: Reduce image size or batch size, use fewer num_workers

**Q: Discriminator loss goes to zero**
- A: Increase disc_start threshold, reduce disc_weight

**Q: Codebook collapse (low usage)**
- A: Increase commitment_cost, add codebook reset mechanism

**Q: LPIPS not available**
- A: Install with `pip install lpips`

**Q: Out of memory**
- A: Reduce batch size or image resolution

## Contact

For questions or issues, please open an issue on GitHub.
