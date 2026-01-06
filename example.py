"""
Example script demonstrating how to use the models
"""

import torch
from torchvision.utils import save_image
from utils import get_data_loaders, compare_reconstructions


def example_vqgan():
    """Example: Train and use VQ-GAN"""
    print("=" * 50)
    print("VQ-GAN Example")
    print("=" * 50)

    from vqgan import VQGAN, Discriminator
    from vqgan.train import VQGANTrainer

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Model
    vqgan = VQGAN(
        in_channels=3,
        hidden_dims=[128, 256, 512],
        latent_dim=256,
        num_embeddings=1024
    )
    discriminator = Discriminator(in_channels=3)

    print(f"VQ-GAN parameters: {sum(p.numel() for p in vqgan.parameters()) / 1e6:.2f}M")

    # Data
    train_loader, val_loader = get_data_loaders(
        dataset_name='cifar10',
        batch_size=4,
        image_size=256
    )

    # Trainer
    trainer = VQGANTrainer(
        vqgan=vqgan,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    # Train for a few epochs
    print("\nTraining for 2 epochs...")
    for epoch in range(2):
        losses = trainer.train_epoch(epoch)
        print(f"Epoch {epoch}: {losses}")

    # Test reconstruction
    print("\nTesting reconstruction...")
    batch = next(iter(val_loader))
    images = batch[0][:4].to(device)

    with torch.no_grad():
        recon, vq_loss, indices = vqgan(images)

    print(f"VQ Loss: {vq_loss.item():.4f}")
    print(f"Indices shape: {indices.shape}")

    # Save results
    compare_reconstructions(
        images,
        recon,
        save_path='vqgan_example.png'
    )
    print("Saved reconstruction to vqgan_example.png")


def example_efficient_vqgan():
    """Example: Train and use Efficient-VQGAN"""
    print("\n" + "=" * 50)
    print("Efficient-VQGAN Example")
    print("=" * 50)

    from efficient_vqgan import EfficientVQGAN, MultiScaleDiscriminator
    from efficient_vqgan.train import EfficientVQGANTrainer

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Model
    model = EfficientVQGAN(
        in_channels=3,
        hidden_dims=[64, 128, 256],
        latent_dim=256,
        num_codebooks=4,
        codebook_size=256
    )
    discriminator = MultiScaleDiscriminator(in_channels=3, num_scales=3)

    print(f"Efficient-VQGAN parameters: {model.count_parameters() / 1e6:.2f}M")

    # Data
    train_loader, val_loader = get_data_loaders(
        dataset_name='cifar10',
        batch_size=8,
        image_size=256
    )

    # Trainer
    trainer = EfficientVQGANTrainer(
        model=model,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    # Train for a few epochs
    print("\nTraining for 2 epochs...")
    for epoch in range(2):
        losses = trainer.train_epoch(epoch)
        print(f"Epoch {epoch}:")
        for k, v in losses.items():
            print(f"  {k}: {v:.4f}")

    # Test reconstruction
    print("\nTesting reconstruction...")
    batch = next(iter(val_loader))
    images = batch[0][:4].to(device)

    with torch.no_grad():
        recon, vq_loss, indices_list = model(images)

    print(f"VQ Loss: {vq_loss.item():.4f}")
    print(f"Number of codebooks: {len(indices_list)}")
    for i, indices in enumerate(indices_list):
        print(f"  Codebook {i} - Indices shape: {indices.shape}")

    # Save results
    compare_reconstructions(
        images,
        recon,
        save_path='efficient_vqgan_example.png'
    )
    print("Saved reconstruction to efficient_vqgan_example.png")


def example_vq_diffusion():
    """Example: Use VQ-Diffusion (requires pretrained VQ-GAN)"""
    print("\n" + "=" * 50)
    print("VQ-Diffusion Example")
    print("=" * 50)

    from vq_diffusion import VQDiffusion
    from vqgan import VQGAN
    import os

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Check for pretrained VQ-GAN
    vqgan_path = 'checkpoints/vqgan_best.pt'
    if not os.path.exists(vqgan_path):
        print(f"Warning: Pretrained VQ-GAN not found at {vqgan_path}")
        print("Please train VQ-GAN first before using VQ-Diffusion")
        return

    # Load VQ-GAN
    print("Loading pretrained VQ-GAN...")
    vqgan = VQGAN(
        in_channels=3,
        hidden_dims=[128, 256, 512],
        latent_dim=256,
        num_embeddings=1024
    )
    checkpoint = torch.load(vqgan_path, map_location=device)
    vqgan.load_state_dict(checkpoint['vqgan_state_dict'])
    vqgan.to(device)
    vqgan.eval()

    # VQ-Diffusion model
    model = VQDiffusion(
        num_embeddings=1024,
        embedding_dim=256,
        image_size=32,  # Latent size
        hidden_dims=[128, 256, 512, 512],
        num_timesteps=100
    )
    model.to(device)

    print(f"VQ-Diffusion parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Generate samples
    print("\nGenerating samples...")
    with torch.no_grad():
        # Sample indices
        indices = model.sample(batch_size=4, device=device)
        print(f"Sampled indices shape: {indices.shape}")

        # Convert to images using VQ-GAN decoder
        indices_flat = indices.reshape(-1)
        z_q = vqgan.quantizer.embedding(indices_flat)

        h = w = int(indices.shape[1] ** 0.5)
        z_q = z_q.reshape(indices.shape[0], h, w, -1)
        z_q = z_q.permute(0, 3, 1, 2)

        images = vqgan.decode(z_q)

    # Save samples
    save_image(
        images,
        'vq_diffusion_samples.png',
        nrow=2,
        normalize=True,
        value_range=(-1, 1)
    )
    print("Saved samples to vq_diffusion_samples.png")


def example_evaluation():
    """Example: Evaluate model quality"""
    print("\n" + "=" * 50)
    print("Evaluation Example")
    print("=" * 50)

    from vqgan import VQGAN
    from utils import (
        evaluate_reconstruction_quality,
        calculate_codebook_usage,
        get_data_loaders
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model
    model = VQGAN(
        in_channels=3,
        hidden_dims=[128, 256, 512],
        latent_dim=256,
        num_embeddings=1024
    ).to(device)

    # Data
    _, val_loader = get_data_loaders(
        dataset_name='cifar10',
        batch_size=32,
        image_size=256
    )

    # Evaluate reconstruction quality
    print("\nEvaluating reconstruction quality...")
    metrics = evaluate_reconstruction_quality(
        model=model,
        dataloader=val_loader,
        device=device,
        num_batches=5
    )

    print("Reconstruction Metrics:")
    print(f"  PSNR: {metrics['psnr']:.2f} ± {metrics['psnr_std']:.2f} dB")
    print(f"  SSIM: {metrics['ssim']:.4f} ± {metrics['ssim_std']:.4f}")

    # Evaluate codebook usage
    print("\nEvaluating codebook usage...")
    usage = calculate_codebook_usage(
        model=model,
        dataloader=val_loader,
        device=device,
        num_batches=10
    )

    print("Codebook Usage:")
    print(f"  Total codes used: {usage['total_codes']} / 1024")
    print(f"  Usage rate: {usage['usage_rate'] * 100:.1f}%")


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("VQ-GAN, VQ-Diffusion, and Efficient-VQGAN Examples")
    print("=" * 70)

    # Run examples
    try:
        example_vqgan()
    except Exception as e:
        print(f"VQ-GAN example failed: {e}")

    try:
        example_efficient_vqgan()
    except Exception as e:
        print(f"Efficient-VQGAN example failed: {e}")

    try:
        example_vq_diffusion()
    except Exception as e:
        print(f"VQ-Diffusion example failed: {e}")

    try:
        example_evaluation()
    except Exception as e:
        print(f"Evaluation example failed: {e}")

    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)


if __name__ == '__main__':
    main()
