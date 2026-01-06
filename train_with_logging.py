"""
Unified training script with comprehensive logging
Trains models and logs all metrics to files, TensorBoard, and CSV
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import argparse
import sys

from utils.logger import TrainingLogger
from utils.data_loader import get_data_loaders


def train_vqgan_with_logging(config):
    """Train VQ-GAN with comprehensive logging"""
    from vqgan.model import VQGAN, Discriminator
    from vqgan.train import LPIPSLoss, hinge_d_loss, weights_init

    # Initialize logger
    logger = TrainingLogger(
        model_name='VQGAN',
        log_dir=config.get('log_dir', 'logs'),
        save_dir=config.get('save_dir', 'checkpoints'),
        use_tensorboard=True
    )

    # Log configuration
    logger.log_config(config)

    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    logger.logger.info(f"Using device: {device}")

    # Data loaders
    logger.logger.info("Loading dataset...")
    train_loader, val_loader = get_data_loaders(
        dataset_name=config.get('dataset', 'cifar10'),
        batch_size=config.get('batch_size', 4),
        image_size=config.get('image_size', 256),
        num_workers=config.get('num_workers', 4)
    )
    logger.logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.logger.info(f"Validation samples: {len(val_loader.dataset)}")

    # Models
    logger.logger.info("Initializing models...")
    vqgan = VQGAN(
        in_channels=3,
        hidden_dims=config.get('hidden_dims', [128, 256, 512]),
        latent_dim=config.get('latent_dim', 256),
        num_embeddings=config.get('num_embeddings', 1024)
    ).to(device)

    discriminator = Discriminator(in_channels=3).to(device)
    discriminator.apply(weights_init)

    # Log model info
    logger.log_model_info(vqgan)

    # Optimizers
    opt_vq = optim.Adam(
        list(vqgan.encoder.parameters()) +
        list(vqgan.decoder.parameters()) +
        list(vqgan.quantizer.parameters()),
        lr=config.get('lr', 4.5e-6),
        betas=(0.5, 0.9)
    )

    opt_disc = optim.Adam(
        discriminator.parameters(),
        lr=config.get('lr', 4.5e-6),
        betas=(0.5, 0.9)
    )

    # Loss
    perceptual_loss = LPIPSLoss().to(device)

    # Training config
    num_epochs = config.get('num_epochs', 10)
    disc_start = config.get('disc_start', 1000)
    disc_weight = config.get('disc_weight', 0.8)
    perceptual_weight = config.get('perceptual_weight', 1.0)

    global_step = 0

    logger.logger.info("\n" + "="*80)
    logger.logger.info("Starting Training")
    logger.logger.info("="*80)

    # Training loop
    for epoch in range(num_epochs):
        vqgan.train()
        discriminator.train()

        epoch_metrics = {
            'loss_gen': 0,
            'loss_disc': 0,
            'recon_loss': 0,
            'perceptual_loss': 0,
            'vq_loss': 0,
            'g_loss': 0
        }

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for batch_idx, batch in enumerate(pbar):
            images = batch[0].to(device)

            # Train Generator
            opt_vq.zero_grad()

            recon, vq_loss, indices = vqgan(images)

            # Reconstruction loss
            recon_loss = torch.abs(images - recon).mean()

            # Perceptual loss
            p_loss = perceptual_loss(images, recon)

            # GAN loss
            if global_step > disc_start:
                logits_fake = discriminator(recon)
                g_loss = -torch.mean(logits_fake)
                disc_factor = disc_weight
            else:
                g_loss = torch.tensor(0.0, device=device)
                disc_factor = 0.0

            # Total generator loss
            loss_gen = recon_loss + perceptual_weight * p_loss + disc_factor * g_loss + vq_loss

            loss_gen.backward()
            opt_vq.step()

            # Train Discriminator
            if global_step > disc_start:
                opt_disc.zero_grad()

                logits_real = discriminator(images.detach())
                logits_fake = discriminator(recon.detach())

                disc_loss = hinge_d_loss(logits_real, logits_fake)
                disc_loss.backward()
                opt_disc.step()
            else:
                disc_loss = torch.tensor(0.0)

            # Log step metrics
            step_metrics = {
                'loss_gen': loss_gen.item(),
                'loss_disc': disc_loss.item(),
                'recon_loss': recon_loss.item(),
                'perceptual_loss': p_loss.item(),
                'vq_loss': vq_loss.item(),
                'g_loss': g_loss.item() if isinstance(g_loss, torch.Tensor) else 0.0
            }

            logger.log_step(global_step, step_metrics, prefix='train')

            # Update epoch metrics
            for key in epoch_metrics:
                epoch_metrics[key] += step_metrics[key]

            # Update progress bar
            pbar.set_postfix({
                'gen': f'{loss_gen.item():.4f}',
                'disc': f'{disc_loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}'
            })

            global_step += 1

            # Log images periodically
            if global_step % 500 == 0:
                logger.log_images('train/original', images[:8], global_step)
                logger.log_images('train/reconstructed', recon[:8], global_step)

        # Average epoch metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= len(train_loader)

        # Log epoch metrics
        logger.log_epoch(epoch, epoch_metrics, prefix='train')

        # Validation
        if val_loader is not None:
            vqgan.eval()
            val_metrics = {
                'loss': 0,
                'recon_loss': 0,
                'vq_loss': 0
            }

            with torch.no_grad():
                for batch in val_loader:
                    images = batch[0].to(device)
                    recon, vq_loss, _ = vqgan(images)

                    recon_loss = torch.abs(images - recon).mean()

                    val_metrics['recon_loss'] += recon_loss.item()
                    val_metrics['vq_loss'] += vq_loss.item()
                    val_metrics['loss'] += (recon_loss + vq_loss).item()

            for key in val_metrics:
                val_metrics[key] /= len(val_loader)

            logger.log_epoch(epoch, val_metrics, prefix='val')

            # Check if best model
            is_best = logger.check_best_metric(val_metrics['loss'], metric_name='val_loss', mode='min')

            # Save checkpoint
            logger.save_checkpoint(
                model=vqgan,
                optimizer=opt_vq,
                discriminator=discriminator,
                disc_optimizer=opt_disc,
                metrics={'train': epoch_metrics, 'val': val_metrics},
                is_best=is_best
            )

    # Close logger
    logger.close()

    return logger.get_summary()


def train_efficient_vqgan_with_logging(config):
    """Train Efficient-VQGAN with comprehensive logging"""
    from efficient_vqgan.model import EfficientVQGAN, MultiScaleDiscriminator
    from efficient_vqgan.train import LPIPSLoss, hinge_d_loss, adopt_weight

    # Initialize logger
    logger = TrainingLogger(
        model_name='Efficient-VQGAN',
        log_dir=config.get('log_dir', 'logs'),
        save_dir=config.get('save_dir', 'checkpoints'),
        use_tensorboard=True
    )

    # Log configuration
    logger.log_config(config)

    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    logger.logger.info(f"Using device: {device}")

    # Data loaders
    logger.logger.info("Loading dataset...")
    train_loader, val_loader = get_data_loaders(
        dataset_name=config.get('dataset', 'cifar10'),
        batch_size=config.get('batch_size', 8),
        image_size=config.get('image_size', 256),
        num_workers=config.get('num_workers', 4)
    )
    logger.logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.logger.info(f"Validation samples: {len(val_loader.dataset)}")

    # Models
    logger.logger.info("Initializing models...")
    model = EfficientVQGAN(
        in_channels=3,
        hidden_dims=config.get('hidden_dims', [64, 128, 256]),
        latent_dim=config.get('latent_dim', 256),
        num_codebooks=config.get('num_codebooks', 4),
        codebook_size=config.get('codebook_size', 256)
    ).to(device)

    discriminator = MultiScaleDiscriminator(in_channels=3, num_scales=3).to(device)

    # Log model info
    logger.log_model_info(model)

    # Optimizers
    opt_ae = optim.Adam(
        list(model.encoder.parameters()) +
        list(model.decoder.parameters()) +
        list(model.quantizer.parameters()),
        lr=config.get('lr', 4.5e-6),
        betas=(0.5, 0.9)
    )

    opt_disc = optim.Adam(
        discriminator.parameters(),
        lr=config.get('lr', 4.5e-6),
        betas=(0.5, 0.9)
    )

    # Schedulers
    scheduler_ae = optim.lr_scheduler.CosineAnnealingLR(
        opt_ae,
        T_max=len(train_loader) * config.get('num_epochs', 10)
    )

    scheduler_disc = optim.lr_scheduler.CosineAnnealingLR(
        opt_disc,
        T_max=len(train_loader) * config.get('num_epochs', 10)
    )

    # Loss
    perceptual_loss = LPIPSLoss().to(device)

    # Training config
    num_epochs = config.get('num_epochs', 10)
    disc_start = config.get('disc_start', 1000)
    disc_weight = config.get('disc_weight', 0.8)
    perceptual_weight = config.get('perceptual_weight', 1.0)
    codebook_weight = config.get('codebook_weight', 1.0)

    global_step = 0

    logger.logger.info("\n" + "="*80)
    logger.logger.info("Starting Training")
    logger.logger.info("="*80)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        discriminator.train()

        epoch_metrics = {
            'loss_ae': 0,
            'loss_disc': 0,
            'recon_loss': 0,
            'perceptual_loss': 0,
            'vq_loss': 0,
            'g_loss': 0
        }

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for batch_idx, batch in enumerate(pbar):
            images = batch[0].to(device)

            # Train Generator
            opt_ae.zero_grad()

            recon, vq_loss, _ = model(images)

            # Reconstruction loss
            recon_loss = torch.abs(images - recon).mean()

            # Perceptual loss
            p_loss = perceptual_loss(images, recon)

            # GAN loss
            if global_step > disc_start:
                logits_fake = discriminator(recon)
                g_loss = -sum([torch.mean(logit) for logit in logits_fake]) / len(logits_fake)
                disc_factor = adopt_weight(disc_weight, global_step, threshold=disc_start)
            else:
                g_loss = torch.tensor(0.0, device=device)
                disc_factor = 0.0

            # Total autoencoder loss
            loss_ae = recon_loss + perceptual_weight * p_loss + disc_factor * g_loss + codebook_weight * vq_loss

            loss_ae.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt_ae.step()
            scheduler_ae.step()

            # Train Discriminator
            if global_step > disc_start:
                opt_disc.zero_grad()

                logits_real = discriminator(images.detach())
                logits_fake = discriminator(recon.detach())

                disc_loss = 0
                for lr, lf in zip(logits_real, logits_fake):
                    disc_loss += hinge_d_loss(lr, lf)
                disc_loss = disc_loss / len(logits_real)

                disc_loss.backward()
                opt_disc.step()
                scheduler_disc.step()
            else:
                disc_loss = torch.tensor(0.0)

            # Log step metrics
            step_metrics = {
                'loss_ae': loss_ae.item(),
                'loss_disc': disc_loss.item(),
                'recon_loss': recon_loss.item(),
                'perceptual_loss': p_loss.item(),
                'vq_loss': vq_loss.item(),
                'g_loss': g_loss.item() if isinstance(g_loss, torch.Tensor) else 0.0,
                'lr': scheduler_ae.get_last_lr()[0]
            }

            logger.log_step(global_step, step_metrics, prefix='train')

            # Update epoch metrics
            for key in epoch_metrics:
                epoch_metrics[key] += step_metrics[key]

            # Update progress bar
            pbar.set_postfix({
                'ae': f'{loss_ae.item():.4f}',
                'disc': f'{disc_loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}'
            })

            global_step += 1

            # Log images periodically
            if global_step % 500 == 0:
                logger.log_images('train/original', images[:8], global_step)
                logger.log_images('train/reconstructed', recon[:8], global_step)

        # Average epoch metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= len(train_loader)

        # Log epoch metrics
        logger.log_epoch(epoch, epoch_metrics, prefix='train')

        # Validation
        if val_loader is not None:
            model.eval()
            val_metrics = {
                'loss': 0,
                'recon_loss': 0,
                'vq_loss': 0
            }

            with torch.no_grad():
                for batch in val_loader:
                    images = batch[0].to(device)
                    recon, vq_loss, _ = model(images)

                    recon_loss = torch.abs(images - recon).mean()

                    val_metrics['recon_loss'] += recon_loss.item()
                    val_metrics['vq_loss'] += vq_loss.item()
                    val_metrics['loss'] += (recon_loss + vq_loss).item()

            for key in val_metrics:
                val_metrics[key] /= len(val_loader)

            logger.log_epoch(epoch, val_metrics, prefix='val')

            # Check if best model
            is_best = logger.check_best_metric(val_metrics['loss'], metric_name='val_loss', mode='min')

            # Save checkpoint
            logger.save_checkpoint(
                model=model,
                optimizer=opt_ae,
                scheduler=scheduler_ae,
                discriminator=discriminator,
                disc_optimizer=opt_disc,
                metrics={'train': epoch_metrics, 'val': val_metrics},
                is_best=is_best
            )

    # Close logger
    logger.close()

    return logger.get_summary()


def main():
    parser = argparse.ArgumentParser(description='Train models with comprehensive logging')
    parser.add_argument('--model', type=str, choices=['vqgan', 'efficient-vqgan', 'both'],
                       default='both', help='Model to train')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')

    args = parser.parse_args()

    # Base config
    config = {
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'dataset': args.dataset,
        'device': args.device if torch.cuda.is_available() else 'cpu',
        'image_size': 256,
        'num_workers': 4,
        'lr': 4.5e-6,
        'disc_start': 1000,
        'disc_weight': 0.8,
        'perceptual_weight': 1.0
    }

    print("\n" + "="*80)
    print("Training with Comprehensive Logging")
    print("="*80)

    # Train models
    if args.model in ['vqgan', 'both']:
        print("\n>>> Training VQ-GAN...")
        vqgan_config = config.copy()
        vqgan_config.update({
            'hidden_dims': [128, 256, 512],
            'latent_dim': 256,
            'num_embeddings': 1024,
            'batch_size': 4
        })
        vqgan_summary = train_vqgan_with_logging(vqgan_config)
        print(f"\nVQ-GAN training completed: {vqgan_summary['run_name']}")

    if args.model in ['efficient-vqgan', 'both']:
        print("\n>>> Training Efficient-VQGAN...")
        eff_config = config.copy()
        eff_config.update({
            'hidden_dims': [64, 128, 256],
            'latent_dim': 256,
            'num_codebooks': 4,
            'codebook_size': 256,
            'batch_size': 8,
            'codebook_weight': 1.0
        })
        eff_summary = train_efficient_vqgan_with_logging(eff_config)
        print(f"\nEfficient-VQGAN training completed: {eff_summary['run_name']}")

    print("\n" + "="*80)
    print("All training completed!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
