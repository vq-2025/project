import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image
from tqdm import tqdm
import os
from model import EfficientVQGAN, MultiScaleDiscriminator


class LPIPSLoss(nn.Module):
    """Perceptual loss using LPIPS"""
    def __init__(self):
        super().__init__()
        try:
            import lpips
            self.loss_fn = lpips.LPIPS(net='vgg')
        except:
            print("LPIPS not available, using MSE instead")
            self.loss_fn = None

    def forward(self, x, y):
        if self.loss_fn is not None:
            return self.loss_fn(x, y).mean()
        else:
            return F.mse_loss(x, y)


def hinge_d_loss(logits_real, logits_fake):
    """Hinge loss for discriminator"""
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def adopt_weight(weight, global_step, threshold=0, value=0.):
    """Gradually increase weight after threshold"""
    if global_step < threshold:
        return value
    return weight


class EfficientVQGANTrainer:
    def __init__(
        self,
        model,
        discriminator,
        train_loader,
        val_loader=None,
        lr=4.5e-6,
        disc_start=10000,
        disc_weight=0.8,
        perceptual_weight=1.0,
        codebook_weight=1.0,
        device='cuda'
    ):
        self.model = model.to(device)
        self.discriminator = discriminator.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.disc_start = disc_start
        self.disc_weight = disc_weight
        self.perceptual_weight = perceptual_weight
        self.codebook_weight = codebook_weight

        # Optimizers
        self.opt_ae = optim.Adam(
            list(model.encoder.parameters()) +
            list(model.decoder.parameters()) +
            list(model.quantizer.parameters()),
            lr=lr, betas=(0.5, 0.9)
        )

        self.opt_disc = optim.Adam(
            discriminator.parameters(),
            lr=lr, betas=(0.5, 0.9)
        )

        # Learning rate schedulers
        self.scheduler_ae = optim.lr_scheduler.CosineAnnealingLR(
            self.opt_ae,
            T_max=len(train_loader) * 100
        )
        self.scheduler_disc = optim.lr_scheduler.CosineAnnealingLR(
            self.opt_disc,
            T_max=len(train_loader) * 100
        )

        self.perceptual_loss = LPIPSLoss().to(device)
        self.global_step = 0

    def train_step(self, batch):
        images = batch[0].to(self.device)

        # ==================== Train Generator ====================
        self.opt_ae.zero_grad()

        recon, vq_loss, _ = self.model(images)

        # Reconstruction loss (L1)
        recon_loss = torch.abs(images - recon).mean()

        # Perceptual loss
        perceptual_loss = self.perceptual_loss(images, recon)

        # Adversarial loss (multi-scale)
        if self.global_step > self.disc_start:
            logits_fake = self.discriminator(recon)
            g_loss = -sum([torch.mean(logit) for logit in logits_fake]) / len(logits_fake)

            disc_factor = adopt_weight(
                self.disc_weight,
                self.global_step,
                threshold=self.disc_start
            )
        else:
            g_loss = torch.tensor(0.0, device=self.device)
            disc_factor = 0.0

        # Total generator loss
        loss_ae = recon_loss + \
                  self.perceptual_weight * perceptual_loss + \
                  disc_factor * g_loss + \
                  self.codebook_weight * vq_loss

        loss_ae.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()),
            max_norm=1.0
        )

        self.opt_ae.step()
        self.scheduler_ae.step()

        # ==================== Train Discriminator ====================
        if self.global_step > self.disc_start:
            self.opt_disc.zero_grad()

            logits_real = self.discriminator(images.detach())
            logits_fake = self.discriminator(recon.detach())

            # Multi-scale discriminator loss
            disc_loss = 0
            for lr, lf in zip(logits_real, logits_fake):
                disc_loss += hinge_d_loss(lr, lf)
            disc_loss = disc_loss / len(logits_real)

            disc_loss.backward()
            self.opt_disc.step()
            self.scheduler_disc.step()
        else:
            disc_loss = torch.tensor(0.0)

        self.global_step += 1

        return {
            'loss_ae': loss_ae.item(),
            'loss_disc': disc_loss.item(),
            'recon_loss': recon_loss.item(),
            'perceptual_loss': perceptual_loss.item(),
            'vq_loss': vq_loss.item(),
            'g_loss': g_loss.item() if isinstance(g_loss, torch.Tensor) else 0.0
        }

    def train_epoch(self, epoch):
        self.model.train()
        self.discriminator.train()

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        epoch_losses = {
            'loss_ae': 0,
            'loss_disc': 0,
            'recon_loss': 0,
            'perceptual_loss': 0,
            'vq_loss': 0,
            'g_loss': 0
        }

        for batch in pbar:
            losses = self.train_step(batch)
            for k, v in losses.items():
                epoch_losses[k] += v

            pbar.set_postfix({k: f'{v:.4f}' for k, v in losses.items()})

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= len(self.train_loader)

        return epoch_losses

    @torch.no_grad()
    def validate(self):
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_recon_loss = 0
        total_vq_loss = 0

        for batch in self.val_loader:
            images = batch[0].to(self.device)
            recon, vq_loss, _ = self.model(images)

            recon_loss = torch.abs(images - recon).mean()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()

        return {
            'val_recon_loss': total_recon_loss / len(self.val_loader),
            'val_vq_loss': total_vq_loss / len(self.val_loader)
        }

    @torch.no_grad()
    def sample_reconstructions(self, num_samples=8):
        """Generate reconstruction samples"""
        self.model.eval()

        # Get a batch from the data loader
        batch = next(iter(self.train_loader))
        images = batch[0][:num_samples].to(self.device)

        recon, _, _ = self.model(images)

        return torch.cat([images, recon], dim=0)

    def save_checkpoint(self, path, epoch):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'opt_ae_state_dict': self.opt_ae.state_dict(),
            'opt_disc_state_dict': self.opt_disc.state_dict(),
            'scheduler_ae_state_dict': self.scheduler_ae.state_dict(),
            'scheduler_disc_state_dict': self.scheduler_disc.state_dict(),
            'global_step': self.global_step
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.opt_ae.load_state_dict(checkpoint['opt_ae_state_dict'])
        self.opt_disc.load_state_dict(checkpoint['opt_disc_state_dict'])
        self.scheduler_ae.load_state_dict(checkpoint['scheduler_ae_state_dict'])
        self.scheduler_disc.load_state_dict(checkpoint['scheduler_disc_state_dict'])
        self.global_step = checkpoint['global_step']
        return checkpoint['epoch']


def main():
    # Configuration
    batch_size = 8
    image_size = 256
    num_epochs = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")

    # Data
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Replace with your dataset
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    val_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Model
    model = EfficientVQGAN(
        in_channels=3,
        hidden_dims=[64, 128, 256],
        latent_dim=256,
        num_codebooks=4,
        codebook_size=256
    )

    discriminator = MultiScaleDiscriminator(in_channels=3, num_scales=3)

    print(f"Model parameters: {model.count_parameters() / 1e6:.2f}M")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()) / 1e6:.2f}M")

    # Trainer
    trainer = EfficientVQGANTrainer(
        model=model,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=4.5e-6,
        device=device
    )

    # Create directories
    os.makedirs('samples', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        train_losses = trainer.train_epoch(epoch)
        print(f'Epoch {epoch}:')
        for k, v in train_losses.items():
            print(f'  {k}: {v:.4f}')

        # Validation
        val_metrics = trainer.validate()
        if val_metrics:
            print(f'  Validation: {val_metrics}')

            # Save best model
            val_loss = val_metrics['val_recon_loss']
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trainer.save_checkpoint('checkpoints/efficient_vqgan_best.pt', epoch)
                print(f'  Saved best model with val_loss={val_loss:.4f}')

        # Sample and save reconstructions
        if (epoch + 1) % 5 == 0:
            samples = trainer.sample_reconstructions(num_samples=8)
            save_image(
                samples,
                f'samples/epoch_{epoch+1}.png',
                nrow=8,
                normalize=True,
                value_range=(-1, 1)
            )

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(f'checkpoints/efficient_vqgan_epoch_{epoch+1}.pt', epoch)

    print('Training completed!')


if __name__ == '__main__':
    main()
