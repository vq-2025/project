import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import os
from model import VQGAN, Discriminator


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


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def vanilla_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.softplus(-logits_real))
    loss_fake = torch.mean(F.softplus(logits_fake))
    return 0.5 * (loss_real + loss_fake)


class VQGANTrainer:
    def __init__(
        self,
        vqgan,
        discriminator,
        train_loader,
        val_loader=None,
        lr=4.5e-6,
        disc_start=10000,
        disc_weight=0.8,
        perceptual_weight=1.0,
        device='cuda'
    ):
        self.vqgan = vqgan.to(device)
        self.discriminator = discriminator.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.disc_start = disc_start
        self.disc_weight = disc_weight
        self.perceptual_weight = perceptual_weight

        self.opt_vq = optim.Adam(
            list(vqgan.encoder.parameters()) +
            list(vqgan.decoder.parameters()) +
            list(vqgan.quantizer.parameters()),
            lr=lr, betas=(0.5, 0.9)
        )
        self.opt_disc = optim.Adam(
            discriminator.parameters(),
            lr=lr, betas=(0.5, 0.9)
        )

        self.perceptual_loss = LPIPSLoss().to(device)
        self.global_step = 0

    def train_step(self, batch):
        images = batch[0].to(self.device)

        # Train Generator
        self.opt_vq.zero_grad()

        recon, vq_loss, _ = self.vqgan(images)

        # Reconstruction loss
        recon_loss = torch.abs(images - recon).mean()

        # Perceptual loss
        perceptual_loss = self.perceptual_loss(images, recon)

        # GAN loss
        if self.global_step > self.disc_start:
            logits_fake = self.discriminator(recon)
            g_loss = -torch.mean(logits_fake)
            disc_factor = self.disc_weight
        else:
            g_loss = torch.tensor(0.0, device=self.device)
            disc_factor = 0.0

        # Total generator loss
        loss_gen = recon_loss + \
                   self.perceptual_weight * perceptual_loss + \
                   disc_factor * g_loss + \
                   vq_loss

        loss_gen.backward()
        self.opt_vq.step()

        # Train Discriminator
        if self.global_step > self.disc_start:
            self.opt_disc.zero_grad()

            logits_real = self.discriminator(images.detach())
            logits_fake = self.discriminator(recon.detach())

            disc_loss = hinge_d_loss(logits_real, logits_fake)
            disc_loss.backward()
            self.opt_disc.step()
        else:
            disc_loss = torch.tensor(0.0)

        self.global_step += 1

        return {
            'loss_gen': loss_gen.item(),
            'loss_disc': disc_loss.item(),
            'recon_loss': recon_loss.item(),
            'perceptual_loss': perceptual_loss.item(),
            'vq_loss': vq_loss.item(),
            'g_loss': g_loss.item() if isinstance(g_loss, torch.Tensor) else 0.0
        }

    def train_epoch(self, epoch):
        self.vqgan.train()
        self.discriminator.train()

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch in pbar:
            losses = self.train_step(batch)
            pbar.set_postfix(losses)

        return losses

    def validate(self):
        if self.val_loader is None:
            return {}

        self.vqgan.eval()
        total_loss = 0
        count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch[0].to(self.device)
                recon, vq_loss, _ = self.vqgan(images)
                recon_loss = torch.abs(images - recon).mean()
                total_loss += recon_loss.item()
                count += 1

        return {'val_recon_loss': total_loss / count}

    def save_checkpoint(self, path, epoch):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'vqgan_state_dict': self.vqgan.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'opt_vq_state_dict': self.opt_vq.state_dict(),
            'opt_disc_state_dict': self.opt_disc.state_dict(),
            'global_step': self.global_step
        }, path)


def main():
    # Configuration
    batch_size = 4
    image_size = 256
    num_epochs = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    # Model
    vqgan = VQGAN(
        in_channels=3,
        hidden_dims=[128, 256, 512],
        latent_dim=256,
        num_embeddings=1024
    )

    discriminator = Discriminator(in_channels=3)
    discriminator.apply(weights_init)

    # Trainer
    trainer = VQGANTrainer(
        vqgan=vqgan,
        discriminator=discriminator,
        train_loader=train_loader,
        device=device
    )

    # Training loop
    for epoch in range(num_epochs):
        losses = trainer.train_epoch(epoch)
        print(f'Epoch {epoch}: {losses}')

        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(f'checkpoints/vqgan_epoch_{epoch+1}.pt', epoch)

    print('Training completed!')


if __name__ == '__main__':
    main()
