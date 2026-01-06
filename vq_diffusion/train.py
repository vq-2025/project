import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torchvision.utils import save_image
from tqdm import tqdm
import os
import sys
sys.path.append('../vqgan')
from model import VQDiffusion


class CodebookDataset(Dataset):
    """Dataset that converts images to codebook indices using a pretrained VQ-GAN"""
    def __init__(self, image_dataset, vqgan_model, device='cuda'):
        self.image_dataset = image_dataset
        self.vqgan = vqgan_model.to(device)
        self.vqgan.eval()
        self.device = device

        # Precompute all indices
        print("Precomputing codebook indices...")
        self.indices = []
        with torch.no_grad():
            for i in tqdm(range(len(image_dataset))):
                img, label = image_dataset[i]
                img = img.unsqueeze(0).to(device)
                _, _, indices = self.vqgan.encode(img)

                # Reshape indices to spatial dimensions
                h = w = int(indices.shape[0] ** 0.5)
                indices = indices.reshape(h, w)
                self.indices.append(indices.cpu())

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.indices[idx]


class VQDiffusionTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        lr=1e-4,
        device='cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_loader) * 100
        )

    def train_step(self, batch):
        self.model.train()
        indices = batch.to(self.device)

        self.optimizer.zero_grad()
        loss = self.model(indices)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def train_epoch(self, epoch):
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')

        for batch in pbar:
            loss = self.train_step(batch)
            total_loss += loss
            pbar.set_postfix({'loss': loss})

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    @torch.no_grad()
    def validate(self):
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0

        for batch in self.val_loader:
            indices = batch.to(self.device)
            loss = self.model(indices)
            total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        return {'val_loss': avg_loss}

    @torch.no_grad()
    def sample(self, num_samples=16):
        self.model.eval()
        samples = self.model.sample(num_samples, self.device)
        return samples

    def save_checkpoint(self, path, epoch):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch']


def indices_to_images(indices, vqgan_model, device='cuda'):
    """Convert codebook indices back to images using VQ-GAN decoder"""
    vqgan_model.eval()
    with torch.no_grad():
        # Get embeddings from indices
        h = w = int(indices.shape[1] ** 0.5)
        indices_flat = indices.reshape(-1)

        # Get quantized vectors
        z_q = vqgan_model.quantizer.embedding(indices_flat)
        z_q = z_q.reshape(indices.shape[0], h, w, -1)
        z_q = z_q.permute(0, 3, 1, 2)

        # Decode
        images = vqgan_model.decode(z_q)

    return images


def main():
    # Configuration
    batch_size = 32
    image_size = 256
    latent_size = 32  # Assuming VQ-GAN downsamples by 8x
    num_epochs = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load pretrained VQ-GAN (you need to train this first)
    print("Loading pretrained VQ-GAN...")
    from vqgan.model import VQGAN
    vqgan = VQGAN(
        in_channels=3,
        hidden_dims=[128, 256, 512],
        latent_dim=256,
        num_embeddings=1024
    )

    # Load checkpoint if exists
    vqgan_ckpt_path = '../vqgan/checkpoints/vqgan_best.pt'
    if os.path.exists(vqgan_ckpt_path):
        checkpoint = torch.load(vqgan_ckpt_path)
        vqgan.load_state_dict(checkpoint['vqgan_state_dict'])
        print("VQ-GAN loaded successfully!")
    else:
        print("Warning: No pretrained VQ-GAN found. You need to train VQ-GAN first!")
        print(f"Expected path: {vqgan_ckpt_path}")
        return

    # Prepare data
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Image dataset
    image_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    # Convert to codebook indices
    codebook_dataset = CodebookDataset(image_dataset, vqgan, device)
    train_loader = DataLoader(
        codebook_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    # Model
    model = VQDiffusion(
        num_embeddings=1024,
        embedding_dim=256,
        image_size=latent_size,
        hidden_dims=[128, 256, 512, 512],
        num_timesteps=100
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Trainer
    trainer = VQDiffusionTrainer(
        model=model,
        train_loader=train_loader,
        lr=1e-4,
        device=device
    )

    # Training loop
    os.makedirs('samples', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(num_epochs):
        avg_loss = trainer.train_epoch(epoch)
        print(f'Epoch {epoch}: avg_loss={avg_loss:.4f}')

        # Sample and save images
        if (epoch + 1) % 10 == 0:
            samples = trainer.sample(num_samples=16)
            images = indices_to_images(samples, vqgan, device)
            save_image(
                images,
                f'samples/epoch_{epoch+1}.png',
                nrow=4,
                normalize=True,
                value_range=(-1, 1)
            )

            trainer.save_checkpoint(f'checkpoints/vqdiffusion_epoch_{epoch+1}.pt', epoch)

    print('Training completed!')


if __name__ == '__main__':
    main()
