import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np


def denormalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """Denormalize tensor from [-1, 1] to [0, 1]"""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def show_images(images, nrow=8, title=None, denorm=True):
    """Display a grid of images"""
    if denorm:
        images = denormalize(images.clone())

    grid = make_grid(images, nrow=nrow, padding=2, normalize=False)
    grid = grid.cpu().numpy().transpose((1, 2, 0))
    grid = np.clip(grid, 0, 1)

    plt.figure(figsize=(12, 12))
    plt.imshow(grid)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def compare_reconstructions(original, reconstructed, nrow=8, save_path=None):
    """Compare original and reconstructed images side by side"""
    batch_size = min(original.size(0), reconstructed.size(0))

    # Interleave original and reconstructed
    comparison = torch.stack([
        original[:batch_size],
        reconstructed[:batch_size]
    ], dim=1).flatten(0, 1)

    if save_path:
        from torchvision.utils import save_image
        save_image(
            comparison,
            save_path,
            nrow=nrow,
            normalize=True,
            value_range=(-1, 1)
        )
    else:
        show_images(comparison, nrow=nrow, title='Original vs Reconstructed')


def plot_training_curves(history, save_path=None):
    """Plot training curves from history dict"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Training Curves')

    # Loss curves
    if 'loss_ae' in history:
        axes[0, 0].plot(history['loss_ae'], label='Generator Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Generator Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

    if 'loss_disc' in history:
        axes[0, 1].plot(history['loss_disc'], label='Discriminator Loss', color='orange')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Discriminator Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

    if 'recon_loss' in history:
        axes[1, 0].plot(history['recon_loss'], label='Reconstruction Loss', color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Reconstruction Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

    if 'vq_loss' in history:
        axes[1, 1].plot(history['vq_loss'], label='VQ Loss', color='red')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('VQ Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def visualize_codebook_usage(indices_list, num_codebooks, codebook_size, save_path=None):
    """Visualize codebook usage distribution"""
    fig, axes = plt.subplots(1, num_codebooks, figsize=(4 * num_codebooks, 4))

    if num_codebooks == 1:
        axes = [axes]

    for i in range(num_codebooks):
        indices = torch.cat([idx[i].flatten() for idx in indices_list])
        counts = torch.bincount(indices, minlength=codebook_size)

        axes[i].bar(range(codebook_size), counts.cpu().numpy())
        axes[i].set_xlabel('Code Index')
        axes[i].set_ylabel('Usage Count')
        axes[i].set_title(f'Codebook {i} Usage')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def interpolate_latents(model, z1, z2, num_steps=10):
    """Interpolate between two latent codes"""
    model.eval()
    alphas = torch.linspace(0, 1, num_steps).to(z1.device)

    interpolations = []
    with torch.no_grad():
        for alpha in alphas:
            z = (1 - alpha) * z1 + alpha * z2
            img = model.decode(z)
            interpolations.append(img)

    return torch.cat(interpolations, dim=0)
