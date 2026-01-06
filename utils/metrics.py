import torch
import numpy as np
from scipy import linalg


def calculate_psnr(img1, img2):
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(2.0 / torch.sqrt(mse))


def calculate_ssim(img1, img2, window_size=11):
    """Calculate Structural Similarity Index for RGB images"""
    from torch.nn.functional import conv2d

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Get number of channels
    channels = img1.size(1)

    # Create Gaussian window
    sigma = 1.5
    gauss = torch.Tensor([
        np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
        for x in range(window_size)
    ])
    gauss = gauss / gauss.sum()
    window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0)
    
    # Expand window for all channels
    window = window.expand(channels, 1, window_size, window_size).contiguous()
    window = window.to(img1.device)

    # Calculate SSIM for each channel separately using groups
    mu1 = conv2d(img1, window, padding=window_size // 2, groups=channels)
    mu2 = conv2d(img2, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = conv2d(img1 * img1, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = conv2d(img2 * img2, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = conv2d(img1 * img2, window, padding=window_size // 2, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


class InceptionV3FeatureExtractor(torch.nn.Module):
    """Extract features from InceptionV3 for FID calculation"""
    def __init__(self):
        super().__init__()
        try:
            from torchvision.models import inception_v3, Inception_V3_Weights
            inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
            self.model = torch.nn.Sequential(
                inception.Conv2d_1a_3x3,
                inception.Conv2d_2a_3x3,
                inception.Conv2d_2b_3x3,
                torch.nn.MaxPool2d(kernel_size=3, stride=2),
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                torch.nn.MaxPool2d(kernel_size=3, stride=2),
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
            )
            self.model.eval()
        except:
            print("Warning: Could not load InceptionV3. FID calculation will not work.")
            self.model = None

    def forward(self, x):
        if self.model is None:
            return None

        # Resize to 299x299 for InceptionV3
        x = torch.nn.functional.interpolate(
            x, size=(299, 299), mode='bilinear', align_corners=False
        )
        features = self.model(x)
        return features.squeeze(-1).squeeze(-1)


def calculate_fid(real_features, fake_features):
    """Calculate Frechet Inception Distance"""
    mu1 = np.mean(real_features, axis=0)
    sigma1 = np.cov(real_features, rowvar=False)

    mu2 = np.mean(fake_features, axis=0)
    sigma2 = np.cov(fake_features, rowvar=False)

    # Calculate FID
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


@torch.no_grad()
def evaluate_reconstruction_quality(model, dataloader, device='cuda', num_batches=10):
    """Evaluate reconstruction quality with multiple metrics"""
    model.eval()

    psnr_values = []
    ssim_values = []

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        images = batch[0].to(device)
        recon, _, _ = model(images)

        # Calculate metrics
        psnr = calculate_psnr(images, recon)
        ssim = calculate_ssim(images, recon)

        psnr_values.append(psnr.item())
        ssim_values.append(ssim.item())

    return {
        'psnr': np.mean(psnr_values),
        'ssim': np.mean(ssim_values),
        'psnr_std': np.std(psnr_values),
        'ssim_std': np.std(ssim_values)
    }


@torch.no_grad()
def calculate_codebook_usage(model, dataloader, device='cuda', num_batches=50):
    """Calculate codebook usage statistics"""
    model.eval()

    all_indices = []

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        images = batch[0].to(device)
        _, _, indices = model.encode(images)

        if isinstance(indices, list):
            # Factorized codebooks
            all_indices.append([idx.cpu() for idx in indices])
        else:
            # Single codebook
            all_indices.append(indices.cpu())

    if isinstance(all_indices[0], list):
        # Factorized
        num_codebooks = len(all_indices[0])
        usage_stats = []

        for i in range(num_codebooks):
            indices = torch.cat([batch[i].flatten() for batch in all_indices])
            unique = torch.unique(indices)
            usage_stats.append({
                'total_codes': len(unique),
                'usage_rate': len(unique) / model.quantizer.codebook_size
            })

        return usage_stats
    else:
        # Single codebook
        indices = torch.cat([batch.flatten() for batch in all_indices])
        unique = torch.unique(indices)

        return {
            'total_codes': len(unique),
            'usage_rate': len(unique) / model.quantizer.num_embeddings
        }
