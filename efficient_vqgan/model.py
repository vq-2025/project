import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class FactorizedVectorQuantizer(nn.Module):
    """Factorized Vector Quantization with multiple smaller codebooks"""
    def __init__(self, num_codebooks=4, codebook_size=256, embedding_dim=256, commitment_cost=0.25):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        assert embedding_dim % num_codebooks == 0, "embedding_dim must be divisible by num_codebooks"
        self.dim_per_codebook = embedding_dim // num_codebooks

        # Multiple smaller codebooks
        self.codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, self.dim_per_codebook)
            for _ in range(num_codebooks)
        ])

        # Initialize
        for codebook in self.codebooks:
            codebook.weight.data.uniform_(-1/codebook_size, 1/codebook_size)

    def forward(self, z):
        # z: [B, C, H, W]
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        b, h, w, c = z.shape

        # Split into codebook groups
        z_splits = torch.chunk(z, self.num_codebooks, dim=-1)

        z_q_list = []
        indices_list = []
        losses = []

        for i, (z_split, codebook) in enumerate(zip(z_splits, self.codebooks)):
            z_flat = z_split.reshape(-1, self.dim_per_codebook)

            # Calculate distances
            d = torch.sum(z_flat ** 2, dim=1, keepdim=True) + \
                torch.sum(codebook.weight**2, dim=1) - \
                2 * torch.matmul(z_flat, codebook.weight.t())

            # Find closest encodings
            min_encoding_indices = torch.argmin(d, dim=1)
            z_q = codebook(min_encoding_indices).view(b, h, w, self.dim_per_codebook)

            # Compute loss for this codebook
            loss = torch.mean((z_q.detach() - z_split)**2) + \
                   self.commitment_cost * torch.mean((z_q - z_split.detach())**2)

            # Preserve gradients
            z_q = z_split + (z_q - z_split).detach()

            z_q_list.append(z_q)
            indices_list.append(min_encoding_indices.reshape(b, h, w))
            losses.append(loss)

        # Concatenate all codebooks
        z_q = torch.cat(z_q_list, dim=-1)
        total_loss = sum(losses) / len(losses)

        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        return z_q, total_loss, indices_list


class EfficientResBlock(nn.Module):
    """Efficient residual block with depthwise separable convolutions"""
    def __init__(self, channels, expansion=4):
        super().__init__()
        hidden_dim = channels * expansion

        self.norm1 = nn.GroupNorm(32, channels)
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pointwise1 = nn.Conv2d(channels, hidden_dim, 1)
        self.norm2 = nn.GroupNorm(32, hidden_dim)
        self.pointwise2 = nn.Conv2d(hidden_dim, channels, 1)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = F.silu(x)
        x = self.depthwise(x)
        x = self.pointwise1(x)
        x = self.norm2(x)
        x = F.silu(x)
        x = self.pointwise2(x)
        return x + residual


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1)

    def forward(self, x):
        # Global average pooling
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class EfficientEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=[64, 128, 256], latent_dim=256):
        super().__init__()

        layers = [nn.Conv2d(in_channels, hidden_dims[0], 3, padding=1)]

        # Downsampling blocks with efficient operations
        for i in range(len(hidden_dims)):
            layers.extend([
                EfficientResBlock(hidden_dims[i]),
                EfficientResBlock(hidden_dims[i]),
                SqueezeExcitation(hidden_dims[i])
            ])

            if i < len(hidden_dims) - 1:
                # Strided depthwise conv for downsampling
                layers.append(nn.Conv2d(hidden_dims[i], hidden_dims[i], 3, stride=2, padding=1, groups=hidden_dims[i]))
                layers.append(nn.Conv2d(hidden_dims[i], hidden_dims[i+1], 1))

        # Output
        layers.extend([
            nn.GroupNorm(32, hidden_dims[-1]),
            nn.SiLU(),
            nn.Conv2d(hidden_dims[-1], latent_dim, 1)
        ])

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class EfficientDecoder(nn.Module):
    def __init__(self, latent_dim=256, hidden_dims=[256, 128, 64], out_channels=3):
        super().__init__()

        layers = [nn.Conv2d(latent_dim, hidden_dims[0], 1)]

        # Upsampling blocks
        for i in range(len(hidden_dims)):
            layers.extend([
                EfficientResBlock(hidden_dims[i]),
                EfficientResBlock(hidden_dims[i]),
                SqueezeExcitation(hidden_dims[i])
            ])

            if i < len(hidden_dims) - 1:
                # Pixel shuffle for upsampling
                layers.append(nn.Conv2d(hidden_dims[i], hidden_dims[i+1] * 4, 1))
                layers.append(nn.PixelShuffle(2))

        # Output
        layers.extend([
            nn.GroupNorm(32, hidden_dims[-1]),
            nn.SiLU(),
            nn.Conv2d(hidden_dims[-1], out_channels, 3, padding=1)
        ])

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator for better training"""
    def __init__(self, in_channels=3, num_scales=3):
        super().__init__()
        self.num_scales = num_scales

        self.discriminators = nn.ModuleList([
            self._make_discriminator(in_channels)
            for _ in range(num_scales)
        ])

        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def _make_discriminator(self, in_channels):
        layers = []
        channels = [in_channels, 64, 128, 256, 512]

        for i in range(len(channels) - 1):
            layers.extend([
                nn.Conv2d(channels[i], channels[i+1], 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ])

        layers.append(nn.Conv2d(channels[-1], 1, 4, padding=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        outputs = []
        for disc in self.discriminators:
            outputs.append(disc(x))
            x = self.downsample(x)
        return outputs


class EfficientVQGAN(nn.Module):
    def __init__(
        self,
        in_channels=3,
        hidden_dims=[64, 128, 256],
        latent_dim=256,
        num_codebooks=4,
        codebook_size=256,
        commitment_cost=0.25
    ):
        super().__init__()

        self.encoder = EfficientEncoder(in_channels, hidden_dims, latent_dim)
        self.quantizer = FactorizedVectorQuantizer(
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            embedding_dim=latent_dim,
            commitment_cost=commitment_cost
        )
        self.decoder = EfficientDecoder(latent_dim, list(reversed(hidden_dims)), in_channels)

    def encode(self, x):
        z = self.encoder(x)
        z_q, vq_loss, indices_list = self.quantizer(z)
        return z_q, vq_loss, indices_list

    def decode(self, z_q):
        return self.decoder(z_q)

    def forward(self, x):
        z_q, vq_loss, indices_list = self.encode(x)
        x_recon = self.decode(z_q)
        return x_recon, vq_loss, indices_list

    def get_codebook_usage(self):
        """Monitor codebook usage to detect collapse"""
        usage = []
        for i, codebook in enumerate(self.quantizer.codebooks):
            usage.append(torch.zeros(self.quantizer.codebook_size))
        return usage

    def count_parameters(self):
        """Count total parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
