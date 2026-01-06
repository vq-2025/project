import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class VectorQuantizer(nn.Module):
    """Vector Quantization layer for VQ-GAN"""
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, z):
        # z: [B, C, H, W]
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.embedding_dim)

        # Calculate distances
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())

        # Find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # Compute loss
        loss = torch.mean((z_q.detach() - z)**2) + \
               self.commitment_cost * torch.mean((z_q - z.detach())**2)

        # Preserve gradients
        z_q = z + (z_q - z).detach()

        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        return z_q, loss, min_encoding_indices


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x):
        residual = self.residual_conv(x)
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)
        return x + residual


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        # Compute attention
        b, c, h, w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        v = rearrange(v, 'b c h w -> b (h w) c')

        attn = torch.softmax(torch.bmm(q, k) / (c ** 0.5), dim=2)
        out = torch.bmm(attn, v)
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)
        out = self.proj_out(out)

        return x + out


class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=[128, 256, 512], latent_dim=256):
        super().__init__()

        layers = [nn.Conv2d(in_channels, hidden_dims[0], 3, padding=1)]

        # Downsampling blocks
        for i in range(len(hidden_dims)):
            in_ch = hidden_dims[i]
            out_ch = hidden_dims[i]

            layers.append(ResidualBlock(in_ch, out_ch))
            layers.append(ResidualBlock(out_ch, out_ch))

            if i < len(hidden_dims) - 1:
                layers.append(nn.Conv2d(out_ch, hidden_dims[i+1], 4, stride=2, padding=1))

        # Middle blocks
        layers.append(ResidualBlock(hidden_dims[-1], hidden_dims[-1]))
        layers.append(AttentionBlock(hidden_dims[-1]))
        layers.append(ResidualBlock(hidden_dims[-1], hidden_dims[-1]))

        # Output
        layers.append(nn.GroupNorm(32, hidden_dims[-1]))
        layers.append(nn.SiLU())
        layers.append(nn.Conv2d(hidden_dims[-1], latent_dim, 3, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim=256, hidden_dims=[512, 256, 128], out_channels=3):
        super().__init__()

        layers = [nn.Conv2d(latent_dim, hidden_dims[0], 3, padding=1)]

        # Middle blocks
        layers.append(ResidualBlock(hidden_dims[0], hidden_dims[0]))
        layers.append(AttentionBlock(hidden_dims[0]))
        layers.append(ResidualBlock(hidden_dims[0], hidden_dims[0]))

        # Upsampling blocks
        for i in range(len(hidden_dims)):
            in_ch = hidden_dims[i]
            out_ch = hidden_dims[i]

            layers.append(ResidualBlock(in_ch, out_ch))
            layers.append(ResidualBlock(out_ch, out_ch))

            if i < len(hidden_dims) - 1:
                layers.append(nn.ConvTranspose2d(out_ch, hidden_dims[i+1], 4, stride=2, padding=1))

        # Output
        layers.append(nn.GroupNorm(32, hidden_dims[-1]))
        layers.append(nn.SiLU())
        layers.append(nn.Conv2d(hidden_dims[-1], out_channels, 3, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=[64, 128, 256, 512]):
        super().__init__()

        layers = []
        prev_dim = in_channels

        for dim in hidden_dims:
            layers.extend([
                nn.Conv2d(prev_dim, dim, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            prev_dim = dim

        layers.append(nn.Conv2d(prev_dim, 1, 4, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class VQGAN(nn.Module):
    def __init__(
        self,
        in_channels=3,
        hidden_dims=[128, 256, 512],
        latent_dim=256,
        num_embeddings=1024,
        commitment_cost=0.25
    ):
        super().__init__()

        self.encoder = Encoder(in_channels, hidden_dims, latent_dim)
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.decoder = Decoder(latent_dim, list(reversed(hidden_dims)), in_channels)

    def encode(self, x):
        z = self.encoder(x)
        z_q, vq_loss, indices = self.quantizer(z)
        return z_q, vq_loss, indices

    def decode(self, z_q):
        return self.decoder(z_q)

    def forward(self, x):
        z_q, vq_loss, indices = self.encode(x)
        x_recon = self.decode(z_q)
        return x_recon, vq_loss, indices
