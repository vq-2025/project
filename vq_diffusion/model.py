import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, t):
        h = self.block1(x)
        h = h + self.time_mlp(t)[:, :, None, None]
        h = self.block2(h)
        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = rearrange(qkv, 'b (three heads c) h w -> three b heads c (h w)',
                           three=3, heads=self.num_heads)

        scale = (c // self.num_heads) ** -0.5
        attn = torch.softmax(torch.einsum('bhci,bhcj->bhij', q, k) * scale, dim=-1)
        out = torch.einsum('bhij,bhcj->bhci', attn, v)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', h=h, w=w)

        return x + self.proj(out)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        hidden_dims=[128, 256, 512, 512],
        time_emb_dim=256,
        num_res_blocks=2,
        attn_resolutions=[16, 8]
    ):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # Initial conv
        self.init_conv = nn.Conv2d(in_channels, hidden_dims[0], 3, padding=1)

        # Downsampling
        self.downs = nn.ModuleList([])
        prev_dim = hidden_dims[0]

        for i, dim in enumerate(hidden_dims):
            is_last = i == len(hidden_dims) - 1

            blocks = nn.ModuleList([])
            for _ in range(num_res_blocks):
                blocks.append(ResidualBlock(prev_dim, dim, time_emb_dim))
                prev_dim = dim

            down = nn.Module()
            down.blocks = blocks
            down.downsample = nn.Conv2d(dim, dim, 4, 2, 1) if not is_last else nn.Identity()

            self.downs.append(down)

        # Middle
        self.mid_block1 = ResidualBlock(hidden_dims[-1], hidden_dims[-1], time_emb_dim)
        self.mid_attn = AttentionBlock(hidden_dims[-1])
        self.mid_block2 = ResidualBlock(hidden_dims[-1], hidden_dims[-1], time_emb_dim)

        # Upsampling
        self.ups = nn.ModuleList([])
        prev_dim = hidden_dims[-1]

        for i, dim in enumerate(reversed(hidden_dims)):
            is_last = i == len(hidden_dims) - 1

            blocks = nn.ModuleList([])
            for _ in range(num_res_blocks + 1):
                blocks.append(ResidualBlock(prev_dim + dim, dim, time_emb_dim))
                prev_dim = dim

            up = nn.Module()
            up.blocks = blocks
            up.upsample = nn.ConvTranspose2d(dim, dim, 4, 2, 1) if not is_last else nn.Identity()

            self.ups.append(up)

        # Output
        self.final_conv = nn.Sequential(
            nn.GroupNorm(32, hidden_dims[0]),
            nn.SiLU(),
            nn.Conv2d(hidden_dims[0], out_channels, 3, padding=1)
        )

    def forward(self, x, t):
        t = self.time_mlp(t)

        x = self.init_conv(x)

        # Downsampling
        skips = []
        for down in self.downs:
            for block in down.blocks:
                x = block(x, t)
                skips.append(x)
            x = down.downsample(x)

        # Middle
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # Upsampling
        for up in self.ups:
            for block in up.blocks:
                x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, t)
            x = up.upsample(x)

        return self.final_conv(x)


class VQDiffusion(nn.Module):
    """VQ-Diffusion model for discrete diffusion on codebook indices"""
    def __init__(
        self,
        num_embeddings=1024,
        embedding_dim=256,
        image_size=32,
        hidden_dims=[128, 256, 512, 512],
        num_timesteps=100
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_timesteps = num_timesteps
        self.image_size = image_size

        # Embedding layer for discrete tokens
        self.token_embedding = nn.Embedding(num_embeddings + 1, embedding_dim)  # +1 for mask token

        # Denoising network
        self.denoise_fn = UNet(
            in_channels=embedding_dim,
            out_channels=num_embeddings,
            hidden_dims=hidden_dims
        )

        # Noise schedule
        self.register_buffer('betas', self._cosine_schedule(num_timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))

    def _cosine_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def q_sample(self, x_0, t):
        """Forward diffusion process - add noise to clean indices"""
        # x_0: [B, H, W] with values in [0, num_embeddings-1]
        batch_size = x_0.shape[0]

        # Sample mask probabilities
        alphas_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1)

        # Randomly mask tokens based on noise schedule
        mask = torch.rand_like(x_0.float()) > alphas_cumprod_t

        # Replace masked tokens with mask token (num_embeddings)
        x_t = torch.where(mask, torch.ones_like(x_0) * self.num_embeddings, x_0)

        return x_t.long()

    def p_losses(self, x_0, t):
        """Compute training loss"""
        # x_0: clean codebook indices [B, H, W]
        x_t = self.q_sample(x_0, t)

        # Embed tokens
        x_emb = self.token_embedding(x_t)
        x_emb = rearrange(x_emb, 'b h w c -> b c h w')

        # Predict logits for each position
        logits = self.denoise_fn(x_emb, t)
        logits = rearrange(logits, 'b c h w -> b h w c')

        # Cross entropy loss
        loss = F.cross_entropy(
            logits.reshape(-1, self.num_embeddings),
            x_0.reshape(-1),
            reduction='mean'
        )

        return loss

    @torch.no_grad()
    def p_sample(self, x_t, t):
        """Single denoising step"""
        x_emb = self.token_embedding(x_t)
        x_emb = rearrange(x_emb, 'b h w c -> b c h w')

        logits = self.denoise_fn(x_emb, t)
        logits = rearrange(logits, 'b c h w -> b h w c')

        # Sample from predicted distribution
        probs = F.softmax(logits, dim=-1)
        x_t_minus_1 = torch.multinomial(
            probs.reshape(-1, self.num_embeddings),
            num_samples=1
        ).reshape(x_t.shape)

        return x_t_minus_1

    @torch.no_grad()
    def sample(self, batch_size, device):
        """Generate samples from noise"""
        # Start from all mask tokens
        x_t = torch.ones(batch_size, self.image_size, self.image_size,
                        dtype=torch.long, device=device) * self.num_embeddings

        # Iterative denoising
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x_t = self.p_sample(x_t, t_batch)

        return x_t

    def forward(self, x_0):
        """Training forward pass"""
        batch_size = x_0.shape[0]
        device = x_0.device

        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()

        return self.p_losses(x_0, t)
