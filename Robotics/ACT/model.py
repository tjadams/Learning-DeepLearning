"""
ACT (Action Chunking with Transformers) model implementation.
Based on: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (arxiv 2304.13705)

Architecture: Conditional VAE (CVAE) that predicts chunks of k actions at once.
- CVAEEncoder: BERT-like encoder, training only, produces latent z
- ACTPolicy: ResNet18 backbone + transformer encoder/decoder, used at train and inference
- ACT: Full model wrapper combining both components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.models import resnet18


def get_device(no_cuda=False, no_mps=False):
    """Return the best available device: CUDA > MPS > CPU."""
    use_cuda = not no_cuda and torch.cuda.is_available()
    use_mps = not no_mps and torch.backends.mps.is_available()
    if use_cuda:
        return torch.device("cuda")
    elif use_mps:
        return torch.device("mps")
    else:
        return torch.device("cpu")


class SinusoidalPositionEmbedding2D(nn.Module):
    """
    Fixed 2D sinusoidal position embeddings for image feature tokens.

    Splits hidden_dim in half: first half encodes y-position, second half encodes x-position.
    Registered as a buffer so it moves to the correct device automatically and is not trained.
    """

    def __init__(self, hidden_dim: int, feat_h: int, feat_w: int):
        super().__init__()
        assert hidden_dim % 2 == 0, "hidden_dim must be even for 2D sinusoidal embeddings"
        half_dim = hidden_dim // 2

        # Compute sinusoidal frequencies
        # Shape: (half_dim // 2,) — each half is further split for sin/cos
        dim_t = torch.arange(half_dim // 2, dtype=torch.float32)
        dim_t = 10000 ** (2 * dim_t / half_dim)

        # y-position embeddings: (feat_h, half_dim)
        pos_y = torch.arange(feat_h, dtype=torch.float32).unsqueeze(1)  # (H, 1)
        y_embed = torch.zeros(feat_h, half_dim)
        y_embed[:, 0::2] = torch.sin(pos_y / dim_t)
        y_embed[:, 1::2] = torch.cos(pos_y / dim_t)

        # x-position embeddings: (feat_w, half_dim)
        pos_x = torch.arange(feat_w, dtype=torch.float32).unsqueeze(1)  # (W, 1)
        x_embed = torch.zeros(feat_w, half_dim)
        x_embed[:, 0::2] = torch.sin(pos_x / dim_t)
        x_embed[:, 1::2] = torch.cos(pos_x / dim_t)

        # Combine: tile y over W and x over H, then cat along hidden_dim
        # y: (H, W, half_dim) — same y embedding for every x position
        # x: (H, W, half_dim) — same x embedding for every y position
        y_embed = y_embed.unsqueeze(1).expand(feat_h, feat_w, half_dim)
        x_embed = x_embed.unsqueeze(0).expand(feat_h, feat_w, half_dim)

        # pos_embed: (H*W, hidden_dim)
        pos_embed = torch.cat([y_embed, x_embed], dim=-1).view(feat_h * feat_w, hidden_dim)
        self.register_buffer("pos_embed", pos_embed)

    def forward(self):
        """Returns (feat_h * feat_w, hidden_dim) sinusoidal position embeddings."""
        return self.pos_embed


class ImageBackbone(nn.Module):
    """
    ResNet18 feature extractor with projection to hidden_dim.

    Removes avgpool and fc layers, keeps up through layer4.
    Input:  (B, 3, H, W)
    Output: (B, hidden_dim, H//32, W//32)

    For 96×96 input: output is (B, hidden_dim, 3, 3)
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, hidden_dim: int):
        super().__init__()
        backbone = resnet18(weights=None)
        # Remove avgpool and fc, keep up to layer4
        self.feature_extractor = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        # ResNet18 layer4 outputs 512 channels; project to hidden_dim
        self.proj = nn.Conv2d(512, hidden_dim, kernel_size=1)

        # Register ImageNet normalization stats as buffers
        mean = torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) image tensor in [0, 1]
        Returns:
            (B, hidden_dim, H//32, W//32)
        """
        x = (x - self.mean) / self.std
        x = self.feature_extractor(x)
        x = self.proj(x)
        return x


class CVAEEncoder(nn.Module):
    """
    CVAE Encoder (training only).

    Processes [CLS, joint_pos, a_1, ..., a_k] sequence of length k+2 through
    a BERT-like transformer encoder. The CLS token output is projected to
    produce the latent distribution parameters (mu, log_var).
    """

    def __init__(
        self,
        joint_dim: int,
        hidden_dim: int,
        latent_dim: int,
        chunk_size: int,
        num_enc_layers: int,
        num_heads: int,
    ):
        super().__init__()
        self.chunk_size = chunk_size

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Input projections
        self.joint_pos_embedding = nn.Linear(joint_dim, hidden_dim)
        self.action_embedding = nn.Linear(joint_dim, hidden_dim)

        # Learned 1D positional IDs for the k+2 sequence positions
        self.pos_embedding = nn.Embedding(chunk_size + 2, hidden_dim)

        # Transformer encoder (BERT-style)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_enc_layers)

        # Project CLS output to latent distribution parameters
        self.mu_proj = nn.Linear(hidden_dim, latent_dim)
        self.log_var_proj = nn.Linear(hidden_dim, latent_dim)

        nn.init.normal_(self.cls_token, std=0.02)

    def forward(
        self, joint_positions: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            joint_positions: (B, joint_dim) current joint state
            actions:         (B, chunk_size, joint_dim) action chunk targets
        Returns:
            mu:      (B, latent_dim)
            log_var: (B, latent_dim)
        """
        B = joint_positions.shape[0]

        # Project inputs to hidden_dim tokens
        jpos_token = self.joint_pos_embedding(joint_positions).unsqueeze(1)  # (B, 1, hidden_dim)
        action_tokens = self.action_embedding(actions)                        # (B, k, hidden_dim)

        # Expand CLS token over batch
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, hidden_dim)

        # Concatenate: [CLS | joint_pos | a_1 ... a_k] → (B, k+2, hidden_dim)
        seq = torch.cat([cls_tokens, jpos_token, action_tokens], dim=1)

        # Add learned positional embeddings
        positions = torch.arange(self.chunk_size + 2, device=seq.device)
        seq = seq + self.pos_embedding(positions).unsqueeze(0)  # broadcast over batch

        # Encode
        encoded = self.transformer_encoder(seq)  # (B, k+2, hidden_dim)

        # Extract CLS token (index 0) output for latent distribution
        cls_out = encoded[:, 0, :]  # (B, hidden_dim)
        mu = self.mu_proj(cls_out)
        log_var = self.log_var_proj(cls_out)

        return mu, log_var


class ACTPolicy(nn.Module):
    """
    ACT Policy — the CVAE decoder, used at both train and inference time.

    Processes camera images through ResNet18, combines with joint state and
    latent z via transformer encoder, then uses learned action queries to
    cross-attend and produce an action chunk.
    """

    def __init__(
        self,
        joint_dim: int,
        hidden_dim: int,
        latent_dim: int,
        chunk_size: int,
        num_cameras: int,
        feat_h: int,
        feat_w: int,
        num_enc_layers: int,
        num_dec_layers: int,
        num_heads: int,
    ):
        super().__init__()
        self.num_cameras = num_cameras
        self.feat_h = feat_h
        self.feat_w = feat_w
        self.chunk_size = chunk_size

        self.backbone = ImageBackbone(hidden_dim)
        self.image_pos_embed = SinusoidalPositionEmbedding2D(hidden_dim, feat_h, feat_w)

        self.joint_proj = nn.Linear(joint_dim, hidden_dim)
        self.z_proj = nn.Linear(latent_dim, hidden_dim)

        # Transformer encoder: encodes [z_token | jpos_token | image_tokens]
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_enc_layers)

        # DETR-style learned action queries
        self.action_queries = nn.Parameter(torch.zeros(1, chunk_size, hidden_dim))

        # Transformer decoder: action queries cross-attend to encoder memory
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(dec_layer, num_layers=num_dec_layers)

        # Project each decoded query to a joint action
        self.action_head = nn.Linear(hidden_dim, joint_dim)

        nn.init.normal_(self.action_queries, std=0.02)

    def forward(
        self,
        images: torch.Tensor,
        joint_positions: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            images:          (B, num_cameras, 3, H, W)
            joint_positions: (B, joint_dim)
            z:               (B, latent_dim)
        Returns:
            pred_actions: (B, chunk_size, joint_dim)
        """
        B, num_cam, C, H, W = images.shape

        # --- Image processing ---
        # Fold cameras into batch for efficient backbone processing
        imgs_flat = images.view(B * num_cam, C, H, W)         # (B*num_cam, 3, H, W)
        feat_flat = self.backbone(imgs_flat)                    # (B*num_cam, hidden_dim, fh, fw)
        _, hidden_dim, fh, fw = feat_flat.shape

        # Reshape to (B, num_cam, hidden_dim, fh, fw) then flatten spatial dims
        feat = feat_flat.view(B, num_cam, hidden_dim, fh, fw)
        feat = feat.permute(0, 1, 3, 4, 2)                     # (B, num_cam, fh, fw, hidden_dim)
        feat = feat.reshape(B, num_cam * fh * fw, hidden_dim)  # (B, num_cam*fh*fw, hidden_dim)

        # Add 2D sinusoidal position embeddings (tiled per camera)
        pos = self.image_pos_embed()                            # (fh*fw, hidden_dim)
        pos = pos.unsqueeze(0).expand(num_cam, -1, -1)         # (num_cam, fh*fw, hidden_dim)
        pos = pos.reshape(num_cam * fh * fw, hidden_dim)       # (num_cam*fh*fw, hidden_dim)
        feat = feat + pos.unsqueeze(0)                          # broadcast over batch

        # --- Non-image tokens ---
        jpos_token = self.joint_proj(joint_positions).unsqueeze(1)  # (B, 1, hidden_dim)
        z_token = self.z_proj(z).unsqueeze(1)                       # (B, 1, hidden_dim)

        # --- Encoder ---
        # Concat [z_token | jpos_token | image_tokens] → (B, 2 + num_cam*fh*fw, hidden_dim)
        enc_input = torch.cat([z_token, jpos_token, feat], dim=1)
        memory = self.transformer_encoder(enc_input)                # (B, seq_len, hidden_dim)

        # --- Decoder ---
        # Expand learned action queries over batch
        queries = self.action_queries.expand(B, -1, -1)            # (B, chunk_size, hidden_dim)
        decoded = self.transformer_decoder(queries, memory)        # (B, chunk_size, hidden_dim)

        # Project to joint actions
        pred_actions = self.action_head(decoded)                   # (B, chunk_size, joint_dim)
        return pred_actions


class ACT(nn.Module):
    """
    Full ACT model: CVAE Encoder + ACTPolicy (decoder).

    At training time: runs the CVAE encoder to produce z ~ N(mu, sigma^2).
    At inference time: uses z = 0 (prior mean), CVAE encoder is discarded.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.latent_dim = config["latent_dim"]
        self.beta = config["beta"]

        self.cvae_encoder = CVAEEncoder(
            joint_dim=config["joint_dim"],
            hidden_dim=config["hidden_dim"],
            latent_dim=config["latent_dim"],
            chunk_size=config["chunk_size"],
            num_enc_layers=config["num_enc_layers"],
            num_heads=config["num_heads"],
        )

        feat_h = config["img_h"] // 32
        feat_w = config["img_w"] // 32

        self.policy = ACTPolicy(
            joint_dim=config["joint_dim"],
            hidden_dim=config["hidden_dim"],
            latent_dim=config["latent_dim"],
            chunk_size=config["chunk_size"],
            num_cameras=config["num_cameras"],
            feat_h=feat_h,
            feat_w=feat_w,
            num_enc_layers=config["num_enc_layers"],
            num_dec_layers=config["num_dec_layers"],
            num_heads=config["num_heads"],
        )

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + eps * std."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        images: torch.Tensor,
        joint_positions: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            images:          (B, num_cameras, 3, H, W)
            joint_positions: (B, joint_dim)
            actions:         (B, chunk_size, joint_dim) — only provided during training
        Returns:
            pred_actions: (B, chunk_size, joint_dim)
            mu:           (B, latent_dim)  — zeros at inference
            log_var:      (B, latent_dim)  — zeros at inference
        """
        B = images.shape[0]
        device = images.device

        if self.training and actions is not None:
            mu, log_var = self.cvae_encoder(joint_positions, actions)
            z = self.reparameterize(mu, log_var)
        else:
            # Inference: use prior mean z = 0
            mu = torch.zeros(B, self.latent_dim, device=device)
            log_var = torch.zeros(B, self.latent_dim, device=device)
            z = mu  # z = 0

        pred_actions = self.policy(images, joint_positions, z)
        return pred_actions, mu, log_var

    def compute_loss(
        self,
        pred_actions: torch.Tensor,
        target_actions: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute CVAE training loss: L1 reconstruction + beta-weighted KL divergence.

        Args:
            pred_actions:   (B, chunk_size, joint_dim)
            target_actions: (B, chunk_size, joint_dim)
            mu:             (B, latent_dim)
            log_var:        (B, latent_dim)
        Returns:
            total_loss, recon_loss, kl_loss  — all scalar tensors
        """
        recon_loss = F.l1_loss(pred_actions, target_actions)

        # KL divergence: -0.5 * mean(1 + log_var - mu^2 - exp(log_var))
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        total_loss = recon_loss + self.beta * kl_loss
        return total_loss, recon_loss, kl_loss


class SyntheticDemoDataset(Dataset):
    """
    Synthetic dataset of Gaussian random-walk trajectories for testing ACT.

    Generates `num_demos` trajectories at init time. Each trajectory has
    `traj_len` timesteps of `joint_dim` joint positions, produced by
    cumulative sum of Gaussian noise (random walk).

    __getitem__ returns one (observation, action_chunk) sample from a flat
    index over all (demo, timestep) pairs.
    """

    def __init__(
        self,
        num_demos: int = 100,
        traj_len: int = 50,
        chunk_size: int = 10,
        joint_dim: int = 7,
        num_cameras: int = 1,
        img_h: int = 96,
        img_w: int = 96,
        seed: int = 42,
    ):
        self.traj_len = traj_len
        self.chunk_size = chunk_size
        self.joint_dim = joint_dim
        self.num_cameras = num_cameras
        self.img_h = img_h
        self.img_w = img_w

        rng = torch.Generator()
        rng.manual_seed(seed)

        # Generate trajectories: (num_demos, traj_len, joint_dim)
        noise = torch.randn(num_demos, traj_len, joint_dim, generator=rng) * 0.05
        self.trajectories = torch.cumsum(noise, dim=1)

        # Total samples = num_demos * traj_len
        self._length = num_demos * traj_len

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int):
        demo_idx = idx // self.traj_len
        step_idx = idx % self.traj_len

        # Current joint state
        joint_pos = self.trajectories[demo_idx, step_idx]  # (joint_dim,)

        # Action chunk: next chunk_size actions, padding by repeating last action
        end_idx = min(step_idx + self.chunk_size, self.traj_len)
        chunk = self.trajectories[demo_idx, step_idx:end_idx]  # (<=chunk_size, joint_dim)

        if chunk.shape[0] < self.chunk_size:
            # Pad by repeating the last action
            pad_len = self.chunk_size - chunk.shape[0]
            last_action = chunk[-1:].expand(pad_len, -1)
            chunk = torch.cat([chunk, last_action], dim=0)

        # Random image (in practice this would be a real camera frame)
        # Shape: (num_cameras, 3, img_h, img_w) in [0, 1]
        image = torch.rand(self.num_cameras, 3, self.img_h, self.img_w)

        return image, joint_pos, chunk
