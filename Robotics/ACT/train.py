"""
Training script for ACT (Action Chunking with Transformers).

Trains the model on synthetic Gaussian random-walk demonstrations and saves a checkpoint.

Usage:
    python train.py                           # full training
    python train.py --dry-run --save-model    # single batch, save checkpoint
    python train.py --epochs 5 --num-demos 20 --save-model
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import ACT, SyntheticDemoDataset, get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ACT on synthetic demonstrations")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50, metavar="N")
    parser.add_argument("--batch-size", type=int, default=16, metavar="N")
    parser.add_argument("--lr", type=float, default=1e-4, metavar="LR")

    # Model architecture
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--joint-dim", type=int, default=7)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--beta", type=float, default=10.0, help="KL weight in CVAE loss")
    parser.add_argument("--num-enc-layers", type=int, default=4)
    parser.add_argument("--num-dec-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)

    # Dataset
    parser.add_argument("--num-demos", type=int, default=100)
    parser.add_argument("--traj-len", type=int, default=50)
    parser.add_argument("--img-h", type=int, default=96)
    parser.add_argument("--img-w", type=int, default=96)
    parser.add_argument("--num-cameras", type=int, default=1)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=10, metavar="N")
    parser.add_argument("--save-model", action="store_true", default=False)
    parser.add_argument("--model-path", type=str, default="act_model.pt")
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--no-mps", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Run a single batch to verify shapes, then exit")

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    """Collect all model hyperparameters into a single dict for checkpoint saving."""
    return {
        "chunk_size": args.chunk_size,
        "joint_dim": args.joint_dim,
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "beta": args.beta,
        "num_enc_layers": args.num_enc_layers,
        "num_dec_layers": args.num_dec_layers,
        "num_heads": args.num_heads,
        "img_h": args.img_h,
        "img_w": args.img_w,
        "num_cameras": args.num_cameras,
    }


def train_epoch(
    model: ACT,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    dataset_size: int,
    args: argparse.Namespace,
) -> None:
    model.train()

    for batch_idx, (images, joint_positions, action_chunks) in enumerate(train_loader):
        images = images.to(device)
        joint_positions = joint_positions.to(device)
        action_chunks = action_chunks.to(device)

        optimizer.zero_grad()

        pred_actions, mu, log_var = model(images, joint_positions, actions=action_chunks)
        total_loss, recon_loss, kl_loss = model.compute_loss(
            pred_actions, action_chunks, mu, log_var
        )

        total_loss.backward()

        # Gradient clipping prevents transformer gradient explosion
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print(
                f"Train Epoch {epoch:3d} [{batch_idx * len(images):5d}/{dataset_size}]"
                f"  total: {total_loss.item():.4f}"
                f"  recon: {recon_loss.item():.4f}"
                f"  kl: {kl_loss.item():.4f}"
            )

        if args.dry_run:
            break


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    device = get_device(args.no_cuda, args.no_mps)
    print(f"Device: {device}")

    # Dataset and DataLoader
    dataset = SyntheticDemoDataset(
        num_demos=args.num_demos,
        traj_len=args.traj_len,
        chunk_size=args.chunk_size,
        joint_dim=args.joint_dim,
        num_cameras=args.num_cameras,
        img_h=args.img_h,
        img_w=args.img_w,
        seed=args.seed,
    )
    dataset_size = len(dataset)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Dataset: {dataset_size} samples ({args.num_demos} demos × {args.traj_len} steps)")

    # Model
    config = build_config(args)
    model = ACT(config).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_epoch(model, device, train_loader, optimizer, epoch, dataset_size, args)
        if args.dry_run:
            break

    # Save checkpoint
    if args.save_model:
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": config,
        }
        torch.save(checkpoint, args.model_path)
        print(f"Checkpoint saved to {args.model_path}")


if __name__ == "__main__":
    main()
