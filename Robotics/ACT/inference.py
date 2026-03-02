"""
Inference script for ACT (Action Chunking with Transformers).

Loads a trained checkpoint and runs inference with temporal ensembling:
- At each step t, the policy predicts a chunk of k future actions.
- Each predicted action for future step t+i is registered with weight exp(-m*i).
- The executed action at step t is the weighted average of all predictions targeting t.
- This smooths out prediction noise and reduces jerky motion.

Usage:
    python inference.py                         # requires act_model.pt in current dir
    python inference.py --model-path act_model.pt --num-steps 50
"""

import argparse
import math
from collections import defaultdict

import numpy as np
import torch

from model import ACT, get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ACT inference with temporal ensembling")
    parser.add_argument("--model-path", type=str, default="act_model.pt")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--temporal-m", type=float, default=0.1,
                        help="Exponential decay rate for temporal ensembling weights")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--no-mps", action="store_true", default=False)
    return parser.parse_args()


def load_model(model_path: str, device: torch.device) -> tuple[ACT, dict]:
    """Load a saved ACT checkpoint and return (model, config)."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    config = checkpoint["config"]
    model = ACT(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def run_inference(
    model: ACT,
    config: dict,
    device: torch.device,
    num_steps: int,
    m: float,
    seed: int,
) -> np.ndarray:
    """
    Simulate num_steps timesteps with temporal ensembling.

    At each step t:
    1. Query the policy → chunk of k predicted actions
    2. Register pred[i] for future step t+i with weight exp(-m*i)
    3. Execute weighted-average action for the current step
    4. Advance simulated joint state

    Args:
        model:     Trained ACT model in eval mode
        config:    Model hyperparameter dict from checkpoint
        device:    Torch device
        num_steps: Number of timesteps to simulate
        m:         Temporal ensembling decay rate (higher → prefer recent predictions)
        seed:      Random seed for synthetic observations

    Returns:
        executed_actions: (num_steps, joint_dim) array of executed actions
    """
    torch.manual_seed(seed)

    joint_dim = config["joint_dim"]
    chunk_size = config["chunk_size"]
    num_cameras = config["num_cameras"]
    img_h = config["img_h"]
    img_w = config["img_w"]

    # Simulated joint state — starts at zeros
    joint_state = torch.zeros(joint_dim, device=device)

    # Temporal ensembling accumulators:
    # accum_actions[t] = sum of weighted predictions for step t
    # accum_weights[t] = sum of weights for step t
    accum_actions: dict[int, torch.Tensor] = defaultdict(lambda: torch.zeros(joint_dim, device=device))
    accum_weights: dict[int, float] = defaultdict(float)

    executed_actions = np.zeros((num_steps, joint_dim))

    with torch.no_grad():
        for t in range(num_steps):
            # --- Build inputs ---
            # Random image observation (in a real system this would be a camera feed)
            images = torch.rand(1, num_cameras, 3, img_h, img_w, device=device)
            joint_pos = joint_state.unsqueeze(0)  # (1, joint_dim)

            # --- Policy query (z = 0 since model.eval()) ---
            pred_actions, _, _ = model(images, joint_pos, actions=None)
            pred_actions = pred_actions.squeeze(0)  # (chunk_size, joint_dim)

            # --- Register predictions with temporal ensembling weights ---
            for i in range(chunk_size):
                future_step = t + i
                if future_step < num_steps:
                    weight = math.exp(-m * i)
                    accum_actions[future_step] += weight * pred_actions[i]
                    accum_weights[future_step] += weight

            # --- Execute ensembled action for current step ---
            executed = accum_actions[t] / accum_weights[t]
            executed_actions[t] = executed.cpu().numpy()

            # Advance simulated joint state (simple integration)
            joint_state = executed.detach()

            print(
                f"Step {t:3d}: "
                f"action = [{', '.join(f'{v:.4f}' for v in executed_actions[t])}]"
            )

            # Clean up past entries to avoid unbounded memory growth
            del accum_actions[t]
            del accum_weights[t]

    return executed_actions


def main() -> None:
    args = parse_args()

    device = get_device(args.no_cuda, args.no_mps)
    print(f"Device: {device}")

    model, config = load_model(args.model_path, device)
    print(f"Loaded model from {args.model_path}")
    print(f"Config: {config}")

    executed_actions = run_inference(
        model=model,
        config=config,
        device=device,
        num_steps=args.num_steps,
        m=args.temporal_m,
        seed=args.seed,
    )

    print(f"\nInference complete: {args.num_steps} steps")
    print(f"Action mean:  {executed_actions.mean(axis=0)}")
    print(f"Action std:   {executed_actions.std(axis=0)}")
    print(f"Action range: [{executed_actions.min():.4f}, {executed_actions.max():.4f}]")


if __name__ == "__main__":
    main()
