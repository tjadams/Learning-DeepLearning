"""
Unit and integration tests for the ACT (Action Chunking with Transformers) implementation.

Run with:
    python -m unittest tests/test_act.py -v
    # from the Robotics/ACT directory
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

# Add parent directory to path so we can import from model.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import (
    ACT,
    CVAEEncoder,
    ImageBackbone,
    SinusoidalPositionEmbedding2D,
    ACTPolicy,
    SyntheticDemoDataset,
)


# Shared small config used across all tests — small enough to run fast on CPU
TEST_CONFIG = {
    "joint_dim": 7,
    "hidden_dim": 64,    # Smaller than production (256) for fast tests
    "latent_dim": 32,
    "chunk_size": 10,
    "num_cameras": 1,
    "img_h": 96,
    "img_w": 96,
    "num_enc_layers": 2,
    "num_dec_layers": 2,
    "num_heads": 8,      # hidden_dim / num_heads = 64/8 = 8 — valid
    "beta": 10.0,
}

BATCH = 2
DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

class TestShapes(unittest.TestCase):
    """Verify output tensor shapes match expectations from the paper."""

    def test_image_backbone_output_shape(self):
        """ResNet18 backbone: (B, 3, 96, 96) → (B, hidden_dim, 3, 3)."""
        model = ImageBackbone(hidden_dim=TEST_CONFIG["hidden_dim"])
        x = torch.rand(BATCH, 3, 96, 96)
        out = model(x)
        self.assertEqual(out.shape, (BATCH, TEST_CONFIG["hidden_dim"], 3, 3))

    def test_sinusoidal_pos_embed_shape(self):
        """2D sinusoidal embeddings: forward() → (feat_h * feat_w, hidden_dim)."""
        hidden_dim = TEST_CONFIG["hidden_dim"]
        h, w = 3, 3
        embed = SinusoidalPositionEmbedding2D(hidden_dim=hidden_dim, feat_h=h, feat_w=w)
        out = embed()
        self.assertEqual(out.shape, (h * w, hidden_dim))

    def test_cvae_encoder_output_shape(self):
        """CVAE encoder: (B, joint_dim) + (B, k, joint_dim) → mu (B, latent_dim), log_var (B, latent_dim)."""
        cfg = TEST_CONFIG
        encoder = CVAEEncoder(
            joint_dim=cfg["joint_dim"],
            hidden_dim=cfg["hidden_dim"],
            latent_dim=cfg["latent_dim"],
            chunk_size=cfg["chunk_size"],
            num_enc_layers=cfg["num_enc_layers"],
            num_heads=cfg["num_heads"],
        )
        joint_pos = torch.rand(BATCH, cfg["joint_dim"])
        actions = torch.rand(BATCH, cfg["chunk_size"], cfg["joint_dim"])
        mu, log_var = encoder(joint_pos, actions)
        self.assertEqual(mu.shape, (BATCH, cfg["latent_dim"]))
        self.assertEqual(log_var.shape, (BATCH, cfg["latent_dim"]))

    def test_act_policy_output_shape(self):
        """ACTPolicy: images (B,1,3,96,96) + joint_pos (B,7) + z (B,32) → (B, chunk_size, joint_dim)."""
        cfg = TEST_CONFIG
        feat_h = cfg["img_h"] // 32
        feat_w = cfg["img_w"] // 32
        policy = ACTPolicy(
            joint_dim=cfg["joint_dim"],
            hidden_dim=cfg["hidden_dim"],
            latent_dim=cfg["latent_dim"],
            chunk_size=cfg["chunk_size"],
            num_cameras=cfg["num_cameras"],
            feat_h=feat_h,
            feat_w=feat_w,
            num_enc_layers=cfg["num_enc_layers"],
            num_dec_layers=cfg["num_dec_layers"],
            num_heads=cfg["num_heads"],
        )
        images = torch.rand(BATCH, cfg["num_cameras"], 3, cfg["img_h"], cfg["img_w"])
        joint_pos = torch.rand(BATCH, cfg["joint_dim"])
        z = torch.rand(BATCH, cfg["latent_dim"])
        out = policy(images, joint_pos, z)
        self.assertEqual(out.shape, (BATCH, cfg["chunk_size"], cfg["joint_dim"]))

    def test_act_full_forward_train(self):
        """ACT training forward: returns pred_actions, mu, log_var with correct shapes."""
        model = ACT(TEST_CONFIG)
        model.train()
        cfg = TEST_CONFIG
        images = torch.rand(BATCH, cfg["num_cameras"], 3, cfg["img_h"], cfg["img_w"])
        joint_pos = torch.rand(BATCH, cfg["joint_dim"])
        actions = torch.rand(BATCH, cfg["chunk_size"], cfg["joint_dim"])
        pred_actions, mu, log_var = model(images, joint_pos, actions=actions)
        self.assertEqual(pred_actions.shape, (BATCH, cfg["chunk_size"], cfg["joint_dim"]))
        self.assertEqual(mu.shape, (BATCH, cfg["latent_dim"]))
        self.assertEqual(log_var.shape, (BATCH, cfg["latent_dim"]))

    def test_act_full_forward_inference(self):
        """ACT inference forward: z=0, pred_actions has correct shape, mu is all zeros."""
        model = ACT(TEST_CONFIG)
        model.eval()
        cfg = TEST_CONFIG
        images = torch.rand(BATCH, cfg["num_cameras"], 3, cfg["img_h"], cfg["img_w"])
        joint_pos = torch.rand(BATCH, cfg["joint_dim"])
        with torch.no_grad():
            pred_actions, mu, _ = model(images, joint_pos, actions=None)
        self.assertEqual(pred_actions.shape, (BATCH, cfg["chunk_size"], cfg["joint_dim"]))
        # At inference, mu should be all zeros (prior mean)
        self.assertTrue(torch.all(mu == 0), "mu should be zeros at inference")


# ---------------------------------------------------------------------------
# Loss tests
# ---------------------------------------------------------------------------

class TestLoss(unittest.TestCase):

    def _make_model_and_outputs(self):
        model = ACT(TEST_CONFIG)
        model.train()
        cfg = TEST_CONFIG
        images = torch.rand(BATCH, cfg["num_cameras"], 3, cfg["img_h"], cfg["img_w"])
        joint_pos = torch.rand(BATCH, cfg["joint_dim"])
        actions = torch.rand(BATCH, cfg["chunk_size"], cfg["joint_dim"])
        pred_actions, mu, log_var = model(images, joint_pos, actions=actions)
        return model, pred_actions, actions, mu, log_var

    def test_compute_loss_values(self):
        """Loss values are non-negative and total ≈ recon + beta * kl."""
        model, pred_actions, target_actions, mu, log_var = self._make_model_and_outputs()
        total_loss, recon_loss, kl_loss = model.compute_loss(
            pred_actions, target_actions, mu, log_var
        )
        self.assertGreaterEqual(total_loss.item(), 0.0, "total_loss must be non-negative")
        self.assertGreaterEqual(recon_loss.item(), 0.0, "recon_loss must be non-negative")
        self.assertTrue(torch.isfinite(kl_loss), "kl_loss must be finite")

        expected_total = recon_loss + TEST_CONFIG["beta"] * kl_loss
        self.assertAlmostEqual(total_loss.item(), expected_total.item(), places=5)

    def test_kl_loss_zero_when_mu_zero_logvar_zero(self):
        """KL divergence should be ~0 when mu=0 and log_var=0 (z = N(0,1) = prior)."""
        model = ACT(TEST_CONFIG)
        B = BATCH
        mu = torch.zeros(B, TEST_CONFIG["latent_dim"])
        log_var = torch.zeros(B, TEST_CONFIG["latent_dim"])
        pred_actions = torch.rand(B, TEST_CONFIG["chunk_size"], TEST_CONFIG["joint_dim"])
        target_actions = torch.rand(B, TEST_CONFIG["chunk_size"], TEST_CONFIG["joint_dim"])
        _, _, kl_loss = model.compute_loss(pred_actions, target_actions, mu, log_var)
        self.assertAlmostEqual(kl_loss.item(), 0.0, places=5)


# ---------------------------------------------------------------------------
# Gradient test
# ---------------------------------------------------------------------------

class TestGradients(unittest.TestCase):

    def test_backward_pass(self):
        """All trainable parameters should have gradients after a backward pass."""
        model = ACT(TEST_CONFIG)
        model.train()
        cfg = TEST_CONFIG
        images = torch.rand(BATCH, cfg["num_cameras"], 3, cfg["img_h"], cfg["img_w"])
        joint_pos = torch.rand(BATCH, cfg["joint_dim"])
        actions = torch.rand(BATCH, cfg["chunk_size"], cfg["joint_dim"])

        pred_actions, mu, log_var = model(images, joint_pos, actions=actions)
        total_loss, _, _ = model.compute_loss(pred_actions, actions, mu, log_var)
        total_loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(
                    param.grad,
                    f"Parameter '{name}' has no gradient after backward()"
                )


# ---------------------------------------------------------------------------
# Dataset test
# ---------------------------------------------------------------------------

class TestDataset(unittest.TestCase):

    def test_synthetic_dataset_shapes(self):
        """Dataset length and item shapes are correct."""
        cfg = TEST_CONFIG
        dataset = SyntheticDemoDataset(
            num_demos=3,
            traj_len=20,
            chunk_size=cfg["chunk_size"],
            joint_dim=cfg["joint_dim"],
            num_cameras=cfg["num_cameras"],
            img_h=cfg["img_h"],
            img_w=cfg["img_w"],
            seed=0,
        )
        # Total samples = num_demos * traj_len
        self.assertEqual(len(dataset), 3 * 20)

        image, joint_pos, action_chunk = dataset[0]
        self.assertEqual(image.shape, (cfg["num_cameras"], 3, cfg["img_h"], cfg["img_w"]))
        self.assertEqual(joint_pos.shape, (cfg["joint_dim"],))
        self.assertEqual(action_chunk.shape, (cfg["chunk_size"], cfg["joint_dim"]))

    def test_action_chunk_padding(self):
        """Last timestep's action chunk should be padded by repeating the final action."""
        cfg = TEST_CONFIG
        traj_len = 15
        dataset = SyntheticDemoDataset(
            num_demos=1,
            traj_len=traj_len,
            chunk_size=cfg["chunk_size"],
            joint_dim=cfg["joint_dim"],
            seed=1,
        )
        # Last timestep: idx = traj_len - 1, only 1 action available before end
        last_idx = traj_len - 1
        _, _, chunk = dataset[last_idx]
        self.assertEqual(chunk.shape, (cfg["chunk_size"], cfg["joint_dim"]))
        # All padded steps should equal the last real action
        self.assertTrue(torch.all(chunk == chunk[-1]))


# ---------------------------------------------------------------------------
# Integration tests (subprocess)
# ---------------------------------------------------------------------------

class TestIntegration(unittest.TestCase):
    """Run train.py and inference.py as subprocesses to catch import/runtime errors."""

    ACT_DIR = Path(__file__).parent.parent

    def test_dry_run_train(self):
        """Dry-run training completes without error and saves a checkpoint."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp_path = f.name

        result = subprocess.run(
            [
                sys.executable, str(self.ACT_DIR / "train.py"),
                "--dry-run",
                "--save-model",
                "--model-path", tmp_path,
                "--num-demos", "5",
                "--traj-len", "10",
                "--hidden-dim", "64",
                "--num-enc-layers", "2",
                "--num-dec-layers", "2",
            ],
            capture_output=True,
            text=True,
            cwd=str(self.ACT_DIR),
        )
        self.assertEqual(
            result.returncode, 0,
            f"train.py --dry-run failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        self.assertTrue(
            Path(tmp_path).exists(),
            f"Checkpoint not saved to {tmp_path}"
        )
        self._tmp_checkpoint = tmp_path

    def test_inference_from_checkpoint(self):
        """Inference runs to completion after a dry-run training checkpoint."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp_path = f.name

        # First create a checkpoint
        train_result = subprocess.run(
            [
                sys.executable, str(self.ACT_DIR / "train.py"),
                "--dry-run",
                "--save-model",
                "--model-path", tmp_path,
                "--num-demos", "5",
                "--traj-len", "10",
                "--hidden-dim", "64",
                "--num-enc-layers", "2",
                "--num-dec-layers", "2",
            ],
            capture_output=True,
            text=True,
            cwd=str(self.ACT_DIR),
        )
        self.assertEqual(train_result.returncode, 0, f"Train failed: {train_result.stderr}")

        # Then run inference
        infer_result = subprocess.run(
            [
                sys.executable, str(self.ACT_DIR / "inference.py"),
                "--model-path", tmp_path,
                "--num-steps", "5",
                "--no-cuda",
                "--no-mps",
            ],
            capture_output=True,
            text=True,
            cwd=str(self.ACT_DIR),
        )
        self.assertEqual(
            infer_result.returncode, 0,
            f"inference.py failed:\nSTDOUT:\n{infer_result.stdout}\nSTDERR:\n{infer_result.stderr}"
        )


if __name__ == "__main__":
    unittest.main()
