import unittest
import torch
import torch.nn as nn

import sys
from pathlib import Path

# Add parent directory to path to allow importing backprop modules
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from backprop_core import zero_gradients, compute_nll_loss, backward_pass

# Add mnist-example to path so we can import Net
mnist_dir = Path(__file__).parent.parent / 'mnist-example'
sys.path.insert(0, str(mnist_dir))


class SimpleModel(nn.Module):
  """Simple model for testing purposes."""

  def __init__(self):
    super(SimpleModel, self).__init__()
    self.linear1 = nn.Linear(10, 5)
    self.linear2 = nn.Linear(5, 2)

  def forward(self, x):
    x = torch.relu(self.linear1(x))
    x = self.linear2(x)
    return x


class TestZeroGradients(unittest.TestCase):
  """Unit tests for the zero_gradients function."""

  def test_zero_gradients(self):
    """Test that zero_gradients sets all gradients to zero."""
    # Create a simple model
    model = SimpleModel()

    # Create dummy input and compute a loss to generate gradients
    x = torch.randn(4, 10)
    target = torch.randn(4, 2)

    # Forward pass
    output = model(x)
    loss = nn.MSELoss()(output, target)

    # Backward pass to generate gradients
    loss.backward()

    # Verify gradients exist and are non-zero before zeroing
    has_gradients = False
    gradients_before = {}
    for name, param in model.named_parameters():
      if param.grad is not None:
        has_gradients = True
        gradients_before[name] = param.grad.clone()
        # Verify gradient is not all zeros
        self.assertFalse(
            torch.allclose(param.grad, torch.zeros_like(param.grad)),
            f"Gradient for {name} is already zero before zero_gradients call"
        )

    self.assertTrue(has_gradients, "No gradients were generated")

    # Call zero_gradients
    zero_gradients(model)

    # Verify all gradients are now zero
    for name, param in model.named_parameters():
      if param.grad is not None:
        self.assertTrue(
            # torch.zeros_like creates a tensor of zeros 
            # with the same shape and dtype as param.grad

            # torch.allclose checks if 2 tensors are element-wise
            # close in value, within some tolerance
            # default rtol=1e-05, atol=1e-08)

            # Using allclose instead of == because of floating 
            # point precision

            torch.allclose(param.grad, torch.zeros_like(param.grad)),
            f"Gradient for {name} is not zero after zero_gradients call"
        )
        # Verify the shape is preserved
        self.assertEqual(
            param.grad.shape, gradients_before[name].shape,
            f"Gradient shape for {name} changed after zero_gradients"
        )

  def test_zero_gradients_with_no_gradients(self):
    """Test that zero_gradients handles models with no gradients gracefully."""
    model = SimpleModel()

    # Don't call backward, so no gradients exist
    # This should not raise an error
    zero_gradients(model)

    # Verify no gradients exist
    for name, param in model.named_parameters():
      self.assertIsNone(
          param.grad,
          f"Gradient for {name} should be None"
      )


class TestComputeNllLoss(unittest.TestCase):
  """Unit tests for the compute_nll_loss function."""

  def test_compute_nll_loss(self):
    """Test that compute_nll_loss computes negative log likelihood loss correctly."""
    # Create dummy log probabilities (output from log_softmax)
    # Shape: [batch_size, num_classes]
    batch_size = 4
    num_classes = 10
    output = torch.randn(batch_size, num_classes)
    # Apply log_softmax to get proper log probabilities
    output = torch.nn.functional.log_softmax(output, dim=1)

    # Create target class indices
    # Shape: [batch_size]
    target = torch.randint(0, num_classes, (batch_size,))

    # Call compute_nll_loss
    loss = compute_nll_loss(output, target)

    # Verify loss is a tensor
    self.assertIsInstance(loss, torch.Tensor, "Loss should be a tensor")

    # Verify loss is a scalar (0-dimensional tensor)
    self.assertEqual(loss.dim(), 0, "Loss should be a scalar")

    # Verify loss matches direct PyTorch call
    expected_loss = torch.nn.functional.nll_loss(output, target)
    self.assertTrue(
        torch.allclose(loss, expected_loss),
        f"Loss {loss.item()} does not match expected loss {expected_loss.item()}"
    )

    # Verify loss is non-negative (NLL loss should always be >= 0)
    self.assertGreaterEqual(
        loss.item(), 0.0,
        "NLL loss should be non-negative"
    )

  def test_compute_nll_loss_different_shapes(self):
    """Test compute_nll_loss with different batch sizes and number of classes."""
    # Test with different batch size
    output1 = torch.nn.functional.log_softmax(torch.randn(8, 5), dim=1)
    target1 = torch.randint(0, 5, (8,))
    loss1 = compute_nll_loss(output1, target1)

    # Test with different number of classes
    output2 = torch.nn.functional.log_softmax(torch.randn(2, 3), dim=1)
    target2 = torch.randint(0, 3, (2,))
    loss2 = compute_nll_loss(output2, target2)

    # Verify both are valid scalar tensors
    self.assertEqual(loss1.dim(), 0, "Loss 1 should be a scalar")
    self.assertEqual(loss2.dim(), 0, "Loss 2 should be a scalar")
    self.assertGreaterEqual(loss1.item(), 0.0, "Loss 1 should be non-negative")
    self.assertGreaterEqual(loss2.item(), 0.0, "Loss 2 should be non-negative")


class TestBackwardPass(unittest.TestCase):
  """Compare manual backward_pass gradients against PyTorch autograd."""

  def test_gradients_match_autograd(self):
    """All 8 parameter gradients (4 layers x weight+bias) should match PyTorch autograd."""
    from net import Net

    torch.manual_seed(42)

    # Use eval mode so dropout is disabled — this makes the comparison deterministic.
    # In training mode, our manual dropout mask would differ from PyTorch's internal one.
    model = Net()
    model.eval()

    # Create a random batch of "images" and targets
    batch_size = 8
    data = torch.randn(batch_size, 1, 28, 28)
    target = torch.randint(0, 10, (batch_size,))

    # --- Run our manual backward pass ---
    output = model(data)
    loss = compute_nll_loss(output, target)
    backward_pass(model, loss, output, target, args=None)

    # Save our manually computed gradients
    manual_grads = {}
    for name, param in model.named_parameters():
      self.assertIsNotNone(param.grad, f"Manual backward_pass didn't set grad for {name}")
      manual_grads[name] = param.grad.clone()

    # --- Run PyTorch autograd for comparison ---
    # Zero all gradients first
    for param in model.parameters():
      param.grad = None

    # Re-run forward pass and use PyTorch's autograd backward
    # We need to recompute using the same loss function path.
    # compute_nll_loss takes log_softmax output and applies softmax + -log,
    # so we replicate that same path here with autograd enabled.
    output2 = model(data)
    loss2 = compute_nll_loss(output2, target)
    loss2.backward()

    # --- Compare all 8 parameter gradients ---
    param_names = [name for name, _ in model.named_parameters()]
    self.assertEqual(len(param_names), 8, "Expected 8 parameters (4 layers x weight+bias)")

    for name in param_names:
      param = dict(model.named_parameters())[name]
      self.assertIsNotNone(param.grad, f"Autograd didn't set grad for {name}")

      manual = manual_grads[name]
      autograd = param.grad

      self.assertEqual(manual.shape, autograd.shape,
                       f"Shape mismatch for {name}: manual {manual.shape} vs autograd {autograd.shape}")

      matches = torch.allclose(manual, autograd, atol=1e-5, rtol=1e-4)
      if not matches:
        max_diff = (manual - autograd).abs().max().item()
        mean_diff = (manual - autograd).abs().mean().item()
        self.fail(
          f"Gradient mismatch for {name}:\n"
          f"  max abs diff: {max_diff:.6e}\n"
          f"  mean abs diff: {mean_diff:.6e}\n"
          f"  manual grad range: [{manual.min().item():.6e}, {manual.max().item():.6e}]\n"
          f"  autograd range: [{autograd.min().item():.6e}, {autograd.max().item():.6e}]"
        )


if __name__ == "__main__":
  unittest.main()
