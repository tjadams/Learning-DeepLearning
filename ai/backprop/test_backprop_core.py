import unittest
import torch
import torch.nn as nn
from backprop_core import zero_gradients


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


if __name__ == "__main__":
  unittest.main()
