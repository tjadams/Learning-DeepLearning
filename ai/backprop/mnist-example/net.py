import torch
import torch.nn as nn
import torch.nn.functional as F


# The following modules have weights and biases (parameters):
# Conv2D (convolutional)
# Linear (fully connected layer)

# These don't have weights or biases
# Dropout (randomly drops out elements)
# Anything from F. are functional only: relu, max_pool2d, flatten, log_softmax
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, 1)
    self.conv2 = nn.Conv2d(32, 64, 3, 1)
    # Dropout is a regularization technique to prevent overfitting
    self.dropout1 = nn.Dropout(0.25)
    self.dropout2 = nn.Dropout(0.5)
    self.fc1 = nn.Linear(9216, 128)
    # 10 outputs for the 10 MNIST prediction classes (digits 0-9)
    self.fc2 = nn.Linear(128, 10)

    # Cache for intermediate activations needed by backward pass
    self.cache = {}

  def forward(self, x):
    # Save the original input (needed for conv1 weight gradient)
    self.cache['input'] = x

    # Conv1: [batch, 1, 28, 28] -> [batch, 32, 26, 26]
    x = self.conv1(x)
    self.cache['conv1_out'] = x  # pre-ReLU (needed to compute ReLU backward mask)

    # ReLU1: zero out negatives
    x = F.relu(x)
    self.cache['relu1_out'] = x  # post-ReLU (this is the input to conv2)

    # Conv2: [batch, 32, 26, 26] -> [batch, 64, 24, 24]
    x = self.conv2(x)
    self.cache['conv2_out'] = x  # pre-ReLU

    # ReLU2
    x = F.relu(x)
    self.cache['relu2_out'] = x  # post-ReLU (needed for max_unpool output_size)

    # MaxPool2d: [batch, 64, 24, 24] -> [batch, 64, 12, 12]
    # return_indices=True gives us the positions of the max values,
    # which we need during backward pass to route gradients back to the correct positions
    x, pool_indices = F.max_pool2d(x, 2, return_indices=True)
    self.cache['pool_indices'] = pool_indices

    # TODO: why did Claude Code replace x = self.dropout1(x) with this?
    # Dropout1: randomly zero elements with p=0.25, scale remaining by 1/(1-p)
    # We implement dropout manually so we can save the mask for backward pass.
    # During eval (model.eval()), dropout is disabled (all elements kept).
    if self.training:
      # Create binary mask: 1 = keep, 0 = drop. Each element has (1-p) chance of being kept.
      dropout1_mask = (torch.rand_like(x) >= 0.25).float()
      # Inverted dropout: scale by 1/(1-p) so expected value is unchanged
      dropout1_mask = dropout1_mask / (1.0 - 0.25)
      x = x * dropout1_mask
      self.cache['dropout1_mask'] = dropout1_mask
    else:
      # In eval mode, dropout is identity. Mask of all ones means "keep everything".
      self.cache['dropout1_mask'] = torch.ones_like(x)

    # Flatten: [batch, 64, 12, 12] -> [batch, 9216]
    self.cache['pre_flatten_shape'] = x.shape  # save shape for unflatten in backward
    x = torch.flatten(x, 1)
    self.cache['flatten_out'] = x  # this is the input to fc1

    # FC1: [batch, 9216] -> [batch, 128]
    x = self.fc1(x)
    self.cache['fc1_out'] = x  # pre-ReLU

    # ReLU3
    x = F.relu(x)

    # TODO: why did Claude Code replace x = self.dropout2(x) with this?
    # Dropout2: randomly zero elements with p=0.5
    if self.training:
      dropout2_mask = (torch.rand_like(x) >= 0.5).float()
      dropout2_mask = dropout2_mask / (1.0 - 0.5)
      x = x * dropout2_mask
      self.cache['dropout2_mask'] = dropout2_mask
    else:
      self.cache['dropout2_mask'] = torch.ones_like(x)

    # Save the input to fc2 (post-ReLU, post-dropout2)
    self.cache['fc2_input'] = x

    # FC2: [batch, 128] -> [batch, 10]
    x = self.fc2(x)
    self.cache['fc2_out'] = x  # raw logits (before log_softmax)

    # Log softmax to convert the output to a probability distribution over the 10 classes
    output = F.log_softmax(x, dim=1)
    return output
