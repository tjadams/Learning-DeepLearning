import torch
import torch.nn as nn
import torch.nn.functional as F
from backprop_utils import print_gradient


def zero_gradients(model, debug=False):
  for name, param in model.named_parameters():
    if param.grad is not None:
      if debug:
        print("Before:")
        print_gradient(name, param, debug)

      # Fill the grad tensor with zeros that match its shape
      param.grad.zero_()

      if debug:
        print("After:")
        print_gradient(name, param, debug)


# Compute loss: negative log likelihood loss
########################################
########################################
# Output:
# values are logits like -2.3290
# torch.Size([64, 10])
# So an array with 64 rows and 10 columns.
# In ML terms, this often means: batch size = 64, classes = 10.
# E.g. -2.3290, -2.3698, -2.2902, -2.1538, -2.3503, -2.2076, ... 10 columns
########################################
########################################
# B) Target:
# values are the classes (digit 0-9)
# torch.Size([64]) so an array with 64 rows.
########################################
########################################
# Why the shapes of the output and target are different:
# 64 images go into the model. The model does 10-class classification.
# The forward pass of the model runs on each image to produce logits.
# Logits are produced for each of the classes.
# 1 image -> 10 logits (10 classes)
# 64 images * 10 classes, because doing the above 64 times.
########################################
# How to read the output tensor?
# Row i is the ith image.
# Column j is the logit for class j (e.g. classes 0-9), for the respective image.
# After doing softmax on the logits, you get the probability.
# So basically you'll find the one with the highest probability.
# Highest probability is the predicted class for that image
# Then you'll compare that with the target to see if it was accurate.
def compute_nll_loss(output, target):
  # Convert logits to probabilities
  output_probabilities = F.softmax(output, dim=1)

  # batch_size = 64
  batch_size = len(target)

  loss_per_image_in_batch = torch.zeros(batch_size)

  # Compute loss for each image in the batch
  # for i in range(0, batch_size):
  for i in range(batch_size):
    correct_class = target[i]

    probability_of_correct_class = output_probabilities[i][correct_class]

    loss_per_image_in_batch[i] = -1*torch.log(probability_of_correct_class)

  # Loss is basically an average of the loss_per_image_in_batch
  # loss = (1/batch_size) * sum(loss_per_image_in_batch)
  # Using sum is less efficient so use a tensor operation to do the average
  loss = loss_per_image_in_batch.mean()

  return loss

# More details at backward_pass.md
# For the most fundamental implementation, using Mini-batch Gradient Descent with plain (vanilla) SGD (Stochastic Gradient Descent)
# For every weight, take the derivative of the loss function with respect to that weight.
# This will be the gradient of the loss function with respect to that weight, a.k.a.
# the gradient of that weight.

# Hardcoded to MNIST Net for now, starting from last layer
  #  x = self.conv1(x)
  # x = F.relu(x)
  # x = self.conv2(x)
  # x = F.relu(x)
  # x = F.max_pool2d(x, 2)
  # x = self.dropout1(x)
  # x = torch.flatten(x, 1)
  # x = self.fc1(x)
  # x = F.relu(x)
  # x = self.dropout2(x)
  # x = self.fc2(x)
  # output = F.log_softmax(x, dim=1)


def view_model(model):
  # View all model parameters (weights and biases)
  for name, param in model.named_parameters():
    print(f"{name}: {param.data}")
    print(f"Shape: {param.shape}\n")

  # View weights/biases of specific layer
  print("conv1 weights:", model.conv1.weight.data)
  print("conv1 bias:", model.conv1.bias.data)
  print("fc1 weights:", model.fc1.weight.data)
  print("fc1 bias:", model.fc1.bias.data)
  print("fc2 weights:", model.fc2.weight.data)
  print("fc2 bias:", model.fc2.bias.data)

# Backpropagation is just the chain rule applied to a graph.
# Chain rule example: dL/dW = dL/dA * dA/dW (dA terms cancel out)
# We start with the loss and apply the chain rule to each layer to compute the gradients.
# We will later (outside of this function) use the gradients to update the weights of the model.


def backward_pass(model, loss, output, target, args):
  trainable_layers = []
  for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
      trainable_layers.append((name, module))

  # Process layers backwards, from last to first
  iterate_over_layers = reversed(trainable_layers)
  kept_layers = []

  # Expect to see this in iterate_over_layers for MNIST:
  # 1st element: Linear(128, 10)
  # 2nd element: Linear(9216, 128)
  # 3rd element: Conv2d(32, 64, 3, 1)
  # 4th element: Conv2d(1, 32, 3, 1)

  # Actual output looks right except for
  # 1. The shape elements are in different positions, not sure if that matters
  # 2. Shapes for conv2ds are a bit unexpected
  # Processing layer: fc2 (Linear)
  # Layer has weights with shape: torch.Size([10, 128])
  # Processing layer: fc1 (Linear)
  # Layer has weights with shape: torch.Size([128, 9216])
  # Processing layer: conv2 (Conv2d)
  # Layer has weights with shape: torch.Size([64, 32, 3, 3])
  # Processing layer: conv1 (Conv2d)
  # Layer has weights with shape: torch.Size([32, 1, 3, 3])

  for name, module in iterate_over_layers:
    layer_has_weight = hasattr(module, 'weight') and module.weight is not None
    layer_has_bias = hasattr(module, 'bias') and module.bias is not None

    if layer_has_weight and layer_has_bias:
      kept_layers.append((name, module))

  for name, module in kept_layers:
    print(f"Processing layer: {name} ({type(module).__name__})")
    print(f"Layer has weights with shape: {module.weight.shape}")

  # TODO: Compute gradients for each layer (linear, linear, conv2d, conv2d)

  # Step 1: Gradient w.r.t. loss itself = 1.0
  # f(x) = x, so f'(x) = 1.0
  # dL/dL = 1.0
  grad_loss = 1.0
