import torch
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

# Backpropagation is just the chain rule applied to a graph.
# Chain rule example: dL/dW = dL/dA * dA/dW (dA terms cancel out)
# We start with the loss and apply the chain rule to each layer to compute the gradients.
# We will later (outside of this function) use the gradients to update the weights of the model.

# -  trainable_layers = []
# -  for name, module in model.named_modules():
# -    if isinstance(module, (nn.Conv2d, nn.Linear)):
# -      trainable_layers.append((name, module))
# -
# -  # Process layers backwards, from last to first
# -  iterate_over_layers = reversed(trainable_layers)
# -  kept_layers = []
# -
# -  # Expect to see this in iterate_over_layers for MNIST:
# -  # 1st element: Linear(128, 10)
# -  # 2nd element: Linear(9216, 128)
# -  # 3rd element: Conv2d(32, 64, 3, 1)
# -  # 4th element: Conv2d(1, 32, 3, 1)
# -
# -  # Actual output looks right except for
# -  # 1. The shape elements are in different positions, not sure if that matters
# -  # 2. Shapes for conv2ds are a bit unexpected
# -  # Processing layer: fc2 (Linear)
# -  # Layer has weights with shape: torch.Size([10, 128])
# -  # Processing layer: fc1 (Linear)
# -  # Layer has weights with shape: torch.Size([128, 9216])
# -  # Processing layer: conv2 (Conv2d)

# -  # Layer has weights with shape: torch.Size([32, 1, 3, 3])
# -
# -  for name, module in iterate_over_layers:
# -    layer_has_weight = hasattr(module, 'weight') and module.weight is not None
# -    layer_has_bias = hasattr(module, 'bias') and module.bias is not None
# -
# -    if layer_has_weight and layer_has_bias:
# -      kept_layers.append((name, module))
# -
# -  for name, module in kept_layers:
# -    print(f"Processing layer: {name} ({type(module).__name__})")
# -    print(f"Layer has weights with shape: {module.weight.shape}")
# -
# -  # TODO: Compute gradients for each layer (linear, linear, conv2d, conv2d)
# -
# -  # Step 1: Gradient w.r.t. loss itself = 1.0
# -  # f(x) = x, so f'(x) = 1.0
# -  # dL/dL = 1.0
# -  grad_loss = 1.0


def backward_pass(model, loss, output, target, args):
  cache = model.cache
  batch_size = target.shape[0]

  ##############################################################################
  # Phase A: Loss Gradient — combined softmax + NLL (cross-entropy)
  ##############################################################################
  # Our forward pass ends with: logits -> log_softmax -> NLL loss
  # compute_nll_loss does: softmax(log_softmax(logits)) then -log of correct class
  # softmax(log_softmax(x)) = softmax of log-probs, which is NOT the same as softmax(x).
  #
  # But the key insight: the output we passed to compute_nll_loss is log_softmax(logits),
  # and compute_nll_loss applies softmax to that. So the full chain from raw logits is:
  #   logits -> log_softmax -> softmax -> -log(p[target]) -> mean
  #
  # To match PyTorch autograd (which backprops through log_softmax + nll_loss),
  # we compute the gradient of NLL(log_softmax(logits)) w.r.t. logits.
  #
  # For log_softmax output s_i = log(softmax(logits))_i, NLL loss = -s[target] / batch_size
  # The gradient of NLL w.r.t. log-softmax output s is:
  #   dL/ds_i = -1/batch_size if i == target, else 0
  #
  # The Jacobian of log_softmax w.r.t. logits z is:
  #   d(s_i)/d(z_j) = delta(i,j) - softmax(z)_j
  #
  # Combining via chain rule: dL/dz = softmax(z) - one_hot(target)  (divided by batch_size)
  # This is the classic cross-entropy gradient formula.
  #
  # We use the raw logits (fc2_out) saved before log_softmax was applied.
  logits = cache['fc2_out']                             # [batch, 10]
  softmax_probs = F.softmax(logits, dim=1)              # [batch, 10]

  # one_hot: a tensor of zeros with a 1 at the correct class position
  # e.g. target=3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  one_hot_target = torch.zeros_like(softmax_probs)      # [batch, 10]
  one_hot_target.scatter_(1, target.unsqueeze(1), 1.0)

  # The gradient of the loss w.r.t. the raw logits (fc2 output)
  # This is the starting point — we propagate this backward through all layers.
  # Divide by batch_size because the loss is a mean over the batch.
  grad = (softmax_probs - one_hot_target) / batch_size  # [batch, 10]

  ##############################################################################
  # Phase B: FC2 Backward (Linear layer: y = x @ W^T + b)
  ##############################################################################
  # For a linear layer y = x @ W^T + b:
  #   dL/dW = grad^T @ x     (weight gradient: how loss changes when weights change)
  #   dL/db = grad.sum(dim=0) (bias gradient: sum over batch dimension)
  #   dL/dx = grad @ W        (input gradient: passed backward to previous layer)
  #
  # fc2_input is the input that was fed into fc2 during the forward pass.
  fc2_input = cache['fc2_input']                        # [batch, 128]

  # Weight gradient: [10, batch] @ [batch, 128] -> [10, 128]
  # Each row of the result is the gradient for one output neuron's weights
  model.fc2.weight.grad = grad.t() @ fc2_input

  # Bias gradient: sum the per-sample gradients across the batch
  # [batch, 10] -> [10]
  model.fc2.bias.grad = grad.sum(dim=0)

  # Input gradient: propagate gradient backward through fc2
  # [batch, 10] @ [10, 128] -> [batch, 128]
  grad = grad @ model.fc2.weight

  ##############################################################################
  # Phase C: Dropout2 Backward
  ##############################################################################
  # Dropout during forward: x_out = x_in * mask (mask includes the 1/(1-p) scaling)
  # Backward: just multiply gradient by the same mask.
  # In eval mode, mask is all ones so this is a no-op.
  grad = grad * cache['dropout2_mask']                  # [batch, 128]

  ##############################################################################
  # Phase D: ReLU Backward (after fc1)
  ##############################################################################
  # ReLU forward: y = max(0, x)
  # ReLU backward: gradient passes through where input > 0, is zeroed where input <= 0
  # This is because the derivative of max(0, x) is 1 if x > 0, else 0.
  # fc1_out is the pre-ReLU value (the raw fc1 output before ReLU was applied).
  relu_mask = (cache['fc1_out'] > 0).float()            # [batch, 128]
  grad = grad * relu_mask                               # [batch, 128]

  ##############################################################################
  # Phase E: FC1 Backward (Linear layer)
  ##############################################################################
  # Same pattern as FC2 backward.
  fc1_input = cache['flatten_out']                      # [batch, 9216]

  # Weight gradient: [128, batch] @ [batch, 9216] -> [128, 9216]
  model.fc1.weight.grad = grad.t() @ fc1_input

  # Bias gradient: [batch, 128] -> [128]
  model.fc1.bias.grad = grad.sum(dim=0)

  # Input gradient: [batch, 128] @ [128, 9216] -> [batch, 9216]
  grad = grad @ model.fc1.weight

  ##############################################################################
  # Phase F: Unflatten
  ##############################################################################
  # Flatten during forward reshaped [batch, 64, 12, 12] -> [batch, 9216]
  # Unflatten is the inverse: just reshape the gradient back to the original shape.
  # No math needed — flatten doesn't change values, just reshapes them.
  grad = grad.view(cache['pre_flatten_shape'])           # [batch, 64, 12, 12]

  ##############################################################################
  # Phase G: Dropout1 Backward
  ##############################################################################
  # Same as Phase C — multiply by the saved mask.
  grad = grad * cache['dropout1_mask']                   # [batch, 64, 12, 12]

  ##############################################################################
  # Phase H: MaxPool2d Backward
  ##############################################################################
  # MaxPool forward: picks the maximum value from each 2x2 window.
  # Backward: the gradient goes ONLY to the position that was the maximum.
  #           All other positions in the 2x2 window get zero gradient.
  # F.max_unpool2d does exactly this: it places gradient values at the positions
  # indicated by pool_indices, and fills the rest with zeros.
  # This expands [batch, 64, 12, 12] back to [batch, 64, 24, 24].
  grad = F.max_unpool2d(
    grad,
    cache['pool_indices'],
    kernel_size=2,
    output_size=cache['relu2_out'].shape                 # [batch, 64, 24, 24]
  )

  ##############################################################################
  # Phase I: ReLU Backward (after conv2)
  ##############################################################################
  # Same pattern as Phase D: zero gradient where pre-ReLU input was <= 0.
  relu_mask = (cache['conv2_out'] > 0).float()           # [batch, 64, 24, 24]
  grad = grad * relu_mask                                # [batch, 64, 24, 24]

  ##############################################################################
  # Phase J: Conv2 Backward (Conv2d)
  ##############################################################################
  # Conv2d forward: output = conv(input, weight) + bias
  # The input to conv2 was relu1_out (the output of ReLU after conv1).
  #
  # For convolution, computing gradients is more complex than linear layers because
  # weights are shared across spatial positions. We use three techniques:
  #
  # 1. Bias gradient: sum over batch and spatial dimensions
  #    Each output channel has one bias value added to every spatial position,
  #    so its gradient is the sum of all gradients at those positions.
  #
  # 2. Weight gradient (using unfold/im2col):
  #    Convolution can be expressed as matrix multiplication by "unfolding" the input
  #    into columns (im2col). Each column contains the values from one receptive field.
  #    F.unfold extracts all 3x3 patches from the input, laid out as columns.
  #    Then: weight_grad = sum_over_batch(grad_reshaped @ input_unfolded^T)
  #
  # 3. Input gradient: F.conv_transpose2d (transposed/deconvolution)
  #    This is the adjoint operation of convolution — it spreads gradients back
  #    to all input positions that contributed to each output position.

  conv2_input = cache['relu1_out']                       # [batch, 32, 26, 26]

  # Bias gradient: sum over batch (dim 0) and spatial dims (dims 2, 3)
  # [batch, 64, 24, 24] -> [64]
  model.conv2.bias.grad = grad.sum(dim=(0, 2, 3))

  # Weight gradient using unfold (im2col approach):
  # Step 1: Unfold the input into patches matching the kernel size
  # Each 3x3 patch of the input is stretched into a column of length C_in * k * k = 32*3*3 = 288
  # There are H_out * W_out = 24*24 = 576 such patches per image
  input_unfold = F.unfold(conv2_input, kernel_size=3)    # [batch, 288, 576]

  # Step 2: Reshape the gradient to combine spatial dims
  grad_reshaped = grad.reshape(batch_size, 64, -1)       # [batch, 64, 576]

  # Step 3: Batch matrix multiply: for each image in the batch,
  # multiply the gradient (how much each output position contributed to loss)
  # by the unfolded input (what input values were at each position).
  # This gives the weight gradient contribution from each image.
  # [batch, 64, 576] @ [batch, 576, 288] -> [batch, 64, 288]
  weight_grad = torch.bmm(grad_reshaped, input_unfold.transpose(1, 2))

  # Step 4: Sum over batch and reshape to kernel shape [C_out, C_in, k, k]
  model.conv2.weight.grad = weight_grad.sum(dim=0).view(model.conv2.weight.shape)

  # Input gradient: use transposed convolution to spread gradients back to input space
  # conv_transpose2d is the mathematical adjoint of conv2d — it reverses the operation.
  # [batch, 64, 24, 24] -> [batch, 32, 26, 26]
  grad = F.conv_transpose2d(grad, model.conv2.weight)

  ##############################################################################
  # Phase K: ReLU Backward (after conv1)
  ##############################################################################
  relu_mask = (cache['conv1_out'] > 0).float()           # [batch, 32, 26, 26]
  grad = grad * relu_mask                                # [batch, 32, 26, 26]

  ##############################################################################
  # Phase L: Conv1 Backward
  ##############################################################################
  # Same pattern as Phase J. We don't need the input gradient after this
  # because conv1 is the first layer (nothing before it has learnable parameters).

  conv1_input = cache['input']                           # [batch, 1, 28, 28]

  # Bias gradient: [batch, 32, 26, 26] -> [32]
  model.conv1.bias.grad = grad.sum(dim=(0, 2, 3))

  # Weight gradient using unfold:
  # Unfold input into patches: [batch, 1*3*3, 26*26] = [batch, 9, 676]
  input_unfold = F.unfold(conv1_input, kernel_size=3)    # [batch, 9, 676]
  grad_reshaped = grad.reshape(batch_size, 32, -1)       # [batch, 32, 676]

  # [batch, 32, 676] @ [batch, 676, 9] -> [batch, 32, 9]
  weight_grad = torch.bmm(grad_reshaped, input_unfold.transpose(1, 2))

  # Sum over batch, reshape to [32, 1, 3, 3]
  model.conv1.weight.grad = weight_grad.sum(dim=0).view(model.conv1.weight.shape)

  # No need to compute input gradient — nothing before conv1 has parameters.

# Update Weights using gradients found in gradient descent
# `torch.no_grad()` prevents autograd from tracking the update on leaf tensors (required, else RuntimeError)
# `param -= ...` updates in-place. Short form for param.data = param.date - ...
# `args.lr` — learning rate from argparse (`--lr`, default `1.0`)
# Reuses `model.named_parameters()` pattern from `zero_gradients` (lines 7–8)
def update_weights(model, args):
  with torch.no_grad():
    for name, param in model.named_parameters():
      if param.grad is not None:
        param -= args.lr * param.grad
