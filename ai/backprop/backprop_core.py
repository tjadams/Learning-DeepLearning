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
  probabilities = F.softmax(output, dim=1)

  # Can have this be an arg
  batch_size = 64

  loss_per_image_in_batch = torch.zeros(batch_size)

  # Compute loss for each image in the batch
  for i in range(0, batch_size):
    correct_class = target[i]

    probability_predicted = output[i][correct_class]

    loss_per_image_in_batch[i] = -1*torch.log(probability_predicted)
    

  # TODO: calculate loss based on loss_per_image_in_batch

  return loss
