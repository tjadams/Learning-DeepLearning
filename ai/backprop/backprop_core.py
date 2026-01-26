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

import torch.nn.functional as F

# Compute loss: negative log likelihood loss
def compute_nll_loss(output, target):
  # TODO:
  # return target
  loss = F.nll_loss(output, target)
  return loss