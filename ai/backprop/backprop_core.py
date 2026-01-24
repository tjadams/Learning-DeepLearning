from backprop_utils import print_gradient


def zero_gradients(model):
  for name, param in model.named_parameters():
    if param.grad is not None:
      print("Before:")
      print_gradient(name, param)
      
      # Fill the grad tensor with zeros that match its shape
      param.grad.zero_()

      print("After:")
      print_gradient(name, param)
