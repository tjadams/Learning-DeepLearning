def print_gradients(model):
  for name, param in model.named_parameters():
    if param.grad is not None:
      print_gradient(name, param)


def print_gradient(name, param):
  print(f"{name}: gradient shape = {param.grad.shape}")
  print(f"{name}: gradient values = {param.grad}")
