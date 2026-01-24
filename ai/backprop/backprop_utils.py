def print_gradients(model, debug=False):
  if not debug:
    return
  for name, param in model.named_parameters():
    if param.grad is not None:
      print_gradient(name, param, debug)


def print_gradient(name, param, debug=False):
  if not debug:
    return
  print(f"{name}: gradient shape = {param.grad.shape}")
  print(f"{name}: gradient values = {param.grad}")
