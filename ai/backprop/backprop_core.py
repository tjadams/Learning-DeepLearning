def zero_gradients(model):
  for name, param in model.named_parameters():
    if param.grad is not None:
      param.grad._zero()
