import argparse
import torch
from shared import get_device, setup_data_loaders, test
import os
from net import Net


def main():
  """Main entry point for inference."""
  args = parse_args()

  device = get_device(args.no_cuda, args.no_mps)
  use_cuda = not args.no_cuda and torch.cuda.is_available()
  _, test_loader = setup_data_loaders(64, args.test_batch_size, use_cuda)

  print("Loading model from file...")
  model = load_model(args.model_path, device)

  print("Calling inference on model:")
  test(model, device, test_loader)


def parse_args():
  """Parse command line arguments for inference."""
  parser = argparse.ArgumentParser(description='PyTorch MNIST Inference')
  parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                      help='input batch size for testing (default: 1000)')
  parser.add_argument('--no-cuda', action='store_true', default=False,
                      help='disables CUDA training')
  parser.add_argument('--no-mps', action='store_true', default=False,
                      help='disables macOS GPU training')
  parser.add_argument('--model-path', type=str, default='../artifacts/mnist_cnn_backup.pt',
                      help='path to the model file (default: ../artifacts/mnist_cnn_backup.pt)')
  return parser.parse_args()


def load_model(model_path, device):
  """Load the model from a file path."""
  if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

  model = Net().to(device)
  model.load_state_dict(torch.load(model_path, map_location=device))
  return model


if __name__ == '__main__':
  main()
