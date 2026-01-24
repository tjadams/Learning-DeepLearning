import argparse
import sys
from pathlib import Path

# Add parent directory to path to allow importing backprop modules
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from net import Net
from utils import get_device, setup_data_loaders, test
from backprop_utils import print_gradients
from backprop_core import zero_gradients


def main():
  """Main entry point for training."""
  args = parse_args()

  # Init seed for reproducibility of random numbers, and therefore the results
  torch.manual_seed(args.seed)

  device = get_device(args.no_cuda, args.no_mps)
  use_cuda = not args.no_cuda and torch.cuda.is_available()
  train_loader, test_loader = setup_data_loaders(
      args.batch_size, args.test_batch_size, use_cuda)

  print("Starting training...")
  run_training(args, device, train_loader, test_loader)


def parse_args():
  """Parse command line arguments for training."""
  parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
  parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                      help='input batch size for training (default: 64)')
  parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                      help='input batch size for testing (default: 1000)')
  parser.add_argument('--epochs', type=int, default=14, metavar='N',
                      help='number of epochs to train (default: 14)')
  parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                      help='learning rate (default: 1.0)')
  parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                      help='Learning rate step gamma (default: 0.7)')
  parser.add_argument('--no-cuda', action='store_true', default=False,
                      help='disables CUDA training')
  parser.add_argument('--no-mps', action='store_true', default=False,
                      help='disables macOS GPU training')
  parser.add_argument('--dry-run', action='store_true', default=False,
                      help='quickly check a single pass')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
  parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                      help='how many batches to wait before logging training status')
  parser.add_argument('--save-model', action='store_true', default=False,
                      help='For Saving the current Model')
  parser.add_argument('--backprop-from-scratch', action='store_true', default=False,
                      help='For doing the internals of backprop from scratch')
  parser.add_argument('--debug-logs', action='store_true', default=False,
                      help='Enable debug logging for gradients')
  return parser.parse_args()


def run_training(args, device, train_loader, test_loader):
  """Init model and training loop variables, then run training."""
  # Init model and training loop variables
  # The model and data (data.to(device), target.to(device)) must be on the same
  # device otherwise the forward pass model(data) will fail with a device mismatch error
  model = Net().to(device)

  # Optimizer is for updating the model's weights (input: gradients, output: weights)
  # Can be changed to different optimizers, like SGD, Adam, etc.
  optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

  # Init a scheduler which will be used to adjust the learning rate hyperparam each epoch
  scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

  # Training loop
  for epoch in range(1, args.epochs + 1):
    # Each epoch, we train the model and test the latest version
    train(args, model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

    # Change the learning rate at the end of the epoch
    scheduler.step()

  if args.save_model:
    torch.save(model.state_dict(), "mnist_cnn.pt")


# TODO: remove optimizer param or use args to switch between using it or using the from-scratch backprop implementation
def train(args, model, device, train_loader, optimizer, epoch):
  """Training function for a single epoch."""
  # Set model to training mode (dropout is turned on)
  model.train()

  # Iterate over the training data loader
  for batch_idx, (data, target) in enumerate(train_loader):
    # Move data (training set) and target (validation set) to the device
    data, target = data.to(device), target.to(device)

    # Zero/clear the gradients from the previous batch.
    # Gradients accumulate by default so each backprop adds to the previous gradients
    print_gradients(model, args.debug_logs)
    if args.debug_logs:
      print("Starting to zero gradients...")
    if args.backprop_from_scratch:
      zero_gradients(model, args.debug_logs)
    else:
      optimizer.zero_grad()
    print_gradients(model, args.debug_logs)

    # Forward pass: compute predicted y by passing x to the model (calls nn.Module.forward via __call__ syntax. I defined forward in the Net class)
    output = model(data)

    # Compute the loss: negative log likelihood loss
    # TODO: compute loss from scratch
    loss = F.nll_loss(output, target)

    # Backward pass: compute gradients via backpropogation (gradients of the loss)
    # TODO: compute gradients from scratch
    loss.backward()

    # Update the model's weights (using gradient descent results stored in .grad field)
    # weight = weight - learning_rate * weight.grad
    # TODO: update model weights from scratch
    optimizer.step()

    # Log the training status every log_interval batches
    if batch_idx % args.log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item()))
      if args.dry_run:
        break


if __name__ == '__main__':
  main()
