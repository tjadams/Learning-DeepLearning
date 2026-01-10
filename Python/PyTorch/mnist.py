# This code is mostly copied from the PyTorch repo, as an example to learn.
# https://github.com/pytorch/examples/tree/main/mnist

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # Dropout is a regularization technique to prevent overfitting
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        # 10 outputs for the 10 MNIST prediction classes (digits 0-9)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # Log softmax to convert the output to a probability distribution over the 10 classes
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    # Set model to training mode (dropout is turned on)
    model.train() 

    # Iterate over the training data loader
    for batch_idx, (data, target) in enumerate(train_loader): 
        # Move data (training set) and target (validation set) to the device
        data, target = data.to(device), target.to(device)

        # Zero/clear the gradients from the previous batch.
        # Gradients accumulate by default so each backprop adds to the previous gradients
        optimizer.zero_grad()

        # Forward pass: compute predicted y by passing x to the model (calls nn.Module.forward via __call__ syntax. I defined forward in the Net class)
        output = model(data)

        # Compute the loss: negative log likelihood loss
        loss = F.nll_loss(output, target)

        # Backward pass: compute gradients via backpropogation (gradients of the loss)
        loss.backward()

        # Update the model's weights (using gradient descent results stored in .grad field)
        # weight = weight - learning_rate * weight.grad
        optimizer.step()

        # Log the training status every log_interval batches
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    # Set model to evaluation/testing/inference mode (dropout + batch norm turned off)
    model.eval()

    test_loss = 0
    correct = 0

    # Disable gradient calculation (no_grad) for faster inference. Not needed for inference, only training.
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)

            # Compute the loss: negative log likelihood loss, summed up over the batch
            # .item() is to get the scalar value of the loss (previously stored in a tensor)
            test_loss += F.nll_loss(output, target, reduction='sum').item()

            # Get the index of the max log-probability, a.k.a. what the model thinks is the highest probability MNIST digit, based on its training
            pred = output.argmax(dim=1, keepdim=True)  
            
            # pred: Shape [batch_size, 1] (indices of predicted classes).
            # target: Shape [batch_size] (true class labels).
            # target.view_as(pred): Reshapes target to match pred’s shape [batch_size, 1].
            # pred.eq(...): Element-wise equality, returns a boolean tensor [batch_size, 1] (True where equal, False otherwise).
            # .sum(): Sums the booleans (True=1, False=0), giving the count of correct predictions (scalar tensor).
            # .item(): Converts the scalar tensor to a Python int.
            # correct += ...: Adds that count to the running total.
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings parsed from command line arguments
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
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
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    # Init seed for reproducibility of random numbers, and therefore the results
    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Set up the data loaders for the training and test sets
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Transform param to datasets.MNIST: "transform (callable, optional): A function/
    # transform that  takes in an PIL image and returns a transformed version.
    # E.g, ``transforms.RandomCrop``"
    transform=transforms.Compose([
        transforms.ToTensor(),
        # I guess this is the mean and standard deviation of the MNIST dataset?.
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

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


if __name__ == '__main__':
    main()