import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from net import Net


def get_device(no_cuda=False, no_mps=False):
    """Set up and return the appropriate device (CUDA, MPS, or CPU)."""
    use_cuda = not no_cuda and torch.cuda.is_available()
    use_mps = not no_mps and torch.backends.mps.is_available()

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    return device


def setup_data_loaders(batch_size, test_batch_size, use_cuda=False):
    """Set up the data loaders for the training and test sets."""
    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Transform param to datasets.MNIST: "transform (callable, optional): A function/
    # transform that  takes in an PIL image and returns a transformed version.
    # E.g, ``transforms.RandomCrop``"
    transform = transforms.Compose([
        transforms.ToTensor(),
        # I guess this is the mean and standard deviation of the MNIST dataset?.
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    
    return train_loader, test_loader


def test(model, device, test_loader):
    """Test/inference function for the model."""
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
            # target.view_as(pred): Reshapes target to match pred's shape [batch_size, 1].
            # pred.eq(...): Element-wise equality, returns a boolean tensor [batch_size, 1] (True where equal, False otherwise).
            # .sum(): Sums the booleans (True=1, False=0), giving the count of correct predictions (scalar tensor).
            # .item(): Converts the scalar tensor to a Python int.
            # correct += ...: Adds that count to the running total.
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
