# PyTorch
This folder is about PyTorch for learning purposes.

Manually created a conda env ahead of time.

## Usage - MNIST 
Training:
0. conda activate pytorch
1. cd Python/PyTorch/MNIST/
2. python mnist.py --save-model

<In Development> Inference:
0. conda activate pytorch
1. cd Python/PyTorch/MNIST/
2. python infer.py --model-file "mnist_cnn.pt"