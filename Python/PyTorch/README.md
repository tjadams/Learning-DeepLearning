# PyTorch
This folder is about PyTorch for learning purposes.

Manually created a conda env ahead of time.

## Usage - MNIST 
This code is mostly copied from the PyTorch repo, as an example to learn: https://github.com/pytorch/examples/tree/main/mnist.

Training:
0. conda activate pytorch
1. cd Python/PyTorch/MNIST/
2. python mnist.py --save-model

<In Development> Inference:
0. conda activate pytorch
1. cd Python/PyTorch/MNIST/
2. python mnist.py --load-model-and-infer
<!-- 2. python infer.py --model-file "mnist_cnn.pt" -->