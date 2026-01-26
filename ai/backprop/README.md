# Backpropagation
This folder is about building backprop from scratch in PyTorch for learning purposes.

Manually created a conda env ahead of time.

## Usage - execution:
0. `conda activate pytorch`
1. `cd ai/backprop/`
2. Execute: `python mnist-example/train.py --backprop-from-scratch --debug-logs`

## Usage - unit tests:
0. `conda activate pytorch`
1. `cd ai/backprop/tests/`
2. Unit test: `python -m unittest test_backprop_core.py -v`

## Next steps: close TODOs in train.py step-by-step

## Notes
### Loss 
Recap on loss: 
- Loss functions measure “how wrong” the NN prediction is when compared to target data.
- High loss means the NN's prediction is very wrong
- Low loss means the NN's prediction is accurate. We want low loss

Why NLL for MNIST?
- We use NLL for this neural network because MNIST is a classification problem (predict 1 of 10 digits), and NLL works well on classification problems


How does one pick a loss function?
- Picking a loss function depends on 
  - task type (regression, classification, etc) 
  - and what behavior you want to encourage (robustness to outliers, calibrated probabilities, margin separation, etc.).

What does NLL look like?
- NLL = -log(P(X)), where P(x) = the probability that the NN classifies the correct digit. Log base epsilon, like "lawn" ln function. The ln function is the most common when we say log in ML libraries
- Let's look at a case where the classification is accurate: NLL = -log(0.9) = ~0.1. That is, if the probability that the NN classifies the correct digit is 0.9 a.k.a. 90%, then the loss will be low which is what we want for accurate predictions
- Let's look at a case where the classification is inaccurate: NLL = -log(0.01) = ~4.6. So if the probability that the NN classifies the correct digit is 0.01, a.k.a. 1%, then the loss will be high which is what we want for inaccurate predictions

How to calculate NLL?
<!-- TODO:  -->