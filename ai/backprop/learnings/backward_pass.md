# Recap of what is a derivative?
 - A derivative is the rate of change at a point in a real-valued function.
 - The derivative f'(x) of function f() for variable x is the rate that the function f() changes at the point x.
  - A derivative looks like the slope of a line at a given point X.
 - https://machinelearningmastery.com/wp-content/uploads/2020/12/Tangent-Line-of-a-Function-at-a-Given-Point.png
 - The function might change a lot at point x, e.g. be very curved, or might change a little, e.g. slight curve, or it might not change at all, e.g. flat or stationary.
 - We can use derivatives in optimization problems as they tell us how to change inputs to the target function in a way that increases or decreases the output of the function, so we can get closer to the minimum or maximum of the function.
 - Once you say “minimize a function”, calculus enters the chat.

 # How is neural network training like optimization?
 - We want to find the weights that minimize the loss function.
 - We can use derivatives to tell us how to change the weights in a way that decreases the loss.
 - We're looking to minimize a function, so calculus is applicable.
 - The derivative tells us: If I nudge this weight a tiny bit, does the loss go up or down — and by how much?
 - That is, dL/dW (derivative of loss with respect to weight)
 - if dL/dW is positive, then increasing the weight will increase the loss, so we should decrease the weight
 - if dL/dW is negative, then increasing the weight will decrease the loss, so we should increase the weight
 - if dL/dW is 0, then the weight is already at the minimum, so we should leave it alone
 - We can use this to update the weights in a way that decreases the loss.
 - direction: does increasing W increase or decrease loss?
 - magnitude: how sensitive is the loss to that change in W?

 # How do we find the weights that minimize the loss function?
 - We can use gradient descent to find the weights that minimize the loss.
 - Gradient descent is a type of optimization algorithm that uses derivatives to find the minimum of a function.
 
# What is a gradient?
 - A gradient is a derivative of a function that has more than one input variable.
 - The gradient applies here because the loss function has multiple input variables (multiple weights as parameters)

 # How does gradient descent work?
- Intuitively, we need to pass the end result's (forward pass's) feedback towards the beginning layer
- So while we're going through end to front, we can calculate the gradients (and even update the weights in-place if we want, or we can do a separate end to front pass for that)
- In summary, the gradient is just the partial derivative of the Loss with respect to the respective weight that we're calculating the gradient for
- When we're at layer N, we do calculate the partial derivative of the loss w.r.t each weight in that layer (a.ka. gradient of the weight)
- Repeat for all layers
- To calculate the gradient of the weight:

# What's a spelled-out example of gradient descent?
- Network: x -> h -> y_hat -> L
- Network with weights : x -> (w1) -> h -> y_hat -> (w2) -> L
- w1: weight of x -> h
- w2: weight of h to y_hat
- Using MSE for Loss
- h = w1 * x
- y_hat = w2 * h
- L = (y_hat - y)^2
- y is the target data used for training, and y_hat is the forward pass result

- Gradient for w2 = dL/dw2 = dL/dy_hat * dy_hat/dw2 
- dL/dy_hat = 2*(y_hat - y), from calculus derivatives
- dy_hat/dw2 = 1*h = h, from calculus derivatives
- Subbing into the equation:
- Gradient for w2 = dL/dy_hat * dy_hat/dw2 = 2*(y_hat - y) * h = 2h(y_hat - y)
- Sub in h
- Gradient for w2 = 2*(w1*x)(y_hat - y) = 2*w1*x(y_hat - y)

- It makes sense for earlier weights (closer to network entrance, like w1) to have more factors in their gradient
- Gradient for w1 = dL/dw1 = <RHS>
- RHS is dL/dy_hat * dy_hat/dh * dh/dw1
- Options of terms to include: dL/dy_hat, dy_hat/dw2, dy_hat/dh, dh/dx, dh/dw1, dx/dw1
- dL/dy_hat makes sense because you need numerator to have dL and dy_hat is the output, so each term is needed
- You need some other term with dy_hat in numerator, and there are 2
- Why do we exclude dy_hat/dw2 and keep dy_hat/dh?
- Answer: we include every intermediate variables that lie on the path from w1 to L, excluding other weights. So that is: w1, h, y_hat, L. So from right to left: dL/dy_hat * dy_hat/dh * dh/dw1
- What do these terms mean - dL/dy_hat: how sensitive is the loss to the network output?
- What do these terms mean - dy_hat/dh: how sensitive is the network output to the hidden layer?
 - What do these terms mean - dh/dw1: how sensitive is the hidden layer to weight 1?



 # What is the intuition behind updating the weights after the gradient calculation?
 - Recall from calculus that given an input value x and the derivative at that point f'(x), we can estimate the value of the function f(x) at a nearby point x + delta_x (change in x) using the derivative, as follows: f(x + delta_x) = f(x) + f'(x) * delta_x
- So f'(x) * delta_x will give us how much we need to add to f(x) in order to reach f(x + delta_x)





# How does optimization work in Robotics?
 - You are always minimizing some objective:
 - a. prediction error
 - b. action mismatch
 - c. negative reward
 - d. energy + task cost