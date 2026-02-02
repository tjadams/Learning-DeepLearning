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


 # What is the intuition behind updating the weights after the gradient calculation?
 - Recall from calculus that given an input value x and the derivative at that point f'(x), we can estimate the value of the function f(x) at a nearby point x + delta_x (change in x) using the derivative, as follows: f(x + delta_x) = f(x) + f'(x) * delta_x
- So f'(x) * delta_x will give us how much we need to add to f(x) in order to reach f(x + delta_x)





# How does optimization work in Robotics?
 - You are always minimizing some objective:
 - a. prediction error
 - b. action mismatch
 - c. negative reward
 - d. energy + task cost