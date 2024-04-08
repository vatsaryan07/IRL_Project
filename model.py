import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RewardNet(nn.Module):
    def __init__(self, feat_dim):
        super(RewardNet, self).__init__()
        self.fc1 = nn.Linear(feat_dim, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    # def backprop(self, X, grad):

    #     num_layers = 2                                           # Get the number of layers of our neural network
    #     batch_size = X.shape[1]

    #     Y_hat, outputs, linear_outputs = self.forward(X)  # Perform the forward pass on the neural network
    #     delta = [0.0] * num_layers                                      # Initialize the cell to store the error at level i
    #     # propagate error
    #     for i in reversed(range(num_layers)):                           # Iterate over all layers
    #         if i == num_layers - 1:
    #             delta[i] = grad.reshape(1, -1)

    #         else:
    #             linear_output = linear_outputs[i].copy()
    #             """
    #             ``derivative'' is an  n^{(i)} x 1 vector where element
    #             j \\in {1,..., n^{(i)}} is the derivative of the output of node j
    #             in layer i w.r.t. the input to that node. In other words, this
    #             term is the derivative of the activation function w.r.t. its
    #             inputs. We assume that the activation function for all non-final
    #             layers is ReLU, which is defined as

    #             output = f(input) =   / input   if  input >= 0
    #                                 \\  0       otherwise

    #             Therefore, the derivative is given by

    #                         /
    #             d output      \\   1   if  input >= 0
    #             -------- =    |
    #             d input       /   0   otherwise
    #                         \\
    #             """

    #             derivative = linear_output  # Get the output of the activation function
    #             # Compute the error term for layer i
    #             derivative[linear_output >= 0] = 1.0  # Compute the derivative for elements >= 0
    #             derivative[linear_output < 0] = 0.0  # Compute the derivative for elements < 0
    #             delta[i] = np.dot(nn[i + 1][0].T, delta[i + 1]) * derivative

    #     # Compute the gradients of all the neural network's weights using the error term, delta
    #     grad = defaultdict(list)  # Initialize a cell array, where cell i contains the gradients for the weight matrix in layer i of the neural network

    #     for i in range(num_layers):
    #         if i == 0:
    #             # Gradients for the first layer are calculated using the examples, X.
    #             grad[i].append(np.dot(delta[i], X.T))
    #             grad[i].append(np.dot(delta[i], np.ones((batch_size, 1))))
    #         else:
    #             # Gradients for subsequent layers are calculated using the output of the previous layers.
    #             grad[i].append(np.dot(delta[i], outputs[i - 1].T))
    #             grad[i].append(np.dot(delta[i], np.ones((batch_size, 1))))
    #         if np.isnan(grad[i][0]).any() or np.isnan(grad[i][1]).any():
    #             print("WARNING: Gradients/biases are nan")
    #             exit(-1)
    #     return grad