import numpy as np
from collections import defaultdict


def create_neural_net(numNodesPerLayer, numInputDims, numOutputDims):
    """
    This function takes as input the number of nodes per hidden layer as well
    as the size of the input and outputs of the neural network and returns a
    randomly initialized neural network. Weights for the network are
    generated randomly using the method of He et al. ICCV'15.

    Args:
        numNodesPerLayer: This list contains natural numbers for the quantity of nodes contained in each hidden layer.
        numInputDims: This number represents the cardinality of the input vector for the neural network.
        numOutputDims: This number represents the cardinality of the output vector of the neural network.

    Returns: the neural network created
    """
    nn = []
    num_layers = len(numNodesPerLayer)
    for i in range(num_layers + 1):
        if i == 0:
            # Use numInputDims for the input size
            nn.append([np.random.random((numNodesPerLayer[i], numInputDims)) * np.sqrt(2.0 / numInputDims), np.zeros((numNodesPerLayer[i], 1))])
        elif i == num_layers:
            # Use numOutputDims for the output size
            nn.append([np.random.random((numOutputDims, numNodesPerLayer[i-1])) * np.sqrt(2.0 / numNodesPerLayer[i-1]), np.zeros((numOutputDims, 1))])
        else:
            nn.append([np.random.random((numNodesPerLayer[i], numNodesPerLayer[i-1])) * np.sqrt(2.0 / numNodesPerLayer[i-1]), np.zeros((numNodesPerLayer[i], 1))])
    return nn


def forward_pass(nn, X, full_return=False):
    """
    This function takes as input a neural network, nn, and inputs, X, and
    performs a forward pass on the neural network. The code assumes that
    layers, {1,2,...,l-1}, have ReLU activations and the final layer has a
    linear activation (or softmax if final_softmax=True) function.

    Args:
        nn: The weights and biases of the neural network. nn[i][0] corresponds to the weights for the ith layer and
            nn[i][1] corresponds to the biases for the ith layer
        X: This matrix is d x n matrix and contains the input features for n states, each with d features.
        full_return: if True, returns all the intermediate results with final outputs; if False, returns final outputs.

    Returns:
        if full_return is True, returns all the intermediate results with final outputs;
        else, return 1 x n vector of predicted rewards for the n states in Deep MaxEnt.
    """
    assert isinstance(X, np.ndarray)
    assert len(X.shape) == 2

    num_layers = len(nn)             # Get the number of layers of our neural network

    linear_outputs = []  # Outputs after linear transformation but before activation for each hidden layer. To be used in backprop.
    outputs = []         # Outputs after activation for each hidden layer.
    for i in range(num_layers):
        # Compute the result of linear transformation of layer i
        if i == 0:
            # z[0] = dot(W[0].T, X) + b[0]   (0-> first layer)
            linear_outputs.append(np.matmul(nn[i][0], X) + nn[i][1])
        else:
            # Subsequent z computations use output from previous layer
            # z[i] = dot(W[i].T, output[i - 1]) + b[i]
            linear_outputs.append(np.matmul(nn[i][0], outputs[i-1]) + nn[i][1])

        # Compute the result after activation of layer i
        if i < num_layers-1:
            # If layer i is not the output layer, then apply the ReLU activation function for the nodes at this layer.
            outputs.append(linear_outputs[i].copy())
            outputs[i][outputs[i] < 0] = 0
        else:
            # Apply a linear activation (i.e., no activation)
            outputs.append(linear_outputs[i].copy())

    Y_hat = outputs[-1]
    if full_return:
        return Y_hat, outputs, linear_outputs
    else:
        return Y_hat


def backprop(nn, X, grad):
    """
    This function takes as input a neural network, nn, and input-output
    pairs, <X,Y>, and computes the gradient of the loss for the neural
    network's current predictions. The code assumes that layers,
    {1,2,...,l-1}, have ReLU activations and the final layer has a linear
    activation function.

    Args:
        nn: The weights and biases of the neural network. nn[i][0] corresponds to the weights for the ith layer and
            nn[i][1] corresponds to the biases for the ith layer
        X: This matrix is d x n matrix and contains the input features for n states, each with d features.
        grad: The difference in state visitation frequencies between the expert and the current policy

    Returns: A cell of dimension num_layers where num_layers is the number of layers in neural network, nn. grad[i][0]
             corresponds to the gradients for ith layer weights, and grad[i][1] corresponds to the gradient for ith
             layer bias.

    """
    assert isinstance(X, np.ndarray)
    assert len(X.shape) == 2

    num_layers = len(nn)                                            # Get the number of layers of our neural network
    batch_size = X.shape[1]

    Y_hat, outputs, linear_outputs = forward_pass(nn, X, full_return=True)  # Perform the forward pass on the neural network
    delta = [0.0] * num_layers                                      # Initialize the cell to store the error at level i
    # propagate error
    for i in reversed(range(num_layers)):                           # Iterate over all layers
        if i == num_layers - 1:
            delta[i] = grad.reshape(1, -1)

        else:
            linear_output = linear_outputs[i].copy()
            """
            ``derivative'' is an  n^{(i)} x 1 vector where element
             j \\in {1,..., n^{(i)}} is the derivative of the output of node j
             in layer i w.r.t. the input to that node. In other words, this
             term is the derivative of the activation function w.r.t. its
             inputs. We assume that the activation function for all non-final
             layers is ReLU, which is defined as

             output = f(input) =   / input   if  input >= 0
                                   \\  0       otherwise

             Therefore, the derivative is given by

                           /
             d output      \\   1   if  input >= 0
             -------- =    |
             d input       /   0   otherwise
                           \\
            """

            derivative = linear_output  # Get the output of the activation function
            # Compute the error term for layer i
            derivative[linear_output >= 0] = 1.0  # Compute the derivative for elements >= 0
            derivative[linear_output < 0] = 0.0  # Compute the derivative for elements < 0
            delta[i] = np.dot(nn[i + 1][0].T, delta[i + 1]) * derivative

    # Compute the gradients of all the neural network's weights using the error term, delta
    grad = defaultdict(list)  # Initialize a cell array, where cell i contains the gradients for the weight matrix in layer i of the neural network

    for i in range(num_layers):
        if i == 0:
            # Gradients for the first layer are calculated using the examples, X.
            grad[i].append(np.dot(delta[i], X.T))
            grad[i].append(np.dot(delta[i], np.ones((batch_size, 1))))
        else:
            # Gradients for subsequent layers are calculated using the output of the previous layers.
            grad[i].append(np.dot(delta[i], outputs[i - 1].T))
            grad[i].append(np.dot(delta[i], np.ones((batch_size, 1))))
        if np.isnan(grad[i][0]).any() or np.isnan(grad[i][1]).any():
            print("WARNING: Gradients/biases are nan")
            exit(-1)
    return grad
