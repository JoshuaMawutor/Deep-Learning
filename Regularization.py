import numpy as np


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def relu(x):
    s = np.maximum(0, x)
    return s


def tanh(x):
    s = np.tanh(x)
    return s


def tanh_prime(x):
    s = 1 - tanh(x) ** 2
    return s


def initialize_weights(np_x, np_h1, np_h2, np_y):
    W1 = np.random.randn(np_h1, np_x) * 0.01
    b1 = np.zeros((np_h1, 1))
    W2 = np.random.randn(np_h2, np_h1) * 0.01
    b2 = np.zeros((np_h2, 1))
    W3 = np.random.randn(np_y, np_h2) * 0.01
    b3 = np.zeros((np_y, 1))
    parameters = {'W1': W1, 'b1': b1, 'W2': W2,
                  'b2': b2, 'W3': W3, 'b3': b3}
    return parameters


def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    forward = {'Z1': Z1, 'A1': A1, 'Z2': Z2,
               'A2': A2, 'Z3': Z3, 'A3': A3}
    return A3, forward


def cost_function(A3, Y):
    m = Y.shape[1]
    cost = -1 / m * (np.dot(Y, np.log(A3).T) + np.dot(1 - Y, np.log(1 - A3).T))
    cost = np.squeeze(cost)
    return cost


def backward_propagation(X, Y, A3, parameters, forward):
    m = Y.shape[1]
    W3 = parameters['W3']
    W2 = parameters['W2']
    A2 = forward['A2']
    A1 = forward['A1']

    dZ3 = A3 - Y
    dW3 = 1 / m * np.dot(dZ3, A2.T)
    db3 = 1 / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = dA2 * np.int64(A2 > 0)
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * np.int64(A1 > 0)
    dW1 = 1 / m * np.dot(dZ1, X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    backward = {'dW1': dW1, 'db1': db1, 'dW2': dW2,
                'db2': db2, 'dW3': dW3, 'db3': db3}
    return backward


def update_parameters(parameters, backward, learning_rate=0.001):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    dW3 = backward['dW3']
    db3 = backward['db3']
    dW2 = backward['dW2']
    db2 = backward['db2']
    dW1 = backward['dW1']
    db1 = backward['db1']

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    W3 = W3 - learning_rate * dW3
    b3 = b3 - learning_rate * db3
    parameters = {'W1': W1, 'b1': b1, 'W2': W2,
                  'b2': b2, 'W3': W3, 'b3': b3}
    return parameters


def Models(X, Y, iterations, print_cost=False):
    np_x = X.shape[0]
    np_h1 = 10
    np_h2 = 3
    np_y = 1
    costs = []
    iters = []
    parameters = initialize_weights(np_x, np_h1, np_h2, np_y)

    for i in range(iterations):
        A3, forward = forward_propagation(X, parameters)
        cost = cost_function(A3, Y)
        backward = backward_propagation(X, Y, A3, parameters, forward)
        parameters = update_parameters(parameters, backward, learning_rate=0.001)
        if print_cost and i % 1000 == 0:
            a = f'Epochs: {i}: >>>>>>>>>>>>>>>>>>>>>>>>>>>>> Cost: {cost}'
            print(a)
            costs.append(cost)
            # total_cost = np.squeeze(total_cost)
            iters.append(i)

    all_params = {'params': parameters, 'costs': costs, 'iters': iters}
    return all_params
