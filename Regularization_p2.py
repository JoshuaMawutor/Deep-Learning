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


def forward_propagation(X, parameters, drop_prob):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    D1 = np.random.rand(A1.shape[0], A1.shape[1]) / drop_prob
    A1 = A1 * D1
    A1 = A1 / drop_prob

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    D2 = np.random.rand(A2.shape[0], A2.shape[1]) / drop_prob
    A2 = A2 * D2
    A2 = A2 / drop_prob

    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    forward_output = {'A1': A1, 'A2': A2, 'A3': A3,
                      'Z1': Z1, 'Z2': Z2, 'Z3': Z3,
                      'D1': D1, 'D2': D2}
    return A3, forward_output


def cost_function(Y, A3):
    m = Y.shape[1]
    cost = -1 / m * (np.dot(Y, np.log(A3).T) + np.dot(1 - Y, np.log(1 - A3).T))
    cost = np.squeeze(cost)
    return cost


def backward_propagation(X, A3, Y, forward_output, parameters, drop_prob):
    m = Y.shape[1]
    W2 = parameters['W2']
    W3 = parameters['W3']
    A2 = forward_output['A2']
    A1 = forward_output['A1']
    D1 = forward_output['D1']
    D2 = forward_output['D2']

    dZ3 = A3 - Y
    dW3 = 1 / m * np.dot(dZ3, A2.T)
    db3 = 1 / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dA2 = np.multiply(dA2, D2)
    dA2 = dA2 / drop_prob
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dA1 = np.multiply(dA1, D1)
    dA1 = dA1 / drop_prob
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1 / m * np.dot(dZ1, X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    backward = {'dW3': dW3, 'db3': db3, 'dW2': dW2,
                'db2': db2, 'dW1': dW1, 'db1': db1}
    return backward


def update_parameters(parameters, backward, learning_rate):
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

def model(X, Y, iterations, drop_prob, print_cost=False):
    np_x = X.shape[0]
    np_h1 = 20
    np_h2 = 10
    np_y = 1

    costs = []
    iters = []
    parameters = initialize_weights(np_x, np_h1, np_h2, np_y)

    for i in range(iterations):
        A3, forward_output = forward_propagation(X, parameters, drop_prob)
        cost = cost_function(Y, A3)
        backward = backward_propagation(X,A3,Y,forward_output, parameters, drop_prob)
        parameters = update_parameters(parameters, backward, 0.001)

        if print_cost and i % 1000 == 0:
            a = f'Epochs: {i}: >>>>>>>>>>>>>>>>>>>>>>>>>>>>> Cost: {cost}'
            print(a)
            costs.append(cost)
            # total_cost = np.squeeze(total_cost)
            iters.append(i)
    all_params = {'params': parameters,'iters': iters,'costs': costs}
    return all_params


