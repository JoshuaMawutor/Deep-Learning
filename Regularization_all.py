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


def tanh_derivative(x):
    s = 1 - tanh(x) ** 2
    return s


def initialize_weights(shape):
    # Weight and Bias initialization
    W1 = np.random.randn(shape[1], shape[0]) * 0.01  # Weight initialization by *0.01
    b1 = np.zeros((shape[1], 1))  # Bias initialization
    W2 = np.random.randn(shape[2], shape[1]) * 0.01
    b2 = np.zeros((shape[2], 1))
    W3 = np.random.randn(shape[3], shape[2]) * 0.01
    b3 = np.zeros((shape[3], 1))

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

    forward_out = {'Z1': Z1, 'A1': A1, 'Z2': Z2,
                   'A2': A2, 'Z3': Z3, 'A3': A3}

    return A3, forward_out


def forward_propagation_drop_out(X, parameters, drop_prob):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    D1 = np.random.rand(A1.shape[0], A1.shape[1]) < drop_prob
    A1 = (A1 * D1) / drop_prob

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    D2 = np.random.rand(A2.shape[0], A2.shape[1]) < drop_prob
    A2 = (A2 * D2) / drop_prob

    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    forward_out = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2,
                   'Z3': Z3, 'A3': A3, 'D1': D1, 'D2': D2}

    return A3, forward_out


def cost_function(A3, Y):
    m = Y.shape[1]
    cost = -1. / m * (np.dot(Y, np.log(A3).T) + np.dot(1 - Y, np.log(1 - A3).T))
    cost = np.squeeze(cost)
    return cost


def cost_function_reg(A3, Y, parameters, lambed):
    m = Y.shape[1]
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']

    cost1 = cost_function(A3, Y)
    cost2 = lambed / (2 * m) * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
    cost = cost1 + cost2
    cost = np.squeeze(cost)
    return cost


def backward_propagation(A3, Y, X, parameters, forward_out):
    m = Y.shape[1]
    W3 = parameters['W3']
    W2 = parameters['W2']
    A2 = forward_out['A2']
    A1 = forward_out['A1']

    dZ3 = A3 - Y
    dW3 = 1. / m * np.dot(dZ3, A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    backward = {'dW3': dW3, 'db3': db3, 'dW2': dW2,
                'db2': db2, 'dW1': dW1, 'db1': db1}

    return backward


def backward_propagation_reg(X, A3, Y, parameters, forward_out, lambed):
    m = Y.shape[1]
    W3 = parameters['W3']
    W2 = parameters['W2']
    W1 = parameters['W1']
    A2 = forward_out['A2']
    A1 = forward_out['A1']

    dZ3 = A3 - Y
    dW3 = 1. / m * np.dot(dZ3, A2.T) + (lambed / m) * W3
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T) + (lambed / m) * W2
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T) + (lambed / m) * W1
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    backward = {'dW3': dW3, 'db3': db3, 'dW2': dW2,
                'db2': db2, 'dW1': dW1, 'db1': db1}

    return backward


def backward_propagation_drop(X, A3, Y, parameters, forward_out, drop_prob):
    m = Y.shape[1]
    W3 = parameters['W3']
    W2 = parameters['W2']
    W1 = parameters['W1']
    A2 = forward_out['A2']
    A1 = forward_out['A1']
    D1 = forward_out['D1']
    D2 = forward_out['D2']

    dZ3 = A3 - Y
    dW3 = 1. / m * np.dot(dZ3, A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dA2 = (dA2 * D2) / drop_prob
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dA1 = (dA1 * D1) / drop_prob
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    backward = {'dW3': dW3, 'db3': db3, 'dW2': dW2,
                'db2': db2, 'dW1': dW1, 'db1': db1}

    return backward


def update_weights(parameters, backward, learning_rate):
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


def predict(X, y, parameters, drop_prob):
    m = y.shape[1]
    pred_L = np.zeros((1, m))

    if drop_prob < 1:
        A, forward = forward_propagation_drop_out(X, parameters, drop_prob)
    else:
        A, forward = forward_propagation(X, parameters)

    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            pred_L[0, i] = 1
        else:
            pred_L[0, i] = 0

    accuracy = np.sum(pred_L == y) / m

    return accuracy, pred_L


def models_all(X, Y,learning_rate, keep_prob, lambed, iterations, print_cost=False):
    costs = []
    iters = []
    accuracy = []
    shape = [X.shape[0], 30, 10, 1]
    parameter = initialize_weights(shape)

    for i in range(iterations):
        if keep_prob == 1:
            A3, forward = forward_propagation(X, parameter)

        elif keep_prob < 1:
            A3, forward = forward_propagation_drop_out(X, parameter, keep_prob)

        if lambed == 0:
            cost = cost_function(A3, Y)
        else:
            cost = cost_function_reg(A3, Y, parameter, lambed)

        if keep_prob == 1 and lambed == 0:
            backward = backward_propagation(A3, Y, X, parameter, forward)

        elif lambed != 0:
            backward = backward_propagation_reg(X, A3, Y, parameter, forward, lambed)

        elif keep_prob != 0:
            backward = backward_propagation_drop(X, A3, Y, parameter, forward, keep_prob)

        parameter = update_weights(parameter, backward, learning_rate)
        train_acc, train_pred = predict(X, Y, parameter, keep_prob)

        if print_cost and i % 100 == 0:
            a = f'Epochs: {i}: >>>>>>>>>>>>>>>>>>>>>>>>>>>>> Cost: {cost}, Accuracy: {train_acc}'
            print(a)
            costs.append(cost)
            # total_cost = np.squeeze(total_cost)
            iters.append(i)
            accuracy.append(train_acc)
    reg = {'params': parameter, 'iters': iters, 'costs': costs, 'accuracy': accuracy, 'test_pred1': test_pred1}
    return reg
