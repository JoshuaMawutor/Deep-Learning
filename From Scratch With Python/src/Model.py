import numpy as np
import matplotlib.pyplot as plt
import copy


def sigmoid(x):
    sfor = 1 / (1 + np.exp(-x))
    return sfor


def tanh(x):
    tfor = np.tanh(x)
    return tfor


def sig_derivative(a):
    sb = sigmoid(a)
    sback = sb(1 - sb)
    return sback


def tanh_derivative(a):
    tbackc = 1 - np.tanh(a) ** 2
    return tbackc


def relu(x):
    s = np.maximum(0, x)
    return s


def initialize(np_x, np_h, np_y):
    # Initialization of weight
    # Initialization of bias
    # first Weight (W1)
    W1 = np.random.randn(np_h, np_x) * 0.01
    # First bias (b1)
    b1 = np.zeros((np_h, 1))
    # Second weight (W2)
    W2 = np.random.randn(np_y, np_h) * 0.01
    # Second bias (b2)
    b2 = np.zeros((np_y, 1))
    parameter = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return parameter


def forward(x, parameter):
    # Retrieve the Weigh and bias from the parameters
    W1 = parameter['W1']
    b1 = parameter['b1']
    W2 = parameter['W2']
    b2 = parameter['b2']

    # DOING DOT PRODUCT OF WEIGHT AND X INPUT FEATURES
    Z1 = np.dot(W1, x) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    forward_params = {'Z1': Z1, 'Z2': Z2, 'A1': A1, 'A2': A2}
    return forward_params


def cost_function(y, forward_params):
    m = y.shape[1]
    A2 = forward_params['A2']
    cost = -1 / m * (np.dot(y, np.log(A2).T) + np.dot(1 - y, np.log(1 - A2).T))
    cost = np.squeeze(cost)
    return cost


def backward(x, y, parameter, forward_params):
    m = y.shape[1]
    W1 = parameter['W1']
    W2 = parameter['W2']

    Z1 = forward_params['Z1']
    A1 = forward_params['A1']
    A2 = forward_params['A2']

    tback = tanh_derivative(Z1)
    dZ2 = A2 - y
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    db2 = 1 / m * np.sum(np.sum(dZ2, axis=1, keepdims=True))

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * np.int64(A1 > 0)
    dW1 = 1 / m * np.dot(dZ1, x.T)
    db1 = 1 / m * np.sum(np.sum(dZ1, axis=1, keepdims=True))
    backward_params = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    return backward_params


def update_parameters(parameter, backward_params, learning_rate=0.01):
    W1 = parameter['W1']
    W2 = parameter['W2']
    b1 = parameter['b1']
    b2 = parameter['b2']

    dW1 = backward_params['dW1']
    db1 = backward_params['db1']
    dW2 = backward_params['dW2']
    db2 = backward_params['db2']

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    parameters = {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}
    return parameters


def predict(y, x, parameters):
    m = y.shape[1]
    pred_list = np.zeros((1, m))
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    pred = forward(x, parameters)
    A2 = pred['A2']
    for i in range(A2.shape[1]):
        if A2[0, i] > 0.5:
            pred_list[0, i] = 1
        else:
            pred_list[0, i] = 0
        accuracy = np.sum(pred_list == y) / m
    train_accuracy = 1 - np.mean(np.abs(pred_list - y))
    return pred_list, train_accuracy


def model(x, y, num_iterations=1000, print_cost=False):
    np_x = x.shape[0]
    np_h = 10
    np_y = 1
    total_cost = []
    iter = []
    accuracies = []
    parameters = initialize(np_x, np_h, np_y)
    old_params = copy.deepcopy(parameters)
    for i in range(num_iterations):

        f_params = forward(x, parameters)
        cost = cost_function(y, f_params)
        b_params = backward(x, y, parameters, f_params)
        parameters = update_parameters(parameters, b_params, learning_rate=0.001)
        pred, accuracy = predict(y, x, parameters)
        if print_cost and i % 1000 == 0:
            a = f'Epochs: {i}: >>>>>>>>>>>>>>>>>>>>>>>>>>>>> Cost: {cost}, Accuracy: {accuracy}'
            print(a)
            total_cost.append(cost)
            # total_cost = np.squeeze(total_cost)
            iter.append(i)
            accuracies.append(accuracy)
    alls = {'params': parameters, 'cost': total_cost, 'f_params': f_params, 'iter': iter, 'pred': pred,
            'accuracy': accuracies}
    return alls
