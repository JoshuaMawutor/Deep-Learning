import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split

def load_data(path):
    data_list = []
    label_list = []
    for i in os.listdir(path):
        if i.endswith('.jpg') or i.endswith('.png'):
            img = cv2.imread(os.path.join(path, i))
            img = cv2.resize(img, (64,64))
            data_list.append(img)

            if i.startswith('dog'):
                lab = 0
            else:
                lab = 1
            label_list.append(lab)
    label = np.array(label_list)
    data = np.array(data_list)
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
    X_train = X_train.reshape(X_train.shape[0], -1).T
    X_test = X_test.reshape(X_test.shape[0], -1).T
    y_train = y_train.reshape(y_train.shape[0], -1).T
    y_test = y_test.reshape(y_test.shape[0], -1).T
    return X_train, y_train, X_test, y_test


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def relu(x):
    s = np.maximum(0, x)
    return s


def initialize_weights(shape):
    W1 = np.random.randn(shape[1], shape[0]) * 0.01
    b1 = np.zeros((shape[1], 1))
    W2 = np.random.randn(shape[2], shape[1]) * 0.01
    b2 = np.zeros((shape[2], 1))
    W3 = np.random.randn(shape[3], shape[2]) * 0.01
    b3 = np.zeros((shape[2], 1))

    parameters = {'W1': W1, 'b1': b1, 'W2': W2,
                  'b2': b2, 'W3': W3, 'b3': b3}
    return parameters


def forward_propagation(parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = np.dot(W1, W1.T) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, W2.T) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, W3.T) + b3
    A3 = sigmoid(Z3)

    forward = {'Z1': Z1, 'A1': A1, 'Z2': Z2,
               'A2': A2, 'Z3': Z3, 'A3': A3}
    return A3, forward

def cost_function(A3, Y):
    m = Y.shape[1]
    cost = -1 / m * (np.dot(Y, np.log(A3).T) + np.dot(1 - Y, np.log(1 - A3).T))
    cost = np.squeeze(cost)
    return cost

def backward_propagation(A3, Y, X , forward, parameters):
    m = Y.shape[1]
