import os
import cv2
import  pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn 
from sklearn.model_selection import train_test_split
import mlflow
import argparse
import mlflow
import numpy as np

parser = argparse.ArgumentParser(description='Train a deep learning model on Azure ML')
parser.add_argument('--input', type=str, default='./cat_dog', help='Path to the training data')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for training')
parser.add_argument('--keep_prob', type=float, default=1.0, help='Keep probability for dropout')
parser.add_argument('--lambd', type=float, default=0.0, help='Regularization parameter')
parser.add_argument('--iterations', type=int, default=100000, help='Number of training iterations')
arg = parser.parse_args()



from Regularization_all import (sigmoid, relu, tanh, tanh_derivative, initialize_weights,
                                forward_propagation,forward_propagation_drop_out, cost_function, 
                                cost_function_reg, backward_propagation,backward_propagation_reg, backward_propagation_drop,
                                update_weights, predict, models_all)

data = arg.input

with mlflow.start_run() as run:

    images = []
    label = []
    for i in os.listdir(data):
        if i.endswith('jpg') or i.endswith('png'):
            elements = os.path.join(data, i)
            img = cv2.imread(elements)
            img = cv2.resize(img, (64,64))
            images.append(img)
            if i.startswith('dog'):
                lab = 1
            else:
                lab = 0
            label.append(lab)
    Images = np.array(images)
    label = np.array(label)
    train_x, test_x, train_y, test_y = train_test_split(Images, label, test_size=0.2, random_state=0)

    print("train_x.shape: \n", train_x.shape)
    print("train_y.shape: \n", train_y.shape)
    print("test_x.shape: \n", test_x.shape)
    print("test_y.shape: \n", test_y.shape)

    Xtrain = train_x.reshape(train_x.shape[0], -1).T
    Ytrain = np.array(train_y).reshape(1, -1)
    Xtest = test_x.reshape(test_x.shape[0], -1).T
    Ytest = np.array(test_y).reshape(1, -1)

    print("Xtrain.shape: \n", Xtrain.shape)
    print("Ytrain.shape: \n", Ytrain.shape)
    print("Xtest.shape: \n", Xtest.shape)
    print("Ytest.shape: \n", Ytest.shape)

    plt.imshow(train_x[50])
    plt.imsave('sample_image.png', train_x[50])
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/sample_image.png")

    model = models_all(Xtrain, Ytrain, arg.learning_rate, arg.keep_prob, arg.lambd, arg.iterations, print_cost=True)
    print(f"parameters: {model['params']}")
    print(f"train_cc: {model['train_cc']}")
    print(f"Cost: {model['Cost']}")
    print("======================================================================================")    
    predict = predict(Xtest, Ytest, model['params'], arg.keep_prob)
    print(f'Test Accuracy: {predict[0]}')
    print(f'Loss: {predict[1]}')

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=model['iters'], y=model['costs'])
    plt.xlabel('Iterations')
    plt.ylabel('Cost') 
    plt.title('Cost vs Iterations')
    plt.savefig("cost_iterations.png")
    mlflow.log_artifact("cost_iterations.png")

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=model['iters'], y=model['accuracy'])
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Iterations')
    plt.savefig("accuracy_iterations.png")
    mlflow.log_artifact("accuracy_iterations.png")



