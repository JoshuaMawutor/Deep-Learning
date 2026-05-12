import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import argparse
import keras
import tensorflow as tf
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization,Rescaling, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.utils import image_dataset_from_directory,img_to_array, array_to_img, save_img
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import cv2
import os
import mlflow
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Train a Keras model on image data.')
parser.add_argument('--input', type=str, required=True, help='Path to the directory containing the image data.')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training the model.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training the model.')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model.')
parser.add_argument('--img_size', type=int, default=128, help='Size to which images will be resized (img_size x img_size).')

args = parser.parse_args()

#----------------------------------------------------------------------------------------------------------
#Parameters
path = args.input
learning_rate = args.learning_rate
batch_size = args.batch_size
epochs = args.epochs
img_size = args.img_size

#----------------------------------------------------------------------------------------------------------
# Data Sorting
def get_data(path):
    image_data_list = []
    label = []
    for i in os.listdir(path):
        if i.endswith('jpg') or i.endswith('png'):
            elements = os.path.join(path, i)
            img = cv2.imread(elements)
            img = cv2.resize(img, (128,128))
            image_data_list.append(img)
            if i.startswith('dog'):
                lab = 1
            else:
                lab = 0
            label.append(lab)
    label = np.array(label)
    image_data_list = np.array(image_data_list)
    plt.imshow(image_data_list[50])
    plt.show() 
    image = image_data_list.shape[0]
    print(image)
    return image_data_list, label
#----------------------------------------------------------------------------------------------------------
# Data Sorting
image_data_list, label = get_data(path)
#----------------------------------------------------------------------------------------------------------

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(image_data_list, label, test_size=0.2, random_state=42)

#----------------------------------------------------------------------------------------------------------
# Data augmentation
data_augmentation = Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1),
])
#----------------------------------------------------------------------------------------------------------
# Model Creation
model = Sequential()
model.add(Rescaling(1./255, input_shape=(128, 128, 3)))
model.add(data_augmentation)
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', name='conv2d_1'))
model.add(BatchNormalization(name='batch_norm_1'))
model.add(MaxPooling2D(pool_size=(2, 2), name='max_pooling_1'))

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', name='conv2d_2'))
model.add(BatchNormalization(name='batch_norm_2'))
model.add(MaxPooling2D(pool_size=(2, 2), name='max_pooling_2'))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', name='conv2d_3'))
model.add(MaxPooling2D(pool_size=(2, 2), name='max_pooling_3'))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', name='conv2d_4'))
model.add(MaxPooling2D(pool_size=(2, 2), name='max_pooling_4'))

model.add(Flatten(name='flatten'))

model.add(Dense(64, activation='relu', name='dense_2'))
model.add(Dropout(0.3))
model.add(Dense(32,  activation='relu', name='Next_output_layer'))
model.add(Dropout(0.3))
#model.add(Dense(8,  activation='relu', name='Next_output_layer1'))
model.add(Dense(1, activation='sigmoid', name='output_layer'))

model.summary()
#----------------------------------------------------------------------------------------------------------
# Model Compilation
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
#----------------------------------------------------------------------------------------------------------
# Model Training
history = model.fit(X_train, y_train, epochs=20, validation_split=0.2, batch_size=32)
#----------------------------------------------------------------------------------------------------------
# Model Evaluation
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
