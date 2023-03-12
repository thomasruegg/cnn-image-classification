import matplotlib
import matplotlib.pyplot as plt

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, Dropout

from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.optimizers import SGD

import numpy



#  pip install extra-keras-datasets first
from extra_keras_datasets import kmnist



# Model configuration
no_classes = 10
validation_split = 0.2


# Load KMNIST dataset
(input_train, target_train), (input_test, target_test) = kmnist.load_data(type='kmnist')

# Shape of the input sets
input_train_shape = input_train.shape
input_test_shape = input_test.shape 

# Keras layer input shape.
input_shape = (input_train_shape[1], input_train_shape[2], 1)

# Reshape the training data to include channels
input_train = input_train.reshape(input_train_shape[0], input_train_shape[1], input_train_shape[2], 1)
input_test = input_test.reshape(input_test_shape[0], input_test_shape[1], input_test_shape[2], 1)

# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Normalize input data
input_train = input_train / 255
input_test = input_test / 255


# Function to create model
def create_model_SGD(neurons):
    model = Sequential()
    model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = input_shape, padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
    model.add(Conv2D(128, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(neurons, activation = 'relu'))
    model.add(Dropout(rate = 0.2))
    model.add(BatchNormalization())
    model.add(Dense(neurons, activation = 'relu'))
    model.add(Dropout(rate = 0.2))
    model.add(BatchNormalization())
    
    model.add(Dense(no_classes, activation = 'softmax'))
    
    # compilation of the model
    
    model.compile(loss=tensorflow.keras.losses.sparse_categorical_crossentropy, optimizer='SGD', metrics=['accuracy'])
    
    return model

# fix random seed for reproducibility
seed = 7
tensorflow.random.set_seed(seed)

# create model
model = KerasClassifier(model=create_model_SGD)

# define the grid search parameters
learn_rate = [0.001, 0.01, 0.1]
momentum = [0.0, 0.5, 0.9]
neurons = [256, 512, 1024]
batch_size = [100, 250, 350]
epochs = [10, 25, 50]

param_grid = dict(model__neurons=neurons, optimizer__learning_rate=learn_rate, optimizer__momentum=momentum,
                 batch_size=batch_size, epochs=epochs)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, verbose=3)

grid_result = grid.fit(input_train, target_train)