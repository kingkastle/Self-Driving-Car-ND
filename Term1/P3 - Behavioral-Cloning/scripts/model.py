"""
model to predict steering angles from the previous sequence of images
model architecture is taken from: https://github.com/jamesmf/mnistCRNN/blob/master/scripts/addMNISTrnn.py

References used:
https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.a60sq6l6p
https://github.com/fchollet/keras/issues/1638
"""

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import warnings

# Fix error with TF and Keras
import tensorflow as tf

from config import *
import process_data

tf.python.control_flow_ops = tf

from keras.models import Sequential

from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.callbacks import EarlyStopping

def generator(samples, labels, batch_size):
    """
    Generator with augmented data to feed the model_RNN
    :param samples: numpy array with samples
    :param labels: numpy array with corresponding labels
    :param batch_size: int batch size
    :yields: batched samples augmented and corresponding labels
    """
    while 1:
        batch_images = []
        batch_steering = []
        for batch_sample in range(0, batch_size):
            # random value:
            intensity = np.random.uniform()
            # random flipping:
            flipping = np.random.choice([True, False])
            # random sample
            idx = np.random.randint(samples.shape[0])
            img_aug, steering_aug = process_data.augmented_images(samples[idx], labels[idx], flipping, intensity)
            batch_images.append(img_aug)
            batch_steering.append(steering_aug)
        batch_images = np.asarray(batch_images)
        batch_steering = np.asarray(batch_steering)
        yield batch_images, batch_steering


############################################################
# Configuration:
############################################################
batch_size = 512
nb_epochs = 3
seed = 2016
test_size = 0.2


# for reproducibility
np.random.seed(seed)

############################################################
# Load and process Data_test:
############################################################
with open(CURRENT_PATH + 'features_{0}.pickle'.format(VERSION), 'rb') as handle:
    X = pickle.load(handle)
with open(CURRENT_PATH + 'labels_{0}.pickle'.format(VERSION), 'rb') as handle:
    y = pickle.load(handle)

X = X.astype('float32')
y = y.astype('float32')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)


############################################################
# Define our time-distributed setup
############################################################
model = Sequential()
model.add(
Convolution2D(nb_filter=24, nb_row=5, nb_col=5, border_mode='valid', subsample=(2, 2), input_shape=(Y_PIX, X_PIX, 3)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filter=36, nb_row=5, nb_col=5, border_mode='valid', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filter=48, nb_row=5, nb_col=5, border_mode='valid', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, border_mode='valid'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(1164))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))

# keras model compile, choose optimizer and loss func
model.compile(optimizer='adam', loss='mse')

# train generator:
train_generator = generator(X_train, y_train, batch_size=batch_size)
validation_generator = generator(X_test, y_test, batch_size=batch_size)

# callback:
early_stopping = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')

# run epochs of sampling data then training
model.fit_generator(train_generator, samples_per_epoch=batch_size*100, nb_epoch=nb_epochs, verbose=1,
                    validation_data=validation_generator, nb_val_samples=X_test.shape[0])

# evaluate:
print("Model Evaluation: ", model.evaluate(X_test, y_test, batch_size=32, verbose=0, sample_weight=None))

# save the model
model.save(PATH_TO_MODEL + 'model_{0}.h5'.format(VERSION))
print("model Saved!", PATH_TO_MODEL + 'model_{0}.h5'.format(VERSION))

print("Model structure:")
print(model.summary())
