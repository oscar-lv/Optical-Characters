# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

dftrain,dftest = pd.read_csv('./emnist-balanced-train.csv'), pd.read_csv('./emnist-balanced-test.csv')

from numpy import genfromtxt
my_data = genfromtxt('./emnist-balanced-test.csv', delimiter=',')

image_array = my_data[0]

import tensorflow as tf

(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()



from keras.utils import np_utils
from keras.layers import Input, Dense, Dropout, Conv2D
from keras.models import Model

num_classes = 47
y_train = dftrain.iloc[:,0]
y_train = np_utils.to_categorical(y_train, num_classes)
print ("y_train:", y_train.shape)

x_train = dftrain.iloc[:,1:]
x_train = x_train.astype('float32')
x_train /= 255
print ("x_train:",x_train.shape)

inp = Input(shape=(784,))
conv1_ = Conv2D(64, kernel_size=3, activation='relu')(inp)
hidden_1 = Dense(1024, activation='relu')(conv1_)
dropout_1 = Dropout(0.2)(hidden_1)
out = Dense(num_classes, activation='softmax')(hidden_1) 
model = Model(input=inp, output=out)

model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy

model.fit(x_train, y_train, # Train the model using the training set...
          batch_size=512, nb_epoch=10,
          verbose=1, validation_split=0.1) # ...holding out 10% of the data for validation

y_test = dftest.iloc[:,0]
y_test = np_utils.to_categorical(y_test, num_classes)
print ("y_test:", y_test.shape)

x_test = dftest.iloc[:,1:]
x_test = x_test.astype('float32')
x_test /= 255
print ("x_test:",x_train.shape)

print(model.evaluate(x_test, y_test, verbose=1)) # Evaluate the trained model on the test set!

model.save('my_model2.h5') 