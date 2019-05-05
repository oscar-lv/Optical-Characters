# Imports
import os

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# Paths and mapping
DATADIR = './training'
MAPPING = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
           'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Creating, shuffling and splitting data
training_data = []


def create_training_data():
    for category in MAPPING:
        path = os.path.join(DATADIR, category)
        class_num = MAPPING.index(category)
        for image in os.listdir(path):
            try:
                image_op = Image.open(os.path.join(path, image))
                np_im = np.array(image_op)
                training_data.append([np_im, class_num])
            except:
                pass


create_training_data()

import random

random.shuffle(training_data)

X0 = []
y = []

for features, label in training_data:
    X0.append(features)
    y.append(label)

X = np.array(X0).reshape(-1, 20, 20, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D


def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, input_shape=(20, 20, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(35, activation='softmax'))
    return model


# Training and saving the model
model = create_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=60)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))
#model.save('task1_cnn.h5')
