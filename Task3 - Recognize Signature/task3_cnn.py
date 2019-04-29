# Imports

import os

import matplotlib.pyplot as plt
import numpy as  np
from PIL import Image
from sklearn.model_selection import train_test_split

# Paths and mapping
DATADIR = './training'
MAPPING = ['o', 'j']

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.26)

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D


# Creating, training and saving model
def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, input_shape=(20, 20, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.6))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return model


model = create_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=10)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))
model.save('my_model.h5')

# Plotting training phase
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=10, verbose=0)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

