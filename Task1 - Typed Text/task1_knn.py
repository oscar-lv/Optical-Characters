# Imports
import os

import numpy as  np
from PIL import Image
from sklearn.model_selection import train_test_split

# Path and mapping
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

X = np.array(X0).reshape(6394, 400)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Training and saving the model

from sklearn.neighbors import KNeighborsClassifier

k = 1
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)
model.score(X_test, y_test)

#import pickle
#pickle.dump(model, open('task1_knn.sav', 'wb'))

