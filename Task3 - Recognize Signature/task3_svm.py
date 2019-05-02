# Import

import os

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

X = np.array(X0).reshape(len(X0), 400)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Creating, training and saving model

from sklearn.svm import SVC

# create the SVC
clf = SVC(gamma=0.044) # Low gamma for low bias
# train the svm
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
