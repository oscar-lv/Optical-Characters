# Import

import os

import numpy as  np
from PIL import Image
from sklearn.model_selection import train_test_split
from skimage.feature import hog

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
                a,c = hog(np_im,orientations=10, pixels_per_cell=(20,20),cells_per_block=(1, 1),block_norm= 'L2',visualize=True) #Extracting HoG Features
                training_data.append([c, class_num]) # Using HoG Image
            except:
                pass


create_training_data()

from skimage import data
import random

random.shuffle(training_data)


import matplotlib.pyplot as plt


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
clf = SVC()
# train the svm
clf.fit(X_train, y_train)
clf.score(X_test, y_test)