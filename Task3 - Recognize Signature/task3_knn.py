# Import

import os

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import pickle
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
from sklearn.neighbors import KNeighborsClassifier

k = 4
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)
model.score(X_test, y_test)

       
# Model evaluation

model = pickle.load(open('model_knn.sav', 'rb'))
y = model.predict(X_test)
    
print("EVALUATION ON TESTING DATA")
print(classification_report(y, y_test))

print ("Confusion matrix")
b = classification_report(y_test,y)
a = confusion_matrix(y_test,y)
fig = plt.figure()
plt.matshow(a)
plt.title('Task 2: Confusion Matrix On EMNIST DATA')
plt.colorbar()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
