# This class can be used to try the different models on the testing samples

import os
import string
from collections import Counter

import keras
import numpy as  np
from extractor import Extract_Letters

# Importing the file and ground truth
path = './testing/screen.png'
letters = Extract_Letters.exece([path])
gt = ((open('./testing/adobe_ground_truth.txt', 'r').read()).upper()).replace(" ", "")

# Normalizing string for comparison 
gt = gt.translate(str.maketrans('', '', string.punctuation)).replace("’", '').replace('”', '')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MAPPING = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
              'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Loading model
model = keras.models.load_model('task1_cnn.h5')

predictions = []

for image in letters:
    im = np.array(image)
    a = im.reshape(1, 20, 20, 1)
    prediction = model.predict(a)[0]
    predictions.append(MAPPING[np.argmax(prediction)])

output = str(predictions).replace(",", '').replace('[', '').replace(']', '').replace("'", '').replace(" ", '')


# Classification report on each letter or confusion matrix 

from sklearn.metrics import classification_report

print (classification_report(output, gt))

# Optionally counting correct strings classified
count = 0
score = 0
for l in output:
    if l == gt[count]:
        score += 1
    count += 1

print(score)
print(len(output), len(gt))

a, b = set(gt), set(output)
print(a & b)

res = Counter(output)
res2 = Counter(gt)
