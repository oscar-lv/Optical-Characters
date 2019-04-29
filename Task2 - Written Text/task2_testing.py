import os
import string
from collections import Counter

import keras
import numpy as  np

from extractor2 import Extract_Letters

path = './mnist.png'
letters = Extract_Letters.exece([path])
#gt = ((open('./ocr/testing/adobe_ground_truth.txt', 'r').read()).upper()).replace(" ", "")
#gt = gt.translate(str.maketrans('', '', string.punctuation)).replace("’", '').replace('”', '')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MAPPING = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
              'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','a','b','d','e','f','g','h','n','q','r','t']

model = keras.models.load_model('my_model4.h5')
model.summary()

predictions = []

for image in letters:
    im = np.array(image)
    a = im.reshape(1,1, 28, 28)
    prediction = model.predict(a)
    predictions.append(CATEGORIES[np.argmax(prediction)])


output = str(predictions).replace(",", '').replace('[', '').replace(']', '').replace("'", '').replace(" ", '')

---


preds = model.predict(data)
preds = np.argmax(preds, axis=1)
return ''.join(mapping[x] for x in preds)



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
# -*- coding: utf-8 -*-

