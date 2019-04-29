# Imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt

from mnist import MNIST

mdata = MNIST('emnist')
mdata.select_emnist('balanced')
X_train, y_train = mdata.load_training()
X_test, y_test = mdata.load_testing()

# Optional resizing
X_train,y_train = X_train[0:60000], y_train[:60000]
X_test, y_test = X_test[:9000], y_test[:9000]

# Assinging data
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)


# Defining and training model, saving to pickle
k = 3
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)
model.score(X_test,y_test)
import pickle 
pickle.dump(model, open('model_knn.sav', 'wb'))



# =============================================================================
# Finding best number of K

# for k in range(1,8):
#           # train the k-Nearest Neighbor classifier with the current value of `k`
#           model = KNeighborsClassifier(n_neighbors=k)
#           model.fit(X_train, y_train)
#           # evaluate the model and update the accuracies list
#           score = model.score(X_test, y_test)
#           print("k=%d, accuracy=%.2f%%" % (k, score * 100))
#           accuracies.append(score)
# =============================================================================
          
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

