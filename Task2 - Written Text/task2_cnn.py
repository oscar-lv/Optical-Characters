# Imports and data loading

import load_emnist as emnist

X_train, y_train, X_test, y_test, mapping = emnist.load_data('./training')

from keras.models import Sequential
from keras.layers import Conv2D, Convolution2D, MaxPooling2D, Dropout, Flatten, Dense


# Defining different models

def create_model(): 
    model = Sequential()
    model.add(Dense(512, activation='relu',input_shape=(1, 28, 28)))                   
    model.add(Dropout(0.2))
    model.add(Dropout(0.2))
    model.add(Dense(47, activation='softmax'))
    return model

def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(1,1), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(47, activation='softmax'))
    return model

def larger_model():
	# create model
	model = Sequential()
	model.add(Conv2D(30, (1, 1), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(1, 1)))
	model.add(Conv2D(15, (1, 1), activation='relu'))
	model.add(MaxPooling2D(pool_size=(1, 1)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(47, activation='softmax'))
	return model

# Training and saving the model
model = larger_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))
model.save('task2_cnnx.h5')

# Evaluation
score = model.evaluate(X_test, y_test)
print('Loss:', score[0])
print('Accuracy:', score[1])


