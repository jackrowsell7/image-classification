import tensorflow as tf



from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.python.keras import layers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python.keras.callbacks import TensorBoard
import datetime

import os
import tensorflow_datasets as tfds



plt.style.use('fivethirtyeight')

#Load the data

from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()   #Training and testing datasets (170498071 images!)

#Checking datatypes of variables

print(type(x_train))
print(type(y_train))
print(type(x_test))
print(type(y_test))

#Get the shape of the arrays

print('x_train shape:', x_train.shape) #50,000 rows of data, 32x32 images, depth 3 (RGB)
print('y_train shape:', y_train.shape) #50,000 rows, 1 column
print('x_test shape:', x_test.shape) 
print('y_test shape:', y_test.shape)

#Looking at the first image as an array

index = 10
print(x_train[index])

#Show the image 

img = plt.imshow(x_train[index])
plt.show()

#Get the img label

print('The image label is:', y_train[index])

#Get the classification of the image

classification = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']



print("The img class is:", classification[y_train[index][0]])

#Convert the labels into a set of 10 numbers to input into the neural network
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

print(y_train_one_hot)
print(y_test_one_hot)

#Print the new label (using one hot encoding) of the image
print('The one hot label is:', y_train_one_hot[index])

#Normailse the pixels to be values between 0 and 1
x_train = x_train / 255
x_test = x_test /255

print(x_train[index])

#Create the models architecture

model = Sequential()

# convolutional layer
model.add(Conv2D(50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(32, 32, 3)))

# convolutional layer
model.add(Conv2D(75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(125, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# flatten output of conv
model.add(Flatten())

# hidden layer
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.3))
# output layer
model.add(Dense(10, activation='softmax'))


#Compile the model

model.compile(loss = 'categorical_crossentropy',
optimizer = 'adam',
metrics = ['accuracy'])

#Train the model

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

hist = model.fit(x_train, y_train_one_hot,
batch_size = 128,
epochs = 10,
validation_split = 0.2,
callbacks = [tensorboard_callback])

#Model evaluate
model.evaluate(x_test,y_test_one_hot)[1]
print(model.evaluate(x_test,y_test_one_hot)[1])

#Visualize the accuracy

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title("Model accuracy")
plt.ylabel("Accuracy")
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

