# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 10:31:12 2018

@author: Saurav
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.constraints import maxnorm
from PIL import Image
from keras.datasets import cifar10
from matplotlib import pyplot as plt
from keras.utils import np_utils
import h5py

labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

(x_train, y_train),(x_test, y_test) = cifar10.load_data()

new_x_train = x_train.astype('float32')
new_x_test = x_train.astype('float32')
new_x_train /= 255
new_x_test /= 255
new_y_train = np_utils.to_categorical(y_train)
new_y_test = np_utils.to_categorical(y_test)


model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(32,32,3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])

model.fit(new_x_train, new_y_train, epochs=50, batch_size=32)
model.save('Train_model.h5')




