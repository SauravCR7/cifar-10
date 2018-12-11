# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 07:38:30 2018

@author: Saurav
"""
import cv2
from PIL import Image
from keras.models import load_model
import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils
from matplotlib import pyplot as plt

labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

model = load_model('Train_model.h5')
'''
(x_train, y_train),(x_test, y_test) = cifar10.load_data()

new_x_train = x_train.astype('float32')
new_x_test = x_train.astype('float32')
new_x_test = new_x_test[:10000]
new_x_train /= 255
new_x_test /= 255
new_y_train = np_utils.to_categorical(y_train)
new_y_test = np_utils.to_categorical(y_test)


print(len(new_y_test))
print(len(new_x_test))

display_label=y_test[0][0]
i=plt.imshow(new_x_test[0])
plt.show()
print(labels[display_label])

s=model.evaluate(new_x_test , new_y_test, batch_size=100, verbose=1)
print(type(s))
print(s)
'''
input_img = Image.open('C:/Users/Saurav/Desktop/stan-lee.jpg')
input_img = input_img.resize((32,32), resample=Image.LANCZOS)
image_array = np.array(input_img)
image_array = image_array.astype('float32')
image_array /= 255.0
image_array = image_array.reshape(1,32,32,3)
result = model.predict(image_array)
input_img.show()
print(labels[np.argmax(result)])
#class_prob= model.predict_proba(result)
#print(class_prob)

