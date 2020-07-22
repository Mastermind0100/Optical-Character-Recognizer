"""
@author: Atharva
"""
##This code is to train the model to recognize typed characters

import numpy as np
import cv2
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import scipy.fftpack 

trdata = 71999
vltdata = 21600
batch = 16
#tst = cv2.inpaint(tst, thresh2,3, cv2.INPAINT_TELEA)   
arr_result = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

training_data = 'nist_final/training'
validation_data = 'nist_final/validation'

model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(64,64,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=36,activation='sigmoid'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

train_datagen=ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                    horizontal_flip = False)

test_datagen=ImageDataGenerator(rescale = 1./255)

training_set=train_datagen.flow_from_directory(directory = training_data,
                                                 target_size = (64, 64),
                                                 color_mode='grayscale',
                                                 batch_size = batch,
                                                 class_mode = 'sparse')

test_set=test_datagen.flow_from_directory(directory = validation_data,
                                            target_size = (64, 64),
                                            color_mode='grayscale',
                                            batch_size = batch,
                                            class_mode = 'sparse')
 
model.fit_generator(training_set,steps_per_epoch = 4500,         
                         epochs = 15,
                         validation_data = test_set,
                         validation_steps = 1350)                 

model.save('fmodelwts.h5')