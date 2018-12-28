#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 09:07:09 2018

@author: Jee Wen Jie
"""

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv1D, BatchNormalization
from keras.layers import GlobalMaxPooling1D, MaxPooling1D

model = Sequential()

## BLOCK no. 1
# Input layer: F = 5. S=1 by default. 
model.add(Conv1D(filters = 8, kernel_size = 5,input_shape = (16000,1))) 
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv1D(filters = 8, kernel_size = 5))
model.add(BatchNormalization())
model.add(Activation('relu'))
  
model.add(MaxPooling1D(pool_size = 4))

## BLOCK no. 2
model.add(Conv1D(filters = 16, kernel_size = 5))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv1D(filters = 16, kernel_size = 5))
model.add(BatchNormalization())
model.add(Activation('relu'))
  
model.add(MaxPooling1D(pool_size = 4))

## Block No. 3
model.add(Conv1D(filters = 32, kernel_size = 5))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv1D(filters = 32, kernel_size = 5))
model.add(BatchNormalization())
model.add(Activation('relu'))
  
model.add(MaxPooling1D(pool_size = 4))

## Block No. 4
model.add(Conv1D(filters = 64, kernel_size = 5))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv1D(filters = 64, kernel_size = 5))
model.add(BatchNormalization())
model.add(Activation('relu'))
  
model.add(MaxPooling1D(pool_size = 4))

## Block No. 5
model.add(Conv1D(filters = 128, kernel_size = 5))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv1D(filters = 128, kernel_size = 5))
model.add(BatchNormalization())
model.add(Activation('relu'))
  
model.add(MaxPooling1D(pool_size = 4))

## Block No. 6
model.add(Conv1D(filters = 256, kernel_size = 5))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv1D(filters = 256, kernel_size = 5))
model.add(BatchNormalization())
model.add(Activation('relu'))
  
model.add(GlobalMaxPooling1D())

## FC layers
model.add(Dense(128))#input_shape=(None,256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

## Output
model.add(Dense(12))

model.summary()