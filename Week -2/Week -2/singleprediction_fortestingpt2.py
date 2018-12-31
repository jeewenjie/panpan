#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 08:40:48 2018

@author: audiodsp
"""
import numpy as np
import librosa 
import librosa.display
import pandas as pd
from keras.models import load_model

# Load file 
mean = np.load('/media/audiodsp/ExternalVolume/Google_speech_command/Codes/mean_npy.npy')
std = np.load('/media/audiodsp/ExternalVolume/Google_speech_command/Codes/std_npy.npy')
## CHANGE THIS FOR EACH TEST CASE
test = pd.read_csv("/media/audiodsp/ExternalVolume/Google_speech_command/recordings_19_06_2018/audio_recordings/11_others/11others.txt")
## CHANGE THIS FOR EACH TEST CASE
s = 'unknown'

# Define labels
LABELS = ['unknown','down','right','no','yes','up','go','off','on','silence','stop','left']

# Load model 
model = load_model('/media/audiodsp/ExternalVolume/Google_speech_command/Real_time_materials/models/best_5.h5')
model.load_weights('/media/audiodsp/ExternalVolume/Google_speech_command/Real_time_materials/models/best_5.h5')

class Config(object):
    def __init__(self,
                 sampling_rate=16000, audio_duration=1, n_classes=12,
                 use_mfcc=False, n_folds=2, learning_rate=0.0001, 
                 max_epochs=50, n_mfcc=40):
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.n_classes = n_classes
        self.use_mfcc = use_mfcc
        self.n_mfcc = n_mfcc
        self.n_folds = n_folds
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        self.audio_length = self.sampling_rate * self.audio_duration
        if self.use_mfcc:
            self.dim = (32,120, 1) #hardcoded for now, check size and then in shape of time, freq
        else:
            self.dim = (self.audio_length, 1)
    
    
def prepare_data(df, config, data_dir):
    X = np.empty(shape=(1, config.dim[0], config.dim[1], 1))

    input_length = config.audio_length
    
    fname = df.fname

    file_path = data_dir + fname
    data, _ = librosa.core.load(file_path, sr=config.sampling_rate, res_type="kaiser_fast")

    # Random offset / Padding
    if len(data) > input_length:
        max_offset = len(data) - input_length
        offset = np.random.randint(max_offset)
        data = data[offset:(input_length+offset)]
    else:
        if input_length > len(data):
           max_offset = input_length - len(data)
           offset = np.random.randint(max_offset)
        else:
            offset = 0
        data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

    ps = librosa.feature.melspectrogram(y=data, sr=config.sampling_rate,  n_fft=1024, hop_length=512, n_mels=120)

        
    log_ps = np.log(ps.T + 1e-10)
    data = log_ps


    data = np.expand_dims(data, axis=-1)

    X[0,] = data
    return X

# Define configuration
config = Config(sampling_rate=16000, audio_duration=1, n_folds=10, 
                learning_rate=0.001, use_mfcc=True, n_mfcc=14)

length = len(test)
counter = 0 
(down_count, go_count, left_count,
 no_count, off_count, on_count, 
 right_count, stop_count, up_count,
 yes_count, unknown_count)= 0,0,0,0,0,0,0,0,0,0,0


for i in range(length):
    ## Load file. CHANGE THE DIRECTORY FOR EACH TEST CASE
    file_1 = test.iloc[i]
    X_file = prepare_data(file_1,config,'/media/audiodsp/ExternalVolume/Google_speech_command/recordings_19_06_2018/audio_recordings/11_others/')
    X_file = (X_file - mean)/std
    predictions = model.predict(X_file,batch_size=1,verbose=1)
    prediction = (np.array(LABELS)[np.argsort(-predictions, axis=1)[:, :1]])[0][0]
    
    
    if prediction == s:
       counter+=1
       
    elif prediction == 'down':
       down_count+=1
    elif prediction == 'go':
       go_count+=1
    elif prediction == 'left':
       left_count+=1
    elif prediction == 'no':
       no_count+=1
    elif prediction == 'off':
       off_count+=1
    elif prediction == 'on':
       on_count+=1
    elif prediction == 'right':
       right_count+=1
    elif prediction == 'stop':
       stop_count+=1
    elif prediction == 'up':
       up_count+=1
    elif prediction == 'yes':
       yes_count+=1
    elif prediction == 'unknown':
       unknown_count+=1

print('Accuracy = {0}'.format(counter/length))

if s == 'down':
   print('Predicted "down" {0} times.'.format(counter))
else: 
   print('Predicted "down" {0} times.'.format(down_count))
   
if s == 'go':
   print('Predicted "go" {0} times.'.format(counter))
else: 
   print('Predicted "go" {0} times.'.format(go_count))
   
if s == 'left':
   print('Predicted "left" {0} times.'.format(counter))
else: 
   print('Predicted "left" {0} times.'.format(left_count))
   
if s == 'no':
   print('Predicted "no" {0} times.'.format(counter))
else: 
   print('Predicted "no" {0} times.'.format(no_count))
   
if s == 'off':
   print('Predicted "off" {0} times.'.format(counter))
else: 
   print('Predicted "off" {0} times.'.format(off_count))
   
if s == 'on':
   print('Predicted "on" {0} times.'.format(counter))
else: 
   print('Predicted "on" {0} times.'.format(on_count))

if s == 'right':
   print('Predicted "right" {0} times.'.format(counter))
else: 
   print('Predicted "right" {0} times.'.format(right_count))

if s == 'stop':
   print('Predicted "stop" {0} times.'.format(counter))
else: 
   print('Predicted "stop" {0} times.'.format(stop_count))

if s == 'up':
   print('Predicted "up" {0} times.'.format(counter))
else: 
   print('Predicted "up" {0} times.'.format(up_count))

if s == 'yes':
   print('Predicted "yes" {0} times.'.format(counter))
else: 
   print('Predicted "yes" {0} times.'.format(yes_count))

if s == 'unknown':
   print('Predicted "unknown" {0} times.'.format(counter))
else: 
   print('Predicted "unknown" {0} times.'.format(unknown_count))

