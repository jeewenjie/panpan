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
# CHANGE THIS FOR EACH TEST CASE
test = pd.read_csv("/media/audiodsp/ExternalVolume/Google_speech_command/recordings_19_06_2018/audio_recordings/11_others/11others.txt")
#train = pd.read_csv("/media/audiodsp/ExternalVolume/Google_speech_command/train_wav_reference.csv")
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
#   print(X.shape)
#   exit()
    input_length = config.audio_length
    #for i, fname in enumerate(df.index):
    fname = df.fname
#   print(fname)
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

    # print(data.shape)
    # print(ps.shape)
    # plt.plot([1,2,3])
    # plt.subplot(211)
    # plt.ylabel('Amplitude')
    # plt.title('Raw audio')
    # librosa.display.waveplot(data, 16000,x_axis='time', offset=0.0)
    # plt.subplot(212)
    # librosa.display.specshow(librosa.power_to_db(ps,ref=np.max), y_axis='mel', fmax=8000)
    # plt.title('Mel spectrogram')
    # plt.tight_layout()
    # plt.show()
    # exit()
        
    log_ps = np.log(ps.T + 1e-10)
    data = log_ps

  
    #print(log_ps.shape)
    #exit()

    #mfcc = librosa.feature.mfcc(data, sr=config.sampling_rate, n_mfcc=config.n_mfcc)
    #mfcc_delta = librosa.feature.delta(mfcc)
    #mfcc_delta2 = librosa.feature.delta(mfcc, order=2)        
    
    #concatenate mfcc, delta and delta-delta
    #data = np.concatenate((mfcc, mfcc_delta, mfcc_delta2), axis=0)
    data = np.expand_dims(data, axis=-1)

    # print(data.shape)
    # exit()
    X[0,] = data
    return X


# Define configuration
config = Config(sampling_rate=16000, audio_duration=1, n_folds=10, 
                learning_rate=0.001, use_mfcc=True, n_mfcc=14)


#for i, (train_split, val_split) in enumerate(skf):
    #K.clear_session()
#    X, y, X_val, y_val = X_train[train_split], y_train[train_split], X_train[val_split], y_train[val_split]
#    checkpoint = ModelCheckpoint('best_%d.h5'%i, monitor='val_loss', verbose=1, save_best_only=True)
#    early = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    #tb = TensorBoard(log_dir='./logs/' + PREDICTION_FOLDER + '/fold_%i'%i, write_graph=True)
    #callbacks_list = [checkpoint, early, tb]
#    callbacks_list = [checkpoint, early]
#    print("#"*50)
#    print("Fold: ", i)
#    model = get_2d_conv_model(config)
    #model.summary()
    #plot_model(model, to_file = 'final_model.png', show_shapes = 'true')
    #exit()
#    history = model.fit(X, y, validation_data=(X_val, y_val), callbacks=callbacks_list, 
#                        batch_size=64, epochs=config.max_epochs)


    # Save train predictions
#    predictions = model.predict(X_train, batch_size=64, verbose=1)
#    np.save(PREDICTION_FOLDER + "/train_predictions_%d.npy"%i, predictions)

    # Save test predictions
#    predictions = model.predict(X_test, batch_size=64, verbose=1)
#    np.save(PREDICTION_FOLDER + "/test_predictions_%d.npy"%i, predictions)

    # Make a submission file
#    top_1 = np.array(LABELS)[np.argsort(-predictions, axis=1)[:, :1]]
#    predicted_labels = [' '.join(list(x)) for x in top_1]
#    test['label'] = predicted_labels
#    test[['label']].to_csv(PREDICTION_FOLDER + "/predictions_%d.csv"%i)

length = len(test)
counter = 0 


#tic=timeit.default_timer()
#predictions = model.predict(X_file,batch_size=1,verbose=1)
#get_output = K.function([model.layers[0].input],[model.layers[-1].output])
#predictions = get_output([X_file])[0]
#toc=timeit.default_timer()
#print(toc-tic)


for i in range(length):
    # Load file. CHANGE THE DIRECTORY FOR EACH TEST CASE
    file_1 = test.iloc[i]
    X_file = prepare_data(file_1,config,'/media/audiodsp/ExternalVolume/Google_speech_command/recordings_19_06_2018/audio_recordings/11_others/')
    #file_1 = train.iloc[7]
    #X_file = prepare_data(file_1,config,'/media/audiodsp/ExternalVolume/Google_speech_command/Real_time_materials/train_wav/')
    X_file = (X_file - mean)/std
    predictions = model.predict(X_file,batch_size=1,verbose=1)
    prediction = np.array(LABELS)[np.argsort(-predictions, axis=1)[:, :1]]
    
    #CHANGE THIS FOR EACH TEST CASE
    if prediction[0][0] == 'unknown':
       counter+=1

print(counter/length)
