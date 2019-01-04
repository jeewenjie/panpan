#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 11:26:07 2018

@author: audiodsp
"""
import pyaudio
import wave
import tkinter as tk
import threading
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
import librosa
from tkinter import Label
from PIL import Image, ImageTk
import pandas as pd
from keras.models import load_model
from keras import backend as K
import timeit
import tensorflow as tf

main  = tk.Tk()
#canvas = tk.Canvas(main, width = 300, height = 300)
lbl = Label(main,compound="top", text=" ", font=("Helvetica", 24))
lbl.pack()
#image = Image.open("normalizedsignal.png")
#photo = ImageTk.PhotoImage(image)
#canvas.create_image(20,20, image=photo)
mean = np.load('/media/audiodsp/ExternalVolume/Google_speech_command/Codes/mean_npy.npy')
std = np.load('/media/audiodsp/ExternalVolume/Google_speech_command/Codes/std_npy.npy')
#df = pd.DataFrame({'fname' : ['newoutput.wav']})
#LABELS = ['unknown','down','right','no','yes','up','go','off','on','silence','stop','left']
y = np.linspace(0,1,15976) # Depends on sample rate
model = load_model('/media/audiodsp/ExternalVolume/Google_speech_command/Real_time_materials/models/best_5.h5')
model.load_weights('/media/audiodsp/ExternalVolume/Google_speech_command/Real_time_materials/models/best_5.h5')
graph = tf.get_default_graph()

class RecordApp():
    
    def __init__(self,lbl,master):
        self.master = master
        self.lbl = lbl
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.RECORD_SECONDS = 1
        self.WAVE_OUTPUT_FILENAME = "newoutput.wav" 
        self.isrecording = False
        self.button = tk.Button(main,text='Record', height = 10, width = 20)
        self.button.bind("<ButtonRelease-1>", self.startrecording)
        self.button.pack()
        self.LABELS = ['unknown','down','right','no','yes','up','go','off','on','silence','stop','left']  
        
    def startrecording(self, event): 
        self.isrecording = True
        t = threading.Thread(target = self._record)
        t.start()
    
    def _record(self):

        print('Recording')
        frames = []
        p = pyaudio.PyAudio()  

        for i in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
            stream = p.open(format=self.FORMAT,channels=self.CHANNELS,rate=self.RATE,input=True,input_device_index=1,frames_per_buffer=self.CHUNK)
            data = stream.read(self.CHUNK)
            frames.append(data)
            stream.stop_stream()
            stream.close()
         #  p.terminate()

        print('Done recording')

        wf = wave.open(self.WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        _, signal = scipy.io.wavfile.read('newoutput.wav')
        signal = signal/max(abs(signal))
        signal = np.asarray(signal, dtype=np.float32)
        signal = librosa.resample(signal,44100,16000)
        scipy.io.wavfile.write('newoutput.wav',16000,signal)

        t2 = threading.Thread(target = self.predict)
        t2.start()
        
    def display_label(self):
        self.image = tk.PhotoImage(file="normalizedsignal.png")
        lbl.config(text=self.predicted_label,image=self.image)
        
    def prepare_data(self):
        X = np.empty(shape=(1,32,120,1))

        input_length = 16000

        fname = '/home/audiodsp/Desktop/Weekly Reports/Week -1/newoutput.wav'

        data, _ = librosa.core.load(fname, sr=16000, res_type="kaiser_fast")

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

        ps = librosa.feature.melspectrogram(y=data, sr=16000,  n_fft=1024, hop_length=512, n_mels=120)
     
        log_ps = np.log(ps.T + 1e-10)
        data = log_ps
        
        data = np.expand_dims(data, axis=-1)

        X[0,] = data
        return X

    def predict(self):
        with graph.as_default(): # For circumventing keras's problem
             X_file = self.prepare_data()

             X_file = (X_file - mean)/std 
#        tic = timeit.default_timer()
#       model = load_model('/media/audiodsp/ExternalVolume/Google_speech_command/Real_time_materials/models/best_5.h5')
#        model.load_weights('/media/audiodsp/ExternalVolume/Google_speech_command/Real_time_materials/models/best_5.h5')
#        toc = timeit.default_timer()
#        print(toc-tic) # = 1.8 s
        
             predictions = model.predict(X_file,batch_size=1,verbose=1)
             
             top_1 = np.array(self.LABELS)[np.argsort(-predictions, axis=1)[:, :1]]
             self.predicted_label = ([' '.join(list(x)) for x in top_1])[0]
#             K.clear_session()
             _, data = scipy.io.wavfile.read('newoutput.wav')
             fig = plt.figure()
             plt.title('Signal Graph')
             plt.ylabel('Signal')
             plt.xlabel('Time')
             plt.plot(y,data)
             fig.savefig("normalizedsignal.png")
             plt.close(fig)
             plt.show()
             t3 = threading.Thread(target = self.display_label)
             t3.start()
        
#    def display(self):
#        _, signal = scipy.io.wavfile.read('newoutput.wav')
#        plt.plot(signal)
#        plt.savefig('normalizedsignal.png', bbox_inches='tight')
#        self.canvas.after(1,self.display)
 
app = RecordApp(lbl,main)
main.mainloop()
