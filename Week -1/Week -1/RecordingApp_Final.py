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
from keras.models import load_model
import timeit
# tic = timeit.default_timer()
# toc = timeit.default_timer()
# print(toc - tic)
import tensorflow as tf

main  = tk.Tk()

# Title
main.title('Speech Command Recognition')

# Plot empty graph for placeholder
fig = plt.figure()
plt.title('Signal Graph')
plt.ylabel('Signal')
plt.xlabel('Time')
fig.savefig("normalizedsignal.png")
plt.close(fig)


# Load preprocessing data
mean = np.load('/media/audiodsp/ExternalVolume/Google_speech_command/Codes/mean_npy.npy')
std = np.load('/media/audiodsp/ExternalVolume/Google_speech_command/Codes/std_npy.npy')
#LABELS = ['unknown','down','right','no','yes','up','go','off','on','silence','stop','left']

# For time axis
y = np.linspace(0,1,15976) # Depends on sample rate
model = load_model('/media/audiodsp/ExternalVolume/Google_speech_command/Real_time_materials/models/best_5.h5')
model.load_weights('/media/audiodsp/ExternalVolume/Google_speech_command/Real_time_materials/models/best_5.h5')
graph = tf.get_default_graph() # For storing Keras's model and using them in other thread.
# Refer here for more: https://www.tensorflow.org/api_docs/python/tf/get_default_graph

class RecordApp():
    
    def __init__(self,master):
        self.master = master 
        self.placeholderimage = tk.PhotoImage(file="normalizedsignal.png")

        self.lbl = Label(main, image = self.placeholderimage)
        self.lbl.grid(row = 0, column=0, rowspan = 11, sticky="nsew")
        self.lbl2 = Label(main, text=" Available commands:", font=("Helvetica", 12))
        self.lbl2.grid(row = 0, column = 1, columnspan=2, sticky = "w")
        
        self.num1 = Label(main, text="1. down", font=("Helvetica", 15))
        self.num1.grid(row = 1, column = 1, columnspan = 2, sticky = "w")
        
        self.num2 = Label(main, text="2. right", font=("Helvetica", 15))
        self.num2.grid(row = 2, column = 1, columnspan = 2, sticky = "w")
        
        self.num3 = Label(main, text="3. no", font=("Helvetica", 15))
        self.num3.grid(row = 3, column = 1, columnspan = 2, sticky = "w") 
        
        self.num4 = Label(main, text="4. yes", font=("Helvetica", 15))
        self.num4.grid(row = 4, column = 1, columnspan = 2, sticky = "w")

        self.num5 = Label(main, text="5. up", font=("Helvetica", 15))
        self.num5.grid(row = 5, column = 1, columnspan = 2, sticky = "w")
        
        self.num6 = Label(main, text="6. go", font=("Helvetica", 15))
        self.num6.grid(row = 6, column = 1, columnspan = 2, sticky = "w")
        
        self.num7 = Label(main, text="7. off", font=("Helvetica", 15))
        self.num7.grid(row = 7, column = 1,  columnspan = 2, sticky = "w")
        
        self.num8 = Label(main, text="8. on", font=("Helvetica", 15))
        self.num8.grid(row = 8, column = 1, columnspan = 2, sticky = "w")

        self.num9 = Label(main, text="9. stop", font=("Helvetica", 15))
        self.num9.grid(row = 9, column = 1, columnspan = 2, sticky = "w")

        self.num10 = Label(main, text="10. left", font=("Helvetica", 15))
        self.num10.grid(row = 10, column = 1, columnspan = 2, sticky = "w")
        
        self.lbl3 = Label(main, text="Prediction: ", font=("Helvetica", 15))
        self.lbl3.grid(row = 11, column =1, sticky = "nsew")
        self.lbl4 = Label(main, text = "       ", font=("Helvetica", 15))
        self.lbl4.grid(row = 11, column =2, sticky = "nsew")
        self.predtime = Label(main, text = "Time taken: ", font = ("Helvetica", 15))
        self.predtime.grid(row = 12, column =1, sticky = "nsew")
        self.time = Label(main, text = "    ", font=("Helvetica", 15))
        self.time.grid(row = 12, column =2, sticky = "nsew")
        
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.RECORD_SECONDS = 1
        self.WAVE_OUTPUT_FILENAME = "newoutput.wav" 
        self.button = tk.Button(main,text='Record', font=("Helvetica", 20))
        self.button.bind("<ButtonRelease-1>", self.startrecording)
        self.button.grid(row = 11, column = 0, rowspan=2, sticky = "nsew")
        self.LABELS = ['unknown','down','right','no','yes','up','go','off','on','silence','stop','left']  
        
    def startrecording(self, event): 
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
        self.lbl4.config(text=self.predicted_label)
        self.lbl.config(image = self.image)
        self.time.config(text = self.predict_time)
        
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
        
        X_file = self.prepare_data()
        X_file = (X_file - mean)/std 
#        tic = timeit.default_timer()
#       model = load_model('/media/audiodsp/ExternalVolume/Google_speech_command/Real_time_materials/models/best_5.h5')
#        model.load_weights('/media/audiodsp/ExternalVolume/Google_speech_command/Real_time_materials/models/best_5.h5')
#        toc = timeit.default_timer()
#        print(toc-tic) # = 1.8 s
        
        tic = timeit.default_timer()
        with graph.as_default(): # For circumventing keras's problem
             predictions = model.predict(X_file,batch_size=1,verbose=1)
        toc = timeit.default_timer()
        self.predict_time = round((toc-tic),2)
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
        
app = RecordApp(main)
main.mainloop()
