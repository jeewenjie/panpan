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
from PIL import Image, ImageTk

class RecordApp():
    
    def __init__(self,master):
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.RECORD_SECONDS = 2
        self.WAVE_OUTPUT_FILENAME = "output.wav"
        self.isrecording = False
        self.button = tk.Button(main,text='Record', height = 5, width = 10)
        self.button.bind("<ButtonRelease-1>", self.startrecording)
        self.button.pack()

    def startrecording(self, event): 
        self.isrecording = True
        t = threading.Thread(target = self._record)
        t.start()
    
    def _record(self):

        print('Recording')
        frames = []
        p = pyaudio.PyAudio()  

        for i in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
            stream = p.open(format=self.FORMAT,channels=self.CHANNELS,rate=self.RATE,input=True,input_device_index=3,frames_per_buffer=self.CHUNK)
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
        
        _, signal = scipy.io.wavfile.read('output.wav')

        plt.figure(0)
        signal = signal/abs(max(signal)) 
        signal = np.asarray(signal, dtype=np.float32)
        signal = librosa.resample(signal,44100,16000)
        scipy.io.wavfile.write('newoutput.wav',16000,signal)
        
#        canvas = tk.Canvas(main, width = 300, height = 300)
#        image = Image.open("normalizedsignal.png")
#        photo = ImageTk.PhotoImage(image)
#        canvas.create_image(20,20, image=photo) 

main  = tk.Tk()
app = RecordApp(main)
main.mainloop()

#_, signal = scipy.io.wavfile.read('output.wav')

#plt.figure(0)
#plt.title('Signal Wave')
#plt.plot(signal)
#plt.show()

#signal = signal/abs(max(signal))

#signal = np.asarray(signal, dtype=np.float32)
#signal = librosa.resample(signal,44100,16000)
#scipy.io.wavfile.write('newoutput.wav',16000,signal)

#rate, data = scipy.io.wavfile.read('newoutput.wav')

#plt.figure(1)
#plt.title('Normalized Signal Wave')
#plt.plot(data)
#plt.savefig('normalizedsignal.png', bbox_inches='tight')
#plt.show()

