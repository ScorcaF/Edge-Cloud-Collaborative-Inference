#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cherrypy
import json
from datetime import datetime
import time
from base64 import b64decode
import pyaudio
import requests
import tensorflow as tf
import wave
from scipy.io import wavfile
from scipy import signal
import tensorflow.lite as tflite


class Classifier(object):
    exposed = True
    
    def __init__(self):
        self.commands = ['stop', 'up', 'yes', 'right', 'left', 'no', 'down', 'go']        

        self.sampling_rate = 16000
        self.frame_length =  640
        self.frame_step = 320
        self.lower_frequency = 20
        self.upper_frequency = 4000 
        self.num_mel_bins = 40
        self.num_coefficients = 10
        self.num_spectrogram_bins = (self.frame_length) // 2 + 1
        self.linear_to_mel_weight_matrix  = tf.signal.linear_to_mel_weight_matrix(self.num_mel_bins, 
                                                                                  self.num_spectrogram_bins, 
                                                                                  self.sampling_rate,
                                                                                  self.lower_frequency,
                                                                                  self.upper_frequency)
                
        #Preparing tflite model
        self.interpreter = tflite.Interpreter('kws_dscnn_True.tflite')
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()


    def GET(self, *path, **query):        
        pass
     
        
    def POST(self, *path, **query):
        pass
        
    def PUT(self, *path, **query):
        
        if len(path) > 0:
            raise cherrypy.HTTPError(400, 'Wrong path')
            
        if len(query) >0:
            raise cherrypy.HTTPError(400, 'Wrong query')
                

        body = cherrypy.request.body.read()
        body = json.loads(body)
        audio_binary = b64decode(body.get("audio"))
        
    
        audio, _ = tf.audio.decode_wav(audio_binary)
        audio = tf.squeeze(audio, axis=1)
        audio = tf.convert_to_tensor(audio, dtype=tf.float32)
        
        #padding
        zero_padding = tf.zeros([self.sampling_rate] - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio, zero_padding], 0)
        audio.set_shape([self.sampling_rate])

        #stft
        stft = tf.signal.stft(audio, frame_length=self.frame_length,
                frame_step=self.frame_step, fft_length=self.frame_length)
        spectrogram = tf.abs(stft)

        #mfcc
        mel_spectrogram = tf.tensordot(spectrogram,
                self.linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :self.num_coefficients]
        mfccs = tf.expand_dims(mfccs, 0)
        mfccs = tf.expand_dims(mfccs, -1)

    
    
        #Predict keyword in input 
        self.interpreter.set_tensor(self.input_details[0]['index'], mfccs.numpy())
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
                    
            
        #Sending data 
        output = {'keyword': self.commands[tf.math.argmax(output[0])]}
        output_str = json.dumps(output)
        return output_str
            
            
    def DELETE(self, *path, **query):
        pass
    
    
if __name__== '__main__':
    conf = {'/':{'request.dispatch': cherrypy.dispatch.MethodDispatcher()}}
    cherrypy.tree.mount(Classifier(), '', conf)
    cherrypy.config.update({'server.socket_host': '0.0.0.0'})
    cherrypy.config.update({'server.socket_port': 8080})
    cherrypy.engine.start()
    cherrypy.engine.block()

