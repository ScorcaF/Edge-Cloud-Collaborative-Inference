#!/usr/bin/env python
# coding: utf-8

# In[225]:


import tensorflow as tf
import numpy as np
import os
import argparse
from tensorflow import keras
import zlib
import tensorflow.lite as tflite
from scipy import signal
from scipy.special import softmax
from base64 import b64encode
from base64 import b64decode
import requests

#Create list with sample names
with open('kws_test_split.txt', 'r') as f:
    test_ds = f.readlines()
test_ds = list(map(lambda line: line[:-1], test_ds))


#Preparing for preprocessing

#Need to tune
sampling_rate = 16000
frame_length =  640
frame_step = 320
lower_frequency = 20
upper_frequency = 4000 
num_mel_bins = 40
num_coefficients = 10
commands = ['stop', 'up', 'yes', 'right', 'left', 'no', 'down', 'go']
num_spectrogram_bins = (frame_length) // 2 + 1

#Preparing tflite model
interpreter = tflite.Interpreter('kws_dscnn_True.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#compute matrix once
linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                                                 num_mel_bins, num_spectrogram_bins, sampling_rate, lower_frequency,
                                                 upper_frequency)
y_true, y_pred = [], []
for file_path in test_ds:
    #Preprocessing-------------------------------
    
    #reading
    parts = tf.strings.split(file_path, '/')
    label = parts[-2]
    
#     label_id = tf.argmax(label == commands)
    audio_binary = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio_binary)
    audio = tf.squeeze(audio, axis=1)
    audio = tf.convert_to_tensor(audio, dtype=tf.float32)
        
    #padding
    zero_padding = tf.zeros([sampling_rate] - tf.shape(audio), dtype=tf.float32)
    audio = tf.concat([audio, zero_padding], 0)
    audio.set_shape([sampling_rate])

    #stft
    stft = tf.signal.stft(audio, frame_length=frame_length,
                frame_step=frame_step, fft_length=frame_length)
    spectrogram = tf.abs(stft)

    #mfcc
    mel_spectrogram = tf.tensordot(spectrogram,
                linear_to_mel_weight_matrix, 1)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    mfccs = mfccs[..., :num_coefficients]
    mfccs = tf.expand_dims(mfccs, 0)
    mfccs = tf.expand_dims(mfccs, -1)

    y_true.append(label)
    
    #Inference-----------------------------------------------------
    
    interpreter.set_tensor(input_details[0]['index'], mfccs)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    
    first = np.sort(softmax(output))[-1]
    second = np.sort(softmax(output))[-2]
    
    #server invoked 7.5% of the times with this policy
    if first - second < 0.4:
        audio_encoded = b64encode(audio_binary.numpy())
        audio_string = audio_encoded.decode()
        
        
        body = {'audio': audio_string}
        r = requests.put('http://localhost:8080', json=body) 
        
        if r.status_code == 200:
            response = r.json()
            y_pred.append(response['keyword'])
            
        else:
            print(r.status_code)
                      
    else:
        keyword = commands[np.argmax(output)]
        y_pred.append(keyword) 
        
    
#Measure accuracy
if len(y_true) ==  len(y_pred):
    num_corrects = (np.array(y_pred) == np.array(y_true)).sum()
    print("Accuracy: {:0.03}%".format(num_corrects*100/len(y_true)))
else:
    print("Error. Predictions missing. Restart.")  

