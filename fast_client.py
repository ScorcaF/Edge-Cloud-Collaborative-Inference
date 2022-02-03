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
import time
from sys import getsizeof
from datetime import datetime

# Create list with sample names
with open('kws_test_split.txt', 'r') as f:
    test_ds = f.readlines()
test_ds = list(map(lambda line: line[:-1], test_ds))


# Preparing for preprocessing

# Need to tune
sampling_rate = 16000
frame_length =  480
frame_step = 320
lower_frequency = 20
upper_frequency = 4000
num_mel_bins = 32
num_coefficients = 10

commands = ['stop', 'up', 'yes', 'right', 'left', 'no', 'down', 'go']

num_spectrogram_bins = (frame_length) // 2 + 1

# Preparing tflite model
interpreter = tflite.Interpreter('kws_dscnn_True.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Compute matrix once
linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                                                 num_mel_bins, num_spectrogram_bins, sampling_rate, lower_frequency,
                                                 upper_frequency)
y_true, y_pred = [], []
tot_inf_time = []

communication_cost = []

for file_path in test_ds:

    parts = tf.strings.split(file_path, '/')
    label = parts[-2]

    start_t = time.time()

    # Preprocessing ------------------------------------------------

    # Audio conversion

    audio_binary = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio_binary)
    audio = tf.squeeze(audio, axis=1)
    audio = tf.convert_to_tensor(audio, dtype=tf.float32)

    # Padding
    zero_padding = tf.zeros([sampling_rate] - tf.shape(audio), dtype=tf.float32)
    audio = tf.concat([audio, zero_padding], 0)
    audio.set_shape([sampling_rate])

    # STFT
    stft = tf.signal.stft(audio, frame_length=frame_length,
                frame_step=frame_step, fft_length=frame_length)
    spectrogram = tf.abs(stft)

    # MFCC
    mel_spectrogram = tf.tensordot(spectrogram,
                linear_to_mel_weight_matrix, 1)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    mfccs = mfccs[..., :num_coefficients]
    mfccs = tf.expand_dims(mfccs, 0)
    mfccs = tf.expand_dims(mfccs, -1)

    y_true.append(label)

    # Inference-----------------------------------------------------

    interpreter.set_tensor(input_details[0]['index'], mfccs)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    end_inf = time.time()
    tot_inf_time.append(end_inf - start_t)

    # Success Checker-----------------------------------------------

    first = np.sort(softmax(output))[-1]
    second = np.sort(softmax(output))[-2]

    # Server invocation policy
    if first - second < 0.28:
        audio_encoded = b64encode(audio_binary.numpy())
        audio_string = audio_encoded.decode()
        posixTime = datetime.now().timestamp()
        body = {'bn': 'rpi',
                'e': [{'n':'audio_to_send', 't': posixTime, 'v': audio_string}]}
        communication_cost.append(getsizeof(body))
        communication_cost.append(getsizeof(audio_string))
        # Change with server IP
        r = requests.put('http://169.254.104.205:8080', json=body)

        if r.status_code == 200:
            response = r.json()
            communication_cost.append(getsizeof(response))
            y_pred.append(response['keyword'])

        else:
            print(r.status_code)

    else:
        keyword = commands[np.argmax(output)]
        y_pred.append(keyword)

# Measure accuracy, printing execution times and transfered MB

if len(y_true) ==  len(y_pred):
    num_corrects = (np.array(y_pred) == np.array(y_true)).sum()
    print("Accuracy: {:0.03}%".format(num_corrects*100/len(y_true)))
    print('Fast Total Inference Time: {:.2f}ms'.format(np.mean(tot_inf_time)*1000.))
    print("Communication cost: {:.2f}MB".format(np.sum(communication_cost)/(2**20)))

else:
    print("Error. Predictions missing. Restart.")