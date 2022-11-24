#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 11:09:37 2022

@author: nuopel
"""

import numpy as np
from numpy import sin, log, log10, pi, arange, exp, zeros, conj, logspace, sqrt, real, imag, concatenate, linspace
from numpy.fft import fft, ifft

import matplotlib.pyplot as plt

from pymeasfrf.signal_generation import ChirpSignal, RandomSignal
from pymeasfrf import signal_record as recsig
from pymeasfrf import signal_processing as sprs

import sounddevice as sd

# %% Define parameters
fs = 48000
te = 1 / fs
t_tot = 2
f_0 = 50
f_end = 20000
n_avg = 3

sig = ChirpSignal(fs=fs, te=te, t_tot=t_tot, f_0=f_0, f_end=f_end)  # ♣initiate struct
sig.gen_time()  # generate time vector
sig.gen_chirp()  # generate sweep
sig.gen_window(alpha=0.05).add_window  # gen tukey window and add it
sig.add_zeros(time_of_zeros=0.2)  # gen tukey window and add it

plt.figure(1)
sig.plot_time_sig()
sig.plot_time_windows
plt.show()


# %% Detect sound card
cx = 4
if cx == 0:

    print(sd.query_devices())
    print('\n Chose your soundcard : \n')
    soundcard = int(input(''))
    recsig.select_sound_card(([soundcard, ] * 2), fs=fs)
    sd.default.latency = ('high', 'high')

elif cx == 1:

    print(sd.query_devices())
    print('\n Chose your soundcard input: \n')
    soundcard_in = int(input(''))
    print('\n Chose your soundcard output: \n')
    soundcard_out = int(input(''))
    recsig.select_sound_card(([soundcard_in, soundcard_out]), fs=fs)
    sd.default.latency = ('high', 'high')

else:
    # if you know your sound card
    recsig.select_sound_card(([0, ] * 2), fs=fs)
    sd.default.latency = ('low', 'low')

# %%
test = True
if test == True:
    # help to set the sound card and avoid weird trigger effect at the beginning of the measurement for bad sound card
    signal_in = recsig.record_1_mic(channel2record=1, record_time=1, fs=fs, path2save_record='./record/', opt_plot=0)

latency_feedback_spk = 1  # channel output for the loop back
latency_feedback_mic = 1  # channel input for the loop back
mic2record = [0, 1]  # desired input
spk2record = 0  # desired output channel
path2save_record = './record/latency/sweep/'

signal_mic = recsig.record_frf_1spk_2_1mic_latency_measurement_short_signal_input(mic2record, spk2record,
                                                                                  latency_feedback_mic,
                                                                                  latency_feedback_spk,
                                                                                  sig2play=sig.x, fs=fs, gain_spk=0.05,
                                                                                  n_avg=n_avg,
                                                                                  t_tot=0.5, t_burst=0.1,
                                                                                  gain_burst=0.7,
                                                                                  path2save_record=path2save_record,
                                                                                  opt_plot=1, opt_save=1)
plt.show()

# %%

time = np.arange(0, signal_mic.shape[0]) / fs
plt.figure()
for ii in range(len(mic2record)):
    plt.plot(time, signal_mic[:,0,:] / np.max(signal_mic[:,ii,:], axis=0), '--', label=" sweep")

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

plt.show()
