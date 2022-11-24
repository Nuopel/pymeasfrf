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

from pymeasfrf.signal_generation import ChirpSignal
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
    # recsig.test_one_channel(0,sig.x,fs, gain = 0.09)
    # help to set the sound card and avoid weird trigger effect at the beginning of the measurement for bad sound card
    signal_in = recsig.record_1_mic(channel2record=1, record_time=1, fs=fs, path2save_record='./record/', opt_plot=0)

latency_feedback_spk = 1  # channel output for the loop back
latency_feedback_mic = 1  # channel input for the loop back
mic2record = 0  # desired input
spk2record = 0  # desired output channel
path2save_record = './record/latency/'

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
plt.plot(time, signal_mic / np.max(signal_mic, axis=0), '--', label=" IR S")

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

plt.show()

# %% processing signal  using A.novak method
n_fft = sig.n_pts
ri_S_fft, ri_S, S_fft, y_fft = sprs.ri_S_fft(sig.f_0, sig.R, fs, signal_mic, n_fft)

time = np.arange(0, n_fft) / fs
f_nfft = np.arange(0, n_fft) * (fs / n_fft)

plt.figure()
plt.subplot(211)
plt.plot(time, ri_S, '--')
plt.title(' IR from A novak Method')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(212)
with np.errstate(divide='ignore'):
    plt.semilogx(f_nfft, 20 * np.log10(abs(y_fft) * 2), '--', label='fft(y)')
    plt.semilogx(f_nfft, 20 * np.log10(abs(S_fft) * 2), '--', label='fft(S)')
    plt.semilogx(f_nfft, 20 * np.log10(abs(ri_S_fft) * 2), '--', label='fft(y/S)')
plt.xlim(50, fs / 2)
plt.legend()

plt.subplot(212)

plt.title(' Fft A.Novak')
plt.xlabel('Frequency')
plt.ylabel('dB')

plt.show()

# %%

ri_S_fft_final = ri_S_fft.mean(axis=1)
ri_S_final = np.fft.irfft(ri_S_fft_final, n_fft, axis=0)

with np.errstate(divide='ignore'):
    plt.figure(8)
    plt.subplot(211)
    plt.plot(time, ri_S_final, '--')
    plt.title(' IR from A novak Method')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.subplot(212)
    plt.semilogx(f_nfft, 20 * np.log10(abs(ri_S_fft_final) * 2), '--', label='fft(y/S)')
    plt.xlim(sig.f_0 / 2, fs / 2)
    # plt.ylim(-120, 0)

    plt.title(' Fft A.Novak')
    plt.xlabel('Frequency')
    plt.ylabel('dB')
    plt.show()

# %% extract nonlin

n_harmonics = 2
len_ir = int(0.25 * fs)  # length of the impulse responses hm [samples]
pre_ir = int(0.05 * fs)  # number of samples before the IR

hm = sprs.extract_harmonics(ri_S_final, sig.R, n_harmonics, fs, len_ir=len_ir, pre_ir=pre_ir)
time_h = np.arange(0, hm.shape[0]) / fs

# % fft of impulse
h_fft = fft(hm, axis=0)
f_nfft_h = np.arange(0, hm.shape[0]) * (fs / hm.shape[0])

# get THD
thd, thd_mat = sprs.extract_thd(h_fft, f_nfft_h, n_harmonics, fs)

# plot
np.seterr(divide='ignore')
plt.figure()
plt.subplot(211)
label1 = list(map(lambda _: ('hrmnc = {}').format(_), arange(n_harmonics + 1)))
plt.plot(time, ri_S_final / np.max(abs(ri_S_final), axis=0), 'red', label=" IR S")

plt.plot(time_h, real(hm) / np.max(abs(hm), axis=0), '--', label=label1)
plt.title(' IR from A novak Method')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.xlim([0.00, len_ir / fs])
# plt.legend()

plt.subplot(212)
plt.semilogx(f_nfft, 20 * np.log10(abs(ri_S_fft_final) * 2), 'red', label='fft(y/S)')
plt.semilogx(f_nfft_h, 20 * np.log10(abs(h_fft) * 2), '--', label=label1)
plt.xlim(sig.f_0 / 2, fs / 2)
plt.ylim(-120, 0)

plt.title(' Fft A.Novak')
plt.xlabel('Frequency')
plt.ylabel('dB')
plt.legend()
plt.show()

plt.figure()
plt.subplot(211)
plt.semilogx(f_nfft_h, thd)
plt.ylim([0, 20])
plt.xlim([sig.f_0, sig.f_end])

plt.subplot(212)
label1 = list(map(lambda _: ('h = {}').format(_), arange(n_harmonics + 1)))
plt.semilogx(f_nfft_h, 20 * np.log10(abs(thd_mat) * 2), '--', label=label1)
plt.xlim([sig.f_0, sig.f_end])
# plt.ylim(-120,0)

plt.title(' Scaled FFTs')
plt.xlabel('Frequency')
plt.ylabel('dB')
plt.legend()
plt.show()
