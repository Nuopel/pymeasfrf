# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 10:17:03 2022

@author: samuel
"""

import numpy as np
from numpy import sin, log, log10, pi, arange, exp, zeros, conj, logspace, sqrt, real, imag
import matplotlib.pyplot as plt
from scipy.io import wavfile

from pathlib import Path

from pymeasfrf import signal_generation as sgen
from pymeasfrf.signal_generation import ChirpSignal
from pymeasfrf import signal_record as recsig
from pymeasfrf import signal_processing as sprs

import sounddevice as sd

# %% Define parameters

fs = 48000
te = 1 / fs
t_tot = 1
f_0 = 100
f_end = 16000
n_avg = 2  # note that if other than 1 delay of the sound card might mess up the result

sig = ChirpSignal(fs=fs, te=te, t_tot=t_tot, f_0=f_0, f_end=f_end)  # ♣initiate struct
sig.gen_time()  # generate time vector
sig.gen_chirp()  # generate sweep
sig.gen_window(alpha=0.15).add_window  # gen tukey window and add it
sig.add_zeros(time_of_zeros=0.2)  # gen tukey window and add it

plt.figure(1)
sig.plot_time_sig(undersample=5)  # plot 1 over (undersample) points
sig.plot_time_windows
plt.show()

# %% Detect sound card
if 0:

    print(sd.query_devices())
    print('\n Chose your soundcard : \n')
    soundcard = int(input(''))
    recsig.select_sound_card(([soundcard, ] * 2), fs=fs)
    # sd.default.latency = ('high', 'high')

else:
    # if you know your sound card
    recsig.select_sound_card(([1,3 ] ), fs=fs)

# %% record the signal
test = True
if test == True:
    # recsig.test_one_channel(0,sig.x,fs, gain = 0.09)
    # help to set the sound card and avoid weird trigger effect at the beginning of the measurement for bad sound card
    signal_in = recsig.record_1_mic(channel2record=1, record_time=1, fs=fs, path2save_record='./record/', opt_plot=0)

if test == True:
    path2save_record = './record/sweep/'

    Path(path2save_record).mkdir(parents=True, exist_ok=True)
    sig.save_self(path2save_record + sig.name + '_conf')
    y = recsig.record_frf_1spk_2_1mic(mic2record=0, spk2record=0, sig2play=sig.x, gain=0.9, n_avg=n_avg, FS=fs,
                                      path2save_record=path2save_record, opt_plot=0, opt_save=1)

else:
    path2save_record = './record/sweep/'
    sig = sgen.load_object(path2save_record + 'logarithmic_chirp_conf')
    fs, y = wavfile.read(path2save_record + 'LS[0]Mic[0]_avg1over3.wav')

# % plot measured the signal
plt.figure()
time = np.arange(0, y.shape[0]) / fs
plt.subplot(211)
plt.plot(time, y, '--')
plt.title(' Measured signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
# %% processing signal  using A.novak method

n_fft = sig.n_pts * 2
ri_S_fft, ri_S, S_fft, y_fft = sprs.ri_S_fft(sig.f_0, sig.R, fs, y, n_fft)

time = np.arange(0, n_fft) / fs
f_nfft = np.arange(0, n_fft) * (fs / n_fft)

with np.errstate(divide='ignore'):
    plt.figure()
    plt.subplot(211)
    plt.plot(time, ri_S, '--', label=" IR S")
    plt.title(' IR from A novak Method')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    # plt.xlim([0.00,0.006])
    plt.legend()

    plt.subplot(212)
    plt.semilogx(f_nfft, 20 * np.log10(abs(y_fft) * 2), '--', label='fft(y)')
    plt.semilogx(f_nfft, 20 * np.log10(abs(S_fft) * 2), '--', label='fft(S)')
    plt.semilogx(f_nfft, 20 * np.log10(abs(ri_S_fft) * 2), '--', label='fft(y/S)')
    plt.xlim(1, fs / 2)
    plt.ylim(-50, 80)

    plt.title(' Fft A.Novak')
    plt.xlabel('Frequency')
    plt.ylabel('dB')
    plt.legend()
    plt.show()

# %% do the mean
ri_S_fft_mean = np.mean(ri_S_fft, axis=1)
ri_S_mean = np.fft.irfft(ri_S_fft_mean, n_fft, axis=0)

with np.errstate(divide='ignore'):
    plt.figure()
    plt.subplot(211)
    time = np.arange(0, ri_S_mean.size) / fs

    plt.plot(time, ri_S_mean, '--')
    plt.title(' IR mean')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    # plt.xlim([0.00,0.006])

    plt.subplot(212)
    plt.semilogx(f_nfft, 20 * np.log10(abs(ri_S_fft_mean) * 2), '-')
    plt.xlim(1, fs / 2)
    # plt.ylim(-80, 0)

    plt.title(' Fft mean')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('dB')
    plt.show()

