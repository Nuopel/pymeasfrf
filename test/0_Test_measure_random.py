# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 10:17:03 2022

@author: samuel
"""

import numpy as np
from numpy import sin, log, log10, pi, arange, exp, zeros, conj, logspace, sqrt, real, imag
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft
from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy.signal import firwin, freqz, butter
import sys

from pymeasfrf import signal_generation as sgen
from pymeasfrf.signal_generation import RandomSignal
from pymeasfrf import signal_record as recsig
from pymeasfrf import signal_processing as sprs

import sounddevice as sd

# %% Define parameters

fs = 48000
te = 1 / fs
t_tot = 2

sig = RandomSignal(fs=fs, te=te, t_tot=t_tot)
sig.gen_time()  # generate time vector
sig.gen_random_uniform()  # generate sweep
sig.gen_window(alpha=0.15).add_window  # gen tukey window and add it
sig.add_zeros(time_of_zeros=0.2)  # gen tukey window and add it

plt.figure(1)
sig.plot_time_sig(undersample=20)  # plot 1 over (undersample) points
sig.plot_time_windows
plt.show()

# %% Detect sound card
if 1:
    print(sd.query_devices())
    print('\n Chose your soundcard : \n')
    soundcard = int(input(''))
    recsig.select_sound_card(([soundcard, ] * 2), fs=fs)

else:
    # if you know your sound card
    recsig.select_sound_card(([28, ] * 2), fs=fs)

# %% play sound test, then record test
test = True
if test == True:
    # recsig.test_one_channel(0,sig.x,fs, gain = 0.09)
    # help to set the sound card and avoid possible weird trigger effect at the beginning of the measurement
    signal_in = recsig.record_1_mic(channel2record=0, record_time=1, fs=fs, path2save_record='./record/', opt_plot=0)

if test == True:

    path2save_record = './record/random/'
    y = recsig.record_frf_1spk_2_1mic(mic2record=0, spk2record=0, sig2play=sig.x, gain=0.5, FS=fs,
                                      path2save_record=path2save_record, opt_plot=0)
    sig.save_self(path2save_record + sig.name + '_conf.plk')

else:
    path2save_record = './record/random/'
    sig = sgen.load_object(path2save_record + 'random_conf')
    fs, y = wavfile.read(path2save_record + 'LS0Mic0.wav')

# %% processing signal  using y/x
n_fft = int(sig.t_tot * fs)
sig.x_fft = fft(sig.x, n_fft)
y_fft = fft(y, n_fft)

ri, ri_fft = sprs.ri_fft_y_over_x(y_fft, sig.x_fft, n_fft)

time = np.arange(0, (ri.size) / fs, 1 / fs)
f_nfft = np.arange(0, n_fft) * (fs / n_fft)

timexy = np.arange(0, (ri.size)) / fs

plt.figure()
plt.subplot(211)
plt.plot(timexy, np.real(ri), '--', label=" IR S")
plt.title(' IR sortie/entree')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
# plt.xlim([0.00,0.09])
plt.legend()

plt.subplot(212)
plt.semilogx(f_nfft, 20 * np.log10(abs(y_fft) * 2), '--', label='fft(y)')
plt.semilogx(f_nfft, 20 * np.log10(abs(sig.x_fft) * 2), '--', label='fft(x)')
plt.semilogx(f_nfft, 20 * np.log10(abs(ri_fft) * 2), '--', label='fft(y/x)')

plt.xlim(1, fs / 2)
plt.ylim(-170, 150)

plt.title(' Fft sortie/entree')
plt.xlabel('Frequency')
plt.ylabel('dB')
plt.legend()

# %% using Cttm approach
f_estm, H_estm = sprs.transfunct(sig.x, y, fs, nperseg=sig.n_pts / 10, noverlap=None, H_type='H3', window='hann')
ri_estm = np.fft.irfft(H_estm, H_estm.size * 2)

plt.figure()
time_estm = np.arange(0, (ri_estm.size)) / fs
plt.subplot(211)
plt.plot(time_estm, ri_estm, '--', label=" IR H3")
plt.title(' IR from estimator Method')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(212)

plt.semilogx(f_estm, 20 * np.log10(abs((H_estm)) * 2), '-', label='Estm H3')
plt.xlim(50, fs / 2)
plt.ylim(-50, 80)
# plt.xlim(2500,3000)
# plt.ylim(-30,5)
plt.title(' Fft Estm')
plt.xlabel('Frequency')
plt.ylabel('dB')
plt.legend()

# %% using convolution deconvolution
import scipy.signal as scsig

x_flipped = np.flip(sig.x)

ri_conv = scsig.convolve(y, x_flipped)

ri_conv = np.roll(ri_conv, int(sig.t_tot * fs))

x_fft_flip = fft(x_flipped, n_fft)

norm = np.sqrt(sum(abs((sig.x_fft * np.conj(x_fft_flip)) ** 2)) / ri_conv.size)

yeff = norm
ri_conv = ri_conv / yeff
ri_conv_fft = fft(ri_conv, n_fft)

time_conv = np.arange(0, ri_conv.size) / fs

plt.figure()
plt.subplot(211)
plt.plot(time_conv, ri_conv, '--', label=" IR conv")
plt.title(' IR from convolution Method')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
# plt.xlim([0.00,0.006])
plt.legend()

plt.subplot(212)
plt.semilogx(f_nfft, 20 * np.log10(abs(ri_conv_fft) * 2), '--', label='convolution')
plt.xlim(1, fs / 2)
# plt.ylim(-50,80)
plt.title(' Fft Estm')
plt.xlabel('Frequency')
plt.ylabel('dB')
plt.legend()

# %%
plt.figure()
plt.semilogx(f_nfft, 20 * np.log10(abs(ri_fft) * 2), '--', label='fft(y/x)')
plt.semilogx(f_nfft, 20 * np.log10(abs(ri_conv_fft) * 2), '--', label='convolution')
plt.semilogx(f_estm, 20 * np.log10(abs(H_estm) * 2), '--', label='H3')

plt.xlim(150, fs / 4)
plt.ylim(-80, 20)

plt.title(' Fft Estm')
plt.xlabel('Frequency')
plt.ylabel('dB')
plt.legend()

# %% compra time domain
plt.figure()
plt.plot(timexy, ri, '-', label=" IR yx")
plt.plot(time_conv, ri_conv, '-', label=" IR conv")
plt.plot(time_estm, ri_estm, '-.', label=" IR H3")

plt.title(' IR from convolution Method')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
# plt.xlim([0.05,0.055])
plt.legend()
