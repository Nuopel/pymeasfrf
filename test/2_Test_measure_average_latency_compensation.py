#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 11:09:37 2022

@author: nuopel
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 10:17:03 2022

@author: samuel
"""

import numpy as np
from numpy import sin,log,log10,pi, arange, exp, zeros, conj,logspace,sqrt, real,imag
import matplotlib.pyplot as plt
from numpy.fft import fft,ifft
from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy.signal import firwin,freqz,butter
import sys

from pymeasfrf import signal_generation as sgen
from pymeasfrf.signal_generation import ChirpSignal
from pymeasfrf import signal_record as recsig
from pymeasfrf import signal_processing as sprs

import sounddevice as sd


#%% Define parameters

fs = 48000
te = 1/fs
t_tot = 1
f_0 = 100
f_end = 4000
n_avg = 2

sig = ChirpSignal(fs=fs,te=te,t_tot=t_tot,f_0 = f_0, f_end = f_end) #♣initiate struct
sig.gen_time() # generate time vector
sig.gen_chirp() #  generate sweep
sig.gen_window(alpha = 0.15).add_window # gen tukey window and add it
sig.add_zeros(time_of_zeros = 0.2) # gen tukey window and add it

plt.figure(1)
sig.plot_time_sig
sig.plot_time_windows
plt.show()

#%% Detect sound card
if 0:
    
    print(sd.query_devices())
    print('\n Chose your soundcard : \n')
    soundcard = int(input(''))
    recsig.select_sound_card(([soundcard,]*2), fs = fs)

else:
    # if you know your sound card
    recsig.select_sound_card(([28,]*2), fs = fs)
    
#%%  
latency_feedback_spk = 1 # channel output for the loop back
latency_feedback_mic = 1 # channel input for the loop back
mic2record = 0 # desired input
spk2record = 0 # desired output channel
path2save_record = './record/latency/'

signal_mic = recsig.record_frf_1spk_2_1mic_latency_measurement_short_signal_input(mic2record,spk2record,latency_feedback_mic,latency_feedback_spk,
                                                sig2play= sig.x, fs=fs, gain_spk = 0.15,n_avg = n_avg, 
                                                t_tot = 0.5, t_burst = 0.01, gain_burst = 0.5,
                                                path2save_record= path2save_record, opt_plot = 1, opt_save = 1 )  

#%%

time = np.arange(0,signal_mic.shape[0])/fs
plt.figure()
plt.plot(time,signal_mic/np.max(signal_mic,axis=0),'--',label=" IR S")

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
          
#%% processing signal  using A.novak method
n_fft = sig.n_pts
ri_S,ri_S_fft,S_fft,y_fft = sprs.ri_S_fft(sig.f_0,sig.R,fs,signal_mic,n_fft)


time = np.arange(0,n_fft)/fs
f_nfft = np.arange(0,n_fft) * (fs / n_fft)


plt.figure()
plt.subplot(211)
plt.plot(time,ri_S,'--')
plt.title(' IR from A novak Method')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(212)
plt.semilogx(f_nfft,20*np.log10(abs(y_fft)*2), '--', label='fft(y)')
plt.semilogx(f_nfft,20*np.log10(abs(S_fft)*2), '--', label='fft(S)')
plt.semilogx(f_nfft,20*np.log10(abs(ri_S_fft)*2), '--', label='fft(y/S)')
plt.xlim(50,fs/2)

plt.subplot(212)

plt.title(' Fft A.Novak' )
plt.xlabel('Frequency')
plt.ylabel('dB')

#%%

ri_S_fft_final = ri_S_fft.mean(axis=1)
ri_S_final =  np.fft.irfft(ri_S_fft_final,n_fft,axis=0)


plt.figure(8)
plt.subplot(211)
plt.plot(time,ri_S_final,'--')
plt.title(' IR from A novak Method')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(212)
plt.semilogx(f_nfft,20*np.log10(abs(ri_S_fft_final)*2), '--', label='fft(y/S)')
plt.ylim(-100,0)

plt.title(' Fft A.Novak' )
plt.xlabel('Frequency')
plt.ylabel('dB')

#%%
