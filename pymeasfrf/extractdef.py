#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 16:56:51 2022

@author: dupont
"""
import numpy  as np
import matplotlib.pyplot as plt
from pathlib import Path


from scipy.io import wavfile
import os 
import sys
from matplotlib import gridspec

sys.path.append("../../../Manip_test/")
from Toolbox_manip import signal_generation as sgen
from Toolbox_manip.signal_generation import  SignalGen, ChirpSignal, gen_burst
from Toolbox_manip import signal_record as recsig
from Toolbox_manip import signal_processing as sprs
from scipy.io import wavfile


sys.path.append("../../../")
from Toolbox.array_defclass import  Array

#%%


def extract_frf_from_tf(path_2get,n_avg,n_fft = int(48000/3), n_harmonics = 3,len_ir = int( 0.25*48000),pre_ir = int( 0.01*48000)):
    sig = sgen.load_object(path_2get+'/conf_logarithmic_chirp')
    mics = sgen.load_object(path_2get+'/conf_mics_plan')
    sources = sgen.load_object(path_2get+'/conf_sources_plan')
    
    #% processing signal  using A.novak method
    fs = sig.fs
    # n_fft = int(fs/3)
    # len_ir = int( 0.25*fs) # length of the impulse responses hm [samples]
    # pre_ir = int( 0.01*fs) # number of samples before the IR
    
    H = np.zeros((sources.nodes, mics.nodes, n_fft), dtype=complex)
    H_origin = np.zeros((sources.nodes, mics.nodes, sig.n_pts), dtype=complex)
    thd_mat = np.zeros((sources.nodes, mics.nodes, n_fft), dtype=complex)
    
    
    freq_origin = np.arange(0, sig.n_pts) * (fs / sig.n_pts )#% frequency vector of fft IR

    offset_nsource = 0
    
    for i_source in np.arange(sources.nodes) : 
        for i_mic in np.arange(mics.nodes) : 
            print('Source %i∕%i mic %i/%i '%(i_source+1,sources.nodes,i_mic+1,mics.nodes))
            y = []
            for i_avg in np.arange(n_avg):
                filename = 'LS%iMic%i_avg%iover%i.wav'%(i_source+offset_nsource, i_mic, i_avg+1, n_avg)
                fs, y_temp = wavfile.read(path_2get + filename) 
                y.append(y_temp)
            
            # perform impulse resp and mean
            y = np.array(y).T # stacked signal 2 array
            ri_S_fft = sprs.ri_S_fft(sig.f_0,sig.R,fs,y,n_fft = sig.n_pts, opt_short=1) # get frf novak method 
            ri_S =  np.fft.irfft( np.mean(ri_S_fft, axis=1), sig.n_pts ,axis=0) # mean IR 
            
            H_origin[i_source,i_mic,:] = np.mean(ri_S_fft,axis=1)

            #extract non lin
            ir_harmonics = sprs.extract_harmonics(ri_S, sig.R, n_harmonics, fs, len_ir = len_ir, pre_ir = pre_ir )
            
            #assign clean fundamental  
            h_fft = np.fft.fft(ir_harmonics,n_fft, axis=0)
            H[i_source,i_mic,:] = h_fft[:,0]
            
            # get THD 
            f_nfft_h = np.arange(0,n_fft) * (fs /n_fft )#% frequency vector of fft IR
            thd_mat[i_source,i_mic,:]  =  sprs.extract_thd(h_fft, f_nfft_h, n_harmonics, fs,opt_plot=0 )[0]
            
    #% save
    indx = np.arange(int(f_nfft_h.size/2)+1)
    np.savez( path_2get + 'freq.npz', freq = f_nfft_h[indx])
    np.savez( path_2get + 'H_3hz.npz', H =  H[:,:,indx])
    np.savez( path_2get + 'thd_all.npz', thd =  thd_mat[:,:,indx])
    
    # np.savez( path_2get + 'H_origin.npz', H_origin[:,:,np.arange(int(sig.n_pts/2)+1)])
    # np.savez( path_2get + 'freq_origin.npz', freq_origin[indx])

    return f_nfft_h, H, thd_mat, sig