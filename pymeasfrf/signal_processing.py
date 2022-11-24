#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 17:30:27 2022

@author: dupont
"""

import numpy as np
from numpy import sin,log,log10,pi, arange, exp, zeros, conj,logspace,sqrt, real,imag, concatenate,linspace
from numpy.fft import fft,ifft

from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import freqz
import scipy.signal as scsig

#%% processing methods



def ri_fft_y_over_x(y_fft,x_fft,n_ifft=None):
    if n_ifft is None:
        n_ifft  = y_fft.size
    ri_fft = y_fft/x_fft
    ri =  np.fft.ifft(ri_fft,n_ifft)
    return ri, ri_fft

def synchronized_swept_sine_spectra(f_0,R,fs,n_pts):
    f = np.linspace(0,fs,n_pts+1)
    f = f[0:-1]
    with np.errstate(divide='ignore', invalid='ignore'): # remove /0 error due to f = 0
        S = 1/2*np.sqrt(R/f)*np.exp(1j*2*pi*f*R*(1 - np.log(f/f_0)) - 1j*pi/4)*fs
    S[0] = np.inf
    return S

def ri_S_fft(f_0,R,fs,y,n_fft =[], opt_short = 0):
    if isinstance(n_fft, list):
        n_fft = y.shape[0]
        
    if n_fft < y.shape[0] :
        print('--------------------------------------------------------')
        print('Be carefull Nfft smaller than signal, thus it is cropped')
        print('--------------------------------------------------------')
        
    y_fft =  np.fft.fft(y,n_fft,axis=0)
    S_fft = synchronized_swept_sine_spectra(f_0, R ,fs ,n_fft)
    if len(y.shape) > 1:
        S_fft = S_fft[:,np.newaxis]
    ri_S_fft =  y_fft/S_fft
    if opt_short == 1:
        return ri_S_fft
    else:
        ri_S =  np.fft.irfft(ri_S_fft,n_fft,axis=0) # working because everything is cropped after n_fft/2+1
        return ri_S_fft,ri_S,S_fft,y_fft



def transfunct(x,y,fs, nperseg=None, noverlap=None, H_type='H3',  window = 'hann'):
    '''
     x : array_like
         Time series of measurement values
     y : array_like
         Time series of measurement values
     fs : float, optional
         Sampling frequency of the `x` and `y` time series. Defaults
         to 1.0.
     window : str or tuple or array_like, optional
         Desired window to use. If `window` is a string or tuple, it is
         passed to `get_window` to generate the window values, which are
         DFT-even by default. See `get_window` for a list of windows and
         required parameters. If `window` is array_like it will be used
         directly as the window and its length must be nperseg. Defaults
         to a Hann window.
     nperseg : int, optional
         Length of each segment. Defaults to None, but if window is str or
         tuple, is set to 256, and if window is array_like, is set to the
         length of the window.
     noverlap: int, optional
         Number of points to overlap between segments. If `None`,
         ``noverlap = nperseg // 2``. Defaults to `None`.   
     '''
    if H_type == 'H1':
        f, Sxy = signal.csd(x, y, fs=fs, window=window, 
                      nperseg=nperseg, noverlap=noverlap, detrend=False, 
                      return_onesided=True, scaling='spectrum', axis=0, average='mean')
        f, Sxx = signal.csd(x, x, fs=fs, window=window, 
                      nperseg=nperseg, noverlap=noverlap,  detrend=False, 
                      return_onesided=True, scaling='spectrum', axis=0, average='mean')
    
        H_mod = abs(Sxy/Sxx)
        H_phase = np.angle(Sxy,deg=False)
        H = H_mod*np.exp(1j*H_phase)
        # H = Sxy/Sxx
    
    elif H_type == 'H2': 
        f, Syx = signal.csd(y, x, fs=fs, window=window, 
                      nperseg=nperseg, noverlap=noverlap, detrend=False, 
                      return_onesided=True, scaling='spectrum', axis=0, average='mean')
        f, Syy = signal.csd(y, y, fs=fs, window=window, 
                      nperseg=nperseg, noverlap=noverlap,  detrend=False, 
                      return_onesided=True, scaling='spectrum', axis=0, average='mean')
        H_mod = abs(Syy/Syx)
        H_phase = np.angle(Syx, deg=False)
        H = H_mod*np.exp(1j*H_phase)
        
    elif H_type=='H3':
        f, Sxy = signal.csd(x, y, fs=fs, window=window, 
                      nperseg=nperseg, noverlap=noverlap,  detrend=False, 
                      return_onesided=True, scaling='spectrum', axis=0, average='mean')
        f, Sxx = signal.csd(x, x, fs=fs, window=window, 
                      nperseg=nperseg, noverlap=noverlap,  detrend=False, 
                          return_onesided=True, scaling='spectrum', axis=0, average='mean')
        f, Syx = signal.csd(y, x, fs=fs, window=window, 
                      nperseg=nperseg, noverlap=noverlap, detrend=False, 
                      return_onesided=True, scaling='spectrum', axis=0, average='mean')
        f, Syy = signal.csd(y, y, fs=fs, window=window, 
                      nperseg=nperseg, noverlap=noverlap, detrend=False, 
                      return_onesided=True, scaling='spectrum', axis=0, average='mean')
    
        H = np.sqrt(Syy/Sxx)*Sxy/abs(Sxy)
    
    else: print('ERROR: Transfer Function estimator H_type is not defined correctly')
    
    return f, H

def ri_conv_time(f_end,f_0,time,t_tot,fs,y,x):
    # R = np.log(f_end/f_0)      
    # k = np.exp(time*R/t_tot)
    x_flipped = np.flip(x)
    ri_conv = scsig.convolve( y, x_flipped,'same')/fs
        
    ri_conv = np.roll(ri_conv , int(t_tot*fs/2))

    return ri_conv, x_flipped



def freq2time(Q,f_q,M,fs,b = [],a =[],axis = -1,n_ifft = None):
    '''
    freq2time(Q,f_q,M,fs,b = [],a =[],axis = -1,n_ifft = None)
        Convert a frequency signal to a impulse response filter 

        Parameters: 
                    :param Q: frequency signal vectors to convert (N_channel,N_freq)
                    :param f_q: frequency vector (N_freq,)
                    :param M: Number of frequency in the interpolation vector
                    :param fs: sampling frequency
                    :param b: b ir/fir filterparameter
                    :param a: a ir/fir filterparameter
                    :param axis: axis of fft, default = -1
                    :param n_ifft: number of point in the ifft
        Return:
                    :return ir: impulse response filter (N_channel, ...axis(,N_time))
                    :return ir_fft: interpolated frequency signal vectors (N_channel, ...axis(,N_freq))
                    :return f_new: interpolated frequency vector


        Author1: Samuel Dupont
        Date:    Juin 2022
    '''

    M = int(M) # force  to int

    if n_ifft == None:
        n_ifft = M
    n_ifft = int(n_ifft) # force  to int
        
    
# Full freq. vector
    f_new = np.arange(0, M)/M*fs/2     

# Real-Imag Interpolation
    Re = real(Q)
    Im = imag(Q)
                
    # Complete frequency band
    if f_q[0] != 0 :
        shape2add = np.array(Re.shape) # get shape for later reshape (allow axis use)
        shape2add[axis] = 1
        
        
        f_q = np.concatenate((np.array([0]),f_q))         
        Re = np.concatenate(( np.ones(shape2add)*1e-6, Re),axis=axis)
        Im = np.concatenate(( np.ones(shape2add)*1e-6, Im),axis=axis)
        # Re = np.hstack((Re[0], Re))
        # Im = np.hstack((Im[0], Im))

    if f_q[-1] != fs/2 :
        shape2add = np.array(Re.shape) # get shape for later reshape (allow axis use)
        shape2add[axis] = 1
    
        f_q = np.concatenate(( f_q, np.array([fs/2]) ))   
        Re = np.concatenate((Re, np.ones(shape2add)*1e-6), axis=axis)
        Im = np.concatenate((Im, np.ones(shape2add)*1e-6), axis=axis)
        
        # Re = np.hstack((Re, Re[-1])) 
        # Im = np.hstack((Im, Im[-1]))            
    
# Interpolation    
    Q_2interp = interp1d(f_q, Re + 1j * Im, kind = 'linear',fill_value = (0,0), bounds_error = False, axis = axis )                      
    ir_fft = Q_2interp(f_new) 
                    
# No-Imag at 0 and Nyqst.
    slc = [slice(None)] * len(ir_fft.shape)
    slc[axis] = [0,-1]
    ir_fft[tuple(slc)]   = real(ir_fft[tuple(slc)])
    
    
    if isinstance(a, list)!=1 and isinstance(b, list)!=1:
        shape2add = np.ones(len(ir_fft.shape),dtype=int)
        shape2add[axis] = ir_fft.shape[axis]
        
        # Filter to avoid Gibbs.
        w, h = freqz(b, a, worN=int(M), fs = fs)
        ir_fft = ir_fft * h.reshape(shape2add)
        
    elif  isinstance(b, list)!=1:
        shape2add = np.ones(len(ir_fft.shape),dtype=int)
        shape2add[axis] = ir_fft.shape[axis]
        # Filter to avoid Gibbs.
        w, h = freqz(b, [1], worN=int(M), fs = fs)
        ir_fft = ir_fft * abs(h.reshape(shape2add))

                      
    # compute impulse response
    ir = np.fft.irfft(ir_fft, n = n_ifft,  axis = axis)  
    
    # compensate the delay introduced by the filter
    # IR = np.roll(IR , -int(len(b)/2),axis=axis)

    return ir,ir_fft,f_new



def roll_0_pad(a,n_shift,axis=0):
    """ roll_0_fwrd(a,n_shift,axis=0):
        Shift an array of n_shift and put zero in the first n_shift sample.
        
    Parameters
    ----------
        a : array to roll
        n_shift : number of shift (int)
        axis : default 0
    """
    a = np.roll(a,n_shift, axis=axis)
    if n_shift > 0 :
        slc = [slice(None)] * len(a.shape)
        slc[axis] = slice(0, n_shift)
        a[tuple(slc)] = 0
    else :
        slc = [slice(None)] * len(a.shape)
        slc[axis] = slice(n_shift, None)
        a[tuple(slc)] = 0
        
    return a

def extract_harmonics(h, R, n_harmonics, fs, len_ir = 2^13, pre_ir = 0 ):
    '''
    Parameters
    ----------
    h : array
        impulse response with nonlinearity from sweep method.
    R : float
        Logaritmic increment of the sweep.
    n_harmonics : int
        Number of desired harmonics + the fundamental.
    fs : int
        Sampling frequency.
    len_ir : int , optional
        [sample] length of output impulse response . The default is 2^13.
    pre_ir : int, optional
        [sample] delay before impulse response. The default is 0.

    Returns
    -------
    hm : Array [len_ir x n_harmonics+1]
        impulse responses of the fundamental + the harmonics .
        
    Definition : Extract nonlinearity from a sweep impulse response
    
    Author : Samuel Dupont 03/2022 (Adapted from A.novak matlab code)
    ----------

    '''
    n_harmonics = n_harmonics + 1
    pos_harmonics = R * log(arange(n_harmonics)+1) # position of the harmonics  [seconds]
    
        
    # rounded positions [samples]
    dt_ = (np.round(pos_harmonics * fs)).astype(int)
    
    # non-integer difference
    dt_rem = pos_harmonics * fs - dt_
    
    
    # circular periodisation of IR
    h_pos = concatenate((h, h[0:len_ir]))
    
    
    # frequency axis definition (0 .. 2pi)
    axe_w = linspace(0,2*pi,len_ir+1)
    axe_w = np.delete(axe_w,-1)
    
    # memory localisation for IRs
    hm = zeros((len_ir,n_harmonics),dtype=complex)
    
    # last sample poition
    st0 = h_pos.size
    
    for n in arange(n_harmonics):
        
        if n > 0 and R * (log(n+1)-log(n)) *fs < (len_ir) and  pre_ir == 0:
            print('From the sweep params, delay of harmonic %i is %g seconds and windows is %g s'%(n, np.round(R * (log(n+1)-log(n)),2),np.round(len_ir/fs,2) ))
            print('You might got tail effect from harmonic %i if pre_ir not set up '%(n-1))
            
        if n > 0 and R * (log(n+2)-log(n+1)) *fs < pre_ir +0.01 * fs :
            print('You might got  the harmonic %i along with the harmonic %i  due to too long pre_ir'%(n+1,n))
            print('')
        # start position of n-th IR
        st = h.size - dt_[n] - pre_ir + 1
        
        # end position of n-th IR
        ed = min(st + len_ir, st0)
        
        # separation of IRs
        hm[0:(ed-st),n] = h_pos[st:ed]
        
        # Non-integer sample delay correction
        Hx = fft(hm[:,n]) * exp(-1j*dt_rem[n]*axe_w)
        hm[:,n] = ifft(Hx)
        
        # last sample poition for the next IR
        st0 = st - 1
    return hm


def extract_thd(ir_harms_fft, freq_ir, n_harms, fs, opt_plot = 0 ):
    '''
    thd, thd_mat = extract_thd(ir_harms_fft, freq_ir, n_harms, fs, opt_plot = 0 )
    
    Parameters
    ----------
    ir_harms_fft :  array [N_freq x N_harmonics]
        impulse response matrix (fundamental and harmonics)
    freq_ir : array
        frequency vector associated to frf [N_freq,]
    n_harms : int
        number of harmonics.
    fs : int
        Sampling frequency.

    Returns
    -------
    thd : array [N_freq,]
        Total harmonic distortion of the given matrix
    thd_mat : array [N_freq x N_harmonics]
        Scaled matrix of harmonics (harmonics in front of the fundamental frequency_wise) 
    
    
    Definition : Calculate THD from frf matrix of fundamental and harmonics  [N_pts x N_harmonics]
    
    Author : Samuel Dupont 03/2022 
    ----------

    '''
    # set harmonique equal 2 zeros after limit
    for i_harmo in arange(1,n_harms+1):
        ir_harms_fft[freq_ir>fs/2,i_harmo]=10**(-150/20)
    
    
    thd_mat = zeros(ir_harms_fft.shape,dtype=complex)# matrix for matching frequency scaling 
    thd_mat[:,0]=ir_harms_fft[:,0] # fundamental saty the same
    
    # interpolate inbeween points for each harmoniques to match 
    for idx, i_harmo in enumerate(arange(1,n_harms+1)):
        test = freq_ir < fs/2
        f_2interp = freq_ir[test]/(i_harmo+1)
        f_2interp[-1]=  freq_ir[-1]
        
        # Interpolation    
        Q_2interp = interp1d(f_2interp,ir_harms_fft[test,i_harmo], kind = 'linear',fill_value = (0,0), bounds_error = False )                      
        thd_mat[:,idx+1]= Q_2interp(freq_ir) 
        
    thd = abs(sqrt(np.sum(abs(thd_mat[:,1::]**2),1))/abs(thd_mat[:,0])*100)
    
    if opt_plot == 1:
        import matplotlib.pyplot as plt
        np.seterr(divide = 'ignore') 
        plt.figure()
        plt.subplot(211)
        plt.semilogx(freq_ir, thd)
        plt.ylim([0, 25])
        plt.xlim([5, fs/2-1])
        plt.title(' THD' )
        plt.xlabel('Frequency')
        plt.ylabel('THD %')
    
        plt.subplot(212)
        label1 = list( map(lambda _: ('h = {}').format(_),arange(n_harms+1))) 
        plt.semilogx(freq_ir,20*np.log10(abs(thd_mat)*2), '--', label=label1)
        plt.title(' Scaled harmonics fft' )
        plt.xlabel('Frequency')
        plt.ylabel('dB')
        plt.legend()
        plt.xlim([5, fs/2-1])
        plt.show()
        plt.pause(0.05)

    return thd, thd_mat
