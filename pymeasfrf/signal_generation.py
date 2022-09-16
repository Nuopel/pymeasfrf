# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 10:06:13 2022

@author: samuel
"""

import numpy as np
import matplotlib.pyplot as plt


from scipy.signal import butter, sosfreqz, tukey

from numpy import pi as pi
from numpy import sin as sin
import pickle as pickle


class SignalGen:
    def __init__(self, **kwargs):
        self.fs = []
        self.te = []
        self.fe = []
        self.t_tot = []
        self.channel = []
 
        self.n_pts = []
        self.n_avg = 1
        self.time = []
        self.x = []
        self.name = []
        self.window_t = []
        
        self.n_fft = []
        self.f = []
        self.x_fft = []
        
        self.sos = []
        self.w_bp_sos = []
        self.h_bp_sos = []
        self.__dict__.update(kwargs)
        
    def gen_time(self):
        self.n_pts = int(self.t_tot * self.fs)
        self.t_tot = self.n_pts / self.fs
        self.time = gen_time_npts(self.n_pts,self.fs)
        return self

    def gen_time_npts(self):
        self.t_tot = self.n_pts / self.fs
        self.time = gen_time_npts(self.n_pts,self.fs)
        return self
        
 
    def gen_window(self,alpha):
        self.window_t = tukey(self.n_pts, alpha)
        return self
    
    @property
    def add_window(self):
        self.x = self.window_t * self.x
        return self
    
    def add_zeros(self,time_of_zeros):
        n_0 = int(time_of_zeros *self.fs)
        self.n_pts = self.n_pts + n_0
        self.t_tot =  self.n_pts / self.fs
        self.x = np.concatenate([ self.x, np.zeros( n_0) ])
        self.gen_time_npts()  
        return self
    
    

    @property
    def gen_f_vec(self):
        # self.f = np.linspace(0,(self.n_fft-1)/self.n_fft*self.fs,self.n_fft)
        self.f = np.arange(0,self.n_fft)*self.fs/self.n_fft
        return self

    def butter_bandpass_sos(self, lowcut, highcut, order=5, worN=2000):
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        self.sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        self.w_bp_sos, self.h_bp_sos = sosfreqz(self.sos, worN=worN)
        return self.sos 

    def plot_time_sig_undersample(self, undersample = 1):
        if undersample !=0 :
            pas = np.linspace(0,self.n_pts-1, int(self.n_pts/undersample)).astype(int)
            plt.plot(self.time[pas], self.x[pas], label=self.name)
        else :
            plt.plot(self.time, self.x, label=self.name)
            
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        return self
    
    def plot_time_sig(self, undersample = 1):
        if undersample !=0 :
            pas = np.linspace(0,self.n_pts-1, int(self.n_pts/undersample)).astype(int)
            plt.plot(self.time[pas], self.x[pas], label=self.name)
        else :
            plt.plot(self.time, self.x, label=self.name)
            
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        return self

    @property 
    def plot_time_windows(self):
        time = np.arange(0,self.window_t.size)/self.fs
        plt.plot(time, self.window_t, label = 'tukey windows')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
    
    @property
    def plot_fft(self):
        plt.semilogx(self.f, 20*np.log10(abs(self.x_fft)/self.n_fft), label = self.name)
        plt.title(' Fft')
        plt.xlabel('Frequency')
        plt.ylabel('dB')
        plt.xlim([self.f[0], self.fs/2])
        plt.legend()
        
    @property
    def plot_fft_sos_bp(self):
        plt.semilogx( abs((self.fs * 0.5 / np.pi) * self.w_bp_sos), 20*np.log10(abs(self.h_bp_sos)), label="butter window BP")
        plt.title(' Fft')
        plt.xlabel('Frequency')
        plt.ylabel('dB')
        plt.xlim([abs(self.f[0]), abs(self.fs/2)])
        plt.legend()
        
    def save_self(self,filename = []):
        if len(filename)==0:
            filename = '_conf_'+self.name
        save_object(self, filename)
        
class ChirpSignal(SignalGen):
    def __init__(self, **kwargs):
        self.R = []
        self.f_end = []
        self.f_0 = []
        self.__dict__.update(kwargs)

        
    def gen_chirp(self, method='logarithmic'):
        self.x, self.R = gen_chirp( self.f_0,self.f_end,self.time,self.t_tot)
        self.name = '%s_chirp'%(method)
        return self


class RandomSignal(SignalGen):
    def __init__(self, **kwargs):
        self.name = 'normal_random'
        self.__dict__.update(kwargs)

    def gen_random_normal(self):
        self.x = np.random.normal( 0, 1, self.n_pts )
        return self
     
    def gen_random_uniform(self):
        self.x =  np.random.uniform(low=-1, high=1, size=(self.n_pts,))
        self.name = 'uniform_random'
        return self
   
class SinusSignal(SignalGen):
    def __init__(self, **kwargs):
        SignalGen.__init__(self, kwargs)
        
    def gen_sin(self,f):
        self.x =  np.sin(2*np.pi*f*self.time)
        self.name = 'sinus %i Hz'%(int(f))

        return self

class Multitone(SignalGen):
    def __init__(self, **kwargs):
        self.name = 'multitone'
        self.n_freq = []
        self.f_multitone = []
        self.__dict__.update(kwargs)
        
    def gen_multitone(self):
        self.x, self.f_multitone = gen_multitone(self.time, self.f_0, self.f_end, self.n_freq)


    
def gen_chirp( f_0,f_end,time,t_tot, method ='logarithmic'):
    R = t_tot/np.log(f_end/f_0)
    x = sin((2*pi*f_0*R)*(np.exp(time/R)-1)) # phase 0 for 0
    # x = sin(2*pi*f_0*R*np.exp(time/R)) # Formulation novak 
    # x = chirp(time, f0=f_0, f1=f_end, t1=t_tot, method = method)

    return x,R


def gen_multitone(t, f_0, f_end, n_freq):
    sig = np.zeros(t.shape)
    frequencies = np.linspace(f_0, f_end, n_freq)
    for n in range(int(n_freq)):
        sig = sig + sin(2*pi*frequencies[n]*t + np.random.rand()*2*pi)    
    sig = sig/max(abs(sig))
    return sig, frequencies


#%% Save data as pikcle
def save_object(obj, filename):
    """ save_object(obj, filename)
    
    Method which save the object in a file 
    
    :filename: name and location of the file
    
    use : obj = load_object(filename) to load back

    """    
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    """ load_object(obj, filename)
    
    Method which load the object from a pickle file 
    
    :filename: name and location of the file
    

    """ 
    with open(filename, 'rb') as pickle_file:
        obj = pickle.load(pickle_file)
    return obj


#%% generate vector or signals
def gen_time(t_tot,fs):
    time = np.arange(0,int(t_tot*fs))/fs
    return time

def gen_time_npts(n_pts,fs):
    time = np.arange(0,n_pts)/fs
    return time


def gen_burst(fs, t_tot = 1, t_burst = 0.01, f_0 = 1e3, gain = 0.5  ):
    # t_tot = 1                 # time of the burst
    # f_0 = 1e3             # burst main frequency
    # t_burst = 0.01     # burst signal signal position
    time  = gen_time(t_tot,fs)  
    burst = gain * np.real(np.exp(-(1000*(time-t_burst))**2 + 1j*2*np.pi*f_0*time))
    
    return burst, t_tot, t_burst

def duplicate_signal(signal,n_avg):
       return np.tile(signal, n_avg) 