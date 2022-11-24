# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 10:22:43 2022

@author: samuel
"""
import numpy as np
import sounddevice as sd
from time import sleep
from pathlib import Path
from scipy.io import wavfile
import matplotlib.pyplot as plt

# from pymeasfrf import signal_generation as sgen
from pymeasfrf.signal_generation import gen_burst
from pymeasfrf import signal_processing as sprs


def select_sound_card(device_name, max_inputs=None, max_outputs=None, fs=48000):
    """ select_sound_card(device_name, max_inputs, max_outputs, fs = 48000)
    
     Author1: Samuel Dupont
     
    Description: define the sound card parameters when already known
    
    :param: device_name,
            max_inputs,
            max_output,
            fs, default = 48000
 
    :return: 
    """
    sd.default.samplerate = fs
    sd.default.device = device_name

    if (max_inputs is None) or (max_outputs is None):
        device_in = sd.query_devices()[device_name[0]]
        device_out = sd.query_devices()[device_name[1]]
        max_inputs = device_in['max_input_channels']
        max_outputs = device_out['max_output_channels']
        sd.default.channels = [max_inputs, max_outputs]
    else:
        sd.default.channels = [max_inputs, max_outputs]


def test_one_channel(out_ch, sig2play, FS, gain=0.09):
    """ test_one_channel(out_ch,sig2play,FS)
    
     Author1: Marina Sanalatii
     Author2: Samuel Dupont
     
    Description: Play a given signal on one of the out channel for test
    
    :param: out_ch, selected output channel
            sig2play, signal to play 
            FS, sampling frequency
            gain = 0.09
 
    :return: (signal_iN_SPK) all the microphones signals for played channel
 
    """
    N_SPK = sd.default.channels[1]
    ns = sig2play.size
    signal_out = np.zeros((ns, N_SPK))
    signal_out[::, out_ch] = sig2play * gain
    print('\n Loudspeaker n°' + str(out_ch + 1) + '/' + str(N_SPK))
    signal_in = sd.play(signal_out, samplerate=FS)

    return signal_in


def record_1_mic(channel2record, record_time, fs, path2save_record='./record', name_record='/record_mic_', opt_plot=1,
                 opt_save=1):
    """ absolute_calibration_1mic(mic2calibrate, record_time, fs, path2save_record, opt_plot=1)
     Author1: Samuel Dupont
     
    Description: Record the signal on the selected microphone (channel2record) 
                 for a given nuber of second (record_time) and save in the defined 
                 folder path (path2save_record).
        
    
    :param: 
        channel2record,
        record_time, number of second of record
        fs,
        path2save_record,
        opt_plot=1
        opt_save = 1
    :return: 
    
    """

    # record calibrator
    signal_in = sd.rec(int(record_time * fs), samplerate=fs, channels=sd.default.channels[0])
    sleep(1.2 * record_time)

    # save meas in a folder
    if opt_save == 1:
        # create destination folder if inexistant
        Path(path2save_record).mkdir(parents=True, exist_ok=True)

        filename_rec = path2save_record + name_record + str(channel2record) + '.wav'
        wavfile.write(filename_rec, int(fs), (signal_in[::, channel2record]))

    # plot
    if opt_plot == 1:
        timevec = np.linspace(0, (signal_in.shape[0] - 1) / fs, signal_in.shape[0])
        plt.figure()
        plt.plot(timevec, signal_in[:, channel2record])
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
    return signal_in[:, channel2record]


# % REcord FRF
def record_frf_1spk_2_1mic(mic2record, spk2record, sig2play, gain, FS, n_avg=1, path2save_record='./record/',
                           opt_plot=1, opt_save=1, filename_recTF=[]):
    """ record_frf_1spk_2_1mic(mic2record,spk2record, sig2play, gain, FS, path2save_record = './record/', opt_plot = 1 )
    Author1: Marina Sanalatii
    Author2: Samuel Dupont
     
    Description: Record the signal on 1 microphone (mic2record) given a signal to be played 
                (sig2play) on channel (spk2record).
        
    
    :param: 
        mic_array_pos,
        sig2play,
        path2save_record,
    
    :return: (signal_iN_SPK) all the microphones signal for the last channel output
    
    """

    if type(mic2record) == int:
        mic2record = [mic2record]

    if type(spk2record) == int:
        spk2record = [spk2record]

    signal_in = record_frf_1spk_2_allmics(spk2record=spk2record, sig2play=sig2play, gain=gain, n_avg=n_avg, FS=FS,
                                          opt_plot=opt_plot, opt_save=0)
    signal_in = signal_in[::, mic2record]

    # save meas in a folder
    if opt_save == 1:
        # create destination folder if inexistant
        Path(path2save_record).mkdir(parents=True, exist_ok=True)

        if len(filename_recTF) == 0:
            if len(spk2record) > 1:
                spk2record = '_'.join(map(str, spk2record))
            if len(mic2record) > 1:
                mic2record = '_'.join(map(str, mic2record))

            if n_avg > 1:
                for i_avg in range(n_avg):
                    filename_recTF = '/LS' + str(spk2record) + 'Mic' + str(mic2record) + '_avg%iover%i' % (
                    i_avg + 1, n_avg) + '.wav'
                    wavfile.write(path2save_record + filename_recTF, int(FS),
                                  (signal_in[:, :, i_avg]).astype(np.float32))
            else:
                filename_recTF = '/LS' + str(spk2record) + 'Mic' + str(mic2record) + '.wav'
                wavfile.write(path2save_record + filename_recTF, int(FS), (signal_in).astype(np.float32))

    return signal_in.squeeze()


# save all available microphones
def record_frf_1spk_2_allmics(spk2record, sig2play, gain, FS, n_avg=1, path2save_record='./record', opt_plot=1,
                              opt_save=1):
    """ record_frf_1spk_2_1mic(spk2record, sig2play, gain, FS, path2save_record = './record', opt_plot = 1 )
    
    Author1: Marina Sanalatii
    Author2: Samuel Dupont
     
    Description: Record the signal on all available microphones (mic2record) given a signal to be played 
                (sig2play) on channel (spk2record).
        
    
    :param: 
        mic_array_pos,
        sig2play,
        path2save_record,
    
    :return: (signal_iN_SPK) all the microphones signal for the last channel output
    
    """
    if type(spk2record) == int:
        spk2record = [spk2record]

    # config
    M_MIC = sd.default.channels[0]
    npts_sig = sig2play.size
    T = npts_sig / FS

    # def signal outpout
    signal_out = np.zeros((npts_sig, sd.default.channels[1]))
    signal_out[::, spk2record] = np.array([sig2play] * len(spk2record)).T * gain  # reshape to spk2record numbers
    # signal_out[::, spk2record[0]] = np.tile(np.array([sig2play]*len(spk2record)).T *gain,(1,len(spk2record[0])))  # reshape to spk2record numbers
    print('\nLoudspeaker n°' + str(np.array(spk2record) + 1) + '/' + str(sd.default.channels[1]))

    if n_avg == 1:
        signal_in = sd.playrec(signal_out, samplerate=FS)
        sleep(1.15 * T)

    elif n_avg > 1:
        signal_in = np.zeros((npts_sig, sd.default.channels[0], n_avg))
        for i_avg in range(n_avg):
            meas = sd.playrec(signal_out, samplerate=FS)
            sleep(1.15 * T)
            # intermediate variable else, initialisation problems with empty first meas
            signal_in[:, :, i_avg] = meas

    else:
        raise ValueError('Number of averages must be > 0')

    # save meas in a folder
    if opt_save == 1:
        # create destination folder if inexistant
        Path(path2save_record).mkdir(parents=True, exist_ok=True)

        for m in range(M_MIC):  # write the measurements to the wav file
            if len(spk2record) > 1:
                spk2record = '_'.join(map(str, spk2record))
            filename_recTF = path2save_record + '/LS' + str(spk2record) + 'Mic' + str(m) + '.wav'

            wavfile.write(filename_recTF, int(FS), (signal_in[::, m]).astype(np.float32))

    if opt_plot == 1:
        timevec = np.linspace(0, (signal_in.shape[0] - 1) / FS, signal_in.shape[0])
        plt.figure()
        plt.plot(timevec, signal_in.reshape(signal_in.shape[0], -1))

    return signal_in


# save all available mics vs all speakers
def record_tf_allspeakers_2_allmics(sig2play, gain, FS, path2save_record, n_avg=1, mic_array_pos=0, opt_plot=1,
                                    opt_save=1):
    """ record_tf_allmics_of_allspeakers(mic_array_pos,sig2play,FS,path2save_record, gain ):
        
     Author1: Marina Sanalatii
     Author2: Samuel Dupont
     
    Description: Record the signal on all microphones considering a possible 
                displacement of an array (mic_array_pos, default = 0), given a signal to be played 
                (sig2play) on each channel, one after another.
        
    
    :param: 
        mic_array_pos,
        sig2play,
        path2save_record,
    
    :return: (signal_iN_SPK) all the microphones signal for the last channel output
    
    """

    # create destination folder if inexistant
    Path(path2save_record).mkdir(parents=True, exist_ok=True)

    # config
    M_MIC = sd.default.channels[0]
    N_SPK = sd.default.channels[1]

    for out_ch in range(N_SPK):
        signal_in = record_frf_1spk_2_allmics(spk2record=out_ch, sig2play=sig2play, gain=gain, n_avg=n_avg, FS=FS,
                                              path2save_record=path2save_record, opt_plot=opt_plot, opt_save=0)

        if opt_save == 1:
            if n_avg == 1:
                for m in range(M_MIC):  # write the measurements to the wav file
                    filename_recTF = path2save_record + '/LS' + str(out_ch) + 'Mic' + str(
                        m + mic_array_pos * M_MIC) + '.wav'
                    wavfile.write(filename_recTF, int(FS), (signal_in[::, m]).astype(np.float32))
            else:
                for m in range(M_MIC):  # write the measurements to the wav file
                    for i_avg in range(n_avg):
                        filename_recTF = path2save_record + '/LS' + str(out_ch) + 'Mic' + str(
                            m + mic_array_pos * M_MIC) + 'avg_%iover%i' % (i_avg, n_avg) + '.wav'
                        wavfile.write(filename_recTF, int(FS), (signal_in[::, m, i_avg]).astype(np.float32))
    return signal_in


# %%

def record_frf_1spk_2_1mic_latency_measurement_short_signal_input(mic2record, spk2record, latency_feedback_mic,
                                                                  latency_feedback_spk,
                                                                  sig2play, fs, gain_spk=0.05, n_avg=1,
                                                                  t_tot=0.5, t_burst=0.01, gain_burst=0.5,
                                                                  path2save_record='./record/', opt_plot=1, opt_save=1):
    def force2list(a):
        return a if isinstance(a, list) else [a]

    mic2record = force2list(mic2record)
    spk2record = force2list(spk2record)

    # do as normal but concatenate al signal avg instead of reapting the measurement ->  np.tile(sig2play, n_avg)         
    signal_mics = record_frf_1spk_2_1mic_latency_measurement(mic2record, spk2record, latency_feedback_mic,
                                                             latency_feedback_spk,
                                                             np.tile(sig2play, n_avg), fs, gain_spk=gain_spk, n_avg=1,
                                                             t_tot=t_tot, t_burst=t_burst, gain_burst=gain_burst,
                                                             path2save_record=path2save_record, opt_plot=opt_plot,
                                                             opt_save=0)

    # force reshape to [n_pts*navg, n_mics] due to squeeze in previous function (...)->(...,1) if needed
    signal_mics = signal_mics.reshape(signal_mics.shape[0], len(mic2record))

    # % split to expected
    # [n_mics x n_pts,n_avg] -> [navg, n_pts, n_mics ] -> [n_pts, n_mics, navg]
    signal_mics = signal_mics.reshape(n_avg, sig2play.shape[0], len(mic2record)).transpose((1, 2, 0))

    # timevec = np.linspace(0,(signal_mics.shape[0]-1)/fs,signal_mics.shape[0])
    # plt.figure()
    # plt.plot(timevec,signal_mics[:,0].reshape(signal_mics.shape[0],-1))

    # save meas in a folder
    if opt_save == 1:
        # create destination folder if inexistant
        Path(path2save_record).mkdir(parents=True, exist_ok=True)

        if len(spk2record) > 1:
            spk2record = '_'.join(map(str, spk2record))
        if len(mic2record) > 1:
            mic2record = '_'.join(map(str, mic2record))

        if n_avg > 1:
            for i_avg in range(n_avg):
                filename_recTF = '/LS' + str(spk2record) + 'Mic' + str(mic2record) + '_avg%iover%i' % (
                i_avg + 1, n_avg) + '.wav'
                wavfile.write(path2save_record + filename_recTF, int(fs), (signal_mics[:, :, i_avg]).astype(np.float32))
        else:
            filename_recTF = '/LS' + str(spk2record) + 'Mic' + str(mic2record) + '.wav'
            wavfile.write(path2save_record + filename_recTF, int(fs), (signal_mics).astype(np.float32))

    return signal_mics.squeeze()


# %%


def record_frf_1spk_2_1mic_latency_measurement(mic2record, spk2record, latency_feedback_mic, latency_feedback_spk,
                                               sig2play, fs, gain_spk=0.05, n_avg=1,
                                               t_tot=0.5, t_burst=0.01, gain_burst=0.5,
                                               path2save_record='./record/', opt_plot=1, opt_save=1):
    def force2list(a):
        return a if isinstance(a, list) else [a]

    mic2record = force2list(mic2record)
    spk2record = force2list(spk2record)

    mic2record_tot = force2list(latency_feedback_mic) + mic2record
    spk2record_tot = force2list(latency_feedback_spk) + spk2record

    burst, t_tot, t_burst = gen_burst(fs, t_tot=t_tot, t_burst=t_burst, gain=gain_burst)  # gen burst
    input_sig = np.concatenate((burst, sig2play))  # add it to the signal
    indx_burst = int(t_burst * fs)
    indx_tot = burst.size  # should be int(t_tot*fs)

    # record
    signal_mics = record_frf_1spk_2_allmics(spk2record_tot, input_sig, gain=gain_spk, n_avg=n_avg, FS=fs,
                                            path2save_record='./record1/', opt_plot=opt_plot, opt_save=0)

    # estimate the latency from the burst signal
    if n_avg == 1:
        out_burst = signal_mics[0:indx_tot, latency_feedback_mic,
                    np.newaxis]  # must be in the burst + null beginning part
        signal_mics = signal_mics[..., np.newaxis]
    else:
        out_burst = signal_mics[0:indx_tot, latency_feedback_mic, :]  # must be in the burst + null beginning part

    pos = np.argmax(out_burst, axis=0)  # get position of the burst in the recorded signal (from the source)
    latency = pos - indx_burst  # latency is the max position minus the specified pos of the burst

    # keep desired mic only
    signal_mics = signal_mics[:, mic2record, ...]

    ## shift true zero at 0
    for i_avg in range(n_avg):
        signal_mics[:, :, i_avg] = sprs.roll_0_pad(signal_mics[:, :, i_avg], -latency[i_avg])

        # remove burst
    signal_mics = signal_mics[indx_tot::, ...]

    # %%%

    # save meas in a folder
    if opt_save == 1:
        # create destination folder if inexistant
        from pathlib import Path

        Path(path2save_record).mkdir(parents=True, exist_ok=True)

        if len(spk2record) > 1:
            spk2record = '_'.join(map(str, spk2record))
        if len(mic2record) > 1:
            mic2record = '_'.join(map(str, mic2record))

        if n_avg > 1:
            for i_avg in range(n_avg):
                filename_recTF = '/LS' + str(spk2record) + 'Mic' + str(mic2record) + '_avg%iover%i' % (
                i_avg + 1, n_avg) + '.wav'
                wavfile.write(path2save_record + filename_recTF, int(fs), (signal_mics[:, :, i_avg]).astype(np.float32))
        else:
            filename_recTF = '/LS' + str(spk2record) + 'Mic' + str(mic2record) + '.wav'
            wavfile.write(path2save_record + filename_recTF, int(fs), (signal_mics).astype(np.float32))

    return signal_mics.squeeze()


def record_frf_multioutput_allmics(sig2play, gain, fs, n_avg=1):
    # create destination folder if inexistant

    # config
    M_MIC = sd.default.channels[0]
    npts_sig = sig2play.shape[0]
    T = npts_sig / fs

    if n_avg == 1:
        signal_in = sd.playrec(gain * sig2play, samplerate=fs)
        sleep(1.15 * T)

    elif n_avg > 1:
        signal_in = np.zeros((npts_sig, sd.default.channels[0], n_avg))
        for i_avg in range(n_avg):
            meas = sd.playrec(gain * sig2play, samplerate=fs)
            sleep(1.15 * T)
            # intermediate variable else, initialisation problems with empty first meas
            signal_in[:, :, i_avg] = meas

    else:
        raise ValueError('Number of averages must be > 0')

    return signal_in


def record_frf_multioutput_allmics_latency_measurement(mic2record, latency_feedback_mic, latency_feedback_spk,
                                                       sig2play, fs, gain_spk=0.05, n_avg=1,
                                                       t_tot=0.5, t_burst=0.01, gain_burst=0.5,
                                                       ):
    def force2list(a):
        return a if isinstance(a, list) else [a]

    mic2record = force2list(mic2record)

    n_origin = sig2play.shape[0]
    sig2play = np.tile(sig2play, (n_avg, 1))

    burst, t_tot, t_burst = gen_burst(fs, t_tot=t_tot, t_burst=t_burst, gain=gain_burst)  # gen burst
    shape2add = np.array(sig2play.shape);
    shape2add[0] = 1

    reshape2add = np.ones(len(sig2play.shape), dtype=int);
    reshape2add[0] = burst.shape[0]
    input_sig = np.concatenate((np.tile(burst.reshape(reshape2add), shape2add), sig2play))  # add it to the signal
    indx_burst = int(t_burst * fs)
    indx_tot = burst.size  # should be int(t_tot*fs)

    # record
    signal_mics = record_frf_multioutput_allmics(input_sig, gain=gain_spk, n_avg=1, fs=fs)

    # estimate the latency from the burst signal
    out_burst = signal_mics[0:indx_tot, latency_feedback_mic, np.newaxis]  # must be in the burst + null beginning part
    signal_mics = signal_mics[..., np.newaxis]

    pos = np.argmax(out_burst, axis=0)  # get position of the burst in the recorded signal (from the source)
    latency = pos - indx_burst  # latency is the max position minus the specified pos of the burst

    # keep desired mic only
    signal_mics = signal_mics[:, mic2record, ...]

    ## shift true zero at 0
    signal_mics[:, :, :] = sprs.roll_0_pad(signal_mics[:, :, :], -latency[0])

    # remove burst 
    signal_mics = signal_mics[indx_tot::, ...]

    # force reshape to [n_pts*navg, n_mics] due to squeeze in previous function (...)->(...,1) if needed
    signal_mics = signal_mics.reshape(signal_mics.shape[0], len(mic2record))

    # % split to expected
    # [n_mics x n_pts,n_avg] -> [navg, n_pts, n_mics ] -> [n_pts, n_mics, navg]
    signal_mics = signal_mics.reshape(n_avg, n_origin, len(mic2record)).transpose((1, 2, 0))

    # if 1 :
    #     timevec = np.linspace(0,(signal_mics.shape[0]-1)/fs,signal_mics.shape[0])
    #     plt.figure()
    #     plt.plot(timevec,signal_mics[:,0,:].reshape(signal_mics.shape[0],-1))

    return signal_mics.squeeze()
