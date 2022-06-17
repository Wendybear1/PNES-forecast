from __future__ import division
import mne
import numpy as np
import scipy.signal
from scipy.signal import butter, lfilter
from matplotlib import pyplot
import math
from scipy.fftpack import fft, ifft
from scipy import signal
from scipy.signal import butter, lfilter, iirfilter, filtfilt
from scipy.signal import hilbert
from biosppy.signals import tools
import pandas as pd
from matplotlib import rc


def butter_bandpass(lowcut, highcut, fs, order=7):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


# def butter_bandpass_filter(data, lowcut, highcut, fs, order=7):
#     b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#     y = lfilter(b, a, data)
#     return y

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def Implement_Notch_Filter(fs, band, freq, ripple, order, filter_type, data):
    nyq = fs / 2.0
    low = freq - band / 2.0
    high = freq + band / 2.0
    low = low / nyq
    high = high / nyq
    b, a = iirfilter(order, [low, high], rp=ripple, btype='bandstop', analog=False, ftype=filter_type)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def split(arr, size):
    arrs = []
    while len(arr) > size:
        pice = arr[:size]
        arrs.append(pice)
        arr = arr[int(size):]
    arrs.append(arr)
    return arrs


def movingaverage(values, window_size):
    weights = (np.ones(window_size)) / window_size
    a = np.ones(1)
    return lfilter(weights, a, values)


csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC1012/Cz_EEGvariance_VIC1012_15s_3h.csv', sep=',', header=None)
Raw_variance_EEG = csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC1012/Cz_EEGauto_VIC1012_15s_3h.csv', sep=',', header=None)
Raw_auto_EEG = csv_reader.values

Raw_variance_EEG_arr = []
for item in Raw_variance_EEG:
    Raw_variance_EEG_arr.append(float(item))
Raw_auto_EEG_arr = []
for item in Raw_auto_EEG:
    Raw_auto_EEG_arr.append(float(item))
t = np.linspace(0 + 0.00416667, 0 + 0.00416667 + 0.00416667 * (len(Raw_variance_EEG_arr) - 1),
                len(Raw_variance_EEG_arr))
t_window_arr = t

print(len(t_window_arr));print(t_window_arr[0]);
print(t_window_arr[-1] - t_window_arr[0]);
print(t_window_arr[18000]);
print(t_window_arr[18000] - t_window_arr[0]);
print(t_window_arr[-1] - t_window_arr[18000]);

window_time_arr = t_window_arr
# pyplot.plot(window_time_arr,Raw_variance_EEG_arr,'grey',alpha=0.5)
# pyplot.ylabel('Voltage',fontsize=13)
# pyplot.xlabel('Time (hours)',fontsize=13)
# pyplot.title('raw EEG variance in VIC1012',fontsize=13)
# pyplot.show()
var_arr = []
for item in Raw_variance_EEG_arr:
    if item < 1e-8:
        var_arr.append(item)
    else:
        var_arr.append(var_arr[-1])
Raw_variance_EEG = var_arr
# pyplot.plot(window_time_arr,Raw_variance_EEG,'grey',alpha=0.5)
# pyplot.ylabel('Voltage',fontsize=13)
# pyplot.xlabel('Time (hours)',fontsize=13)
# pyplot.title('EEG variance in VIC1012',fontsize=13)
# pyplot.show()


seizure_timing_index = []
for k in range(len(window_time_arr)):
    if window_time_arr[k] < 17.30305 and window_time_arr[k + 1] >= 17.30305:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 43.636667 and window_time_arr[k + 1] >= 43.636667:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 46.30361 and window_time_arr[k + 1] >= 46.30361:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 49.76528 and window_time_arr[k + 1] >= 49.76528:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 65.136667 and window_time_arr[k + 1] >= 65.136667:
        seizure_timing_index.append(k)
    # if window_time_arr[k] < 120.803 and window_time_arr[k + 1] >= 120.803:
    #     seizure_timing_index.append(k)
print(seizure_timing_index)


# # # # # # ### EEG variance
window_time_arr=t_window_arr[0:18000]
Raw_variance_EEG=Raw_variance_EEG[0:18000]
# window_time_arr=t_window_arr
# Raw_variance_EEG=Raw_variance_EEG

long_rhythm_var_arr=movingaverage(Raw_variance_EEG,5760)
medium_rhythm_var_arr=movingaverage(Raw_variance_EEG,240)
medium_rhythm_var_arr_2=movingaverage(Raw_variance_EEG,240*3)
medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG,240*6)
medium_rhythm_var_arr_4=movingaverage(Raw_variance_EEG,240*12)
short_rhythm_var_arr_plot=movingaverage(Raw_variance_EEG,20)


long_rhythm_var_arr=short_rhythm_var_arr_plot*(10**12)
var_trans=hilbert(long_rhythm_var_arr)
var_trans_nomal=[]
for m in var_trans:
    var_trans_nomal.append(m/abs(m))
SIvarlong=sum(var_trans_nomal)/len(var_trans_nomal)
print(SIvarlong)
seizure_phase=[]
for item in seizure_timing_index:
     seizure_phase.append(var_trans_nomal[item])
SIvarlongseizure=sum(seizure_phase)/len(seizure_phase)
print(SIvarlongseizure)
var_phase=np.angle(var_trans)
phase_long_EEGvariance_arr=var_phase
seizure_phase_var_long=[]
for item in seizure_timing_index:
    seizure_phase_var_long.append(phase_long_EEGvariance_arr[item])
print(seizure_phase_var_long)


from matplotlib import gridspec
fig = pyplot.figure(figsize=(12, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 3])
ax1=pyplot.subplot(gs[0])
ax1.plot(window_time_arr[240*6:],long_rhythm_var_arr[240*6:],'darkblue',alpha=0.8)
ax1.set_title('EEG variance in VIC1012',fontsize=23)
ax1.set_xlabel('Time (hours)',fontsize=23)
ax1.set_ylabel('$\mathregular{\u03BCV^2}$',fontsize=23)
locs, labels = pyplot.xticks(fontsize=23)
locs, labels = pyplot.yticks(fontsize=23)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.scatter(window_time_arr[4151],long_rhythm_var_arr[4151],s=60,c='g')
ax1.scatter(window_time_arr[10471],long_rhythm_var_arr[10471],s=60,c='g')
ax1.scatter(window_time_arr[11111],long_rhythm_var_arr[11111],s=60,c='g')
ax1.scatter(window_time_arr[11942],long_rhythm_var_arr[11942],s=60,c='r')
ax1.scatter(window_time_arr[15631],long_rhythm_var_arr[15631],s=60,c='g')
# ax1.scatter(window_time_arr[28991],long_rhythm_var_arr[28991],s=60,c='g')
# pyplot.xlim(window_time_arr[0],window_time_arr[-1])
ax2=pyplot.subplot(gs[1])
ax2.set_xlabel('Time (hours)',fontsize=23)
ax2.set_title('Instantaneous Phase',fontsize=23)
# pyplot.xlim(window_time_arr[0],window_time_arr[-1])
ax2.plot(window_time_arr[240*6:],phase_long_EEGvariance_arr[240*6:],c='k',alpha=0.7,label='instantaneous phase')
pyplot.hlines(0,window_time_arr[240*6],window_time_arr[-1],'k','dashed')
ax2.set_xlabel('Time (hours)',fontsize=23)
locs, labels = pyplot.xticks(fontsize=23)
locs, labels = pyplot.yticks(fontsize=23)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.scatter(window_time_arr[4151],phase_long_EEGvariance_arr[4151],s=60,c='g')
ax2.scatter(window_time_arr[10471],phase_long_EEGvariance_arr[10471],s=60,c='g')
ax2.scatter(window_time_arr[11111],phase_long_EEGvariance_arr[11111],s=60,c='g')
ax2.scatter(window_time_arr[11942],phase_long_EEGvariance_arr[11942],s=60,c='r')
ax2.scatter(window_time_arr[15631],phase_long_EEGvariance_arr[15631],s=60,c='g')
# ax2.scatter(window_time_arr[28991],phase_long_EEGvariance_arr[28991],s=60,c='g')
# locs, labels = pyplot.yticks([-np.pi,-0.5*np.pi,0,0.5*np.pi,np.pi],['-$\pi$','Rising','0','Falling','$\pi$'],rotation='vertical',fontsize=23)
locs, labels = pyplot.yticks([-0.5*np.pi,-0.15*np.pi,0,0.3*np.pi,0.5*np.pi],['-0.5$\pi$','Rising','0','Falling','0.5$\pi$'],rotation='vertical',fontsize=23)
pyplot.tight_layout()
pyplot.show()

bins_number = 18
bins = np.linspace(-np.pi, np.pi, bins_number + 1)
nEEGsvar, _, _ = pyplot.hist(phase_long_EEGvariance_arr, bins)
nEEGsvarsei, _, _ = pyplot.hist(seizure_phase_var_long, bins)
print(nEEGsvar)
print(nEEGsvarsei)
width = 2*np.pi / bins_number
params = dict(projection='polar')
fig, ax2 = pyplot.subplots(subplot_kw=params)
ax2.bar(bins[:bins_number], nEEGsvarsei/sum(nEEGsvarsei),width=width, color='grey',alpha=0.7,linewidth=2,edgecolor='k')
locs, labels = pyplot.yticks([0.1,0.3,0.5],['0.1','0.3','0.5'],fontsize=16)
ax2.set_title('EEG variance',fontsize=16)
ax2.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',],fontsize=16)
pyplot.show()





# pyplot.plot(t_window_arr,Raw_auto_EEG_arr,'grey',alpha=0.5)
# pyplot.xlabel('Time (hours)',fontsize=13)
# pyplot.title('raw EEG autocorrelation in VIC1012',fontsize=13)
# pyplot.show()
value_arr=[]
for item in Raw_auto_EEG_arr:
    if item<500:
        value_arr.append(item)
    else:
        value_arr.append(value_arr[-1])
Raw_auto_EEG_arr=value_arr
# pyplot.plot(t_window_arr,Raw_auto_EEG_arr,'grey',alpha=0.5)
# pyplot.xlabel('Time (hours)',fontsize=13)
# pyplot.title('EEG autocorrelation in VIC1012',fontsize=13)
# pyplot.show()


Raw_auto_EEG=Raw_auto_EEG_arr[0:18000]
window_time_arr=t_window_arr[0:18000]
# Raw_auto_EEG=Raw_auto_EEG_arr
# window_time_arr=t_window_arr

long_rhythm_value_arr=movingaverage(Raw_auto_EEG,5760)
medium_rhythm_value_arr=movingaverage(Raw_auto_EEG,240)
medium_rhythm_value_arr_2=movingaverage(Raw_auto_EEG,240*3)
medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG,240*6)
medium_rhythm_value_arr_4=movingaverage(Raw_auto_EEG,240*12)
short_rhythm_value_arr_plot=movingaverage(Raw_auto_EEG,20)




long_rhythm_value_arr=short_rhythm_value_arr_plot
value_trans=hilbert(long_rhythm_value_arr)
value_trans_nomal=[]
for m in value_trans:
    value_trans_nomal.append(m/abs(m))
SIvaluelong=sum(value_trans_nomal)/len(value_trans_nomal)
print(SIvaluelong)
seizure_phase=[]
for item in seizure_timing_index:
    seizure_phase.append(value_trans_nomal[item])
SIvaluelongseizure=sum(seizure_phase)/len(seizure_phase)
print(SIvaluelongseizure)
value_phase=np.angle(value_trans)
phase_long_EEGauto_arr=value_phase
seizure_phase_value_long=[]
for item in seizure_timing_index:
    seizure_phase_value_long.append(phase_long_EEGauto_arr[item])
print(seizure_phase_value_long)
# item_arr=[]
# for item in seizure_phase_value_long:
#     item_arr.append(item/np.pi)
# print(item_arr)
from matplotlib import gridspec
fig = pyplot.figure(figsize=(12, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 3])
ax2=pyplot.subplot(gs[0])
ax2.plot(window_time_arr[240*6:],long_rhythm_value_arr[240*6:],'darkblue',alpha=0.8)
ax2.set_title('EEG autocorrelation in VIC1012',fontsize=23)
ax2.set_xlabel('Time (hours)',fontsize=23)
locs, labels = pyplot.xticks(fontsize=23)
locs, labels = pyplot.yticks(fontsize=23)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.scatter(window_time_arr[4151],long_rhythm_value_arr[4151],s=60,c='g')
ax2.scatter(window_time_arr[10471],long_rhythm_value_arr[10471],s=60,c='g')
ax2.scatter(window_time_arr[11111],long_rhythm_value_arr[11111],s=60,c='r')
ax2.scatter(window_time_arr[11942],long_rhythm_value_arr[11942],s=60,c='r')
ax2.scatter(window_time_arr[15631],long_rhythm_value_arr[15631],s=60,c='r')
# ax2.scatter(window_time_arr[28991],long_rhythm_value_arr[28991],s=60,c='g')
ax3=pyplot.subplot(gs[1])
ax3.set_xlabel('Time (hours)',fontsize=23)
ax3.set_title('Instantaneous Phase',fontsize=23)
# pyplot.xlim(window_time_arr[0],window_time_arr[-1])
locs, labels = pyplot.xticks(fontsize=23)
locs, labels = pyplot.yticks(fontsize=23)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.plot(window_time_arr[240*6:],phase_long_EEGauto_arr[240*6:],'k',alpha=0.7,label='instantaneous phase')
pyplot.hlines(0,window_time_arr[240*6],window_time_arr[-1],'k','dashed')
ax3.scatter(window_time_arr[4151],phase_long_EEGauto_arr[4151],s=60,c='g')
ax3.scatter(window_time_arr[10471],phase_long_EEGauto_arr[10471],s=60,c='g')
ax3.scatter(window_time_arr[11111],phase_long_EEGauto_arr[11111],s=60,c='r')
ax3.scatter(window_time_arr[11942],phase_long_EEGauto_arr[11942],s=60,c='r')
ax3.scatter(window_time_arr[15631],phase_long_EEGauto_arr[15631],s=60,c='r')
# ax3.scatter(window_time_arr[28991],phase_long_EEGauto_arr[28991],s=60,c='g')
ax3.set_xlabel('Time (hours)',fontsize=23)
locs, labels = pyplot.yticks([-0.5*np.pi,-0.15*np.pi,0,0.3*np.pi,0.5*np.pi],['-0.5$\pi$','Rising','0','Falling','0.5$\pi$'],rotation='vertical',fontsize=23)
# locs, labels = pyplot.yticks([-0.25*np.pi,-0.125*np.pi,0,0.125*np.pi,0.25*np.pi],['-0.25$\pi$','Rising','0','Falling','0.25$\pi$'],rotation='vertical',fontsize=23)
pyplot.tight_layout()
pyplot.show()

bins_number = 18
bins = np.linspace(-np.pi, np.pi, bins_number + 1)
nEEGsauto, _, _ = pyplot.hist(phase_long_EEGauto_arr, bins)
nEEGsautosei, _, _ = pyplot.hist(seizure_phase_value_long, bins)
print(nEEGsauto)
print(nEEGsautosei)
width = 2*np.pi / bins_number
params = dict(projection='polar')
fig, ax2 = pyplot.subplots(subplot_kw=params)
ax2.bar(bins[:bins_number], nEEGsautosei/sum(nEEGsautosei),width=width, color='grey',alpha=0.7,linewidth=2,edgecolor='k')
locs, labels = pyplot.yticks([0.1,0.3,0.5],['0.1','0.3','0.5'],fontsize=16)
ax2.set_title('EEG autocorrelation',fontsize=16)
ax2.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',],fontsize=16)
pyplot.show()









# # ### ECG data
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC1012/RRI_ch21_timewindowarr_VIC1012_15s_3h.csv', sep=',',
                         header=None)
rri_t = csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC1012/RRI_ch21_rawvariance_VIC1012_15s_3h.csv', sep=',',
                         header=None)
RRI_var = csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC1012/RRI_ch21_rawauto_VIC1012_15s_3h.csv', sep=',',
                         header=None)
Raw_auto_RRI31 = csv_reader.values

rri_t_arr = []
for item in rri_t:
    rri_t_arr.append(0 + float(item))

Raw_variance_RRI31_arr = []
for item in RRI_var:
    Raw_variance_RRI31_arr.append(float(item))
Raw_auto_RRI31_arr = []
for item in Raw_auto_RRI31:
    Raw_auto_RRI31_arr.append(float(item))
print(len(Raw_variance_RRI31_arr))

# pyplot.plot(Raw_variance_RRI31_arr,'grey',alpha=0.5)
# pyplot.xlabel('Time (hours)',fontsize=13)
# pyplot.title('RRI variance in VIC1012',fontsize=13)
# pyplot.show()
#
# pyplot.plot(Raw_auto_RRI31_arr,'grey',alpha=0.5)
# pyplot.xlabel('Time (hours)',fontsize=13)
# pyplot.title('RRI autocorrelation in VIC1012',fontsize=13)
# pyplot.show()



# window_time_arr=t_window_arr
# Raw_variance_RRI31=Raw_variance_RRI31_arr
window_time_arr = t_window_arr[0:18000]
Raw_variance_RRI31 = Raw_variance_RRI31_arr[0:18000]

long_rhythm_var_arr = movingaverage(Raw_variance_RRI31, 5760)
medium_rhythm_var_arr = movingaverage(Raw_variance_RRI31, 240)
medium_rhythm_var_arr_2 = movingaverage(Raw_variance_RRI31, 240 * 3)
medium_rhythm_var_arr_3 = movingaverage(Raw_variance_RRI31, 240 * 6)
medium_rhythm_var_arr_4 = movingaverage(Raw_variance_RRI31, 240 * 12)
short_rhythm_var_arr_plot = movingaverage(Raw_variance_RRI31, 20)



long_rhythm_var_arr = short_rhythm_var_arr_plot
var_trans = hilbert(long_rhythm_var_arr)
var_trans_nomal = []
for m in var_trans:
    var_trans_nomal.append(m / abs(m))
SIvarlong = sum(var_trans_nomal) / len(var_trans_nomal)
print(SIvarlong)
seizure_phase = []
for item in seizure_timing_index:
    seizure_phase.append(var_trans_nomal[item])
SIvarlongseizure = sum(seizure_phase) / len(seizure_phase)
print(SIvarlongseizure)
var_phase = np.angle(var_trans)
phase_whole_long = var_phase
seizure_phase_var_long = []
for item in seizure_timing_index:
    seizure_phase_var_long.append(phase_whole_long[item])
print(seizure_phase_var_long)

from matplotlib import gridspec
fig = pyplot.figure(figsize=(12, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 3])
ax1 = pyplot.subplot(gs[0])
ax1.plot(window_time_arr[240 * 6:], long_rhythm_var_arr[240 * 6:len(window_time_arr)], 'darkblue', alpha=0.8,
         label='channel 2')
# pyplot.xlim(window_time_arr[0],window_time_arr[-1])
ax1.set_title('RRI variance in VIC1012', fontsize=23)
ax1.set_xlabel('Time (hours)', fontsize=23)
ax1.set_ylabel('$\mathregular{S^2}$', fontsize=23)
locs, labels = pyplot.xticks(fontsize=23)
locs, labels = pyplot.yticks(fontsize=23)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.scatter(window_time_arr[4151], long_rhythm_var_arr[4151], c='g', s=60)
ax1.scatter(window_time_arr[10471], long_rhythm_var_arr[10471], c='g', s=60)
ax1.scatter(window_time_arr[11111], long_rhythm_var_arr[11111], c='g', s=60)
ax1.scatter(window_time_arr[11942], long_rhythm_var_arr[11942], c='r', s=60)
ax1.scatter(window_time_arr[15631], long_rhythm_var_arr[15631], c='g', s=60)
# ax1.scatter(window_time_arr[28991], long_rhythm_var_arr[28991], c='g', s=60)
# pyplot.xlim(window_time_arr[0],window_time_arr[33000])
ax2 = pyplot.subplot(gs[1])
ax2.set_xlabel('Time (hours)', fontsize=23)
ax2.set_title('Instantaneous Phase', fontsize=23)
# pyplot.xlim(window_time_arr[0],window_time_arr[-1])
ax2.plot(window_time_arr[240 * 6:], phase_whole_long[240 * 6:len(window_time_arr)], 'k', alpha=0.7)
ax2.scatter(window_time_arr[4151], phase_whole_long[4151], c='g', s=60)
ax2.scatter(window_time_arr[10471], phase_whole_long[10471], c='g', s=60)
ax2.scatter(window_time_arr[11111], phase_whole_long[11111], c='g', s=60)
ax2.scatter(window_time_arr[11942], phase_whole_long[11942], c='r', s=60)
ax2.scatter(window_time_arr[15631], phase_whole_long[15631], c='g', s=60)
# ax2.scatter(window_time_arr[28991], phase_whole_long[28991], c='g', s=60)
ax2.set_xlabel('Time (hours)', fontsize=23)
locs, labels = pyplot.xticks(fontsize=23)
locs, labels = pyplot.yticks(fontsize=23)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
pyplot.hlines(0, window_time_arr[240 * 6], window_time_arr[-1], 'k', 'dashed')
locs, labels = pyplot.yticks([-0.5 * np.pi, -0.15 * np.pi, 0, 0.3 * np.pi, 0.5 * np.pi],
                             ['-0.5$\pi$', 'Rising', '0', 'Falling', '0.5$\pi$'], rotation='vertical', fontsize=23)
pyplot.tight_layout()
pyplot.show()
bins_number = 18
bins = np.linspace(-np.pi, np.pi, bins_number + 1)
nRRIsvar, _, _ = pyplot.hist(phase_whole_long, bins)
nRRIsvarsei, _, _ = pyplot.hist(seizure_phase_var_long, bins)
print(nRRIsvar)
print(nRRIsvarsei)
params = dict(projection='polar')
fig, ax2 = pyplot.subplots(subplot_kw=params)
ax2.bar(bins[:bins_number], nRRIsvarsei/sum(nRRIsvarsei),width=width, color='grey',alpha=0.7,edgecolor='k',linewidth=2)
ax2.set_title('RRI variance',fontsize=16)
locs, labels = pyplot.yticks([0.1,0.3,0.5],['0.1','0.3','0.5'],fontsize=16)
# ax2.set_rlim([0,0.002])
ax2.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',],fontsize=16)
pyplot.show()





# Raw_auto_RRI31=Raw_auto_RRI31_arr
Raw_auto_RRI31 = Raw_auto_RRI31_arr[0:18000]

long_rhythm_value_arr = movingaverage(Raw_auto_RRI31, 5760)
medium_rhythm_value_arr = movingaverage(Raw_auto_RRI31, 240)
medium_rhythm_value_arr_2 = movingaverage(Raw_auto_RRI31, 240 * 3)
medium_rhythm_value_arr_3 = movingaverage(Raw_auto_RRI31, 240 * 6)
medium_rhythm_value_arr_4 = movingaverage(Raw_auto_RRI31, 240 * 12)
short_rhythm_value_arr_plot = movingaverage(Raw_auto_RRI31, 20)



long_rhythm_value_arr = short_rhythm_value_arr_plot
value_trans = hilbert(long_rhythm_value_arr)
value_trans_nomal = []
for m in value_trans:
    value_trans_nomal.append(m / abs(m))
SIvaluelong = sum(value_trans_nomal) / len(value_trans_nomal)
print(SIvaluelong)
seizure_phase = []
for item in seizure_timing_index:
    seizure_phase.append(value_trans_nomal[item])
SIvaluelongseizure = sum(seizure_phase) / len(seizure_phase)
print(SIvaluelongseizure)
value_phase = np.angle(value_trans)
phase_whole_value_long = value_phase
seizure_phase_value_long = []
for item in seizure_timing_index:
    seizure_phase_value_long.append(phase_whole_value_long[item])
print(seizure_phase_value_long)

from matplotlib import gridspec
fig = pyplot.figure(figsize=(12, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 3])
ax1 = pyplot.subplot(gs[0])
ax1.set_title('RRI autocorrelation in VIC1012', fontsize=23)
ax1.set_xlabel('Time (hours)', fontsize=23)
locs, labels = pyplot.xticks(fontsize=23)
locs, labels = pyplot.yticks(fontsize=23)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.plot(window_time_arr[240 * 6:], long_rhythm_value_arr[240 * 6:len(window_time_arr)], 'darkblue', alpha=0.8)
# pyplot.xlim(window_time_arr[0],window_time_arr[-1])
ax1.scatter(window_time_arr[4151], long_rhythm_value_arr[4151], c='g', s=60)
ax1.scatter(window_time_arr[10471], long_rhythm_value_arr[10471], c='g', s=60)
ax1.scatter(window_time_arr[11111], long_rhythm_value_arr[11111], c='g', s=60)
ax1.scatter(window_time_arr[11942], long_rhythm_value_arr[11942], c='r', s=60)
ax1.scatter(window_time_arr[15631], long_rhythm_value_arr[15631], c='g', s=60)
# ax1.scatter(window_time_arr[28991], long_rhythm_value_arr[28991], c='r', s=60)
ax2 = pyplot.subplot(gs[1])
ax2.set_title('Instantaneous Phase', fontsize=23)
ax2.set_xlabel('Time (hours)', fontsize=23)
locs, labels = pyplot.xticks(fontsize=23)
locs, labels = pyplot.yticks(fontsize=23)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
pyplot.hlines(0, window_time_arr[240 * 6], window_time_arr[-1], 'k', 'dashed')
# pyplot.xlim(window_time_arr[0],window_time_arr[-1])
ax2.plot(window_time_arr[240 * 6:], phase_whole_value_long[240 * 6:len(window_time_arr)], 'k', alpha=0.7)
ax2.scatter(window_time_arr[4151], phase_whole_value_long[4151], c='g', s=60)
ax2.scatter(window_time_arr[10471], phase_whole_value_long[10471], c='g', s=60)
ax2.scatter(window_time_arr[11111], phase_whole_value_long[11111], c='g', s=60)
ax2.scatter(window_time_arr[11942], phase_whole_value_long[11942], c='r', s=60)
ax2.scatter(window_time_arr[15631], phase_whole_value_long[15631], c='g', s=60)
# ax2.scatter(window_time_arr[28991], phase_whole_value_long[28991], c='r', s=60)
ax2.set_xlabel('Time (hours)', fontsize=23)
# locs, labels = pyplot.yticks([-np.pi,-0.5*np.pi,0,0.5*np.pi,np.pi],['-$\pi$','Rising','0','Falling','$\pi$'],rotation='vertical',fontsize=23)
# locs, labels = pyplot.yticks([-0.5*np.pi,-0.25*np.pi,0,0.25*np.pi,0.5*np.pi],['-0.5$\pi$','Rising','0','Falling','0.5$\pi$'],rotation='vertical',fontsize=23)
locs, labels = pyplot.yticks([-0.5 * np.pi, -0.15 * np.pi, 0, 0.3 * np.pi, 0.5 * np.pi],
                             ['-0.5$\pi$', 'Rising', '0', 'Falling', '0.5$\pi$'], rotation='vertical', fontsize=23)
# locs, labels = pyplot.yticks([-0.25*np.pi,-0.125*np.pi,0,0.125*np.pi,0.25*np.pi],['-0.25$\pi$','Rising','0','Falling','0.25$\pi$'],rotation='vertical',fontsize=23)
pyplot.tight_layout()
pyplot.show()
bins_number = 18
bins = np.linspace(-np.pi, np.pi, bins_number + 1)
nRRIsauto, _, _ = pyplot.hist(phase_whole_value_long, bins)
nRRIsautosei, _, _ = pyplot.hist(seizure_phase_value_long, bins)
print(nRRIsauto)
print(nRRIsautosei)
width = 2*np.pi / bins_number
params = dict(projection='polar')
fig, ax2 = pyplot.subplots(subplot_kw=params)
ax2.bar(bins[:bins_number], nRRIsautosei/sum(nRRIsautosei),width=width, color='grey',alpha=0.7,edgecolor='k',linewidth=2)
ax2.set_title('RRI autocorrelation',fontsize=16)
locs, labels = pyplot.yticks([0.1,0.3,0.5],['0.1','0.3','0.5'],fontsize=16)
ax2.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',],fontsize=16)
pyplot.show()






t=np.linspace(0+0.00416667,0+0.00416667+0.00416667*(len(Raw_variance_EEG_arr)-1),len(Raw_variance_EEG_arr))
window_time_arr=t
a=np.where(t<2.80305556+0)
print(a);print(t[671]);print(t[672])
t[0:672]=t[0:672]-0+21.1969444
t[672:]=t[672:]-2.80305556-0
print(t[671]);print(t[672]);print(t);print(type(t));

time_feature_arr=[]
for i in range(len(t)):
    if t[i]>24:
        time_feature_arr.append(t[i] - (t[i] // 24) * 24)
    else:
        time_feature_arr.append(t[i])
seizure_time=[time_feature_arr[4151],time_feature_arr[10471],time_feature_arr[11111],time_feature_arr[11942],time_feature_arr[15631],
]
print(seizure_time)
bins_number = 18
bins = np.linspace(0, 24, bins_number + 1)
nEEGsvar, _, _ = pyplot.hist(time_feature_arr[0:18000], bins)
nEEGsvarsei, _, _ = pyplot.hist(seizure_time, bins)


# bins = np.linspace(0, 2*np.pi, bins_number + 1)
# width = 2*np.pi / bins_number
# params = dict(projection='polar')
# fig, ax = pyplot.subplots(subplot_kw=params)
# ax.bar(bins[:bins_number], nEEGsvarsei/sum(nEEGsvarsei),width=width, color='grey',alpha=0.7,edgecolor='k',linewidth=2)
# pyplot.setp(ax.get_yticklabels(), color='k')
# # ax.set_title('seizure timing histogram (SA0124)',fontsize=23)
# ax.set_xticks(np.linspace(0, 2*np.pi, 24, endpoint=False))
# ax.set_xticklabels(['0 am','','','Night','','','6 am','','','Morning','','','12 am','','','Afternoon','','','18 pm','','','Evening','','','24 pm'],fontsize=16)
# # locs, labels = pyplot.xticks([0,3,6,9,12,15,18,21,24],['0','Night','6','Morning','12','Afternoon','18','Evening','24'],fontsize=16)
# locs, labels = pyplot.yticks([0.1,0.2,0.3],['0.1','0.2','0.3'],fontsize=16)
# pyplot.show()
bins = np.linspace(0, 2*np.pi, bins_number + 1)
width = 2*np.pi / bins_number
params = dict(projection='polar')
fig, ax = pyplot.subplots(subplot_kw=params)
ax.bar(bins[:bins_number], nEEGsvarsei/sum(nEEGsvarsei),width=width, color='grey',alpha=0.7,edgecolor='k',linewidth=2)
pyplot.setp(ax.get_yticklabels(), color='k')
# ax.set_title('seizure timing histogram (SA0124)',fontsize=23)
ax.set_xticks(np.linspace(0, 2*np.pi, 24, endpoint=False))
ax.set_xticklabels(['0 am','','','Night','','','6 am','','','Morning','','','12 am','','','Afternoon','','','18 pm','','','Evening','','','24 pm'],fontsize=16)
# locs, labels = pyplot.xticks([0,3,6,9,12,15,18,21,24],['0','Night','6','Morning','12','Afternoon','18','Evening','24'],fontsize=16)
locs, labels = pyplot.yticks([0.1,0.3,0.5],['0.1','0.3','0.5'],fontsize=16)
pyplot.show()
bins_number = 18
bins = np.linspace(0, 24, bins_number + 1)
ntimes, _, _ = pyplot.hist(time_feature_arr[0:18000], bins)
ntimesei, _, _ = pyplot.hist(seizure_time, bins)
print(ntimes)
print(ntimesei)



# #### section 2 training training
medium_rhythm_var_arr_3 = movingaverage(Raw_variance_EEG, 240 * 12)
long_rhythm_var_arr = medium_rhythm_var_arr_3
var_trans = hilbert(long_rhythm_var_arr)
var_phase = np.angle(var_trans)
phase_long_EEGvariance_arr = var_phase
print(len(phase_long_EEGvariance_arr));
medium_rhythm_value_arr_3 = movingaverage(Raw_auto_EEG, 240 * 12)
long_rhythm_value_arr = medium_rhythm_value_arr_3
value_trans = hilbert(long_rhythm_value_arr)
value_phase = np.angle(value_trans)
phase_long_EEGauto_arr = value_phase
print(len(phase_long_EEGauto_arr));
medium_rhythm_RRIvar_arr_3 = movingaverage(Raw_variance_RRI31, 240 * 12)
long_rhythm_RRIvar_arr = medium_rhythm_RRIvar_arr_3
var_trans = hilbert(long_rhythm_RRIvar_arr)
var_phase = np.angle(var_trans)
phase_long_RRIvariance_arr = var_phase
print(len(phase_long_RRIvariance_arr));
medium_rhythm_RRIvalue_arr_3 = movingaverage(Raw_auto_RRI31, 240 * 12)
long_rhythm_RRIvalue_arr = medium_rhythm_RRIvalue_arr_3
value_trans = hilbert(long_rhythm_RRIvalue_arr)
value_phase = np.angle(value_trans)
phase_long_RRIauto_arr = value_phase
print(len(phase_long_RRIauto_arr));

#### combined probability calculation
###### 12h
bins_number = 18
bins = np.linspace(-np.pi, np.pi, bins_number + 1)
pro_eegvars_time = []
pro_eegvars_time_false = []
for i in range(len(phase_long_EEGvariance_arr)):
    if phase_long_EEGvariance_arr[i] >= bins[0] and phase_long_EEGvariance_arr[i] < bins[1]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] >= bins[1] and phase_long_EEGvariance_arr[i] < bins[2]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] >= bins[2] and phase_long_EEGvariance_arr[i] < bins[3]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] >= bins[3] and phase_long_EEGvariance_arr[i] < bins[4]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] >= bins[4] and phase_long_EEGvariance_arr[i] < bins[5]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] >= bins[5] and phase_long_EEGvariance_arr[i] < bins[6]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] >= bins[6] and phase_long_EEGvariance_arr[i] <= bins[7]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] > bins[7] and phase_long_EEGvariance_arr[i] < bins[8]:
        pro_eegvars_time_false.append(0.146262851)
        pro_eegvars_time.append(0.4)
    elif phase_long_EEGvariance_arr[i] >= bins[8] and phase_long_EEGvariance_arr[i] <= bins[9]:
        pro_eegvars_time_false.append(0.405557099)
        pro_eegvars_time.append(0.6)
    elif phase_long_EEGvariance_arr[i] > bins[9] and phase_long_EEGvariance_arr[i] < bins[10]:
        pro_eegvars_time_false.append(0.376215615)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] >= bins[10] and phase_long_EEGvariance_arr[i] <= bins[11]:
        pro_eegvars_time_false.append(0.030619617)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] > bins[11] and phase_long_EEGvariance_arr[i] < bins[12]:
        pro_eegvars_time_false.append(0.013781606)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] > bins[12] and phase_long_EEGvariance_arr[i] < bins[13]:
        pro_eegvars_time_false.append(0.014059461)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] >= bins[13] and phase_long_EEGvariance_arr[i] < bins[14]:
        pro_eegvars_time_false.append(0.013503751)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] >= bins[14] and phase_long_EEGvariance_arr[i] < bins[15]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] >= bins[15] and phase_long_EEGvariance_arr[i] < bins[16]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] >= bins[16] and phase_long_EEGvariance_arr[i] < bins[17]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] >= bins[17]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
print(pro_eegvars_time[4151]);print(pro_eegvars_time[10471]);
print(pro_eegvars_time[11111]);print(pro_eegvars_time[11942]);print(pro_eegvars_time[15631]);
pro_eegautos_time = []
pro_eegautos_time_false = []
for i in range(len(phase_long_EEGauto_arr)):
    if phase_long_EEGauto_arr[i] >= bins[0] and phase_long_EEGauto_arr[i] < bins[1]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[1] and phase_long_EEGauto_arr[i] < bins[2]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[2] and phase_long_EEGauto_arr[i] < bins[3]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[3] and phase_long_EEGauto_arr[i] < bins[4]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[4] and phase_long_EEGauto_arr[i] < bins[5]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[5] and phase_long_EEGauto_arr[i] < bins[6]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[6] and phase_long_EEGauto_arr[i] < bins[7]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[7] and phase_long_EEGauto_arr[i] < bins[8]:
        pro_eegautos_time_false.append(0.115754376)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[8] and phase_long_EEGauto_arr[i] < bins[9]:
        pro_eegautos_time_false.append(0.353375938)
        pro_eegautos_time.append(0.8)
    elif phase_long_EEGauto_arr[i] >= bins[9] and phase_long_EEGauto_arr[i] <= bins[10]:
        pro_eegautos_time_false.append(0.468241178)
        pro_eegautos_time.append(0.2)
    elif phase_long_EEGauto_arr[i] > bins[10] and phase_long_EEGauto_arr[i] < bins[11]:
        pro_eegautos_time_false.append(0.033787163)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[11] and phase_long_EEGauto_arr[i] < bins[12]:
        pro_eegautos_time_false.append(0.009613782)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[12] and phase_long_EEGauto_arr[i] < bins[13]:
        pro_eegautos_time_false.append(0.008057794)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[13] and phase_long_EEGauto_arr[i] < bins[14]:
        pro_eegautos_time_false.append(0.011169769)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[14] and phase_long_EEGauto_arr[i] < bins[15]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[15] and phase_long_EEGauto_arr[i] < bins[16]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[16] and phase_long_EEGauto_arr[i] < bins[17]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[17]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
print(pro_eegautos_time[4151]);print(pro_eegautos_time[10471]);
print(pro_eegautos_time[11111]);print(pro_eegautos_time[11942]);print(pro_eegautos_time[15631]);
pro_RRIvars_time = []
pro_RRIvars_time_false = []
for i in range(len(phase_long_RRIvariance_arr)):
    if phase_long_RRIvariance_arr[i] >= bins[0] and phase_long_RRIvariance_arr[i] < bins[1]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[1] and phase_long_RRIvariance_arr[i] < bins[2]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[2] and phase_long_RRIvariance_arr[i] < bins[3]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[3] and phase_long_RRIvariance_arr[i] < bins[4]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[4] and phase_long_RRIvariance_arr[i] < bins[5]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[5] and phase_long_RRIvariance_arr[i] < bins[6]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[6] and phase_long_RRIvariance_arr[i] < bins[7]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[7] and phase_long_RRIvariance_arr[i] <= bins[8]:
        pro_RRIvars_time_false.append(0.163267574)
        pro_RRIvars_time.append(0.6)
    elif phase_long_RRIvariance_arr[i] > bins[8] and phase_long_RRIvariance_arr[i] < bins[9]:
        pro_RRIvars_time_false.append(0.308419005)
        pro_RRIvars_time.append(0.4)
    elif phase_long_RRIvariance_arr[i] >= bins[9] and phase_long_RRIvariance_arr[i] <= bins[10]:
        pro_RRIvars_time_false.append(0.415504307)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] > bins[10] and phase_long_RRIvariance_arr[i] <= bins[11]:
        pro_RRIvars_time_false.append(0.07618783)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] > bins[11] and phase_long_RRIvariance_arr[i] < bins[12]:
        pro_RRIvars_time_false.append(0.018949708)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[12] and phase_long_RRIvariance_arr[i] < bins[13]:
        pro_RRIvars_time_false.append(0.010502917)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[13] and phase_long_RRIvariance_arr[i] < bins[14]:
        pro_RRIvars_time_false.append(0.007168658)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[14] and phase_long_RRIvariance_arr[i] < bins[15]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[15] and phase_long_RRIvariance_arr[i] < bins[16]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[16] and phase_long_RRIvariance_arr[i] < bins[17]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[17]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
print(pro_RRIvars_time[4151]);print(pro_RRIvars_time[10471]);
print(pro_RRIvars_time[11111]);print(pro_RRIvars_time[11942]);print(pro_RRIvars_time[15631]);
pro_RRIautos_time = []
pro_RRIautos_time_false = []
for i in range(len(phase_long_RRIauto_arr)):
    if phase_long_RRIauto_arr[i] >= bins[0] and phase_long_RRIauto_arr[i] < bins[1]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[1] and phase_long_RRIauto_arr[i] < bins[2]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[2] and phase_long_RRIauto_arr[i] < bins[3]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[3] and phase_long_RRIauto_arr[i] < bins[4]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[4] and phase_long_RRIauto_arr[i] < bins[5]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[5] and phase_long_RRIauto_arr[i] < bins[6]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[6] and phase_long_RRIauto_arr[i] < bins[7]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[7] and phase_long_RRIauto_arr[i] <= bins[8]:
        pro_RRIautos_time_false.append(0.087524312)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] > bins[8] and phase_long_RRIauto_arr[i] <= bins[9]:
        pro_RRIautos_time_false.append(0.3901083639)
        pro_RRIautos_time.append(1)
    elif phase_long_RRIauto_arr[i] > bins[9] and phase_long_RRIauto_arr[i] <= bins[10]:
        pro_RRIautos_time_false.append(0.456682412)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] > bins[10] and phase_long_RRIauto_arr[i] < bins[11]:
        pro_RRIautos_time_false.append(0.030008336)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[11] and phase_long_RRIauto_arr[i] < bins[12]:
        pro_RRIautos_time_false.append(0.012114476)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[12] and phase_long_RRIauto_arr[i] < bins[13]:
        pro_RRIautos_time_false.append(0.010225063)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[13] and phase_long_RRIauto_arr[i] < bins[14]:
        pro_RRIautos_time_false.append(0.013337038)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[14] and phase_long_RRIauto_arr[i] < bins[15]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[15] and phase_long_RRIauto_arr[i] < bins[16]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[16] and phase_long_RRIauto_arr[i] < bins[17]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[17]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
print(pro_RRIautos_time[4151]);print(pro_RRIautos_time[10471]);
print(pro_RRIautos_time[11111]);print(pro_RRIautos_time[11942]);print(pro_RRIautos_time[15631]);

Pseizureeegvar = 0.000277778;
Pnonseizureeegvar = 0.999722222;
t=np.linspace(0+0.00416667,0+0.00416667+0.00416667*(len(Raw_variance_EEG)-1),len(Raw_variance_EEG))
window_time_arr=t

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_RRIvars_time[m]*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))

Pcombined=[]
for m in range(len(pro_eegvars_time)):
    P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]
    P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m])
    Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_RRIvars_time[m]*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))
# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=Pseizureeegvar*pro_RRIvars_time[m]*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))
# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m])
#     Pcombined.append(P1/(P1+P2))
# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=Pseizureeegvar*pro_RRIvars_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m])
#     Pcombined.append(P1/(P1+P2))
# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=Pseizureeegvar*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))


pyplot.figure(figsize=(12, 5))
pyplot.plot(window_time_arr, Pcombined)
pyplot.title('combined probability in VIC1012', fontsize=15)
pyplot.annotate('', xy=(17.30305, np.max(Pcombined)), xytext=(17.30305, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(43.636667, np.max(Pcombined)), xytext=(43.636667, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(46.30361, np.max(Pcombined)), xytext=(46.30361, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(49.76528, np.max(Pcombined)), xytext=(49.76528, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(65.136667, np.max(Pcombined)), xytext=(65.136667, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.tight_layout()
pyplot.xlabel('Time(h)', fontsize=15)
pyplot.ylabel('seizure probability', fontsize=15)
pyplot.show()
for item in seizure_timing_index:
    print(Pcombined[item])




t=np.linspace(0,0+0.00416667*(len(Raw_variance_EEG_arr)-1),len(Raw_variance_EEG_arr))
a=np.where(t<2.80305556+0)
t[0:672]=t[0:672]-0+21.1969444
t[672:]=t[672:]-2.80305556-0
time_feature_arr=[]
for i in range(len(t)):
    if t[i]>24:
        time_feature_arr.append(t[i] - (t[i] // 24) * 24)
    else:
        time_feature_arr.append(t[i])


bins_number = 18
bins = np.linspace(0, 24, bins_number + 1)
pro_circadian_time=[]
pro_circadian_time_false=[]
for i in range(len(time_feature_arr)):
    if time_feature_arr[i] >= bins[0] and time_feature_arr[i] <= bins[1]:
        pro_circadian_time_false.append(0.05601556)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] > bins[1] and time_feature_arr[i] < bins[2]:
        pro_circadian_time_false.append(0.053348152)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] >= bins[2] and time_feature_arr[i] < bins[3]:
        pro_circadian_time_false.append(0.05334815)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] >= bins[3] and time_feature_arr[i] < bins[4]:
        pro_circadian_time_false.append(0.05334815)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] >= bins[4] and time_feature_arr[i] < bins[5]:
        pro_circadian_time_false.append(0.05334815)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] >= bins[5] and time_feature_arr[i] <= bins[6]:
        pro_circadian_time_false.append(0.05334815)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] > bins[6] and time_feature_arr[i] < bins[7]:
        pro_circadian_time_false.append(0.05334815)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] >= bins[7] and time_feature_arr[i] <= bins[8]:
        pro_circadian_time_false.append(0.05334815)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] > bins[8] and time_feature_arr[i] < bins[9]:
        pro_circadian_time_false.append(0.05334815)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] >= bins[9] and time_feature_arr[i] < bins[10]:
        pro_circadian_time_false.append(0.05334815)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] >= bins[10] and time_feature_arr[i] < bins[11]:
        pro_circadian_time_false.append(0.05323701)
        pro_circadian_time.append(0.4)
    elif time_feature_arr[i] >= bins[11] and time_feature_arr[i] < bins[12]:
        pro_circadian_time_false.append(0.053348152)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] >= bins[12] and time_feature_arr[i] < bins[13]:
        pro_circadian_time_false.append(0.053292581)
        pro_circadian_time.append(0.2)
    elif time_feature_arr[i] >= bins[13] and time_feature_arr[i] < bins[14]:
        pro_circadian_time_false.append(0.053348152)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] >= bins[14] and time_feature_arr[i] < bins[15]:
        pro_circadian_time_false.append(0.053292581)
        pro_circadian_time.append(0.2)
    elif time_feature_arr[i] >= bins[15] and time_feature_arr[i] < bins[16]:
        pro_circadian_time_false.append(0.055126424)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] >= bins[16] and time_feature_arr[i] < bins[17]:
        pro_circadian_time_false.append(0.07113087)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] >= bins[17] and time_feature_arr[i] <= bins[18]:
        pro_circadian_time_false.append(0.071075299)
        pro_circadian_time.append(0.2)

# Pcombined=[]
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_RRIvars_time[m]*pro_RRIautos_time[m]*pro_circadian_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
#     Pcombined.append(P1/(P1+P2))
# Pcombined=[]
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_RRIvars_time[m]*pro_RRIautos_time[m]*pro_circadian_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
#     Pcombined.append(P1/(P1+P2))

Pcombined=[]
for m in range(len(pro_eegvars_time)):
    P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_circadian_time[m]
    P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_circadian_time_false[m])
    Pcombined.append(P1/(P1+P2))

# Pcombined=[]
# for m in range(len(pro_eegvars_time)):
#     P1=Pseizureeegvar*pro_RRIvars_time[m]*pro_RRIautos_time[m]*pro_circadian_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
#     Pcombined.append(P1/(P1+P2))
# Pcombined=[]
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_circadian_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_circadian_time_false[m])
#     Pcombined.append(P1/(P1+P2))
# Pcombined=[]
# for m in range(len(pro_eegvars_time)):
#     P1=Pseizureeegvar*pro_RRIvars_time[m]*pro_circadian_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m]*pro_circadian_time_false[m])
#     Pcombined.append(P1/(P1+P2))
# Pcombined=[]
# for m in range(len(pro_eegvars_time)):
#     P1=Pseizureeegvar*pro_RRIautos_time[m]*pro_circadian_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
#     Pcombined.append(P1/(P1+P2))


t=np.linspace(0+0.00416667,0+0.00416667+0.00416667*(len(Raw_variance_EEG)-1),len(Raw_variance_EEG))
window_time_arr=t
pyplot.figure(figsize=(12, 5))
pyplot.plot(window_time_arr, Pcombined)
pyplot.title('combined probability in VIC1012', fontsize=15)
pyplot.annotate('', xy=(17.30305, np.max(Pcombined)), xytext=(17.30305, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(43.636667, np.max(Pcombined)), xytext=(43.636667, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(46.30361, np.max(Pcombined)), xytext=(46.30361, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(49.76528, np.max(Pcombined)), xytext=(49.76528, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(65.136667, np.max(Pcombined)), xytext=(65.136667, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.tight_layout()
pyplot.xlim(window_time_arr[0], window_time_arr[-1])
pyplot.xlabel('Time(h)', fontsize=15)
pyplot.ylabel('seizure probability', fontsize=15)
pyplot.show()
for item in seizure_timing_index:
    print(Pcombined[item])








# ## section 3 froecast
t=np.linspace(0,0+0.00416667*(len(Raw_variance_EEG_arr)-1),len(Raw_variance_EEG_arr))
t_window_arr=t
fore_arr_EEGvars=[]
for k in range(75,76):
# for k in range(60, 61):
    variance_arr = Raw_variance_EEG_arr[0:(18000+240*k)]
    long_rhythm_var_arr=movingaverage(variance_arr,240*12)
    pyplot.figure(figsize=(6, 3))
    pyplot.title('EEG variance')
    pyplot.ylabel('Voltage ($\mathregular{v^2}$)')
    pyplot.xlabel('Time(h)')
    pyplot.plot(t_window_arr[240*6:(18000+240*k)], long_rhythm_var_arr[240*6:],'orange')
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC1012/cycles12h_Cz_forecast75hsignal_3hcycle_EEGvar_VIC1012.csv',sep=',',header=None)
forecast_var_EEG= csv_reader.values
forecast_var_EEG_arr=[]
for item in forecast_var_EEG:
    forecast_var_EEG_arr=forecast_var_EEG_arr+list(item)
t=np.linspace(t_window_arr[18000],t_window_arr[18000]+0.1666667*(len(forecast_var_EEG_arr)-1),len(forecast_var_EEG_arr))
pyplot.plot(t, forecast_var_EEG_arr,'k',label='forecast EEG var')
pyplot.legend()
pyplot.show()

fore_arr_EEGauto=[]
for k in range(75,76):
    auto_arr = Raw_auto_EEG_arr[0:(18000+240*k)]
    long_rhythm_auto_arr=movingaverage(auto_arr,240*12)
    pyplot.figure(figsize=(6, 3))
    pyplot.title('EEG autocorrelation')
    pyplot.xlabel('time(h)')
    pyplot.plot(t_window_arr[240*6:(18000+240*k)], long_rhythm_auto_arr[240*6:],'orange')
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC1012/cycles12h_Cz_forecast75hsignal_3hcycle_EEGauto_VIC1012.csv',sep=',',header=None)
forecast_auto_EEG= csv_reader.values
forecast_auto_EEG_arr=[]
for item in forecast_auto_EEG:
    forecast_auto_EEG_arr=forecast_auto_EEG_arr+list(item)
t=np.linspace(t_window_arr[18000],t_window_arr[18000]+0.1666667*(len(forecast_auto_EEG_arr)-1),len(forecast_auto_EEG_arr))
pyplot.plot(t, forecast_auto_EEG_arr,'k',label='forecast EEG auto')
pyplot.legend()
pyplot.show()

fore_arr_RRIvars=[]
for k in range(75, 76):
    variance_arr = Raw_variance_RRI31_arr[0:(18000+240*k)]
    long_rhythm_var_arr=movingaverage(variance_arr,240*12)
    pyplot.figure(figsize=(6, 3))
    pyplot.title('RRI variance')
    pyplot.ylabel('Second ($\mathregular{s^2}$)')
    pyplot.xlabel('Time(h)')
    pyplot.plot(t_window_arr[240*6:(18000+240*k)], long_rhythm_var_arr[240*6:],'orange')
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC1012/cycles12h_ch31_forecast75hsignal_3hcycle_RRIvar_VIC1012.csv', sep=',',header=None)
forecast_var_RRI31= csv_reader.values
forecast_var_RRI31_arr=[]
for item in forecast_var_RRI31:
    forecast_var_RRI31_arr=forecast_var_RRI31_arr+list(item)
t=np.linspace(t_window_arr[18000],t_window_arr[18000]+0.1666667*(len(forecast_var_RRI31_arr)-1),len(forecast_var_RRI31_arr))
pyplot.plot(t, forecast_var_RRI31_arr,'k',label='forecast RRI var')
pyplot.legend()
pyplot.show()

fore_arr_RRIautos=[]
save_data_RRIautos=[]
for k in range(75,76):
    auto_arr = Raw_auto_RRI31_arr[0:18000+240*k]
    long_rhythm_auto_arr=movingaverage(auto_arr,240*12)
    pyplot.figure(figsize=(6,3))
    pyplot.title('RRI autocorrelation')
    pyplot.xlabel('Time(h)')
    pyplot.plot(t_window_arr[240*6:18000+240*k], long_rhythm_auto_arr[240*6:],'orange')
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC1012/cycles12h_ch31_forecast75hsignal_3hcycle_RRIauto_VIC1012.csv',sep=',',header=None)
forecast_auto_RRI31= csv_reader.values
forecast_auto_RRI31_arr=[]
for item in forecast_auto_RRI31:
    forecast_auto_RRI31_arr=forecast_auto_RRI31_arr+list(item)
t=np.linspace(t_window_arr[18000],t_window_arr[18000]+0.1666667*(len(forecast_auto_RRI31_arr)-1),len(forecast_auto_RRI31_arr))
pyplot.plot(t, forecast_auto_RRI31_arr,'k',label='forecast RRI auto')
pyplot.legend()
pyplot.show()
# print(len(forecast_var_EEG_arr));print(len(forecast_auto_EEG_arr));
# # print(len(forecast_var_RRI31_arr));
# print(len(forecast_auto_RRI31_arr));



# ### predict, forecast data
var_trans=hilbert(forecast_var_EEG_arr)
var_phase=np.angle(var_trans)
rolmean_short_EEGvar=var_phase

var_trans=hilbert(forecast_auto_EEG_arr)
var_phase=np.angle(var_trans)
rolmean_short_EEGauto=var_phase

var_trans=hilbert(forecast_var_RRI31_arr)
var_phase=np.angle(var_trans)
rolmean_short_RRIvar=var_phase

var_trans=hilbert(forecast_auto_RRI31_arr)
var_phase=np.angle(var_trans)
rolmean_short_RRIauto=var_phase




##### 12h 12h 12h
bins_number = 18
bins = np.linspace(-np.pi, np.pi, bins_number + 1)
pro_eegvars_time = []
pro_eegvars_time_false = []
for i in range(len(rolmean_short_EEGvar)):
    if rolmean_short_EEGvar[i] >= bins[0] and rolmean_short_EEGvar[i] < bins[1]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] >= bins[1] and rolmean_short_EEGvar[i] < bins[2]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] >= bins[2] and rolmean_short_EEGvar[i] < bins[3]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] >= bins[3] and rolmean_short_EEGvar[i] < bins[4]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] >= bins[4] and rolmean_short_EEGvar[i] < bins[5]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] >= bins[5] and rolmean_short_EEGvar[i] < bins[6]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] >= bins[6] and rolmean_short_EEGvar[i] <= bins[7]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] > bins[7] and rolmean_short_EEGvar[i] < bins[8]:
        pro_eegvars_time_false.append(0.146262851)
        pro_eegvars_time.append(0.4)
    elif rolmean_short_EEGvar[i] >= bins[8] and rolmean_short_EEGvar[i] <= bins[9]:
        pro_eegvars_time_false.append(0.405557099)
        pro_eegvars_time.append(0.6)
    elif rolmean_short_EEGvar[i] > bins[9] and rolmean_short_EEGvar[i] < bins[10]:
        pro_eegvars_time_false.append(0.376215615)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] >= bins[10] and rolmean_short_EEGvar[i] <= bins[11]:
        pro_eegvars_time_false.append(0.030619617)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] > bins[11] and rolmean_short_EEGvar[i] < bins[12]:
        pro_eegvars_time_false.append(0.013781606)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] > bins[12] and rolmean_short_EEGvar[i] < bins[13]:
        pro_eegvars_time_false.append(0.014059461)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] >= bins[13] and rolmean_short_EEGvar[i] < bins[14]:
        pro_eegvars_time_false.append(0.013503751)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] >= bins[14] and rolmean_short_EEGvar[i] < bins[15]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] >= bins[15] and rolmean_short_EEGvar[i] < bins[16]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] >= bins[16] and rolmean_short_EEGvar[i] < bins[17]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] >= bins[17]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
pro_eegautos_time = []
pro_eegautos_time_false = []
for i in range(len(rolmean_short_EEGauto)):
    if rolmean_short_EEGauto[i] >= bins[0] and rolmean_short_EEGauto[i] < bins[1]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[1] and rolmean_short_EEGauto[i] < bins[2]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[2] and rolmean_short_EEGauto[i] < bins[3]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[3] and rolmean_short_EEGauto[i] < bins[4]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[4] and rolmean_short_EEGauto[i] < bins[5]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[5] and rolmean_short_EEGauto[i] < bins[6]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[6] and rolmean_short_EEGauto[i] < bins[7]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[7] and rolmean_short_EEGauto[i] < bins[8]:
        pro_eegautos_time_false.append(0.115754376)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[8] and rolmean_short_EEGauto[i] < bins[9]:
        pro_eegautos_time_false.append(0.353375938)
        pro_eegautos_time.append(0.8)
    elif rolmean_short_EEGauto[i] >= bins[9] and rolmean_short_EEGauto[i] <= bins[10]:
        pro_eegautos_time_false.append(0.468241178)
        pro_eegautos_time.append(0.2)
    elif rolmean_short_EEGauto[i] > bins[10] and rolmean_short_EEGauto[i] < bins[11]:
        pro_eegautos_time_false.append(0.033787163)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[11] and rolmean_short_EEGauto[i] < bins[12]:
        pro_eegautos_time_false.append(0.009613782)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[12] and rolmean_short_EEGauto[i] < bins[13]:
        pro_eegautos_time_false.append(0.008057794)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[13] and rolmean_short_EEGauto[i] < bins[14]:
        pro_eegautos_time_false.append(0.011169769)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[14] and rolmean_short_EEGauto[i] < bins[15]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[15] and rolmean_short_EEGauto[i] < bins[16]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[16] and rolmean_short_EEGauto[i] < bins[17]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[17]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
pro_RRIvars_time = []
pro_RRIvars_time_false = []
for i in range(len(rolmean_short_RRIvar)):
    if rolmean_short_RRIvar[i] >= bins[0] and rolmean_short_RRIvar[i] < bins[1]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] >= bins[1] and rolmean_short_RRIvar[i] < bins[2]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] >= bins[2] and rolmean_short_RRIvar[i] < bins[3]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] >= bins[3] and rolmean_short_RRIvar[i] < bins[4]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] >= bins[4] and rolmean_short_RRIvar[i] < bins[5]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] >= bins[5] and rolmean_short_RRIvar[i] < bins[6]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] >= bins[6] and rolmean_short_RRIvar[i] < bins[7]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] >= bins[7] and rolmean_short_RRIvar[i] <= bins[8]:
        pro_RRIvars_time_false.append(0.163267574)
        pro_RRIvars_time.append(0.6)
    elif rolmean_short_RRIvar[i] > bins[8] and rolmean_short_RRIvar[i] < bins[9]:
        pro_RRIvars_time_false.append(0.308419005)
        pro_RRIvars_time.append(0.4)
    elif rolmean_short_RRIvar[i] >= bins[9] and rolmean_short_RRIvar[i] <= bins[10]:
        pro_RRIvars_time_false.append(0.415504307)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] > bins[10] and rolmean_short_RRIvar[i] <= bins[11]:
        pro_RRIvars_time_false.append(0.07618783)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] > bins[11] and rolmean_short_RRIvar[i] < bins[12]:
        pro_RRIvars_time_false.append(0.018949708)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] >= bins[12] and rolmean_short_RRIvar[i] < bins[13]:
        pro_RRIvars_time_false.append(0.010502917)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] >= bins[13] and rolmean_short_RRIvar[i] < bins[14]:
        pro_RRIvars_time_false.append(0.007168658)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] >= bins[14] and rolmean_short_RRIvar[i] < bins[15]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] >= bins[15] and rolmean_short_RRIvar[i] < bins[16]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] >= bins[16] and rolmean_short_RRIvar[i] < bins[17]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] >= bins[17]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
pro_RRIautos_time = []
pro_RRIautos_time_false = []
for i in range(len(rolmean_short_RRIauto)):
    if rolmean_short_RRIauto[i] >= bins[0] and rolmean_short_RRIauto[i] < bins[1]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[1] and rolmean_short_RRIauto[i] < bins[2]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[2] and rolmean_short_RRIauto[i] < bins[3]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[3] and rolmean_short_RRIauto[i] < bins[4]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[4] and rolmean_short_RRIauto[i] < bins[5]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[5] and rolmean_short_RRIauto[i] < bins[6]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[6] and rolmean_short_RRIauto[i] < bins[7]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[7] and rolmean_short_RRIauto[i] <= bins[8]:
        pro_RRIautos_time_false.append(0.087524312)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] > bins[8] and rolmean_short_RRIauto[i] <= bins[9]:
        pro_RRIautos_time_false.append(0.3901083639)
        pro_RRIautos_time.append(1)
    elif rolmean_short_RRIauto[i] > bins[9] and rolmean_short_RRIauto[i] <= bins[10]:
        pro_RRIautos_time_false.append(0.456682412)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] > bins[10] and rolmean_short_RRIauto[i] < bins[11]:
        pro_RRIautos_time_false.append(0.030008336)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[11] and rolmean_short_RRIauto[i] < bins[12]:
        pro_RRIautos_time_false.append(0.012114476)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[12] and rolmean_short_RRIauto[i] < bins[13]:
        pro_RRIautos_time_false.append(0.010225063)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[13] and rolmean_short_RRIauto[i] < bins[14]:
        pro_RRIautos_time_false.append(0.013337038)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[14] and rolmean_short_RRIauto[i] < bins[15]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[15] and rolmean_short_RRIauto[i] < bins[16]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[16] and rolmean_short_RRIauto[i] < bins[17]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[17]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)




Pseizureeegvar = 0.000277778;
Pnonseizureeegvar = 0.999722222;


# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_RRIvars_time[m]*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))

Pcombined = []
for m in range(len(pro_eegvars_time)):
    P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]
    P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m])
    Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=Pseizureeegvar*pro_RRIvars_time[m]*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=Pseizureeegvar*pro_RRIvars_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=Pseizureeegvar*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))

pyplot.figure(figsize=(8,4))
RRI_timewindow_arr=t
pyplot.plot(RRI_timewindow_arr,Pcombined)
pyplot.annotate('',xy=(120.803,np.max(Pcombined)),xytext=(120.803,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='k',shrink=0.05))
pyplot.title('Forecast seizures in VIC1012')
pyplot.xlabel('Time(h)')
pyplot.ylabel('Seizure probability')
pyplot.show()


Pcombined=split(Pcombined,6)
print(len(Pcombined))
index=[]
for i in range(len(Pcombined)):
    for item in Pcombined[i]:
        if item >= 4.115688090795725e-05:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[120.803]
k=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k=k+1
            n_arr.append(n)
print(k)
index=[]
for i in range(len(Pcombined)):
    for item in Pcombined[i]:
        if item >= 0.3*4.115688090795725e-05:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[120.803]
k=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k=k+1
            n_arr.append(n)
print(k)
index=[]
for i in range(len(Pcombined)):
    for item in Pcombined[i]:
        if item >= 0.6*4.115688090795725e-05:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[120.803]
k=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k=k+1
            n_arr.append(n)
print(k)
index=[]
for i in range(len(Pcombined)):
    for item in Pcombined[i]:
        if item >= 1.2*4.115688090795725e-05:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[120.803]
k=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k=k+1
            n_arr.append(n)
print(k)
index=[]
for i in range(len(Pcombined)):
    for item in Pcombined[i]:
        if item >= 2*4.115688090795725e-05:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[120.803]
k=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k=k+1
            n_arr.append(n)
print(k)



t1=np.linspace(0,0+0.00416667*(len(Raw_variance_EEG_arr)-1),len(Raw_variance_EEG_arr))
a=np.where(t1<2.80305556+0)
t1[0:672]=t1[0:672]-0+21.1969444
t1[672:]=t1[672:]-2.80305556-0
time_feature_arr=[]
for i in range(len(t1)):
    if t1[i]>24:
        time_feature_arr.append(t1[i] - (t1[i] // 24) * 24)
    else:
        time_feature_arr.append(t1[i])

print(len(time_feature_arr))
time_arr=time_feature_arr[18000:]
print(len(time_arr))
new_arr=[]
for j in range(0,450):
    new_arr.append(time_arr[40*j])

bins_number = 18
bins = np.linspace(0, 24, bins_number + 1)
pro_circadian_time=[]
pro_circadian_time_false=[]
for i in range(len(new_arr)):
    if new_arr[i] >= bins[0] and new_arr[i] <= bins[1]:
        pro_circadian_time_false.append(0.049367479)
        pro_circadian_time.append(0)
    elif new_arr[i] > bins[1] and new_arr[i] < bins[2]:
        pro_circadian_time_false.append(0.049367479)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[2] and new_arr[i] < bins[3]:
        pro_circadian_time_false.append(0.049367479)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[3] and new_arr[i] < bins[4]:
        pro_circadian_time_false.append(0.049367479)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[4] and new_arr[i] < bins[5]:
        pro_circadian_time_false.append(0.049367479)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[5] and new_arr[i] <= bins[6]:
        pro_circadian_time_false.append(0.049367479)
        pro_circadian_time.append(0)
    elif new_arr[i] > bins[6] and new_arr[i] < bins[7]:
        pro_circadian_time_false.append(0.049316055)
        pro_circadian_time.append(0.25)
    elif new_arr[i] >= bins[7] and new_arr[i] <= bins[8]:
        pro_circadian_time_false.append(0.049316055)
        pro_circadian_time.append(0.25)
    elif new_arr[i] > bins[8] and new_arr[i] < bins[9]:
        pro_circadian_time_false.append(0.049367479)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[9] and new_arr[i] < bins[10]:
        pro_circadian_time_false.append(0.049367479)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[10] and new_arr[i] < bins[11]:
        pro_circadian_time_false.append(0.05075594)
        pro_circadian_time.append(0.25)
    elif new_arr[i] >= bins[11] and new_arr[i] < bins[12]:
        pro_circadian_time_false.append(0.065771881)
        pro_circadian_time.append(0.25)
    elif new_arr[i] >= bins[12] and new_arr[i] < bins[13]:
        pro_circadian_time_false.append(0.065823306)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[13] and new_arr[i] < bins[14]:
        pro_circadian_time_false.append(0.065823306)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[14] and new_arr[i] < bins[15]:
        pro_circadian_time_false.append(0.065823306)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[15] and new_arr[i] < bins[16]:
        pro_circadian_time_false.append(0.065823306)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[16] and new_arr[i] < bins[17]:
        pro_circadian_time_false.append(0.065823306)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[17] and new_arr[i] <= bins[18]:
        pro_circadian_time_false.append(0.060783709)
        pro_circadian_time.append(0)





# RRI_timewindow_arr=t[0:len(pro_circadian_time)]
# print(RRI_timewindow_arr[-1]-RRI_timewindow_arr[0])
# pyplot.figure(figsize=(8,4))
# pyplot.plot(RRI_timewindow_arr,pro_circadian_time)
# pyplot.annotate('',xy=(120.803,np.max(pro_circadian_time)),xytext=(120.803,np.max(pro_circadian_time)+0.000000000001),arrowprops=dict(facecolor='r',shrink=0.05))
# pyplot.hlines(0.2, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'r')
# # pyplot.hlines(0.11111, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'orange')
# # pyplot.hlines(0.22222, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'orange')
# pyplot.title('Forecast seizures in VIC1012')
# pyplot.xlabel('Time(h)')
# pyplot.ylabel('Seizure probability')
# pyplot.show()
# Pcombined=split(pro_circadian_time,6)
# print(len(Pcombined))
# index=[]
# for i in range(len(Pcombined)):
#     for item in Pcombined[i]:
#         if item >= 0.2:
#             index.append(6*i+0)
# print(RRI_timewindow_arr[index])
# a=np.unique(RRI_timewindow_arr[index])
# print(a); print(len(a))
# time_arr=[120.803]
# k=0
# n_arr=[]
# for m in time_arr:
#     for n in a:
#         if m-n<=1 and m-n>=0:
#             k=k+1
#             n_arr.append(n)
# print(k)
# index=[]
# for i in range(len(Pcombined)):
#     for item in Pcombined[i]:
#         if item >= 0.3*0.2:
#             index.append(6*i+0)
# print(RRI_timewindow_arr[index])
# a=np.unique(RRI_timewindow_arr[index])
# print(a); print(len(a))
# time_arr=[120.803]
# k=0
# n_arr=[]
# for m in time_arr:
#     for n in a:
#         if m-n<=1 and m-n>=0:
#             k=k+1
#             n_arr.append(n)
# print(k)
# index=[]
# for i in range(len(Pcombined)):
#     for item in Pcombined[i]:
#         if item >= 0.6*0.2:
#             index.append(6*i+0)
# print(RRI_timewindow_arr[index])
# a=np.unique(RRI_timewindow_arr[index])
# print(a); print(len(a))
# time_arr=[120.803]
# k=0
# n_arr=[]
# for m in time_arr:
#     for n in a:
#         if m-n<=1 and m-n>=0:
#             k=k+1
#             n_arr.append(n)
# print(k)
# index=[]
# for i in range(len(Pcombined)):
#     for item in Pcombined[i]:
#         if item >= 1.2*0.2:
#             index.append(6*i+0)
# print(RRI_timewindow_arr[index])
# a=np.unique(RRI_timewindow_arr[index])
# print(a); print(len(a))
# time_arr=[120.803]
# k=0
# n_arr=[]
# for m in time_arr:
#     for n in a:
#         if m-n<=1 and m-n>=0:
#             k=k+1
#             n_arr.append(n)
# print(k)
# index=[]
# for i in range(len(Pcombined)):
#     for item in Pcombined[i]:
#         if item >= 2*0.2:
#             index.append(6*i+0)
# print(RRI_timewindow_arr[index])
# a=np.unique(RRI_timewindow_arr[index])
# print(a); print(len(a))
# time_arr=[120.803]
# k=0
# n_arr=[]
# for m in time_arr:
#     for n in a:
#         if m-n<=1 and m-n>=0:
#             k=k+1
#             n_arr.append(n)
# print(k)






Pseizureeegvar = 0.000277778;
Pnonseizureeegvar = 0.999722222;


# Pcombined = []
# for m in range(len(pro_circadian_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_RRIvars_time[m]*pro_RRIautos_time[m]*pro_circadian_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
#     Pcombined.append(P1/(P1+P2))
# print(len(Pcombined))

Pcombined = []
for m in range(len(pro_eegvars_time)):
    P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_circadian_time[m]
    P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_circadian_time_false[m])
    Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_circadian_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_circadian_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_circadian_time_false[m])
#     Pcombined.append(P1/(P1+P2))
# print(len(Pcombined))

# Pcombined = []
# for m in range(len(pro_circadian_time)):
#     P1=Pseizureeegvar*pro_RRIvars_time[m]*pro_RRIautos_time[m]*pro_circadian_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
#     Pcombined.append(P1/(P1+P2))
# print(len(Pcombined))


# Pcombined = []
# for m in range(len(pro_circadian_time)):
#     P1=Pseizureeegvar*pro_RRIvars_time[m]*pro_circadian_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m]*pro_circadian_time_false[m])
#     Pcombined.append(P1/(P1+P2))
# print(len(Pcombined))

# Pcombined = []
# for m in range(len(pro_circadian_time)):
#     P1=Pseizureeegvar*pro_RRIautos_time[m]*pro_circadian_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
#     Pcombined.append(P1/(P1+P2))
# print(len(Pcombined))

pyplot.figure(figsize=(8,4))
RRI_timewindow_arr=t[0:len(pro_circadian_time)]
pyplot.plot(RRI_timewindow_arr,Pcombined)
pyplot.annotate('',xy=(120.803,np.max(Pcombined)),xytext=(120.803,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='r',shrink=0.05))
pyplot.title('Forecast seizures in VIC1012')
pyplot.xlabel('Time(h)')
pyplot.ylabel('Seizure probability')
pyplot.show()


Pcombined=split(Pcombined,6)
print(len(Pcombined))
index=[]
for i in range(len(Pcombined)):
    for item in Pcombined[i]:
        if item >= 1.3473076871688326e-05:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[120.803]
k=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k=k+1
            n_arr.append(n)
print(k)
index=[]
for i in range(len(Pcombined)):
    for item in Pcombined[i]:
        if item >= 0.3*1.3473076871688326e-05:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[120.803]
k=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k=k+1
            n_arr.append(n)
print(k)
index=[]
for i in range(len(Pcombined)):
    for item in Pcombined[i]:
        if item >= 0.6*1.3473076871688326e-05:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[120.803]
k=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k=k+1
            n_arr.append(n)
print(k)
index=[]
for i in range(len(Pcombined)):
    for item in Pcombined[i]:
        if item >= 1.2*1.3473076871688326e-05:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[120.803]
k=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k=k+1
            n_arr.append(n)
print(k)
index=[]
for i in range(len(Pcombined)):
    for item in Pcombined[i]:
        if item >= 2*1.3473076871688326e-05:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[120.803]
k=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k=k+1
            n_arr.append(n)
print(k)