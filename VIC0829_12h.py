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






csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC0829/Cz_EEGvariance_VIC0829_15s_3h.csv', sep=',', header=None)
Raw_variance_EEG = csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC0829/Cz_EEGauto_VIC0829_15s_3h.csv', sep=',', header=None)
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
print(t_window_arr[19450]);
print(t_window_arr[19450] - t_window_arr[0]);
print(t_window_arr[-1] - t_window_arr[19450]);

window_time_arr = t_window_arr
# pyplot.plot(window_time_arr,Raw_variance_EEG_arr,'grey',alpha=0.5)
# pyplot.ylabel('Voltage',fontsize=13)
# pyplot.xlabel('Time (hours)',fontsize=13)
# pyplot.title('raw EEG variance in VIC0829',fontsize=13)
# pyplot.show()
var_arr = [0]
for item in Raw_variance_EEG_arr:
    if item < 1e-8:
        var_arr.append(item)
    else:
        var_arr.append(var_arr[-1])
Raw_variance_EEG = var_arr
# pyplot.plot(window_time_arr,Raw_variance_EEG,'grey',alpha=0.5)
# pyplot.ylabel('Voltage',fontsize=13)
# pyplot.xlabel('Time (hours)',fontsize=13)
# pyplot.title('EEG variance in VIC0829',fontsize=13)
# pyplot.show()


seizure_timing_index = []
for k in range(len(window_time_arr)):
    # if window_time_arr[k] < 2.4472 and window_time_arr[k + 1] >= 2.4472:
    #     seizure_timing_index.append(k)
    if window_time_arr[k] < 28.64 and window_time_arr[k + 1] >= 28.64:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 43.1678 and window_time_arr[k + 1] >= 43.1678:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 68.875 and window_time_arr[k + 1] >= 68.875:
        seizure_timing_index.append(k)
    # if window_time_arr[k] < 139.3575 and window_time_arr[k + 1] >= 139.3575:
    #     seizure_timing_index.append(k)
    # if window_time_arr[k] < 145.3427 and window_time_arr[k + 1] >= 145.3427:
    #     seizure_timing_index.append(k)
    # if window_time_arr[k] < 152.11 and window_time_arr[k + 1] >= 152.11:
    #     seizure_timing_index.append(k)
    # if window_time_arr[k] < 160.5378 and window_time_arr[k + 1] >= 160.5378:
    #     seizure_timing_index.append(k)
print(seizure_timing_index)

# # #
# # # # # # ### EEG variance
window_time_arr=t_window_arr[0:19450]
Raw_variance_EEG=Raw_variance_EEG[0:19450]
# window_time_arr=t_window_arr
# Raw_variance_EEG=Raw_variance_EEG


long_rhythm_var_arr=movingaverage(Raw_variance_EEG,5760)
medium_rhythm_var_arr=movingaverage(Raw_variance_EEG,240)
medium_rhythm_var_arr_2=movingaverage(Raw_variance_EEG,240*3)
medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG,240*6)
medium_rhythm_var_arr_4=movingaverage(Raw_variance_EEG,240*12)
short_rhythm_var_arr_plot=movingaverage(Raw_variance_EEG,240*24)


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
n=0
for item in seizure_phase_var_long:
    if item <0:
        n=n+1
print(n/len(seizure_phase_var_long))

bins_number = 18
bins = np.linspace(-np.pi, np.pi, bins_number + 1)
nEEGsvar, _, _ = pyplot.hist(phase_long_EEGvariance_arr, bins)
nEEGsvarsei, _, _ = pyplot.hist(seizure_phase_var_long, bins)
print(nEEGsvar)
print(nEEGsvarsei)
# width = 2*np.pi / bins_number
# params = dict(projection='polar')
# fig, ax2 = pyplot.subplots(subplot_kw=params)
# ax2.bar(bins[:bins_number], nEEGsvarsei/sum(nEEGsvarsei),width=width, color='grey',alpha=0.7,linewidth=2,edgecolor='k')
# locs, labels = pyplot.yticks([0.1,0.3,0.5],['0.1','0.3','0.5'],fontsize=16)
# ax2.set_title('EEG variance',fontsize=16)
# ax2.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',],fontsize=16)
# pyplot.show()




# pyplot.plot(t_window_arr,Raw_auto_EEG_arr,'grey',alpha=0.5)
# pyplot.xlabel('Time (hours)',fontsize=13)
# pyplot.title('raw EEG autocorrelation in VIC0829',fontsize=13)
# pyplot.show()
value_arr=[0]
for item in Raw_auto_EEG_arr:
    if item<500:
        value_arr.append(item)
    else:
        value_arr.append(value_arr[-1])
Raw_auto_EEG_arr=value_arr[1:]
# pyplot.plot(t_window_arr,Raw_auto_EEG_arr,'grey',alpha=0.5)
# pyplot.xlabel('Time (hours)',fontsize=13)
# pyplot.title('EEG autocorrelation in VIC0829',fontsize=13)
# pyplot.show()


Raw_auto_EEG=Raw_auto_EEG_arr[0:19450]
window_time_arr=t_window_arr[0:19450]
# Raw_auto_EEG=Raw_auto_EEG_arr
# window_time_arr=t_window_arr

long_rhythm_value_arr=movingaverage(Raw_auto_EEG,5760)
medium_rhythm_value_arr=movingaverage(Raw_auto_EEG,240)
medium_rhythm_value_arr_2=movingaverage(Raw_auto_EEG,240*3)
medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG,240*6)
medium_rhythm_value_arr_4=movingaverage(Raw_auto_EEG,240*12)
short_rhythm_value_arr_plot=movingaverage(Raw_auto_EEG,240*24)


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
n=0
for item in seizure_phase_value_long:
    if item <0:
        n=n+1
print(n/len(seizure_phase_value_long))

bins_number = 18
bins = np.linspace(-np.pi, np.pi, bins_number + 1)
nEEGsauto, _, _ = pyplot.hist(phase_long_EEGauto_arr, bins)
nEEGsautosei, _, _ = pyplot.hist(seizure_phase_value_long, bins)
print(nEEGsauto)
print(nEEGsautosei)
# width = 2*np.pi / bins_number
# params = dict(projection='polar')
# fig, ax2 = pyplot.subplots(subplot_kw=params)
# ax2.bar(bins[:bins_number], nEEGsautosei/sum(nEEGsautosei),width=width, color='grey',alpha=0.7,linewidth=2,edgecolor='k')
# locs, labels = pyplot.yticks([0.1,0.3,0.5],['0.1','0.3','0.5'],fontsize=16)
# ax2.set_title('EEG autocorrelation',fontsize=16)
# ax2.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',],fontsize=16)
# pyplot.show()







# # ### ECG data
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC0829/RRI_ch31_timewindowarr_VIC0829_15s_3h.csv', sep=',',
                         header=None)
rri_t = csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC0829/RRI_ch31_rawvariance_VIC0829_15s_3h.csv', sep=',',
                         header=None)
RRI_var = csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC0829/RRI_ch31_rawauto_VIC0829_15s_3h.csv', sep=',',
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
# pyplot.title('RRI variance in VIC0829',fontsize=13)
# pyplot.show()
#
# pyplot.plot(Raw_auto_RRI31_arr,'grey',alpha=0.5)
# pyplot.xlabel('Time (hours)',fontsize=13)
# pyplot.title('RRI autocorrelation in VIC0829',fontsize=13)
# pyplot.show()



# window_time_arr=t_window_arr
# Raw_variance_RRI31=Raw_variance_RRI31_arr
window_time_arr = t_window_arr[0:19450]
Raw_variance_RRI31 = Raw_variance_RRI31_arr[0:19450]

long_rhythm_var_arr = movingaverage(Raw_variance_RRI31, 5760)
medium_rhythm_var_arr = movingaverage(Raw_variance_RRI31, 240)
medium_rhythm_var_arr_2 = movingaverage(Raw_variance_RRI31, 240 * 3)
medium_rhythm_var_arr_3 = movingaverage(Raw_variance_RRI31, 240 * 6)
medium_rhythm_var_arr_4 = movingaverage(Raw_variance_RRI31, 240 * 12)
short_rhythm_var_arr_plot = movingaverage(Raw_variance_RRI31, 240*24)


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
n = 0
for item in seizure_phase_var_long:
    if item < 0:
        n = n + 1
print(n / len(seizure_phase_var_long))

bins_number = 18
bins = np.linspace(-np.pi, np.pi, bins_number + 1)
nRRIsvar, _, _ = pyplot.hist(phase_whole_long, bins)
nRRIsvarsei, _, _ = pyplot.hist(seizure_phase_var_long, bins)
print(nRRIsvar)
print(nRRIsvarsei)
# params = dict(projection='polar')
# fig, ax2 = pyplot.subplots(subplot_kw=params)
# ax2.bar(bins[:bins_number], nRRIsvarsei/sum(nRRIsvarsei),width=width, color='grey',alpha=0.7,edgecolor='k',linewidth=2)
# ax2.set_title('RRI variance',fontsize=16)
# locs, labels = pyplot.yticks([0.1,0.3,0.5],['0.1','0.3','0.5'],fontsize=16)
# # ax2.set_rlim([0,0.002])
# ax2.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',],fontsize=16)
# pyplot.show()




# Raw_auto_RRI31=Raw_auto_RRI31_arr
Raw_auto_RRI31 = Raw_auto_RRI31_arr[0:19450]

long_rhythm_value_arr = movingaverage(Raw_auto_RRI31, 5760)
medium_rhythm_value_arr = movingaverage(Raw_auto_RRI31, 240)
medium_rhythm_value_arr_2 = movingaverage(Raw_auto_RRI31, 240 * 3)
medium_rhythm_value_arr_3 = movingaverage(Raw_auto_RRI31, 240 * 6)
medium_rhythm_value_arr_4 = movingaverage(Raw_auto_RRI31, 240 * 12)
short_rhythm_value_arr_plot = movingaverage(Raw_auto_RRI31, 240*24)


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
n=0
for item in seizure_phase_value_long:
    if item <0:
        n=n+1
print(n/len(seizure_phase_value_long))

bins_number = 18
bins = np.linspace(-np.pi, np.pi, bins_number + 1)
nRRIsauto, _, _ = pyplot.hist(phase_whole_value_long, bins)
nRRIsautosei, _, _ = pyplot.hist(seizure_phase_value_long, bins)
print(nRRIsauto)
print(nRRIsautosei)
# width = 2*np.pi / bins_number
# params = dict(projection='polar')
# fig, ax2 = pyplot.subplots(subplot_kw=params)
# ax2.bar(bins[:bins_number], nRRIsautosei/sum(nRRIsautosei),width=width, color='grey',alpha=0.7,edgecolor='k',linewidth=2)
# ax2.set_title('RRI autocorrelation',fontsize=16)
# locs, labels = pyplot.yticks([0.1,0.3,0.5],['0.1','0.3','0.5'],fontsize=16)
# ax2.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',],fontsize=16)
# pyplot.show()







# t=np.linspace(0+0.00416667,0+0.00416667+0.00416667*(len(Raw_variance_EEG_arr)-1),len(Raw_variance_EEG_arr))
# window_time_arr=t
# a=np.where(t<10.4536+0)
# print(a);print(t[2267]);print(t[2508])
# t[0:2508]=t[0:2508]-0+13.5463889
# t[2508:]=t[2508:]-10.4536-0
# print(t[2508]);print(t);print(type(t));
#
# time_feature_arr=[]
# for i in range(len(t)):
#     if t[i]>24:
#         time_feature_arr.append(t[i] - (t[i] // 24) * 24)
#     else:
#         time_feature_arr.append(t[i])
# seizure_time=[time_feature_arr[586],time_feature_arr[6872],time_feature_arr[10359],time_feature_arr[16528],
# ]
# # seizure_time=[time_feature_arr[6872],time_feature_arr[10359],time_feature_arr[16528],
# # ]
# print(seizure_time)
# bins_number = 18
# bins = np.linspace(0, 24, bins_number + 1)
# nEEGsvar, _, _ = pyplot.hist(time_feature_arr[0:19450], bins)
# nEEGsvarsei, _, _ = pyplot.hist(seizure_time, bins)
#
# # bins = np.linspace(0, 2*np.pi, bins_number + 1)
# # width = 2*np.pi / bins_number
# # params = dict(projection='polar')
# # fig, ax = pyplot.subplots(subplot_kw=params)
# # ax.bar(bins[:bins_number], nEEGsvarsei/sum(nEEGsvarsei),width=width, color='grey',alpha=0.7,edgecolor='k',linewidth=2)
# # pyplot.setp(ax.get_yticklabels(), color='k')
# # # ax.set_title('seizure timing histogram (SA0124)',fontsize=23)
# # ax.set_xticks(np.linspace(0, 2*np.pi, 24, endpoint=False))
# # ax.set_xticklabels(['0 am','','','Night','','','6 am','','','Morning','','','12 am','','','Afternoon','','','18 pm','','','Evening','','','24 pm'],fontsize=16)
# # # locs, labels = pyplot.xticks([0,3,6,9,12,15,18,21,24],['0','Night','6','Morning','12','Afternoon','18','Evening','24'],fontsize=16)
# # locs, labels = pyplot.yticks([0.1,0.2,0.3],['0.1','0.2','0.3'],fontsize=16)
# # pyplot.show()
# # bins = np.linspace(0, 2*np.pi, bins_number + 1)
# # width = 2*np.pi / bins_number
# # params = dict(projection='polar')
# # fig, ax = pyplot.subplots(subplot_kw=params)
# # ax.bar(bins[:bins_number], nEEGsvarsei/sum(nEEGsvarsei),width=width, color='grey',alpha=0.7,edgecolor='k',linewidth=2)
# # pyplot.setp(ax.get_yticklabels(), color='k')
# # # ax.set_title('seizure timing histogram (SA0124)',fontsize=23)
# # ax.set_xticks(np.linspace(0, 2*np.pi, 24, endpoint=False))
# # ax.set_xticklabels(['0 am','','','Night','','','6 am','','','Morning','','','12 am','','','Afternoon','','','18 pm','','','Evening','','','24 pm'],fontsize=16)
# # # locs, labels = pyplot.xticks([0,3,6,9,12,15,18,21,24],['0','Night','6','Morning','12','Afternoon','18','Evening','24'],fontsize=16)
# # locs, labels = pyplot.yticks([0.1,0.3,0.5],['0.1','0.3','0.5'],fontsize=16)
# # pyplot.show()
# # bins_number = 18
# # bins = np.linspace(0, 24, bins_number + 1)
# # ntimes, _, _ = pyplot.hist(time_feature_arr[0:19450], bins)
# # ntimesei, _, _ = pyplot.hist(seizure_time, bins)
# # print(ntimes)
# # print(ntimesei)
#
#
#
#
#
#
#
#
#
# #### section 2 training training
medium_rhythm_var_arr_3 = movingaverage(Raw_variance_EEG, 240 * 6)
long_rhythm_var_arr = medium_rhythm_var_arr_3
var_trans = hilbert(long_rhythm_var_arr)
var_phase = np.angle(var_trans)
phase_long_EEGvariance_arr = var_phase
print(len(phase_long_EEGvariance_arr));
medium_rhythm_value_arr_3 = movingaverage(Raw_auto_EEG, 240 * 6)
long_rhythm_value_arr = medium_rhythm_value_arr_3
value_trans = hilbert(long_rhythm_value_arr)
value_phase = np.angle(value_trans)
phase_long_EEGauto_arr = value_phase
print(len(phase_long_EEGauto_arr));
medium_rhythm_RRIvar_arr_3 = movingaverage(Raw_variance_RRI31, 240 * 6)
long_rhythm_RRIvar_arr = medium_rhythm_RRIvar_arr_3
var_trans = hilbert(long_rhythm_RRIvar_arr)
var_phase = np.angle(var_trans)
phase_long_RRIvariance_arr = var_phase
print(len(phase_long_RRIvariance_arr));
medium_rhythm_RRIvalue_arr_3 = movingaverage(Raw_auto_RRI31, 240 * 6)
long_rhythm_RRIvalue_arr = medium_rhythm_RRIvalue_arr_3
value_trans = hilbert(long_rhythm_RRIvalue_arr)
value_phase = np.angle(value_trans)
phase_long_RRIauto_arr = value_phase
print(len(phase_long_RRIauto_arr));



### combined probability calculation
# ##### 24 h 24 h 24 h
# bins_number = 18
# bins = np.linspace(-np.pi, np.pi, bins_number + 1)
# pro_eegvars_time = []
# pro_eegvars_time_false = []
# for i in range(len(phase_long_EEGvariance_arr)):
#     if phase_long_EEGvariance_arr[i] >= bins[0] and phase_long_EEGvariance_arr[i] < bins[1]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[1] and phase_long_EEGvariance_arr[i] < bins[2]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[2] and phase_long_EEGvariance_arr[i] < bins[3]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[3] and phase_long_EEGvariance_arr[i] < bins[4]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[4] and phase_long_EEGvariance_arr[i] < bins[5]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[5] and phase_long_EEGvariance_arr[i] < bins[6]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[6] and phase_long_EEGvariance_arr[i] <= bins[7]:
#         pro_eegvars_time_false.append(0.000102844)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] > bins[7] and phase_long_EEGvariance_arr[i] < bins[8]:
#         pro_eegvars_time_false.append(0.130097187)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[8] and phase_long_EEGvariance_arr[i] <= bins[9]:
#         pro_eegvars_time_false.append(0.471795135)
#         pro_eegvars_time.append(0.666666667)
#     elif phase_long_EEGvariance_arr[i] > bins[9] and phase_long_EEGvariance_arr[i] < bins[10]:
#         pro_eegvars_time_false.append(0.27850054)
#         pro_eegvars_time.append(0.333333333)
#     elif phase_long_EEGvariance_arr[i] >= bins[10] and phase_long_EEGvariance_arr[i] <= bins[11]:
#         pro_eegvars_time_false.append(0.060266365)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] > bins[11] and phase_long_EEGvariance_arr[i] < bins[12]:
#         pro_eegvars_time_false.append(0.030236026)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] > bins[12] and phase_long_EEGvariance_arr[i] < bins[13]:
#         pro_eegvars_time_false.append(0.021237209)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[13] and phase_long_EEGvariance_arr[i] < bins[14]:
#         pro_eegvars_time_false.append(0.007764694)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[14] and phase_long_EEGvariance_arr[i] < bins[15]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[15] and phase_long_EEGvariance_arr[i] < bins[16]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[16] and phase_long_EEGvariance_arr[i] < bins[17]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[17]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
# # print(pro_eegvars_time[586]);
# print(pro_eegvars_time[6872]);print(pro_eegvars_time[10359]);print(pro_eegvars_time[16528]);
# pro_eegautos_time = []
# pro_eegautos_time_false = []
# for i in range(len(phase_long_EEGauto_arr)):
#     if phase_long_EEGauto_arr[i] >= bins[0] and phase_long_EEGauto_arr[i] < bins[1]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[1] and phase_long_EEGauto_arr[i] < bins[2]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[2] and phase_long_EEGauto_arr[i] < bins[3]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[3] and phase_long_EEGauto_arr[i] < bins[4]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[4] and phase_long_EEGauto_arr[i] < bins[5]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[5] and phase_long_EEGauto_arr[i] < bins[6]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[6] and phase_long_EEGauto_arr[i] < bins[7]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[7] and phase_long_EEGauto_arr[i] < bins[8]:
#         pro_eegautos_time_false.append(0.17894791)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[8] and phase_long_EEGauto_arr[i] < bins[9]:
#         pro_eegautos_time_false.append(0.339178279)
#         pro_eegautos_time.append(0.666666667)
#     elif phase_long_EEGauto_arr[i] >= bins[9] and phase_long_EEGauto_arr[i] <= bins[10]:
#         pro_eegautos_time_false.append(0.386332082)
#         pro_eegautos_time.append(0.333333333)
#     elif phase_long_EEGauto_arr[i] > bins[10] and phase_long_EEGauto_arr[i] < bins[11]:
#         pro_eegautos_time_false.append(0.055535558)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[11] and phase_long_EEGauto_arr[i] < bins[12]:
#         pro_eegautos_time_false.append(0.013163984)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[12] and phase_long_EEGauto_arr[i] < bins[13]:
#         pro_eegautos_time_false.append(0.010335784)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[13] and phase_long_EEGauto_arr[i] < bins[14]:
#         pro_eegautos_time_false.append(0.016506402)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[14] and phase_long_EEGauto_arr[i] < bins[15]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[15] and phase_long_EEGauto_arr[i] < bins[16]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[16] and phase_long_EEGauto_arr[i] < bins[17]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[17]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
# # print(pro_eegautos_time[586]);
# print(pro_eegautos_time[6872]);print(pro_eegautos_time[10359]);print(pro_eegautos_time[16528]);
# pro_RRIautos_time = []
# pro_RRIautos_time_false = []
# for i in range(len(phase_long_RRIauto_arr)):
#     if phase_long_RRIauto_arr[i] >= bins[0] and phase_long_RRIauto_arr[i] < bins[1]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[1] and phase_long_RRIauto_arr[i] < bins[2]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[2] and phase_long_RRIauto_arr[i] < bins[3]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[3] and phase_long_RRIauto_arr[i] < bins[4]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[4] and phase_long_RRIauto_arr[i] < bins[5]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[5] and phase_long_RRIauto_arr[i] < bins[6]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[6] and phase_long_RRIauto_arr[i] < bins[7]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[7] and phase_long_RRIauto_arr[i] <= bins[8]:
#         pro_RRIautos_time_false.append(0.171851699)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] > bins[8] and phase_long_RRIauto_arr[i] <= bins[9]:
#         pro_RRIautos_time_false.append(0.405769527)
#         pro_RRIautos_time.append(0.666666667)
#     elif phase_long_RRIauto_arr[i] > bins[9] and phase_long_RRIauto_arr[i] <= bins[10]:
#         pro_RRIautos_time_false.append(0.316964056)
#         pro_RRIautos_time.append(0.333333333)
#     elif phase_long_RRIauto_arr[i] > bins[10] and phase_long_RRIauto_arr[i] < bins[11]:
#         pro_RRIautos_time_false.append(0.049879159)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[11] and phase_long_RRIauto_arr[i] < bins[12]:
#         pro_RRIautos_time_false.append(0.017174886)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[12] and phase_long_RRIauto_arr[i] < bins[13]:
#         pro_RRIautos_time_false.append(0.014346686)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[13] and phase_long_RRIauto_arr[i] < bins[14]:
#         pro_RRIautos_time_false.append(0.024013987)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[14] and phase_long_RRIauto_arr[i] < bins[15]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[15] and phase_long_RRIauto_arr[i] < bins[16]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[16] and phase_long_RRIauto_arr[i] < bins[17]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[17]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
# # print(pro_RRIautos_time[586]);
# print(pro_RRIautos_time[6872]);print(pro_RRIautos_time[10359]);print(pro_RRIautos_time[16528]);



###### 12 h 12 h 12 h
# #### combined probability calculation
# bins_number = 18
# bins = np.linspace(-np.pi, np.pi, bins_number + 1)
# pro_eegvars_time = []
# pro_eegvars_time_false = []
# for i in range(len(phase_long_EEGvariance_arr)):
#     if phase_long_EEGvariance_arr[i] >= bins[0] and phase_long_EEGvariance_arr[i] < bins[1]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[1] and phase_long_EEGvariance_arr[i] < bins[2]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[2] and phase_long_EEGvariance_arr[i] < bins[3]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[3] and phase_long_EEGvariance_arr[i] < bins[4]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[4] and phase_long_EEGvariance_arr[i] < bins[5]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[5] and phase_long_EEGvariance_arr[i] < bins[6]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[6] and phase_long_EEGvariance_arr[i] <= bins[7]:
#         pro_eegvars_time_false.append(0.047102381)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] > bins[7] and phase_long_EEGvariance_arr[i] < bins[8]:
#         pro_eegvars_time_false.append(0.271352908)
#         pro_eegvars_time.append(0.333333333)
#     elif phase_long_EEGvariance_arr[i] >= bins[8] and phase_long_EEGvariance_arr[i] <= bins[9]:
#         pro_eegvars_time_false.append(0.147580604)
#         pro_eegvars_time.append(0.333333333)
#     elif phase_long_EEGvariance_arr[i] > bins[9] and phase_long_EEGvariance_arr[i] < bins[10]:
#         pro_eegvars_time_false.append(0.29778372)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[10] and phase_long_EEGvariance_arr[i] <= bins[11]:
#         pro_eegvars_time_false.append(0.163881318)
#         pro_eegvars_time.append(0.333333333)
#     elif phase_long_EEGvariance_arr[i] > bins[11] and phase_long_EEGvariance_arr[i] < bins[12]:
#         pro_eegvars_time_false.append(0.052501671)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] > bins[12] and phase_long_EEGvariance_arr[i] < bins[13]:
#         pro_eegvars_time_false.append(0.013678202)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[13] and phase_long_EEGvariance_arr[i] < bins[14]:
#         pro_eegvars_time_false.append(0.006119196)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[14] and phase_long_EEGvariance_arr[i] < bins[15]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[15] and phase_long_EEGvariance_arr[i] < bins[16]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[16] and phase_long_EEGvariance_arr[i] < bins[17]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[17]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
# # print(pro_eegvars_time[586]);
# print(pro_eegvars_time[6872]);print(pro_eegvars_time[10359]);print(pro_eegvars_time[16528]);
# pro_eegautos_time = []
# pro_eegautos_time_false = []
# for i in range(len(phase_long_EEGauto_arr)):
#     if phase_long_EEGauto_arr[i] >= bins[0] and phase_long_EEGauto_arr[i] < bins[1]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[1] and phase_long_EEGauto_arr[i] < bins[2]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[2] and phase_long_EEGauto_arr[i] < bins[3]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[3] and phase_long_EEGauto_arr[i] < bins[4]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[4] and phase_long_EEGauto_arr[i] < bins[5]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[5] and phase_long_EEGauto_arr[i] < bins[6]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[6] and phase_long_EEGauto_arr[i] < bins[7]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[7] and phase_long_EEGauto_arr[i] < bins[8]:
#         pro_eegautos_time_false.append(0.11770453)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[8] and phase_long_EEGauto_arr[i] < bins[9]:
#         pro_eegautos_time_false.append(0.326117139)
#         pro_eegautos_time.append(0.666666667)
#     elif phase_long_EEGauto_arr[i] >= bins[9] and phase_long_EEGauto_arr[i] <= bins[10]:
#         pro_eegautos_time_false.append(0.528616239)
#         pro_eegautos_time.append(0.333333333)
#     elif phase_long_EEGauto_arr[i] > bins[10] and phase_long_EEGauto_arr[i] < bins[11]:
#         pro_eegautos_time_false.append(0.011209955)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[11] and phase_long_EEGauto_arr[i] < bins[12]:
#         pro_eegautos_time_false.append(0.003496683)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[12] and phase_long_EEGauto_arr[i] < bins[13]:
#         pro_eegautos_time_false.append(0.003239574)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[13] and phase_long_EEGauto_arr[i] < bins[14]:
#         pro_eegautos_time_false.append(0.009615879)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[14] and phase_long_EEGauto_arr[i] < bins[15]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[15] and phase_long_EEGauto_arr[i] < bins[16]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[16] and phase_long_EEGauto_arr[i] < bins[17]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[17]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
# # print(pro_eegautos_time[586]);
# print(pro_eegautos_time[6872]);print(pro_eegautos_time[10359]);print(pro_eegautos_time[16528]);
# pro_RRIautos_time = []
# pro_RRIautos_time_false = []
# for i in range(len(phase_long_RRIauto_arr)):
#     if phase_long_RRIauto_arr[i] >= bins[0] and phase_long_RRIauto_arr[i] < bins[1]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[1] and phase_long_RRIauto_arr[i] < bins[2]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[2] and phase_long_RRIauto_arr[i] < bins[3]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[3] and phase_long_RRIauto_arr[i] < bins[4]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[4] and phase_long_RRIauto_arr[i] < bins[5]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[5] and phase_long_RRIauto_arr[i] < bins[6]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[6] and phase_long_RRIauto_arr[i] < bins[7]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[7] and phase_long_RRIauto_arr[i] <= bins[8]:
#         pro_RRIautos_time_false.append(0.081452152)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] > bins[8] and phase_long_RRIauto_arr[i] <= bins[9]:
#         pro_RRIautos_time_false.append(0.504293721)
#         pro_RRIautos_time.append(0.666666667)
#     elif phase_long_RRIauto_arr[i] > bins[9] and phase_long_RRIauto_arr[i] <= bins[10]:
#         pro_RRIautos_time_false.append(0.355427572)
#         pro_RRIautos_time.append(0.333333333)
#     elif phase_long_RRIauto_arr[i] > bins[10] and phase_long_RRIauto_arr[i] < bins[11]:
#         pro_RRIautos_time_false.append(0.029104746)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[11] and phase_long_RRIauto_arr[i] < bins[12]:
#         pro_RRIautos_time_false.append(0.008433177)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[12] and phase_long_RRIauto_arr[i] < bins[13]:
#         pro_RRIautos_time_false.append(0.007044788)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[13] and phase_long_RRIauto_arr[i] < bins[14]:
#         pro_RRIautos_time_false.append(0.014243842)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[14] and phase_long_RRIauto_arr[i] < bins[15]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[15] and phase_long_RRIauto_arr[i] < bins[16]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[16] and phase_long_RRIauto_arr[i] < bins[17]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[17]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
# # print(pro_RRIautos_time[586]);
# print(pro_RRIautos_time[6872]);print(pro_RRIautos_time[10359]);print(pro_RRIautos_time[16528]);



#### combined probability calculation 6h 6h 6h
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
        pro_eegvars_time_false.append(0.02581375)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] >= bins[6] and phase_long_EEGvariance_arr[i] <= bins[7]:
        pro_eegvars_time_false.append(0.183627295)
        pro_eegvars_time.append(0.333333333)
    elif phase_long_EEGvariance_arr[i] > bins[7] and phase_long_EEGvariance_arr[i] < bins[8]:
        pro_eegvars_time_false.append(0.172211652)
        pro_eegvars_time.append(0.333333333)
    elif phase_long_EEGvariance_arr[i] >= bins[8] and phase_long_EEGvariance_arr[i] <= bins[9]:
        pro_eegvars_time_false.append(0.155242454)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] > bins[9] and phase_long_EEGvariance_arr[i] < bins[10]:
        pro_eegvars_time_false.append(0.107317324)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] >= bins[10] and phase_long_EEGvariance_arr[i] <= bins[11]:
        pro_eegvars_time_false.append(0.13158842)
        pro_eegvars_time.append(0.333333333)
    elif phase_long_EEGvariance_arr[i] > bins[11] and phase_long_EEGvariance_arr[i] < bins[12]:
        pro_eegvars_time_false.append(0.111945287)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] > bins[12] and phase_long_EEGvariance_arr[i] < bins[13]:
        pro_eegvars_time_false.append(0.106186044)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] >= bins[13] and phase_long_EEGvariance_arr[i] < bins[14]:
        pro_eegvars_time_false.append(0.006067774)
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
# print(pro_eegvars_time[586]);
print(pro_eegvars_time[6872]);print(pro_eegvars_time[10359]);print(pro_eegvars_time[16528]);
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
        pro_eegautos_time_false.append(0.02154574)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[7] and phase_long_EEGauto_arr[i] < bins[8]:
        pro_eegautos_time_false.append(0.05049622)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[8] and phase_long_EEGauto_arr[i] < bins[9]:
        pro_eegautos_time_false.append(0.40443256)
        pro_eegautos_time.append(0.666666667)
    elif phase_long_EEGauto_arr[i] >= bins[9] and phase_long_EEGauto_arr[i] <= bins[10]:
        pro_eegautos_time_false.append(0.496169075)
        pro_eegautos_time.append(0.333333333)
    elif phase_long_EEGauto_arr[i] > bins[10] and phase_long_EEGauto_arr[i] < bins[11]:
        pro_eegautos_time_false.append(0.017894791)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[11] and phase_long_EEGauto_arr[i] < bins[12]:
        pro_eegautos_time_false.append(0.000977014)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[12] and phase_long_EEGauto_arr[i] < bins[13]:
        pro_eegautos_time_false.append(0.000308531)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[13] and phase_long_EEGauto_arr[i] < bins[14]:
        pro_eegautos_time_false.append(0.008176068)
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
# print(pro_eegautos_time[586]);
print(pro_eegautos_time[6872]);print(pro_eegautos_time[10359]);print(pro_eegautos_time[16528]);
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
        pro_RRIvars_time_false.append(0.020928678)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[5] and phase_long_RRIvariance_arr[i] < bins[6]:
        pro_RRIvars_time_false.append(0.147374916)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[6] and phase_long_RRIvariance_arr[i] < bins[7]:
        pro_RRIvars_time_false.append(0.110814007)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[7] and phase_long_RRIvariance_arr[i] <= bins[8]:
        pro_RRIvars_time_false.append(0.137913303)
        pro_RRIvars_time.append(0.333333333)
    elif phase_long_RRIvariance_arr[i] > bins[8] and phase_long_RRIvariance_arr[i] < bins[9]:
        pro_RRIvars_time_false.append(0.061500489)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[9] and phase_long_RRIvariance_arr[i] <= bins[10]:
        pro_RRIvars_time_false.append(0.091119453)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] > bins[10] and phase_long_RRIvariance_arr[i] <= bins[11]:
        pro_RRIvars_time_false.append(0.14860904)
        pro_RRIvars_time.append(0.666666667)
    elif phase_long_RRIvariance_arr[i] > bins[11] and phase_long_RRIvariance_arr[i] < bins[12]:
        pro_RRIvars_time_false.append(0.167635111)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[12] and phase_long_RRIvariance_arr[i] < bins[13]:
        pro_RRIvars_time_false.append(0.069727979)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[13] and phase_long_RRIvariance_arr[i] < bins[14]:
        pro_RRIvars_time_false.append(0.044377025)
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
# print(pro_RRIvars_time[586]);
print(pro_RRIvars_time[6872]);print(pro_RRIvars_time[10359]);print(pro_RRIvars_time[16528]);
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
        pro_RRIautos_time_false.append(0.05049622)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] > bins[8] and phase_long_RRIauto_arr[i] <= bins[9]:
        pro_RRIautos_time_false.append(0.408546305)
        pro_RRIautos_time.append(0.666666667)
    elif phase_long_RRIauto_arr[i] > bins[9] and phase_long_RRIauto_arr[i] <= bins[10]:
        pro_RRIautos_time_false.append(0.501105569)
        pro_RRIautos_time.append(0.333333333)
    elif phase_long_RRIauto_arr[i] > bins[10] and phase_long_RRIauto_arr[i] < bins[11]:
        pro_RRIautos_time_false.append(0.02298555)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[11] and phase_long_RRIauto_arr[i] < bins[12]:
        pro_RRIautos_time_false.append(0.004422276)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[12] and phase_long_RRIauto_arr[i] < bins[13]:
        pro_RRIautos_time_false.append(0.003136731)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[13] and phase_long_RRIauto_arr[i] < bins[14]:
        pro_RRIautos_time_false.append(0.009307348)
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
# print(pro_RRIautos_time[586]);
print(pro_RRIautos_time[6872]);print(pro_RRIautos_time[10359]);print(pro_RRIautos_time[16528]);

# # ###### 1 h
# # #### combined probability calculation
# # # bins_number = 18
# # # bins = np.linspace(-np.pi, np.pi, bins_number + 1)
# # # pro_eegautos_time = []
# # # pro_eegautos_time_false = []
# # # for i in range(len(phase_long_EEGauto_arr)):
# # #     if phase_long_EEGauto_arr[i] >= bins[0] and phase_long_EEGauto_arr[i] < bins[1]:
# # #         pro_eegautos_time_false.append(0)
# # #         pro_eegautos_time.append(0)
# # #     elif phase_long_EEGauto_arr[i] >= bins[1] and phase_long_EEGauto_arr[i] < bins[2]:
# # #         pro_eegautos_time_false.append(0)
# # #         pro_eegautos_time.append(0)
# # #     elif phase_long_EEGauto_arr[i] >= bins[2] and phase_long_EEGauto_arr[i] < bins[3]:
# # #         pro_eegautos_time_false.append(0)
# # #         pro_eegautos_time.append(0)
# # #     elif phase_long_EEGauto_arr[i] >= bins[3] and phase_long_EEGauto_arr[i] < bins[4]:
# # #         pro_eegautos_time_false.append(0)
# # #         pro_eegautos_time.append(0)
# # #     elif phase_long_EEGauto_arr[i] >= bins[4] and phase_long_EEGauto_arr[i] < bins[5]:
# # #         pro_eegautos_time_false.append(0.003496683)
# # #         pro_eegautos_time.append(0)
# # #     elif phase_long_EEGauto_arr[i] >= bins[5] and phase_long_EEGauto_arr[i] < bins[6]:
# # #         pro_eegautos_time_false.append(0.00169692)
# # #         pro_eegautos_time.append(0)
# # #     elif phase_long_EEGauto_arr[i] >= bins[6] and phase_long_EEGauto_arr[i] < bins[7]:
# # #         pro_eegautos_time_false.append(0.00992441)
# # #         pro_eegautos_time.append(0)
# # #     elif phase_long_EEGauto_arr[i] >= bins[7] and phase_long_EEGauto_arr[i] < bins[8]:
# # #         pro_eegautos_time_false.append(0.096210212)
# # #         pro_eegautos_time.append(0)
# # #     elif phase_long_EEGauto_arr[i] >= bins[8] and phase_long_EEGauto_arr[i] < bins[9]:
# # #         pro_eegautos_time_false.append(0.388491798)
# # #         pro_eegautos_time.append(0.666666667)
# # #     elif phase_long_EEGauto_arr[i] >= bins[9] and phase_long_EEGauto_arr[i] <= bins[10]:
# # #         pro_eegautos_time_false.append(0.398724739)
# # #         pro_eegautos_time.append(0.333333333)
# # #     elif phase_long_EEGauto_arr[i] > bins[10] and phase_long_EEGauto_arr[i] < bins[11]:
# # #         pro_eegautos_time_false.append(0.080577981)
# # #         pro_eegautos_time.append(0)
# # #     elif phase_long_EEGauto_arr[i] >= bins[11] and phase_long_EEGauto_arr[i] < bins[12]:
# # #         pro_eegautos_time_false.append(0.011672752)
# # #         pro_eegautos_time.append(0)
# # #     elif phase_long_EEGauto_arr[i] >= bins[12] and phase_long_EEGauto_arr[i] < bins[13]:
# # #         pro_eegautos_time_false.append(0.004010901)
# # #         pro_eegautos_time.append(0)
# # #     elif phase_long_EEGauto_arr[i] >= bins[13] and phase_long_EEGauto_arr[i] < bins[14]:
# # #         pro_eegautos_time_false.append(0.005193603)
# # #         pro_eegautos_time.append(0)
# # #     elif phase_long_EEGauto_arr[i] >= bins[14] and phase_long_EEGauto_arr[i] < bins[15]:
# # #         pro_eegautos_time_false.append(0)
# # #         pro_eegautos_time.append(0)
# # #     elif phase_long_EEGauto_arr[i] >= bins[15] and phase_long_EEGauto_arr[i] < bins[16]:
# # #         pro_eegautos_time_false.append(0)
# # #         pro_eegautos_time.append(0)
# # #     elif phase_long_EEGauto_arr[i] >= bins[16] and phase_long_EEGauto_arr[i] < bins[17]:
# # #         pro_eegautos_time_false.append(0)
# # #         pro_eegautos_time.append(0)
# # #     elif phase_long_EEGauto_arr[i] >= bins[17]:
# # #         pro_eegautos_time_false.append(0)
# # #         pro_eegautos_time.append(0)
# # # print(pro_eegautos_time[586]);print(pro_eegautos_time[6872]);
# # # print(pro_eegautos_time[10359]);print(pro_eegautos_time[16528]);
# # # pro_RRIautos_time = []
# # # pro_RRIautos_time_false = []
# # # for i in range(len(phase_long_RRIauto_arr)):
# # #     if phase_long_RRIauto_arr[i] >= bins[0] and phase_long_RRIauto_arr[i] < bins[1]:
# # #         pro_RRIautos_time_false.append(0)
# # #         pro_RRIautos_time.append(0)
# # #     elif phase_long_RRIauto_arr[i] >= bins[1] and phase_long_RRIauto_arr[i] < bins[2]:
# # #         pro_RRIautos_time_false.append(0)
# # #         pro_RRIautos_time.append(0)
# # #     elif phase_long_RRIauto_arr[i] >= bins[2] and phase_long_RRIauto_arr[i] < bins[3]:
# # #         pro_RRIautos_time_false.append(0)
# # #         pro_RRIautos_time.append(0)
# # #     elif phase_long_RRIauto_arr[i] >= bins[3] and phase_long_RRIauto_arr[i] < bins[4]:
# # #         pro_RRIautos_time_false.append(0)
# # #         pro_RRIautos_time.append(0)
# # #     elif phase_long_RRIauto_arr[i] >= bins[4] and phase_long_RRIauto_arr[i] < bins[5]:
# # #         pro_RRIautos_time_false.append(0)
# # #         pro_RRIautos_time.append(0)
# # #     elif phase_long_RRIauto_arr[i] >= bins[5] and phase_long_RRIauto_arr[i] < bins[6]:
# # #         pro_RRIautos_time_false.append(0.000668484)
# # #         pro_RRIautos_time.append(0)
# # #     elif phase_long_RRIauto_arr[i] >= bins[6] and phase_long_RRIauto_arr[i] < bins[7]:
# # #         pro_RRIautos_time_false.append(0.007970381)
# # #         pro_RRIautos_time.append(0)
# # #     elif phase_long_RRIauto_arr[i] >= bins[7] and phase_long_RRIauto_arr[i] <= bins[8]:
# # #         pro_RRIautos_time_false.append(0.030955932)
# # #         pro_RRIautos_time.append(0.333333333)
# # #     elif phase_long_RRIauto_arr[i] > bins[8] and phase_long_RRIauto_arr[i] <= bins[9]:
# # #         pro_RRIautos_time_false.append(0.423047257)
# # #         pro_RRIautos_time.append(0.666666667)
# # #     elif phase_long_RRIauto_arr[i] > bins[9] and phase_long_RRIauto_arr[i] <= bins[10]:
# # #         pro_RRIautos_time_false.append(0.53005605)
# # #         pro_RRIautos_time.append(0)
# # #     elif phase_long_RRIauto_arr[i] > bins[10] and phase_long_RRIauto_arr[i] < bins[11]:
# # #         pro_RRIautos_time_false.append(0.002725356)
# # #         pro_RRIautos_time.append(0)
# # #     elif phase_long_RRIauto_arr[i] >= bins[11] and phase_long_RRIauto_arr[i] < bins[12]:
# # #         pro_RRIautos_time_false.append(0.001079858)
# # #         pro_RRIautos_time.append(0)
# # #     elif phase_long_RRIauto_arr[i] >= bins[12] and phase_long_RRIauto_arr[i] < bins[13]:
# # #         pro_RRIautos_time_false.append(0.001388389)
# # #         pro_RRIautos_time.append(0)
# # #     elif phase_long_RRIauto_arr[i] >= bins[13] and phase_long_RRIauto_arr[i] < bins[14]:
# # #         pro_RRIautos_time_false.append(0.002108294)
# # #         pro_RRIautos_time.append(0)
# # #     elif phase_long_RRIauto_arr[i] >= bins[14] and phase_long_RRIauto_arr[i] < bins[15]:
# # #         pro_RRIautos_time_false.append(0)
# # #         pro_RRIautos_time.append(0)
# # #     elif phase_long_RRIauto_arr[i] >= bins[15] and phase_long_RRIauto_arr[i] < bins[16]:
# # #         pro_RRIautos_time_false.append(0)
# # #         pro_RRIautos_time.append(0)
# # #     elif phase_long_RRIauto_arr[i] >= bins[16] and phase_long_RRIauto_arr[i] < bins[17]:
# # #         pro_RRIautos_time_false.append(0)
# # #         pro_RRIautos_time.append(0)
# # #     elif phase_long_RRIauto_arr[i] >= bins[17]:
# # #         pro_RRIautos_time_false.append(0)
# # #         pro_RRIautos_time.append(0)
# # # print(pro_RRIautos_time[586]);print(pro_RRIautos_time[6872]);print(pro_RRIautos_time[10359]);print(pro_RRIautos_time[16528]);
# #
# #
# #
# # ###### 5 min
# # #### combined probability calculation
# # bins_number = 18
# # bins = np.linspace(-np.pi, np.pi, bins_number + 1)
# # pro_eegvars_time = []
# # pro_eegvars_time_false = []
# # for i in range(len(phase_long_EEGvariance_arr)):
# #     if phase_long_EEGvariance_arr[i] >= bins[0] and phase_long_EEGvariance_arr[i] < bins[1]:
# #         pro_eegvars_time_false.append(0)
# #         pro_eegvars_time.append(0)
# #     elif phase_long_EEGvariance_arr[i] >= bins[1] and phase_long_EEGvariance_arr[i] < bins[2]:
# #         pro_eegvars_time_false.append(0)
# #         pro_eegvars_time.append(0)
# #     elif phase_long_EEGvariance_arr[i] >= bins[2] and phase_long_EEGvariance_arr[i] < bins[3]:
# #         pro_eegvars_time_false.append(0)
# #         pro_eegvars_time.append(0)
# #     elif phase_long_EEGvariance_arr[i] >= bins[3] and phase_long_EEGvariance_arr[i] < bins[4]:
# #         pro_eegvars_time_false.append(0)
# #         pro_eegvars_time.append(0)
# #     elif phase_long_EEGvariance_arr[i] >= bins[4] and phase_long_EEGvariance_arr[i] < bins[5]:
# #         pro_eegvars_time_false.append(0.138787474)
# #         pro_eegvars_time.append(0)
# #     elif phase_long_EEGvariance_arr[i] >= bins[5] and phase_long_EEGvariance_arr[i] < bins[6]:
# #         pro_eegvars_time_false.append(0.189386538)
# #         pro_eegvars_time.append(0.333333333)
# #     elif phase_long_EEGvariance_arr[i] >= bins[6] and phase_long_EEGvariance_arr[i] <= bins[7]:
# #         pro_eegvars_time_false.append(0.094770402)
# #         pro_eegvars_time.append(0)
# #     elif phase_long_EEGvariance_arr[i] > bins[7] and phase_long_EEGvariance_arr[i] < bins[8]:
# #         pro_eegvars_time_false.append(0.053375842)
# #         pro_eegvars_time.append(0)
# #     elif phase_long_EEGvariance_arr[i] >= bins[8] and phase_long_EEGvariance_arr[i] <= bins[9]:
# #         pro_eegvars_time_false.append(0.0419602)
# #         pro_eegvars_time.append(0.333333333)
# #     elif phase_long_EEGvariance_arr[i] > bins[9] and phase_long_EEGvariance_arr[i] < bins[10]:
# #         pro_eegvars_time_false.append(0.035789582)
# #         pro_eegvars_time.append(0)
# #     elif phase_long_EEGvariance_arr[i] >= bins[10] and phase_long_EEGvariance_arr[i] <= bins[11]:
# #         pro_eegvars_time_false.append(0.032807117)
# #         pro_eegvars_time.append(0)
# #     elif phase_long_EEGvariance_arr[i] > bins[11] and phase_long_EEGvariance_arr[i] < bins[12]:
# #         pro_eegvars_time_false.append(0.06947087)
# #         pro_eegvars_time.append(0)
# #     elif phase_long_EEGvariance_arr[i] > bins[12] and phase_long_EEGvariance_arr[i] < bins[13]:
# #         pro_eegvars_time_false.append(0.164395537)
# #         pro_eegvars_time.append(0.333333333)
# #     elif phase_long_EEGvariance_arr[i] >= bins[13] and phase_long_EEGvariance_arr[i] < bins[14]:
# #         pro_eegvars_time_false.append(0.179256441)
# #         pro_eegvars_time.append(0)
# #     elif phase_long_EEGvariance_arr[i] >= bins[14] and phase_long_EEGvariance_arr[i] < bins[15]:
# #         pro_eegvars_time_false.append(0)
# #         pro_eegvars_time.append(0)
# #     elif phase_long_EEGvariance_arr[i] >= bins[15] and phase_long_EEGvariance_arr[i] < bins[16]:
# #         pro_eegvars_time_false.append(0)
# #         pro_eegvars_time.append(0)
# #     elif phase_long_EEGvariance_arr[i] >= bins[16] and phase_long_EEGvariance_arr[i] < bins[17]:
# #         pro_eegvars_time_false.append(0)
# #         pro_eegvars_time.append(0)
# #     elif phase_long_EEGvariance_arr[i] >= bins[17]:
# #         pro_eegvars_time_false.append(0)
# #         pro_eegvars_time.append(0)
# # # print(pro_eegvars_time[586]);
# # print(pro_eegvars_time[6872]);print(pro_eegvars_time[10359]);print(pro_eegvars_time[16528]);
# # pro_eegautos_time = []
# # pro_eegautos_time_false = []
# # for i in range(len(phase_long_EEGauto_arr)):
# #     if phase_long_EEGauto_arr[i] >= bins[0] and phase_long_EEGauto_arr[i] < bins[1]:
# #         pro_eegautos_time_false.append(0)
# #         pro_eegautos_time.append(0)
# #     elif phase_long_EEGauto_arr[i] >= bins[1] and phase_long_EEGauto_arr[i] < bins[2]:
# #         pro_eegautos_time_false.append(0)
# #         pro_eegautos_time.append(0)
# #     elif phase_long_EEGauto_arr[i] >= bins[2] and phase_long_EEGauto_arr[i] < bins[3]:
# #         pro_eegautos_time_false.append(0)
# #         pro_eegautos_time.append(0)
# #     elif phase_long_EEGauto_arr[i] >= bins[3] and phase_long_EEGauto_arr[i] < bins[4]:
# #         pro_eegautos_time_false.append(0)
# #         pro_eegautos_time.append(0)
# #     elif phase_long_EEGauto_arr[i] >= bins[4] and phase_long_EEGauto_arr[i] < bins[5]:
# #         pro_eegautos_time_false.append(0.004576541)
# #         pro_eegautos_time.append(0)
# #     elif phase_long_EEGauto_arr[i] >= bins[5] and phase_long_EEGauto_arr[i] < bins[6]:
# #         pro_eegautos_time_false.append(0.006016352)
# #         pro_eegautos_time.append(0)
# #     elif phase_long_EEGauto_arr[i] >= bins[6] and phase_long_EEGauto_arr[i] < bins[7]:
# #         pro_eegautos_time_false.append(0.03290996)
# #         pro_eegautos_time.append(0)
# #     elif phase_long_EEGauto_arr[i] >= bins[7] and phase_long_EEGauto_arr[i] < bins[8]:
# #         pro_eegautos_time_false.append(0.124132257)
# #         pro_eegautos_time.append(0)
# #     elif phase_long_EEGauto_arr[i] >= bins[8] and phase_long_EEGauto_arr[i] < bins[9]:
# #         pro_eegautos_time_false.append(0.32642567)
# #         pro_eegautos_time.append(0.666666667)
# #     elif phase_long_EEGauto_arr[i] >= bins[9] and phase_long_EEGauto_arr[i] <= bins[10]:
# #         pro_eegautos_time_false.append(0.351313827)
# #         pro_eegautos_time.append(0.333333333)
# #     elif phase_long_EEGauto_arr[i] > bins[10] and phase_long_EEGauto_arr[i] < bins[11]:
# #         pro_eegautos_time_false.append(0.108551448)
# #         pro_eegautos_time.append(0)
# #     elif phase_long_EEGauto_arr[i] >= bins[11] and phase_long_EEGauto_arr[i] < bins[12]:
# #         pro_eegautos_time_false.append(0.031161619)
# #         pro_eegautos_time.append(0)
# #     elif phase_long_EEGauto_arr[i] >= bins[12] and phase_long_EEGauto_arr[i] < bins[13]:
# #         pro_eegautos_time_false.append(0.011415643)
# #         pro_eegautos_time.append(0)
# #     elif phase_long_EEGauto_arr[i] >= bins[13] and phase_long_EEGauto_arr[i] < bins[14]:
# #         pro_eegautos_time_false.append(0.003496683)
# #         pro_eegautos_time.append(0)
# #     elif phase_long_EEGauto_arr[i] >= bins[14] and phase_long_EEGauto_arr[i] < bins[15]:
# #         pro_eegautos_time_false.append(0)
# #         pro_eegautos_time.append(0)
# #     elif phase_long_EEGauto_arr[i] >= bins[15] and phase_long_EEGauto_arr[i] < bins[16]:
# #         pro_eegautos_time_false.append(0)
# #         pro_eegautos_time.append(0)
# #     elif phase_long_EEGauto_arr[i] >= bins[16] and phase_long_EEGauto_arr[i] < bins[17]:
# #         pro_eegautos_time_false.append(0)
# #         pro_eegautos_time.append(0)
# #     elif phase_long_EEGauto_arr[i] >= bins[17]:
# #         pro_eegautos_time_false.append(0)
# #         pro_eegautos_time.append(0)
# # # print(pro_eegautos_time[586]);
# # print(pro_eegautos_time[6872]);print(pro_eegautos_time[10359]);print(pro_eegautos_time[16528]);
# # pro_RRIautos_time = []
# # pro_RRIautos_time_false = []
# # for i in range(len(phase_long_RRIauto_arr)):
# #     if phase_long_RRIauto_arr[i] >= bins[0] and phase_long_RRIauto_arr[i] < bins[1]:
# #         pro_RRIautos_time_false.append(0)
# #         pro_RRIautos_time.append(0)
# #     elif phase_long_RRIauto_arr[i] >= bins[1] and phase_long_RRIauto_arr[i] < bins[2]:
# #         pro_RRIautos_time_false.append(0)
# #         pro_RRIautos_time.append(0)
# #     elif phase_long_RRIauto_arr[i] >= bins[2] and phase_long_RRIauto_arr[i] < bins[3]:
# #         pro_RRIautos_time_false.append(0)
# #         pro_RRIautos_time.append(0)
# #     elif phase_long_RRIauto_arr[i] >= bins[3] and phase_long_RRIauto_arr[i] < bins[4]:
# #         pro_RRIautos_time_false.append(0)
# #         pro_RRIautos_time.append(0)
# #     elif phase_long_RRIauto_arr[i] >= bins[4] and phase_long_RRIauto_arr[i] < bins[5]:
# #         pro_RRIautos_time_false.append(0)
# #         pro_RRIautos_time.append(0)
# #     elif phase_long_RRIauto_arr[i] >= bins[5] and phase_long_RRIauto_arr[i] < bins[6]:
# #         pro_RRIautos_time_false.append(0.00226256)
# #         pro_RRIautos_time.append(0)
# #     elif phase_long_RRIauto_arr[i] >= bins[6] and phase_long_RRIauto_arr[i] < bins[7]:
# #         pro_RRIautos_time_false.append(0.006839101)
# #         pro_RRIautos_time.append(0)
# #     elif phase_long_RRIauto_arr[i] >= bins[7] and phase_long_RRIauto_arr[i] <= bins[8]:
# #         pro_RRIautos_time_false.append(0.060677739)
# #         pro_RRIautos_time.append(0.333333333)
# #     elif phase_long_RRIauto_arr[i] > bins[8] and phase_long_RRIauto_arr[i] <= bins[9]:
# #         pro_RRIautos_time_false.append(0.409111945)
# #         pro_RRIautos_time.append(0.666666667)
# #     elif phase_long_RRIauto_arr[i] > bins[9] and phase_long_RRIauto_arr[i] <= bins[10]:
# #         pro_RRIautos_time_false.append(0.465470252)
# #         pro_RRIautos_time.append(0)
# #     elif phase_long_RRIauto_arr[i] > bins[10] and phase_long_RRIauto_arr[i] < bins[11]:
# #         pro_RRIautos_time_false.append(0.05106186)
# #         pro_RRIautos_time.append(0)
# #     elif phase_long_RRIauto_arr[i] >= bins[11] and phase_long_RRIauto_arr[i] < bins[12]:
# #         pro_RRIautos_time_false.append(0.00395948)
# #         pro_RRIautos_time.append(0)
# #     elif phase_long_RRIauto_arr[i] >= bins[12] and phase_long_RRIauto_arr[i] < bins[13]:
# #         pro_RRIautos_time_false.append(0.000462796)
# #         pro_RRIautos_time.append(0)
# #     elif phase_long_RRIauto_arr[i] >= bins[13] and phase_long_RRIauto_arr[i] < bins[14]:
# #         pro_RRIautos_time_false.append(0.000154265)
# #         pro_RRIautos_time.append(0)
# #     elif phase_long_RRIauto_arr[i] >= bins[14] and phase_long_RRIauto_arr[i] < bins[15]:
# #         pro_RRIautos_time_false.append(0)
# #         pro_RRIautos_time.append(0)
# #     elif phase_long_RRIauto_arr[i] >= bins[15] and phase_long_RRIauto_arr[i] < bins[16]:
# #         pro_RRIautos_time_false.append(0)
# #         pro_RRIautos_time.append(0)
# #     elif phase_long_RRIauto_arr[i] >= bins[16] and phase_long_RRIauto_arr[i] < bins[17]:
# #         pro_RRIautos_time_false.append(0)
# #         pro_RRIautos_time.append(0)
# #     elif phase_long_RRIauto_arr[i] >= bins[17]:
# #         pro_RRIautos_time_false.append(0)
# #         pro_RRIautos_time.append(0)
# # # print(pro_RRIautos_time[586]);
# # print(pro_RRIautos_time[6872]);print(pro_RRIautos_time[10359]);print(pro_RRIautos_time[16528]);
#


Pseizureeegvar = 0.000154242;
Pnonseizureeegvar = 0.999845758;
t=np.linspace(0+0.00416667,0+0.00416667+0.00416667*(len(Raw_variance_EEG)-1),len(Raw_variance_EEG))
window_time_arr=t

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_eegautos_time)):
#     P1=pro_RRIautos_time[m]*Pseizureeegvar*pro_eegautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIautos_time_false[m]*pro_eegautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))

Pcombined = []
for m in range(len(pro_RRIautos_time)):
    P1=Pseizureeegvar*pro_RRIautos_time[m]
    P2=Pnonseizureeegvar*(1-pro_RRIautos_time_false[m])
    Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_eegautos_time)):
#     P1=Pseizureeegvar*pro_eegautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))


pyplot.figure(figsize=(12, 5))
pyplot.plot(window_time_arr, Pcombined)
pyplot.title('combined probability in VIC0829', fontsize=15)
# pyplot.annotate('', xy=(2.4472, np.max(Pcombined)), xytext=(2.4472, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(28.64, np.max(Pcombined)), xytext=(28.64, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(43.1678, np.max(Pcombined)), xytext=(43.1678, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(68.875, np.max(Pcombined)), xytext=(68.875, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.tight_layout()
pyplot.xlabel('Time(h)', fontsize=15)
pyplot.ylabel('seizure probability', fontsize=15)
pyplot.show()
pro=[]
for item in seizure_timing_index:
    pro.append(float(Pcombined[item]))
    print(Pcombined[item])
print(pro)
Th1=np.min(pro)
print(Th1)



# # t=np.linspace(0,0+0.00416667*(len(Raw_variance_EEG_arr)-1),len(Raw_variance_EEG_arr))
# # a=np.where(t<10.4536+0)
# # t[0:2508]=t[0:2508]-0+13.5463889
# # t[2508:]=t[2508:]-10.4536-0
# # time_feature_arr=[]
# # for i in range(len(t)):
# #     if t[i]>24:
# #         time_feature_arr.append(t[i] - (t[i] // 24) * 24)
# #     else:
# #         time_feature_arr.append(t[i])
# # bins_number = 18
# # bins = np.linspace(0, 24, bins_number + 1)
# #
# #
# # pro_circadian_time=[]
# # pro_circadian_time_false=[]
# # for i in range(len(time_feature_arr)):
# #     if time_feature_arr[i] >= bins[0] and time_feature_arr[i] <= bins[1]:
# #         pro_circadian_time_false.append(0.049364941)
# #         pro_circadian_time.append(0)
# #     elif time_feature_arr[i] > bins[1] and time_feature_arr[i] < bins[2]:
# #         pro_circadian_time_false.append(0.049364941)
# #         pro_circadian_time.append(0)
# #     elif time_feature_arr[i] >= bins[2] and time_feature_arr[i] < bins[3]:
# #         pro_circadian_time_false.append(0.049364941)
# #         pro_circadian_time.append(0)
# #     elif time_feature_arr[i] >= bins[3] and time_feature_arr[i] < bins[4]:
# #         pro_circadian_time_false.append(0.049364941)
# #         pro_circadian_time.append(0)
# #     elif time_feature_arr[i] >= bins[4] and time_feature_arr[i] < bins[5]:
# #         pro_circadian_time_false.append(0.049364941)
# #         pro_circadian_time.append(0)
# #     elif time_feature_arr[i] >= bins[5] and time_feature_arr[i] <= bins[6]:
# #         pro_circadian_time_false.append(0.049364941)
# #         pro_circadian_time.append(0)
# #     elif time_feature_arr[i] > bins[6] and time_feature_arr[i] < bins[7]:
# #         pro_circadian_time_false.append(0.049313519)
# #         pro_circadian_time.append(0.333333333)
# #     elif time_feature_arr[i] >= bins[7] and time_feature_arr[i] <= bins[8]:
# #         pro_circadian_time_false.append(0.049313519)
# #         pro_circadian_time.append(0.333333333)
# #     elif time_feature_arr[i] > bins[8] and time_feature_arr[i] < bins[9]:
# #         pro_circadian_time_false.append(0.05106186)
# #         pro_circadian_time.append(0)
# #     elif time_feature_arr[i] >= bins[9] and time_feature_arr[i] < bins[10]:
# #         pro_circadian_time_false.append(0.065819921)
# #         pro_circadian_time.append(0)
# #     elif time_feature_arr[i] >= bins[10] and time_feature_arr[i] < bins[11]:
# #         pro_circadian_time_false.append(0.065819921)
# #         pro_circadian_time.append(0)
# #     elif time_feature_arr[i] >= bins[11] and time_feature_arr[i] < bins[12]:
# #         pro_circadian_time_false.append(0.065819921)
# #         pro_circadian_time.append(0)
# #     elif time_feature_arr[i] >= bins[12] and time_feature_arr[i] < bins[13]:
# #         pro_circadian_time_false.append(0.065819921)
# #         pro_circadian_time.append(0)
# #     elif time_feature_arr[i] >= bins[13] and time_feature_arr[i] < bins[14]:
# #         pro_circadian_time_false.append(0.065768499)
# #         pro_circadian_time.append(0.333333333)
# #     elif time_feature_arr[i] >= bins[14] and time_feature_arr[i] < bins[15]:
# #         pro_circadian_time_false.append(0.065819921)
# #         pro_circadian_time.append(0)
# #     elif time_feature_arr[i] >= bins[15] and time_feature_arr[i] < bins[16]:
# #         pro_circadian_time_false.append(0.060523474)
# #         pro_circadian_time.append(0)
# #     elif time_feature_arr[i] >= bins[16] and time_feature_arr[i] < bins[17]:
# #         pro_circadian_time_false.append(0.049364941)
# #         pro_circadian_time.append(0)
# #     elif time_feature_arr[i] >= bins[17] and time_feature_arr[i] <= bins[18]:
# #         pro_circadian_time_false.append(0.049364941)
# #         pro_circadian_time.append(0)
# #
# # # Pcombined=[]
# # # for m in range(len(pro_eegvars_time)):
# # #     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_RRIautos_time[m]*pro_circadian_time[m]
# # #     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
# # #     Pcombined.append(P1/(P1+P2))
# #
# # # Pcombined = []
# # # for m in range(len(pro_eegautos_time)):
# # #     P1=pro_RRIautos_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_circadian_time[m]
# # #     P2=Pnonseizureeegvar*(1-pro_RRIautos_time_false[m]*pro_eegautos_time_false[m]*pro_circadian_time_false[m])
# # #     Pcombined.append(P1/(P1+P2))
# #
# # # Pcombined=[]
# # # for m in range(len(pro_eegvars_time)):
# # #     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_circadian_time[m]
# # #     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_circadian_time_false[m])
# # #     Pcombined.append(P1/(P1+P2))
# #
# # # Pcombined=[]
# # # for m in range(len(pro_RRIautos_time)):
# # #     P1=Pseizureeegvar*pro_RRIautos_time[m]*pro_circadian_time[m]
# # #     P2=Pnonseizureeegvar*(1-pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
# # #     Pcombined.append(P1/(P1+P2))
# #
# # # Pcombined=[]
# # # for m in range(len(pro_eegvars_time)):
# # #     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_circadian_time[m]
# # #     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_circadian_time_false[m])
# # #     Pcombined.append(P1/(P1+P2))
# #
# # # Pcombined=[]
# # # for m in range(len(pro_eegautos_time)):
# # #     P1=Pseizureeegvar*pro_eegautos_time[m]*pro_circadian_time[m]
# # #     P2=Pnonseizureeegvar*(1-pro_eegautos_time_false[m]*pro_circadian_time_false[m])
# # #     Pcombined.append(P1/(P1+P2))
# #
# # # t=np.linspace(0+0.00416667,0+0.00416667+0.00416667*(len(Raw_variance_EEG)-1),len(Raw_variance_EEG))
# # # window_time_arr=t
# # # pyplot.figure(figsize=(12, 5))
# # # pyplot.plot(window_time_arr, Pcombined)
# # # pyplot.title('combined probability in VIC0829', fontsize=15)
# # # # pyplot.annotate('', xy=(2.4472, np.max(Pcombined)), xytext=(2.4472, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# # # pyplot.annotate('', xy=(28.64, np.max(Pcombined)), xytext=(28.64, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# # # pyplot.annotate('', xy=(43.1678, np.max(Pcombined)), xytext=(43.1678, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# # # pyplot.annotate('', xy=(68.875, np.max(Pcombined)), xytext=(68.875, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# # # pyplot.tight_layout()
# # # pyplot.xlim(window_time_arr[0], window_time_arr[-1])
# # # pyplot.xlabel('Time(h)', fontsize=15)
# # # pyplot.ylabel('seizure probability', fontsize=15)
# # # pyplot.show()
# # # pro=[]
# # # for item in seizure_timing_index:
# # #     pro.append(float(Pcombined[item]))
# # #     print(Pcombined[item])
# # # print(pro)
# # # Th2=np.min(pro)
# # # print(Th2)
# #
# #






# ## section 3 froecast
t=np.linspace(0,0+0.00416667*(len(Raw_variance_EEG_arr)-1),len(Raw_variance_EEG_arr))
t_window_arr=t
fore_arr_EEGvars=[]
for k in range(81,82):
    variance_arr = Raw_variance_EEG_arr[0:(19450+240*k)]
    long_rhythm_var_arr=movingaverage(variance_arr,240 * 6)
    pyplot.figure(figsize=(6, 3))
    pyplot.title('EEG variance')
    pyplot.ylabel('Voltage ($\mathregular{v^2}$)')
    pyplot.xlabel('Time(h)')
    pyplot.plot(t_window_arr[240*24:(19450+240*k)], long_rhythm_var_arr[240*24:],'orange')
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC0829/cycles6h_Cz_forecast81hsignal_3hcycle_EEGvar_VIC0829.csv',sep=',',header=None)
forecast_var_EEG= csv_reader.values
forecast_var_EEG_arr=[]
for item in forecast_var_EEG:
    forecast_var_EEG_arr=forecast_var_EEG_arr+list(item)
t=np.linspace(t_window_arr[19450],t_window_arr[19450]+0.1666667*(len(forecast_var_EEG_arr)-1),len(forecast_var_EEG_arr))
pyplot.plot(t, forecast_var_EEG_arr,'k',label='forecast EEG var')
pyplot.legend()
pyplot.show()

fore_arr_EEGauto=[]
for k in range(81,82):
    auto_arr = Raw_auto_EEG_arr[0:(19450+240*k)]
    long_rhythm_auto_arr=movingaverage(auto_arr,240 * 6)
    pyplot.figure(figsize=(6, 3))
    pyplot.title('EEG autocorrelation')
    pyplot.xlabel('time(h)')
    pyplot.plot(t_window_arr[240*24:(19450+240*k)], long_rhythm_auto_arr[240*24:],'orange')
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC0829/cycles6h_Cz_forecast81hsignal_3hcycle_EEGauto_VIC0829.csv',sep=',',header=None)
forecast_auto_EEG= csv_reader.values
forecast_auto_EEG_arr=[]
for item in forecast_auto_EEG:
    forecast_auto_EEG_arr=forecast_auto_EEG_arr+list(item)
t=np.linspace(t_window_arr[19450],t_window_arr[19450]+0.1666667*(len(forecast_auto_EEG_arr)-1),len(forecast_auto_EEG_arr))
pyplot.plot(t, forecast_auto_EEG_arr,'k',label='forecast EEG auto')
pyplot.legend()
pyplot.show()

# fore_arr_RRIvars=[]
# for k in range(81, 82):
#     variance_arr = Raw_variance_RRI31_arr[0:(19450+240*k)]
#     long_rhythm_var_arr=movingaverage(variance_arr,240*24)
#     pyplot.figure(figsize=(6, 3))
#     pyplot.title('RRI variance')
#     pyplot.ylabel('Second ($\mathregular{s^2}$)')
#     pyplot.xlabel('Time(h)')
#     pyplot.plot(t_window_arr[240*6:(19450+240*k)], long_rhythm_var_arr[240*6:],'orange')
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC0829/cycles24h_ch31_forecast81hsignal_3hcycle_RRIvar_VIC0829.csv', sep=',',header=None)
# forecast_var_RRI31= csv_reader.values
# forecast_var_RRI31_arr=[]
# for item in forecast_var_RRI31:
#     forecast_var_RRI31_arr=forecast_var_RRI31_arr+list(item)
# t=np.linspace(t_window_arr[19450],t_window_arr[19450]+0.1666667*(len(forecast_var_RRI31_arr)-1),len(forecast_var_RRI31_arr))
# pyplot.plot(t, forecast_var_RRI31_arr,'k',label='forecast RRI var')
# pyplot.legend()
# pyplot.show()

fore_arr_RRIautos=[]
save_data_RRIautos=[]
for k in range(81,82):
    auto_arr = Raw_auto_RRI31_arr[0:19450+240*k]
    long_rhythm_auto_arr=movingaverage(auto_arr,240 * 6)
    pyplot.figure(figsize=(6,3))
    pyplot.title('RRI autocorrelation')
    pyplot.xlabel('Time(h)')
    pyplot.plot(t_window_arr[240*24:19450+240*k], long_rhythm_auto_arr[240*24:],'orange')
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC0829/cycles6h_ch31_forecast81hsignal_3hcycle_RRIauto_VIC0829.csv',sep=',',header=None)
forecast_auto_RRI31= csv_reader.values
forecast_auto_RRI31_arr=[]
for item in forecast_auto_RRI31:
    forecast_auto_RRI31_arr=forecast_auto_RRI31_arr+list(item)
t=np.linspace(t_window_arr[19450],t_window_arr[19450]+0.1666667*(len(forecast_auto_RRI31_arr)-1),len(forecast_auto_RRI31_arr))
pyplot.plot(t, forecast_auto_RRI31_arr,'k',label='forecast RRI auto')
pyplot.legend()
pyplot.show()




# ### predict, forecast data
var_trans=hilbert(forecast_var_EEG_arr)
var_phase=np.angle(var_trans)
rolmean_short_EEGvar=var_phase
var_trans=hilbert(forecast_auto_EEG_arr)
var_phase=np.angle(var_trans)
rolmean_short_EEGauto=var_phase
# var_trans=hilbert(forecast_var_RRI31_arr)
# var_phase=np.angle(var_trans)
# rolmean_short_RRIvar=var_phase
var_trans=hilbert(forecast_auto_RRI31_arr)
var_phase=np.angle(var_trans)
rolmean_short_RRIauto=var_phase




# ##### 24 h 24 h 24 h 24 h
# bins_number = 18
# bins = np.linspace(-np.pi, np.pi, bins_number + 1)
# pro_eegvars_time = []
# pro_eegvars_time_false = []
# for i in range(len(rolmean_short_EEGvar)):
#     if rolmean_short_EEGvar[i] >= bins[0] and rolmean_short_EEGvar[i] < bins[1]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[1] and rolmean_short_EEGvar[i] < bins[2]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[2] and rolmean_short_EEGvar[i] < bins[3]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[3] and rolmean_short_EEGvar[i] < bins[4]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[4] and rolmean_short_EEGvar[i] < bins[5]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[5] and rolmean_short_EEGvar[i] < bins[6]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[6] and rolmean_short_EEGvar[i] <= bins[7]:
#         pro_eegvars_time_false.append(0.000102844)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] > bins[7] and rolmean_short_EEGvar[i] < bins[8]:
#         pro_eegvars_time_false.append(0.130097187)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[8] and rolmean_short_EEGvar[i] <= bins[9]:
#         pro_eegvars_time_false.append(0.471795135)
#         pro_eegvars_time.append(0.666666667)
#     elif rolmean_short_EEGvar[i] > bins[9] and rolmean_short_EEGvar[i] < bins[10]:
#         pro_eegvars_time_false.append(0.27850054)
#         pro_eegvars_time.append(0.333333333)
#     elif rolmean_short_EEGvar[i] >= bins[10] and rolmean_short_EEGvar[i] <= bins[11]:
#         pro_eegvars_time_false.append(0.060266365)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] > bins[11] and rolmean_short_EEGvar[i] < bins[12]:
#         pro_eegvars_time_false.append(0.030236026)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] > bins[12] and rolmean_short_EEGvar[i] < bins[13]:
#         pro_eegvars_time_false.append(0.021237209)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[13] and rolmean_short_EEGvar[i] < bins[14]:
#         pro_eegvars_time_false.append(0.007764694)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[14] and rolmean_short_EEGvar[i] < bins[15]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[15] and rolmean_short_EEGvar[i] < bins[16]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[16] and rolmean_short_EEGvar[i] < bins[17]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[17]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
# pro_eegautos_time = []
# pro_eegautos_time_false = []
# for i in range(len(rolmean_short_EEGauto)):
#     if rolmean_short_EEGauto[i] >= bins[0] and rolmean_short_EEGauto[i] < bins[1]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[1] and rolmean_short_EEGauto[i] < bins[2]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[2] and rolmean_short_EEGauto[i] < bins[3]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[3] and rolmean_short_EEGauto[i] < bins[4]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[4] and rolmean_short_EEGauto[i] < bins[5]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[5] and rolmean_short_EEGauto[i] < bins[6]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[6] and rolmean_short_EEGauto[i] < bins[7]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[7] and rolmean_short_EEGauto[i] < bins[8]:
#         pro_eegautos_time_false.append(0.17894791)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[8] and rolmean_short_EEGauto[i] < bins[9]:
#         pro_eegautos_time_false.append(0.339178279)
#         pro_eegautos_time.append(0.666666667)
#     elif rolmean_short_EEGauto[i] >= bins[9] and rolmean_short_EEGauto[i] <= bins[10]:
#         pro_eegautos_time_false.append(0.386332082)
#         pro_eegautos_time.append(0.333333333)
#     elif rolmean_short_EEGauto[i] > bins[10] and rolmean_short_EEGauto[i] < bins[11]:
#         pro_eegautos_time_false.append(0.055535558)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[11] and rolmean_short_EEGauto[i] < bins[12]:
#         pro_eegautos_time_false.append(0.013163984)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[12] and rolmean_short_EEGauto[i] < bins[13]:
#         pro_eegautos_time_false.append(0.010335784)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[13] and rolmean_short_EEGauto[i] < bins[14]:
#         pro_eegautos_time_false.append(0.016506402)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[14] and rolmean_short_EEGauto[i] < bins[15]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[15] and rolmean_short_EEGauto[i] < bins[16]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[16] and rolmean_short_EEGauto[i] < bins[17]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[17]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
# pro_RRIautos_time = []
# pro_RRIautos_time_false = []
# for i in range(len(rolmean_short_RRIauto)):
#     if rolmean_short_RRIauto[i] >= bins[0] and rolmean_short_RRIauto[i] < bins[1]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[1] and rolmean_short_RRIauto[i] < bins[2]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[2] and rolmean_short_RRIauto[i] < bins[3]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[3] and rolmean_short_RRIauto[i] < bins[4]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[4] and rolmean_short_RRIauto[i] < bins[5]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[5] and rolmean_short_RRIauto[i] < bins[6]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[6] and rolmean_short_RRIauto[i] < bins[7]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[7] and rolmean_short_RRIauto[i] <= bins[8]:
#         pro_RRIautos_time_false.append(0.171851699)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] > bins[8] and rolmean_short_RRIauto[i] <= bins[9]:
#         pro_RRIautos_time_false.append(0.405769527)
#         pro_RRIautos_time.append(0.666666667)
#     elif rolmean_short_RRIauto[i] > bins[9] and rolmean_short_RRIauto[i] <= bins[10]:
#         pro_RRIautos_time_false.append(0.316964056)
#         pro_RRIautos_time.append(0.333333333)
#     elif rolmean_short_RRIauto[i] > bins[10] and rolmean_short_RRIauto[i] < bins[11]:
#         pro_RRIautos_time_false.append(0.049879159)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[11] and rolmean_short_RRIauto[i] < bins[12]:
#         pro_RRIautos_time_false.append(0.017174886)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[12] and rolmean_short_RRIauto[i] < bins[13]:
#         pro_RRIautos_time_false.append(0.014346686)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[13] and rolmean_short_RRIauto[i] < bins[14]:
#         pro_RRIautos_time_false.append(0.024013987)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[14] and rolmean_short_RRIauto[i] < bins[15]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[15] and rolmean_short_RRIauto[i] < bins[16]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[16] and rolmean_short_RRIauto[i] < bins[17]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[17]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)


#### 12h 12h 12h
# bins_number = 18
# bins = np.linspace(-np.pi, np.pi, bins_number + 1)
# pro_eegvars_time = []
# pro_eegvars_time_false = []
# for i in range(len(rolmean_short_EEGvar)):
#     if rolmean_short_EEGvar[i] >= bins[0] and rolmean_short_EEGvar[i] < bins[1]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[1] and rolmean_short_EEGvar[i] < bins[2]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[2] and rolmean_short_EEGvar[i] < bins[3]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[3] and rolmean_short_EEGvar[i] < bins[4]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[4] and rolmean_short_EEGvar[i] < bins[5]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[5] and rolmean_short_EEGvar[i] < bins[6]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[6] and rolmean_short_EEGvar[i] <= bins[7]:
#         pro_eegvars_time_false.append(0.047102381)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] > bins[7] and rolmean_short_EEGvar[i] < bins[8]:
#         pro_eegvars_time_false.append(0.271352908)
#         pro_eegvars_time.append(0.333333333)
#     elif rolmean_short_EEGvar[i] >= bins[8] and rolmean_short_EEGvar[i] <= bins[9]:
#         pro_eegvars_time_false.append(0.147580604)
#         pro_eegvars_time.append(0.333333333)
#     elif rolmean_short_EEGvar[i] > bins[9] and rolmean_short_EEGvar[i] < bins[10]:
#         pro_eegvars_time_false.append(0.29778372)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[10] and rolmean_short_EEGvar[i] <= bins[11]:
#         pro_eegvars_time_false.append(0.163881318)
#         pro_eegvars_time.append(0.333333333)
#     elif rolmean_short_EEGvar[i] > bins[11] and rolmean_short_EEGvar[i] < bins[12]:
#         pro_eegvars_time_false.append(0.052501671)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] > bins[12] and rolmean_short_EEGvar[i] < bins[13]:
#         pro_eegvars_time_false.append(0.013678202)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[13] and rolmean_short_EEGvar[i] < bins[14]:
#         pro_eegvars_time_false.append(0.006119196)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[14] and rolmean_short_EEGvar[i] < bins[15]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[15] and rolmean_short_EEGvar[i] < bins[16]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[16] and rolmean_short_EEGvar[i] < bins[17]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[17]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
# pro_eegautos_time = []
# pro_eegautos_time_false = []
# for i in range(len(rolmean_short_EEGauto)):
#     if rolmean_short_EEGauto[i] >= bins[0] and rolmean_short_EEGauto[i] < bins[1]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[1] and rolmean_short_EEGauto[i] < bins[2]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[2] and rolmean_short_EEGauto[i] < bins[3]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[3] and rolmean_short_EEGauto[i] < bins[4]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[4] and rolmean_short_EEGauto[i] < bins[5]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[5] and rolmean_short_EEGauto[i] < bins[6]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[6] and rolmean_short_EEGauto[i] < bins[7]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[7] and rolmean_short_EEGauto[i] < bins[8]:
#         pro_eegautos_time_false.append(0.11770453)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[8] and rolmean_short_EEGauto[i] < bins[9]:
#         pro_eegautos_time_false.append(0.326117139)
#         pro_eegautos_time.append(0.666666667)
#     elif rolmean_short_EEGauto[i] >= bins[9] and rolmean_short_EEGauto[i] <= bins[10]:
#         pro_eegautos_time_false.append(0.528616239)
#         pro_eegautos_time.append(0.333333333)
#     elif rolmean_short_EEGauto[i] > bins[10] and rolmean_short_EEGauto[i] < bins[11]:
#         pro_eegautos_time_false.append(0.011209955)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[11] and rolmean_short_EEGauto[i] < bins[12]:
#         pro_eegautos_time_false.append(0.003496683)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[12] and rolmean_short_EEGauto[i] < bins[13]:
#         pro_eegautos_time_false.append(0.003239574)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[13] and rolmean_short_EEGauto[i] < bins[14]:
#         pro_eegautos_time_false.append(0.009615879)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[14] and rolmean_short_EEGauto[i] < bins[15]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[15] and rolmean_short_EEGauto[i] < bins[16]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[16] and rolmean_short_EEGauto[i] < bins[17]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[17]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
# pro_RRIautos_time = []
# pro_RRIautos_time_false = []
# for i in range(len(rolmean_short_RRIauto)):
#     if rolmean_short_RRIauto[i] >= bins[0] and rolmean_short_RRIauto[i] < bins[1]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[1] and rolmean_short_RRIauto[i] < bins[2]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[2] and rolmean_short_RRIauto[i] < bins[3]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[3] and rolmean_short_RRIauto[i] < bins[4]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[4] and rolmean_short_RRIauto[i] < bins[5]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[5] and rolmean_short_RRIauto[i] < bins[6]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[6] and rolmean_short_RRIauto[i] < bins[7]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[7] and rolmean_short_RRIauto[i] <= bins[8]:
#         pro_RRIautos_time_false.append(0.081452152)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] > bins[8] and rolmean_short_RRIauto[i] <= bins[9]:
#         pro_RRIautos_time_false.append(0.504293721)
#         pro_RRIautos_time.append(0.666666667)
#     elif rolmean_short_RRIauto[i] > bins[9] and rolmean_short_RRIauto[i] <= bins[10]:
#         pro_RRIautos_time_false.append(0.355427572)
#         pro_RRIautos_time.append(0.333333333)
#     elif rolmean_short_RRIauto[i] > bins[10] and rolmean_short_RRIauto[i] < bins[11]:
#         pro_RRIautos_time_false.append(0.029104746)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[11] and rolmean_short_RRIauto[i] < bins[12]:
#         pro_RRIautos_time_false.append(0.008433177)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[12] and rolmean_short_RRIauto[i] < bins[13]:
#         pro_RRIautos_time_false.append(0.007044788)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[13] and rolmean_short_RRIauto[i] < bins[14]:
#         pro_RRIautos_time_false.append(0.014243842)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[14] and rolmean_short_RRIauto[i] < bins[15]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[15] and rolmean_short_RRIauto[i] < bins[16]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[16] and rolmean_short_RRIauto[i] < bins[17]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[17]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)


#### 6h 6h 6h
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
        pro_eegvars_time_false.append(0.02581375)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] >= bins[6] and rolmean_short_EEGvar[i] <= bins[7]:
        pro_eegvars_time_false.append(0.183627295)
        pro_eegvars_time.append(0.333333333)
    elif rolmean_short_EEGvar[i] > bins[7] and rolmean_short_EEGvar[i] < bins[8]:
        pro_eegvars_time_false.append(0.172211652)
        pro_eegvars_time.append(0.333333333)
    elif rolmean_short_EEGvar[i] >= bins[8] and rolmean_short_EEGvar[i] <= bins[9]:
        pro_eegvars_time_false.append(0.155242454)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] > bins[9] and rolmean_short_EEGvar[i] < bins[10]:
        pro_eegvars_time_false.append(0.107317324)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] >= bins[10] and rolmean_short_EEGvar[i] <= bins[11]:
        pro_eegvars_time_false.append(0.13158842)
        pro_eegvars_time.append(0.333333333)
    elif rolmean_short_EEGvar[i] > bins[11] and rolmean_short_EEGvar[i] < bins[12]:
        pro_eegvars_time_false.append(0.111945287)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] > bins[12] and rolmean_short_EEGvar[i] < bins[13]:
        pro_eegvars_time_false.append(0.106186044)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] >= bins[13] and rolmean_short_EEGvar[i] < bins[14]:
        pro_eegvars_time_false.append(0.006067774)
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
        pro_eegautos_time_false.append(0.02154574)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[7] and rolmean_short_EEGauto[i] < bins[8]:
        pro_eegautos_time_false.append(0.05049622)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[8] and rolmean_short_EEGauto[i] < bins[9]:
        pro_eegautos_time_false.append(0.40443256)
        pro_eegautos_time.append(0.666666667)
    elif rolmean_short_EEGauto[i] >= bins[9] and rolmean_short_EEGauto[i] <= bins[10]:
        pro_eegautos_time_false.append(0.496169075)
        pro_eegautos_time.append(0.333333333)
    elif rolmean_short_EEGauto[i] > bins[10] and rolmean_short_EEGauto[i] < bins[11]:
        pro_eegautos_time_false.append(0.017894791)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[11] and rolmean_short_EEGauto[i] < bins[12]:
        pro_eegautos_time_false.append(0.000977014)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[12] and rolmean_short_EEGauto[i] < bins[13]:
        pro_eegautos_time_false.append(0.000308531)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[13] and rolmean_short_EEGauto[i] < bins[14]:
        pro_eegautos_time_false.append(0.008176068)
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
# pro_RRIvars_time = []
# pro_RRIvars_time_false = []
# for i in range(len(rolmean_short_RRIvar)):
#     if rolmean_short_RRIvar[i] >= bins[0] and rolmean_short_RRIvar[i] < bins[1]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[1] and rolmean_short_RRIvar[i] < bins[2]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[2] and rolmean_short_RRIvar[i] < bins[3]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[3] and rolmean_short_RRIvar[i] < bins[4]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[4] and rolmean_short_RRIvar[i] < bins[5]:
#         pro_RRIvars_time_false.append(0.020928678)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[5] and rolmean_short_RRIvar[i] < bins[6]:
#         pro_RRIvars_time_false.append(0.147374916)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[6] and rolmean_short_RRIvar[i] < bins[7]:
#         pro_RRIvars_time_false.append(0.110814007)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[7] and rolmean_short_RRIvar[i] <= bins[8]:
#         pro_RRIvars_time_false.append(0.137913303)
#         pro_RRIvars_time.append(0.333333333)
#     elif rolmean_short_RRIvar[i] > bins[8] and rolmean_short_RRIvar[i] < bins[9]:
#         pro_RRIvars_time_false.append(0.061500489)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[9] and rolmean_short_RRIvar[i] <= bins[10]:
#         pro_RRIvars_time_false.append(0.091119453)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] > bins[10] and rolmean_short_RRIvar[i] <= bins[11]:
#         pro_RRIvars_time_false.append(0.14860904)
#         pro_RRIvars_time.append(0.666666667)
#     elif rolmean_short_RRIvar[i] > bins[11] and rolmean_short_RRIvar[i] < bins[12]:
#         pro_RRIvars_time_false.append(0.167635111)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[12] and rolmean_short_RRIvar[i] < bins[13]:
#         pro_RRIvars_time_false.append(0.069727979)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[13] and rolmean_short_RRIvar[i] < bins[14]:
#         pro_RRIvars_time_false.append(0.044377025)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[14] and rolmean_short_RRIvar[i] < bins[15]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[15] and rolmean_short_RRIvar[i] < bins[16]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[16] and rolmean_short_RRIvar[i] < bins[17]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[17]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
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
        pro_RRIautos_time_false.append(0.05049622)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] > bins[8] and rolmean_short_RRIauto[i] <= bins[9]:
        pro_RRIautos_time_false.append(0.408546305)
        pro_RRIautos_time.append(0.666666667)
    elif rolmean_short_RRIauto[i] > bins[9] and rolmean_short_RRIauto[i] <= bins[10]:
        pro_RRIautos_time_false.append(0.501105569)
        pro_RRIautos_time.append(0.333333333)
    elif rolmean_short_RRIauto[i] > bins[10] and rolmean_short_RRIauto[i] < bins[11]:
        pro_RRIautos_time_false.append(0.02298555)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[11] and rolmean_short_RRIauto[i] < bins[12]:
        pro_RRIautos_time_false.append(0.004422276)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[12] and rolmean_short_RRIauto[i] < bins[13]:
        pro_RRIautos_time_false.append(0.003136731)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[13] and rolmean_short_RRIauto[i] < bins[14]:
        pro_RRIautos_time_false.append(0.009307348)
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


# # ###### 1h 1h 1h
# # # bins_number = 18
# # # bins = np.linspace(-np.pi, np.pi, bins_number + 1)
# # # pro_eegautos_time = []
# # # pro_eegautos_time_false = []
# # # for i in range(len(rolmean_short_EEGauto)):
# # #     if rolmean_short_EEGauto[i] >= bins[0] and rolmean_short_EEGauto[i] < bins[1]:
# # #         pro_eegautos_time_false.append(0)
# # #         pro_eegautos_time.append(0)
# # #     elif rolmean_short_EEGauto[i] >= bins[1] and rolmean_short_EEGauto[i] < bins[2]:
# # #         pro_eegautos_time_false.append(0)
# # #         pro_eegautos_time.append(0)
# # #     elif rolmean_short_EEGauto[i] >= bins[2] and rolmean_short_EEGauto[i] < bins[3]:
# # #         pro_eegautos_time_false.append(0)
# # #         pro_eegautos_time.append(0)
# # #     elif rolmean_short_EEGauto[i] >= bins[3] and rolmean_short_EEGauto[i] < bins[4]:
# # #         pro_eegautos_time_false.append(0)
# # #         pro_eegautos_time.append(0)
# # #     elif rolmean_short_EEGauto[i] >= bins[4] and rolmean_short_EEGauto[i] < bins[5]:
# # #         pro_eegautos_time_false.append(0.003496683)
# # #         pro_eegautos_time.append(0)
# # #     elif rolmean_short_EEGauto[i] >= bins[5] and rolmean_short_EEGauto[i] < bins[6]:
# # #         pro_eegautos_time_false.append(0.00169692)
# # #         pro_eegautos_time.append(0)
# # #     elif rolmean_short_EEGauto[i] >= bins[6] and rolmean_short_EEGauto[i] < bins[7]:
# # #         pro_eegautos_time_false.append(0.00992441)
# # #         pro_eegautos_time.append(0)
# # #     elif rolmean_short_EEGauto[i] >= bins[7] and rolmean_short_EEGauto[i] < bins[8]:
# # #         pro_eegautos_time_false.append(0.096210212)
# # #         pro_eegautos_time.append(0)
# # #     elif rolmean_short_EEGauto[i] >= bins[8] and rolmean_short_EEGauto[i] < bins[9]:
# # #         pro_eegautos_time_false.append(0.388491798)
# # #         pro_eegautos_time.append(0.666666667)
# # #     elif rolmean_short_EEGauto[i] >= bins[9] and rolmean_short_EEGauto[i] <= bins[10]:
# # #         pro_eegautos_time_false.append(0.398724739)
# # #         pro_eegautos_time.append(0.333333333)
# # #     elif rolmean_short_EEGauto[i] > bins[10] and rolmean_short_EEGauto[i] < bins[11]:
# # #         pro_eegautos_time_false.append(0.080577981)
# # #         pro_eegautos_time.append(0)
# # #     elif rolmean_short_EEGauto[i] >= bins[11] and rolmean_short_EEGauto[i] < bins[12]:
# # #         pro_eegautos_time_false.append(0.011672752)
# # #         pro_eegautos_time.append(0)
# # #     elif rolmean_short_EEGauto[i] >= bins[12] and rolmean_short_EEGauto[i] < bins[13]:
# # #         pro_eegautos_time_false.append(0.004010901)
# # #         pro_eegautos_time.append(0)
# # #     elif rolmean_short_EEGauto[i] >= bins[13] and rolmean_short_EEGauto[i] < bins[14]:
# # #         pro_eegautos_time_false.append(0.005193603)
# # #         pro_eegautos_time.append(0)
# # #     elif rolmean_short_EEGauto[i] >= bins[14] and rolmean_short_EEGauto[i] < bins[15]:
# # #         pro_eegautos_time_false.append(0)
# # #         pro_eegautos_time.append(0)
# # #     elif rolmean_short_EEGauto[i] >= bins[15] and rolmean_short_EEGauto[i] < bins[16]:
# # #         pro_eegautos_time_false.append(0)
# # #         pro_eegautos_time.append(0)
# # #     elif rolmean_short_EEGauto[i] >= bins[16] and rolmean_short_EEGauto[i] < bins[17]:
# # #         pro_eegautos_time_false.append(0)
# # #         pro_eegautos_time.append(0)
# # #     elif rolmean_short_EEGauto[i] >= bins[17]:
# # #         pro_eegautos_time_false.append(0)
# # #         pro_eegautos_time.append(0)
# # # pro_RRIautos_time = []
# # # pro_RRIautos_time_false = []
# # # for i in range(len(rolmean_short_RRIauto)):
# # #     if rolmean_short_RRIauto[i] >= bins[0] and rolmean_short_RRIauto[i] < bins[1]:
# # #         pro_RRIautos_time_false.append(0)
# # #         pro_RRIautos_time.append(0)
# # #     elif rolmean_short_RRIauto[i] >= bins[1] and rolmean_short_RRIauto[i] < bins[2]:
# # #         pro_RRIautos_time_false.append(0)
# # #         pro_RRIautos_time.append(0)
# # #     elif rolmean_short_RRIauto[i] >= bins[2] and rolmean_short_RRIauto[i] < bins[3]:
# # #         pro_RRIautos_time_false.append(0)
# # #         pro_RRIautos_time.append(0)
# # #     elif rolmean_short_RRIauto[i] >= bins[3] and rolmean_short_RRIauto[i] < bins[4]:
# # #         pro_RRIautos_time_false.append(0)
# # #         pro_RRIautos_time.append(0)
# # #     elif rolmean_short_RRIauto[i] >= bins[4] and rolmean_short_RRIauto[i] < bins[5]:
# # #         pro_RRIautos_time_false.append(0)
# # #         pro_RRIautos_time.append(0)
# # #     elif rolmean_short_RRIauto[i] >= bins[5] and rolmean_short_RRIauto[i] < bins[6]:
# # #         pro_RRIautos_time_false.append(0.000668484)
# # #         pro_RRIautos_time.append(0)
# # #     elif rolmean_short_RRIauto[i] >= bins[6] and rolmean_short_RRIauto[i] < bins[7]:
# # #         pro_RRIautos_time_false.append(0.007970381)
# # #         pro_RRIautos_time.append(0)
# # #     elif rolmean_short_RRIauto[i] >= bins[7] and rolmean_short_RRIauto[i] <= bins[8]:
# # #         pro_RRIautos_time_false.append(0.030955932)
# # #         pro_RRIautos_time.append(0.333333333)
# # #     elif rolmean_short_RRIauto[i] > bins[8] and rolmean_short_RRIauto[i] <= bins[9]:
# # #         pro_RRIautos_time_false.append(0.423047257)
# # #         pro_RRIautos_time.append(0.666666667)
# # #     elif rolmean_short_RRIauto[i] > bins[9] and rolmean_short_RRIauto[i] <= bins[10]:
# # #         pro_RRIautos_time_false.append(0.53005605)
# # #         pro_RRIautos_time.append(0)
# # #     elif rolmean_short_RRIauto[i] > bins[10] and rolmean_short_RRIauto[i] < bins[11]:
# # #         pro_RRIautos_time_false.append(0.002725356)
# # #         pro_RRIautos_time.append(0)
# # #     elif rolmean_short_RRIauto[i] >= bins[11] and rolmean_short_RRIauto[i] < bins[12]:
# # #         pro_RRIautos_time_false.append(0.001079858)
# # #         pro_RRIautos_time.append(0)
# # #     elif rolmean_short_RRIauto[i] >= bins[12] and rolmean_short_RRIauto[i] < bins[13]:
# # #         pro_RRIautos_time_false.append(0.001388389)
# # #         pro_RRIautos_time.append(0)
# # #     elif rolmean_short_RRIauto[i] >= bins[13] and rolmean_short_RRIauto[i] < bins[14]:
# # #         pro_RRIautos_time_false.append(0.002108294)
# # #         pro_RRIautos_time.append(0)
# # #     elif rolmean_short_RRIauto[i] >= bins[14] and rolmean_short_RRIauto[i] < bins[15]:
# # #         pro_RRIautos_time_false.append(0)
# # #         pro_RRIautos_time.append(0)
# # #     elif rolmean_short_RRIauto[i] >= bins[15] and rolmean_short_RRIauto[i] < bins[16]:
# # #         pro_RRIautos_time_false.append(0)
# # #         pro_RRIautos_time.append(0)
# # #     elif rolmean_short_RRIauto[i] >= bins[16] and rolmean_short_RRIauto[i] < bins[17]:
# # #         pro_RRIautos_time_false.append(0)
# # #         pro_RRIautos_time.append(0)
# # #     elif rolmean_short_RRIauto[i] >= bins[17]:
# # #         pro_RRIautos_time_false.append(0)
# # #         pro_RRIautos_time.append(0)
# #
# #
# # ###### 5 min 5 min 5 min
# # #### combined probability calculation
# # bins_number = 18
# # bins = np.linspace(-np.pi, np.pi, bins_number + 1)
# # pro_eegvars_time = []
# # pro_eegvars_time_false = []
# # for i in range(len(rolmean_short_EEGvar)):
# #     if rolmean_short_EEGvar[i] >= bins[0] and rolmean_short_EEGvar[i] < bins[1]:
# #         pro_eegvars_time_false.append(0)
# #         pro_eegvars_time.append(0)
# #     elif rolmean_short_EEGvar[i] >= bins[1] and rolmean_short_EEGvar[i] < bins[2]:
# #         pro_eegvars_time_false.append(0)
# #         pro_eegvars_time.append(0)
# #     elif rolmean_short_EEGvar[i] >= bins[2] and rolmean_short_EEGvar[i] < bins[3]:
# #         pro_eegvars_time_false.append(0)
# #         pro_eegvars_time.append(0)
# #     elif rolmean_short_EEGvar[i] >= bins[3] and rolmean_short_EEGvar[i] < bins[4]:
# #         pro_eegvars_time_false.append(0)
# #         pro_eegvars_time.append(0)
# #     elif rolmean_short_EEGvar[i] >= bins[4] and rolmean_short_EEGvar[i] < bins[5]:
# #         pro_eegvars_time_false.append(0.138787474)
# #         pro_eegvars_time.append(0)
# #     elif rolmean_short_EEGvar[i] >= bins[5] and rolmean_short_EEGvar[i] < bins[6]:
# #         pro_eegvars_time_false.append(0.189386538)
# #         pro_eegvars_time.append(0.333333333)
# #     elif rolmean_short_EEGvar[i] >= bins[6] and rolmean_short_EEGvar[i] <= bins[7]:
# #         pro_eegvars_time_false.append(0.094770402)
# #         pro_eegvars_time.append(0)
# #     elif rolmean_short_EEGvar[i] > bins[7] and rolmean_short_EEGvar[i] < bins[8]:
# #         pro_eegvars_time_false.append(0.053375842)
# #         pro_eegvars_time.append(0)
# #     elif rolmean_short_EEGvar[i] >= bins[8] and rolmean_short_EEGvar[i] <= bins[9]:
# #         pro_eegvars_time_false.append(0.0419602)
# #         pro_eegvars_time.append(0.333333333)
# #     elif rolmean_short_EEGvar[i] > bins[9] and rolmean_short_EEGvar[i] < bins[10]:
# #         pro_eegvars_time_false.append(0.035789582)
# #         pro_eegvars_time.append(0)
# #     elif rolmean_short_EEGvar[i] >= bins[10] and rolmean_short_EEGvar[i] <= bins[11]:
# #         pro_eegvars_time_false.append(0.032807117)
# #         pro_eegvars_time.append(0)
# #     elif rolmean_short_EEGvar[i] > bins[11] and rolmean_short_EEGvar[i] < bins[12]:
# #         pro_eegvars_time_false.append(0.06947087)
# #         pro_eegvars_time.append(0)
# #     elif rolmean_short_EEGvar[i] > bins[12] and rolmean_short_EEGvar[i] < bins[13]:
# #         pro_eegvars_time_false.append(0.164395537)
# #         pro_eegvars_time.append(0.333333333)
# #     elif rolmean_short_EEGvar[i] >= bins[13] and rolmean_short_EEGvar[i] < bins[14]:
# #         pro_eegvars_time_false.append(0.179256441)
# #         pro_eegvars_time.append(0)
# #     elif rolmean_short_EEGvar[i] >= bins[14] and rolmean_short_EEGvar[i] < bins[15]:
# #         pro_eegvars_time_false.append(0)
# #         pro_eegvars_time.append(0)
# #     elif rolmean_short_EEGvar[i] >= bins[15] and rolmean_short_EEGvar[i] < bins[16]:
# #         pro_eegvars_time_false.append(0)
# #         pro_eegvars_time.append(0)
# #     elif rolmean_short_EEGvar[i] >= bins[16] and rolmean_short_EEGvar[i] < bins[17]:
# #         pro_eegvars_time_false.append(0)
# #         pro_eegvars_time.append(0)
# #     elif rolmean_short_EEGvar[i] >= bins[17]:
# #         pro_eegvars_time_false.append(0)
# #         pro_eegvars_time.append(0)
# # pro_eegautos_time = []
# # pro_eegautos_time_false = []
# # for i in range(len(rolmean_short_EEGauto)):
# #     if rolmean_short_EEGauto[i] >= bins[0] and rolmean_short_EEGauto[i] < bins[1]:
# #         pro_eegautos_time_false.append(0)
# #         pro_eegautos_time.append(0)
# #     elif rolmean_short_EEGauto[i] >= bins[1] and rolmean_short_EEGauto[i] < bins[2]:
# #         pro_eegautos_time_false.append(0)
# #         pro_eegautos_time.append(0)
# #     elif rolmean_short_EEGauto[i] >= bins[2] and rolmean_short_EEGauto[i] < bins[3]:
# #         pro_eegautos_time_false.append(0)
# #         pro_eegautos_time.append(0)
# #     elif rolmean_short_EEGauto[i] >= bins[3] and rolmean_short_EEGauto[i] < bins[4]:
# #         pro_eegautos_time_false.append(0)
# #         pro_eegautos_time.append(0)
# #     elif rolmean_short_EEGauto[i] >= bins[4] and rolmean_short_EEGauto[i] < bins[5]:
# #         pro_eegautos_time_false.append(0.004576541)
# #         pro_eegautos_time.append(0)
# #     elif rolmean_short_EEGauto[i] >= bins[5] and rolmean_short_EEGauto[i] < bins[6]:
# #         pro_eegautos_time_false.append(0.006016352)
# #         pro_eegautos_time.append(0)
# #     elif rolmean_short_EEGauto[i] >= bins[6] and rolmean_short_EEGauto[i] < bins[7]:
# #         pro_eegautos_time_false.append(0.03290996)
# #         pro_eegautos_time.append(0)
# #     elif rolmean_short_EEGauto[i] >= bins[7] and rolmean_short_EEGauto[i] < bins[8]:
# #         pro_eegautos_time_false.append(0.124132257)
# #         pro_eegautos_time.append(0)
# #     elif rolmean_short_EEGauto[i] >= bins[8] and rolmean_short_EEGauto[i] < bins[9]:
# #         pro_eegautos_time_false.append(0.32642567)
# #         pro_eegautos_time.append(0.666666667)
# #     elif rolmean_short_EEGauto[i] >= bins[9] and rolmean_short_EEGauto[i] <= bins[10]:
# #         pro_eegautos_time_false.append(0.351313827)
# #         pro_eegautos_time.append(0.333333333)
# #     elif rolmean_short_EEGauto[i] > bins[10] and rolmean_short_EEGauto[i] < bins[11]:
# #         pro_eegautos_time_false.append(0.108551448)
# #         pro_eegautos_time.append(0)
# #     elif rolmean_short_EEGauto[i] >= bins[11] and rolmean_short_EEGauto[i] < bins[12]:
# #         pro_eegautos_time_false.append(0.031161619)
# #         pro_eegautos_time.append(0)
# #     elif rolmean_short_EEGauto[i] >= bins[12] and rolmean_short_EEGauto[i] < bins[13]:
# #         pro_eegautos_time_false.append(0.011415643)
# #         pro_eegautos_time.append(0)
# #     elif rolmean_short_EEGauto[i] >= bins[13] and rolmean_short_EEGauto[i] < bins[14]:
# #         pro_eegautos_time_false.append(0.003496683)
# #         pro_eegautos_time.append(0)
# #     elif rolmean_short_EEGauto[i] >= bins[14] and rolmean_short_EEGauto[i] < bins[15]:
# #         pro_eegautos_time_false.append(0)
# #         pro_eegautos_time.append(0)
# #     elif rolmean_short_EEGauto[i] >= bins[15] and rolmean_short_EEGauto[i] < bins[16]:
# #         pro_eegautos_time_false.append(0)
# #         pro_eegautos_time.append(0)
# #     elif rolmean_short_EEGauto[i] >= bins[16] and rolmean_short_EEGauto[i] < bins[17]:
# #         pro_eegautos_time_false.append(0)
# #         pro_eegautos_time.append(0)
# #     elif rolmean_short_EEGauto[i] >= bins[17]:
# #         pro_eegautos_time_false.append(0)
# #         pro_eegautos_time.append(0)
# # pro_RRIautos_time = []
# # pro_RRIautos_time_false = []
# # for i in range(len(rolmean_short_RRIauto)):
# #     if rolmean_short_RRIauto[i] >= bins[0] and rolmean_short_RRIauto[i] < bins[1]:
# #         pro_RRIautos_time_false.append(0)
# #         pro_RRIautos_time.append(0)
# #     elif rolmean_short_RRIauto[i] >= bins[1] and rolmean_short_RRIauto[i] < bins[2]:
# #         pro_RRIautos_time_false.append(0)
# #         pro_RRIautos_time.append(0)
# #     elif rolmean_short_RRIauto[i] >= bins[2] and rolmean_short_RRIauto[i] < bins[3]:
# #         pro_RRIautos_time_false.append(0)
# #         pro_RRIautos_time.append(0)
# #     elif rolmean_short_RRIauto[i] >= bins[3] and rolmean_short_RRIauto[i] < bins[4]:
# #         pro_RRIautos_time_false.append(0)
# #         pro_RRIautos_time.append(0)
# #     elif rolmean_short_RRIauto[i] >= bins[4] and rolmean_short_RRIauto[i] < bins[5]:
# #         pro_RRIautos_time_false.append(0)
# #         pro_RRIautos_time.append(0)
# #     elif rolmean_short_RRIauto[i] >= bins[5] and rolmean_short_RRIauto[i] < bins[6]:
# #         pro_RRIautos_time_false.append(0.00226256)
# #         pro_RRIautos_time.append(0)
# #     elif rolmean_short_RRIauto[i] >= bins[6] and rolmean_short_RRIauto[i] < bins[7]:
# #         pro_RRIautos_time_false.append(0.006839101)
# #         pro_RRIautos_time.append(0)
# #     elif rolmean_short_RRIauto[i] >= bins[7] and rolmean_short_RRIauto[i] <= bins[8]:
# #         pro_RRIautos_time_false.append(0.060677739)
# #         pro_RRIautos_time.append(0.333333333)
# #     elif rolmean_short_RRIauto[i] > bins[8] and rolmean_short_RRIauto[i] <= bins[9]:
# #         pro_RRIautos_time_false.append(0.409111945)
# #         pro_RRIautos_time.append(0.666666667)
# #     elif rolmean_short_RRIauto[i] > bins[9] and rolmean_short_RRIauto[i] <= bins[10]:
# #         pro_RRIautos_time_false.append(0.465470252)
# #         pro_RRIautos_time.append(0)
# #     elif rolmean_short_RRIauto[i] > bins[10] and rolmean_short_RRIauto[i] < bins[11]:
# #         pro_RRIautos_time_false.append(0.05106186)
# #         pro_RRIautos_time.append(0)
# #     elif rolmean_short_RRIauto[i] >= bins[11] and rolmean_short_RRIauto[i] < bins[12]:
# #         pro_RRIautos_time_false.append(0.00395948)
# #         pro_RRIautos_time.append(0)
# #     elif rolmean_short_RRIauto[i] >= bins[12] and rolmean_short_RRIauto[i] < bins[13]:
# #         pro_RRIautos_time_false.append(0.000462796)
# #         pro_RRIautos_time.append(0)
# #     elif rolmean_short_RRIauto[i] >= bins[13] and rolmean_short_RRIauto[i] < bins[14]:
# #         pro_RRIautos_time_false.append(0.000154265)
# #         pro_RRIautos_time.append(0)
# #     elif rolmean_short_RRIauto[i] >= bins[14] and rolmean_short_RRIauto[i] < bins[15]:
# #         pro_RRIautos_time_false.append(0)
# #         pro_RRIautos_time.append(0)
# #     elif rolmean_short_RRIauto[i] >= bins[15] and rolmean_short_RRIauto[i] < bins[16]:
# #         pro_RRIautos_time_false.append(0)
# #         pro_RRIautos_time.append(0)
# #     elif rolmean_short_RRIauto[i] >= bins[16] and rolmean_short_RRIauto[i] < bins[17]:
# #         pro_RRIautos_time_false.append(0)
# #         pro_RRIautos_time.append(0)
# #     elif rolmean_short_RRIauto[i] >= bins[17]:
# #         pro_RRIautos_time_false.append(0)
# #         pro_RRIautos_time.append(0)



Pseizureeegvar = 0.000154242;
Pnonseizureeegvar = 0.999845758;

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_eegautos_time)):
#     P1=pro_RRIautos_time[m]*Pseizureeegvar*pro_eegautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIautos_time_false[m]*pro_eegautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))

Pcombined = []
for m in range(len(pro_RRIautos_time)):
    P1=Pseizureeegvar*pro_RRIautos_time[m]
    P2=Pnonseizureeegvar*(1-pro_RRIautos_time_false[m])
    Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_eegautos_time)):
#     P1=pro_eegautos_time[m]*Pseizureeegvar
#     P2=Pnonseizureeegvar*(1-pro_eegautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))

pyplot.figure(figsize=(8,4))
RRI_timewindow_arr=t
pyplot.plot(RRI_timewindow_arr,Pcombined)
pyplot.annotate('',xy=(139.3575,np.max(Pcombined)),xytext=(139.3575,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='k',shrink=0.05))
pyplot.annotate('',xy=(145.3427,np.max(Pcombined)),xytext=(145.3427,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='k',shrink=0.05))
pyplot.annotate('',xy=(152.11,np.max(Pcombined)),xytext=(152.11,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='k',shrink=0.05))
pyplot.annotate('',xy=(160.5378,np.max(Pcombined)),xytext=(160.5378,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='k',shrink=0.05))
pyplot.title('Forecast seizures in VIC0829')
pyplot.xlabel('Time(h)')
pyplot.ylabel('Seizure probability')
pyplot.show()

Pcombined_X=Pcombined
Pcombined=split(Pcombined,6)
print(len(Pcombined))
index=[]
for i in range(len(Pcombined)):
    for item in Pcombined[i]:
        if item >= Th1:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[139.3575,145.3427,152.11,160.5378]
k=0
n_arr=[]
pretime=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k=k+1
            n_arr.append(n)
            pretime.append(m - n)
print(k)
index=[]
for i in range(len(Pcombined)):
    for item in Pcombined[i]:
        if item >= 0.3*Th1:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[139.3575,145.3427,152.11,160.5378]
k=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k=k+1
            n_arr.append(n)
            pretime.append(m - n)
print(k)
index=[]
for i in range(len(Pcombined)):
    for item in Pcombined[i]:
        if item >= 0.6*Th1:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[139.3575,145.3427,152.11,160.5378]
k=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k=k+1
            n_arr.append(n)
            pretime.append(m - n)
print(k)
index=[]
for i in range(len(Pcombined)):
    for item in Pcombined[i]:
        if item >= 1.2*Th1:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[139.3575,145.3427,152.11,160.5378]
k=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k=k+1
            n_arr.append(n)
            pretime.append(m - n)
print(k)
index=[]
for i in range(len(Pcombined)):
    for item in Pcombined[i]:
        if item >= 2*Th1:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[139.3575,145.3427,152.11,160.5378]
k=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k=k+1
            n_arr.append(n)
            pretime.append(m - n)
print(k)
print(pretime)
print(np.mean(pretime))

Pcombined = split(Pcombined_X, 6)
print(len(Pcombined))
time_arr_arr=[]
AUC_cs_arr=[]
for i in range(50000):
    time_arr = np.random.uniform(low=t_window_arr[19450], high=t_window_arr[-1], size=4)
    time_arr_arr.append(time_arr)
    time_arr=np.sort(time_arr)

    index = []
    for i in range(len(Pcombined)):
        for item in Pcombined[i]:
            if item >= Th1:
                index.append(6 * i + 0)
    # print(RRI_timewindow_arr[index])
    a1 = np.unique(RRI_timewindow_arr[index])
    # print(a1);
    # print(len(a1))
    k1 = 0
    n_arr = []
    for m in time_arr:
        for n in a1:
            if m - n <= 1 and m - n >= 0:
                k1 = k1 + 1
                n_arr.append(n)
    # print(k1)

    index = []
    for i in range(len(Pcombined)):
        for item in Pcombined[i]:
            if item >= 0.3 * Th1:
                index.append(6 * i + 0)
    # print(RRI_timewindow_arr[index])
    a2 = np.unique(RRI_timewindow_arr[index])
    # print(a2);
    # print(len(a2))
    k2 = 0
    n_arr = []
    for m in time_arr:
        for n in a2:
            if m - n <= 1 and m - n >= 0:
                k2 = k2 + 1
                n_arr.append(n)
    # print(k2)

    index = []
    for i in range(len(Pcombined)):
        for item in Pcombined[i]:
            if item >= 0.6 * Th1:
                index.append(6 * i + 0)
    # print(RRI_timewindow_arr[index])
    a3 = np.unique(RRI_timewindow_arr[index])
    # print(a3);
    # print(len(a3))
    k3 = 0
    n_arr = []
    for m in time_arr:
        for n in a3:
            if m - n <= 1 and m - n >= 0:
                k3 = k3 + 1
                n_arr.append(n)
    # print(k3)

    index = []
    for i in range(len(Pcombined)):
        for item in Pcombined[i]:
            if item >= 1.2 * Th1:
                index.append(6 * i + 0)
    # print(RRI_timewindow_arr[index])
    a4 = np.unique(RRI_timewindow_arr[index])
    # print(a);
    # print(len(a4))
    k4 = 0
    n_arr = []
    for m in time_arr:
        for n in a4:
            if m - n <= 1 and m - n >= 0:
                k4 = k4 + 1
                n_arr.append(n)
    # print(k4)

    index = []
    for i in range(len(Pcombined)):
        for item in Pcombined[i]:
            if item >= 2 * Th1:
                index.append(6 * i + 0)
    # print(RRI_timewindow_arr[index])
    a5 = np.unique(RRI_timewindow_arr[index])
    # print(a5);
    # print(len(a5))
    k5 = 0
    n_arr = []
    for m in time_arr:
        for n in a5:
            if m - n <= 1 and m - n >= 0:
                k5 = k5 + 1
                n_arr.append(n)
    # print(k5)

    Sen1 = k1 / len(time_arr);
    Sen2 = k2 / len(time_arr);
    Sen3 = k3 / len(time_arr);
    Sen4 = k4 / len(time_arr);
    Sen5 = k5 / len(time_arr);
    FPR1 = (len(a1) - k1) / len(Pcombined);
    FPR2 = (len(a2) - k2) / len(Pcombined);
    FPR3 = (len(a3) - k3) / len(Pcombined);
    FPR4 = (len(a4) - k4) / len(Pcombined);
    FPR5 = (len(a5) - k5) / len(Pcombined);
    Sen_arr_CS = [0, Sen1, Sen2, Sen3, Sen4, Sen5, 1]
    FPR_arr_CS = [0, FPR1, FPR2, FPR3, FPR4, FPR5, 1]
    from sklearn.metrics import auc

    AUC_cs = auc(np.sort(FPR_arr_CS), np.sort(Sen_arr_CS))
    # print(AUC_cs)
    AUC_cs_arr.append(AUC_cs)

# print(AUC_cs_arr)
# print(time_arr_arr)
# np.savetxt("C:/Users/wxiong/PycharmProjects/EEGstudy/VIC0829chance/AUC_ECG_auto_12h_VIC0829_2022.csv", AUC_cs_arr, delimiter=",", fmt='%s')
# np.savetxt("C:/Users/wxiong/PycharmProjects/EEGstudy/VIC0829chance/seizure_labels_ECG_auto_12h_VIC0829_2022.csv", time_arr_arr, delimiter=",", fmt='%s')
np.savetxt("C:/Users/wxiong/Documents/PHD/2011.1/VIC0829/chance/AUC_ECGauto_6h_VIC0829_2022.csv", AUC_cs_arr, delimiter=",", fmt='%s')




# # t1=np.linspace(0+0.00416667,0+0.00416667+0.00416667*(len(Raw_variance_EEG_arr)-1),len(Raw_variance_EEG_arr))
# # a=np.where(t1<10.4536+0)
# # t1[0:2508]=t1[0:2508]-0+13.5463889
# # t1[2508:]=t1[2508:]-10.4536-0
# # time_feature_arr=[]
# # for i in range(len(t1)):
# #     if t1[i]>24:
# #         time_feature_arr.append(t1[i] - (t1[i] // 24) * 24)
# #     else:
# #         time_feature_arr.append(t1[i])
# #
# # print(len(time_feature_arr))
# # time_arr=time_feature_arr[19450:]
# # print(len(time_arr))
# # new_arr=[]
# # for j in range(0,486):
# #     new_arr.append(time_arr[40*j])
# #
# # bins_number = 18
# # bins = np.linspace(0, 24, bins_number + 1)
# # pro_circadian_time=[]
# # pro_circadian_time_false=[]
# # for i in range(len(new_arr)):
# #     if new_arr[i] >= bins[0] and new_arr[i] <= bins[1]:
# #         pro_circadian_time_false.append(0.049364941)
# #         pro_circadian_time.append(0)
# #     elif new_arr[i] > bins[1] and new_arr[i] < bins[2]:
# #         pro_circadian_time_false.append(0.049364941)
# #         pro_circadian_time.append(0)
# #     elif new_arr[i] >= bins[2] and new_arr[i] < bins[3]:
# #         pro_circadian_time_false.append(0.049364941)
# #         pro_circadian_time.append(0)
# #     elif new_arr[i] >= bins[3] and new_arr[i] < bins[4]:
# #         pro_circadian_time_false.append(0.049364941)
# #         pro_circadian_time.append(0)
# #     elif new_arr[i] >= bins[4] and new_arr[i] < bins[5]:
# #         pro_circadian_time_false.append(0.049364941)
# #         pro_circadian_time.append(0)
# #     elif new_arr[i] >= bins[5] and new_arr[i] <= bins[6]:
# #         pro_circadian_time_false.append(0.049364941)
# #         pro_circadian_time.append(0)
# #     elif new_arr[i] > bins[6] and new_arr[i] < bins[7]:
# #         pro_circadian_time_false.append(0.049313519)
# #         pro_circadian_time.append(0.333333333)
# #     elif new_arr[i] >= bins[7] and new_arr[i] <= bins[8]:
# #         pro_circadian_time_false.append(0.049313519)
# #         pro_circadian_time.append(0.333333333)
# #     elif new_arr[i] > bins[8] and new_arr[i] < bins[9]:
# #         pro_circadian_time_false.append(0.05106186)
# #         pro_circadian_time.append(0)
# #     elif new_arr[i] >= bins[9] and new_arr[i] < bins[10]:
# #         pro_circadian_time_false.append(0.065819921)
# #         pro_circadian_time.append(0)
# #     elif new_arr[i] >= bins[10] and new_arr[i] < bins[11]:
# #         pro_circadian_time_false.append(0.065819921)
# #         pro_circadian_time.append(0)
# #     elif new_arr[i] >= bins[11] and new_arr[i] < bins[12]:
# #         pro_circadian_time_false.append(0.065819921)
# #         pro_circadian_time.append(0)
# #     elif new_arr[i] >= bins[12] and new_arr[i] < bins[13]:
# #         pro_circadian_time_false.append(0.065819921)
# #         pro_circadian_time.append(0)
# #     elif new_arr[i] >= bins[13] and new_arr[i] < bins[14]:
# #         pro_circadian_time_false.append(0.065768499)
# #         pro_circadian_time.append(0.333333333)
# #     elif new_arr[i] >= bins[14] and new_arr[i] < bins[15]:
# #         pro_circadian_time_false.append(0.065819921)
# #         pro_circadian_time.append(0)
# #     elif new_arr[i] >= bins[15] and new_arr[i] < bins[16]:
# #         pro_circadian_time_false.append(0.060523474)
# #         pro_circadian_time.append(0)
# #     elif new_arr[i] >= bins[16] and new_arr[i] < bins[17]:
# #         pro_circadian_time_false.append(0.049364941)
# #         pro_circadian_time.append(0)
# #     elif new_arr[i] >= bins[17] and new_arr[i] <= bins[18]:
# #         pro_circadian_time_false.append(0.049364941)
# #         pro_circadian_time.append(0)
# #
# #
# # # RRI_timewindow_arr=t[0:len(pro_circadian_time)]
# # # print(RRI_timewindow_arr[-1]-RRI_timewindow_arr[0])
# # # pyplot.figure(figsize=(8,4))
# # # pyplot.plot(RRI_timewindow_arr,pro_circadian_time)
# # # pyplot.annotate('',xy=(139.3575,np.max(pro_circadian_time)),xytext=(139.3575,np.max(pro_circadian_time)+0.000000000001),arrowprops=dict(facecolor='k',shrink=0.05))
# # # pyplot.annotate('',xy=(145.3427,np.max(pro_circadian_time)),xytext=(145.3427,np.max(pro_circadian_time)+0.000000000001),arrowprops=dict(facecolor='r',shrink=0.05))
# # # pyplot.annotate('',xy=(152.11,np.max(pro_circadian_time)),xytext=(152.11,np.max(pro_circadian_time)+0.00000000001),arrowprops=dict(facecolor='r',shrink=0.05))
# # # pyplot.annotate('',xy=(160.5378,np.max(pro_circadian_time)),xytext=(160.5378,np.max(pro_circadian_time)+0.000000000001),arrowprops=dict(facecolor='r',shrink=0.05))
# # # pyplot.hlines(1/3, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'r')
# # # # pyplot.hlines(0.11111, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'orange')
# # # # pyplot.hlines(0.22222, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'orange')
# # # pyplot.title('Forecast seizures in VIC0829')
# # # pyplot.xlabel('Time(h)')
# # # pyplot.ylabel('Seizure probability')
# # # pyplot.show()
# # # Pcombined=split(pro_circadian_time,6)
# # # print(len(Pcombined))
# # # index=[]
# # # for i in range(len(Pcombined)):
# # #     for item in Pcombined[i]:
# # #         if item >= 1/3:
# # #             index.append(6*i+0)
# # # print(RRI_timewindow_arr[index])
# # # a=np.unique(RRI_timewindow_arr[index])
# # # print(a); print(len(a))
# # # time_arr=[139.3575,145.3427,152.11,160.5378]
# # # k=0
# # # n_arr=[]
# # # for m in time_arr:
# # #     for n in a:
# # #         if m-n<=1 and m-n>=0:
# # #             k=k+1
# # #             n_arr.append(n)
# # # print(k)
# # # index=[]
# # # for i in range(len(Pcombined)):
# # #     for item in Pcombined[i]:
# # #         if item >= 0.3*1/3:
# # #             index.append(6*i+0)
# # # print(RRI_timewindow_arr[index])
# # # a=np.unique(RRI_timewindow_arr[index])
# # # print(a); print(len(a))
# # # time_arr=[139.3575,145.3427,152.11,160.5378]
# # # k=0
# # # n_arr=[]
# # # for m in time_arr:
# # #     for n in a:
# # #         if m-n<=1 and m-n>=0:
# # #             k=k+1
# # #             n_arr.append(n)
# # # print(k)
# # # index=[]
# # # for i in range(len(Pcombined)):
# # #     for item in Pcombined[i]:
# # #         if item >= 0.6*1/3:
# # #             index.append(6*i+0)
# # # print(RRI_timewindow_arr[index])
# # # a=np.unique(RRI_timewindow_arr[index])
# # # print(a); print(len(a))
# # # time_arr=[139.3575,145.3427,152.11,160.5378]
# # # k=0
# # # n_arr=[]
# # # for m in time_arr:
# # #     for n in a:
# # #         if m-n<=1 and m-n>=0:
# # #             k=k+1
# # #             n_arr.append(n)
# # # print(k)
# # # index=[]
# # # for i in range(len(Pcombined)):
# # #     for item in Pcombined[i]:
# # #         if item >= 1.2*1/3:
# # #             index.append(6*i+0)
# # # print(RRI_timewindow_arr[index])
# # # a=np.unique(RRI_timewindow_arr[index])
# # # print(a); print(len(a))
# # # time_arr=[139.3575,145.3427,152.11,160.5378]
# # # k=0
# # # n_arr=[]
# # # for m in time_arr:
# # #     for n in a:
# # #         if m-n<=1 and m-n>=0:
# # #             k=k+1
# # #             n_arr.append(n)
# # # print(k)
# # # index=[]
# # # for i in range(len(Pcombined)):
# # #     for item in Pcombined[i]:
# # #         if item >= 2*1/3:
# # #             index.append(6*i+0)
# # # print(RRI_timewindow_arr[index])
# # # a=np.unique(RRI_timewindow_arr[index])
# # # print(a); print(len(a))
# # # time_arr=[139.3575,145.3427,152.11,160.5378]
# # # k=0
# # # n_arr=[]
# # # for m in time_arr:
# # #     for n in a:
# # #         if m-n<=1 and m-n>=0:
# # #             k=k+1
# # #             n_arr.append(n)
# # # print(k)
# #
# #
# #
# #
# # Pseizureeegvar = 0.000154242;
# # Pnonseizureeegvar = 0.999845758;
# #
# # # Pcombined = []
# # # for m in range(len(pro_circadian_time)):
# # #     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_RRIautos_time[m]*pro_circadian_time[m]
# # #     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
# # #     Pcombined.append(P1/(P1+P2))
# # # print(len(Pcombined))
# #
# # # Pcombined = []
# # # for m in range(len(pro_eegautos_time)):
# # #     P1=pro_RRIautos_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_circadian_time[m]
# # #     P2=Pnonseizureeegvar*(1-pro_RRIautos_time_false[m]*pro_eegautos_time_false[m]*pro_circadian_time_false[m])
# # #     Pcombined.append(P1/(P1+P2))
# #
# # # Pcombined = []
# # # for m in range(len(pro_circadian_time)):
# # #     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_circadian_time[m]
# # #     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_circadian_time_false[m])
# # #     Pcombined.append(P1/(P1+P2))
# # # print(len(Pcombined))
# #
# # # Pcombined = []
# # # for m in range(len(pro_circadian_time)):
# # #     P1=Pseizureeegvar*pro_RRIautos_time[m]*pro_circadian_time[m]
# # #     P2=Pnonseizureeegvar*(1-pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
# # #     Pcombined.append(P1/(P1+P2))
# # # print(len(Pcombined))
# #
# # # Pcombined = []
# # # for m in range(len(pro_circadian_time)):
# # #     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_circadian_time[m]
# # #     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_circadian_time_false[m])
# # #     Pcombined.append(P1/(P1+P2))
# # # print(len(Pcombined))
# #
# # # Pcombined = []
# # # for m in range(len(pro_circadian_time)):
# # #     P1=pro_eegautos_time[m]*Pseizureeegvar*pro_circadian_time[m]
# # #     P2=Pnonseizureeegvar*(1-pro_eegautos_time_false[m]*pro_circadian_time_false[m])
# # #     Pcombined.append(P1/(P1+P2))
# # # print(len(Pcombined))
# #
# # # pyplot.figure(figsize=(8,4))
# # # RRI_timewindow_arr=t[0:len(pro_circadian_time)]
# # # pyplot.plot(RRI_timewindow_arr,Pcombined)
# # # pyplot.annotate('',xy=(139.3575,np.max(Pcombined)),xytext=(139.3575,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='k',shrink=0.05))
# # # pyplot.annotate('',xy=(145.3427,np.max(Pcombined)),xytext=(145.3427,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='r',shrink=0.05))
# # # pyplot.annotate('',xy=(152.11,np.max(Pcombined)),xytext=(152.11,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='r',shrink=0.05))
# # # pyplot.annotate('',xy=(160.5378,np.max(Pcombined)),xytext=(160.5378,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='r',shrink=0.05))
# # # pyplot.title('Forecast seizures in VIC0829')
# # # pyplot.xlabel('Time(h)')
# # # pyplot.ylabel('Seizure probability')
# # # pyplot.show()
# # #
# # # Pcombined=split(Pcombined,6)
# # # print(len(Pcombined))
# # # index=[]
# # # for i in range(len(Pcombined)):
# # #     for item in Pcombined[i]:
# # #         if item >= Th2:
# # #             index.append(6*i+0)
# # # print(RRI_timewindow_arr[index])
# # # a=np.unique(RRI_timewindow_arr[index])
# # # print(a); print(len(a))
# # # time_arr=[139.3575,145.3427,152.11,160.5378]
# # # k=0
# # # n_arr=[]
# # # for m in time_arr:
# # #     for n in a:
# # #         if m-n<=1 and m-n>=0:
# # #             k=k+1
# # #             n_arr.append(n)
# # # print(k)
# # # index=[]
# # # for i in range(len(Pcombined)):
# # #     for item in Pcombined[i]:
# # #         if item >= 0.3*Th2:
# # #             index.append(6*i+0)
# # # print(RRI_timewindow_arr[index])
# # # a=np.unique(RRI_timewindow_arr[index])
# # # print(a); print(len(a))
# # # time_arr=[139.3575,145.3427,152.11,160.5378]
# # # k=0
# # # n_arr=[]
# # # for m in time_arr:
# # #     for n in a:
# # #         if m-n<=1 and m-n>=0:
# # #             k=k+1
# # #             n_arr.append(n)
# # # print(k)
# # # index=[]
# # # for i in range(len(Pcombined)):
# # #     for item in Pcombined[i]:
# # #         if item >= 0.6*Th2:
# # #             index.append(6*i+0)
# # # print(RRI_timewindow_arr[index])
# # # a=np.unique(RRI_timewindow_arr[index])
# # # print(a); print(len(a))
# # # time_arr=[139.3575,145.3427,152.11,160.5378]
# # # k=0
# # # n_arr=[]
# # # for m in time_arr:
# # #     for n in a:
# # #         if m-n<=1 and m-n>=0:
# # #             k=k+1
# # #             n_arr.append(n)
# # # print(k)
# # # index=[]
# # # for i in range(len(Pcombined)):
# # #     for item in Pcombined[i]:
# # #         if item >= 1.2*Th2:
# # #             index.append(6*i+0)
# # # print(RRI_timewindow_arr[index])
# # # a=np.unique(RRI_timewindow_arr[index])
# # # print(a); print(len(a))
# # # time_arr=[139.3575,145.3427,152.11,160.5378]
# # # k=0
# # # n_arr=[]
# # # for m in time_arr:
# # #     for n in a:
# # #         if m-n<=1 and m-n>=0:
# # #             k=k+1
# # #             n_arr.append(n)
# # # print(k)
# # # index=[]
# # # for i in range(len(Pcombined)):
# # #     for item in Pcombined[i]:
# # #         if item >= 2*Th2:
# # #             index.append(6*i+0)
# # # print(RRI_timewindow_arr[index])
# # # a=np.unique(RRI_timewindow_arr[index])
# # # print(a); print(len(a))
# # # time_arr=[139.3575,145.3427,152.11,160.5378]
# # # k=0
# # # n_arr=[]
# # # for m in time_arr:
# # #     for n in a:
# # #         if m-n<=1 and m-n>=0:
# # #             k=k+1
# # #             n_arr.append(n)
# # # print(k)