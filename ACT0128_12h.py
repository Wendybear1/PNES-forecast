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


csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/ACT0128/Cz_EEGvariance_ACT0128_15s_3h.csv', sep=',',
                         header=None)
Raw_variance_EEG = csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/ACT0128/Cz_EEGauto_ACT0128_15s_3h.csv', sep=',',
                         header=None)
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

print(len(t_window_arr));
print(t_window_arr[0]);
print(t_window_arr[-1] - t_window_arr[0]);
print(t_window_arr[17280]);
print(t_window_arr[17280] - t_window_arr[0]);
print(t_window_arr[-1] - t_window_arr[17280]);

window_time_arr = t_window_arr
# pyplot.plot(window_time_arr, Raw_variance_EEG_arr, 'grey', alpha=0.5)
# pyplot.ylabel('Voltage', fontsize=13)
# pyplot.xlabel('Time (hours)', fontsize=13)
# pyplot.title('raw EEG variance in ACT0128', fontsize=13)
# pyplot.show()
var_arr = []
for item in Raw_variance_EEG_arr:
    if item < 1e-8:
        var_arr.append(item)
    else:
        var_arr.append(var_arr[-1])
Raw_variance_EEG = var_arr
# pyplot.plot(window_time_arr, Raw_variance_EEG, 'grey', alpha=0.5)
# pyplot.ylabel('Voltage', fontsize=13)
# pyplot.xlabel('Time (hours)', fontsize=13)
# pyplot.title('EEG variance in ACT0128', fontsize=13)
# pyplot.show()

seizure_timing_index = []
for k in range(len(window_time_arr)):
    if window_time_arr[k] < 7.0936 and window_time_arr[k + 1] >= 7.0936:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 28.258611 and window_time_arr[k + 1] >= 28.258611:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 47.048333 and window_time_arr[k + 1] >= 47.048333:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 56.151389 and window_time_arr[k + 1] >= 56.151389:
        seizure_timing_index.append(k)
    # if window_time_arr[k] < 76.88111 and window_time_arr[k + 1] >= 76.88111:
    #     seizure_timing_index.append(k)
    # if window_time_arr[k] < 97.923611 and window_time_arr[k + 1] >= 97.923611:
    #     seizure_timing_index.append(k)
    # if window_time_arr[k] < 119.25278 and window_time_arr[k + 1] >= 119.25278:
    #     seizure_timing_index.append(k)
print(seizure_timing_index)

# seizure_timing_index = []
# for k in range(len(window_time_arr)):
#     if window_time_arr[k] < 30.9036 and window_time_arr[k + 1] >= 30.9036:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 119.44028 and window_time_arr[k + 1] >= 119.44028:
#         seizure_timing_index.append(k)
# print(seizure_timing_index)


# # # # # # # ### EEG variance
window_time_arr = t_window_arr[0:17280]
Raw_variance_EEG = Raw_variance_EEG[0:17280]
# window_time_arr=t_window_arr
# Raw_variance_EEG=Raw_variance_EEG

long_rhythm_var_arr = movingaverage(Raw_variance_EEG, 5760)
medium_rhythm_var_arr = movingaverage(Raw_variance_EEG, 240)
medium_rhythm_var_arr_2 = movingaverage(Raw_variance_EEG, 240 * 3)
medium_rhythm_var_arr_3 = movingaverage(Raw_variance_EEG, 240 * 6)
medium_rhythm_var_arr_4 = movingaverage(Raw_variance_EEG, 240 * 12)
short_rhythm_var_arr_plot = movingaverage(Raw_variance_EEG, 240*24)


long_rhythm_var_arr = short_rhythm_var_arr_plot * (10 ** 12)
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
phase_long_EEGvariance_arr = var_phase
seizure_phase_var_long = []
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
# width = 2 * np.pi / bins_number
# params = dict(projection='polar')
# fig, ax2 = pyplot.subplots(subplot_kw=params)
# ax2.bar(bins[:bins_number], nEEGsvarsei / sum(nEEGsvarsei), width=width, color='grey', alpha=0.7, linewidth=2,
#         edgecolor='k')
# locs, labels = pyplot.yticks([0.1, 0.3, 0.5], ['0.1', '0.3', '0.5'], fontsize=16)
# ax2.set_title('EEG variance', fontsize=16)
# ax2.set_xticklabels(['0', '0.25$\pi$', '0.5$\pi$', '0.75$\pi$', '-$\pi$', '-0.75$\pi$', '-0.5$\pi$', '-0.25$\pi$', ],
#                     fontsize=16)
# pyplot.show()







# pyplot.plot(t_window_arr, Raw_auto_EEG_arr, 'grey', alpha=0.5)
# pyplot.xlabel('Time (hours)', fontsize=13)
# pyplot.title('raw EEG autocorrelation in ACT0128', fontsize=13)
# pyplot.show()
value_arr = []
for item in Raw_auto_EEG_arr:
    if item < 500:
        value_arr.append(item)
    else:
        value_arr.append(value_arr[-1])
Raw_auto_EEG_arr = value_arr
# pyplot.plot(t_window_arr, Raw_auto_EEG_arr, 'grey', alpha=0.5)
# pyplot.xlabel('Time (hours)', fontsize=13)
# pyplot.title('EEG autocorrelation in ACT0128', fontsize=13)
# pyplot.show()

Raw_auto_EEG = Raw_auto_EEG_arr[0:17280]
window_time_arr = t_window_arr[0:17280]
# Raw_auto_EEG=Raw_auto_EEG_arr
# window_time_arr=t_window_arr

long_rhythm_value_arr = movingaverage(Raw_auto_EEG, 5760)
medium_rhythm_value_arr = movingaverage(Raw_auto_EEG, 240)
medium_rhythm_value_arr_2 = movingaverage(Raw_auto_EEG, 240 * 3)
medium_rhythm_value_arr_3 = movingaverage(Raw_auto_EEG, 240 * 6)
medium_rhythm_value_arr_4 = movingaverage(Raw_auto_EEG, 240 * 12)
short_rhythm_value_arr_plot = movingaverage(Raw_auto_EEG, 240*24)

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
phase_long_EEGauto_arr = value_phase
seizure_phase_value_long = []
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
# width = 2 * np.pi / bins_number
# params = dict(projection='polar')
# fig, ax2 = pyplot.subplots(subplot_kw=params)
# ax2.bar(bins[:bins_number], nEEGsautosei / sum(nEEGsautosei), width=width, color='grey', alpha=0.7, linewidth=2,
#         edgecolor='k')
# locs, labels = pyplot.yticks([0.1, 0.3, 0.5], ['0.1', '0.3', '0.5'], fontsize=16)
# ax2.set_title('EEG autocorrelation', fontsize=16)
# ax2.set_xticklabels(['0', '0.25$\pi$', '0.5$\pi$', '0.75$\pi$', '-$\pi$', '-0.75$\pi$', '-0.5$\pi$', '-0.25$\pi$', ],
#                     fontsize=16)
# pyplot.show()







# # ### ECG data
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/ACT0128/RRI_ch31_timewindowarr_ACT0128_15s_3h.csv',
                         sep=',', header=None)
rri_t = csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/ACT0128/RRI_ch31_rawvariance_ACT0128_15s_3h.csv',
                         sep=',', header=None)
RRI_var = csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/ACT0128/RRI_ch31_rawauto_ACT0128_15s_3h.csv', sep=',',
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

# pyplot.plot(Raw_variance_RRI31_arr, 'grey', alpha=0.5)
# pyplot.xlabel('Time (hours)', fontsize=13)
# pyplot.title('RRI variance in ACT0128', fontsize=13)
# pyplot.show()
#
# pyplot.plot(Raw_auto_RRI31_arr, 'grey', alpha=0.5)
# pyplot.xlabel('Time (hours)', fontsize=13)
# pyplot.title('RRI autocorrelation in ACT0128', fontsize=13)
# pyplot.show()

# window_time_arr=t_window_arr
# Raw_variance_RRI31=Raw_variance_RRI31_arr
window_time_arr = t_window_arr[0:17280]
Raw_variance_RRI31 = Raw_variance_RRI31_arr[0:17280]

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
n=0
for item in seizure_phase_var_long:
    if item <0:
        n=n+1
print(n/len(seizure_phase_var_long))

bins_number = 18
bins = np.linspace(-np.pi, np.pi, bins_number + 1)
nRRIsvar, _, _ = pyplot.hist(phase_whole_long, bins)
nRRIsvarsei, _, _ = pyplot.hist(seizure_phase_var_long, bins)
print(nRRIsvar)
print(nRRIsvarsei)
# params = dict(projection='polar')
# fig, ax2 = pyplot.subplots(subplot_kw=params)
# ax2.bar(bins[:bins_number], nRRIsvarsei / sum(nRRIsvarsei), width=width, color='grey', alpha=0.7, edgecolor='k',
#         linewidth=2)
# ax2.set_title('RRI variance', fontsize=16)
# locs, labels = pyplot.yticks([0.1, 0.3, 0.5], ['0.1', '0.3', '0.5'], fontsize=16)
# # ax2.set_rlim([0,0.002])
# ax2.set_xticklabels(['0', '0.25$\pi$', '0.5$\pi$', '0.75$\pi$', '-$\pi$', '-0.75$\pi$', '-0.5$\pi$', '-0.25$\pi$', ],
#                     fontsize=16)
# pyplot.show()




# Raw_auto_RRI31=Raw_auto_RRI31_arr
Raw_auto_RRI31 = Raw_auto_RRI31_arr[0:17280]

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
# width = 2 * np.pi / bins_number
# params = dict(projection='polar')
# fig, ax2 = pyplot.subplots(subplot_kw=params)
# ax2.bar(bins[:bins_number], nRRIsautosei / sum(nRRIsautosei), width=width, color='grey', alpha=0.7, edgecolor='k',
#         linewidth=2)
# ax2.set_title('RRI autocorrelation', fontsize=16)
# locs, labels = pyplot.yticks([0.1, 0.45, 0.8], ['0.1', '0.45', '0.8'], fontsize=16)
# ax2.set_xticklabels(['0', '0.25$\pi$', '0.5$\pi$', '0.75$\pi$', '-$\pi$', '-0.75$\pi$', '-0.5$\pi$', '-0.25$\pi$', ],
#                     fontsize=16)
# pyplot.show()





# t = np.linspace(0 + 0.00416667, 0 + 0.00416667 + 0.00416667 * (len(Raw_variance_EEG_arr) - 1),
#                 len(Raw_variance_EEG_arr))
# window_time_arr = t
# a = np.where(t < 12.7977778 + 0)
# print(a);
# print(t[3070]);
# print(t[3071])
# t[0:3071] = t[0:3071] - 0 + 11.2188889
# t[3071:] = t[3071:] - 12.7977778 - 0
# print(t[3071]);
# print(t);
# print(type(t));
# print(t[0])
#
# time_feature_arr = []
# for i in range(len(t)):
#     if t[i] > 24:
#         time_feature_arr.append(t[i] - (t[i] // 24) * 24)
#     else:
#         time_feature_arr.append(t[i])
# seizure_time = [time_feature_arr[1701], time_feature_arr[6781], time_feature_arr[11290], time_feature_arr[13475],
#                 # time_feature_arr[18450],
#                 # time_feature_arr[23500], time_feature_arr[28619],
#                 ]
# print(seizure_time)
# bins_number = 18
# bins = np.linspace(0, 24, bins_number + 1)
# nEEGsvar, _, _ = pyplot.hist(time_feature_arr[0:17280], bins)
# nEEGsvarsei, _, _ = pyplot.hist(seizure_time, bins)
#
# # bins = np.linspace(0, 2 * np.pi, bins_number + 1)
# # width = 2 * np.pi / bins_number
# # params = dict(projection='polar')
# # fig, ax = pyplot.subplots(subplot_kw=params)
# # ax.bar(bins[:bins_number], nEEGsvarsei / sum(nEEGsvarsei), width=width, color='grey', alpha=0.7, edgecolor='k',
# #        linewidth=2)
# # pyplot.setp(ax.get_yticklabels(), color='k')
# # # ax.set_title('seizure timing histogram (SA0124)',fontsize=23)
# # ax.set_xticks(np.linspace(0, 2 * np.pi, 24, endpoint=False))
# # ax.set_xticklabels(
# #     ['0 am', '', '', 'Night', '', '', '6 am', '', '', 'Morning', '', '', '12 am', '', '', 'Afternoon', '', '', '18 pm',
# #      '', '', 'Evening', '', '', '24 pm'], fontsize=16)
# # # locs, labels = pyplot.xticks([0,3,6,9,12,15,18,21,24],['0','Night','6','Morning','12','Afternoon','18','Evening','24'],fontsize=16)
# # locs, labels = pyplot.yticks([0.1, 0.2, 0.3], ['0.1', '0.2', '0.3'], fontsize=16)
# # pyplot.show()
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
# locs, labels = pyplot.yticks([0.1,0.3,0.5],['0.1','0.3','0.5'],fontsize=16)
# pyplot.show()
#
# bins_number = 18
# bins = np.linspace(0, 24, bins_number + 1)
# ntimes, _, _ = pyplot.hist(time_feature_arr[0:17280], bins)
# ntimesei, _, _ = pyplot.hist(seizure_time, bins)
# print(ntimes)
# print(ntimesei)
#
#
#
#
# #### section 2 training training
medium_rhythm_var_arr_3 = movingaverage(Raw_variance_EEG, 240*6)
long_rhythm_var_arr = medium_rhythm_var_arr_3
var_trans = hilbert(long_rhythm_var_arr)
var_phase = np.angle(var_trans)
phase_long_EEGvariance_arr = var_phase
print(len(phase_long_EEGvariance_arr));
medium_rhythm_value_arr_3 = movingaverage(Raw_auto_EEG, 240*6)
long_rhythm_value_arr = medium_rhythm_value_arr_3
value_trans = hilbert(long_rhythm_value_arr)
value_phase = np.angle(value_trans)
phase_long_EEGauto_arr = value_phase
print(len(phase_long_EEGauto_arr));
medium_rhythm_RRIvar_arr_3 = movingaverage(Raw_variance_RRI31, 240*6)
long_rhythm_RRIvar_arr = medium_rhythm_RRIvar_arr_3
var_trans = hilbert(long_rhythm_RRIvar_arr)
var_phase = np.angle(var_trans)
phase_long_RRIvariance_arr = var_phase
print(len(phase_long_RRIvariance_arr));
medium_rhythm_RRIvalue_arr_3 = movingaverage(Raw_auto_RRI31, 240*6)
long_rhythm_RRIvalue_arr = medium_rhythm_RRIvalue_arr_3
value_trans = hilbert(long_rhythm_RRIvalue_arr)
value_phase = np.angle(value_trans)
phase_long_RRIauto_arr = value_phase
print(len(phase_long_RRIauto_arr));



# # ##### 24 h
# ### combined probability calculation
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
#         pro_eegvars_time_false.append(0.047406807)
#         pro_eegvars_time.append(0.25)
#     elif phase_long_EEGvariance_arr[i] >= bins[6] and phase_long_EEGvariance_arr[i] <= bins[7]:
#         pro_eegvars_time_false.append(0.20670294)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] > bins[7] and phase_long_EEGvariance_arr[i] < bins[8]:
#         pro_eegvars_time_false.append(0.114783515)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[8] and phase_long_EEGvariance_arr[i] <= bins[9]:
#         pro_eegvars_time_false.append(0.137300301)
#         pro_eegvars_time.append(0.25)
#     elif phase_long_EEGvariance_arr[i] > bins[9] and phase_long_EEGvariance_arr[i] < bins[10]:
#         pro_eegvars_time_false.append(0.131396157)
#         pro_eegvars_time.append(0.25)
#     elif phase_long_EEGvariance_arr[i] >= bins[10] and phase_long_EEGvariance_arr[i] <= bins[11]:
#         pro_eegvars_time_false.append(0.0884464)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] > bins[11] and phase_long_EEGvariance_arr[i] < bins[12]:
#         pro_eegvars_time_false.append(0.213243806)
#         pro_eegvars_time.append(0.25)
#     elif phase_long_EEGvariance_arr[i] > bins[12] and phase_long_EEGvariance_arr[i] < bins[13]:
#         pro_eegvars_time_false.append(0.056320908)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[13] and phase_long_EEGvariance_arr[i] < bins[14]:
#         pro_eegvars_time_false.append(0.004399166)
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
# print(pro_eegvars_time[1701]);print(pro_eegvars_time[6781]);print(pro_eegvars_time[11290]);print(pro_eegvars_time[13475]);
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
#         pro_eegautos_time_false.append(0.173246122)
#         pro_eegautos_time.append(0.25)
#     elif phase_long_EEGauto_arr[i] >= bins[8] and phase_long_EEGauto_arr[i] < bins[9]:
#         pro_eegautos_time_false.append(0.415952767)
#         pro_eegautos_time.append(0.5)
#     elif phase_long_EEGauto_arr[i] >= bins[9] and phase_long_EEGauto_arr[i] <= bins[10]:
#         pro_eegautos_time_false.append(0.302732114)
#         pro_eegautos_time.append(0.25)
#     elif phase_long_EEGauto_arr[i] > bins[10] and phase_long_EEGauto_arr[i] < bins[11]:
#         pro_eegautos_time_false.append(0.052095392)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[11] and phase_long_EEGauto_arr[i] < bins[12]:
#         pro_eegautos_time_false.append(0.017538782)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[12] and phase_long_EEGauto_arr[i] < bins[13]:
#         pro_eegautos_time_false.append(0.01719148)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[13] and phase_long_EEGauto_arr[i] < bins[14]:
#         pro_eegautos_time_false.append(0.021243343)
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
# print(pro_eegautos_time[1701]);print(pro_eegautos_time[6781]);print(pro_eegautos_time[11290]);print(pro_eegautos_time[13475]);
# pro_RRIvars_time = []
# pro_RRIvars_time_false = []
# for i in range(len(phase_long_RRIvariance_arr)):
#     if phase_long_RRIvariance_arr[i] >= bins[0] and phase_long_RRIvariance_arr[i] < bins[1]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvariance_arr[i] >= bins[1] and phase_long_RRIvariance_arr[i] < bins[2]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvariance_arr[i] >= bins[2] and phase_long_RRIvariance_arr[i] < bins[3]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvariance_arr[i] >= bins[3] and phase_long_RRIvariance_arr[i] < bins[4]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvariance_arr[i] >= bins[4] and phase_long_RRIvariance_arr[i] < bins[5]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvariance_arr[i] >= bins[5] and phase_long_RRIvariance_arr[i] < bins[6]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvariance_arr[i] >= bins[6] and phase_long_RRIvariance_arr[i] < bins[7]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvariance_arr[i] >= bins[7] and phase_long_RRIvariance_arr[i] <= bins[8]:
#         pro_RRIvars_time_false.append(0.02564251)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvariance_arr[i] > bins[8] and phase_long_RRIvariance_arr[i] < bins[9]:
#         pro_RRIvars_time_false.append(0.661611484)
#         pro_RRIvars_time.append(1)
#     elif phase_long_RRIvariance_arr[i] >= bins[9] and phase_long_RRIvariance_arr[i] <= bins[10]:
#         pro_RRIvars_time_false.append(0.20930771)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvariance_arr[i] > bins[10] and phase_long_RRIvariance_arr[i] <= bins[11]:
#         pro_RRIvars_time_false.append(0.055626302)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvariance_arr[i] > bins[11] and phase_long_RRIvariance_arr[i] < bins[12]:
#         pro_RRIvars_time_false.append(0.027089604)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvariance_arr[i] >= bins[12] and phase_long_RRIvariance_arr[i] < bins[13]:
#         pro_RRIvars_time_false.append(0.012792313)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvariance_arr[i] >= bins[13] and phase_long_RRIvariance_arr[i] < bins[14]:
#         pro_RRIvars_time_false.append(0.007930076)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvariance_arr[i] >= bins[14] and phase_long_RRIvariance_arr[i] < bins[15]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvariance_arr[i] >= bins[15] and phase_long_RRIvariance_arr[i] < bins[16]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvariance_arr[i] >= bins[16] and phase_long_RRIvariance_arr[i] < bins[17]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvariance_arr[i] >= bins[17]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
# print(pro_RRIvars_time[1701]);print(pro_RRIvars_time[6781]);print(pro_RRIvars_time[11290]);print(pro_RRIvars_time[13475]);
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
#         pro_RRIautos_time_false.append(0.245137763)
#         pro_RRIautos_time.append(0.25)
#     elif phase_long_RRIauto_arr[i] > bins[8] and phase_long_RRIauto_arr[i] <= bins[9]:
#         pro_RRIautos_time_false.append(0.243285483)
#         pro_RRIautos_time.append(0.25)
#     elif phase_long_RRIauto_arr[i] > bins[9] and phase_long_RRIauto_arr[i] <= bins[10]:
#         pro_RRIautos_time_false.append(0.415142394)
#         pro_RRIautos_time.append(0.5)
#     elif phase_long_RRIauto_arr[i] > bins[10] and phase_long_RRIauto_arr[i] < bins[11]:
#         pro_RRIautos_time_false.append(0.051516555)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[11] and phase_long_RRIauto_arr[i] < bins[12]:
#         pro_RRIautos_time_false.append(0.012850197)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[12] and phase_long_RRIauto_arr[i] < bins[13]:
#         pro_RRIautos_time_false.append(0.009898125)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[13] and phase_long_RRIauto_arr[i] < bins[14]:
#         pro_RRIautos_time_false.append(0.022169484)
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
# print(pro_RRIautos_time[1701]);print(pro_RRIautos_time[6781]);print(pro_RRIautos_time[11290]);print(pro_RRIautos_time[13475]);


# ##### 12h  12h 12h 12h 12h
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
# #         pro_eegvars_time_false.append(0.033051632)
# #         pro_eegvars_time.append(0)
# #     elif phase_long_EEGvariance_arr[i] >= bins[5] and phase_long_EEGvariance_arr[i] < bins[6]:
# #         pro_eegvars_time_false.append(0.129196573)
# #         pro_eegvars_time.append(0.25)
# #     elif phase_long_EEGvariance_arr[i] >= bins[6] and phase_long_EEGvariance_arr[i] <= bins[7]:
# #         pro_eegvars_time_false.append(0.075769854)
# #         pro_eegvars_time.append(0)
# #     elif phase_long_EEGvariance_arr[i] > bins[7] and phase_long_EEGvariance_arr[i] < bins[8]:
# #         pro_eegvars_time_false.append(0.119761519)
# #         pro_eegvars_time.append(0)
# #     elif phase_long_EEGvariance_arr[i] >= bins[8] and phase_long_EEGvariance_arr[i] <= bins[9]:
# #         pro_eegvars_time_false.append(0.146156518)
# #         pro_eegvars_time.append(0.25)
# #     elif phase_long_EEGvariance_arr[i] > bins[9] and phase_long_EEGvariance_arr[i] < bins[10]:
# #         pro_eegvars_time_false.append(0.079879602)
# #         pro_eegvars_time.append(0)
# #     elif phase_long_EEGvariance_arr[i] >= bins[10] and phase_long_EEGvariance_arr[i] <= bins[11]:
# #         pro_eegvars_time_false.append(0.20224589)
# #         pro_eegvars_time.append(0.25)
# #     elif phase_long_EEGvariance_arr[i] > bins[11] and phase_long_EEGvariance_arr[i] < bins[12]:
# #         pro_eegvars_time_false.append(0.110268581)
# #         pro_eegvars_time.append(0)
# #     elif phase_long_EEGvariance_arr[i] > bins[12] and phase_long_EEGvariance_arr[i] < bins[13]:
# #         pro_eegvars_time_false.append(0.090703867)
# #         pro_eegvars_time.append(0.25)
# #     elif phase_long_EEGvariance_arr[i] >= bins[13] and phase_long_EEGvariance_arr[i] < bins[14]:
# #         pro_eegvars_time_false.append(0.012965964)
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
# # print(pro_eegvars_time[1701]);print(pro_eegvars_time[6781]);print(pro_eegvars_time[11290]);print(pro_eegvars_time[13475]);
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
# #         pro_eegautos_time_false.append(0)
# #         pro_eegautos_time.append(0)
# #     elif phase_long_EEGauto_arr[i] >= bins[5] and phase_long_EEGauto_arr[i] < bins[6]:
# #         pro_eegautos_time_false.append(0)
# #         pro_eegautos_time.append(0)
# #     elif phase_long_EEGauto_arr[i] >= bins[6] and phase_long_EEGauto_arr[i] < bins[7]:
# #         pro_eegautos_time_false.append(0)
# #         pro_eegautos_time.append(0)
# #     elif phase_long_EEGauto_arr[i] >= bins[7] and phase_long_EEGauto_arr[i] < bins[8]:
# #         pro_eegautos_time_false.append(0.092382496)
# #         pro_eegautos_time.append(0.25)
# #     elif phase_long_EEGauto_arr[i] >= bins[8] and phase_long_EEGauto_arr[i] < bins[9]:
# #         pro_eegautos_time_false.append(0.470942348)
# #         pro_eegautos_time.append(0.5)
# #     elif phase_long_EEGauto_arr[i] >= bins[9] and phase_long_EEGauto_arr[i] <= bins[10]:
# #         pro_eegautos_time_false.append(0.374160685)
# #         pro_eegautos_time.append(0.25)
# #     elif phase_long_EEGauto_arr[i] > bins[10] and phase_long_EEGauto_arr[i] < bins[11]:
# #         pro_eegautos_time_false.append(0.034382959)
# #         pro_eegautos_time.append(0)
# #     elif phase_long_EEGauto_arr[i] >= bins[11] and phase_long_EEGauto_arr[i] < bins[12]:
# #         pro_eegautos_time_false.append(0.009087752)
# #         pro_eegautos_time.append(0)
# #     elif phase_long_EEGauto_arr[i] >= bins[12] and phase_long_EEGauto_arr[i] < bins[13]:
# #         pro_eegautos_time_false.append(0.008856217)
# #         pro_eegautos_time.append(0)
# #     elif phase_long_EEGauto_arr[i] >= bins[13] and phase_long_EEGauto_arr[i] < bins[14]:
# #         pro_eegautos_time_false.append(0.010187543)
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
# # print(pro_eegautos_time[1701]);print(pro_eegautos_time[6781]);print(pro_eegautos_time[11290]);print(pro_eegautos_time[13475]);
# # pro_RRIvars_time = []
# # pro_RRIvars_time_false = []
# # for i in range(len(phase_long_RRIvariance_arr)):
# #     if phase_long_RRIvariance_arr[i] >= bins[0] and phase_long_RRIvariance_arr[i] < bins[1]:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
# #     elif phase_long_RRIvariance_arr[i] >= bins[1] and phase_long_RRIvariance_arr[i] < bins[2]:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
# #     elif phase_long_RRIvariance_arr[i] >= bins[2] and phase_long_RRIvariance_arr[i] < bins[3]:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
# #     elif phase_long_RRIvariance_arr[i] >= bins[3] and phase_long_RRIvariance_arr[i] < bins[4]:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
# #     elif phase_long_RRIvariance_arr[i] >= bins[4] and phase_long_RRIvariance_arr[i] < bins[5]:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
# #     elif phase_long_RRIvariance_arr[i] >= bins[5] and phase_long_RRIvariance_arr[i] < bins[6]:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
# #     elif phase_long_RRIvariance_arr[i] >= bins[6] and phase_long_RRIvariance_arr[i] < bins[7]:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
# #     elif phase_long_RRIvariance_arr[i] >= bins[7] and phase_long_RRIvariance_arr[i] <= bins[8]:
# #         pro_RRIvars_time_false.append(0.151076638)
# #         pro_RRIvars_time.append(0)
# #     elif phase_long_RRIvariance_arr[i] > bins[8] and phase_long_RRIvariance_arr[i] < bins[9]:
# #         pro_RRIvars_time_false.append(0.360558)
# #         pro_RRIvars_time.append(0.75)
# #     elif phase_long_RRIvariance_arr[i] >= bins[9] and phase_long_RRIvariance_arr[i] <= bins[10]:
# #         pro_RRIvars_time_false.append(0.353669831)
# #         pro_RRIvars_time.append(0)
# #     elif phase_long_RRIvariance_arr[i] > bins[10] and phase_long_RRIvariance_arr[i] <= bins[11]:
# #         pro_RRIvars_time_false.append(0.116114841)
# #         pro_RRIvars_time.append(0.25)
# #     elif phase_long_RRIvariance_arr[i] > bins[11] and phase_long_RRIvariance_arr[i] < bins[12]:
# #         pro_RRIvars_time_false.append(0.008335263)
# #         pro_RRIvars_time.append(0)
# #     elif phase_long_RRIvariance_arr[i] >= bins[12] and phase_long_RRIvariance_arr[i] < bins[13]:
# #         pro_RRIvars_time_false.append(0.005209539)
# #         pro_RRIvars_time.append(0)
# #     elif phase_long_RRIvariance_arr[i] >= bins[13] and phase_long_RRIvariance_arr[i] < bins[14]:
# #         pro_RRIvars_time_false.append(0.005035888)
# #         pro_RRIvars_time.append(0)
# #     elif phase_long_RRIvariance_arr[i] >= bins[14] and phase_long_RRIvariance_arr[i] < bins[15]:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
# #     elif phase_long_RRIvariance_arr[i] >= bins[15] and phase_long_RRIvariance_arr[i] < bins[16]:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
# #     elif phase_long_RRIvariance_arr[i] >= bins[16] and phase_long_RRIvariance_arr[i] < bins[17]:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
# #     elif phase_long_RRIvariance_arr[i] >= bins[17]:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
# # print(pro_RRIvars_time[1701]);print(pro_RRIvars_time[6781]);print(pro_RRIvars_time[11290]);print(pro_RRIvars_time[13475]);
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
# #         pro_RRIautos_time_false.append(0)
# #         pro_RRIautos_time.append(0)
# #     elif phase_long_RRIauto_arr[i] >= bins[6] and phase_long_RRIauto_arr[i] < bins[7]:
# #         pro_RRIautos_time_false.append(0.063324844)
# #         pro_RRIautos_time.append(0)
# #     elif phase_long_RRIauto_arr[i] >= bins[7] and phase_long_RRIauto_arr[i] <= bins[8]:
# #         pro_RRIautos_time_false.append(0.072238944)
# #         pro_RRIautos_time.append(0.25)
# #     elif phase_long_RRIauto_arr[i] > bins[8] and phase_long_RRIauto_arr[i] <= bins[9]:
# #         pro_RRIautos_time_false.append(0.251331327)
# #         pro_RRIautos_time.append(0.25)
# #     elif phase_long_RRIauto_arr[i] > bins[9] and phase_long_RRIauto_arr[i] <= bins[10]:
# #         pro_RRIautos_time_false.append(0.572470479)
# #         pro_RRIautos_time.append(0.5)
# #     elif phase_long_RRIauto_arr[i] > bins[10] and phase_long_RRIauto_arr[i] < bins[11]:
# #         pro_RRIautos_time_false.append(0.020143552)
# #         pro_RRIautos_time.append(0)
# #     elif phase_long_RRIauto_arr[i] >= bins[11] and phase_long_RRIauto_arr[i] < bins[12]:
# #         pro_RRIautos_time_false.append(0.004283399)
# #         pro_RRIautos_time.append(0)
# #     elif phase_long_RRIauto_arr[i] >= bins[12] and phase_long_RRIauto_arr[i] < bins[13]:
# #         pro_RRIautos_time_false.append(0.004341283)
# #         pro_RRIautos_time.append(0)
# #     elif phase_long_RRIauto_arr[i] >= bins[13] and phase_long_RRIauto_arr[i] < bins[14]:
# #         pro_RRIautos_time_false.append(0.011866173)
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
# # print(pro_RRIautos_time[1701]);print(pro_RRIautos_time[6781]);print(pro_RRIautos_time[11290]);print(pro_RRIautos_time[13475]);


#### 6h 6h 6h6h
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
        pro_eegvars_time_false.append(0.078664043)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] >= bins[5] and phase_long_EEGvariance_arr[i] < bins[6]:
        pro_eegvars_time_false.append(0.092382496)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] >= bins[6] and phase_long_EEGvariance_arr[i] <= bins[7]:
        pro_eegvars_time_false.append(0.149976846)
        pro_eegvars_time.append(0.25)
    elif phase_long_EEGvariance_arr[i] > bins[7] and phase_long_EEGvariance_arr[i] < bins[8]:
        pro_eegvars_time_false.append(0.092382496)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] >= bins[8] and phase_long_EEGvariance_arr[i] <= bins[9]:
        pro_eegvars_time_false.append(0.070560315)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] > bins[9] and phase_long_EEGvariance_arr[i] < bins[10]:
        pro_eegvars_time_false.append(0.059041445)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] >= bins[10] and phase_long_EEGvariance_arr[i] <= bins[11]:
        pro_eegvars_time_false.append(0.119645751)
        pro_eegvars_time.append(0.25)
    elif phase_long_EEGvariance_arr[i] > bins[11] and phase_long_EEGvariance_arr[i] < bins[12]:
        pro_eegvars_time_false.append(0.210060199)
        pro_eegvars_time.append(0.5)
    elif phase_long_EEGvariance_arr[i] > bins[12] and phase_long_EEGvariance_arr[i] < bins[13]:
        pro_eegvars_time_false.append(0.057015513)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] >= bins[13] and phase_long_EEGvariance_arr[i] < bins[14]:
        pro_eegvars_time_false.append(0.070270896)
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
print(pro_eegvars_time[1701]);print(pro_eegvars_time[6781]);print(pro_eegvars_time[11290]);print(pro_eegvars_time[13475]);
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
        pro_eegautos_time_false.append(0.055510535)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[8] and phase_long_EEGauto_arr[i] < bins[9]:
        pro_eegautos_time_false.append(0.53056263)
        pro_eegautos_time.append(1)
    elif phase_long_EEGauto_arr[i] >= bins[9] and phase_long_EEGauto_arr[i] <= bins[10]:
        pro_eegautos_time_false.append(0.366809447)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] > bins[10] and phase_long_EEGauto_arr[i] < bins[11]:
        pro_eegautos_time_false.append(0.035656402)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[11] and phase_long_EEGauto_arr[i] < bins[12]:
        pro_eegautos_time_false.append(0.003820329)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[12] and phase_long_EEGauto_arr[i] < bins[13]:
        pro_eegautos_time_false.append(0.003125724)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[13] and phase_long_EEGauto_arr[i] < bins[14]:
        pro_eegautos_time_false.append(0.004514934)
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
print(pro_eegautos_time[1701]);print(pro_eegautos_time[6781]);print(pro_eegautos_time[11290]);print(pro_eegautos_time[13475]);
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
        pro_RRIvars_time_false.append(0.093771706)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[7] and phase_long_RRIvariance_arr[i] <= bins[8]:
        pro_RRIvars_time_false.append(0.163753184)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] > bins[8] and phase_long_RRIvariance_arr[i] < bins[9]:
        pro_RRIvars_time_false.append(0.203577217)
        pro_RRIvars_time.append(0.25)
    elif phase_long_RRIvariance_arr[i] >= bins[9] and phase_long_RRIvariance_arr[i] <= bins[10]:
        pro_RRIvars_time_false.append(0.326117157)
        pro_RRIvars_time.append(0.25)
    elif phase_long_RRIvariance_arr[i] > bins[10] and phase_long_RRIvariance_arr[i] <= bins[11]:
        pro_RRIvars_time_false.append(0.163753184)
        pro_RRIvars_time.append(0.25)
    elif phase_long_RRIvariance_arr[i] > bins[11] and phase_long_RRIvariance_arr[i] < bins[12]:
        pro_RRIvars_time_false.append(0.044107432)
        pro_RRIvars_time.append(0.25)
    elif phase_long_RRIvariance_arr[i] >= bins[12] and phase_long_RRIvariance_arr[i] < bins[13]:
        pro_RRIvars_time_false.append(0.002257467)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[13] and phase_long_RRIvariance_arr[i] < bins[14]:
        pro_RRIvars_time_false.append(0.002662653)
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
print(pro_RRIvars_time[1701]);print(pro_RRIvars_time[6781]);print(pro_RRIvars_time[11290]);print(pro_RRIvars_time[13475]);
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
        pro_RRIautos_time_false.append(0.013139616)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[6] and phase_long_RRIauto_arr[i] < bins[7]:
        pro_RRIautos_time_false.append(0.042660338)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[7] and phase_long_RRIauto_arr[i] <= bins[8]:
        pro_RRIautos_time_false.append(0.030620514)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] > bins[8] and phase_long_RRIauto_arr[i] <= bins[9]:
        pro_RRIautos_time_false.append(0.277957861)
        pro_RRIautos_time.append(0.75)
    elif phase_long_RRIauto_arr[i] > bins[9] and phase_long_RRIauto_arr[i] <= bins[10]:
        pro_RRIautos_time_false.append(0.620456124)
        pro_RRIautos_time.append(0.25)
    elif phase_long_RRIauto_arr[i] > bins[10] and phase_long_RRIauto_arr[i] < bins[11]:
        pro_RRIautos_time_false.append(0.005441074)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[11] and phase_long_RRIauto_arr[i] < bins[12]:
        pro_RRIautos_time_false.append(0.001910164)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[12] and phase_long_RRIauto_arr[i] < bins[13]:
        pro_RRIautos_time_false.append(0.002315351)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[13] and phase_long_RRIauto_arr[i] < bins[14]:
        pro_RRIautos_time_false.append(0.005498958)
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
print(pro_RRIautos_time[1701]);print(pro_RRIautos_time[6781]);print(pro_RRIautos_time[11290]);print(pro_RRIautos_time[13475]);


# # # ### 1h 1h 1h 1h 1h
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
# #         pro_eegvars_time_false.append(0.151192406)
# #         pro_eegvars_time.append(0.25)
# #     elif phase_long_EEGvariance_arr[i] >= bins[5] and phase_long_EEGvariance_arr[i] < bins[6]:
# #         pro_eegvars_time_false.append(0.127633712)
# #         pro_eegvars_time.append(0)
# #     elif phase_long_EEGvariance_arr[i] >= bins[6] and phase_long_EEGvariance_arr[i] <= bins[7]:
# #         pro_eegvars_time_false.append(0.073107201)
# #         pro_eegvars_time.append(0)
# #     elif phase_long_EEGvariance_arr[i] > bins[7] and phase_long_EEGvariance_arr[i] < bins[8]:
# #         pro_eegvars_time_false.append(0.080863626)
# #         pro_eegvars_time.append(0.25)
# #     elif phase_long_EEGvariance_arr[i] >= bins[8] and phase_long_EEGvariance_arr[i] <= bins[9]:
# #         pro_eegvars_time_false.append(0.047696226)
# #         pro_eegvars_time.append(0)
# #     elif phase_long_EEGvariance_arr[i] > bins[9] and phase_long_EEGvariance_arr[i] < bins[10]:
# #         pro_eegvars_time_false.append(0.052558463)
# #         pro_eegvars_time.append(0)
# #     elif phase_long_EEGvariance_arr[i] >= bins[10] and phase_long_EEGvariance_arr[i] <= bins[11]:
# #         pro_eegvars_time_false.append(0.060372771)
# #         pro_eegvars_time.append(0.25)
# #     elif phase_long_EEGvariance_arr[i] > bins[11] and phase_long_EEGvariance_arr[i] < bins[12]:
# #         pro_eegvars_time_false.append(0.198888632)
# #         pro_eegvars_time.append(0.25)
# #     elif phase_long_EEGvariance_arr[i] > bins[12] and phase_long_EEGvariance_arr[i] < bins[13]:
# #         pro_eegvars_time_false.append(0.079648067)
# #         pro_eegvars_time.append(0)
# #     elif phase_long_EEGvariance_arr[i] >= bins[13] and phase_long_EEGvariance_arr[i] < bins[14]:
# #         pro_eegvars_time_false.append(0.128038898)
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
# # print(pro_eegvars_time[1701]);print(pro_eegvars_time[6781]);print(pro_eegvars_time[11290]);print(pro_eegvars_time[13475]);
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
# #         pro_eegautos_time_false.append(0)
# #         pro_eegautos_time.append(0)
# #     elif phase_long_EEGauto_arr[i] >= bins[5] and phase_long_EEGauto_arr[i] < bins[6]:
# #         pro_eegautos_time_false.append(0)
# #         pro_eegautos_time.append(0)
# #     elif phase_long_EEGauto_arr[i] >= bins[6] and phase_long_EEGauto_arr[i] < bins[7]:
# #         pro_eegautos_time_false.append(0.000810373)
# #         pro_eegautos_time.append(0)
# #     elif phase_long_EEGauto_arr[i] >= bins[7] and phase_long_EEGauto_arr[i] < bins[8]:
# #         pro_eegautos_time_false.append(0.06679787)
# #         pro_eegautos_time.append(0)
# #     elif phase_long_EEGauto_arr[i] >= bins[8] and phase_long_EEGauto_arr[i] < bins[9]:
# #         pro_eegautos_time_false.append(0.422377865)
# #         pro_eegautos_time.append(0.5)
# #     elif phase_long_EEGauto_arr[i] >= bins[9] and phase_long_EEGauto_arr[i] <= bins[10]:
# #         pro_eegautos_time_false.append(0.421104422)
# #         pro_eegautos_time.append(0.5)
# #     elif phase_long_EEGauto_arr[i] > bins[10] and phase_long_EEGauto_arr[i] < bins[11]:
# #         pro_eegautos_time_false.append(0.083989349)
# #         pro_eegautos_time.append(0)
# #     elif phase_long_EEGauto_arr[i] >= bins[11] and phase_long_EEGauto_arr[i] < bins[12]:
# #         pro_eegautos_time_false.append(0.00306784)
# #         pro_eegautos_time.append(0)
# #     elif phase_long_EEGauto_arr[i] >= bins[12] and phase_long_EEGauto_arr[i] < bins[13]:
# #         pro_eegautos_time_false.append(0.001331327)
# #         pro_eegautos_time.append(0)
# #     elif phase_long_EEGauto_arr[i] >= bins[13] and phase_long_EEGauto_arr[i] < bins[14]:
# #         pro_eegautos_time_false.append(0.000520954)
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
# # print(pro_eegautos_time[1701]);print(pro_eegautos_time[6781]);print(pro_eegautos_time[11290]);print(pro_eegautos_time[13475]);
# # pro_RRIvars_time = []
# # pro_RRIvars_time_false = []
# # for i in range(len(phase_long_RRIvariance_arr)):
# #     if phase_long_RRIvariance_arr[i] >= bins[0] and phase_long_RRIvariance_arr[i] < bins[1]:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
# #     elif phase_long_RRIvariance_arr[i] >= bins[1] and phase_long_RRIvariance_arr[i] < bins[2]:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
# #     elif phase_long_RRIvariance_arr[i] >= bins[2] and phase_long_RRIvariance_arr[i] < bins[3]:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
# #     elif phase_long_RRIvariance_arr[i] >= bins[3] and phase_long_RRIvariance_arr[i] < bins[4]:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
# #     elif phase_long_RRIvariance_arr[i] >= bins[4] and phase_long_RRIvariance_arr[i] < bins[5]:
# #         pro_RRIvars_time_false.append(0.000405186)
# #         pro_RRIvars_time.append(0)
# #     elif phase_long_RRIvariance_arr[i] >= bins[5] and phase_long_RRIvariance_arr[i] < bins[6]:
# #         pro_RRIvars_time_false.append(0.043239176)
# #         pro_RRIvars_time.append(0)
# #     elif phase_long_RRIvariance_arr[i] >= bins[6] and phase_long_RRIvariance_arr[i] < bins[7]:
# #         pro_RRIvars_time_false.append(0.108126881)
# #         pro_RRIvars_time.append(0)
# #     elif phase_long_RRIvariance_arr[i] >= bins[7] and phase_long_RRIvariance_arr[i] <= bins[8]:
# #         pro_RRIvars_time_false.append(0.169425793)
# #         pro_RRIvars_time.append(0.25)
# #     elif phase_long_RRIvariance_arr[i] > bins[8] and phase_long_RRIvariance_arr[i] < bins[9]:
# #         pro_RRIvars_time_false.append(0.175156286)
# #         pro_RRIvars_time.append(0)
# #     elif phase_long_RRIvariance_arr[i] >= bins[9] and phase_long_RRIvariance_arr[i] <= bins[10]:
# #         pro_RRIvars_time_false.append(0.166184302)
# #         pro_RRIvars_time.append(0.25)
# #     elif phase_long_RRIvariance_arr[i] > bins[10] and phase_long_RRIvariance_arr[i] <= bins[11]:
# #         pro_RRIvars_time_false.append(0.208092151)
# #         pro_RRIvars_time.append(0.25)
# #     elif phase_long_RRIvariance_arr[i] > bins[11] and phase_long_RRIvariance_arr[i] < bins[12]:
# #         pro_RRIvars_time_false.append(0.078548275)
# #         pro_RRIvars_time.append(0.25)
# #     elif phase_long_RRIvariance_arr[i] >= bins[12] and phase_long_RRIvariance_arr[i] < bins[13]:
# #         pro_RRIvars_time_false.append(0.050764066)
# #         pro_RRIvars_time.append(0)
# #     elif phase_long_RRIvariance_arr[i] >= bins[13] and phase_long_RRIvariance_arr[i] < bins[14]:
# #         pro_RRIvars_time_false.append(5.78838E-05)
# #         pro_RRIvars_time.append(0)
# #     elif phase_long_RRIvariance_arr[i] >= bins[14] and phase_long_RRIvariance_arr[i] < bins[15]:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
# #     elif phase_long_RRIvariance_arr[i] >= bins[15] and phase_long_RRIvariance_arr[i] < bins[16]:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
# #     elif phase_long_RRIvariance_arr[i] >= bins[16] and phase_long_RRIvariance_arr[i] < bins[17]:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
# #     elif phase_long_RRIvariance_arr[i] >= bins[17]:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
# # print(pro_RRIvars_time[1701]);print(pro_RRIvars_time[6781]);print(pro_RRIvars_time[11290]);print(pro_RRIvars_time[13475]);
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
# #         pro_RRIautos_time_false.append(0)
# #         pro_RRIautos_time.append(0)
# #     elif phase_long_RRIauto_arr[i] >= bins[6] and phase_long_RRIauto_arr[i] < bins[7]:
# #         pro_RRIautos_time_false.append(0.022921973)
# #         pro_RRIautos_time.append(0)
# #     elif phase_long_RRIauto_arr[i] >= bins[7] and phase_long_RRIauto_arr[i] <= bins[8]:
# #         pro_RRIautos_time_false.append(0.050069461)
# #         pro_RRIautos_time.append(0)
# #     elif phase_long_RRIauto_arr[i] > bins[8] and phase_long_RRIauto_arr[i] <= bins[9]:
# #         pro_RRIautos_time_false.append(0.383827275)
# #         pro_RRIautos_time.append(0.5)
# #     elif phase_long_RRIauto_arr[i] > bins[9] and phase_long_RRIauto_arr[i] <= bins[10]:
# #         pro_RRIautos_time_false.append(0.513023848)
# #         pro_RRIautos_time.append(0.5)
# #     elif phase_long_RRIauto_arr[i] > bins[10] and phase_long_RRIauto_arr[i] < bins[11]:
# #         pro_RRIautos_time_false.append(0.027957861)
# #         pro_RRIautos_time.append(0)
# #     elif phase_long_RRIauto_arr[i] >= bins[11] and phase_long_RRIauto_arr[i] < bins[12]:
# #         pro_RRIautos_time_false.append(0.000520954)
# #         pro_RRIautos_time.append(0)
# #     elif phase_long_RRIauto_arr[i] >= bins[12] and phase_long_RRIauto_arr[i] < bins[13]:
# #         pro_RRIautos_time_false.append(0.000578838)
# #         pro_RRIautos_time.append(0)
# #     elif phase_long_RRIauto_arr[i] >= bins[13] and phase_long_RRIauto_arr[i] < bins[14]:
# #         pro_RRIautos_time_false.append(0.001099792)
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
# # print(pro_RRIautos_time[1701]);print(pro_RRIautos_time[6781]);print(pro_RRIautos_time[11290]);print(pro_RRIautos_time[13475]);
#
#
# ### 5min
# ### combined probability calculation
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
#         pro_eegvars_time_false.append(0.195820792)
#         pro_eegvars_time.append(0.25)
#     elif phase_long_EEGvariance_arr[i] >= bins[5] and phase_long_EEGvariance_arr[i] < bins[6]:
#         pro_eegvars_time_false.append(0.142220421)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[6] and phase_long_EEGvariance_arr[i] <= bins[7]:
#         pro_eegvars_time_false.append(0.062282936)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] > bins[7] and phase_long_EEGvariance_arr[i] < bins[8]:
#         pro_eegvars_time_false.append(0.045786062)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[8] and phase_long_EEGvariance_arr[i] <= bins[9]:
#         pro_eegvars_time_false.append(0.036119472)
#         pro_eegvars_time.append(0.25)
#     elif phase_long_EEGvariance_arr[i] > bins[9] and phase_long_EEGvariance_arr[i] < bins[10]:
#         pro_eegvars_time_false.append(0.034614494)
#         pro_eegvars_time.append(0.25)
#     elif phase_long_EEGvariance_arr[i] >= bins[10] and phase_long_EEGvariance_arr[i] <= bins[11]:
#         pro_eegvars_time_false.append(0.055163232)
#         pro_eegvars_time.append(0.25)
#     elif phase_long_EEGvariance_arr[i] > bins[11] and phase_long_EEGvariance_arr[i] < bins[12]:
#         pro_eegvars_time_false.append(0.128386201)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] > bins[12] and phase_long_EEGvariance_arr[i] < bins[13]:
#         pro_eegvars_time_false.append(0.135158602)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[13] and phase_long_EEGvariance_arr[i] < bins[14]:
#         pro_eegvars_time_false.append(0.164447789)
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
# print(pro_eegvars_time[1701]);print(pro_eegvars_time[6781]);print(pro_eegvars_time[11290]);print(pro_eegvars_time[13475]);
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
#         pro_eegautos_time_false.append(0.003878213)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[6] and phase_long_EEGauto_arr[i] < bins[7]:
#         pro_eegautos_time_false.append(0.034788145)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[7] and phase_long_EEGauto_arr[i] < bins[8]:
#         pro_eegautos_time_false.append(0.115246585)
#         pro_eegautos_time.append(0.25)
#     elif phase_long_EEGauto_arr[i] >= bins[8] and phase_long_EEGauto_arr[i] < bins[9]:
#         pro_eegautos_time_false.append(0.334336652)
#         pro_eegautos_time.append(0.75)
#     elif phase_long_EEGauto_arr[i] >= bins[9] and phase_long_EEGauto_arr[i] <= bins[10]:
#         pro_eegautos_time_false.append(0.361542024)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] > bins[10] and phase_long_EEGauto_arr[i] < bins[11]:
#         pro_eegautos_time_false.append(0.104190785)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[11] and phase_long_EEGauto_arr[i] < bins[12]:
#         pro_eegautos_time_false.append(0.042486687)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[12] and phase_long_EEGauto_arr[i] < bins[13]:
#         pro_eegautos_time_false.append(0.003473026)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[13] and phase_long_EEGauto_arr[i] < bins[14]:
#         pro_eegautos_time_false.append(5.78838E-05)
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
# print(pro_eegautos_time[1701]);print(pro_eegautos_time[6781]);print(pro_eegautos_time[11290]);print(pro_eegautos_time[13475]);




Pseizureeegvar = 0.000231481;
Pnonseizureeegvar = 0.999768519;
t = np.linspace(0 + 0.00416667, 0 + 0.00416667 + 0.00416667 * (len(Raw_variance_EEG) - 1), len(Raw_variance_EEG))
window_time_arr = t

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_RRIvars_time[m]*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined=[]
# for m in range(len(pro_eegvars_time)):
#     P1=Pseizureeegvar*pro_eegautos_time[m]*pro_RRIvars_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegautos_time_false[m]*pro_RRIvars_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=Pseizureeegvar*pro_eegvars_time[m]*pro_eegautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegautos_time_false[m]*pro_eegvars_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=Pseizureeegvar*pro_RRIvars_time[m]*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=Pseizureeegvar*pro_eegvars_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=Pseizureeegvar*pro_eegautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=Pseizureeegvar*pro_RRIvars_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m])
#     Pcombined.append(P1/(P1+P2))

Pcombined = []
for m in range(len(pro_RRIautos_time)):
    P1=Pseizureeegvar*pro_RRIautos_time[m]
    P2=Pnonseizureeegvar*(1-pro_RRIautos_time_false[m])
    Pcombined.append(P1/(P1+P2))

pyplot.figure(figsize=(12, 5))
pyplot.plot(window_time_arr, Pcombined)
pyplot.title('Combined probability in ACT0128', fontsize=15)
pyplot.annotate('', xy=(7.0936, np.max(Pcombined)), xytext=(7.0936, np.max(Pcombined) + 0.00000000001),
                arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(28.258611, np.max(Pcombined)), xytext=(28.258611, np.max(Pcombined) + 0.00000000001),
                arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(47.048333, np.max(Pcombined)), xytext=(47.048333, np.max(Pcombined) + 0.00000000001),
                arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(56.151389, np.max(Pcombined)), xytext=(56.151389, np.max(Pcombined) + 0.00000000001),
                arrowprops=dict(facecolor='black', shrink=0.05))
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


# t = np.linspace(0 + 0.00416667, 0 + 0.00416667 + 0.00416667 * (len(Raw_variance_EEG_arr) - 1),
#                 len(Raw_variance_EEG_arr))
# a = np.where(t < 12.7977778 + 0)
# t[0:3071] = t[0:3071] - 0 + 11.2188889
# t[3071:] = t[3071:] - 12.7977778 - 0
# time_feature_arr = []
# for i in range(len(t)):
#     if t[i] > 24:
#         time_feature_arr.append(t[i] - (t[i] // 24) * 24)
#     else:
#         time_feature_arr.append(t[i])
#
# bins_number = 18
# bins = np.linspace(0, 24, bins_number + 1)
# pro_circadian_time = []
# pro_circadian_time_false = []
# for i in range(len(time_feature_arr)):
#     if time_feature_arr[i] >= bins[0] and time_feature_arr[i] <= bins[1]:
#         pro_circadian_time_false.append(0.055799954)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] > bins[1] and time_feature_arr[i] < bins[2]:
#         pro_circadian_time_false.append(0.055568419)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] >= bins[2] and time_feature_arr[i] < bins[3]:
#         pro_circadian_time_false.append(0.055568419)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] >= bins[3] and time_feature_arr[i] < bins[4]:
#         pro_circadian_time_false.append(0.055568419)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] >= bins[4] and time_feature_arr[i] < bins[5]:
#         pro_circadian_time_false.append(0.055568419)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] >= bins[5] and time_feature_arr[i] <= bins[6]:
#         pro_circadian_time_false.append(0.055568419)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] > bins[6] and time_feature_arr[i] < bins[7]:
#         pro_circadian_time_false.append(0.055568419)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] >= bins[7] and time_feature_arr[i] <= bins[8]:
#         pro_circadian_time_false.append(0.055510535)
#         pro_circadian_time.append(0.25)
#     elif time_feature_arr[i] > bins[8] and time_feature_arr[i] < bins[9]:
#         pro_circadian_time_false.append(0.055336884)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] >= bins[9] and time_feature_arr[i] < bins[10]:
#         pro_circadian_time_false.append(0.055568419)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] >= bins[10] and time_feature_arr[i] < bins[11]:
#         pro_circadian_time_false.append(0.055568419)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] >= bins[11] and time_feature_arr[i] < bins[12]:
#         pro_circadian_time_false.append(0.055510535)
#         pro_circadian_time.append(0.25)
#     elif time_feature_arr[i] >= bins[12] and time_feature_arr[i] < bins[13]:
#         pro_circadian_time_false.append(0.055568419)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] >= bins[13] and time_feature_arr[i] < bins[14]:
#         pro_circadian_time_false.append(0.055510535)
#         pro_circadian_time.append(0.25)
#     elif time_feature_arr[i] >= bins[14] and time_feature_arr[i] < bins[15]:
#         pro_circadian_time_false.append(0.055510535)
#         pro_circadian_time.append(0.25)
#     elif time_feature_arr[i] >= bins[15] and time_feature_arr[i] < bins[16]:
#         pro_circadian_time_false.append(0.055568419)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] >= bins[16] and time_feature_arr[i] < bins[17]:
#         pro_circadian_time_false.append(0.055568419)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] >= bins[17] and time_feature_arr[i] <= bins[18]:
#         pro_circadian_time_false.append(0.055568419)
#         pro_circadian_time.append(0)
#
# # Pcombined=[]
# # for m in range(len(pro_eegvars_time)):
# #     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_RRIvars_time[m]*pro_RRIautos_time[m]*pro_circadian_time[m]
# #     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
# #     Pcombined.append(P1/(P1+P2))
#
# # Pcombined=[]
# # for m in range(len(pro_eegvars_time)):
# #     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_RRIautos_time[m]*pro_circadian_time[m]
# #     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
# #     Pcombined.append(P1/(P1+P2))
#
# # Pcombined=[]
# # for m in range(len(pro_eegvars_time)):
# #     P1=Pseizureeegvar*pro_eegautos_time[m]*pro_RRIvars_time[m]*pro_circadian_time[m]
# #     P2=Pnonseizureeegvar*(1-pro_eegautos_time_false[m]*pro_RRIvars_time_false[m]*pro_circadian_time_false[m])
# #     Pcombined.append(P1/(P1+P2))
#
# # Pcombined=[]
# # for m in range(len(pro_eegvars_time)):
# #     P1=Pseizureeegvar*pro_eegvars_time[m]*pro_eegautos_time[m]*pro_circadian_time[m]
# #     P2=Pnonseizureeegvar*(1-pro_eegautos_time_false[m]*pro_eegvars_time_false[m]*pro_circadian_time_false[m])
# #     Pcombined.append(P1/(P1+P2))
#
# # Pcombined = []
# # for m in range(len(pro_eegvars_time)):
# #     P1 = Pseizureeegvar * pro_RRIvars_time[m]*pro_RRIautos_time[m] * pro_circadian_time[m]
# #     P2 = Pnonseizureeegvar * (1 - pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m] * pro_circadian_time_false[m])
# #     Pcombined.append(P1 / (P1 + P2))
#
# # Pcombined=[]
# # for m in range(len(pro_eegvars_time)):
# #     P1=Pseizureeegvar*pro_eegvars_time[m]*pro_circadian_time[m]
# #     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_circadian_time_false[m])
# #     Pcombined.append(P1/(P1+P2))
#
# # Pcombined=[]
# # for m in range(len(pro_eegvars_time)):
# #     P1=Pseizureeegvar*pro_eegautos_time[m]*pro_circadian_time[m]
# #     P2=Pnonseizureeegvar*(1-pro_eegautos_time_false[m]*pro_circadian_time_false[m])
# #     Pcombined.append(P1/(P1+P2))
#
# # Pcombined = []
# # for m in range(len(pro_eegvars_time)):
# #     P1 = Pseizureeegvar * pro_RRIvars_time[m]* pro_circadian_time[m]
# #     P2 = Pnonseizureeegvar * (1 - pro_RRIvars_time_false[m]* pro_circadian_time_false[m])
# #     Pcombined.append(P1 / (P1 + P2))
#
# # Pcombined = []
# # for m in range(len(pro_eegvars_time)):
# #     P1 = Pseizureeegvar * pro_RRIautos_time[m] * pro_circadian_time[m]
# #     P2 = Pnonseizureeegvar * (1 - pro_RRIautos_time_false[m] * pro_circadian_time_false[m])
# #     Pcombined.append(P1 / (P1 + P2))
#
#
# t = np.linspace(0 + 0.00416667, 0 + 0.00416667 + 0.00416667 * (len(Raw_variance_EEG) - 1), len(Raw_variance_EEG))
# window_time_arr = t
# pyplot.figure(figsize=(12, 5))
# pyplot.plot(window_time_arr, Pcombined)
# pyplot.title('combined probability in ACT0128', fontsize=15)
# pyplot.annotate('', xy=(7.0936, np.max(Pcombined)), xytext=(7.0936, np.max(Pcombined) + 0.00000000001),
#                 arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(28.258611, np.max(Pcombined)), xytext=(28.258611, np.max(Pcombined) + 0.00000000001),
#                 arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(47.048333, np.max(Pcombined)), xytext=(47.048333, np.max(Pcombined) + 0.00000000001),
#                 arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(56.151389, np.max(Pcombined)), xytext=(56.151389, np.max(Pcombined) + 0.00000000001),
#                 arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.tight_layout()
# pyplot.xlim(window_time_arr[0], window_time_arr[-1])
# pyplot.xlabel('Time(h)', fontsize=15)
# pyplot.ylabel('seizure probability', fontsize=15)
# pyplot.show()
# # pro=[]
# # for item in seizure_timing_index:
# #     pro.append(float(Pcombined[item]))
# #     print(Pcombined[item])
# # print(pro)
# # Th2=np.min(pro)
# # print(Th2)





# ## section 3 froecast
t = np.linspace(0, 0 + 0.00416667 * (len(Raw_variance_EEG_arr) - 1), len(Raw_variance_EEG_arr))
t_window_arr = t
fore_arr_EEGvars = []
for k in range(72, 73):
    # for k in range(40, 41):
    variance_arr = Raw_variance_EEG_arr[0:(17280 + 240 * k)]
    long_rhythm_var_arr = movingaverage(variance_arr, 240 * 6)
    pyplot.figure(figsize=(6, 3))
    pyplot.title('EEG variance')
    pyplot.ylabel('Voltage ($\mathregular{v^2}$)')
    pyplot.xlabel('Time(h)')
    pyplot.plot(t_window_arr[240 * 24:(17280 + 240 * k)], long_rhythm_var_arr[240 * 24:], 'orange')
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/ACT0128/cycles6h_Cz_forecast72hsignal_3hcycle_EEGvar_ACT0128.csv',
                         sep=',', header=None)
forecast_var_EEG = csv_reader.values
forecast_var_EEG_arr = []
for item in forecast_var_EEG:
    forecast_var_EEG_arr = forecast_var_EEG_arr + list(item)
t = np.linspace(t_window_arr[17280], t_window_arr[17280] + 0.1666667 * (len(forecast_var_EEG_arr) - 1),
                len(forecast_var_EEG_arr))
pyplot.plot(t, forecast_var_EEG_arr, 'k', label='forecast EEG var')
pyplot.legend()
pyplot.show()

fore_arr_EEGauto = []
for k in range(72, 73):
    auto_arr = Raw_auto_EEG_arr[0:(17280 + 240 * k)]
    long_rhythm_auto_arr = movingaverage(auto_arr, 240 * 6)
    pyplot.figure(figsize=(6, 3))
    pyplot.title('EEG autocorrelation')
    pyplot.xlabel('time(h)')
    pyplot.plot(t_window_arr[240 * 24:(17280 + 240 * k)], long_rhythm_auto_arr[240 * 24:], 'orange')
csv_reader = pd.read_csv(
    'C:/Users/wxiong/Documents/PHD/2011.1/ACT0128/cycles6h_Cz_forecast72hsignal_3hcycle_EEGauto_ACT0128.csv', sep=',',
    header=None)
forecast_auto_EEG = csv_reader.values
forecast_auto_EEG_arr = []
for item in forecast_auto_EEG:
    forecast_auto_EEG_arr = forecast_auto_EEG_arr + list(item)

t = np.linspace(t_window_arr[17280], t_window_arr[17280] + 0.1666667 * (len(forecast_auto_EEG_arr) - 1),
                len(forecast_auto_EEG_arr))
pyplot.plot(t, forecast_auto_EEG_arr, 'k', label='forecast EEG auto')
pyplot.legend()
pyplot.show()

fore_arr_RRIvars = []
for k in range(72, 73):
    variance_arr = Raw_variance_RRI31_arr[0:(17280 + 240 * k)]
    long_rhythm_var_arr = movingaverage(variance_arr, 240 * 6)
    pyplot.figure(figsize=(6, 3))
    pyplot.title('RRI variance')
    pyplot.ylabel('Second ($\mathregular{s^2}$)')
    pyplot.xlabel('Time(h)')
    pyplot.plot(t_window_arr[240 * 24:(17280 + 240 * k)], long_rhythm_var_arr[240 * 24:], 'orange')
csv_reader = pd.read_csv(
    'C:/Users/wxiong/Documents/PHD/2011.1/ACT0128/cycles6h_ch31_forecast72hsignal_3hcycle_RRIvar_ACT0128.csv', sep=',',
    header=None)
forecast_var_RRI31 = csv_reader.values
forecast_var_RRI31_arr = []
for item in forecast_var_RRI31:
    forecast_var_RRI31_arr = forecast_var_RRI31_arr + list(item)
t = np.linspace(t_window_arr[17280], t_window_arr[17280] + 0.1666667 * (len(forecast_var_RRI31_arr) - 1),
                len(forecast_var_RRI31_arr))
pyplot.plot(t, forecast_var_RRI31_arr, 'k', label='forecast RRI var')
pyplot.legend()
pyplot.show()

fore_arr_RRIautos = []
save_data_RRIautos = []
for k in range(72, 73):
    # for k in range(40,41):
    auto_arr = Raw_auto_RRI31_arr[0:17280 + 240 * k]
    long_rhythm_auto_arr = movingaverage(auto_arr, 240 * 6)
    pyplot.figure(figsize=(6, 3))
    pyplot.title('RRI autocorrelation')
    pyplot.xlabel('Time(h)')
    pyplot.plot(t_window_arr[240 * 24:17280 + 240 * k], long_rhythm_auto_arr[240 * 24:], 'orange')
csv_reader = pd.read_csv(
    'C:/Users/wxiong/Documents/PHD/2011.1/ACT0128/cycles6h_ch31_forecast72hsignal_3hcycle_RRIauto_ACT0128.csv', sep=',',
    header=None)
forecast_auto_RRI31 = csv_reader.values
forecast_auto_RRI31_arr = []
for item in forecast_auto_RRI31:
    forecast_auto_RRI31_arr = forecast_auto_RRI31_arr + list(item)
t = np.linspace(t_window_arr[17280], t_window_arr[17280] + 0.1666667 * (len(forecast_auto_RRI31_arr) - 1),
                len(forecast_auto_RRI31_arr))
pyplot.plot(t, forecast_auto_RRI31_arr, 'k', label='forecast RRI auto')
pyplot.legend()
pyplot.show()
print(len(forecast_var_EEG_arr));print(len(forecast_auto_EEG_arr));print(len(forecast_var_RRI31_arr));print(len(forecast_auto_RRI31_arr));



# ### predict, forecast data
var_trans = hilbert(forecast_var_EEG_arr)
var_phase = np.angle(var_trans)
rolmean_short_EEGvar = var_phase

var_trans = hilbert(forecast_auto_EEG_arr)
var_phase = np.angle(var_trans)
rolmean_short_EEGauto = var_phase

var_trans = hilbert(forecast_var_RRI31_arr)
var_phase = np.angle(var_trans)
rolmean_short_RRIvar = var_phase

var_trans = hilbert(forecast_auto_RRI31_arr)
var_phase = np.angle(var_trans)
rolmean_short_RRIauto = var_phase




## combined probability calculation
#### 24 h 24 h 24 h
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
#         pro_eegvars_time_false.append(0.047406807)
#         pro_eegvars_time.append(0.25)
#     elif rolmean_short_EEGvar[i] >= bins[6] and rolmean_short_EEGvar[i] <= bins[7]:
#         pro_eegvars_time_false.append(0.20670294)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] > bins[7] and rolmean_short_EEGvar[i] < bins[8]:
#         pro_eegvars_time_false.append(0.114783515)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[8] and rolmean_short_EEGvar[i] <= bins[9]:
#         pro_eegvars_time_false.append(0.137300301)
#         pro_eegvars_time.append(0.25)
#     elif rolmean_short_EEGvar[i] > bins[9] and rolmean_short_EEGvar[i] < bins[10]:
#         pro_eegvars_time_false.append(0.131396157)
#         pro_eegvars_time.append(0.25)
#     elif rolmean_short_EEGvar[i] >= bins[10] and rolmean_short_EEGvar[i] <= bins[11]:
#         pro_eegvars_time_false.append(0.0884464)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] > bins[11] and rolmean_short_EEGvar[i] < bins[12]:
#         pro_eegvars_time_false.append(0.213243806)
#         pro_eegvars_time.append(0.25)
#     elif rolmean_short_EEGvar[i] > bins[12] and rolmean_short_EEGvar[i] < bins[13]:
#         pro_eegvars_time_false.append(0.056320908)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[13] and rolmean_short_EEGvar[i] < bins[14]:
#         pro_eegvars_time_false.append(0.004399166)
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
#         pro_eegautos_time_false.append(0.173246122)
#         pro_eegautos_time.append(0.25)
#     elif rolmean_short_EEGauto[i] >= bins[8] and rolmean_short_EEGauto[i] < bins[9]:
#         pro_eegautos_time_false.append(0.415952767)
#         pro_eegautos_time.append(0.5)
#     elif rolmean_short_EEGauto[i] >= bins[9] and rolmean_short_EEGauto[i] <= bins[10]:
#         pro_eegautos_time_false.append(0.302732114)
#         pro_eegautos_time.append(0.25)
#     elif rolmean_short_EEGauto[i] > bins[10] and rolmean_short_EEGauto[i] < bins[11]:
#         pro_eegautos_time_false.append(0.052095392)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[11] and rolmean_short_EEGauto[i] < bins[12]:
#         pro_eegautos_time_false.append(0.017538782)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[12] and rolmean_short_EEGauto[i] < bins[13]:
#         pro_eegautos_time_false.append(0.01719148)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[13] and rolmean_short_EEGauto[i] < bins[14]:
#         pro_eegautos_time_false.append(0.021243343)
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
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[5] and rolmean_short_RRIvar[i] < bins[6]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[6] and rolmean_short_RRIvar[i] < bins[7]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[7] and rolmean_short_RRIvar[i] <= bins[8]:
#         pro_RRIvars_time_false.append(0.02564251)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] > bins[8] and rolmean_short_RRIvar[i] < bins[9]:
#         pro_RRIvars_time_false.append(0.661611484)
#         pro_RRIvars_time.append(1)
#     elif rolmean_short_RRIvar[i] >= bins[9] and rolmean_short_RRIvar[i] <= bins[10]:
#         pro_RRIvars_time_false.append(0.20930771)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] > bins[10] and rolmean_short_RRIvar[i] <= bins[11]:
#         pro_RRIvars_time_false.append(0.055626302)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] > bins[11] and rolmean_short_RRIvar[i] < bins[12]:
#         pro_RRIvars_time_false.append(0.027089604)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[12] and rolmean_short_RRIvar[i] < bins[13]:
#         pro_RRIvars_time_false.append(0.012792313)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[13] and rolmean_short_RRIvar[i] < bins[14]:
#         pro_RRIvars_time_false.append(0.007930076)
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
#         pro_RRIautos_time_false.append(0.245137763)
#         pro_RRIautos_time.append(0.25)
#     elif rolmean_short_RRIauto[i] > bins[8] and rolmean_short_RRIauto[i] <= bins[9]:
#         pro_RRIautos_time_false.append(0.243285483)
#         pro_RRIautos_time.append(0.25)
#     elif rolmean_short_RRIauto[i] > bins[9] and rolmean_short_RRIauto[i] <= bins[10]:
#         pro_RRIautos_time_false.append(0.415142394)
#         pro_RRIautos_time.append(0.5)
#     elif rolmean_short_RRIauto[i] > bins[10] and rolmean_short_RRIauto[i] < bins[11]:
#         pro_RRIautos_time_false.append(0.051516555)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[11] and rolmean_short_RRIauto[i] < bins[12]:
#         pro_RRIautos_time_false.append(0.012850197)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[12] and rolmean_short_RRIauto[i] < bins[13]:
#         pro_RRIautos_time_false.append(0.009898125)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[13] and rolmean_short_RRIauto[i] < bins[14]:
#         pro_RRIautos_time_false.append(0.022169484)
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


# # ##### 12 h 12 h 12 h
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
# #         pro_eegvars_time_false.append(0.033051632)
# #         pro_eegvars_time.append(0)
# #     elif rolmean_short_EEGvar[i] >= bins[5] and rolmean_short_EEGvar[i] < bins[6]:
# #         pro_eegvars_time_false.append(0.129196573)
# #         pro_eegvars_time.append(0.25)
# #     elif rolmean_short_EEGvar[i] >= bins[6] and rolmean_short_EEGvar[i] <= bins[7]:
# #         pro_eegvars_time_false.append(0.075769854)
# #         pro_eegvars_time.append(0)
# #     elif rolmean_short_EEGvar[i] > bins[7] and rolmean_short_EEGvar[i] < bins[8]:
# #         pro_eegvars_time_false.append(0.119761519)
# #         pro_eegvars_time.append(0)
# #     elif rolmean_short_EEGvar[i] >= bins[8] and rolmean_short_EEGvar[i] <= bins[9]:
# #         pro_eegvars_time_false.append(0.146156518)
# #         pro_eegvars_time.append(0.25)
# #     elif rolmean_short_EEGvar[i] > bins[9] and rolmean_short_EEGvar[i] < bins[10]:
# #         pro_eegvars_time_false.append(0.079879602)
# #         pro_eegvars_time.append(0)
# #     elif rolmean_short_EEGvar[i] >= bins[10] and rolmean_short_EEGvar[i] <= bins[11]:
# #         pro_eegvars_time_false.append(0.20224589)
# #         pro_eegvars_time.append(0.25)
# #     elif rolmean_short_EEGvar[i] > bins[11] and rolmean_short_EEGvar[i] < bins[12]:
# #         pro_eegvars_time_false.append(0.110268581)
# #         pro_eegvars_time.append(0)
# #     elif rolmean_short_EEGvar[i] > bins[12] and rolmean_short_EEGvar[i] < bins[13]:
# #         pro_eegvars_time_false.append(0.090703867)
# #         pro_eegvars_time.append(0.25)
# #     elif rolmean_short_EEGvar[i] >= bins[13] and rolmean_short_EEGvar[i] < bins[14]:
# #         pro_eegvars_time_false.append(0.012965964)
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
# #         pro_eegautos_time_false.append(0)
# #         pro_eegautos_time.append(0)
# #     elif rolmean_short_EEGauto[i] >= bins[5] and rolmean_short_EEGauto[i] < bins[6]:
# #         pro_eegautos_time_false.append(0)
# #         pro_eegautos_time.append(0)
# #     elif rolmean_short_EEGauto[i] >= bins[6] and rolmean_short_EEGauto[i] < bins[7]:
# #         pro_eegautos_time_false.append(0)
# #         pro_eegautos_time.append(0)
# #     elif rolmean_short_EEGauto[i] >= bins[7] and rolmean_short_EEGauto[i] < bins[8]:
# #         pro_eegautos_time_false.append(0.092382496)
# #         pro_eegautos_time.append(0.25)
# #     elif rolmean_short_EEGauto[i] >= bins[8] and rolmean_short_EEGauto[i] < bins[9]:
# #         pro_eegautos_time_false.append(0.470942348)
# #         pro_eegautos_time.append(0.5)
# #     elif rolmean_short_EEGauto[i] >= bins[9] and rolmean_short_EEGauto[i] <= bins[10]:
# #         pro_eegautos_time_false.append(0.374160685)
# #         pro_eegautos_time.append(0.25)
# #     elif rolmean_short_EEGauto[i] > bins[10] and rolmean_short_EEGauto[i] < bins[11]:
# #         pro_eegautos_time_false.append(0.034382959)
# #         pro_eegautos_time.append(0)
# #     elif rolmean_short_EEGauto[i] >= bins[11] and rolmean_short_EEGauto[i] < bins[12]:
# #         pro_eegautos_time_false.append(0.009087752)
# #         pro_eegautos_time.append(0)
# #     elif rolmean_short_EEGauto[i] >= bins[12] and rolmean_short_EEGauto[i] < bins[13]:
# #         pro_eegautos_time_false.append(0.008856217)
# #         pro_eegautos_time.append(0)
# #     elif rolmean_short_EEGauto[i] >= bins[13] and rolmean_short_EEGauto[i] < bins[14]:
# #         pro_eegautos_time_false.append(0.010187543)
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
# # pro_RRIvars_time = []
# # pro_RRIvars_time_false = []
# # for i in range(len(rolmean_short_RRIvar)):
# #     if rolmean_short_RRIvar[i] >= bins[0] and rolmean_short_RRIvar[i] < bins[1]:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
# #     elif rolmean_short_RRIvar[i] >= bins[1] and rolmean_short_RRIvar[i] < bins[2]:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
# #     elif rolmean_short_RRIvar[i] >= bins[2] and rolmean_short_RRIvar[i] < bins[3]:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
# #     elif rolmean_short_RRIvar[i] >= bins[3] and rolmean_short_RRIvar[i] < bins[4]:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
# #     elif rolmean_short_RRIvar[i] >= bins[4] and rolmean_short_RRIvar[i] < bins[5]:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
# #     elif rolmean_short_RRIvar[i] >= bins[5] and rolmean_short_RRIvar[i] < bins[6]:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
# #     elif rolmean_short_RRIvar[i] >= bins[6] and rolmean_short_RRIvar[i] < bins[7]:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
# #     elif rolmean_short_RRIvar[i] >= bins[7] and rolmean_short_RRIvar[i] <= bins[8]:
# #         pro_RRIvars_time_false.append(0.151076638)
# #         pro_RRIvars_time.append(0)
# #     elif rolmean_short_RRIvar[i] > bins[8] and rolmean_short_RRIvar[i] < bins[9]:
# #         pro_RRIvars_time_false.append(0.360558)
# #         pro_RRIvars_time.append(0.75)
# #     elif rolmean_short_RRIvar[i] >= bins[9] and rolmean_short_RRIvar[i] <= bins[10]:
# #         pro_RRIvars_time_false.append(0.353669831)
# #         pro_RRIvars_time.append(0)
# #     elif rolmean_short_RRIvar[i] > bins[10] and rolmean_short_RRIvar[i] <= bins[11]:
# #         pro_RRIvars_time_false.append(0.116114841)
# #         pro_RRIvars_time.append(0.25)
# #     elif rolmean_short_RRIvar[i] > bins[11] and rolmean_short_RRIvar[i] < bins[12]:
# #         pro_RRIvars_time_false.append(0.008335263)
# #         pro_RRIvars_time.append(0)
# #     elif rolmean_short_RRIvar[i] >= bins[12] and rolmean_short_RRIvar[i] < bins[13]:
# #         pro_RRIvars_time_false.append(0.005209539)
# #         pro_RRIvars_time.append(0)
# #     elif rolmean_short_RRIvar[i] >= bins[13] and rolmean_short_RRIvar[i] < bins[14]:
# #         pro_RRIvars_time_false.append(0.005035888)
# #         pro_RRIvars_time.append(0)
# #     elif rolmean_short_RRIvar[i] >= bins[14] and rolmean_short_RRIvar[i] < bins[15]:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
# #     elif rolmean_short_RRIvar[i] >= bins[15] and rolmean_short_RRIvar[i] < bins[16]:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
# #     elif rolmean_short_RRIvar[i] >= bins[16] and rolmean_short_RRIvar[i] < bins[17]:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
# #     elif rolmean_short_RRIvar[i] >= bins[17]:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
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
# #         pro_RRIautos_time_false.append(0)
# #         pro_RRIautos_time.append(0)
# #     elif rolmean_short_RRIauto[i] >= bins[6] and rolmean_short_RRIauto[i] < bins[7]:
# #         pro_RRIautos_time_false.append(0.063324844)
# #         pro_RRIautos_time.append(0)
# #     elif rolmean_short_RRIauto[i] >= bins[7] and rolmean_short_RRIauto[i] <= bins[8]:
# #         pro_RRIautos_time_false.append(0.072238944)
# #         pro_RRIautos_time.append(0.25)
# #     elif rolmean_short_RRIauto[i] > bins[8] and rolmean_short_RRIauto[i] <= bins[9]:
# #         pro_RRIautos_time_false.append(0.251331327)
# #         pro_RRIautos_time.append(0.25)
# #     elif rolmean_short_RRIauto[i] > bins[9] and rolmean_short_RRIauto[i] <= bins[10]:
# #         pro_RRIautos_time_false.append(0.572470479)
# #         pro_RRIautos_time.append(0.5)
# #     elif rolmean_short_RRIauto[i] > bins[10] and rolmean_short_RRIauto[i] < bins[11]:
# #         pro_RRIautos_time_false.append(0.020143552)
# #         pro_RRIautos_time.append(0)
# #     elif rolmean_short_RRIauto[i] >= bins[11] and rolmean_short_RRIauto[i] < bins[12]:
# #         pro_RRIautos_time_false.append(0.004283399)
# #         pro_RRIautos_time.append(0)
# #     elif rolmean_short_RRIauto[i] >= bins[12] and rolmean_short_RRIauto[i] < bins[13]:
# #         pro_RRIautos_time_false.append(0.004341283)
# #         pro_RRIautos_time.append(0)
# #     elif rolmean_short_RRIauto[i] >= bins[13] and rolmean_short_RRIauto[i] < bins[14]:
# #         pro_RRIautos_time_false.append(0.011866173)
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


# ###### 6h 6h 6h
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
        pro_eegvars_time_false.append(0.078664043)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] >= bins[5] and rolmean_short_EEGvar[i] < bins[6]:
        pro_eegvars_time_false.append(0.092382496)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] >= bins[6] and rolmean_short_EEGvar[i] <= bins[7]:
        pro_eegvars_time_false.append(0.149976846)
        pro_eegvars_time.append(0.25)
    elif rolmean_short_EEGvar[i] > bins[7] and rolmean_short_EEGvar[i] < bins[8]:
        pro_eegvars_time_false.append(0.092382496)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] >= bins[8] and rolmean_short_EEGvar[i] <= bins[9]:
        pro_eegvars_time_false.append(0.070560315)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] > bins[9] and rolmean_short_EEGvar[i] < bins[10]:
        pro_eegvars_time_false.append(0.059041445)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] >= bins[10] and rolmean_short_EEGvar[i] <= bins[11]:
        pro_eegvars_time_false.append(0.119645751)
        pro_eegvars_time.append(0.25)
    elif rolmean_short_EEGvar[i] > bins[11] and rolmean_short_EEGvar[i] < bins[12]:
        pro_eegvars_time_false.append(0.210060199)
        pro_eegvars_time.append(0.5)
    elif rolmean_short_EEGvar[i] > bins[12] and rolmean_short_EEGvar[i] < bins[13]:
        pro_eegvars_time_false.append(0.057015513)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] >= bins[13] and rolmean_short_EEGvar[i] < bins[14]:
        pro_eegvars_time_false.append(0.070270896)
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
        pro_eegautos_time_false.append(0.055510535)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[8] and rolmean_short_EEGauto[i] < bins[9]:
        pro_eegautos_time_false.append(0.53056263)
        pro_eegautos_time.append(1)
    elif rolmean_short_EEGauto[i] >= bins[9] and rolmean_short_EEGauto[i] <= bins[10]:
        pro_eegautos_time_false.append(0.366809447)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] > bins[10] and rolmean_short_EEGauto[i] < bins[11]:
        pro_eegautos_time_false.append(0.035656402)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[11] and rolmean_short_EEGauto[i] < bins[12]:
        pro_eegautos_time_false.append(0.003820329)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[12] and rolmean_short_EEGauto[i] < bins[13]:
        pro_eegautos_time_false.append(0.003125724)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[13] and rolmean_short_EEGauto[i] < bins[14]:
        pro_eegautos_time_false.append(0.004514934)
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
        pro_RRIvars_time_false.append(0.093771706)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] >= bins[7] and rolmean_short_RRIvar[i] <= bins[8]:
        pro_RRIvars_time_false.append(0.163753184)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] > bins[8] and rolmean_short_RRIvar[i] < bins[9]:
        pro_RRIvars_time_false.append(0.203577217)
        pro_RRIvars_time.append(0.25)
    elif rolmean_short_RRIvar[i] >= bins[9] and rolmean_short_RRIvar[i] <= bins[10]:
        pro_RRIvars_time_false.append(0.326117157)
        pro_RRIvars_time.append(0.25)
    elif rolmean_short_RRIvar[i] > bins[10] and rolmean_short_RRIvar[i] <= bins[11]:
        pro_RRIvars_time_false.append(0.163753184)
        pro_RRIvars_time.append(0.25)
    elif rolmean_short_RRIvar[i] > bins[11] and rolmean_short_RRIvar[i] < bins[12]:
        pro_RRIvars_time_false.append(0.044107432)
        pro_RRIvars_time.append(0.25)
    elif rolmean_short_RRIvar[i] >= bins[12] and rolmean_short_RRIvar[i] < bins[13]:
        pro_RRIvars_time_false.append(0.002257467)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] >= bins[13] and rolmean_short_RRIvar[i] < bins[14]:
        pro_RRIvars_time_false.append(0.002662653)
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
        pro_RRIautos_time_false.append(0.013139616)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[6] and rolmean_short_RRIauto[i] < bins[7]:
        pro_RRIautos_time_false.append(0.042660338)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[7] and rolmean_short_RRIauto[i] <= bins[8]:
        pro_RRIautos_time_false.append(0.030620514)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] > bins[8] and rolmean_short_RRIauto[i] <= bins[9]:
        pro_RRIautos_time_false.append(0.277957861)
        pro_RRIautos_time.append(0.75)
    elif rolmean_short_RRIauto[i] > bins[9] and rolmean_short_RRIauto[i] <= bins[10]:
        pro_RRIautos_time_false.append(0.620456124)
        pro_RRIautos_time.append(0.25)
    elif rolmean_short_RRIauto[i] > bins[10] and rolmean_short_RRIauto[i] < bins[11]:
        pro_RRIautos_time_false.append(0.005441074)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[11] and rolmean_short_RRIauto[i] < bins[12]:
        pro_RRIautos_time_false.append(0.001910164)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[12] and rolmean_short_RRIauto[i] < bins[13]:
        pro_RRIautos_time_false.append(0.002315351)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[13] and rolmean_short_RRIauto[i] < bins[14]:
        pro_RRIautos_time_false.append(0.005498958)
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



# # # #### 1hour 1hour 1hour
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
# #         pro_eegvars_time_false.append(0.151192406)
# #         pro_eegvars_time.append(0.25)
# #     elif rolmean_short_EEGvar[i] >= bins[5] and rolmean_short_EEGvar[i] < bins[6]:
# #         pro_eegvars_time_false.append(0.127633712)
# #         pro_eegvars_time.append(0)
# #     elif rolmean_short_EEGvar[i] >= bins[6] and rolmean_short_EEGvar[i] <= bins[7]:
# #         pro_eegvars_time_false.append(0.073107201)
# #         pro_eegvars_time.append(0)
# #     elif rolmean_short_EEGvar[i] > bins[7] and rolmean_short_EEGvar[i] < bins[8]:
# #         pro_eegvars_time_false.append(0.080863626)
# #         pro_eegvars_time.append(0.25)
# #     elif rolmean_short_EEGvar[i] >= bins[8] and rolmean_short_EEGvar[i] <= bins[9]:
# #         pro_eegvars_time_false.append(0.047696226)
# #         pro_eegvars_time.append(0)
# #     elif rolmean_short_EEGvar[i] > bins[9] and rolmean_short_EEGvar[i] < bins[10]:
# #         pro_eegvars_time_false.append(0.052558463)
# #         pro_eegvars_time.append(0)
# #     elif rolmean_short_EEGvar[i] >= bins[10] and rolmean_short_EEGvar[i] <= bins[11]:
# #         pro_eegvars_time_false.append(0.060372771)
# #         pro_eegvars_time.append(0.25)
# #     elif rolmean_short_EEGvar[i] > bins[11] and rolmean_short_EEGvar[i] < bins[12]:
# #         pro_eegvars_time_false.append(0.198888632)
# #         pro_eegvars_time.append(0.25)
# #     elif rolmean_short_EEGvar[i] > bins[12] and rolmean_short_EEGvar[i] < bins[13]:
# #         pro_eegvars_time_false.append(0.079648067)
# #         pro_eegvars_time.append(0)
# #     elif rolmean_short_EEGvar[i] >= bins[13] and rolmean_short_EEGvar[i] < bins[14]:
# #         pro_eegvars_time_false.append(0.128038898)
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
# #         pro_eegautos_time_false.append(0)
# #         pro_eegautos_time.append(0)
# #     elif rolmean_short_EEGauto[i] >= bins[5] and rolmean_short_EEGauto[i] < bins[6]:
# #         pro_eegautos_time_false.append(0)
# #         pro_eegautos_time.append(0)
# #     elif rolmean_short_EEGauto[i] >= bins[6] and rolmean_short_EEGauto[i] < bins[7]:
# #         pro_eegautos_time_false.append(0.000810373)
# #         pro_eegautos_time.append(0)
# #     elif rolmean_short_EEGauto[i] >= bins[7] and rolmean_short_EEGauto[i] < bins[8]:
# #         pro_eegautos_time_false.append(0.06679787)
# #         pro_eegautos_time.append(0)
# #     elif rolmean_short_EEGauto[i] >= bins[8] and rolmean_short_EEGauto[i] < bins[9]:
# #         pro_eegautos_time_false.append(0.422377865)
# #         pro_eegautos_time.append(0.5)
# #     elif rolmean_short_EEGauto[i] >= bins[9] and rolmean_short_EEGauto[i] <= bins[10]:
# #         pro_eegautos_time_false.append(0.421104422)
# #         pro_eegautos_time.append(0.5)
# #     elif rolmean_short_EEGauto[i] > bins[10] and rolmean_short_EEGauto[i] < bins[11]:
# #         pro_eegautos_time_false.append(0.083989349)
# #         pro_eegautos_time.append(0)
# #     elif rolmean_short_EEGauto[i] >= bins[11] and rolmean_short_EEGauto[i] < bins[12]:
# #         pro_eegautos_time_false.append(0.00306784)
# #         pro_eegautos_time.append(0)
# #     elif rolmean_short_EEGauto[i] >= bins[12] and rolmean_short_EEGauto[i] < bins[13]:
# #         pro_eegautos_time_false.append(0.001331327)
# #         pro_eegautos_time.append(0)
# #     elif rolmean_short_EEGauto[i] >= bins[13] and rolmean_short_EEGauto[i] < bins[14]:
# #         pro_eegautos_time_false.append(0.000520954)
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
# # pro_RRIvars_time = []
# # pro_RRIvars_time_false = []
# # for i in range(len(rolmean_short_RRIvar)):
# #     if rolmean_short_RRIvar[i] >= bins[0] and rolmean_short_RRIvar[i] < bins[1]:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
# #     elif rolmean_short_RRIvar[i] >= bins[1] and rolmean_short_RRIvar[i] < bins[2]:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
# #     elif rolmean_short_RRIvar[i] >= bins[2] and rolmean_short_RRIvar[i] < bins[3]:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
# #     elif rolmean_short_RRIvar[i] >= bins[3] and rolmean_short_RRIvar[i] < bins[4]:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
# #     elif rolmean_short_RRIvar[i] >= bins[4] and rolmean_short_RRIvar[i] < bins[5]:
# #         pro_RRIvars_time_false.append(0.000405186)
# #         pro_RRIvars_time.append(0)
# #     elif rolmean_short_RRIvar[i] >= bins[5] and rolmean_short_RRIvar[i] < bins[6]:
# #         pro_RRIvars_time_false.append(0.043239176)
# #         pro_RRIvars_time.append(0)
# #     elif rolmean_short_RRIvar[i] >= bins[6] and rolmean_short_RRIvar[i] < bins[7]:
# #         pro_RRIvars_time_false.append(0.108126881)
# #         pro_RRIvars_time.append(0)
# #     elif rolmean_short_RRIvar[i] >= bins[7] and rolmean_short_RRIvar[i] <= bins[8]:
# #         pro_RRIvars_time_false.append(0.169425793)
# #         pro_RRIvars_time.append(0.25)
# #     elif rolmean_short_RRIvar[i] > bins[8] and rolmean_short_RRIvar[i] < bins[9]:
# #         pro_RRIvars_time_false.append(0.175156286)
# #         pro_RRIvars_time.append(0)
# #     elif rolmean_short_RRIvar[i] >= bins[9] and rolmean_short_RRIvar[i] <= bins[10]:
# #         pro_RRIvars_time_false.append(0.166184302)
# #         pro_RRIvars_time.append(0.25)
# #     elif rolmean_short_RRIvar[i] > bins[10] and rolmean_short_RRIvar[i] <= bins[11]:
# #         pro_RRIvars_time_false.append(0.208092151)
# #         pro_RRIvars_time.append(0.25)
# #     elif rolmean_short_RRIvar[i] > bins[11] and rolmean_short_RRIvar[i] < bins[12]:
# #         pro_RRIvars_time_false.append(0.078548275)
# #         pro_RRIvars_time.append(0.25)
# #     elif rolmean_short_RRIvar[i] >= bins[12] and rolmean_short_RRIvar[i] < bins[13]:
# #         pro_RRIvars_time_false.append(0.050764066)
# #         pro_RRIvars_time.append(0)
# #     elif rolmean_short_RRIvar[i] >= bins[13] and rolmean_short_RRIvar[i] < bins[14]:
# #         pro_RRIvars_time_false.append(5.78838E-05)
# #         pro_RRIvars_time.append(0)
# #     elif rolmean_short_RRIvar[i] >= bins[14] and rolmean_short_RRIvar[i] < bins[15]:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
# #     elif rolmean_short_RRIvar[i] >= bins[15] and rolmean_short_RRIvar[i] < bins[16]:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
# #     elif rolmean_short_RRIvar[i] >= bins[16] and rolmean_short_RRIvar[i] < bins[17]:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
# #     elif rolmean_short_RRIvar[i] >= bins[17]:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
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
# #         pro_RRIautos_time_false.append(0)
# #         pro_RRIautos_time.append(0)
# #     elif rolmean_short_RRIauto[i] >= bins[6] and rolmean_short_RRIauto[i] < bins[7]:
# #         pro_RRIautos_time_false.append(0.022921973)
# #         pro_RRIautos_time.append(0)
# #     elif rolmean_short_RRIauto[i] >= bins[7] and rolmean_short_RRIauto[i] <= bins[8]:
# #         pro_RRIautos_time_false.append(0.050069461)
# #         pro_RRIautos_time.append(0)
# #     elif rolmean_short_RRIauto[i] > bins[8] and rolmean_short_RRIauto[i] <= bins[9]:
# #         pro_RRIautos_time_false.append(0.383827275)
# #         pro_RRIautos_time.append(0.5)
# #     elif rolmean_short_RRIauto[i] > bins[9] and rolmean_short_RRIauto[i] <= bins[10]:
# #         pro_RRIautos_time_false.append(0.513023848)
# #         pro_RRIautos_time.append(0.5)
# #     elif rolmean_short_RRIauto[i] > bins[10] and rolmean_short_RRIauto[i] < bins[11]:
# #         pro_RRIautos_time_false.append(0.027957861)
# #         pro_RRIautos_time.append(0)
# #     elif rolmean_short_RRIauto[i] >= bins[11] and rolmean_short_RRIauto[i] < bins[12]:
# #         pro_RRIautos_time_false.append(0.000520954)
# #         pro_RRIautos_time.append(0)
# #     elif rolmean_short_RRIauto[i] >= bins[12] and rolmean_short_RRIauto[i] < bins[13]:
# #         pro_RRIautos_time_false.append(0.000578838)
# #         pro_RRIautos_time.append(0)
# #     elif rolmean_short_RRIauto[i] >= bins[13] and rolmean_short_RRIauto[i] < bins[14]:
# #         pro_RRIautos_time_false.append(0.001099792)
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
# #
#
#
# ### 5 min
# ### combined probability calculation
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
#         pro_eegvars_time_false.append(0.195820792)
#         pro_eegvars_time.append(0.25)
#     elif rolmean_short_EEGvar[i] >= bins[5] and rolmean_short_EEGvar[i] < bins[6]:
#         pro_eegvars_time_false.append(0.142220421)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[6] and rolmean_short_EEGvar[i] <= bins[7]:
#         pro_eegvars_time_false.append(0.062282936)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] > bins[7] and rolmean_short_EEGvar[i] < bins[8]:
#         pro_eegvars_time_false.append(0.045786062)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[8] and rolmean_short_EEGvar[i] <= bins[9]:
#         pro_eegvars_time_false.append(0.036119472)
#         pro_eegvars_time.append(0.25)
#     elif rolmean_short_EEGvar[i] > bins[9] and rolmean_short_EEGvar[i] < bins[10]:
#         pro_eegvars_time_false.append(0.034614494)
#         pro_eegvars_time.append(0.25)
#     elif rolmean_short_EEGvar[i] >= bins[10] and rolmean_short_EEGvar[i] <= bins[11]:
#         pro_eegvars_time_false.append(0.055163232)
#         pro_eegvars_time.append(0.25)
#     elif rolmean_short_EEGvar[i] > bins[11] and rolmean_short_EEGvar[i] < bins[12]:
#         pro_eegvars_time_false.append(0.128386201)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] > bins[12] and rolmean_short_EEGvar[i] < bins[13]:
#         pro_eegvars_time_false.append(0.135158602)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[13] and rolmean_short_EEGvar[i] < bins[14]:
#         pro_eegvars_time_false.append(0.164447789)
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
#         pro_eegautos_time_false.append(0.003878213)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[6] and rolmean_short_EEGauto[i] < bins[7]:
#         pro_eegautos_time_false.append(0.034788145)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[7] and rolmean_short_EEGauto[i] < bins[8]:
#         pro_eegautos_time_false.append(0.115246585)
#         pro_eegautos_time.append(0.25)
#     elif rolmean_short_EEGauto[i] >= bins[8] and rolmean_short_EEGauto[i] < bins[9]:
#         pro_eegautos_time_false.append(0.334336652)
#         pro_eegautos_time.append(0.75)
#     elif rolmean_short_EEGauto[i] >= bins[9] and rolmean_short_EEGauto[i] <= bins[10]:
#         pro_eegautos_time_false.append(0.361542024)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] > bins[10] and rolmean_short_EEGauto[i] < bins[11]:
#         pro_eegautos_time_false.append(0.104190785)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[11] and rolmean_short_EEGauto[i] < bins[12]:
#         pro_eegautos_time_false.append(0.042486687)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[12] and rolmean_short_EEGauto[i] < bins[13]:
#         pro_eegautos_time_false.append(0.003473026)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[13] and rolmean_short_EEGauto[i] < bins[14]:
#         pro_eegautos_time_false.append(5.78838E-05)
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





Pseizureeegvar = 0.000231481;
Pnonseizureeegvar = 0.999768519;

# Pcombined=[]
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_RRIvars_time[m]*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined=[]
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined=[]
# for m in range(len(pro_eegvars_time)):
#     P1=Pseizureeegvar*pro_eegautos_time[m]*pro_RRIvars_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegautos_time_false[m]*pro_RRIvars_time_false[m])
#     Pcombined.append(P1/(P1+P2))
#
# Pcombined=[]
# for m in range(len(pro_eegvars_time)):
#     P1=Pseizureeegvar*pro_eegvars_time[m]*pro_eegautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegautos_time_false[m]*pro_eegvars_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1 = Pseizureeegvar * pro_RRIvars_time[m]*pro_RRIautos_time[m]
#     P2 = Pnonseizureeegvar * (1 - pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m] )
#     Pcombined.append(P1 / (P1 + P2))

# Pcombined=[]
# for m in range(len(pro_eegvars_time)):
#     P1=Pseizureeegvar*pro_eegvars_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined=[]
# for m in range(len(pro_eegautos_time)):
#     P1=Pseizureeegvar*pro_eegautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1 = Pseizureeegvar * pro_RRIvars_time[m]
#     P2 = Pnonseizureeegvar * (1 - pro_RRIvars_time_false[m])
#     Pcombined.append(P1 / (P1 + P2))

Pcombined = []
for m in range(len(pro_RRIautos_time)):
    P1 = Pseizureeegvar * pro_RRIautos_time[m]
    P2 = Pnonseizureeegvar * (1 - pro_RRIautos_time_false[m] )
    Pcombined.append(P1 / (P1 + P2))


pyplot.figure(figsize=(8, 4))
RRI_timewindow_arr = t
pyplot.plot(RRI_timewindow_arr, Pcombined)
pyplot.annotate('', xy=(76.88111, np.max(Pcombined)), xytext=(76.88111, np.max(Pcombined) + 0.00000000001),
                arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(97.923611, np.max(Pcombined)), xytext=(97.923611, np.max(Pcombined) + 0.00000000001),
                arrowprops=dict(facecolor='k', shrink=0.05))
pyplot.annotate('', xy=(119.25278, np.max(Pcombined)), xytext=(119.25278, np.max(Pcombined) + 0.00000000001),
                arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.hlines(Th1, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'r')
pyplot.title('Forecast seizures in ACT0128')
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
time_arr=[76.88111,97.923611,119.25278]
pretime=[]
k=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k=k+1
            n_arr.append(n)
            pretime.append(m - n)
print(k)
time_arr=[76.88111,97.923611,119.25278,119.44028]
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
        if item >= 0.3*Th1:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[76.88111,97.923611,119.25278]
k=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k=k+1
            n_arr.append(n)
            pretime.append(m - n)
print(k)
time_arr=[76.88111,97.923611,119.25278,119.44028]
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
        if item >= 0.6*Th1:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[76.88111,97.923611,119.25278]
k=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k=k+1
            n_arr.append(n)
            pretime.append(m - n)
print(k)
time_arr=[76.88111,97.923611,119.25278,119.44028]
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
        if item >= 1.2*Th1:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[76.88111,97.923611,119.25278]
k=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k=k+1
            n_arr.append(n)
            pretime.append(m - n)
print(k)
time_arr=[76.88111,97.923611,119.25278,119.44028]
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
        if item >= 2*Th1:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[76.88111,97.923611,119.25278]
k=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k=k+1
            n_arr.append(n)
            pretime.append(m - n)
print(k)
time_arr=[76.88111,97.923611,119.25278,119.44028]
k=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k=k+1
            n_arr.append(n)
print(k)
print(pretime)
print(np.mean(pretime))


# Pcombined = split(Pcombined_X, 6)
# print(len(Pcombined))
# time_arr_arr=[]
# AUC_cs_arr=[]
# for i in range(50000):
#     time_arr = np.random.uniform(low=t_window_arr[17280], high=t_window_arr[-1], size=3)
#     time_arr_arr.append(time_arr)
#     time_arr=np.sort(time_arr)
#
#     index = []
#     for i in range(len(Pcombined)):
#         for item in Pcombined[i]:
#             if item >= Th1:
#                 index.append(6 * i + 0)
#     # print(RRI_timewindow_arr[index])
#     a1 = np.unique(RRI_timewindow_arr[index])
#     # print(a1);
#     # print(len(a1))
#     k1 = 0
#     n_arr = []
#     for m in time_arr:
#         for n in a1:
#             if m - n <= 1 and m - n >= 0:
#                 k1 = k1 + 1
#                 n_arr.append(n)
#     # print(k1)
#     # time_arr=[76.88111,97.923611,119.25278,119.44028]
#     # k11=0
#     # n_arr=[]
#     # for m in time_arr:
#     #     for n in a1:
#     #         if m-n<=1 and m-n>=0:
#     #             k11=k11+1
#     #             n_arr.append(n)
#     # print(k11)
#     index = []
#     for i in range(len(Pcombined)):
#         for item in Pcombined[i]:
#             if item >= 0.3 * Th1:
#                 index.append(6 * i + 0)
#     # print(RRI_timewindow_arr[index])
#     a2 = np.unique(RRI_timewindow_arr[index])
#     # print(a2);
#     # print(len(a2))
#     k2 = 0
#     n_arr = []
#     for m in time_arr:
#         for n in a2:
#             if m - n <= 1 and m - n >= 0:
#                 k2 = k2 + 1
#                 n_arr.append(n)
#     # print(k2)
#     # time_arr=[76.88111,97.923611,119.25278,119.44028]
#     # k22=0
#     # n_arr=[]
#     # for m in time_arr:
#     #     for n in a2:
#     #         if m-n<=1 and m-n>=0:
#     #             k22=k22+1
#     #             n_arr.append(n)
#     # print(k22)
#     index = []
#     for i in range(len(Pcombined)):
#         for item in Pcombined[i]:
#             if item >= 0.6 * Th1:
#                 index.append(6 * i + 0)
#     # print(RRI_timewindow_arr[index])
#     a3 = np.unique(RRI_timewindow_arr[index])
#     # print(a3);
#     # print(len(a3))
#     k3 = 0
#     n_arr = []
#     for m in time_arr:
#         for n in a3:
#             if m - n <= 1 and m - n >= 0:
#                 k3 = k3 + 1
#                 n_arr.append(n)
#     # print(k3)
#     # time_arr=[76.88111,97.923611,119.25278,119.44028]
#     # k33=0
#     # n_arr=[]
#     # for m in time_arr:
#     #     for n in a3:
#     #         if m-n<=1 and m-n>=0:
#     #             k33=k33+1
#     #             n_arr.append(n)
#     # print(k33)
#     index = []
#     for i in range(len(Pcombined)):
#         for item in Pcombined[i]:
#             if item >= 1.2 * Th1:
#                 index.append(6 * i + 0)
#     # print(RRI_timewindow_arr[index])
#     a4 = np.unique(RRI_timewindow_arr[index])
#     # print(a);
#     # print(len(a4))
#     k4 = 0
#     n_arr = []
#     for m in time_arr:
#         for n in a4:
#             if m - n <= 1 and m - n >= 0:
#                 k4 = k4 + 1
#                 n_arr.append(n)
#     # print(k4)
#     # time_arr=[76.88111,97.923611,119.25278,119.44028]
#     # k44=0
#     # n_arr=[]
#     # for m in time_arr:
#     #     for n in a4:
#     #         if m-n<=1 and m-n>=0:
#     #             k44=k44+1
#     #             n_arr.append(n)
#     # print(k44)
#     index = []
#     for i in range(len(Pcombined)):
#         for item in Pcombined[i]:
#             if item >= 2 * Th1:
#                 index.append(6 * i + 0)
#     # print(RRI_timewindow_arr[index])
#     a5 = np.unique(RRI_timewindow_arr[index])
#     # print(a5);
#     # print(len(a5))
#     k5 = 0
#     n_arr = []
#     for m in time_arr:
#         for n in a5:
#             if m - n <= 1 and m - n >= 0:
#                 k5 = k5 + 1
#                 n_arr.append(n)
#     # print(k5)
#     # time_arr=[76.88111,97.923611,119.25278,119.44028]
#     # k55=0
#     # n_arr=[]
#     # for m in time_arr:
#     #     for n in a:
#     #         if m-n<=1 and m-n>=0:
#     #             k55=k55+1
#     #             n_arr.append(n)
#     # print(k55)
#     Sen1 = k1 / len(time_arr);
#     Sen2 = k2 / len(time_arr);
#     Sen3 = k3 / len(time_arr);
#     Sen4 = k4 / len(time_arr);
#     Sen5 = k5 / len(time_arr);
#     FPR1 = (len(a1) - k1) / len(Pcombined);
#     FPR2 = (len(a2) - k2) / len(Pcombined);
#     FPR3 = (len(a3) - k3) / len(Pcombined);
#     FPR4 = (len(a4) - k4) / len(Pcombined);
#     FPR5 = (len(a5) - k5) / len(Pcombined);
#     Sen_arr_CS = [0, Sen1, Sen2, Sen3, Sen4, Sen5, 1]
#     FPR_arr_CS = [0, FPR1, FPR2, FPR3, FPR4, FPR5, 1]
#     from sklearn.metrics import auc
#
#     AUC_cs = auc(np.sort(FPR_arr_CS), np.sort(Sen_arr_CS))
#     # print(AUC_cs)
#     AUC_cs_arr.append(AUC_cs)
#
# # print(AUC_cs_arr)
# # print(time_arr_arr)
# # np.savetxt("AUC_ECG_auto_6h_ACT0128.csv", AUC_cs_arr, delimiter=",", fmt='%s')
# # np.savetxt("seizure_labels_ECG_auto_6h_ACT0128.csv", time_arr_arr, delimiter=",", fmt='%s')
# np.savetxt("C:/Users/wxiong/Documents/PHD/2011.1/ACT0128/chance/AUC_ECGauto_6h_ACT0128_2022.csv", AUC_cs_arr, delimiter=",", fmt='%s')



t1 = np.linspace(0 + 0.00416667, 0 + 0.00416667 + 0.00416667 * (len(Raw_variance_EEG_arr) - 1),
                 len(Raw_variance_EEG_arr))
a = np.where(t1 < 12.7977778 + 0)
t1[0:3071] = t1[0:3071] - 0 + 11.2188889
t1[3071:] = t1[3071:] - 12.7977778 - 0
time_feature_arr = []
for i in range(len(t1)):
    if t1[i] > 24:
        time_feature_arr.append(t1[i] - (t1[i] // 24) * 24)
    else:
        time_feature_arr.append(t1[i])
print(len(time_feature_arr))
time_arr = time_feature_arr[17280:]
print(len(time_arr))
new_arr = []
for j in range(0, 432):
    new_arr.append(time_arr[40 * j])

bins_number = 18
bins = np.linspace(0, 24, bins_number + 1)
pro_circadian_time = []
pro_circadian_time_false = []
for i in range(len(new_arr)):
    if new_arr[i] >= bins[0] and new_arr[i] <= bins[1]:
        pro_circadian_time_false.append(0.055799954)
        pro_circadian_time.append(0)
    elif new_arr[i] > bins[1] and new_arr[i] < bins[2]:
        pro_circadian_time_false.append(0.055568419)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[2] and new_arr[i] < bins[3]:
        pro_circadian_time_false.append(0.055568419)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[3] and new_arr[i] < bins[4]:
        pro_circadian_time_false.append(0.055568419)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[4] and new_arr[i] < bins[5]:
        pro_circadian_time_false.append(0.055568419)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[5] and new_arr[i] <= bins[6]:
        pro_circadian_time_false.append(0.055568419)
        pro_circadian_time.append(0)
    elif new_arr[i] > bins[6] and new_arr[i] < bins[7]:
        pro_circadian_time_false.append(0.055568419)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[7] and new_arr[i] <= bins[8]:
        pro_circadian_time_false.append(0.055510535)
        pro_circadian_time.append(0.25)
    elif new_arr[i] > bins[8] and new_arr[i] < bins[9]:
        pro_circadian_time_false.append(0.055336884)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[9] and new_arr[i] < bins[10]:
        pro_circadian_time_false.append(0.055568419)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[10] and new_arr[i] < bins[11]:
        pro_circadian_time_false.append(0.055568419)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[11] and new_arr[i] < bins[12]:
        pro_circadian_time_false.append(0.055510535)
        pro_circadian_time.append(0.25)
    elif new_arr[i] >= bins[12] and new_arr[i] < bins[13]:
        pro_circadian_time_false.append(0.055568419)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[13] and new_arr[i] < bins[14]:
        pro_circadian_time_false.append(0.055510535)
        pro_circadian_time.append(0.25)
    elif new_arr[i] >= bins[14] and new_arr[i] < bins[15]:
        pro_circadian_time_false.append(0.055510535)
        pro_circadian_time.append(0.25)
    elif new_arr[i] >= bins[15] and new_arr[i] < bins[16]:
        pro_circadian_time_false.append(0.055568419)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[16] and new_arr[i] < bins[17]:
        pro_circadian_time_false.append(0.055568419)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[17] and new_arr[i] <= bins[18]:
        pro_circadian_time_false.append(0.055568419)
        pro_circadian_time.append(0)

# RRI_timewindow_arr = t[0:len(pro_circadian_time)]
# print(RRI_timewindow_arr[-1] - RRI_timewindow_arr[0])
# pyplot.figure(figsize=(8, 4))
# pyplot.plot(RRI_timewindow_arr, pro_circadian_time)
# pyplot.annotate('', xy=(76.88111, np.max(pro_circadian_time)), xytext=(76.88111, np.max(pro_circadian_time) + 0.00000000001),
#                 arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(97.923611, np.max(pro_circadian_time)), xytext=(97.923611, np.max(pro_circadian_time) + 0.00000000001),
#                 arrowprops=dict(facecolor='r', shrink=0.05))
# pyplot.annotate('', xy=(119.25278, np.max(pro_circadian_time)), xytext=(119.25278, np.max(pro_circadian_time) + 0.00000000001),
#                 arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.hlines(0.25, RRI_timewindow_arr[0], RRI_timewindow_arr[-1], 'r')
# # pyplot.hlines(0.11111, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'orange')
# # pyplot.hlines(0.22222, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'orange')
# pyplot.title('Forecast seizures in ACT0128')
# pyplot.xlabel('Time(h)')
# pyplot.ylabel('Seizure probability')
# pyplot.show()


Pcombined=split(pro_circadian_time,6)
print(len(Pcombined))
index=[]
for i in range(len(Pcombined)):
    for item in Pcombined[i]:
        if item >= 0.25:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[76.88111,97.923611,119.25278]
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
time_arr=[76.88111,97.923611,119.25278,119.44028]
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
        if item >= 0.3*0.25:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[76.88111,97.923611,119.25278]
k=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k=k+1
            n_arr.append(n)
            pretime.append(m - n)
print(k)
time_arr=[76.88111,97.923611,119.25278,119.44028]
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
        if item >= 0.6*0.25:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[76.88111,97.923611,119.25278]
k=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k=k+1
            n_arr.append(n)
            pretime.append(m - n)
print(k)
time_arr=[76.88111,97.923611,119.25278,119.44028]
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
        if item >= 1.2*0.25:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[76.88111,97.923611,119.25278]
k=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k=k+1
            n_arr.append(n)
            pretime.append(m - n)
print(k)
time_arr=[76.88111,97.923611,119.25278,119.44028]
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
        if item >= 2*0.25:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[76.88111,97.923611,119.25278]
k=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k=k+1
            n_arr.append(n)
            pretime.append(m - n)
print(k)
time_arr=[76.88111,97.923611,119.25278,119.44028]
k=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k=k+1
            n_arr.append(n)
print(k)
print(pretime)
print(np.mean(pretime))


# Pcombined = split(pro_circadian_time, 6)
# print(len(Pcombined))
# time_arr_arr=[]
# AUC_cs_arr=[]
# for i in range(50000):
#     time_arr = np.random.uniform(low=t_window_arr[17280], high=t_window_arr[-1], size=3)
#     time_arr_arr.append(time_arr)
#     time_arr=np.sort(time_arr)
#
#     index = []
#     for i in range(len(Pcombined)):
#         for item in Pcombined[i]:
#             if item >= 0.25:
#                 index.append(6 * i + 0)
#     # print(RRI_timewindow_arr[index])
#     a1 = np.unique(RRI_timewindow_arr[index])
#     # print(a1);
#     # print(len(a1))
#     k1 = 0
#     n_arr = []
#     for m in time_arr:
#         for n in a1:
#             if m - n <= 1 and m - n >= 0:
#                 k1 = k1 + 1
#                 n_arr.append(n)
#     # print(k1)
#     index = []
#     for i in range(len(Pcombined)):
#         for item in Pcombined[i]:
#             if item >= 0.3 * 0.25:
#                 index.append(6 * i + 0)
#     # print(RRI_timewindow_arr[index])
#     a2 = np.unique(RRI_timewindow_arr[index])
#     # print(a2);
#     # print(len(a2))
#     k2 = 0
#     n_arr = []
#     for m in time_arr:
#         for n in a2:
#             if m - n <= 1 and m - n >= 0:
#                 k2 = k2 + 1
#                 n_arr.append(n)
#     # print(k2)
#     index = []
#     for i in range(len(Pcombined)):
#         for item in Pcombined[i]:
#             if item >= 0.6 * 0.25:
#                 index.append(6 * i + 0)
#     # print(RRI_timewindow_arr[index])
#     a3 = np.unique(RRI_timewindow_arr[index])
#     # print(a3);
#     # print(len(a3))
#     k3 = 0
#     n_arr = []
#     for m in time_arr:
#         for n in a3:
#             if m - n <= 1 and m - n >= 0:
#                 k3 = k3 + 1
#                 n_arr.append(n)
#     # print(k3)
#     index = []
#     for i in range(len(Pcombined)):
#         for item in Pcombined[i]:
#             if item >= 1.2 * 0.25:
#                 index.append(6 * i + 0)
#     # print(RRI_timewindow_arr[index])
#     a4 = np.unique(RRI_timewindow_arr[index])
#     # print(a);
#     # print(len(a4))
#     k4 = 0
#     n_arr = []
#     for m in time_arr:
#         for n in a4:
#             if m - n <= 1 and m - n >= 0:
#                 k4 = k4 + 1
#                 n_arr.append(n)
#     # print(k4)
#     index = []
#     for i in range(len(Pcombined)):
#         for item in Pcombined[i]:
#             if item >= 2 * 0.25:
#                 index.append(6 * i + 0)
#     # print(RRI_timewindow_arr[index])
#     a5 = np.unique(RRI_timewindow_arr[index])
#     # print(a5);
#     # print(len(a5))
#     k5 = 0
#     n_arr = []
#     for m in time_arr:
#         for n in a5:
#             if m - n <= 1 and m - n >= 0:
#                 k5 = k5 + 1
#                 n_arr.append(n)
#     # print(k5)
#     Sen1 = k1 / len(time_arr);
#     Sen2 = k2 / len(time_arr);
#     Sen3 = k3 / len(time_arr);
#     Sen4 = k4 / len(time_arr);
#     Sen5 = k5 / len(time_arr);
#     FPR1 = (len(a1) - k1) / len(Pcombined);
#     FPR2 = (len(a2) - k2) / len(Pcombined);
#     FPR3 = (len(a3) - k3) / len(Pcombined);
#     FPR4 = (len(a4) - k4) / len(Pcombined);
#     FPR5 = (len(a5) - k5) / len(Pcombined);
#     Sen_arr_CS = [0, Sen1, Sen2, Sen3, Sen4, Sen5, 1]
#     FPR_arr_CS = [0, FPR1, FPR2, FPR3, FPR4, FPR5, 1]
#     from sklearn.metrics import auc
#
#     AUC_cs = auc(np.sort(FPR_arr_CS), np.sort(Sen_arr_CS))
#     # print(AUC_cs)
#     AUC_cs_arr.append(AUC_cs)
#
# # print(AUC_cs_arr)
# # print(time_arr_arr)
# np.savetxt("C:/Users/wxiong/Documents/PHD/2011.1/ACT0128/chance/AUC_circadian_ACT0128_2022.csv", AUC_cs_arr, delimiter=",", fmt='%s')








# # Pseizureeegvar = 0.000231481;
# # Pnonseizureeegvar = 0.999768519;
# #
# # Pcombined=[]
# # for m in range(len(pro_eegvars_time)):
# #     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_RRIvars_time[m]*pro_RRIautos_time[m]*pro_circadian_time[m]
# #     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
# #     Pcombined.append(P1/(P1+P2))
#
# # Pcombined=[]
# # for m in range(len(pro_eegvars_time)):
# #     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_RRIautos_time[m]*pro_circadian_time[m]
# #     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
# #     Pcombined.append(P1/(P1+P2))
#
# # Pcombined=[]
# # for m in range(len(pro_eegvars_time)):
# #     P1=Pseizureeegvar*pro_eegautos_time[m]*pro_RRIvars_time[m]*pro_circadian_time[m]
# #     P2=Pnonseizureeegvar*(1-pro_eegautos_time_false[m]*pro_RRIvars_time_false[m]*pro_circadian_time_false[m])
# #     Pcombined.append(P1/(P1+P2))
#
# # Pcombined=[]
# # for m in range(len(pro_eegvars_time)):
# #     P1=Pseizureeegvar*pro_eegvars_time[m]*pro_eegautos_time[m]*pro_circadian_time[m]
# #     P2=Pnonseizureeegvar*(1-pro_eegautos_time_false[m]*pro_eegvars_time_false[m]*pro_circadian_time_false[m])
# #     Pcombined.append(P1/(P1+P2))
#
# # Pcombined = []
# # for m in range(len(pro_eegvars_time)):
# #     P1 = Pseizureeegvar * pro_RRIvars_time[m]*pro_RRIautos_time[m] * pro_circadian_time[m]
# #     P2 = Pnonseizureeegvar * (1 - pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m] * pro_circadian_time_false[m])
# #     Pcombined.append(P1 / (P1 + P2))
#
# # Pcombined=[]
# # for m in range(len(pro_eegvars_time)):
# #     P1=Pseizureeegvar*pro_eegvars_time[m]*pro_circadian_time[m]
# #     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_circadian_time_false[m])
# #     Pcombined.append(P1/(P1+P2))
#
# # Pcombined=[]
# # for m in range(len(pro_eegvars_time)):
# #     P1=Pseizureeegvar*pro_eegautos_time[m]*pro_circadian_time[m]
# #     P2=Pnonseizureeegvar*(1-pro_eegautos_time_false[m]*pro_circadian_time_false[m])
# #     Pcombined.append(P1/(P1+P2))
#
#
# # Pcombined = []
# # for m in range(len(pro_eegvars_time)):
# #     P1 = Pseizureeegvar * pro_RRIvars_time[m]* pro_circadian_time[m]
# #     P2 = Pnonseizureeegvar * (1 - pro_RRIvars_time_false[m]* pro_circadian_time_false[m])
# #     Pcombined.append(P1 / (P1 + P2))
#
# # Pcombined = []
# # for m in range(len(pro_eegvars_time)):
# #     P1 = Pseizureeegvar * pro_RRIautos_time[m] * pro_circadian_time[m]
# #     P2 = Pnonseizureeegvar * (1 - pro_RRIautos_time_false[m] * pro_circadian_time_false[m])
# #     Pcombined.append(P1 / (P1 + P2))
#
#
# # pyplot.figure(figsize=(8, 4))
# # RRI_timewindow_arr = t[0:len(pro_circadian_time)]
# # pyplot.plot(RRI_timewindow_arr, Pcombined)
# # pyplot.annotate('', xy=(76.88111, np.max(Pcombined)), xytext=(76.88111, np.max(Pcombined) + 0.00000000001),
# #                 arrowprops=dict(facecolor='black', shrink=0.05))
# # pyplot.annotate('', xy=(97.923611, np.max(Pcombined)), xytext=(97.923611, np.max(Pcombined) + 0.00000000001),
# #                 arrowprops=dict(facecolor='r', shrink=0.05))
# # pyplot.annotate('', xy=(119.25278, np.max(Pcombined)), xytext=(119.25278, np.max(Pcombined) + 0.00000000001),
# #                 arrowprops=dict(facecolor='black', shrink=0.05))
# # pyplot.title('Forecast seizures in ACT0128')
# # pyplot.xlabel('Time(h)')
# # pyplot.ylabel('Seizure probability')
# # pyplot.show()
#
# # Pcombined=split(Pcombined,6)
# # print(len(Pcombined))
# # index=[]
# # for i in range(len(Pcombined)):
# #     for item in Pcombined[i]:
# #         if item >= Th2:
# #             index.append(6*i+0)
# # print(RRI_timewindow_arr[index])
# # a=np.unique(RRI_timewindow_arr[index])
# # print(a); print(len(a))
# # time_arr=[76.88111,97.923611,119.25278]
# # k=0
# # n_arr=[]
# # for m in time_arr:
# #     for n in a:
# #         if m-n<=1 and m-n>=0:
# #             k=k+1
# #             n_arr.append(n)
# # print(k)
# # time_arr=[76.88111,97.923611,119.25278,119.44028]
# # k=0
# # n_arr=[]
# # for m in time_arr:
# #     for n in a:
# #         if m-n<=1 and m-n>=0:
# #             k=k+1
# #             n_arr.append(n)
# # print(k)
# # index=[]
# # for i in range(len(Pcombined)):
# #     for item in Pcombined[i]:
# #         if item >= 0.3*Th2:
# #             index.append(6*i+0)
# # print(RRI_timewindow_arr[index])
# # a=np.unique(RRI_timewindow_arr[index])
# # print(a); print(len(a))
# # time_arr=[76.88111,97.923611,119.25278]
# # k=0
# # n_arr=[]
# # for m in time_arr:
# #     for n in a:
# #         if m-n<=1 and m-n>=0:
# #             k=k+1
# #             n_arr.append(n)
# # print(k)
# # time_arr=[76.88111,97.923611,119.25278,119.44028]
# # k=0
# # n_arr=[]
# # for m in time_arr:
# #     for n in a:
# #         if m-n<=1 and m-n>=0:
# #             k=k+1
# #             n_arr.append(n)
# # print(k)
# # index=[]
# # for i in range(len(Pcombined)):
# #     for item in Pcombined[i]:
# #         if item >= 0.6*Th2:
# #             index.append(6*i+0)
# # print(RRI_timewindow_arr[index])
# # a=np.unique(RRI_timewindow_arr[index])
# # print(a); print(len(a))
# # time_arr=[76.88111,97.923611,119.25278]
# # k=0
# # n_arr=[]
# # for m in time_arr:
# #     for n in a:
# #         if m-n<=1 and m-n>=0:
# #             k=k+1
# #             n_arr.append(n)
# # print(k)
# # time_arr=[76.88111,97.923611,119.25278,119.44028]
# # k=0
# # n_arr=[]
# # for m in time_arr:
# #     for n in a:
# #         if m-n<=1 and m-n>=0:
# #             k=k+1
# #             n_arr.append(n)
# # print(k)
# # index=[]
# # for i in range(len(Pcombined)):
# #     for item in Pcombined[i]:
# #         if item >= 1.2*Th2:
# #             index.append(6*i+0)
# # print(RRI_timewindow_arr[index])
# # a=np.unique(RRI_timewindow_arr[index])
# # print(a); print(len(a))
# # time_arr=[76.88111,97.923611,119.25278]
# # k=0
# # n_arr=[]
# # for m in time_arr:
# #     for n in a:
# #         if m-n<=1 and m-n>=0:
# #             k=k+1
# #             n_arr.append(n)
# # print(k)
# # time_arr=[76.88111,97.923611,119.25278,119.44028]
# # k=0
# # n_arr=[]
# # for m in time_arr:
# #     for n in a:
# #         if m-n<=1 and m-n>=0:
# #             k=k+1
# #             n_arr.append(n)
# # print(k)
# # index=[]
# # for i in range(len(Pcombined)):
# #     for item in Pcombined[i]:
# #         if item >= 2*Th2:
# #             index.append(6*i+0)
# # print(RRI_timewindow_arr[index])
# # a=np.unique(RRI_timewindow_arr[index])
# # print(a); print(len(a))
# # time_arr=[76.88111,97.923611,119.25278]
# # k=0
# # n_arr=[]
# # for m in time_arr:
# #     for n in a:
# #         if m-n<=1 and m-n>=0:
# #             k=k+1
# #             n_arr.append(n)
# # print(k)
# # time_arr=[76.88111,97.923611,119.25278,119.44028]
# # k=0
# # n_arr=[]
# # for m in time_arr:
# #     for n in a:
# #         if m-n<=1 and m-n>=0:
# #             k=k+1
# #             n_arr.append(n)
# # print(k)