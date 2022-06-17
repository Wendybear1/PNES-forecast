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







csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD1282/Cz_EEGvariance_QLD1282_15s_3h.csv', sep=',',
                         header=None)
Raw_variance_EEG = csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD1282/Cz_EEGauto_QLD1282_15s_3h.csv', sep=',',
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
print(t_window_arr[19450]);
print(t_window_arr[19450] - t_window_arr[0]);
print(t_window_arr[-1] - t_window_arr[19450]);

window_time_arr = t_window_arr
# pyplot.plot(window_time_arr, Raw_variance_EEG_arr, 'grey', alpha=0.5)
# pyplot.ylabel('Voltage', fontsize=13)
# pyplot.xlabel('Time (hours)', fontsize=13)
# pyplot.title('raw EEG variance in QLD1282', fontsize=13)
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
# pyplot.title('EEG variance in QLD1282', fontsize=13)
# pyplot.show()

seizure_timing_index = []
for k in range(len(window_time_arr)):
    # if window_time_arr[k] < 1.8288889 and window_time_arr[k + 1] >= 1.8288889:
    #     seizure_timing_index.append(k)
    if window_time_arr[k] < 6.6688889 and window_time_arr[k + 1] >= 6.6688889:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 18.9325 and window_time_arr[k + 1] >= 18.9325:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 24.342778 and window_time_arr[k + 1] >= 24.342778:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 29.681388 and window_time_arr[k + 1] >= 29.681388:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 43.735 and window_time_arr[k + 1] >= 43.735:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 44.365278 and window_time_arr[k + 1] >= 44.365278:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 49.186389 and window_time_arr[k + 1] >= 49.186389:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 54.781944 and window_time_arr[k + 1] >= 54.781944:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 69.106388 and window_time_arr[k + 1] >= 69.106388:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 73.85222 and window_time_arr[k + 1] >= 73.85222:
        seizure_timing_index.append(k)
    # if window_time_arr[k] < 95.493611 and window_time_arr[k + 1] >= 95.493611:
    #     seizure_timing_index.append(k)
    # if window_time_arr[k] < 97.3675 and window_time_arr[k + 1] >= 97.3675:
    #     seizure_timing_index.append(k)
    # if window_time_arr[k] < 99.079444 and window_time_arr[k + 1] >= 99.079444:
    #     seizure_timing_index.append(k)
    # if window_time_arr[k] < 102.43722 and window_time_arr[k + 1] >= 102.43722:
    #     seizure_timing_index.append(k)
    # if window_time_arr[k] < 120.36472 and window_time_arr[k + 1] >= 120.36472:
    #     seizure_timing_index.append(k)
    # if window_time_arr[k] < 124.15888 and window_time_arr[k + 1] >= 124.15888:
    #     seizure_timing_index.append(k)
    # if window_time_arr[k] < 139.29889 and window_time_arr[k + 1] >= 139.29889:
    #     seizure_timing_index.append(k)
print(seizure_timing_index)




# # # # # # # ### EEG variance
window_time_arr = t_window_arr[0:19450]
Raw_variance_EEG = Raw_variance_EEG[0:19450]
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
# pyplot.title('raw EEG autocorrelation in QLD1282', fontsize=13)
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
# pyplot.title('EEG autocorrelation in QLD1282', fontsize=13)
# pyplot.show()

Raw_auto_EEG = Raw_auto_EEG_arr[0:19450]
window_time_arr = t_window_arr[0:19450]
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
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD1282/RRI_ch31_timewindowarr_QLD1282_15s_3h.csv',
                         sep=',', header=None)
rri_t = csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD1282/RRI_ch31_rawvariance_QLD1282_15s_3h.csv',
                         sep=',', header=None)
RRI_var = csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD1282/RRI_ch31_rawauto_QLD1282_15s_3h.csv', sep=',',
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
# pyplot.title('RRI variance in QLD1282', fontsize=13)
# pyplot.show()
#
# pyplot.plot(Raw_auto_RRI31_arr, 'grey', alpha=0.5)
# pyplot.xlabel('Time (hours)', fontsize=13)
# pyplot.title('RRI autocorrelation in QLD1282', fontsize=13)
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
# width = 2 * np.pi / bins_number
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
# a = np.where(t < 12.1372222 + 0)
# print(a);
# print(t[2911]);print(t[2912]);
# t[0:2912] = t[0:2912] - 0 + 11.8627778
# t[2912:] = t[2912:] - 12.1372222 - 0
# print(t[2911]);print(t[2912]);print(t[2913]);print(t[0])
#
# time_feature_arr = []
# for i in range(len(t)):
#     if t[i] > 24:
#         time_feature_arr.append(t[i] - (t[i] // 24) * 24)
#     else:
#         time_feature_arr.append(t[i])
# seizure_time = [time_feature_arr[1599], time_feature_arr[4542], time_feature_arr[5841], time_feature_arr[7122],
#                 time_feature_arr[10495],time_feature_arr[10646],
#                 time_feature_arr[11803],time_feature_arr[13146],time_feature_arr[16584],time_feature_arr[17723],
#                 ]
# print(seizure_time)
# bins_number = 18
# bins = np.linspace(0, 24, bins_number + 1)
# nEEGsvar, _, _ = pyplot.hist(time_feature_arr[0:19450], bins)
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
# ax.bar(bins[:bins_number], nEEGsvarsei/sum(nEEGsvarsei),width=width, color='grey',alpha=0.7,edgecolor='grey',linewidth=2)
# pyplot.setp(ax.get_yticklabels(), color='k')
# # ax.set_title('seizure timing histogram (SA0124)',fontsize=23)
# ax.set_xticks(np.linspace(0, 2*np.pi, 24, endpoint=False))
# ax.set_xticklabels(['0 am','','','Night','','','6 am','','','Morning','','','12 am','','','Afternoon','','','18 pm','','','Evening','','','24 pm'],fontsize=16)
# # locs, labels = pyplot.xticks([0,3,6,9,12,15,18,21,24],['0','Night','6','Morning','12','Afternoon','18','Evening','24'],fontsize=16)
# # locs, labels = pyplot.yticks([0.1,0.2,0.3],['0.1','0.2','0.3'],fontsize=16)
# locs, labels = pyplot.yticks([0.2,0.6,1],['0.2','0.6','1'],fontsize=16)
# ax.annotate("", xy=(-2.876610224, 0.48249798), xytext=(0, 0),arrowprops=dict(arrowstyle="->",color='g',linewidth=2))
# pyplot.show()
#
# bins_number = 18
# bins = np.linspace(0, 24, bins_number + 1)
# ntimes, _, _ = pyplot.hist(time_feature_arr[0:19450], bins)
# ntimesei, _, _ = pyplot.hist(seizure_time, bins)
# print(ntimes)
# print(ntimesei)




# #### section 2 training training
medium_rhythm_var_arr_3 = movingaverage(Raw_variance_EEG, 20*1)
long_rhythm_var_arr = medium_rhythm_var_arr_3
var_trans = hilbert(long_rhythm_var_arr)
var_phase = np.angle(var_trans)
phase_long_EEGvariance_arr = var_phase
print(len(phase_long_EEGvariance_arr));
medium_rhythm_value_arr_3 = movingaverage(Raw_auto_EEG, 20*1)
long_rhythm_value_arr = medium_rhythm_value_arr_3
value_trans = hilbert(long_rhythm_value_arr)
value_phase = np.angle(value_trans)
phase_long_EEGauto_arr = value_phase
print(len(phase_long_EEGauto_arr));
medium_rhythm_RRIvar_arr_3 = movingaverage(Raw_variance_RRI31, 20*1)
long_rhythm_RRIvar_arr = medium_rhythm_RRIvar_arr_3
var_trans = hilbert(long_rhythm_RRIvar_arr)
var_phase = np.angle(var_trans)
phase_long_RRIvar_arr = var_phase
print(len(phase_long_RRIvar_arr));
medium_rhythm_RRIvalue_arr_3 = movingaverage(Raw_auto_RRI31, 20*1)
long_rhythm_RRIvalue_arr = medium_rhythm_RRIvalue_arr_3
value_trans = hilbert(long_rhythm_RRIvalue_arr)
value_phase = np.angle(value_trans)
phase_long_RRIauto_arr = value_phase
print(len(phase_long_RRIauto_arr));


### combined probability calculation
## 24h 24h 24h
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
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] > bins[7] and phase_long_EEGvariance_arr[i] < bins[8]:
#         pro_eegvars_time_false.append(0.098868313)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[8] and phase_long_EEGvariance_arr[i] <= bins[9]:
#         pro_eegvars_time_false.append(0.624125514)
#         pro_eegvars_time.append(0.7)
#     elif phase_long_EEGvariance_arr[i] > bins[9] and phase_long_EEGvariance_arr[i] < bins[10]:
#         pro_eegvars_time_false.append(0.150977366)
#         pro_eegvars_time.append(0.3)
#     elif phase_long_EEGvariance_arr[i] >= bins[10] and phase_long_EEGvariance_arr[i] <= bins[11]:
#         pro_eegvars_time_false.append(0.053909465)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] > bins[11] and phase_long_EEGvariance_arr[i] < bins[12]:
#         pro_eegvars_time_false.append(0.022685185)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] > bins[12] and phase_long_EEGvariance_arr[i] < bins[13]:
#         pro_eegvars_time_false.append(0.024382716)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[13] and phase_long_EEGvariance_arr[i] < bins[14]:
#         pro_eegvars_time_false.append(0.02505144)
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
# print(pro_eegvars_time[1599]);print(pro_eegvars_time[4542]);print(pro_eegvars_time[5841]);print(pro_eegvars_time[7122]);
# print(pro_eegvars_time[10495]);print(pro_eegvars_time[10646]);print(pro_eegvars_time[11803]);print(pro_eegvars_time[13146]);
# print(pro_eegvars_time[16584]);print(pro_eegvars_time[17723]);
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
#         pro_eegautos_time_false.append(0.170781893)
#         pro_eegautos_time.append(0.1)
#     elif phase_long_EEGauto_arr[i] >= bins[8] and phase_long_EEGauto_arr[i] < bins[9]:
#         pro_eegautos_time_false.append(0.457253086)
#         pro_eegautos_time.append(0.7)
#     elif phase_long_EEGauto_arr[i] >= bins[9] and phase_long_EEGauto_arr[i] <= bins[10]:
#         pro_eegautos_time_false.append(0.253858025)
#         pro_eegautos_time.append(0.2)
#     elif phase_long_EEGauto_arr[i] > bins[10] and phase_long_EEGauto_arr[i] < bins[11]:
#         pro_eegautos_time_false.append(0.059053498)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[11] and phase_long_EEGauto_arr[i] < bins[12]:
#         pro_eegautos_time_false.append(0.020010288)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[12] and phase_long_EEGauto_arr[i] < bins[13]:
#         pro_eegautos_time_false.append(0.019444444)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[13] and phase_long_EEGauto_arr[i] < bins[14]:
#         pro_eegautos_time_false.append(0.019598765)
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
# print(pro_eegautos_time[1599]);print(pro_eegautos_time[4542]);print(pro_eegautos_time[5841]);print(pro_eegautos_time[7122]);
# print(pro_eegautos_time[10495]);print(pro_eegautos_time[10646]);print(pro_eegautos_time[11803]);print(pro_eegautos_time[13146]);
# print(pro_eegautos_time[16584]);print(pro_eegautos_time[17723]);
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
#         pro_RRIautos_time_false.append(0.184516461)
#         pro_RRIautos_time.append(0.1)
#     elif phase_long_RRIauto_arr[i] > bins[8] and phase_long_RRIauto_arr[i] <= bins[9]:
#         pro_RRIautos_time_false.append(0.416820988)
#         pro_RRIautos_time.append(0.7)
#     elif phase_long_RRIauto_arr[i] > bins[9] and phase_long_RRIauto_arr[i] <= bins[10]:
#         pro_RRIautos_time_false.append(0.28117284)
#         pro_RRIautos_time.append(0.2)
#     elif phase_long_RRIauto_arr[i] > bins[10] and phase_long_RRIauto_arr[i] < bins[11]:
#         pro_RRIautos_time_false.append(0.050462963)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[11] and phase_long_RRIauto_arr[i] < bins[12]:
#         pro_RRIautos_time_false.append(0.017849794)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[12] and phase_long_RRIauto_arr[i] < bins[13]:
#         pro_RRIautos_time_false.append(0.019187243)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[13] and phase_long_RRIauto_arr[i] < bins[14]:
#         pro_RRIautos_time_false.append(0.029989712)
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
# print(pro_RRIautos_time[1599]);print(pro_RRIautos_time[4542]);print(pro_RRIautos_time[5841]);print(pro_RRIautos_time[7122]);
# print(pro_RRIautos_time[10495]);print(pro_RRIautos_time[10646]);print(pro_RRIautos_time[11803]);print(pro_RRIautos_time[13146]);
# print(pro_RRIautos_time[16584]);print(pro_RRIautos_time[17723]);


#### 12h 12h 12h
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
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] > bins[7] and phase_long_EEGvariance_arr[i] < bins[8]:
#         pro_eegvars_time_false.append(0.000205761)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[8] and phase_long_EEGvariance_arr[i] <= bins[9]:
#         pro_eegvars_time_false.append(0.634876543)
#         pro_eegvars_time.append(0.6)
#     elif phase_long_EEGvariance_arr[i] > bins[9] and phase_long_EEGvariance_arr[i] < bins[10]:
#         pro_eegvars_time_false.append(0.281790123)
#         pro_eegvars_time.append(0.4)
#     elif phase_long_EEGvariance_arr[i] >= bins[10] and phase_long_EEGvariance_arr[i] <= bins[11]:
#         pro_eegvars_time_false.append(0.039197531)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] > bins[11] and phase_long_EEGvariance_arr[i] < bins[12]:
#         pro_eegvars_time_false.append(0.017592593)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] > bins[12] and phase_long_EEGvariance_arr[i] < bins[13]:
#         pro_eegvars_time_false.append(0.011162551)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[13] and phase_long_EEGvariance_arr[i] < bins[14]:
#         pro_eegvars_time_false.append(0.015174897)
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
# print(pro_eegvars_time[1599]);print(pro_eegvars_time[4542]);print(pro_eegvars_time[5841]);print(pro_eegvars_time[7122]);
# print(pro_eegvars_time[10495]);print(pro_eegvars_time[10646]);print(pro_eegvars_time[11803]);print(pro_eegvars_time[13146]);
# print(pro_eegvars_time[16584]);print(pro_eegvars_time[17723]);
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
#         pro_eegautos_time_false.append(0.082407407)
#         pro_eegautos_time.append(0.1)
#     elif phase_long_EEGauto_arr[i] >= bins[8] and phase_long_EEGauto_arr[i] < bins[9]:
#         pro_eegautos_time_false.append(0.478858025)
#         pro_eegautos_time.append(0.5)
#     elif phase_long_EEGauto_arr[i] >= bins[9] and phase_long_EEGauto_arr[i] <= bins[10]:
#         pro_eegautos_time_false.append(0.375565844)
#         pro_eegautos_time.append(0.4)
#     elif phase_long_EEGauto_arr[i] > bins[10] and phase_long_EEGauto_arr[i] < bins[11]:
#         pro_eegautos_time_false.append(0.034876543)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[11] and phase_long_EEGauto_arr[i] < bins[12]:
#         pro_eegautos_time_false.append(0.010493827)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[12] and phase_long_EEGauto_arr[i] < bins[13]:
#         pro_eegautos_time_false.append(0.0093107)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[13] and phase_long_EEGauto_arr[i] < bins[14]:
#         pro_eegautos_time_false.append(0.008487654)
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
# print(pro_eegautos_time[1599]);print(pro_eegautos_time[4542]);print(pro_eegautos_time[5841]);print(pro_eegautos_time[7122]);
# print(pro_eegautos_time[10495]);print(pro_eegautos_time[10646]);print(pro_eegautos_time[11803]);print(pro_eegautos_time[13146]);
# print(pro_eegautos_time[16584]);print(pro_eegautos_time[17723]);
# pro_RRIvars_time = []
# pro_RRIvars_time_false = []
# for i in range(len(phase_long_RRIvar_arr)):
#     if phase_long_RRIvar_arr[i] >= bins[0] and phase_long_RRIvar_arr[i] < bins[1]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[1] and phase_long_RRIvar_arr[i] < bins[2]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[2] and phase_long_RRIvar_arr[i] < bins[3]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[3] and phase_long_RRIvar_arr[i] < bins[4]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[4] and phase_long_RRIvar_arr[i] < bins[5]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[5] and phase_long_RRIvar_arr[i] < bins[6]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[6] and phase_long_RRIvar_arr[i] < bins[7]:
#         pro_RRIvars_time_false.append(0.07659465)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[7] and phase_long_RRIvar_arr[i] <= bins[8]:
#         pro_RRIvars_time_false.append(0.249279835)
#         pro_RRIvars_time.append(0.5)
#     elif phase_long_RRIvar_arr[i] > bins[8] and phase_long_RRIvar_arr[i] < bins[9]:
#         pro_RRIvars_time_false.append(0.210442387)
#         pro_RRIvars_time.append(0.1)
#     elif phase_long_RRIvar_arr[i] >= bins[9] and phase_long_RRIvar_arr[i] <= bins[10]:
#         pro_RRIvars_time_false.append(0.174382716)
#         pro_RRIvars_time.append(0.1)
#     elif phase_long_RRIvar_arr[i] > bins[10] and phase_long_RRIvar_arr[i] <= bins[11]:
#         pro_RRIvars_time_false.append(0.169804527)
#         pro_RRIvars_time.append(0.2)
#     elif phase_long_RRIvar_arr[i] > bins[11] and phase_long_RRIvar_arr[i] < bins[12]:
#         pro_RRIvars_time_false.append(0.098919753)
#         pro_RRIvars_time.append(0.1)
#     elif phase_long_RRIvar_arr[i] >= bins[12] and phase_long_RRIvar_arr[i] < bins[13]:
#         pro_RRIvars_time_false.append(0.017541152)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[13] and phase_long_RRIvar_arr[i] < bins[14]:
#         pro_RRIvars_time_false.append(0.003034979)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[14] and phase_long_RRIvar_arr[i] < bins[15]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[15] and phase_long_RRIvar_arr[i] < bins[16]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[16] and phase_long_RRIvar_arr[i] < bins[17]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[17]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
# print(pro_RRIvars_time[1599]);print(pro_RRIvars_time[4542]);print(pro_RRIvars_time[5841]);print(pro_RRIvars_time[7122]);
# print(pro_RRIvars_time[10495]);print(pro_RRIvars_time[10646]);print(pro_RRIvars_time[11803]);print(pro_RRIvars_time[13146]);
# print(pro_RRIvars_time[16584]);print(pro_RRIvars_time[17723]);


# # # # ##### 6h 6h 6h 6h
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
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] > bins[7] and phase_long_EEGvariance_arr[i] < bins[8]:
#         pro_eegvars_time_false.append(0.000617284)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[8] and phase_long_EEGvariance_arr[i] <= bins[9]:
#         pro_eegvars_time_false.append(0.574537037)
#         pro_eegvars_time.append(0.5)
#     elif phase_long_EEGvariance_arr[i] > bins[9] and phase_long_EEGvariance_arr[i] < bins[10]:
#         pro_eegvars_time_false.append(0.361111111)
#         pro_eegvars_time.append(0.5)
#     elif phase_long_EEGvariance_arr[i] >= bins[10] and phase_long_EEGvariance_arr[i] <= bins[11]:
#         pro_eegvars_time_false.append(0.042283951)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] > bins[11] and phase_long_EEGvariance_arr[i] < bins[12]:
#         pro_eegvars_time_false.append(0.007150206)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] > bins[12] and phase_long_EEGvariance_arr[i] < bins[13]:
#         pro_eegvars_time_false.append(0.007047325)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[13] and phase_long_EEGvariance_arr[i] < bins[14]:
#         pro_eegvars_time_false.append(0.007253086)
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
# print(pro_eegvars_time[1599]);print(pro_eegvars_time[4542]);print(pro_eegvars_time[5841]);print(pro_eegvars_time[7122]);
# print(pro_eegvars_time[10495]);print(pro_eegvars_time[10646]);print(pro_eegvars_time[11803]);print(pro_eegvars_time[13146]);
# print(pro_eegvars_time[16584]);print(pro_eegvars_time[17723]);
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
#         pro_eegautos_time_false.append(0.030092593)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[8] and phase_long_EEGauto_arr[i] < bins[9]:
#         pro_eegautos_time_false.append(0.46882716)
#         pro_eegautos_time.append(0.5)
#     elif phase_long_EEGauto_arr[i] >= bins[9] and phase_long_EEGauto_arr[i] <= bins[10]:
#         pro_eegautos_time_false.append(0.474331276)
#         pro_eegautos_time.append(0.5)
#     elif phase_long_EEGauto_arr[i] > bins[10] and phase_long_EEGauto_arr[i] < bins[11]:
#         pro_eegautos_time_false.append(0.014300412)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[11] and phase_long_EEGauto_arr[i] < bins[12]:
#         pro_eegautos_time_false.append(0.004835391)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[12] and phase_long_EEGauto_arr[i] < bins[13]:
#         pro_eegautos_time_false.append(0.003549383)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[13] and phase_long_EEGauto_arr[i] < bins[14]:
#         pro_eegautos_time_false.append(0.004063786)
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
# print(pro_eegautos_time[1599]);print(pro_eegautos_time[4542]);print(pro_eegautos_time[5841]);print(pro_eegautos_time[7122]);
# print(pro_eegautos_time[10495]);print(pro_eegautos_time[10646]);print(pro_eegautos_time[11803]);print(pro_eegautos_time[13146]);
# print(pro_eegautos_time[16584]);print(pro_eegautos_time[17723]);
# pro_RRIvars_time = []
# pro_RRIvars_time_false = []
# for i in range(len(phase_long_RRIvar_arr)):
#     if phase_long_RRIvar_arr[i] >= bins[0] and phase_long_RRIvar_arr[i] < bins[1]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[1] and phase_long_RRIvar_arr[i] < bins[2]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[2] and phase_long_RRIvar_arr[i] < bins[3]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[3] and phase_long_RRIvar_arr[i] < bins[4]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[4] and phase_long_RRIvar_arr[i] < bins[5]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[5] and phase_long_RRIvar_arr[i] < bins[6]:
#         pro_RRIvars_time_false.append(0.025)
#         pro_RRIvars_time.append(0.1)
#     elif phase_long_RRIvar_arr[i] >= bins[6] and phase_long_RRIvar_arr[i] < bins[7]:
#         pro_RRIvars_time_false.append(0.172788066)
#         pro_RRIvars_time.append(0.2)
#     elif phase_long_RRIvar_arr[i] >= bins[7] and phase_long_RRIvar_arr[i] <= bins[8]:
#         pro_RRIvars_time_false.append(0.22340535)
#         pro_RRIvars_time.append(0.4)
#     elif phase_long_RRIvar_arr[i] > bins[8] and phase_long_RRIvar_arr[i] < bins[9]:
#         pro_RRIvars_time_false.append(0.178240741)
#         pro_RRIvars_time.append(0.2)
#     elif phase_long_RRIvar_arr[i] >= bins[9] and phase_long_RRIvar_arr[i] <= bins[10]:
#         pro_RRIvars_time_false.append(0.084259259)
#         pro_RRIvars_time.append(0.1)
#     elif phase_long_RRIvar_arr[i] > bins[10] and phase_long_RRIvar_arr[i] <= bins[11]:
#         pro_RRIvars_time_false.append(0.072942387)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] > bins[11] and phase_long_RRIvar_arr[i] < bins[12]:
#         pro_RRIvars_time_false.append(0.10622428)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[12] and phase_long_RRIvar_arr[i] < bins[13]:
#         pro_RRIvars_time_false.append(0.123559671)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[13] and phase_long_RRIvar_arr[i] < bins[14]:
#         pro_RRIvars_time_false.append(0.013580247)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[14] and phase_long_RRIvar_arr[i] < bins[15]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[15] and phase_long_RRIvar_arr[i] < bins[16]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[16] and phase_long_RRIvar_arr[i] < bins[17]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[17]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
# print(pro_RRIvars_time[1599]);print(pro_RRIvars_time[4542]);print(pro_RRIvars_time[5841]);print(pro_RRIvars_time[7122]);
# print(pro_RRIvars_time[10495]);print(pro_RRIvars_time[10646]);print(pro_RRIvars_time[11803]);print(pro_RRIvars_time[13146]);
# print(pro_RRIvars_time[16584]);print(pro_RRIvars_time[17723]);
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
#         pro_RRIautos_time_false.append(0.169393004)
#         pro_RRIautos_time.append(0.1)
#     elif phase_long_RRIauto_arr[i] > bins[8] and phase_long_RRIauto_arr[i] <= bins[9]:
#         pro_RRIautos_time_false.append(0.25617284)
#         pro_RRIautos_time.append(0.2)
#     elif phase_long_RRIauto_arr[i] > bins[9] and phase_long_RRIauto_arr[i] <= bins[10]:
#         pro_RRIautos_time_false.append(0.540226337)
#         pro_RRIautos_time.append(0.7)
#     elif phase_long_RRIauto_arr[i] > bins[10] and phase_long_RRIauto_arr[i] < bins[11]:
#         pro_RRIautos_time_false.append(0.012602881)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[11] and phase_long_RRIauto_arr[i] < bins[12]:
#         pro_RRIautos_time_false.append(0.005401235)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[12] and phase_long_RRIauto_arr[i] < bins[13]:
#         pro_RRIautos_time_false.append(0.006687243)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[13] and phase_long_RRIauto_arr[i] < bins[14]:
#         pro_RRIautos_time_false.append(0.009516461)
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
# print(pro_RRIautos_time[1599]);print(pro_RRIautos_time[4542]);print(pro_RRIautos_time[5841]);print(pro_RRIautos_time[7122]);
# print(pro_RRIautos_time[10495]);print(pro_RRIautos_time[10646]);print(pro_RRIautos_time[11803]);print(pro_RRIautos_time[13146]);
# print(pro_RRIautos_time[16584]);print(pro_RRIautos_time[17723]);


#### 1h 1h 1h 1h
# bins_number = 18
# bins = np.linspace(-np.pi, np.pi, bins_number + 1)
# pro_RRIvars_time = []
# pro_RRIvars_time_false = []
# for i in range(len(phase_long_RRIvar_arr)):
#     if phase_long_RRIvar_arr[i] >= bins[0] and phase_long_RRIvar_arr[i] < bins[1]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[1] and phase_long_RRIvar_arr[i] < bins[2]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[2] and phase_long_RRIvar_arr[i] < bins[3]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[3] and phase_long_RRIvar_arr[i] < bins[4]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[4] and phase_long_RRIvar_arr[i] < bins[5]:
#         pro_RRIvars_time_false.append(0.004475309)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[5] and phase_long_RRIvar_arr[i] < bins[6]:
#         pro_RRIvars_time_false.append(0.094495885)
#         pro_RRIvars_time.append(0.1)
#     elif phase_long_RRIvar_arr[i] >= bins[6] and phase_long_RRIvar_arr[i] < bins[7]:
#         pro_RRIvars_time_false.append(0.149176955)
#         pro_RRIvars_time.append(0.2)
#     elif phase_long_RRIvar_arr[i] >= bins[7] and phase_long_RRIvar_arr[i] <= bins[8]:
#         pro_RRIvars_time_false.append(0.191152263)
#         pro_RRIvars_time.append(0.2)
#     elif phase_long_RRIvar_arr[i] > bins[8] and phase_long_RRIvar_arr[i] < bins[9]:
#         pro_RRIvars_time_false.append(0.146296296)
#         pro_RRIvars_time.append(0.1)
#     elif phase_long_RRIvar_arr[i] >= bins[9] and phase_long_RRIvar_arr[i] <= bins[10]:
#         pro_RRIvars_time_false.append(0.072067901)
#         pro_RRIvars_time.append(0.2)
#     elif phase_long_RRIvar_arr[i] > bins[10] and phase_long_RRIvar_arr[i] <= bins[11]:
#         pro_RRIvars_time_false.append(0.077417695)
#         pro_RRIvars_time.append(0.2)
#     elif phase_long_RRIvar_arr[i] > bins[11] and phase_long_RRIvar_arr[i] < bins[12]:
#         pro_RRIvars_time_false.append(0.066820988)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[12] and phase_long_RRIvar_arr[i] < bins[13]:
#         pro_RRIvars_time_false.append(0.104526749)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[13] and phase_long_RRIvar_arr[i] < bins[14]:
#         pro_RRIvars_time_false.append(0.093569959)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[14] and phase_long_RRIvar_arr[i] < bins[15]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[15] and phase_long_RRIvar_arr[i] < bins[16]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[16] and phase_long_RRIvar_arr[i] < bins[17]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[17]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
# print(pro_RRIvars_time[1599]);print(pro_RRIvars_time[4542]);print(pro_RRIvars_time[5841]);print(pro_RRIvars_time[7122]);
# print(pro_RRIvars_time[10495]);print(pro_RRIvars_time[10646]);print(pro_RRIvars_time[11803]);print(pro_RRIvars_time[13146]);
# print(pro_RRIvars_time[16584]);print(pro_RRIvars_time[17723]);


#### 5min 5min 5min 5min
bins_number = 18
bins = np.linspace(-np.pi, np.pi, bins_number + 1)
pro_RRIvars_time = []
pro_RRIvars_time_false = []
for i in range(len(phase_long_RRIvar_arr)):
    if phase_long_RRIvar_arr[i] >= bins[0] and phase_long_RRIvar_arr[i] < bins[1]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvar_arr[i] >= bins[1] and phase_long_RRIvar_arr[i] < bins[2]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvar_arr[i] >= bins[2] and phase_long_RRIvar_arr[i] < bins[3]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvar_arr[i] >= bins[3] and phase_long_RRIvar_arr[i] < bins[4]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvar_arr[i] >= bins[4] and phase_long_RRIvar_arr[i] < bins[5]:
        pro_RRIvars_time_false.append(0.046347737)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvar_arr[i] >= bins[5] and phase_long_RRIvar_arr[i] < bins[6]:
        pro_RRIvars_time_false.append(0.15473251)
        pro_RRIvars_time.append(0.1)
    elif phase_long_RRIvar_arr[i] >= bins[6] and phase_long_RRIvar_arr[i] < bins[7]:
        pro_RRIvars_time_false.append(0.128960905)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvar_arr[i] >= bins[7] and phase_long_RRIvar_arr[i] <= bins[8]:
        pro_RRIvars_time_false.append(0.1093107)
        pro_RRIvars_time.append(0.4)
    elif phase_long_RRIvar_arr[i] > bins[8] and phase_long_RRIvar_arr[i] < bins[9]:
        pro_RRIvars_time_false.append(0.095679012)
        pro_RRIvars_time.append(0.1)
    elif phase_long_RRIvar_arr[i] >= bins[9] and phase_long_RRIvar_arr[i] <= bins[10]:
        pro_RRIvars_time_false.append(0.088425926)
        pro_RRIvars_time.append(0.2)
    elif phase_long_RRIvar_arr[i] > bins[10] and phase_long_RRIvar_arr[i] <= bins[11]:
        pro_RRIvars_time_false.append(0.063065844)
        pro_RRIvars_time.append(0.1)
    elif phase_long_RRIvar_arr[i] > bins[11] and phase_long_RRIvar_arr[i] < bins[12]:
        pro_RRIvars_time_false.append(0.066100823)
        pro_RRIvars_time.append(0.1)
    elif phase_long_RRIvar_arr[i] >= bins[12] and phase_long_RRIvar_arr[i] < bins[13]:
        pro_RRIvars_time_false.append(0.107304527)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvar_arr[i] >= bins[13] and phase_long_RRIvar_arr[i] < bins[14]:
        pro_RRIvars_time_false.append(0.140072016)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvar_arr[i] >= bins[14] and phase_long_RRIvar_arr[i] < bins[15]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvar_arr[i] >= bins[15] and phase_long_RRIvar_arr[i] < bins[16]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvar_arr[i] >= bins[16] and phase_long_RRIvar_arr[i] < bins[17]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvar_arr[i] >= bins[17]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
print(pro_RRIvars_time[1599]);print(pro_RRIvars_time[4542]);print(pro_RRIvars_time[5841]);print(pro_RRIvars_time[7122]);
print(pro_RRIvars_time[10495]);print(pro_RRIvars_time[10646]);print(pro_RRIvars_time[11803]);print(pro_RRIvars_time[13146]);
print(pro_RRIvars_time[16584]);print(pro_RRIvars_time[17723]);
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
        pro_RRIautos_time_false.append(0.011574074)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[6] and phase_long_RRIauto_arr[i] < bins[7]:
        pro_RRIautos_time_false.append(0.048868313)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[7] and phase_long_RRIauto_arr[i] <= bins[8]:
        pro_RRIautos_time_false.append(0.11867284)
        pro_RRIautos_time.append(0.2)
    elif phase_long_RRIauto_arr[i] > bins[8] and phase_long_RRIauto_arr[i] <= bins[9]:
        pro_RRIautos_time_false.append(0.305452675)
        pro_RRIautos_time.append(0.4)
    elif phase_long_RRIauto_arr[i] > bins[9] and phase_long_RRIauto_arr[i] <= bins[10]:
        pro_RRIautos_time_false.append(0.324434156)
        pro_RRIautos_time.append(0.3)
    elif phase_long_RRIauto_arr[i] > bins[10] and phase_long_RRIauto_arr[i] < bins[11]:
        pro_RRIautos_time_false.append(0.154115226)
        pro_RRIautos_time.append(0.1)
    elif phase_long_RRIauto_arr[i] >= bins[11] and phase_long_RRIauto_arr[i] < bins[12]:
        pro_RRIautos_time_false.append(0.034722222)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[12] and phase_long_RRIauto_arr[i] < bins[13]:
        pro_RRIautos_time_false.append(0.001903292)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[13] and phase_long_RRIauto_arr[i] < bins[14]:
        pro_RRIautos_time_false.append(0.000257202)
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
print(pro_RRIautos_time[1599]);print(pro_RRIautos_time[4542]);print(pro_RRIautos_time[5841]);print(pro_RRIautos_time[7122]);
print(pro_RRIautos_time[10495]);print(pro_RRIautos_time[10646]);print(pro_RRIautos_time[11803]);print(pro_RRIautos_time[13146]);
print(pro_RRIautos_time[16584]);print(pro_RRIautos_time[17723]);




Pseizureeegvar = 0.000514139;
Pnonseizureeegvar = 0.999485861;
t = np.linspace(0 + 0.00416667, 0 + 0.00416667 + 0.00416667 * (len(Raw_variance_EEG) - 1), len(Raw_variance_EEG))
window_time_arr = t

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_RRIvars_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIvars_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m])
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
# for m in range(len(pro_RRIvars_time)):
#     P1=pro_RRIvars_time[m]*Pseizureeegvar*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_RRIvars_time)):
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
pyplot.title('Combined probability in QLD1282', fontsize=15)
pyplot.annotate('', xy=(6.6688889, np.max(Pcombined)), xytext=(6.6688889, np.max(Pcombined) + 0.00000000001),
                arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(18.9325, np.max(Pcombined)), xytext=(18.9325, np.max(Pcombined) + 0.00000000001),
                arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(24.342778, np.max(Pcombined)), xytext=(24.342778, np.max(Pcombined) + 0.00000000001),
                arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(29.681388, np.max(Pcombined)), xytext=(29.681388, np.max(Pcombined) + 0.00000000001),
                arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(43.735, np.max(Pcombined)), xytext=(43.735, np.max(Pcombined) + 0.00000000001),
                arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(44.365278, np.max(Pcombined)), xytext=(44.365278, np.max(Pcombined) + 0.00000000001),
                arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(49.186389, np.max(Pcombined)), xytext=(49.186389, np.max(Pcombined) + 0.00000000001),
                arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(54.781944, np.max(Pcombined)), xytext=(54.781944, np.max(Pcombined) + 0.00000000001),
                arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(69.106388, np.max(Pcombined)), xytext=(69.106388, np.max(Pcombined) + 0.00000000001),
                arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(73.85222, np.max(Pcombined)), xytext=(73.85222, np.max(Pcombined) + 0.00000000001),
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


# # t = np.linspace(0 + 0.00416667, 0 + 0.00416667 + 0.00416667 * (len(Raw_variance_EEG_arr) - 1),
# #                 len(Raw_variance_EEG_arr))
# # a = np.where(t < 12.1372222 + 0)
# # t[0:2912] = t[0:2912] - 0 + 11.8627778
# # t[2912:] = t[2912:] - 12.1372222 - 0
# # time_feature_arr = []
# # for i in range(len(t)):
# #     if t[i] > 24:
# #         time_feature_arr.append(t[i] - (t[i] // 24) * 24)
# #     else:
# #         time_feature_arr.append(t[i])
# # bins_number = 18
# # bins = np.linspace(0, 24, bins_number + 1)
# #
# # pro_circadian_time = []
# # pro_circadian_time_false = []
# # for i in range(len(time_feature_arr)):
# #     if time_feature_arr[i] >= bins[0] and time_feature_arr[i] <= bins[1]:
# #         pro_circadian_time_false.append(0.049382716)
# #         pro_circadian_time.append(0)
# #     elif time_feature_arr[i] > bins[1] and time_feature_arr[i] < bins[2]:
# #         pro_circadian_time_false.append(0.049382716)
# #         pro_circadian_time.append(0)
# #     elif time_feature_arr[i] >= bins[2] and time_feature_arr[i] < bins[3]:
# #         pro_circadian_time_false.append(0.049382716)
# #         pro_circadian_time.append(0)
# #     elif time_feature_arr[i] >= bins[3] and time_feature_arr[i] < bins[4]:
# #         pro_circadian_time_false.append(0.049382716)
# #         pro_circadian_time.append(0)
# #     elif time_feature_arr[i] >= bins[4] and time_feature_arr[i] < bins[5]:
# #         pro_circadian_time_false.append(0.049382716)
# #         pro_circadian_time.append(0)
# #     elif time_feature_arr[i] >= bins[5] and time_feature_arr[i] <= bins[6]:
# #         pro_circadian_time_false.append(0.049279835)
# #         pro_circadian_time.append(0.2)
# #     elif time_feature_arr[i] > bins[6] and time_feature_arr[i] < bins[7]:
# #         pro_circadian_time_false.append(0.049279835)
# #         pro_circadian_time.append(0.2)
# #     elif time_feature_arr[i] >= bins[7] and time_feature_arr[i] <= bins[8]:
# #         pro_circadian_time_false.append(0.051295752)
# #         pro_circadian_time.append(0)
# #     elif time_feature_arr[i] > bins[8] and time_feature_arr[i] < bins[9]:
# #         pro_circadian_time_false.append(0.049382716)
# #         pro_circadian_time.append(0)
# #     elif time_feature_arr[i] >= bins[9] and time_feature_arr[i] < bins[10]:
# #         pro_circadian_time_false.append(0.051295752)
# #         pro_circadian_time.append(0.2)
# #     elif time_feature_arr[i] >= bins[10] and time_feature_arr[i] < bins[11]:
# #         pro_circadian_time_false.append(0.065792181)
# #         pro_circadian_time.append(0.1)
# #     elif time_feature_arr[i] >= bins[11] and time_feature_arr[i] < bins[12]:
# #         pro_circadian_time_false.append(0.065843621)
# #         pro_circadian_time.append(0)
# #     elif time_feature_arr[i] >= bins[12] and time_feature_arr[i] < bins[13]:
# #         pro_circadian_time_false.append(0.065843621)
# #         pro_circadian_time.append(0)
# #     elif time_feature_arr[i] >= bins[13] and time_feature_arr[i] < bins[14]:
# #         pro_circadian_time_false.append(0.0656893)
# #         pro_circadian_time.append(0.3)
# #     elif time_feature_arr[i] >= bins[14] and time_feature_arr[i] < bins[15]:
# #         pro_circadian_time_false.append(0.065843621)
# #         pro_circadian_time.append(0)
# #     elif time_feature_arr[i] >= bins[15] and time_feature_arr[i] < bins[16]:
# #         pro_circadian_time_false.append(0.060596708)
# #         pro_circadian_time.append(0)
# #     elif time_feature_arr[i] >= bins[16] and time_feature_arr[i] < bins[17]:
# #         pro_circadian_time_false.append(0.049382716)
# #         pro_circadian_time.append(0)
# #     elif time_feature_arr[i] >= bins[17] and time_feature_arr[i] <= bins[18]:
# #         pro_circadian_time_false.append(0.049382716)
# #         pro_circadian_time.append(0)
# # print(pro_circadian_time[1599]);print(pro_circadian_time[4542]);print(pro_circadian_time[5841]);print(pro_circadian_time[7122]);
# # print(pro_circadian_time[10495]);print(pro_circadian_time[10646]);print(pro_circadian_time[11803]);print(pro_circadian_time[13146]);
# # print(pro_circadian_time[16584]);print(pro_circadian_time[17723]);
# #
# # # Pcombined=[]
# # # for m in range(len(pro_eegvars_time)):
# # #     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_RRIvars_time[m]*pro_circadian_time[m]
# # #     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIvars_time_false[m]*pro_circadian_time_false[m])
# # #     Pcombined.append(P1/(P1+P2))
# #
# # # Pcombined=[]
# # # for m in range(len(pro_eegvars_time)):
# # #     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_RRIautos_time[m]*pro_circadian_time[m]
# # #     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
# # #     Pcombined.append(P1/(P1+P2))
# #
# # # Pcombined=[]
# # # for m in range(len(pro_eegvars_time)):
# # #     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_circadian_time[m]
# # #     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_circadian_time_false[m])
# # #     Pcombined.append(P1/(P1+P2))
# #
# # # Pcombined=[]
# # # for m in range(len(pro_eegvars_time)):
# # #     P1=Pseizureeegvar*pro_eegvars_time[m]*pro_circadian_time[m]
# # #     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_circadian_time_false[m])
# # #     Pcombined.append(P1/(P1+P2))
# #
# # # Pcombined=[]
# # # for m in range(len(pro_eegvars_time)):
# # #     P1=Pseizureeegvar*pro_eegautos_time[m]*pro_circadian_time[m]
# # #     P2=Pnonseizureeegvar*(1-pro_eegautos_time_false[m]*pro_circadian_time_false[m])
# # #     Pcombined.append(P1/(P1+P2))
# #
# # # Pcombined=[]
# # # for m in range(len(pro_RRIvars_time)):
# # #     P1=pro_RRIvars_time[m]*Pseizureeegvar*pro_RRIautos_time[m]*pro_circadian_time[m]
# # #     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
# # #     Pcombined.append(P1/(P1+P2))
# #
# # # Pcombined=[]
# # # for m in range(len(pro_RRIvars_time)):
# # #     P1=Pseizureeegvar*pro_RRIvars_time[m]*pro_circadian_time[m]
# # #     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m]*pro_circadian_time_false[m])
# # #     Pcombined.append(P1/(P1+P2))
# #
# # # Pcombined=[]
# # # for m in range(len(pro_RRIautos_time)):
# # #     P1=Pseizureeegvar*pro_RRIautos_time[m]*pro_circadian_time[m]
# # #     P2=Pnonseizureeegvar*(1-pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
# # #     Pcombined.append(P1/(P1+P2))
# #
# # # t = np.linspace(0 + 0.00416667, 0 + 0.00416667 + 0.00416667 * (len(Raw_variance_EEG) - 1), len(Raw_variance_EEG))
# # # window_time_arr = t
# # # pyplot.figure(figsize=(12, 5))
# # # pyplot.plot(window_time_arr, Pcombined)
# # # pyplot.title('combined probability in QLD1282', fontsize=15)
# # # pyplot.annotate('', xy=(6.6688889, np.max(Pcombined)), xytext=(6.6688889, np.max(Pcombined) + 0.00000000001),
# # #                 arrowprops=dict(facecolor='black', shrink=0.05))
# # # pyplot.annotate('', xy=(18.9325, np.max(Pcombined)), xytext=(18.9325, np.max(Pcombined) + 0.00000000001),
# # #                 arrowprops=dict(facecolor='black', shrink=0.05))
# # # pyplot.annotate('', xy=(24.342778, np.max(Pcombined)), xytext=(24.342778, np.max(Pcombined) + 0.00000000001),
# # #                 arrowprops=dict(facecolor='black', shrink=0.05))
# # # pyplot.annotate('', xy=(29.681388, np.max(Pcombined)), xytext=(29.681388, np.max(Pcombined) + 0.00000000001),
# # #                 arrowprops=dict(facecolor='black', shrink=0.05))
# # # pyplot.annotate('', xy=(43.735, np.max(Pcombined)), xytext=(43.735, np.max(Pcombined) + 0.00000000001),
# # #                 arrowprops=dict(facecolor='black', shrink=0.05))
# # # pyplot.annotate('', xy=(44.365278, np.max(Pcombined)), xytext=(44.365278, np.max(Pcombined) + 0.00000000001),
# # #                 arrowprops=dict(facecolor='black', shrink=0.05))
# # # pyplot.annotate('', xy=(49.186389, np.max(Pcombined)), xytext=(49.186389, np.max(Pcombined) + 0.00000000001),
# # #                 arrowprops=dict(facecolor='black', shrink=0.05))
# # # pyplot.annotate('', xy=(54.781944, np.max(Pcombined)), xytext=(54.781944, np.max(Pcombined) + 0.00000000001),
# # #                 arrowprops=dict(facecolor='black', shrink=0.05))
# # # pyplot.annotate('', xy=(69.106388, np.max(Pcombined)), xytext=(69.106388, np.max(Pcombined) + 0.00000000001),
# # #                 arrowprops=dict(facecolor='black', shrink=0.05))
# # # pyplot.annotate('', xy=(73.85222, np.max(Pcombined)), xytext=(73.85222, np.max(Pcombined) + 0.00000000001),
# # #                 arrowprops=dict(facecolor='black', shrink=0.05))
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





# # ## section 3 froecast
# t = np.linspace(0, 0 + 0.00416667 * (len(Raw_variance_EEG_arr) - 1), len(Raw_variance_EEG_arr))
# t_window_arr = t
# fore_arr_EEGvars = []
# for k in range(81, 82):
#     variance_arr = Raw_variance_EEG_arr[0:(19450 + 240 * k)]
#     long_rhythm_var_arr = movingaverage(variance_arr, 240*6)
#     pyplot.figure(figsize=(6, 3))
#     pyplot.title('EEG variance')
#     pyplot.ylabel('Voltage ($\mathregular{v^2}$)')
#     pyplot.xlabel('Time(h)')
#     # pyplot.plot(t_window_arr[240 * 24:(19450 + 240 * k)], long_rhythm_var_arr[240 * 24:], 'orange')
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD1282/cycles6h_Cz_forecast81hsignal_3hcycle_EEGvar_QLD1282.csv',
#                          sep=',', header=None)
# forecast_var_EEG = csv_reader.values
# forecast_var_EEG_arr = []
# for item in forecast_var_EEG:
#     forecast_var_EEG_arr = forecast_var_EEG_arr + list(item)
# t = np.linspace(t_window_arr[19450], t_window_arr[19450] + 0.1666667 * (len(forecast_var_EEG_arr) - 1),
#                 len(forecast_var_EEG_arr))
# pyplot.plot(t, forecast_var_EEG_arr, 'k', label='forecast EEG var')
# pyplot.legend()
# pyplot.show()
#
# fore_arr_EEGauto = []
# for k in range(81, 82):
#     auto_arr = Raw_auto_EEG_arr[0:(19450 + 240 * k)]
#     long_rhythm_auto_arr = movingaverage(auto_arr, 240*6)
#     pyplot.figure(figsize=(6, 3))
#     pyplot.title('EEG autocorrelation')
#     pyplot.xlabel('time(h)')
#     pyplot.plot(t_window_arr[240 * 24:(19450 + 240 * k)], long_rhythm_auto_arr[240 * 24:], 'orange')
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD1282/cycles6h_Cz_forecast81hsignal_3hcycle_EEGauto_QLD1282.csv', sep=',',
#     header=None)
# forecast_auto_EEG = csv_reader.values
# forecast_auto_EEG_arr = []
# for item in forecast_auto_EEG:
#     forecast_auto_EEG_arr = forecast_auto_EEG_arr + list(item)
#
# t = np.linspace(t_window_arr[19450], t_window_arr[19450] + 0.1666667 * (len(forecast_auto_EEG_arr) - 1),
#                 len(forecast_auto_EEG_arr))
# pyplot.plot(t, forecast_auto_EEG_arr, 'k', label='forecast EEG auto')
# pyplot.legend()
# pyplot.show()

fore_arr_RRIvars = []
for k in range(81, 82):
    variance_arr = Raw_variance_RRI31_arr[0:(19450 + 240 * k)]
    long_rhythm_var_arr = movingaverage(variance_arr, 20*1)
    pyplot.figure(figsize=(6, 3))
    pyplot.title('RRI variance')
    pyplot.ylabel('Second ($\mathregular{s^2}$)')
    pyplot.xlabel('Time(h)')
    pyplot.plot(t_window_arr[240 * 24:(19450 + 240 * k)], long_rhythm_var_arr[240 * 24:], 'orange')
csv_reader = pd.read_csv( 'C:/Users/wxiong/Documents/PHD/2011.1/QLD1282/cycles5min_ch31_forecast81hsignal_3hcycle_RRIvar_QLD1282.csv', sep=',',header=None)

forecast_var_RRI31 = csv_reader.values
forecast_var_RRI31_arr = []
for item in forecast_var_RRI31:
    forecast_var_RRI31_arr = forecast_var_RRI31_arr + list(item)
t = np.linspace(t_window_arr[19450], t_window_arr[19450] + 0.1666667 * (len(forecast_var_RRI31_arr) - 1),
                len(forecast_var_RRI31_arr))
pyplot.plot(t, forecast_var_RRI31_arr, 'k', label='forecast RRI var')
pyplot.legend()
pyplot.show()

fore_arr_RRIautos = []
save_data_RRIautos = []
for k in range(81, 82):
    auto_arr = Raw_auto_RRI31_arr[0:19450 + 240 * k]
    long_rhythm_auto_arr = movingaverage(auto_arr, 20*1)
    pyplot.figure(figsize=(6, 3))
    pyplot.title('RRI autocorrelation')
    pyplot.xlabel('Time(h)')
    pyplot.plot(t_window_arr[240 * 24:19450 + 240 * k], long_rhythm_auto_arr[240 * 24:], 'orange')
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD1282/cycles5min_ch31_forecast81hsignal_3hcycle_RRIauto_QLD1282.csv', sep=',',
    header=None)

forecast_auto_RRI31 = csv_reader.values
forecast_auto_RRI31_arr = []
for item in forecast_auto_RRI31:
    forecast_auto_RRI31_arr = forecast_auto_RRI31_arr + list(item)
t = np.linspace(t_window_arr[19450], t_window_arr[19450] + 0.1666667 * (len(forecast_auto_RRI31_arr) - 1),
                len(forecast_auto_RRI31_arr))
pyplot.plot(t, forecast_auto_RRI31_arr, 'k', label='forecast RRI auto')
pyplot.legend()
pyplot.show()
# print(len(forecast_var_EEG_arr));print(len(forecast_auto_EEG_arr));
# print(len(forecast_var_RRI31_arr));
# print(len(forecast_auto_RRI31_arr));





# ### predict, forecast data
# var_trans = hilbert(forecast_var_EEG_arr)
# var_phase = np.angle(var_trans)
# rolmean_short_EEGvar = var_phase
# var_trans = hilbert(forecast_auto_EEG_arr)
# var_phase = np.angle(var_trans)
# rolmean_short_EEGauto = var_phase
var_trans = hilbert(forecast_var_RRI31_arr)
var_phase = np.angle(var_trans)
rolmean_short_RRIvar = var_phase
var_trans = hilbert(forecast_auto_RRI31_arr)
var_phase = np.angle(var_trans)
rolmean_short_RRIauto = var_phase



# #### combined probability calculation
#### 24h 24h 24h
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
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] > bins[7] and rolmean_short_EEGvar[i] < bins[8]:
#         pro_eegvars_time_false.append(0.098868313)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[8] and rolmean_short_EEGvar[i] <= bins[9]:
#         pro_eegvars_time_false.append(0.624125514)
#         pro_eegvars_time.append(0.7)
#     elif rolmean_short_EEGvar[i] > bins[9] and rolmean_short_EEGvar[i] < bins[10]:
#         pro_eegvars_time_false.append(0.150977366)
#         pro_eegvars_time.append(0.3)
#     elif rolmean_short_EEGvar[i] >= bins[10] and rolmean_short_EEGvar[i] <= bins[11]:
#         pro_eegvars_time_false.append(0.053909465)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] > bins[11] and rolmean_short_EEGvar[i] < bins[12]:
#         pro_eegvars_time_false.append(0.022685185)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] > bins[12] and rolmean_short_EEGvar[i] < bins[13]:
#         pro_eegvars_time_false.append(0.024382716)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[13] and rolmean_short_EEGvar[i] < bins[14]:
#         pro_eegvars_time_false.append(0.02505144)
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
#         pro_eegautos_time_false.append(0.170781893)
#         pro_eegautos_time.append(0.1)
#     elif rolmean_short_EEGauto[i] >= bins[8] and rolmean_short_EEGauto[i] < bins[9]:
#         pro_eegautos_time_false.append(0.457253086)
#         pro_eegautos_time.append(0.7)
#     elif rolmean_short_EEGauto[i] >= bins[9] and rolmean_short_EEGauto[i] <= bins[10]:
#         pro_eegautos_time_false.append(0.253858025)
#         pro_eegautos_time.append(0.2)
#     elif rolmean_short_EEGauto[i] > bins[10] and rolmean_short_EEGauto[i] < bins[11]:
#         pro_eegautos_time_false.append(0.059053498)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[11] and rolmean_short_EEGauto[i] < bins[12]:
#         pro_eegautos_time_false.append(0.020010288)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[12] and rolmean_short_EEGauto[i] < bins[13]:
#         pro_eegautos_time_false.append(0.019444444)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[13] and rolmean_short_EEGauto[i] < bins[14]:
#         pro_eegautos_time_false.append(0.019598765)
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
#         pro_RRIautos_time_false.append(0.184516461)
#         pro_RRIautos_time.append(0.1)
#     elif rolmean_short_RRIauto[i] > bins[8] and rolmean_short_RRIauto[i] <= bins[9]:
#         pro_RRIautos_time_false.append(0.416820988)
#         pro_RRIautos_time.append(0.7)
#     elif rolmean_short_RRIauto[i] > bins[9] and rolmean_short_RRIauto[i] <= bins[10]:
#         pro_RRIautos_time_false.append(0.28117284)
#         pro_RRIautos_time.append(0.2)
#     elif rolmean_short_RRIauto[i] > bins[10] and rolmean_short_RRIauto[i] < bins[11]:
#         pro_RRIautos_time_false.append(0.050462963)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[11] and rolmean_short_RRIauto[i] < bins[12]:
#         pro_RRIautos_time_false.append(0.017849794)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[12] and rolmean_short_RRIauto[i] < bins[13]:
#         pro_RRIautos_time_false.append(0.019187243)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[13] and rolmean_short_RRIauto[i] < bins[14]:
#         pro_RRIautos_time_false.append(0.029989712)
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
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] > bins[7] and rolmean_short_EEGvar[i] < bins[8]:
#         pro_eegvars_time_false.append(0.000205761)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[8] and rolmean_short_EEGvar[i] <= bins[9]:
#         pro_eegvars_time_false.append(0.634876543)
#         pro_eegvars_time.append(0.6)
#     elif rolmean_short_EEGvar[i] > bins[9] and rolmean_short_EEGvar[i] < bins[10]:
#         pro_eegvars_time_false.append(0.281790123)
#         pro_eegvars_time.append(0.4)
#     elif rolmean_short_EEGvar[i] >= bins[10] and rolmean_short_EEGvar[i] <= bins[11]:
#         pro_eegvars_time_false.append(0.039197531)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] > bins[11] and rolmean_short_EEGvar[i] < bins[12]:
#         pro_eegvars_time_false.append(0.017592593)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] > bins[12] and rolmean_short_EEGvar[i] < bins[13]:
#         pro_eegvars_time_false.append(0.011162551)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[13] and rolmean_short_EEGvar[i] < bins[14]:
#         pro_eegvars_time_false.append(0.015174897)
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
#         pro_eegautos_time_false.append(0.082407407)
#         pro_eegautos_time.append(0.1)
#     elif rolmean_short_EEGauto[i] >= bins[8] and rolmean_short_EEGauto[i] < bins[9]:
#         pro_eegautos_time_false.append(0.478858025)
#         pro_eegautos_time.append(0.5)
#     elif rolmean_short_EEGauto[i] >= bins[9] and rolmean_short_EEGauto[i] <= bins[10]:
#         pro_eegautos_time_false.append(0.375565844)
#         pro_eegautos_time.append(0.4)
#     elif rolmean_short_EEGauto[i] > bins[10] and rolmean_short_EEGauto[i] < bins[11]:
#         pro_eegautos_time_false.append(0.034876543)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[11] and rolmean_short_EEGauto[i] < bins[12]:
#         pro_eegautos_time_false.append(0.010493827)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[12] and rolmean_short_EEGauto[i] < bins[13]:
#         pro_eegautos_time_false.append(0.0093107)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[13] and rolmean_short_EEGauto[i] < bins[14]:
#         pro_eegautos_time_false.append(0.008487654)
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
#         pro_RRIvars_time_false.append(0.07659465)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[7] and rolmean_short_RRIvar[i] <= bins[8]:
#         pro_RRIvars_time_false.append(0.249279835)
#         pro_RRIvars_time.append(0.5)
#     elif rolmean_short_RRIvar[i] > bins[8] and rolmean_short_RRIvar[i] < bins[9]:
#         pro_RRIvars_time_false.append(0.210442387)
#         pro_RRIvars_time.append(0.1)
#     elif rolmean_short_RRIvar[i] >= bins[9] and rolmean_short_RRIvar[i] <= bins[10]:
#         pro_RRIvars_time_false.append(0.174382716)
#         pro_RRIvars_time.append(0.1)
#     elif rolmean_short_RRIvar[i] > bins[10] and rolmean_short_RRIvar[i] <= bins[11]:
#         pro_RRIvars_time_false.append(0.169804527)
#         pro_RRIvars_time.append(0.2)
#     elif rolmean_short_RRIvar[i] > bins[11] and rolmean_short_RRIvar[i] < bins[12]:
#         pro_RRIvars_time_false.append(0.098919753)
#         pro_RRIvars_time.append(0.1)
#     elif rolmean_short_RRIvar[i] >= bins[12] and rolmean_short_RRIvar[i] < bins[13]:
#         pro_RRIvars_time_false.append(0.017541152)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[13] and rolmean_short_RRIvar[i] < bins[14]:
#         pro_RRIvars_time_false.append(0.003034979)
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



# ##### 6h 6h 6h
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
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] > bins[7] and rolmean_short_EEGvar[i] < bins[8]:
#         pro_eegvars_time_false.append(0.000617284)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[8] and rolmean_short_EEGvar[i] <= bins[9]:
#         pro_eegvars_time_false.append(0.574537037)
#         pro_eegvars_time.append(0.5)
#     elif rolmean_short_EEGvar[i] > bins[9] and rolmean_short_EEGvar[i] < bins[10]:
#         pro_eegvars_time_false.append(0.361111111)
#         pro_eegvars_time.append(0.5)
#     elif rolmean_short_EEGvar[i] >= bins[10] and rolmean_short_EEGvar[i] <= bins[11]:
#         pro_eegvars_time_false.append(0.042283951)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] > bins[11] and rolmean_short_EEGvar[i] < bins[12]:
#         pro_eegvars_time_false.append(0.007150206)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] > bins[12] and rolmean_short_EEGvar[i] < bins[13]:
#         pro_eegvars_time_false.append(0.007047325)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[13] and rolmean_short_EEGvar[i] < bins[14]:
#         pro_eegvars_time_false.append(0.007253086)
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
#         pro_eegautos_time_false.append(0.030092593)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[8] and rolmean_short_EEGauto[i] < bins[9]:
#         pro_eegautos_time_false.append(0.46882716)
#         pro_eegautos_time.append(0.5)
#     elif rolmean_short_EEGauto[i] >= bins[9] and rolmean_short_EEGauto[i] <= bins[10]:
#         pro_eegautos_time_false.append(0.474331276)
#         pro_eegautos_time.append(0.5)
#     elif rolmean_short_EEGauto[i] > bins[10] and rolmean_short_EEGauto[i] < bins[11]:
#         pro_eegautos_time_false.append(0.014300412)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[11] and rolmean_short_EEGauto[i] < bins[12]:
#         pro_eegautos_time_false.append(0.004835391)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[12] and rolmean_short_EEGauto[i] < bins[13]:
#         pro_eegautos_time_false.append(0.003549383)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[13] and rolmean_short_EEGauto[i] < bins[14]:
#         pro_eegautos_time_false.append(0.004063786)
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
#         pro_RRIvars_time_false.append(0.025)
#         pro_RRIvars_time.append(0.1)
#     elif rolmean_short_RRIvar[i] >= bins[6] and rolmean_short_RRIvar[i] < bins[7]:
#         pro_RRIvars_time_false.append(0.172788066)
#         pro_RRIvars_time.append(0.2)
#     elif rolmean_short_RRIvar[i] >= bins[7] and rolmean_short_RRIvar[i] <= bins[8]:
#         pro_RRIvars_time_false.append(0.22340535)
#         pro_RRIvars_time.append(0.4)
#     elif rolmean_short_RRIvar[i] > bins[8] and rolmean_short_RRIvar[i] < bins[9]:
#         pro_RRIvars_time_false.append(0.178240741)
#         pro_RRIvars_time.append(0.2)
#     elif rolmean_short_RRIvar[i] >= bins[9] and rolmean_short_RRIvar[i] <= bins[10]:
#         pro_RRIvars_time_false.append(0.084259259)
#         pro_RRIvars_time.append(0.1)
#     elif rolmean_short_RRIvar[i] > bins[10] and rolmean_short_RRIvar[i] <= bins[11]:
#         pro_RRIvars_time_false.append(0.072942387)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] > bins[11] and rolmean_short_RRIvar[i] < bins[12]:
#         pro_RRIvars_time_false.append(0.10622428)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[12] and rolmean_short_RRIvar[i] < bins[13]:
#         pro_RRIvars_time_false.append(0.123559671)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[13] and rolmean_short_RRIvar[i] < bins[14]:
#         pro_RRIvars_time_false.append(0.013580247)
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
# #         pro_RRIautos_time_false.append(0)
# #         pro_RRIautos_time.append(0)
# #     elif rolmean_short_RRIauto[i] >= bins[7] and rolmean_short_RRIauto[i] <= bins[8]:
# #         pro_RRIautos_time_false.append(0.169393004)
# #         pro_RRIautos_time.append(0.1)
# #     elif rolmean_short_RRIauto[i] > bins[8] and rolmean_short_RRIauto[i] <= bins[9]:
# #         pro_RRIautos_time_false.append(0.25617284)
# #         pro_RRIautos_time.append(0.2)
# #     elif rolmean_short_RRIauto[i] > bins[9] and rolmean_short_RRIauto[i] <= bins[10]:
# #         pro_RRIautos_time_false.append(0.540226337)
# #         pro_RRIautos_time.append(0.7)
# #     elif rolmean_short_RRIauto[i] > bins[10] and rolmean_short_RRIauto[i] < bins[11]:
# #         pro_RRIautos_time_false.append(0.012602881)
# #         pro_RRIautos_time.append(0)
# #     elif rolmean_short_RRIauto[i] >= bins[11] and rolmean_short_RRIauto[i] < bins[12]:
# #         pro_RRIautos_time_false.append(0.005401235)
# #         pro_RRIautos_time.append(0)
# #     elif rolmean_short_RRIauto[i] >= bins[12] and rolmean_short_RRIauto[i] < bins[13]:
# #         pro_RRIautos_time_false.append(0.006687243)
# #         pro_RRIautos_time.append(0)
# #     elif rolmean_short_RRIauto[i] >= bins[13] and rolmean_short_RRIauto[i] < bins[14]:
# #         pro_RRIautos_time_false.append(0.009516461)
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



#### 1h 1h 1h 1h
# bins_number = 18
# bins = np.linspace(-np.pi, np.pi, bins_number + 1)
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
#         pro_RRIvars_time_false.append(0.004475309)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[5] and rolmean_short_RRIvar[i] < bins[6]:
#         pro_RRIvars_time_false.append(0.094495885)
#         pro_RRIvars_time.append(0.1)
#     elif rolmean_short_RRIvar[i] >= bins[6] and rolmean_short_RRIvar[i] < bins[7]:
#         pro_RRIvars_time_false.append(0.149176955)
#         pro_RRIvars_time.append(0.2)
#     elif rolmean_short_RRIvar[i] >= bins[7] and rolmean_short_RRIvar[i] <= bins[8]:
#         pro_RRIvars_time_false.append(0.191152263)
#         pro_RRIvars_time.append(0.2)
#     elif rolmean_short_RRIvar[i] > bins[8] and rolmean_short_RRIvar[i] < bins[9]:
#         pro_RRIvars_time_false.append(0.146296296)
#         pro_RRIvars_time.append(0.1)
#     elif rolmean_short_RRIvar[i] >= bins[9] and rolmean_short_RRIvar[i] <= bins[10]:
#         pro_RRIvars_time_false.append(0.072067901)
#         pro_RRIvars_time.append(0.2)
#     elif rolmean_short_RRIvar[i] > bins[10] and rolmean_short_RRIvar[i] <= bins[11]:
#         pro_RRIvars_time_false.append(0.077417695)
#         pro_RRIvars_time.append(0.2)
#     elif rolmean_short_RRIvar[i] > bins[11] and rolmean_short_RRIvar[i] < bins[12]:
#         pro_RRIvars_time_false.append(0.066820988)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[12] and rolmean_short_RRIvar[i] < bins[13]:
#         pro_RRIvars_time_false.append(0.104526749)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[13] and rolmean_short_RRIvar[i] < bins[14]:
#         pro_RRIvars_time_false.append(0.093569959)
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



### 5min 5min 5min 5min
bins_number = 18
bins = np.linspace(-np.pi, np.pi, bins_number + 1)
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
        pro_RRIvars_time_false.append(0.046347737)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] >= bins[5] and rolmean_short_RRIvar[i] < bins[6]:
        pro_RRIvars_time_false.append(0.15473251)
        pro_RRIvars_time.append(0.1)
    elif rolmean_short_RRIvar[i] >= bins[6] and rolmean_short_RRIvar[i] < bins[7]:
        pro_RRIvars_time_false.append(0.128960905)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] >= bins[7] and rolmean_short_RRIvar[i] <= bins[8]:
        pro_RRIvars_time_false.append(0.1093107)
        pro_RRIvars_time.append(0.4)
    elif rolmean_short_RRIvar[i] > bins[8] and rolmean_short_RRIvar[i] < bins[9]:
        pro_RRIvars_time_false.append(0.095679012)
        pro_RRIvars_time.append(0.1)
    elif rolmean_short_RRIvar[i] >= bins[9] and rolmean_short_RRIvar[i] <= bins[10]:
        pro_RRIvars_time_false.append(0.088425926)
        pro_RRIvars_time.append(0.2)
    elif rolmean_short_RRIvar[i] > bins[10] and rolmean_short_RRIvar[i] <= bins[11]:
        pro_RRIvars_time_false.append(0.063065844)
        pro_RRIvars_time.append(0.1)
    elif rolmean_short_RRIvar[i] > bins[11] and rolmean_short_RRIvar[i] < bins[12]:
        pro_RRIvars_time_false.append(0.066100823)
        pro_RRIvars_time.append(0.1)
    elif rolmean_short_RRIvar[i] >= bins[12] and rolmean_short_RRIvar[i] < bins[13]:
        pro_RRIvars_time_false.append(0.107304527)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] >= bins[13] and rolmean_short_RRIvar[i] < bins[14]:
        pro_RRIvars_time_false.append(0.140072016)
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
        pro_RRIautos_time_false.append(0.011574074)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[6] and rolmean_short_RRIauto[i] < bins[7]:
        pro_RRIautos_time_false.append(0.048868313)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[7] and rolmean_short_RRIauto[i] <= bins[8]:
        pro_RRIautos_time_false.append(0.11867284)
        pro_RRIautos_time.append(0.2)
    elif rolmean_short_RRIauto[i] > bins[8] and rolmean_short_RRIauto[i] <= bins[9]:
        pro_RRIautos_time_false.append(0.305452675)
        pro_RRIautos_time.append(0.4)
    elif rolmean_short_RRIauto[i] > bins[9] and rolmean_short_RRIauto[i] <= bins[10]:
        pro_RRIautos_time_false.append(0.324434156)
        pro_RRIautos_time.append(0.3)
    elif rolmean_short_RRIauto[i] > bins[10] and rolmean_short_RRIauto[i] < bins[11]:
        pro_RRIautos_time_false.append(0.154115226)
        pro_RRIautos_time.append(0.1)
    elif rolmean_short_RRIauto[i] >= bins[11] and rolmean_short_RRIauto[i] < bins[12]:
        pro_RRIautos_time_false.append(0.034722222)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[12] and rolmean_short_RRIauto[i] < bins[13]:
        pro_RRIautos_time_false.append(0.001903292)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[13] and rolmean_short_RRIauto[i] < bins[14]:
        pro_RRIautos_time_false.append(0.000257202)
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





Pseizureeegvar = 0.000514139;
Pnonseizureeegvar = 0.999485861;

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_RRIvars_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIvars_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m])
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
# for m in range(len(pro_RRIvars_time)):
#     P1=pro_RRIvars_time[m]*Pseizureeegvar*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))


# Pcombined = []
# for m in range(len(pro_RRIvars_time)):
#     P1=Pseizureeegvar*pro_RRIvars_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m])
#     Pcombined.append(P1/(P1+P2))

Pcombined = []
for m in range(len(pro_RRIautos_time)):
    P1=Pseizureeegvar*pro_RRIautos_time[m]
    P2=Pnonseizureeegvar*(1-pro_RRIautos_time_false[m])
    Pcombined.append(P1/(P1+P2))


pyplot.figure(figsize=(8, 4))
RRI_timewindow_arr = t
pyplot.plot(RRI_timewindow_arr, Pcombined)
pyplot.annotate('', xy=(95.493611, np.max(Pcombined)), xytext=(95.493611, np.max(Pcombined) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
pyplot.annotate('', xy=(97.3675, np.max(Pcombined)), xytext=(97.3675, np.max(Pcombined) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
pyplot.annotate('', xy=(99.079444, np.max(Pcombined)), xytext=(99.079444, np.max(Pcombined) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
pyplot.annotate('', xy=(102.43722, np.max(Pcombined)), xytext=(102.43722, np.max(Pcombined) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
pyplot.annotate('', xy=(120.36472, np.max(Pcombined)), xytext=(120.36472, np.max(Pcombined) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
pyplot.annotate('', xy=(124.15888, np.max(Pcombined)), xytext=(124.15888, np.max(Pcombined) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
pyplot.annotate('', xy=(139.29889, np.max(Pcombined)), xytext=(139.29889, np.max(Pcombined) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
pyplot.annotate('', xy=(145.50861, np.max(Pcombined)), xytext=(145.50861, np.max(Pcombined) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
pyplot.annotate('', xy=(149.65861, np.max(Pcombined)), xytext=(149.65861, np.max(Pcombined) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
pyplot.hlines(Th1, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'r')
pyplot.title('Forecast seizures in QLD1282')
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
time_arr=[95.493611,97.3675,99.079444,102.43722,120.36472,124.15888,139.29889,145.50861,149.65861]
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
time_arr=[95.493611,97.3675,99.079444,102.43722,120.36472,124.15888,139.29889,145.50861,149.65861]
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
time_arr=[95.493611,97.3675,99.079444,102.43722,120.36472,124.15888,139.29889,145.50861,149.65861]
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
time_arr=[95.493611,97.3675,99.079444,102.43722,120.36472,124.15888,139.29889,145.50861,149.65861]
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
time_arr=[95.493611,97.3675,99.079444,102.43722,120.36472,124.15888,139.29889,145.50861,149.65861]
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


# Pcombined = split(Pcombined_X, 6)
# print(len(Pcombined))
# time_arr_arr=[]
# AUC_cs_arr=[]
# for i in range(50000):
#     time_arr = np.random.uniform(low=t_window_arr[19450], high=t_window_arr[-1], size=9)
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
#     AUC_cs = auc(np.sort(FPR_arr_CS), np.sort(Sen_arr_CS))
#     # print(AUC_cs)
#     AUC_cs_arr.append(AUC_cs)
#
# # print(AUC_cs_arr)
# # print(time_arr_arr)
# np.savetxt("C:/Users/wxiong/Documents/PHD/2011.1/QLD1282/chance/AUC_ECGauto_5min_QLD1282_2022.csv", AUC_cs_arr, delimiter=",", fmt='%s')




t1 = np.linspace(0 + 0.00416667, 0 + 0.00416667 + 0.00416667 * (len(Raw_variance_EEG_arr) - 1),
                 len(Raw_variance_EEG_arr))
a = np.where(t1 < 12.1372222 + 0)
t1[0:2912] = t1[0:2912] - 0 + 11.8627778
t1[2912:] = t1[2912:] - 12.1372222 - 0
time_feature_arr = []
for i in range(len(t1)):
    if t1[i] > 24:
        time_feature_arr.append(t1[i] - (t1[i] // 24) * 24)
    else:
        time_feature_arr.append(t1[i])
print(len(time_feature_arr))
time_arr = time_feature_arr[19450:]
print(len(time_arr))
new_arr = []
for j in range(0, 486):
    new_arr.append(time_arr[40 * j])


bins_number = 18
bins = np.linspace(0, 24, bins_number + 1)
pro_circadian_time = []
pro_circadian_time_false = []
for i in range(len(new_arr)):
    if new_arr[i] >= bins[0] and new_arr[i] <= bins[1]:
        pro_circadian_time_false.append(0.049382716)
        pro_circadian_time.append(0)
    elif new_arr[i] > bins[1] and new_arr[i] < bins[2]:
        pro_circadian_time_false.append(0.049382716)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[2] and new_arr[i] < bins[3]:
        pro_circadian_time_false.append(0.049382716)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[3] and new_arr[i] < bins[4]:
        pro_circadian_time_false.append(0.049382716)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[4] and new_arr[i] < bins[5]:
        pro_circadian_time_false.append(0.049382716)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[5] and new_arr[i] <= bins[6]:
        pro_circadian_time_false.append(0.049279835)
        pro_circadian_time.append(0.2)
    elif new_arr[i] > bins[6] and new_arr[i] < bins[7]:
        pro_circadian_time_false.append(0.049279835)
        pro_circadian_time.append(0.2)
    elif new_arr[i] >= bins[7] and new_arr[i] <= bins[8]:
        pro_circadian_time_false.append(0.051295752)
        pro_circadian_time.append(0)
    elif new_arr[i] > bins[8] and new_arr[i] < bins[9]:
        pro_circadian_time_false.append(0.049382716)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[9] and new_arr[i] < bins[10]:
        pro_circadian_time_false.append(0.051295752)
        pro_circadian_time.append(0.2)
    elif new_arr[i] >= bins[10] and new_arr[i] < bins[11]:
        pro_circadian_time_false.append(0.065792181)
        pro_circadian_time.append(0.1)
    elif new_arr[i] >= bins[11] and new_arr[i] < bins[12]:
        pro_circadian_time_false.append(0.065843621)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[12] and new_arr[i] < bins[13]:
        pro_circadian_time_false.append(0.065843621)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[13] and new_arr[i] < bins[14]:
        pro_circadian_time_false.append(0.0656893)
        pro_circadian_time.append(0.3)
    elif new_arr[i] >= bins[14] and new_arr[i] < bins[15]:
        pro_circadian_time_false.append(0.065843621)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[15] and new_arr[i] < bins[16]:
        pro_circadian_time_false.append(0.060596708)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[16] and new_arr[i] < bins[17]:
        pro_circadian_time_false.append(0.049382716)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[17] and new_arr[i] <= bins[18]:
        pro_circadian_time_false.append(0.049382716)
        pro_circadian_time.append(0)


# # RRI_timewindow_arr = t[0:len(pro_circadian_time)]
# # print(RRI_timewindow_arr[-1] - RRI_timewindow_arr[0])
# # pyplot.figure(figsize=(8, 4))
# # pyplot.plot(RRI_timewindow_arr, pro_circadian_time)
# # pyplot.annotate('', xy=(95.493611, np.max(pro_circadian_time)), xytext=(95.493611, np.max(pro_circadian_time) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
# # pyplot.annotate('', xy=(97.3675, np.max(pro_circadian_time)), xytext=(97.3675, np.max(pro_circadian_time) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
# # pyplot.annotate('', xy=(99.079444, np.max(pro_circadian_time)), xytext=(99.079444, np.max(pro_circadian_time) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
# # pyplot.annotate('', xy=(102.43722, np.max(pro_circadian_time)), xytext=(102.43722, np.max(pro_circadian_time) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
# # pyplot.annotate('', xy=(120.36472, np.max(pro_circadian_time)), xytext=(120.36472, np.max(pro_circadian_time) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
# # pyplot.annotate('', xy=(124.15888, np.max(pro_circadian_time)), xytext=(124.15888, np.max(pro_circadian_time) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
# # pyplot.annotate('', xy=(139.29889, np.max(pro_circadian_time)), xytext=(139.29889, np.max(pro_circadian_time) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
# # # pyplot.hlines(0.1, RRI_timewindow_arr[0], RRI_timewindow_arr[-1], 'r')
# # pyplot.title('Forecast seizures in QLD1282')
# # pyplot.xlabel('Time(h)')
# # pyplot.ylabel('Seizure probability')
# # pyplot.show()


Pcombined=split(pro_circadian_time,6)
print(len(Pcombined))
index=[]
for i in range(len(Pcombined)):
    for item in Pcombined[i]:
        if item >= 0.1:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[95.493611,97.3675,99.079444,102.43722,120.36472,124.15888,139.29889,145.50861,149.65861]
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
        if item >= 0.3*0.1:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[95.493611,97.3675,99.079444,102.43722,120.36472,124.15888,139.29889,145.50861,149.65861]
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
        if item >= 0.6*0.1:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[95.493611,97.3675,99.079444,102.43722,120.36472,124.15888,139.29889,145.50861,149.65861]
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
        if item >= 1.2*0.1:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[95.493611,97.3675,99.079444,102.43722,120.36472,124.15888,139.29889,145.50861,149.65861]
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
        if item >= 2*0.1:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[95.493611,97.3675,99.079444,102.43722,120.36472,124.15888,139.29889,145.50861,149.65861]
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


# Pcombined = split(pro_circadian_time, 6)
# print(len(Pcombined))
# time_arr_arr=[]
# AUC_cs_arr=[]
# for i in range(50000):
#     time_arr = np.random.uniform(low=t_window_arr[19450], high=t_window_arr[-1], size=9)
#     time_arr_arr.append(time_arr)
#     time_arr=np.sort(time_arr)
#
#     index = []
#     for i in range(len(Pcombined)):
#         for item in Pcombined[i]:
#             if item >= 0.1:
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
#             if item >= 0.3 * 0.1:
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
#             if item >= 0.6 * 0.1:
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
#             if item >= 1.2 * 0.1:
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
#             if item >= 2 * 0.1:
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
# np.savetxt("C:/Users/wxiong/Documents/PHD/2011.1/QLD1282/chance/AUC_circadian_QLD1282.csv", AUC_cs_arr, delimiter=",", fmt='%s')




#
#
# Pseizureeegvar = 0.000514139;
# Pnonseizureeegvar = 0.999485861;
#
# # Pcombined = []
# # for m in range(len(pro_circadian_time)):
# #     P1 = Pseizureeegvar * pro_eegvars_time[m] * pro_eegautos_time[m]*pro_RRIvars_time[m] * pro_circadian_time[m]
# #     P2 = Pnonseizureeegvar * (1 - pro_eegvars_time_false[m] *pro_eegautos_time_false[m]* pro_RRIvars_time_false[m] * pro_circadian_time_false[m])
# #     Pcombined.append(P1 / (P1 + P2))
#
#
# # Pcombined = []
# # for m in range(len(pro_circadian_time)):
# #     P1 = Pseizureeegvar * pro_eegvars_time[m] * pro_eegautos_time[m]*pro_RRIautos_time[m] * pro_circadian_time[m]
# #     P2 = Pnonseizureeegvar * (1 - pro_eegvars_time_false[m] *pro_eegautos_time_false[m]* pro_RRIautos_time_false[m] * pro_circadian_time_false[m])
# #     Pcombined.append(P1 / (P1 + P2))
#
# # Pcombined = []
# # for m in range(len(pro_circadian_time)):
# #     P1 = Pseizureeegvar * pro_eegvars_time[m] * pro_eegautos_time[m] * pro_circadian_time[m]
# #     P2 = Pnonseizureeegvar * (1 - pro_eegvars_time_false[m] *pro_eegautos_time_false[m]*pro_circadian_time_false[m])
# #     Pcombined.append(P1 / (P1 + P2))
#
# # Pcombined = []
# # for m in range(len(pro_circadian_time)):
# #     P1 = Pseizureeegvar *  pro_eegvars_time[m] * pro_circadian_time[m]
# #     P2 = Pnonseizureeegvar * (1 - pro_eegvars_time_false[m] *pro_circadian_time_false[m])
# #     Pcombined.append(P1 / (P1 + P2))
#
# # Pcombined = []
# # for m in range(len(pro_circadian_time)):
# #     P1 = Pseizureeegvar *  pro_eegautos_time[m] * pro_circadian_time[m]
# #     P2 = Pnonseizureeegvar * (1 - pro_eegautos_time_false[m] *pro_circadian_time_false[m])
# #     Pcombined.append(P1 / (P1 + P2))
#
# # Pcombined = []
# # for m in range(len(pro_circadian_time)):
# #     P1 = Pseizureeegvar * pro_RRIvars_time[m] * pro_RRIautos_time[m] * pro_circadian_time[m]
# #     P2 = Pnonseizureeegvar * (1 - pro_RRIvars_time_false[m] *pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
# #     Pcombined.append(P1 / (P1 + P2))
#
# # Pcombined = []
# # for m in range(len(pro_circadian_time)):
# #     P1 = Pseizureeegvar * pro_RRIvars_time[m] * pro_circadian_time[m]
# #     P2 = Pnonseizureeegvar * (1 - pro_RRIvars_time_false[m] *pro_circadian_time_false[m])
# #     Pcombined.append(P1 / (P1 + P2))
# # Pcombined = []
# # for m in range(len(pro_circadian_time)):
# #     P1 = Pseizureeegvar * pro_RRIautos_time[m] * pro_circadian_time[m]
# #     P2 = Pnonseizureeegvar * (1 - pro_RRIautos_time_false[m] *pro_circadian_time_false[m])
# #     Pcombined.append(P1 / (P1 + P2))
#
#
# # pyplot.figure(figsize=(8, 4))
# # RRI_timewindow_arr = t[0:len(pro_circadian_time)]
# # pyplot.plot(RRI_timewindow_arr, Pcombined)
# # pyplot.annotate('', xy=(95.493611, np.max(Pcombined)), xytext=(95.493611, np.max(Pcombined) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
# # pyplot.annotate('', xy=(97.3675, np.max(Pcombined)), xytext=(97.3675, np.max(Pcombined) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
# # pyplot.annotate('', xy=(99.079444, np.max(Pcombined)), xytext=(99.079444, np.max(Pcombined) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
# # pyplot.annotate('', xy=(102.43722, np.max(Pcombined)), xytext=(102.43722, np.max(Pcombined) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
# # pyplot.annotate('', xy=(120.36472, np.max(Pcombined)), xytext=(120.36472, np.max(Pcombined) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
# # pyplot.annotate('', xy=(124.15888, np.max(Pcombined)), xytext=(124.15888, np.max(Pcombined) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
# # pyplot.annotate('', xy=(139.29889, np.max(Pcombined)), xytext=(139.29889, np.max(Pcombined) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
# # pyplot.title('Forecast seizures in QLD1282')
# # pyplot.xlabel('Time(h)')
# # pyplot.ylabel('Seizure probability')
# # pyplot.show()
#
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
# # time_arr=[95.493611,97.3675,99.079444,102.43722,120.36472,124.15888,139.29889]
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
# # time_arr=[95.493611,97.3675,99.079444,102.43722,120.36472,124.15888,139.29889]
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
# # time_arr=[95.493611,97.3675,99.079444,102.43722,120.36472,124.15888,139.29889]
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
# # time_arr=[95.493611,97.3675,99.079444,102.43722,120.36472,124.15888,139.29889]
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
# # time_arr=[95.493611,97.3675,99.079444,102.43722,120.36472,124.15888,139.29889]
# # k=0
# # n_arr=[]
# # for m in time_arr:
# #     for n in a:
# #         if m-n<=1 and m-n>=0:
# #             k=k+1
# #             n_arr.append(n)
# # print(k)
#
#
#
# # Pcombined = split(Pcombined, 6)
# # print(len(Pcombined))
# # time_arr_arr_EEGcirca=[]
# # AUC_com_arr_EEGcirca=[]
# # for i in range(50000):
# #     time_arr = np.random.uniform(low=t_window_arr[19450], high=t_window_arr[-1], size=7)
# #     time_arr_arr_EEGcirca.append(time_arr)
# #     time_arr=np.sort(time_arr)
# #     index=[]
# #     for i in range(len(Pcombined)):
# #         for item in Pcombined[i]:
# #             if item >= Th2:
# #                 index.append(6*i+0)
# #     # print(RRI_timewindow_arr[index])
# #     a6=np.unique(RRI_timewindow_arr[index])
# #     # print(a6);
# #     # print(len(a6))
# #     k6=0
# #     n_arr=[]
# #     for m in time_arr:
# #         for n in a6:
# #             if m-n<=1 and m-n>=0:
# #                 k6=k6+1
# #                 n_arr.append(n)
# #     # print(k6)
# #
# #     index=[]
# #     for i in range(len(Pcombined)):
# #         for item in Pcombined[i]:
# #             if item >= 0.3*Th2:
# #                 index.append(6*i+0)
# #     # print(RRI_timewindow_arr[index])
# #     a7=np.unique(RRI_timewindow_arr[index])
# #     # print(a7);
# #     # print(len(a7))
# #     k7=0
# #     n_arr=[]
# #     for m in time_arr:
# #         for n in a7:
# #             if m-n<=1 and m-n>=0:
# #                 k7=k7+1
# #                 n_arr.append(n)
# #     # print(k7)
# #
# #     index=[]
# #     for i in range(len(Pcombined)):
# #         for item in Pcombined[i]:
# #             if item >= 0.6*Th2:
# #                 index.append(6*i+0)
# #     # print(RRI_timewindow_arr[index])
# #     a8=np.unique(RRI_timewindow_arr[index])
# #     # print(a8);
# #     # print(len(a8))
# #     k8=0
# #     n_arr=[]
# #     for m in time_arr:
# #         for n in a8:
# #             if m-n<=1 and m-n>=0:
# #                 k8=k8+1
# #                 n_arr.append(n)
# #     # print(k8)
# #
# #     index=[]
# #     for i in range(len(Pcombined)):
# #         for item in Pcombined[i]:
# #             if item >= 1.2*Th2:
# #                 index.append(6*i+0)
# #     # print(RRI_timewindow_arr[index])
# #     a9=np.unique(RRI_timewindow_arr[index])
# #     # print(a9);
# #     # print(len(a9))
# #     k9=0
# #     n_arr=[]
# #     for m in time_arr:
# #         for n in a9:
# #             if m-n<=1 and m-n>=0:
# #                 k9=k9+1
# #                 n_arr.append(n)
# #     # print(k9)
# #
# #     index=[]
# #     for i in range(len(Pcombined)):
# #         for item in Pcombined[i]:
# #             if item >= 2*Th2:
# #                 index.append(6*i+0)
# #     # print(RRI_timewindow_arr[index])
# #     a10=np.unique(RRI_timewindow_arr[index])
# #     k10=0
# #     n_arr=[]
# #     for m in time_arr:
# #         for n in a10:
# #             if m-n<=1 and m-n>=0:
# #                 k10=k10+1
# #                 n_arr.append(n)
# #     # print(k10)
# #
# #     Sen6=k6/len(time_arr);Sen7=k7/len(time_arr);Sen8=k8/len(time_arr);Sen9=k9/len(time_arr);Sen10=k10/len(time_arr);
# #     FPR6=(len(a6)-k6)/len(Pcombined);FPR7=(len(a7)-k7)/len(Pcombined);FPR8=(len(a8)-k8)/len(Pcombined);FPR9=(len(a9)-k9)/len(Pcombined);FPR10=(len(a10)-k10)/len(Pcombined);
# #     Sen_arr_COM=[0,Sen6,Sen7,Sen8,Sen9,Sen10,1]
# #     FPR_arr_COM=[0,FPR6,FPR7,FPR8,FPR9,FPR10,1]
# #     # print(Sen_arr_COM);print(FPR_arr_COM);
# #
# #     from sklearn.metrics import auc
# #     AUC_com=auc(np.sort(FPR_arr_COM),np.sort(Sen_arr_COM))
# #     AUC_com_arr_EEGcirca.append(AUC_com)
# #
# # # print(AUC_com_arr_EEGcirca)
# # # print(time_arr_arr_EEGcirca)
# # np.savetxt("AUC_ECG_var_circa1h_QLD1282.csv", AUC_com_arr_EEGcirca, delimiter=",", fmt='%s')
# # np.savetxt("seizure_labels_ECG_var_circa1h_QLD1282.csv", time_arr_arr_EEGcirca, delimiter=",", fmt='%s')