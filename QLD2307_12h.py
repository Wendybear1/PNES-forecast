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







csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD2307/Cz_EEGvariance_QLD2307_15s_3h.csv', sep=',',
                         header=None)
Raw_variance_EEG = csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD2307/Cz_EEGauto_QLD2307_15s_3h.csv', sep=',',
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
# pyplot.title('raw EEG variance in QLD2307', fontsize=13)
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
# pyplot.title('EEG variance in QLD2307', fontsize=13)
# pyplot.show()

seizure_timing_index = []
for k in range(len(window_time_arr)):
    if window_time_arr[k] < 7.195556 and window_time_arr[k + 1] >= 7.195556:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 26.689722 and window_time_arr[k + 1] >= 26.689722:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 31.439166 and window_time_arr[k + 1] >= 31.439166:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 45.970556 and window_time_arr[k + 1] >= 45.970556:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 50.723056 and window_time_arr[k + 1] >= 50.723056:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 55.04944 and window_time_arr[k + 1] >= 55.04944:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 70.144166 and window_time_arr[k + 1] >= 70.144166:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 75.387222 and window_time_arr[k + 1] >= 75.387222:
        seizure_timing_index.append(k)
    # if window_time_arr[k] < 97.562222 and window_time_arr[k + 1] >= 97.562222:
    #     seizure_timing_index.append(k)
    # if window_time_arr[k] < 100.63694 and window_time_arr[k + 1] >= 100.63694:
    #     seizure_timing_index.append(k)
    # if window_time_arr[k] < 102.63111 and window_time_arr[k + 1] >= 102.63111:
    #     seizure_timing_index.append(k)
    # if window_time_arr[k] < 127.05055 and window_time_arr[k + 1] >= 127.05055:
    #     seizure_timing_index.append(k)
    # if window_time_arr[k] < 147.38778 and window_time_arr[k + 1] >= 147.38778:
    #     seizure_timing_index.append(k)
    # if window_time_arr[k] < 150.84722 and window_time_arr[k + 1] >= 150.84722:
    #     seizure_timing_index.append(k)
    # if window_time_arr[k] < 152.03055 and window_time_arr[k + 1] >= 152.03055:
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
width = 2 * np.pi / bins_number
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
# pyplot.title('raw EEG autocorrelation in QLD2307', fontsize=13)
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
# pyplot.title('EEG autocorrelation in QLD2307', fontsize=13)
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
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD2307/RRI_ch31_timewindowarr_QLD2307_15s_3h.csv',sep=',', header=None)
rri_t = csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD2307/RRI_ch31_rawvariance_QLD2307_15s_3h.csv',sep=',', header=None)
RRI_var = csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD2307/RRI_ch31_rawauto_QLD2307_15s_3h.csv', sep=',',header=None)
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
# pyplot.title('RRI variance in QLD2307', fontsize=13)
# pyplot.show()
#
# pyplot.plot(Raw_auto_RRI31_arr, 'grey', alpha=0.5)
# pyplot.xlabel('Time (hours)', fontsize=13)
# pyplot.title('RRI autocorrelation in QLD2307', fontsize=13)
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
params = dict(projection='polar')
fig, ax2 = pyplot.subplots(subplot_kw=params)
ax2.bar(bins[:bins_number], nRRIsvarsei / sum(nRRIsvarsei), width=width, color='grey', alpha=0.7, edgecolor='k',
        linewidth=2)
ax2.set_title('RRI variance', fontsize=16)
locs, labels = pyplot.yticks([0.1, 0.3, 0.5], ['0.1', '0.3', '0.5'], fontsize=16)
# ax2.set_rlim([0,0.002])
ax2.set_xticklabels(['0', '0.25$\pi$', '0.5$\pi$', '0.75$\pi$', '-$\pi$', '-0.75$\pi$', '-0.5$\pi$', '-0.25$\pi$', ],
                    fontsize=16)
pyplot.show()






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
# a = np.where(t < 11.8797222 + 0)
# print(a);
# print(t[2851]);print(t[2852]);
# t[0:2851] = t[0:2851] - 0 + 12.1202778
# t[2851:] = t[2851:] - 11.8797222 - 0
# print(t[2851]);print(t[2852]);print(t[0])
#
# time_feature_arr = []
# for i in range(len(t)):
#     if t[i] > 24:
#         time_feature_arr.append(t[i] - (t[i] // 24) * 24)
#     else:
#         time_feature_arr.append(t[i])
# seizure_time = [time_feature_arr[1725], time_feature_arr[6404], time_feature_arr[7544], time_feature_arr[11031],
#                 time_feature_arr[12172],time_feature_arr[13210],time_feature_arr[16833],time_feature_arr[18091],
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
# ax.annotate("", xy=(-1.965462527, 0.610983386), xytext=(0, 0),arrowprops=dict(arrowstyle="->",color='g',linewidth=2))
# pyplot.show()
#
# bins_number = 18
# bins = np.linspace(0, 24, bins_number + 1)
# ntimes, _, _ = pyplot.hist(time_feature_arr[0:19450], bins)
# ntimesei, _, _ = pyplot.hist(seizure_time, bins)
# print(ntimes)
# print(ntimesei)




# #### section 2 training training
medium_rhythm_var_arr_3 = movingaverage(Raw_variance_EEG, 20 * 1)
long_rhythm_var_arr = medium_rhythm_var_arr_3
var_trans = hilbert(long_rhythm_var_arr)
var_phase = np.angle(var_trans)
phase_long_EEGvariance_arr = var_phase
print(len(phase_long_EEGvariance_arr));
medium_rhythm_value_arr_3 = movingaverage(Raw_auto_EEG, 20 * 1)
long_rhythm_value_arr = medium_rhythm_value_arr_3
value_trans = hilbert(long_rhythm_value_arr)
value_phase = np.angle(value_trans)
phase_long_EEGauto_arr = value_phase
print(len(phase_long_EEGauto_arr));
medium_rhythm_RRIvar_arr_3 = movingaverage(Raw_variance_RRI31, 20 * 1)
long_rhythm_RRIvar_arr = medium_rhythm_RRIvar_arr_3
var_trans = hilbert(long_rhythm_RRIvar_arr)
var_phase = np.angle(var_trans)
phase_long_RRIvar_arr = var_phase
print(len(phase_long_RRIvar_arr));
medium_rhythm_RRIvalue_arr_3 = movingaverage(Raw_auto_RRI31, 20 * 1)
long_rhythm_RRIvalue_arr = medium_rhythm_RRIvalue_arr_3
value_trans = hilbert(long_rhythm_RRIvalue_arr)
value_phase = np.angle(value_trans)
phase_long_RRIauto_arr = value_phase
print(len(phase_long_RRIauto_arr));



### combined probability calculation
### 24h 24h 24h $$$$
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
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[5] and phase_long_RRIvar_arr[i] < bins[6]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[6] and phase_long_RRIvar_arr[i] < bins[7]:
#         pro_RRIvars_time_false.append(0.006995165)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[7] and phase_long_RRIvar_arr[i] <= bins[8]:
#         pro_RRIvars_time_false.append(0.013836025)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] > bins[8] and phase_long_RRIvar_arr[i] <= bins[9]:
#         pro_RRIvars_time_false.append(0.621643864)
#         pro_RRIvars_time.append(0.75)
#     elif phase_long_RRIvar_arr[i] > bins[9] and phase_long_RRIvar_arr[i] <= bins[10]:
#         pro_RRIvars_time_false.append(0.23330933)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] > bins[10] and phase_long_RRIvar_arr[i] < bins[11]:
#         pro_RRIvars_time_false.append(0.08913692)
#         pro_RRIvars_time.append(0.125)
#     elif phase_long_RRIvar_arr[i] >= bins[11] and phase_long_RRIvar_arr[i] < bins[12]:
#         pro_RRIvars_time_false.append(0.033329904)
#         pro_RRIvars_time.append(0.125)
#     elif phase_long_RRIvar_arr[i] >= bins[12] and phase_long_RRIvar_arr[i] < bins[13]:
#         pro_RRIvars_time_false.append(0.000565785)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[13] and phase_long_RRIvar_arr[i] < bins[14]:
#         pro_RRIvars_time_false.append(0.001183006)
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
# print(pro_RRIvars_time[1725]);print(pro_RRIvars_time[6404]);print(pro_RRIvars_time[7544]);print(pro_RRIvars_time[11031]);
# print(pro_RRIvars_time[12172]);print(pro_RRIvars_time[13210]);print(pro_RRIvars_time[16833]);print(pro_RRIvars_time[18091]);
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
#         pro_RRIautos_time_false.append(0.201882522)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] > bins[8] and phase_long_RRIauto_arr[i] <= bins[9]:
#         pro_RRIautos_time_false.append(0.362308404)
#         pro_RRIautos_time.append(0.625)
#     elif phase_long_RRIauto_arr[i] > bins[9] and phase_long_RRIauto_arr[i] <= bins[10]:
#         pro_RRIautos_time_false.append(0.318434317)
#         pro_RRIautos_time.append(0.375)
#     elif phase_long_RRIauto_arr[i] > bins[10] and phase_long_RRIauto_arr[i] < bins[11]:
#         pro_RRIautos_time_false.append(0.056064191)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[11] and phase_long_RRIauto_arr[i] < bins[12]:
#         pro_RRIautos_time_false.append(0.017847958)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[12] and phase_long_RRIauto_arr[i] < bins[13]:
#         pro_RRIautos_time_false.append(0.016664952)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[13] and phase_long_RRIauto_arr[i] < bins[14]:
#         pro_RRIautos_time_false.append(0.026797655)
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
# print(pro_RRIautos_time[1725]);print(pro_RRIautos_time[6404]);print(pro_RRIautos_time[7544]);print(pro_RRIautos_time[11031]);
# print(pro_RRIautos_time[12172]);print(pro_RRIautos_time[13210]);print(pro_RRIautos_time[16833]);print(pro_RRIautos_time[18091]);


# ##### 12h 12h 12h $$$
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
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[5] and phase_long_RRIvar_arr[i] < bins[6]:
#         pro_RRIvars_time_false.append(0.002828927)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[6] and phase_long_RRIvar_arr[i] < bins[7]:
#         pro_RRIvars_time_false.append(0.011470013)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[7] and phase_long_RRIvar_arr[i] <= bins[8]:
#         pro_RRIvars_time_false.append(0.269365292)
#         pro_RRIvars_time.append(0.5)
#     elif phase_long_RRIvar_arr[i] > bins[8] and phase_long_RRIvar_arr[i] <= bins[9]:
#         pro_RRIvars_time_false.append(0.234338031)
#         pro_RRIvars_time.append(0.125)
#     elif phase_long_RRIvar_arr[i] > bins[9] and phase_long_RRIvar_arr[i] <= bins[10]:
#         pro_RRIvars_time_false.append(0.306655694)
#         pro_RRIvars_time.append(0.375)
#     elif phase_long_RRIvar_arr[i] > bins[10] and phase_long_RRIvar_arr[i] < bins[11]:
#         pro_RRIvars_time_false.append(0.084764942)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[11] and phase_long_RRIvar_arr[i] < bins[12]:
#         pro_RRIvars_time_false.append(0.058893118)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[12] and phase_long_RRIvar_arr[i] < bins[13]:
#         pro_RRIvars_time_false.append(0.031221068)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[13] and phase_long_RRIvar_arr[i] < bins[14]:
#         pro_RRIvars_time_false.append(0.000462915)
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
# print(pro_RRIvars_time[1725]);print(pro_RRIvars_time[6404]);print(pro_RRIvars_time[7544]);print(pro_RRIvars_time[11031]);
# print(pro_RRIvars_time[12172]);print(pro_RRIvars_time[13210]);print(pro_RRIvars_time[16833]);print(pro_RRIvars_time[18091]);
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
#         pro_RRIautos_time_false.append(0.092223022)
#         pro_RRIautos_time.append(0.125)
#     elif phase_long_RRIauto_arr[i] > bins[8] and phase_long_RRIauto_arr[i] <= bins[9]:
#         pro_RRIautos_time_false.append(0.443575764)
#         pro_RRIautos_time.append(0.5)
#     elif phase_long_RRIauto_arr[i] > bins[9] and phase_long_RRIauto_arr[i] <= bins[10]:
#         pro_RRIautos_time_false.append(0.402376299)
#         pro_RRIautos_time.append(0.375)
#     elif phase_long_RRIauto_arr[i] > bins[10] and phase_long_RRIauto_arr[i] < bins[11]:
#         pro_RRIautos_time_false.append(0.027209135)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[11] and phase_long_RRIauto_arr[i] < bins[12]:
#         pro_RRIautos_time_false.append(0.009566917)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[12] and phase_long_RRIauto_arr[i] < bins[13]:
#         pro_RRIautos_time_false.append(0.008075301)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[13] and phase_long_RRIauto_arr[i] < bins[14]:
#         pro_RRIautos_time_false.append(0.016973562)
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
# print(pro_RRIautos_time[1725]);print(pro_RRIautos_time[6404]);print(pro_RRIautos_time[7544]);print(pro_RRIautos_time[11031]);
# print(pro_RRIautos_time[12172]);print(pro_RRIautos_time[13210]);print(pro_RRIautos_time[16833]);print(pro_RRIautos_time[18091]);


# ##### 6h 6h 6h $$$
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
#         pro_eegvars_time_false.append(0.017539348)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[5] and phase_long_EEGvariance_arr[i] < bins[6]:
#         pro_eegvars_time_false.append(0.142989404)
#         pro_eegvars_time.append(0.125)
#     elif phase_long_EEGvariance_arr[i] >= bins[6] and phase_long_EEGvariance_arr[i] <= bins[7]:
#         pro_eegvars_time_false.append(0.188766588)
#         pro_eegvars_time.append(0.25)
#     elif phase_long_EEGvariance_arr[i] > bins[7] and phase_long_EEGvariance_arr[i] < bins[8]:
#         pro_eegvars_time_false.append(0.093868944)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[8] and phase_long_EEGvariance_arr[i] <= bins[9]:
#         pro_eegvars_time_false.append(0.053080959)
#         pro_eegvars_time.append(0.125)
#     elif phase_long_EEGvariance_arr[i] > bins[9] and phase_long_EEGvariance_arr[i] < bins[10]:
#         pro_eegvars_time_false.append(0.100195453)
#         pro_eegvars_time.append(0.125)
#     elif phase_long_EEGvariance_arr[i] >= bins[10] and phase_long_EEGvariance_arr[i] <= bins[11]:
#         pro_eegvars_time_false.append(0.126221582)
#         pro_eegvars_time.append(0.125)
#     elif phase_long_EEGvariance_arr[i] > bins[11] and phase_long_EEGvariance_arr[i] < bins[12]:
#         pro_eegvars_time_false.append(0.064756712)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] > bins[12] and phase_long_EEGvariance_arr[i] < bins[13]:
#         pro_eegvars_time_false.append(0.151630491)
#         pro_eegvars_time.append(0.125)
#     elif phase_long_EEGvariance_arr[i] >= bins[13] and phase_long_EEGvariance_arr[i] < bins[14]:
#         pro_eegvars_time_false.append(0.060950519)
#         pro_eegvars_time.append(0.125)
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
# print(pro_eegvars_time[1725]);print(pro_eegvars_time[6404]);print(pro_eegvars_time[7544]);print(pro_eegvars_time[11031]);
# print(pro_eegvars_time[12172]);print(pro_eegvars_time[13210]);print(pro_eegvars_time[16833]);print(pro_eegvars_time[18091]);
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
#         pro_eegautos_time_false.append(0.001028701)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[6] and phase_long_EEGauto_arr[i] < bins[7]:
#         pro_eegautos_time_false.append(0.021808456)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[7] and phase_long_EEGauto_arr[i] < bins[8]:
#         pro_eegautos_time_false.append(0.090268491)
#         pro_eegautos_time.append(0.125)
#     elif phase_long_EEGauto_arr[i] >= bins[8] and phase_long_EEGauto_arr[i] < bins[9]:
#         pro_eegautos_time_false.append(0.343740356)
#         pro_eegautos_time.append(0.375)
#     elif phase_long_EEGauto_arr[i] >= bins[9] and phase_long_EEGauto_arr[i] <= bins[10]:
#         pro_eegautos_time_false.append(0.432928711)
#         pro_eegautos_time.append(0.5)
#     elif phase_long_EEGauto_arr[i] > bins[10] and phase_long_EEGauto_arr[i] < bins[11]:
#         pro_eegautos_time_false.append(0.092428762)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[11] and phase_long_EEGauto_arr[i] < bins[12]:
#         pro_eegautos_time_false.append(0.005760724)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[12] and phase_long_EEGauto_arr[i] < bins[13]:
#         pro_eegautos_time_false.append(0.004629153)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[13] and phase_long_EEGauto_arr[i] < bins[14]:
#         pro_eegautos_time_false.append(0.007406645)
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
# print(pro_eegautos_time[1725]);print(pro_eegautos_time[6404]);print(pro_eegautos_time[7544]);print(pro_eegautos_time[11031]);
# print(pro_eegautos_time[12172]);print(pro_eegautos_time[13210]);print(pro_eegautos_time[16833]);print(pro_eegautos_time[18091]);
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
#         pro_RRIvars_time_false.append(0.001491616)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[5] and phase_long_RRIvar_arr[i] < bins[6]:
#         pro_RRIvars_time_false.append(0.005863594)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[6] and phase_long_RRIvar_arr[i] < bins[7]:
#         pro_RRIvars_time_false.append(0.085742208)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[7] and phase_long_RRIvar_arr[i] <= bins[8]:
#         pro_RRIvars_time_false.append(0.284590063)
#         pro_RRIvars_time.append(0.375)
#     elif phase_long_RRIvar_arr[i] > bins[8] and phase_long_RRIvar_arr[i] < bins[9]:
#         pro_RRIvars_time_false.append(0.134759798)
#         pro_RRIvars_time.append(0.125)
#     elif phase_long_RRIvar_arr[i] >= bins[9] and phase_long_RRIvar_arr[i] <= bins[10]:
#         pro_RRIvars_time_false.append(0.215512807)
#         pro_RRIvars_time.append(0.125)
#     elif phase_long_RRIvar_arr[i] > bins[10] and phase_long_RRIvar_arr[i] <= bins[11]:
#         pro_RRIvars_time_false.append(0.148647259)
#         pro_RRIvars_time.append(0.25)
#     elif phase_long_RRIvar_arr[i] > bins[11] and phase_long_RRIvar_arr[i] < bins[12]:
#         pro_RRIvars_time_false.append(0.039244934)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[12] and phase_long_RRIvar_arr[i] < bins[13]:
#         pro_RRIvars_time_false.append(0.084147721)
#         pro_RRIvars_time.append(0.125)
#     elif phase_long_RRIvar_arr[i] >= bins[13] and phase_long_RRIvar_arr[i] < bins[14]:
#         pro_RRIvars_time_false.append(0)
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
# print(pro_RRIvars_time[1725]);print(pro_RRIvars_time[6404]);print(pro_RRIvars_time[7544]);print(pro_RRIvars_time[11031]);
# print(pro_RRIvars_time[12172]);print(pro_RRIvars_time[13210]);print(pro_RRIvars_time[16833]);print(pro_RRIvars_time[18091]);
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
#         pro_RRIautos_time_false.append(0.051435038)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] > bins[8] and phase_long_RRIauto_arr[i] <= bins[9]:
#         pro_RRIautos_time_false.append(0.47263656)
#         pro_RRIautos_time.append(0.5)
#     elif phase_long_RRIauto_arr[i] > bins[9] and phase_long_RRIauto_arr[i] <= bins[10]:
#         pro_RRIautos_time_false.append(0.436889209)
#         pro_RRIautos_time.append(0.5)
#     elif phase_long_RRIauto_arr[i] > bins[10] and phase_long_RRIauto_arr[i] < bins[11]:
#         pro_RRIautos_time_false.append(0.017487913)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[11] and phase_long_RRIauto_arr[i] < bins[12]:
#         pro_RRIautos_time_false.append(0.005760724)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[12] and phase_long_RRIauto_arr[i] < bins[13]:
#         pro_RRIautos_time_false.append(0.005040634)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[13] and phase_long_RRIauto_arr[i] < bins[14]:
#         pro_RRIautos_time_false.append(0.010749923)
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
# print(pro_RRIautos_time[1725]);print(pro_RRIautos_time[6404]);print(pro_RRIautos_time[7544]);print(pro_RRIautos_time[11031]);
# print(pro_RRIautos_time[12172]);print(pro_RRIautos_time[13210]);print(pro_RRIautos_time[16833]);print(pro_RRIautos_time[18091]);


# ##### 1h 1h 1h 1h $$$
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
#         pro_eegvars_time_false.append(0.191801255)
#         pro_eegvars_time.append(0.125)
#     elif phase_long_EEGvariance_arr[i] >= bins[5] and phase_long_EEGvariance_arr[i] < bins[6]:
#         pro_eegvars_time_false.append(0.131725131)
#         pro_eegvars_time.append(0.125)
#     elif phase_long_EEGvariance_arr[i] >= bins[6] and phase_long_EEGvariance_arr[i] <= bins[7]:
#         pro_eegvars_time_false.append(0.077975517)
#         pro_eegvars_time.append(0.25)
#     elif phase_long_EEGvariance_arr[i] > bins[7] and phase_long_EEGvariance_arr[i] < bins[8]:
#         pro_eegvars_time_false.append(0.05503549)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[8] and phase_long_EEGvariance_arr[i] <= bins[9]:
#         pro_eegvars_time_false.append(0.044028392)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] > bins[9] and phase_long_EEGvariance_arr[i] < bins[10]:
#         pro_eegvars_time_false.append(0.042228166)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[10] and phase_long_EEGvariance_arr[i] <= bins[11]:
#         pro_eegvars_time_false.append(0.056475671)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] > bins[11] and phase_long_EEGvariance_arr[i] < bins[12]:
#         pro_eegvars_time_false.append(0.066351198)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] > bins[12] and phase_long_EEGvariance_arr[i] < bins[13]:
#         pro_eegvars_time_false.append(0.148081473)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[13] and phase_long_EEGvariance_arr[i] < bins[14]:
#         pro_eegvars_time_false.append(0.186297706)
#         pro_eegvars_time.append(0.5)
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
# print(pro_eegvars_time[1725]);print(pro_eegvars_time[6404]);print(pro_eegvars_time[7544]);print(pro_eegvars_time[11031]);
# print(pro_eegvars_time[12172]);print(pro_eegvars_time[13210]);print(pro_eegvars_time[16833]);print(pro_eegvars_time[18091]);
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
#         pro_RRIvars_time_false.append(0.012910194)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[5] and phase_long_RRIvar_arr[i] < bins[6]:
#         pro_RRIvars_time_false.append(0.063316531)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[6] and phase_long_RRIvar_arr[i] < bins[7]:
#         pro_RRIvars_time_false.append(0.138514556)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] >= bins[7] and phase_long_RRIvar_arr[i] <= bins[8]:
#         pro_RRIvars_time_false.append(0.170249974)
#         pro_RRIvars_time.append(0.25)
#     elif phase_long_RRIvar_arr[i] > bins[8] and phase_long_RRIvar_arr[i] < bins[9]:
#         pro_RRIvars_time_false.append(0.125912972)
#         pro_RRIvars_time.append(0.375)
#     elif phase_long_RRIvar_arr[i] >= bins[9] and phase_long_RRIvar_arr[i] <= bins[10]:
#         pro_RRIvars_time_false.append(0.148852999)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] > bins[10] and phase_long_RRIvar_arr[i] <= bins[11]:
#         pro_RRIvars_time_false.append(0.132805267)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvar_arr[i] > bins[11] and phase_long_RRIvar_arr[i] < bins[12]:
#         pro_RRIvars_time_false.append(0.076792511)
#         pro_RRIvars_time.append(0.25)
#     elif phase_long_RRIvar_arr[i] >= bins[12] and phase_long_RRIvar_arr[i] < bins[13]:
#         pro_RRIvars_time_false.append(0.121901039)
#         pro_RRIvars_time.append(0.125)
#     elif phase_long_RRIvar_arr[i] >= bins[13] and phase_long_RRIvar_arr[i] < bins[14]:
#         pro_RRIvars_time_false.append(0.008743956)
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
# print(pro_RRIvars_time[1725]);print(pro_RRIvars_time[6404]);print(pro_RRIvars_time[7544]);print(pro_RRIvars_time[11031]);
# print(pro_RRIvars_time[12172]);print(pro_RRIvars_time[13210]);print(pro_RRIvars_time[16833]);print(pro_RRIvars_time[18091]);



#### 5min 5min 5min 5min $$
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
        pro_RRIvars_time_false.append(0.062545006)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvar_arr[i] >= bins[5] and phase_long_RRIvar_arr[i] < bins[6]:
        pro_RRIvars_time_false.append(0.139903302)
        pro_RRIvars_time.append(0.125)
    elif phase_long_RRIvar_arr[i] >= bins[6] and phase_long_RRIvar_arr[i] < bins[7]:
        pro_RRIvars_time_false.append(0.112745602)
        pro_RRIvars_time.append(0.125)
    elif phase_long_RRIvar_arr[i] >= bins[7] and phase_long_RRIvar_arr[i] <= bins[8]:
        pro_RRIvars_time_false.append(0.098138052)
        pro_RRIvars_time.append(0.25)
    elif phase_long_RRIvar_arr[i] > bins[8] and phase_long_RRIvar_arr[i] <= bins[9]:
        pro_RRIvars_time_false.append(0.085536467)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvar_arr[i] > bins[9] and phase_long_RRIvar_arr[i] <= bins[10]:
        pro_RRIvars_time_false.append(0.086925213)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvar_arr[i] > bins[10] and phase_long_RRIvar_arr[i] < bins[11]:
        pro_RRIvars_time_false.append(0.09587491)
        pro_RRIvars_time.append(0.125)
    elif phase_long_RRIvar_arr[i] >= bins[11] and phase_long_RRIvar_arr[i] < bins[12]:
        pro_RRIvars_time_false.append(0.11650036)
        pro_RRIvars_time.append(0.125)
    elif phase_long_RRIvar_arr[i] >= bins[12] and phase_long_RRIvar_arr[i] < bins[13]:
        pro_RRIvars_time_false.append(0.140777698)
        pro_RRIvars_time.append(0.125)
    elif phase_long_RRIvar_arr[i] >= bins[13] and phase_long_RRIvar_arr[i] < bins[14]:
        pro_RRIvars_time_false.append(0.06105339)
        pro_RRIvars_time.append(0.125)
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
print(pro_RRIvars_time[1725]);print(pro_RRIvars_time[6404]);print(pro_RRIvars_time[7544]);print(pro_RRIvars_time[11031]);
print(pro_RRIvars_time[12172]);print(pro_RRIvars_time[13210]);print(pro_RRIvars_time[16833]);print(pro_RRIvars_time[18091]);




Pseizureeegvar = 0.000411311;
Pnonseizureeegvar = 0.999588689;
t = np.linspace(0 + 0.00416667, 0 + 0.00416667 + 0.00416667 * (len(Raw_variance_EEG) - 1), len(Raw_variance_EEG))
window_time_arr = t


# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_RRIvars_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_RRIvars_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=Pseizureeegvar*pro_eegvars_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m])
#     Pcombined.append(P1/(P1+P2))


# Pcombined = []
# for m in range(len(pro_RRIvars_time)):
#     P1=pro_RRIvars_time[m]*Pseizureeegvar*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))

Pcombined = []
for m in range(len(pro_RRIvars_time)):
    P1=Pseizureeegvar*pro_RRIvars_time[m]
    P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m])
    Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_RRIautos_time)):
#     P1=Pseizureeegvar*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))


pyplot.figure(figsize=(12, 5))
pyplot.plot(window_time_arr, Pcombined)
pyplot.title('Combined probability in QLD2307', fontsize=15)
pyplot.annotate('', xy=(7.195556, np.max(Pcombined)), xytext=(7.195556, np.max(Pcombined) + 0.00000000001),
                arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(26.689722, np.max(Pcombined)), xytext=(26.689722, np.max(Pcombined) + 0.00000000001),
                arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(31.439166, np.max(Pcombined)), xytext=(31.439166, np.max(Pcombined) + 0.00000000001),
                arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(45.970556, np.max(Pcombined)), xytext=(45.970556, np.max(Pcombined) + 0.00000000001),
                arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(50.723056, np.max(Pcombined)), xytext=(50.723056, np.max(Pcombined) + 0.00000000001),
                arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(55.04944, np.max(Pcombined)), xytext=(55.04944, np.max(Pcombined) + 0.00000000001),
                arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(70.144166, np.max(Pcombined)), xytext=(70.144166, np.max(Pcombined) + 0.00000000001),
                arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(75.387222, np.max(Pcombined)), xytext=(75.387222, np.max(Pcombined) + 0.00000000001),
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
# a = np.where(t < 11.8797222 + 0)
# t[0:2851] = t[0:2851] - 0 + 12.1202778
# t[2851:] = t[2851:] - 11.8797222 - 0
# time_feature_arr = []
# for i in range(len(t)):
#     if t[i] > 24:
#         time_feature_arr.append(t[i] - (t[i] // 24) * 24)
#     else:
#         time_feature_arr.append(t[i])
#
#
# bins_number = 18
# bins = np.linspace(0, 24, bins_number + 1)
# pro_circadian_time = []
# pro_circadian_time_false = []
# for i in range(len(time_feature_arr)):
#     if time_feature_arr[i] >= bins[0] and time_feature_arr[i] <= bins[1]:
#         pro_circadian_time_false.append(0.049377636)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] > bins[1] and time_feature_arr[i] < bins[2]:
#         pro_circadian_time_false.append(0.049377636)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] >= bins[2] and time_feature_arr[i] < bins[3]:
#         pro_circadian_time_false.append(0.049377636)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] >= bins[3] and time_feature_arr[i] < bins[4]:
#         pro_circadian_time_false.append(0.049377636)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] >= bins[4] and time_feature_arr[i] < bins[5]:
#         pro_circadian_time_false.append(0.049377636)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] >= bins[5] and time_feature_arr[i] <= bins[6]:
#         pro_circadian_time_false.append(0.049377636)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] > bins[6] and time_feature_arr[i] < bins[7]:
#         pro_circadian_time_false.append(0.049377636)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] >= bins[7] and time_feature_arr[i] <= bins[8]:
#         pro_circadian_time_false.append(0.049274766)
#         pro_circadian_time.append(0.25)
#     elif time_feature_arr[i] > bins[8] and time_feature_arr[i] < bins[9]:
#         pro_circadian_time_false.append(0.049377636)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] >= bins[9] and time_feature_arr[i] < bins[10]:
#         pro_circadian_time_false.append(0.064345232)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] >= bins[10] and time_feature_arr[i] < bins[11]:
#         pro_circadian_time_false.append(0.065836848)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] >= bins[11] and time_feature_arr[i] < bins[12]:
#         pro_circadian_time_false.append(0.065682543)
#         pro_circadian_time.append(0.375)
#     elif time_feature_arr[i] >= bins[12] and time_feature_arr[i] < bins[13]:
#         pro_circadian_time_false.append(0.065836848)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] >= bins[13] and time_feature_arr[i] < bins[14]:
#         pro_circadian_time_false.append(0.065836848)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] >= bins[14] and time_feature_arr[i] < bins[15]:
#         pro_circadian_time_false.append(0.065682543)
#         pro_circadian_time.append(0.375)
#     elif time_feature_arr[i] >= bins[15] and time_feature_arr[i] < bins[16]:
#         pro_circadian_time_false.append(0.063728012)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] >= bins[16] and time_feature_arr[i] < bins[17]:
#         pro_circadian_time_false.append(0.049377636)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] >= bins[17] and time_feature_arr[i] <= bins[18]:
#         pro_circadian_time_false.append(0.049377636)
#         pro_circadian_time.append(0)
# print(pro_circadian_time[1725]);print(pro_circadian_time[6404]);print(pro_circadian_time[7544]);print(pro_circadian_time[11031]);
# print(pro_circadian_time[12172]);print(pro_circadian_time[13210]);print(pro_circadian_time[16833]);print(pro_circadian_time[18091]);
#
#
# #
# # Pcombined=[]
# # for m in range(len(pro_eegvars_time)):
# #     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_RRIvars_time[m]*pro_circadian_time[m]
# #     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_RRIvars_time_false[m]*pro_circadian_time_false[m])
# #     Pcombined.append(P1/(P1+P2))
#
# # Pcombined=[]
# # for m in range(len(pro_eegvars_time)):
# #     P1=Pseizureeegvar*pro_eegvars_time[m]*pro_circadian_time[m]
# #     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_circadian_time_false[m])
# #     Pcombined.append(P1/(P1+P2))
#
#
# # Pcombined=[]
# # for m in range(len(pro_RRIvars_time)):
# #     P1=pro_RRIvars_time[m]*Pseizureeegvar*pro_RRIautos_time[m]*pro_circadian_time[m]
# #     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
# #     Pcombined.append(P1/(P1+P2))
#
# # Pcombined=[]
# # for m in range(len(pro_RRIvars_time)):
# #     P1=Pseizureeegvar*pro_RRIvars_time[m]*pro_circadian_time[m]
# #     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m]*pro_circadian_time_false[m])
# #     Pcombined.append(P1/(P1+P2))
#
# # Pcombined=[]
# # for m in range(len(pro_RRIautos_time)):
# #     P1=Pseizureeegvar*pro_RRIautos_time[m]*pro_circadian_time[m]
# #     P2=Pnonseizureeegvar*(1-pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
# #     Pcombined.append(P1/(P1+P2))
#
# # t = np.linspace(0 + 0.00416667, 0 + 0.00416667 + 0.00416667 * (len(Raw_variance_EEG) - 1), len(Raw_variance_EEG))
# # window_time_arr = t
# # pyplot.figure(figsize=(12, 5))
# # pyplot.plot(window_time_arr, Pcombined)
# # pyplot.title('combined probability in QLD2307', fontsize=15)
# # pyplot.annotate('', xy=(7.195556, np.max(Pcombined)), xytext=(7.195556, np.max(Pcombined) + 0.00000000001),
# #                 arrowprops=dict(facecolor='black', shrink=0.05))
# # pyplot.annotate('', xy=(26.689722, np.max(Pcombined)), xytext=(26.689722, np.max(Pcombined) + 0.00000000001),
# #                 arrowprops=dict(facecolor='black', shrink=0.05))
# # pyplot.annotate('', xy=(31.439166, np.max(Pcombined)), xytext=(31.439166, np.max(Pcombined) + 0.00000000001),
# #                 arrowprops=dict(facecolor='black', shrink=0.05))
# # pyplot.annotate('', xy=(45.970556, np.max(Pcombined)), xytext=(45.970556, np.max(Pcombined) + 0.00000000001),
# #                 arrowprops=dict(facecolor='black', shrink=0.05))
# # pyplot.annotate('', xy=(50.723056, np.max(Pcombined)), xytext=(50.723056, np.max(Pcombined) + 0.00000000001),
# #                 arrowprops=dict(facecolor='black', shrink=0.05))
# # pyplot.annotate('', xy=(55.04944, np.max(Pcombined)), xytext=(55.04944, np.max(Pcombined) + 0.00000000001),
# #                 arrowprops=dict(facecolor='black', shrink=0.05))
# # pyplot.annotate('', xy=(70.144166, np.max(Pcombined)), xytext=(70.144166, np.max(Pcombined) + 0.00000000001),
# #                 arrowprops=dict(facecolor='black', shrink=0.05))
# # pyplot.annotate('', xy=(75.387222, np.max(Pcombined)), xytext=(75.387222, np.max(Pcombined) + 0.00000000001),
# #                 arrowprops=dict(facecolor='black', shrink=0.05))
# # pyplot.tight_layout()
# # pyplot.xlim(window_time_arr[0], window_time_arr[-1])
# # pyplot.xlabel('Time(h)', fontsize=15)
# # pyplot.ylabel('seizure probability', fontsize=15)
# # pyplot.show()
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
for k in range(81, 82):
    variance_arr = Raw_variance_EEG_arr[0:(19450 + 240 * k)]
    long_rhythm_var_arr = movingaverage(variance_arr, 240 * 1)
    pyplot.figure(figsize=(6, 3))
    pyplot.title('EEG variance')
    pyplot.ylabel('Voltage ($\mathregular{v^2}$)')
    pyplot.xlabel('Time(h)')
    pyplot.plot(t_window_arr[240 * 24:(19450 + 240 * k)], long_rhythm_var_arr[240 * 24:], 'orange')
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD2307/cycles1h_Cz_forecast81hsignal_3hcycle_EEGvar_QLD2307.csv',
                         sep=',', header=None)
forecast_var_EEG = csv_reader.values
forecast_var_EEG_arr = []
for item in forecast_var_EEG:
    forecast_var_EEG_arr = forecast_var_EEG_arr + list(item)
t = np.linspace(t_window_arr[19450], t_window_arr[19450] + 0.1666667 * (len(forecast_var_EEG_arr) - 1),
                len(forecast_var_EEG_arr))
pyplot.plot(t, forecast_var_EEG_arr, 'k', label='forecast EEG var')
pyplot.legend()
pyplot.show()

fore_arr_EEGauto = []
for k in range(81, 82):
    auto_arr = Raw_auto_EEG_arr[0:(19450 + 240 * k)]
    long_rhythm_auto_arr = movingaverage(auto_arr, 240 * 6)
    pyplot.figure(figsize=(6, 3))
    pyplot.title('EEG autocorrelation')
    pyplot.xlabel('time(h)')
    pyplot.plot(t_window_arr[240 * 24:(19450 + 240 * k)], long_rhythm_auto_arr[240 * 24:], 'orange')
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD2307/cycles6h_Cz_forecast81hsignal_3hcycle_EEGauto_QLD2307.csv', sep=',',
    header=None)
forecast_auto_EEG = csv_reader.values
forecast_auto_EEG_arr = []
for item in forecast_auto_EEG:
    forecast_auto_EEG_arr = forecast_auto_EEG_arr + list(item)

t = np.linspace(t_window_arr[19450], t_window_arr[19450] + 0.1666667 * (len(forecast_auto_EEG_arr) - 1),
                len(forecast_auto_EEG_arr))
pyplot.plot(t, forecast_auto_EEG_arr, 'k', label='forecast EEG auto')
pyplot.legend()
pyplot.show()

fore_arr_RRIvars = []
for k in range(81, 82):
    variance_arr = Raw_variance_RRI31_arr[0:(19450 + 240 * k)]
    long_rhythm_var_arr = movingaverage(variance_arr, 20 * 1)
    pyplot.figure(figsize=(6, 3))
    pyplot.title('RRI variance')
    pyplot.ylabel('Second ($\mathregular{s^2}$)')
    pyplot.xlabel('Time(h)')
    pyplot.plot(t_window_arr[240 * 24:(19450 + 240 * k)], long_rhythm_var_arr[240 * 24:], 'orange')
csv_reader = pd.read_csv( 'C:/Users/wxiong/Documents/PHD/2011.1/QLD2307/cycles5min_ch31_forecast81hsignal_3hcycle_RRIvar_QLD2307.csv', sep=',',header=None)

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
    long_rhythm_auto_arr = movingaverage(auto_arr, 240 * 6)
    pyplot.figure(figsize=(6, 3))
    pyplot.title('RRI autocorrelation')
    pyplot.xlabel('Time(h)')
    pyplot.plot(t_window_arr[240 * 24:19450 + 240 * k], long_rhythm_auto_arr[240 * 24:], 'orange')
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD2307/cycles6h_ch31_forecast81hsignal_3hcycle_RRIauto_QLD2307.csv', sep=',',
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
# print(len(forecast_var_EEG_arr));print(len(forecast_auto_EEG_arr));print(len(forecast_var_RRI31_arr));print(len(forecast_auto_RRI31_arr));





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



# #### combined probability calculation
#### 24h 24h 24h $$$$
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
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[5] and rolmean_short_RRIvar[i] < bins[6]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[6] and rolmean_short_RRIvar[i] < bins[7]:
#         pro_RRIvars_time_false.append(0.006995165)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[7] and rolmean_short_RRIvar[i] <= bins[8]:
#         pro_RRIvars_time_false.append(0.013836025)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] > bins[8] and rolmean_short_RRIvar[i] <= bins[9]:
#         pro_RRIvars_time_false.append(0.621643864)
#         pro_RRIvars_time.append(0.75)
#     elif rolmean_short_RRIvar[i] > bins[9] and rolmean_short_RRIvar[i] <= bins[10]:
#         pro_RRIvars_time_false.append(0.23330933)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] > bins[10] and rolmean_short_RRIvar[i] < bins[11]:
#         pro_RRIvars_time_false.append(0.08913692)
#         pro_RRIvars_time.append(0.125)
#     elif rolmean_short_RRIvar[i] >= bins[11] and rolmean_short_RRIvar[i] < bins[12]:
#         pro_RRIvars_time_false.append(0.033329904)
#         pro_RRIvars_time.append(0.125)
#     elif rolmean_short_RRIvar[i] >= bins[12] and rolmean_short_RRIvar[i] < bins[13]:
#         pro_RRIvars_time_false.append(0.000565785)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[13] and rolmean_short_RRIvar[i] < bins[14]:
#         pro_RRIvars_time_false.append(0.001183006)
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
#         pro_RRIautos_time_false.append(0.201882522)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] > bins[8] and rolmean_short_RRIauto[i] <= bins[9]:
#         pro_RRIautos_time_false.append(0.362308404)
#         pro_RRIautos_time.append(0.625)
#     elif rolmean_short_RRIauto[i] > bins[9] and rolmean_short_RRIauto[i] <= bins[10]:
#         pro_RRIautos_time_false.append(0.318434317)
#         pro_RRIautos_time.append(0.375)
#     elif rolmean_short_RRIauto[i] > bins[10] and rolmean_short_RRIauto[i] < bins[11]:
#         pro_RRIautos_time_false.append(0.056064191)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[11] and rolmean_short_RRIauto[i] < bins[12]:
#         pro_RRIautos_time_false.append(0.017847958)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[12] and rolmean_short_RRIauto[i] < bins[13]:
#         pro_RRIautos_time_false.append(0.016664952)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[13] and rolmean_short_RRIauto[i] < bins[14]:
#         pro_RRIautos_time_false.append(0.026797655)
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




# ##### 12h 12h 12h $$$
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
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[5] and rolmean_short_RRIvar[i] < bins[6]:
#         pro_RRIvars_time_false.append(0.002828927)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[6] and rolmean_short_RRIvar[i] < bins[7]:
#         pro_RRIvars_time_false.append(0.011470013)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[7] and rolmean_short_RRIvar[i] <= bins[8]:
#         pro_RRIvars_time_false.append(0.269365292)
#         pro_RRIvars_time.append(0.5)
#     elif rolmean_short_RRIvar[i] > bins[8] and rolmean_short_RRIvar[i] <= bins[9]:
#         pro_RRIvars_time_false.append(0.234338031)
#         pro_RRIvars_time.append(0.125)
#     elif rolmean_short_RRIvar[i] > bins[9] and rolmean_short_RRIvar[i] <= bins[10]:
#         pro_RRIvars_time_false.append(0.306655694)
#         pro_RRIvars_time.append(0.375)
#     elif rolmean_short_RRIvar[i] > bins[10] and rolmean_short_RRIvar[i] < bins[11]:
#         pro_RRIvars_time_false.append(0.084764942)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[11] and rolmean_short_RRIvar[i] < bins[12]:
#         pro_RRIvars_time_false.append(0.058893118)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[12] and rolmean_short_RRIvar[i] < bins[13]:
#         pro_RRIvars_time_false.append(0.031221068)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[13] and rolmean_short_RRIvar[i] < bins[14]:
#         pro_RRIvars_time_false.append(0.000462915)
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
#         pro_RRIautos_time_false.append(0.092223022)
#         pro_RRIautos_time.append(0.125)
#     elif rolmean_short_RRIauto[i] > bins[8] and rolmean_short_RRIauto[i] <= bins[9]:
#         pro_RRIautos_time_false.append(0.443575764)
#         pro_RRIautos_time.append(0.5)
#     elif rolmean_short_RRIauto[i] > bins[9] and rolmean_short_RRIauto[i] <= bins[10]:
#         pro_RRIautos_time_false.append(0.402376299)
#         pro_RRIautos_time.append(0.375)
#     elif rolmean_short_RRIauto[i] > bins[10] and rolmean_short_RRIauto[i] < bins[11]:
#         pro_RRIautos_time_false.append(0.027209135)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[11] and rolmean_short_RRIauto[i] < bins[12]:
#         pro_RRIautos_time_false.append(0.009566917)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[12] and rolmean_short_RRIauto[i] < bins[13]:
#         pro_RRIautos_time_false.append(0.008075301)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[13] and rolmean_short_RRIauto[i] < bins[14]:
#         pro_RRIautos_time_false.append(0.016973562)
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


# ##### 6h 6h 6h $$$
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
#         pro_eegvars_time_false.append(0.017539348)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[5] and rolmean_short_EEGvar[i] < bins[6]:
#         pro_eegvars_time_false.append(0.142989404)
#         pro_eegvars_time.append(0.125)
#     elif rolmean_short_EEGvar[i] >= bins[6] and rolmean_short_EEGvar[i] <= bins[7]:
#         pro_eegvars_time_false.append(0.188766588)
#         pro_eegvars_time.append(0.25)
#     elif rolmean_short_EEGvar[i] > bins[7] and rolmean_short_EEGvar[i] < bins[8]:
#         pro_eegvars_time_false.append(0.093868944)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[8] and rolmean_short_EEGvar[i] <= bins[9]:
#         pro_eegvars_time_false.append(0.053080959)
#         pro_eegvars_time.append(0.125)
#     elif rolmean_short_EEGvar[i] > bins[9] and rolmean_short_EEGvar[i] < bins[10]:
#         pro_eegvars_time_false.append(0.100195453)
#         pro_eegvars_time.append(0.125)
#     elif rolmean_short_EEGvar[i] >= bins[10] and rolmean_short_EEGvar[i] <= bins[11]:
#         pro_eegvars_time_false.append(0.126221582)
#         pro_eegvars_time.append(0.125)
#     elif rolmean_short_EEGvar[i] > bins[11] and rolmean_short_EEGvar[i] < bins[12]:
#         pro_eegvars_time_false.append(0.064756712)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] > bins[12] and rolmean_short_EEGvar[i] < bins[13]:
#         pro_eegvars_time_false.append(0.151630491)
#         pro_eegvars_time.append(0.125)
#     elif rolmean_short_EEGvar[i] >= bins[13] and rolmean_short_EEGvar[i] < bins[14]:
#         pro_eegvars_time_false.append(0.060950519)
#         pro_eegvars_time.append(0.125)
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
#         pro_eegautos_time_false.append(0.001028701)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[6] and rolmean_short_EEGauto[i] < bins[7]:
#         pro_eegautos_time_false.append(0.021808456)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[7] and rolmean_short_EEGauto[i] < bins[8]:
#         pro_eegautos_time_false.append(0.090268491)
#         pro_eegautos_time.append(0.125)
#     elif rolmean_short_EEGauto[i] >= bins[8] and rolmean_short_EEGauto[i] < bins[9]:
#         pro_eegautos_time_false.append(0.343740356)
#         pro_eegautos_time.append(0.375)
#     elif rolmean_short_EEGauto[i] >= bins[9] and rolmean_short_EEGauto[i] <= bins[10]:
#         pro_eegautos_time_false.append(0.432928711)
#         pro_eegautos_time.append(0.5)
#     elif rolmean_short_EEGauto[i] > bins[10] and rolmean_short_EEGauto[i] < bins[11]:
#         pro_eegautos_time_false.append(0.092428762)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[11] and rolmean_short_EEGauto[i] < bins[12]:
#         pro_eegautos_time_false.append(0.005760724)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[12] and rolmean_short_EEGauto[i] < bins[13]:
#         pro_eegautos_time_false.append(0.004629153)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[13] and rolmean_short_EEGauto[i] < bins[14]:
#         pro_eegautos_time_false.append(0.007406645)
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
#         pro_RRIvars_time_false.append(0.001491616)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[5] and rolmean_short_RRIvar[i] < bins[6]:
#         pro_RRIvars_time_false.append(0.005863594)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[6] and rolmean_short_RRIvar[i] < bins[7]:
#         pro_RRIvars_time_false.append(0.085742208)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[7] and rolmean_short_RRIvar[i] <= bins[8]:
#         pro_RRIvars_time_false.append(0.284590063)
#         pro_RRIvars_time.append(0.375)
#     elif rolmean_short_RRIvar[i] > bins[8] and rolmean_short_RRIvar[i] < bins[9]:
#         pro_RRIvars_time_false.append(0.134759798)
#         pro_RRIvars_time.append(0.125)
#     elif rolmean_short_RRIvar[i] >= bins[9] and rolmean_short_RRIvar[i] <= bins[10]:
#         pro_RRIvars_time_false.append(0.215512807)
#         pro_RRIvars_time.append(0.125)
#     elif rolmean_short_RRIvar[i] > bins[10] and rolmean_short_RRIvar[i] <= bins[11]:
#         pro_RRIvars_time_false.append(0.148647259)
#         pro_RRIvars_time.append(0.25)
#     elif rolmean_short_RRIvar[i] > bins[11] and rolmean_short_RRIvar[i] < bins[12]:
#         pro_RRIvars_time_false.append(0.039244934)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[12] and rolmean_short_RRIvar[i] < bins[13]:
#         pro_RRIvars_time_false.append(0.084147721)
#         pro_RRIvars_time.append(0.125)
#     elif rolmean_short_RRIvar[i] >= bins[13] and rolmean_short_RRIvar[i] < bins[14]:
#         pro_RRIvars_time_false.append(0)
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
#         pro_RRIautos_time_false.append(0.051435038)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] > bins[8] and rolmean_short_RRIauto[i] <= bins[9]:
#         pro_RRIautos_time_false.append(0.47263656)
#         pro_RRIautos_time.append(0.5)
#     elif rolmean_short_RRIauto[i] > bins[9] and rolmean_short_RRIauto[i] <= bins[10]:
#         pro_RRIautos_time_false.append(0.436889209)
#         pro_RRIautos_time.append(0.5)
#     elif rolmean_short_RRIauto[i] > bins[10] and rolmean_short_RRIauto[i] < bins[11]:
#         pro_RRIautos_time_false.append(0.017487913)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[11] and rolmean_short_RRIauto[i] < bins[12]:
#         pro_RRIautos_time_false.append(0.005760724)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[12] and rolmean_short_RRIauto[i] < bins[13]:
#         pro_RRIautos_time_false.append(0.005040634)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[13] and rolmean_short_RRIauto[i] < bins[14]:
#         pro_RRIautos_time_false.append(0.010749923)
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


##### 1h 1h 1h 1h $$$
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
#         pro_eegvars_time_false.append(0.191801255)
#         pro_eegvars_time.append(0.125)
#     elif rolmean_short_EEGvar[i] >= bins[5] and rolmean_short_EEGvar[i] < bins[6]:
#         pro_eegvars_time_false.append(0.131725131)
#         pro_eegvars_time.append(0.125)
#     elif rolmean_short_EEGvar[i] >= bins[6] and rolmean_short_EEGvar[i] <= bins[7]:
#         pro_eegvars_time_false.append(0.077975517)
#         pro_eegvars_time.append(0.25)
#     elif rolmean_short_EEGvar[i] > bins[7] and rolmean_short_EEGvar[i] < bins[8]:
#         pro_eegvars_time_false.append(0.05503549)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[8] and rolmean_short_EEGvar[i] <= bins[9]:
#         pro_eegvars_time_false.append(0.044028392)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] > bins[9] and rolmean_short_EEGvar[i] < bins[10]:
#         pro_eegvars_time_false.append(0.042228166)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[10] and rolmean_short_EEGvar[i] <= bins[11]:
#         pro_eegvars_time_false.append(0.056475671)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] > bins[11] and rolmean_short_EEGvar[i] < bins[12]:
#         pro_eegvars_time_false.append(0.066351198)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] > bins[12] and rolmean_short_EEGvar[i] < bins[13]:
#         pro_eegvars_time_false.append(0.148081473)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[13] and rolmean_short_EEGvar[i] < bins[14]:
#         pro_eegvars_time_false.append(0.186297706)
#         pro_eegvars_time.append(0.5)
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
#         pro_RRIvars_time_false.append(0.012910194)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[5] and rolmean_short_RRIvar[i] < bins[6]:
#         pro_RRIvars_time_false.append(0.063316531)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[6] and rolmean_short_RRIvar[i] < bins[7]:
#         pro_RRIvars_time_false.append(0.138514556)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[7] and rolmean_short_RRIvar[i] <= bins[8]:
#         pro_RRIvars_time_false.append(0.170249974)
#         pro_RRIvars_time.append(0.25)
#     elif rolmean_short_RRIvar[i] > bins[8] and rolmean_short_RRIvar[i] < bins[9]:
#         pro_RRIvars_time_false.append(0.125912972)
#         pro_RRIvars_time.append(0.375)
#     elif rolmean_short_RRIvar[i] >= bins[9] and rolmean_short_RRIvar[i] <= bins[10]:
#         pro_RRIvars_time_false.append(0.148852999)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] > bins[10] and rolmean_short_RRIvar[i] <= bins[11]:
#         pro_RRIvars_time_false.append(0.132805267)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] > bins[11] and rolmean_short_RRIvar[i] < bins[12]:
#         pro_RRIvars_time_false.append(0.076792511)
#         pro_RRIvars_time.append(0.25)
#     elif rolmean_short_RRIvar[i] >= bins[12] and rolmean_short_RRIvar[i] < bins[13]:
#         pro_RRIvars_time_false.append(0.121901039)
#         pro_RRIvars_time.append(0.125)
#     elif rolmean_short_RRIvar[i] >= bins[13] and rolmean_short_RRIvar[i] < bins[14]:
#         pro_RRIvars_time_false.append(0.008743956)
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


#### 5min 5min 5min 5min $$
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
        pro_RRIvars_time_false.append(0.062545006)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] >= bins[5] and rolmean_short_RRIvar[i] < bins[6]:
        pro_RRIvars_time_false.append(0.139903302)
        pro_RRIvars_time.append(0.125)
    elif rolmean_short_RRIvar[i] >= bins[6] and rolmean_short_RRIvar[i] < bins[7]:
        pro_RRIvars_time_false.append(0.112745602)
        pro_RRIvars_time.append(0.125)
    elif rolmean_short_RRIvar[i] >= bins[7] and rolmean_short_RRIvar[i] <= bins[8]:
        pro_RRIvars_time_false.append(0.098138052)
        pro_RRIvars_time.append(0.25)
    elif rolmean_short_RRIvar[i] > bins[8] and rolmean_short_RRIvar[i] <= bins[9]:
        pro_RRIvars_time_false.append(0.085536467)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] > bins[9] and rolmean_short_RRIvar[i] <= bins[10]:
        pro_RRIvars_time_false.append(0.086925213)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] > bins[10] and rolmean_short_RRIvar[i] < bins[11]:
        pro_RRIvars_time_false.append(0.09587491)
        pro_RRIvars_time.append(0.125)
    elif rolmean_short_RRIvar[i] >= bins[11] and rolmean_short_RRIvar[i] < bins[12]:
        pro_RRIvars_time_false.append(0.11650036)
        pro_RRIvars_time.append(0.125)
    elif rolmean_short_RRIvar[i] >= bins[12] and rolmean_short_RRIvar[i] < bins[13]:
        pro_RRIvars_time_false.append(0.140777698)
        pro_RRIvars_time.append(0.125)
    elif rolmean_short_RRIvar[i] >= bins[13] and rolmean_short_RRIvar[i] < bins[14]:
        pro_RRIvars_time_false.append(0.06105339)
        pro_RRIvars_time.append(0.125)
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





Pseizureeegvar = 0.000411311;
Pnonseizureeegvar = 0.999588689;

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_RRIvars_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_RRIvars_time_false[m])
#     Pcombined.append(P1/(P1+P2))


# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=Pseizureeegvar*pro_eegvars_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m])
#     Pcombined.append(P1/(P1+P2))


# Pcombined = []
# for m in range(len(pro_RRIvars_time)):
#     P1=pro_RRIvars_time[m]*Pseizureeegvar*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))

Pcombined = []
for m in range(len(pro_RRIvars_time)):
    P1=Pseizureeegvar*pro_RRIvars_time[m]
    P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m])
    Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_RRIautos_time)):
#     P1=Pseizureeegvar*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))


pyplot.figure(figsize=(8, 4))
RRI_timewindow_arr = t
pyplot.plot(RRI_timewindow_arr, Pcombined)
pyplot.annotate('', xy=(97.562222, np.max(Pcombined)), xytext=(97.562222, np.max(Pcombined) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
pyplot.annotate('', xy=(100.63694, np.max(Pcombined)), xytext=(100.63694, np.max(Pcombined) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
pyplot.annotate('', xy=(102.63111, np.max(Pcombined)), xytext=(102.63111, np.max(Pcombined) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
pyplot.annotate('', xy=(127.05055, np.max(Pcombined)), xytext=(127.05055, np.max(Pcombined) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
pyplot.annotate('', xy=(147.38778, np.max(Pcombined)), xytext=(147.38778, np.max(Pcombined) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
pyplot.annotate('', xy=(150.84722, np.max(Pcombined)), xytext=(150.84722, np.max(Pcombined) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
pyplot.annotate('', xy=(152.03055, np.max(Pcombined)), xytext=(152.03055, np.max(Pcombined) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
pyplot.hlines(Th1, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'r')
pyplot.title('Forecast seizures in QLD2307')
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
time_arr=[97.562222,100.63694,102.63111,127.05055,147.38778,150.84722,152.03055]
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
time_arr=[97.562222,100.63694,102.63111,127.05055,147.38778,150.84722,152.03055]
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
time_arr=[97.562222,100.63694,102.63111,127.05055,147.38778,150.84722,152.03055]
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
time_arr=[97.562222,100.63694,102.63111,127.05055,147.38778,150.84722,152.03055]
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
time_arr=[97.562222,100.63694,102.63111,127.05055,147.38778,150.84722,152.03055]
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
#     time_arr = np.random.uniform(low=t_window_arr[19450], high=t_window_arr[-1], size=7)
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
#
#     AUC_cs = auc(np.sort(FPR_arr_CS), np.sort(Sen_arr_CS))
#     # print(AUC_cs)
#     AUC_cs_arr.append(AUC_cs)
#
# # print(AUC_cs_arr)
# # print(time_arr_arr)
#
# np.savetxt("C:/Users/wxiong/Documents/PHD/2011.1/QLD2307/chance/AUC_ECGvar_5min_QLD2307.csv", AUC_cs_arr, delimiter=",", fmt='%s')





t1 = np.linspace(0 + 0.00416667, 0 + 0.00416667 + 0.00416667 * (len(Raw_variance_EEG_arr) - 1),
                len(Raw_variance_EEG_arr))
a = np.where(t1 < 11.8797222 + 0)
t1[0:2851] = t1[0:2851] - 0 + 12.1202778
t1[2851:] = t1[2851:] - 11.8797222 - 0
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
        pro_circadian_time_false.append(0.049377636)
        pro_circadian_time.append(0)
    elif new_arr[i] > bins[1] and new_arr[i] < bins[2]:
        pro_circadian_time_false.append(0.049377636)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[2] and new_arr[i] < bins[3]:
        pro_circadian_time_false.append(0.049377636)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[3] and new_arr[i] < bins[4]:
        pro_circadian_time_false.append(0.049377636)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[4] and new_arr[i] < bins[5]:
        pro_circadian_time_false.append(0.049377636)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[5] and new_arr[i] <= bins[6]:
        pro_circadian_time_false.append(0.049377636)
        pro_circadian_time.append(0)
    elif new_arr[i] > bins[6] and new_arr[i] < bins[7]:
        pro_circadian_time_false.append(0.049377636)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[7] and new_arr[i] <= bins[8]:
        pro_circadian_time_false.append(0.049274766)
        pro_circadian_time.append(0.25)
    elif new_arr[i] > bins[8] and new_arr[i] < bins[9]:
        pro_circadian_time_false.append(0.049377636)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[9] and new_arr[i] < bins[10]:
        pro_circadian_time_false.append(0.064345232)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[10] and new_arr[i] < bins[11]:
        pro_circadian_time_false.append(0.065836848)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[11] and new_arr[i] < bins[12]:
        pro_circadian_time_false.append(0.065682543)
        pro_circadian_time.append(0.375)
    elif new_arr[i] >= bins[12] and new_arr[i] < bins[13]:
        pro_circadian_time_false.append(0.065836848)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[13] and new_arr[i] < bins[14]:
        pro_circadian_time_false.append(0.065836848)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[14] and new_arr[i] < bins[15]:
        pro_circadian_time_false.append(0.065682543)
        pro_circadian_time.append(0.375)
    elif new_arr[i] >= bins[15] and new_arr[i] < bins[16]:
        pro_circadian_time_false.append(0.063728012)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[16] and new_arr[i] < bins[17]:
        pro_circadian_time_false.append(0.049377636)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[17] and new_arr[i] <= bins[18]:
        pro_circadian_time_false.append(0.049377636)
        pro_circadian_time.append(0)

# # RRI_timewindow_arr = t[0:len(pro_circadian_time)]
# # print(RRI_timewindow_arr[-1] - RRI_timewindow_arr[0])
# # pyplot.figure(figsize=(8, 4))
# # pyplot.plot(RRI_timewindow_arr, pro_circadian_time)
# # pyplot.annotate('', xy=(97.562222, np.max(pro_circadian_time)), xytext=(97.562222, np.max(pro_circadian_time) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
# # pyplot.annotate('', xy=(100.63694, np.max(pro_circadian_time)), xytext=(100.63694, np.max(pro_circadian_time) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
# # pyplot.annotate('', xy=(102.63111, np.max(pro_circadian_time)), xytext=(102.63111, np.max(pro_circadian_time) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
# # pyplot.annotate('', xy=(127.05055, np.max(pro_circadian_time)), xytext=(127.05055, np.max(pro_circadian_time) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
# # pyplot.annotate('', xy=(147.38778, np.max(pro_circadian_time)), xytext=(147.38778, np.max(pro_circadian_time) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
# # pyplot.annotate('', xy=(150.84722, np.max(pro_circadian_time)), xytext=(150.84722, np.max(pro_circadian_time) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
# # pyplot.annotate('', xy=(152.03055, np.max(pro_circadian_time)), xytext=(152.03055, np.max(pro_circadian_time) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
# # # pyplot.hlines(0.1, RRI_timewindow_arr[0], RRI_timewindow_arr[-1], 'r')
# # pyplot.title('Forecast seizures in QLD2307')
# # pyplot.xlabel('Time(h)')
# # pyplot.ylabel('Seizure probability')
# # pyplot.show()


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
time_arr=[97.562222,100.63694,102.63111,127.05055,147.38778,150.84722,152.03055]
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
        if item >= 0.3*0.25:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[97.562222,100.63694,102.63111,127.05055,147.38778,150.84722,152.03055]
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
        if item >= 0.6*0.25:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[97.562222,100.63694,102.63111,127.05055,147.38778,150.84722,152.03055]
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
        if item >= 1.2*0.25:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[97.562222,100.63694,102.63111,127.05055,147.38778,150.84722,152.03055]
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
        if item >= 2*0.25:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[97.562222,100.63694,102.63111,127.05055,147.38778,150.84722,152.03055]
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
# np.savetxt("C:/Users/wxiong/Documents/PHD/2011.1/QLD2307/chance/AUC_circadian_QLD2307_2022.csv", AUC_cs_arr, delimiter=",", fmt='%s')



# Pseizureeegvar = 0.000411311;
# Pnonseizureeegvar = 0.999588689;
#
# # Pcombined = []
# # for m in range(len(pro_circadian_time)):
# #     P1 = Pseizureeegvar * pro_eegvars_time[m] *pro_RRIvars_time[m] * pro_circadian_time[m]
# #     P2 = Pnonseizureeegvar * (1 - pro_eegvars_time_false[m] * pro_RRIvars_time_false[m] * pro_circadian_time_false[m])
# #     Pcombined.append(P1 / (P1 + P2))
#
#
# # Pcombined = []
# # for m in range(len(pro_circadian_time)):
# #     P1 = Pseizureeegvar *  pro_eegvars_time[m] * pro_circadian_time[m]
# #     P2 = Pnonseizureeegvar * (1 - pro_eegvars_time_false[m] *pro_circadian_time_false[m])
# #     Pcombined.append(P1 / (P1 + P2))
#
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
#
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
# # pyplot.annotate('', xy=(97.562222, np.max(Pcombined)), xytext=(97.562222, np.max(Pcombined) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
# # pyplot.annotate('', xy=(100.63694, np.max(Pcombined)), xytext=(100.63694, np.max(Pcombined) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
# # pyplot.annotate('', xy=(102.63111, np.max(Pcombined)), xytext=(102.63111, np.max(Pcombined) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
# # pyplot.annotate('', xy=(127.05055, np.max(Pcombined)), xytext=(127.05055, np.max(Pcombined) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
# # pyplot.annotate('', xy=(147.38778, np.max(Pcombined)), xytext=(147.38778, np.max(Pcombined) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
# # pyplot.annotate('', xy=(150.84722, np.max(Pcombined)), xytext=(150.84722, np.max(Pcombined) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
# # pyplot.annotate('', xy=(152.03055, np.max(Pcombined)), xytext=(152.03055, np.max(Pcombined) + 0.000000000001),arrowprops=dict(facecolor='r', shrink=0.05))
# # pyplot.title('Forecast seizures in QLD2307')
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
# # time_arr=[97.562222,100.63694,102.63111,127.05055,147.38778,150.84722,152.03055]
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
# # time_arr=[97.562222,100.63694,102.63111,127.05055,147.38778,150.84722,152.03055]
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
# # time_arr=[97.562222,100.63694,102.63111,127.05055,147.38778,150.84722,152.03055]
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
# # time_arr=[97.562222,100.63694,102.63111,127.05055,147.38778,150.84722,152.03055]
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
# # time_arr=[97.562222,100.63694,102.63111,127.05055,147.38778,150.84722,152.03055]
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
# # np.savetxt("AUC_ECG_var_circa5min_QLD2307.csv", AUC_com_arr_EEGcirca, delimiter=",", fmt='%s')
# # np.savetxt("seizure_labels_ECG_var_circa5min_QLD2307.csv", time_arr_arr_EEGcirca, delimiter=",", fmt='%s')