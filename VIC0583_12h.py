from __future__ import division
import mne
import numpy as np
import scipy.signal
from scipy.signal import butter, lfilter
from matplotlib import pyplot
import math
from scipy.fftpack import fft, ifft
from scipy import signal
from scipy.signal import butter, lfilter,iirfilter,filtfilt
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
    nyq  = fs/2.0
    low  = freq - band/2.0
    high = freq + band/2.0
    low  = low/nyq
    high = high/nyq
    b, a = iirfilter(order, [low, high], rp=ripple, btype='bandstop',analog=False, ftype=filter_type)
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
    weights = (np.ones(window_size))/window_size
    a=np.ones(1)
    return lfilter(weights,a,values)




csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC0583/Cz_EEGvariance_VIC0583_15s_3h.csv',sep=',',header=None)
Raw_variance_EEG= csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC0583/Cz_EEGauto_VIC0583_15s_3h.csv',sep=',',header=None)
Raw_auto_EEG= csv_reader.values

Raw_variance_EEG_arr=[]
for item in Raw_variance_EEG:
    Raw_variance_EEG_arr.append(float(item))
Raw_auto_EEG_arr=[]
for item in Raw_auto_EEG:
    Raw_auto_EEG_arr.append(float(item))
t=np.linspace(0+0.00416667,0+0.00416667+0.00416667*(len(Raw_variance_EEG_arr)-1),len(Raw_variance_EEG_arr))
t_window_arr=t


print(len(t_window_arr));print(t_window_arr[0]);
print(t_window_arr[-1]-t_window_arr[0]);
print(t_window_arr[19450]);print(t_window_arr[19450]-t_window_arr[0]);
print(t_window_arr[-1]-t_window_arr[19450]);

window_time_arr=t_window_arr
# pyplot.plot(window_time_arr,Raw_variance_EEG_arr,'grey',alpha=0.5)
# pyplot.ylabel('Voltage',fontsize=13)
# pyplot.xlabel('Time (hours)',fontsize=13)
# pyplot.title('raw EEG variance in VIC0583',fontsize=13)
# pyplot.show()
var_arr=[0]
for item in Raw_variance_EEG_arr:
    if item<1e-8:
        var_arr.append(item)
    else:
        var_arr.append(var_arr[-1])
Raw_variance_EEG=var_arr
# pyplot.plot(window_time_arr,Raw_variance_EEG,'grey',alpha=0.5)
# pyplot.ylabel('Voltage',fontsize=13)
# pyplot.xlabel('Time (hours)',fontsize=13)
# pyplot.title('EEG variance in VIC0583',fontsize=13)
# pyplot.show()



seizure_timing_index=[]
for k in range(len(window_time_arr)):
    if window_time_arr[k]<4.5183333 and window_time_arr[k+1]>=4.5183333:
        seizure_timing_index.append(k)
    if window_time_arr[k]<18.71833 and window_time_arr[k+1]>=18.71833:
        seizure_timing_index.append(k)
    if window_time_arr[k]<22.51833 and window_time_arr[k+1]>=22.51833:
        seizure_timing_index.append(k)
    if window_time_arr[k]<27.985 and window_time_arr[k+1]>=27.985:
        seizure_timing_index.append(k)
    if window_time_arr[k]<34.91833 and window_time_arr[k+1]>=34.91833:
        seizure_timing_index.append(k)
    if window_time_arr[k]<50.801667 and window_time_arr[k+1]>=50.801667:
        seizure_timing_index.append(k)
    if window_time_arr[k]<56.88611 and window_time_arr[k+1]>=56.88611:
        seizure_timing_index.append(k)
    if window_time_arr[k]<70.98472 and window_time_arr[k+1]>=70.98472:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 75.551667 and window_time_arr[k + 1] >= 75.551667:
        seizure_timing_index.append(k)
    # if window_time_arr[k] < 81.951389 and window_time_arr[k + 1] >= 81.951389:
    #     seizure_timing_index.append(k)
    # if window_time_arr[k] < 101.33667 and window_time_arr[k + 1] >= 101.33667:
    #     seizure_timing_index.append(k)
    # if window_time_arr[k] < 108.53472 and window_time_arr[k + 1] >= 108.53472:
    #     seizure_timing_index.append(k)
    # if window_time_arr[k] < 132.30138 and window_time_arr[k + 1] >= 132.30138:
    #     seizure_timing_index.append(k)
    # if window_time_arr[k] < 146.01833 and window_time_arr[k + 1] >= 146.01833:
    #     seizure_timing_index.append(k)
    # if window_time_arr[k] < 149.87306 and window_time_arr[k + 1] >= 149.87306:
    #     seizure_timing_index.append(k)
print(seizure_timing_index)


# seizure_timing_index=[]
# for k in range(len(window_time_arr)):
#     if window_time_arr[k]<4.5183333 and window_time_arr[k+1]>=4.5183333:
#         seizure_timing_index.append(k)
# print(seizure_timing_index)



# # # # # # ### EEG variance
window_time_arr=t_window_arr[2880:19450]
Raw_variance_EEG=Raw_variance_EEG[2880:19450]
# window_time_arr=t_window_arr
# Raw_variance_EEG=Raw_variance_EEG
# print(window_time_arr[-1]-window_time_arr[0])

long_rhythm_var_arr=movingaverage(Raw_variance_EEG,5760)
medium_rhythm_var_arr=movingaverage(Raw_variance_EEG,240)
medium_rhythm_var_arr_2=movingaverage(Raw_variance_EEG,240*3)
medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG,240*6)
medium_rhythm_var_arr_4=movingaverage(Raw_variance_EEG,240*12)
short_rhythm_var_arr_plot=movingaverage(Raw_variance_EEG,240*6)




long_rhythm_var_arr=short_rhythm_var_arr_plot*(10**12)
var_trans=hilbert(long_rhythm_var_arr)
var_trans_nomal=[]
for m in var_trans:
    var_trans_nomal.append(m/abs(m))
SIvarlong=sum(var_trans_nomal)/len(var_trans_nomal)
print(SIvarlong)
seizure_phase=[]
for item in seizure_timing_index:
     seizure_phase.append(var_trans_nomal[item-2880])
SIvarlongseizure=sum(seizure_phase)/len(seizure_phase)
print(SIvarlongseizure)
var_phase=np.angle(var_trans)
phase_long_EEGvariance_arr=var_phase
seizure_phase_var_long=[]
for item in seizure_timing_index:
    seizure_phase_var_long.append(phase_long_EEGvariance_arr[item-2880])
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
# # fig, ax2 = pyplot.subplots(subplot_kw=params)
# # ax2.bar(bins[:bins_number], nEEGsvarsei/sum(nEEGsvarsei),width=width, color='grey',alpha=0.7,linewidth=2,edgecolor='k')
# # locs, labels = pyplot.yticks([0.1,0.3,0.5],['0.1','0.3','0.5'],fontsize=16)
# # ax2.set_title('EEG variance',fontsize=16)
# # ax2.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',],fontsize=16)
# # pyplot.show()
# fig, ax2 = pyplot.subplots(subplot_kw=params)
# ax2.bar(bins[:bins_number], nEEGsvarsei/sum(nEEGsvar),width=width, color='grey',alpha=0.7,linewidth=2,edgecolor='k')
# # locs, labels = pyplot.yticks([0.1,0.3,0.5],['0.1','0.3','0.5'],fontsize=16)
# locs, labels = pyplot.yticks([0.00010,0.00020],['0.00010','0.00020'],fontsize=12)
# ax2.set_title('EEG variance',fontsize=16)
# ax2.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',],fontsize=16)
# pyplot.show()



# pyplot.plot(t_window_arr,Raw_auto_EEG_arr,'grey',alpha=0.5)
# pyplot.xlabel('Time (hours)',fontsize=13)
# pyplot.title('raw EEG autocorrelation in VIC0583',fontsize=13)
# pyplot.show()
value_arr=[0]
for item in Raw_auto_EEG_arr:
    if item<500:
        value_arr.append(item)
    else:
        value_arr.append(value_arr[-1])
Raw_auto_EEG_arr=value_arr
# pyplot.plot(t_window_arr,Raw_auto_EEG_arr,'grey',alpha=0.5)
# pyplot.xlabel('Time (hours)',fontsize=13)
# pyplot.title('EEG autocorrelation in VIC0583',fontsize=13)
# pyplot.show()

Raw_auto_EEG=Raw_auto_EEG_arr[2880:19450]
window_time_arr=t_window_arr[2880:19450]
# Raw_auto_EEG=Raw_auto_EEG_arr
# window_time_arr=t_window_arr


long_rhythm_value_arr=movingaverage(Raw_auto_EEG,5760)
medium_rhythm_value_arr=movingaverage(Raw_auto_EEG,240)
medium_rhythm_value_arr_2=movingaverage(Raw_auto_EEG,240*3)
medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG,240*6)
medium_rhythm_value_arr_4=movingaverage(Raw_auto_EEG,240*12)
short_rhythm_value_arr_plot=movingaverage(Raw_auto_EEG,240*6)



long_rhythm_value_arr=short_rhythm_value_arr_plot
value_trans=hilbert(long_rhythm_value_arr)
value_trans_nomal=[]
for m in value_trans:
    value_trans_nomal.append(m/abs(m))
SIvaluelong=sum(value_trans_nomal)/len(value_trans_nomal)
print(SIvaluelong)
seizure_phase=[]
for item in seizure_timing_index:
    seizure_phase.append(value_trans_nomal[item-2880])
SIvaluelongseizure=sum(seizure_phase)/len(seizure_phase)
print(SIvaluelongseizure)
value_phase=np.angle(value_trans)
phase_long_EEGauto_arr=value_phase
seizure_phase_value_long=[]
for item in seizure_timing_index:
    seizure_phase_value_long.append(phase_long_EEGauto_arr[item-2880])
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
# params = dict(projection='polar')
# fig, ax2 = pyplot.subplots(subplot_kw=params)
# ax2.bar(bins[:bins_number], nEEGsautosei/sum(nEEGsauto),width=width, color='grey',alpha=0.7,linewidth=2,edgecolor='k')
# # locs, labels = pyplot.yticks([0.1,0.3,0.5],['0.1','0.3','0.5'],fontsize=16)
# locs, labels = pyplot.yticks([0.00010,0.00020],['0.00010','0.00020'],fontsize=12)
# ax2.set_title('EEG autocorrelation',fontsize=16)
# ax2.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',],fontsize=16)
# pyplot.show()








# # ### ECG data
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC0583/RRI_ch31_timewindowarr_VIC0583_15s_3h.csv',sep=',',header=None)
rri_t= csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC0583/RRI_ch31_rawvariance_VIC0583_15s_3h.csv',sep=',',header=None)
RRI_var= csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC0583/RRI_ch31_rawauto_VIC0583_15s_3h.csv',sep=',',header=None)
Raw_auto_RRI31= csv_reader.values

rri_t_arr=[]
for item in rri_t:
    rri_t_arr.append(0+float(item))

Raw_variance_RRI31_arr=[]
for item in RRI_var:
    Raw_variance_RRI31_arr.append(float(item))
Raw_auto_RRI31_arr=[]
for item in Raw_auto_RRI31:
    Raw_auto_RRI31_arr.append(float(item))
print(len(Raw_variance_RRI31_arr))

# pyplot.plot(Raw_variance_RRI31_arr,'grey',alpha=0.5)
# pyplot.xlabel('Time (hours)',fontsize=13)
# pyplot.title('RRI variance in VIC0583',fontsize=13)
# pyplot.show()
#
# pyplot.plot(Raw_auto_RRI31_arr,'grey',alpha=0.5)
# pyplot.xlabel('Time (hours)',fontsize=13)
# pyplot.title('RRI autocorrelation in VIC0583',fontsize=13)
# pyplot.show()


# window_time_arr=t_window_arr
# Raw_variance_RRI31=Raw_variance_RRI31_arr
window_time_arr=t_window_arr[2880:19450]
Raw_variance_RRI31=Raw_variance_RRI31_arr[2880:19450]

long_rhythm_var_arr=movingaverage(Raw_variance_RRI31,5760)
medium_rhythm_var_arr=movingaverage(Raw_variance_RRI31,240)
medium_rhythm_var_arr_2=movingaverage(Raw_variance_RRI31,240*3)
medium_rhythm_var_arr_3=movingaverage(Raw_variance_RRI31,240*6)
medium_rhythm_var_arr_4=movingaverage(Raw_variance_RRI31,240*12)
short_rhythm_var_arr_plot=movingaverage(Raw_variance_RRI31,240*6)



long_rhythm_var_arr=short_rhythm_var_arr_plot
var_trans=hilbert(long_rhythm_var_arr)
var_trans_nomal=[]
for m in var_trans:
    var_trans_nomal.append(m/abs(m))
SIvarlong=sum(var_trans_nomal)/len(var_trans_nomal)
print(SIvarlong)
seizure_phase=[]
for item in seizure_timing_index:
     seizure_phase.append(var_trans_nomal[item-2880])
SIvarlongseizure=sum(seizure_phase)/len(seizure_phase)
print(SIvarlongseizure)
var_phase=np.angle(var_trans)
phase_whole_long=var_phase
seizure_phase_var_long=[]
for item in seizure_timing_index:
    seizure_phase_var_long.append(phase_whole_long[item-2880])
print(seizure_phase_var_long)
n=0
for item in seizure_phase_var_long:
    if item <0:
        n=n+1
print(n/len(seizure_phase_var_long))


bins_number = 18
bins = np.linspace(-np.pi, np.pi, bins_number + 1)
width = 2*np.pi / bins_number
nRRIsvar, _, _ = pyplot.hist(phase_whole_long, bins)
nRRIsvarsei, _, _ = pyplot.hist(seizure_phase_var_long, bins)
print(nRRIsvar)
print(nRRIsvarsei)
params = dict(projection='polar')
# fig, ax2 = pyplot.subplots(subplot_kw=params)
# ax2.bar(bins[:bins_number], nRRIsvarsei/sum(nRRIsvarsei),width=width, color='grey',alpha=0.7,edgecolor='k',linewidth=2)
# ax2.set_title('RRI variance',fontsize=16)
# locs, labels = pyplot.yticks([0.1,0.3,0.5],['0.1','0.3','0.5'],fontsize=16)
# # ax2.set_rlim([0,0.002])
# ax2.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',],fontsize=16)
# pyplot.show()
# fig, ax2 = pyplot.subplots(subplot_kw=params)
# ax2.bar(bins[:bins_number], nRRIsvarsei/sum(nRRIsvar),width=width, color='grey',alpha=0.7,edgecolor='k',linewidth=2)
# ax2.set_title('RRI variance',fontsize=16)
# # locs, labels = pyplot.yticks([0.1,0.3,0.5],['0.1','0.3','0.5'],fontsize=16)
# # ax2.set_rlim([0,0.002])
# locs, labels = pyplot.yticks([0.00010,0.00020],['0.00010','0.00020'],fontsize=12)
# ax2.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',],fontsize=16)
# pyplot.show()



# Raw_auto_RRI31=Raw_auto_RRI31_arr
Raw_auto_RRI31=Raw_auto_RRI31_arr[2880:19450]

long_rhythm_value_arr=movingaverage(Raw_auto_RRI31,5760)
medium_rhythm_value_arr=movingaverage(Raw_auto_RRI31,240)
medium_rhythm_value_arr_2=movingaverage(Raw_auto_RRI31,240*3)
medium_rhythm_value_arr_3=movingaverage(Raw_auto_RRI31,240*6)
medium_rhythm_value_arr_4=movingaverage(Raw_auto_RRI31,240*12)
short_rhythm_value_arr_plot=movingaverage(Raw_auto_RRI31,240*6)


long_rhythm_value_arr=short_rhythm_value_arr_plot
value_trans=hilbert(long_rhythm_value_arr)
value_trans_nomal=[]
for m in value_trans:
    value_trans_nomal.append(m/abs(m))
SIvaluelong=sum(value_trans_nomal)/len(value_trans_nomal)
print(SIvaluelong)
seizure_phase=[]
for item in seizure_timing_index:
    seizure_phase.append(value_trans_nomal[item-2880])
SIvaluelongseizure=sum(seizure_phase)/len(seizure_phase)
print(SIvaluelongseizure)
value_phase=np.angle(value_trans)
phase_whole_value_long=value_phase
seizure_phase_value_long=[]
for item in seizure_timing_index:
    seizure_phase_value_long.append(phase_whole_value_long[item-2880])
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
width = 2*np.pi / bins_number
params = dict(projection='polar')
# fig, ax2 = pyplot.subplots(subplot_kw=params)
# ax2.bar(bins[:bins_number], nRRIsautosei/sum(nRRIsautosei),width=width, color='grey',alpha=0.7,edgecolor='k',linewidth=2)
# ax2.set_title('RRI autocorrelation',fontsize=16)
# locs, labels = pyplot.yticks([0.1,0.45,0.8],['0.1','0.45','0.8'],fontsize=16)
# ax2.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',],fontsize=16)
# pyplot.show()
# fig, ax2 = pyplot.subplots(subplot_kw=params)
# ax2.bar(bins[:bins_number], nRRIsautosei/sum(nRRIsauto),width=width, color='grey',alpha=0.7,edgecolor='k',linewidth=2)
# ax2.set_title('RRI autocorrelation',fontsize=16)
# # locs, labels = pyplot.yticks([0.1,0.3,0.5],['0.1','0.3','0.5'],fontsize=16)
# locs, labels = pyplot.yticks([0.00010,0.00020],['0.00010','0.00020'],fontsize=12)
# ax2.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',],fontsize=16)
# pyplot.show()




t=np.linspace(0+0.00416667,0+0.00416667+0.00416667*(len(Raw_variance_EEG_arr)-1),len(Raw_variance_EEG_arr))
a=np.where(t<11.218333+0)
print(a);print(t[2691]);print(t[2692])
t[0:2692]=t[0:2692]-0+12.7816667
t[2692:]=t[2692:]-11.218333-0
print(t[2692]);print(type(t));print(t[0])

time_feature_arr=[]
for i in range(len(t)):
    if t[i]>24:
        time_feature_arr.append(t[i] - (t[i] // 24) * 24)
    else:
        time_feature_arr.append(t[i])
seizure_time=[time_feature_arr[4491],time_feature_arr[5403],time_feature_arr[6715],time_feature_arr[8379],
time_feature_arr[12191],time_feature_arr[13651],time_feature_arr[17035],time_feature_arr[18131],
]
print(seizure_time)
bins_number = 18
bins = np.linspace(0, 24, bins_number + 1)
nEEGsvar, _, _ = pyplot.hist(time_feature_arr[2280:19450], bins)
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
# fig, ax = pyplot.subplots(subplot_kw=params)
# ax.bar(bins[:bins_number], nEEGsvarsei/sum(nEEGsvarsei),width=width, color='grey',alpha=0.7,edgecolor='k',linewidth=2)
# pyplot.setp(ax.get_yticklabels(), color='k')
# # ax.set_title('seizure timing histogram (SA0124)',fontsize=23)
# ax.set_xticks(np.linspace(0, 2*np.pi, 24, endpoint=False))
# ax.set_xticklabels(['0 am','','','Night','','','6 am','','','Morning','','','12 am','','','Afternoon','','','18 pm','','','Evening','','','24 pm'],fontsize=16)
# # locs, labels = pyplot.xticks([0,3,6,9,12,15,18,21,24],['0','Night','6','Morning','12','Afternoon','18','Evening','24'],fontsize=16)
# locs, labels = pyplot.yticks([0.1,0.3,0.5],['0.1','0.3','0.5'],fontsize=16)
# pyplot.show()

bins_number = 18
bins = np.linspace(0, 24, bins_number + 1)
ntimes, _, _ = pyplot.hist(time_feature_arr[2880:19450], bins)
ntimesei, _, _ = pyplot.hist(seizure_time, bins)
print(ntimes)
print(ntimesei)





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
medium_rhythm_RRIvar_arr_3 = movingaverage(Raw_variance_RRI31,240 * 6)
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


# ### combined probability calculation
# ####  24h 24h 24h 24h
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
#         pro_eegvars_time_false.append(0.123777322)
#         pro_eegvars_time.append(0.125)
#     elif phase_long_EEGvariance_arr[i] > bins[7] and phase_long_EEGvariance_arr[i] < bins[8]:
#         pro_eegvars_time_false.append(0.092259389)
#         pro_eegvars_time.append(0.25)
#     elif phase_long_EEGvariance_arr[i] >= bins[8] and phase_long_EEGvariance_arr[i] <= bins[9]:
#         pro_eegvars_time_false.append(0.273095037)
#         pro_eegvars_time.append(0.125)
#     elif phase_long_EEGvariance_arr[i] > bins[9] and phase_long_EEGvariance_arr[i] < bins[10]:
#         pro_eegvars_time_false.append(0.349414322)
#         pro_eegvars_time.append(0.375)
#     elif phase_long_EEGvariance_arr[i] >= bins[10] and phase_long_EEGvariance_arr[i] <= bins[11]:
#         pro_eegvars_time_false.append(0.050054341)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] > bins[11] and phase_long_EEGvariance_arr[i] < bins[12]:
#         pro_eegvars_time_false.append(0.020468542)
#         pro_eegvars_time.append(0.125)
#     elif phase_long_EEGvariance_arr[i] > bins[12] and phase_long_EEGvariance_arr[i] < bins[13]:
#         pro_eegvars_time_false.append(0.033993479)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[13] and phase_long_EEGvariance_arr[i] < bins[14]:
#         pro_eegvars_time_false.append(0.056937568)
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
# print(pro_eegvars_time[4491-2880]);print(pro_eegvars_time[5403-2880]);
# print(pro_eegvars_time[6715-2880]);print(pro_eegvars_time[8379-2880]);print(pro_eegvars_time[12191-2880]);print(pro_eegvars_time[13651-2880]);print(pro_eegvars_time[17035-2880]);print(pro_eegvars_time[18131-2880]);
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
#         pro_eegautos_time_false.append(0.199070161)
#         pro_eegautos_time.append(0.25)
#     elif phase_long_EEGauto_arr[i] >= bins[8] and phase_long_EEGauto_arr[i] < bins[9]:
#         pro_eegautos_time_false.append(0.441673711)
#         pro_eegautos_time.append(0.5)
#     elif phase_long_EEGauto_arr[i] >= bins[9] and phase_long_EEGauto_arr[i] <= bins[10]:
#         pro_eegautos_time_false.append(0.238618524)
#         pro_eegautos_time.append(0.25)
#     elif phase_long_EEGauto_arr[i] > bins[10] and phase_long_EEGauto_arr[i] < bins[11]:
#         pro_eegautos_time_false.append(0.055669605)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[11] and phase_long_EEGauto_arr[i] < bins[12]:
#         pro_eegautos_time_false.append(0.022219539)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[12] and phase_long_EEGauto_arr[i] < bins[13]:
#         pro_eegautos_time_false.append(0.020408163)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[13] and phase_long_EEGauto_arr[i] < bins[14]:
#         pro_eegautos_time_false.append(0.022340297)
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
# print(pro_eegautos_time[4491-2880]);print(pro_eegautos_time[5403-2880]);
# print(pro_eegautos_time[6715-2880]);print(pro_eegautos_time[8379-2880]);print(pro_eegautos_time[12191-2880]);print(pro_eegautos_time[13651-2880]);print(pro_eegautos_time[17035-2880]);print(pro_eegautos_time[18131-2880]);
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
#         pro_RRIvars_time_false.append(0.060801836)
#         pro_RRIvars_time.append(0.125)
#     elif phase_long_RRIvariance_arr[i] >= bins[7] and phase_long_RRIvariance_arr[i] <= bins[8]:
#         pro_RRIvars_time_false.append(0.183009298)
#         pro_RRIvars_time.append(0.25)
#     elif phase_long_RRIvariance_arr[i] > bins[8] and phase_long_RRIvariance_arr[i] < bins[9]:
#         pro_RRIvars_time_false.append(0.205228837)
#         pro_RRIvars_time.append(0.125)
#     elif phase_long_RRIvariance_arr[i] >= bins[9] and phase_long_RRIvariance_arr[i] <= bins[10]:
#         pro_RRIvars_time_false.append(0.420661756)
#         pro_RRIvars_time.append(0.5)
#     elif phase_long_RRIvariance_arr[i] > bins[10] and phase_long_RRIvariance_arr[i] <= bins[11]:
#         pro_RRIvars_time_false.append(0.066779374)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvariance_arr[i] > bins[11] and phase_long_RRIvariance_arr[i] < bins[12]:
#         pro_RRIvars_time_false.append(0.017449583)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvariance_arr[i] >= bins[12] and phase_long_RRIvariance_arr[i] < bins[13]:
#         pro_RRIvars_time_false.append(0.017208067)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvariance_arr[i] >= bins[13] and phase_long_RRIvariance_arr[i] < bins[14]:
#         pro_RRIvars_time_false.append(0.028861249)
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
# print(pro_RRIvars_time[4491-2880]);print(pro_RRIvars_time[5403-2880]);
# print(pro_RRIvars_time[6715-2880]);print(pro_RRIvars_time[8379-2880]);print(pro_RRIvars_time[12191-2880]);print(pro_RRIvars_time[13651-2880]);print(pro_RRIvars_time[17035-2880]);print(pro_RRIvars_time[18131-2880]);
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
#         pro_RRIautos_time_false.append(0.226240792)
#         pro_RRIautos_time.append(0.25)
#     elif phase_long_RRIauto_arr[i] > bins[8] and phase_long_RRIauto_arr[i] <= bins[9]:
#         pro_RRIautos_time_false.append(0.352433281)
#         pro_RRIautos_time.append(0.5)
#     elif phase_long_RRIauto_arr[i] > bins[9] and phase_long_RRIauto_arr[i] <= bins[10]:
#         pro_RRIautos_time_false.append(0.295133438)
#         pro_RRIautos_time.append(0.25)
#     elif phase_long_RRIauto_arr[i] > bins[10] and phase_long_RRIauto_arr[i] < bins[11]:
#         pro_RRIautos_time_false.append(0.056514914)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[11] and phase_long_RRIauto_arr[i] < bins[12]:
#         pro_RRIautos_time_false.append(0.021676126)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[12] and phase_long_RRIauto_arr[i] < bins[13]:
#         pro_RRIautos_time_false.append(0.019683613)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[13] and phase_long_RRIauto_arr[i] < bins[14]:
#         pro_RRIautos_time_false.append(0.028317836)
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
# print(pro_RRIautos_time[4491-2880]);print(pro_RRIautos_time[5403-2880]);
# print(pro_RRIautos_time[6715-2880]);print(pro_RRIautos_time[8379-2880]);print(pro_RRIautos_time[12191-2880]);print(pro_RRIautos_time[13651-2880]);print(pro_RRIautos_time[17035-2880]);print(pro_RRIautos_time[18131-2880]);



#### combined probability calculation
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
#         pro_eegvars_time_false.append(0.034899167)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] > bins[7] and phase_long_EEGvariance_arr[i] < bins[8]:
#         pro_eegvars_time_false.append(0.248399952)
#         pro_eegvars_time.append(0.375)
#     elif phase_long_EEGvariance_arr[i] >= bins[8] and phase_long_EEGvariance_arr[i] <= bins[9]:
#         pro_eegvars_time_false.append(0.26548726)
#         pro_eegvars_time.append(0.5)
#     elif phase_long_EEGvariance_arr[i] > bins[9] and phase_long_EEGvariance_arr[i] < bins[10]:
#         pro_eegvars_time_false.append(0.24338848)
#         pro_eegvars_time.append(0.125)
#     elif phase_long_EEGvariance_arr[i] >= bins[10] and phase_long_EEGvariance_arr[i] <= bins[11]:
#         pro_eegvars_time_false.append(0.120758363)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] > bins[11] and phase_long_EEGvariance_arr[i] < bins[12]:
#         pro_eegvars_time_false.append(0.019321338)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] > bins[12] and phase_long_EEGvariance_arr[i] < bins[13]:
#         pro_eegvars_time_false.append(0.031457553)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[13] and phase_long_EEGvariance_arr[i] < bins[14]:
#         pro_eegvars_time_false.append(0.036287888)
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
# print(pro_eegvars_time[4491-2880]);print(pro_eegvars_time[5403-2880]);
# print(pro_eegvars_time[6715-2880]);print(pro_eegvars_time[8379-2880]);
# print(pro_eegvars_time[12191-2880]);print(pro_eegvars_time[13651-2880]);
# print(pro_eegvars_time[17035-2880]);print(pro_eegvars_time[18131-2880]);
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
#         pro_eegautos_time_false.append(0.10989011)
#         pro_eegautos_time.append(0.25)
#     elif phase_long_EEGauto_arr[i] >= bins[8] and phase_long_EEGauto_arr[i] < bins[9]:
#         pro_eegautos_time_false.append(0.436118826)
#         pro_eegautos_time.append(0.375)
#     elif phase_long_EEGauto_arr[i] >= bins[9] and phase_long_EEGauto_arr[i] <= bins[10]:
#         pro_eegautos_time_false.append(0.39457795)
#         pro_eegautos_time.append(0.375)
#     elif phase_long_EEGauto_arr[i] > bins[10] and phase_long_EEGauto_arr[i] < bins[11]:
#         pro_eegautos_time_false.append(0.028619732)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[11] and phase_long_EEGauto_arr[i] < bins[12]:
#         pro_eegautos_time_false.append(0.010385219)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[12] and phase_long_EEGauto_arr[i] < bins[13]:
#         pro_eegautos_time_false.append(0.009358773)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[13] and phase_long_EEGauto_arr[i] < bins[14]:
#         pro_eegautos_time_false.append(0.01104939)
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
# print(pro_eegautos_time[4491-2880]);print(pro_eegautos_time[5403-2880]);
# print(pro_eegautos_time[6715-2880]);print(pro_eegautos_time[8379-2880]);
# print(pro_eegautos_time[12191-2880]);print(pro_eegautos_time[13651-2880]);
# print(pro_eegautos_time[17035-2880]);print(pro_eegautos_time[18131-2880]);
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
#         pro_RRIvars_time_false.append(0.052409129)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvariance_arr[i] >= bins[7] and phase_long_RRIvariance_arr[i] <= bins[8]:
#         pro_RRIvars_time_false.append(0.103308779)
#         pro_RRIvars_time.append(0.25)
#     elif phase_long_RRIvariance_arr[i] > bins[8] and phase_long_RRIvariance_arr[i] < bins[9]:
#         pro_RRIvars_time_false.append(0.259690859)
#         pro_RRIvars_time.append(0.375)
#     elif phase_long_RRIvariance_arr[i] >= bins[9] and phase_long_RRIvariance_arr[i] <= bins[10]:
#         pro_RRIvars_time_false.append(0.524513948)
#         pro_RRIvars_time.append(0.375)
#     elif phase_long_RRIvariance_arr[i] > bins[10] and phase_long_RRIvariance_arr[i] <= bins[11]:
#         pro_RRIvars_time_false.append(0.024212052)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvariance_arr[i] > bins[11] and phase_long_RRIvariance_arr[i] < bins[12]:
#         pro_RRIvars_time_false.append(0.009419152)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvariance_arr[i] >= bins[12] and phase_long_RRIvariance_arr[i] < bins[13]:
#         pro_RRIvars_time_false.append(0.009358773)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvariance_arr[i] >= bins[13] and phase_long_RRIvariance_arr[i] < bins[14]:
#         pro_RRIvars_time_false.append(0.017087308)
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
# print(pro_RRIvars_time[4491-2880]);print(pro_RRIvars_time[5403-2880]);
# print(pro_RRIvars_time[6715-2880]);print(pro_RRIvars_time[8379-2880]);
# print(pro_RRIvars_time[12191-2880]);print(pro_RRIvars_time[13651-2880]);
# print(pro_RRIvars_time[17035-2880]);print(pro_RRIvars_time[18131-2880]);
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
#         pro_RRIautos_time_false.append(0.113995894)
#         pro_RRIautos_time.append(0.25)
#     elif phase_long_RRIauto_arr[i] > bins[8] and phase_long_RRIauto_arr[i] <= bins[9]:
#         pro_RRIautos_time_false.append(0.516121241)
#         pro_RRIautos_time.append(0.5)
#     elif phase_long_RRIauto_arr[i] > bins[9] and phase_long_RRIauto_arr[i] <= bins[10]:
#         pro_RRIautos_time_false.append(0.301835527)
#         pro_RRIautos_time.append(0.25)
#     elif phase_long_RRIauto_arr[i] > bins[10] and phase_long_RRIauto_arr[i] < bins[11]:
#         pro_RRIautos_time_false.append(0.031336795)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[11] and phase_long_RRIauto_arr[i] < bins[12]:
#         pro_RRIautos_time_false.append(0.011170149)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[12] and phase_long_RRIauto_arr[i] < bins[13]:
#         pro_RRIautos_time_false.append(0.010264461)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[13] and phase_long_RRIauto_arr[i] < bins[14]:
#         pro_RRIautos_time_false.append(0.015275933)
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
# print(pro_RRIautos_time[4491-2880]);print(pro_RRIautos_time[5403-2880]);
# print(pro_RRIautos_time[6715-2880]);print(pro_RRIautos_time[8379-2880]);
# print(pro_RRIautos_time[12191-2880]);print(pro_RRIautos_time[13651-2880]);
# print(pro_RRIautos_time[17035-2880]);print(pro_RRIautos_time[18131-2880]);


# #### combined probability calculation
### 6h 6h 6h
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
        pro_eegvars_time_false.append(0.004226543)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] >= bins[6] and phase_long_EEGvariance_arr[i] <= bins[7]:
        pro_eegvars_time_false.append(0.098538824)
        pro_eegvars_time.append(0.125)
    elif phase_long_EEGvariance_arr[i] > bins[7] and phase_long_EEGvariance_arr[i] < bins[8]:
        pro_eegvars_time_false.append(0.283057602)
        pro_eegvars_time.append(0.375)
    elif phase_long_EEGvariance_arr[i] >= bins[8] and phase_long_EEGvariance_arr[i] <= bins[9]:
        pro_eegvars_time_false.append(0.159461418)
        pro_eegvars_time.append(0.125)
    elif phase_long_EEGvariance_arr[i] > bins[9] and phase_long_EEGvariance_arr[i] < bins[10]:
        pro_eegvars_time_false.append(0.143762831)
        pro_eegvars_time.append(0.25)
    elif phase_long_EEGvariance_arr[i] >= bins[10] and phase_long_EEGvariance_arr[i] <= bins[11]:
        pro_eegvars_time_false.append(0.174495834)
        pro_eegvars_time.append(0.125)
    elif phase_long_EEGvariance_arr[i] > bins[11] and phase_long_EEGvariance_arr[i] < bins[12]:
        pro_eegvars_time_false.append(0.077224973)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] > bins[12] and phase_long_EEGvariance_arr[i] < bins[13]:
        pro_eegvars_time_false.append(0.034899167)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] >= bins[13] and phase_long_EEGvariance_arr[i] < bins[14]:
        pro_eegvars_time_false.append(0.02433281)
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
print(pro_eegvars_time[4491-2880]);print(pro_eegvars_time[5403-2880]);
print(pro_eegvars_time[6715-2880]);print(pro_eegvars_time[8379-2880]);print(pro_eegvars_time[12191-2880]);print(pro_eegvars_time[13651-2880]);print(pro_eegvars_time[17035-2880]);print(pro_eegvars_time[18131-2880]);
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
        pro_eegautos_time_false.append(0.042325806)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[8] and phase_long_EEGauto_arr[i] < bins[9]:
        pro_eegautos_time_false.append(0.484422171)
        pro_eegautos_time.append(0.625)
    elif phase_long_EEGauto_arr[i] >= bins[9] and phase_long_EEGauto_arr[i] <= bins[10]:
        pro_eegautos_time_false.append(0.436843376)
        pro_eegautos_time.append(0.375)
    elif phase_long_EEGauto_arr[i] > bins[10] and phase_long_EEGauto_arr[i] < bins[11]:
        pro_eegautos_time_false.append(0.020830818)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[11] and phase_long_EEGauto_arr[i] < bins[12]:
        pro_eegautos_time_false.append(0.00519261)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[12] and phase_long_EEGauto_arr[i] < bins[13]:
        pro_eegautos_time_false.append(0.00368313)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[13] and phase_long_EEGauto_arr[i] < bins[14]:
        pro_eegautos_time_false.append(0.006702089)
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
print(pro_eegautos_time[4491-2880]);print(pro_eegautos_time[5403-2880]);
print(pro_eegautos_time[6715-2880]);print(pro_eegautos_time[8379-2880]);
print(pro_eegautos_time[12191-2880]);print(pro_eegautos_time[13651-2880]);
print(pro_eegautos_time[17035-2880]);print(pro_eegautos_time[18131-2880]);
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
        pro_RRIvars_time_false.append(0.000120758)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[7] and phase_long_RRIvariance_arr[i] <= bins[8]:
        pro_RRIvars_time_false.append(0.158797247)
        pro_RRIvars_time.append(0.25)
    elif phase_long_RRIvariance_arr[i] > bins[8] and phase_long_RRIvariance_arr[i] < bins[9]:
        pro_RRIvars_time_false.append(0.298575051)
        pro_RRIvars_time.append(0.125)
    elif phase_long_RRIvariance_arr[i] >= bins[9] and phase_long_RRIvariance_arr[i] <= bins[10]:
        pro_RRIvars_time_false.append(0.499456587)
        pro_RRIvars_time.append(0.625)
    elif phase_long_RRIvariance_arr[i] > bins[10] and phase_long_RRIvariance_arr[i] <= bins[11]:
        pro_RRIvars_time_false.append(0.021978022)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] > bins[11] and phase_long_RRIvariance_arr[i] < bins[12]:
        pro_RRIvars_time_false.append(0.006339814)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[12] and phase_long_RRIvariance_arr[i] < bins[13]:
        pro_RRIvars_time_false.append(0.006098297)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[13] and phase_long_RRIvariance_arr[i] < bins[14]:
        pro_RRIvars_time_false.append(0.008634223)
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
print(pro_RRIvars_time[4491-2880]);print(pro_RRIvars_time[5403-2880]);
print(pro_RRIvars_time[6715-2880]);print(pro_RRIvars_time[8379-2880]);
print(pro_RRIvars_time[12191-2880]);print(pro_RRIvars_time[13651-2880]);
print(pro_RRIvars_time[17035-2880]);print(pro_RRIvars_time[18131-2880]);
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
        pro_RRIautos_time_false.append(0.053556334)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] > bins[8] and phase_long_RRIauto_arr[i] <= bins[9]:
        pro_RRIautos_time_false.append(0.526747977)
        pro_RRIautos_time.append(0.875)
    elif phase_long_RRIauto_arr[i] > bins[9] and phase_long_RRIauto_arr[i] <= bins[10]:
        pro_RRIautos_time_false.append(0.377973675)
        pro_RRIautos_time.append(0.125)
    elif phase_long_RRIauto_arr[i] > bins[10] and phase_long_RRIauto_arr[i] < bins[11]:
        pro_RRIautos_time_false.append(0.021796884)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[11] and phase_long_RRIauto_arr[i] < bins[12]:
        pro_RRIautos_time_false.append(0.006158676)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[12] and phase_long_RRIauto_arr[i] < bins[13]:
        pro_RRIautos_time_false.append(0.006219056)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[13] and phase_long_RRIauto_arr[i] < bins[14]:
        pro_RRIautos_time_false.append(0.007547398)
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
print(pro_RRIautos_time[4491-2880]);print(pro_RRIautos_time[5403-2880]);
print(pro_RRIautos_time[6715-2880]);print(pro_RRIautos_time[8379-2880]);
print(pro_RRIautos_time[12191-2880]);print(pro_RRIautos_time[13651-2880]);
print(pro_RRIautos_time[17035-2880]);print(pro_RRIautos_time[18131-2880]);


# # # #### combined probability calculation
# # #### 1h
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
# #         pro_eegvars_time_false.append(0.000543413)
# #         pro_eegvars_time.append(0)
# #     elif phase_long_EEGvariance_arr[i] >= bins[5] and phase_long_EEGvariance_arr[i] < bins[6]:
# #         pro_eegvars_time_false.append(0.112486415)
# #         pro_eegvars_time.append(0)
# #     elif phase_long_EEGvariance_arr[i] >= bins[6] and phase_long_EEGvariance_arr[i] <= bins[7]:
# #         pro_eegvars_time_false.append(0.137241879)
# #         pro_eegvars_time.append(0.125)
# #     elif phase_long_EEGvariance_arr[i] > bins[7] and phase_long_EEGvariance_arr[i] < bins[8]:
# #         pro_eegvars_time_false.append(0.107233426)
# #         pro_eegvars_time.append(0.125)
# #     elif phase_long_EEGvariance_arr[i] >= bins[8] and phase_long_EEGvariance_arr[i] <= bins[9]:
# #         pro_eegvars_time_false.append(0.141045767)
# #         pro_eegvars_time.append(0.5)
# #     elif phase_long_EEGvariance_arr[i] > bins[9] and phase_long_EEGvariance_arr[i] < bins[10]:
# #         pro_eegvars_time_false.append(0.142253351)
# #         pro_eegvars_time.append(0.125)
# #     elif phase_long_EEGvariance_arr[i] >= bins[10] and phase_long_EEGvariance_arr[i] <= bins[11]:
# #         pro_eegvars_time_false.append(0.147204444)
# #         pro_eegvars_time.append(0)
# #     elif phase_long_EEGvariance_arr[i] > bins[11] and phase_long_EEGvariance_arr[i] < bins[12]:
# #         pro_eegvars_time_false.append(0.085436541)
# #         pro_eegvars_time.append(0)
# #     elif phase_long_EEGvariance_arr[i] > bins[12] and phase_long_EEGvariance_arr[i] < bins[13]:
# #         pro_eegvars_time_false.append(0.115505374)
# #         pro_eegvars_time.append(0)
# #     elif phase_long_EEGvariance_arr[i] >= bins[13] and phase_long_EEGvariance_arr[i] < bins[14]:
# #         pro_eegvars_time_false.append(0.01104939)
# #         pro_eegvars_time.append(0.125)
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
# # print(pro_eegvars_time[4491-2880]);print(pro_eegvars_time[5403-2880]);
# # print(pro_eegvars_time[6715-2880]);print(pro_eegvars_time[8379-2880]);print(pro_eegvars_time[12191-2880]);print(pro_eegvars_time[13651-2880]);print(pro_eegvars_time[17035-2880]);print(pro_eegvars_time[18131-2880]);
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
# #         pro_eegautos_time_false.append(0.004286922)
# #         pro_eegautos_time.append(0)
# #     elif phase_long_EEGauto_arr[i] >= bins[7] and phase_long_EEGauto_arr[i] < bins[8]:
# #         pro_eegautos_time_false.append(0.042567323)
# #         pro_eegautos_time.append(0)
# #     elif phase_long_EEGauto_arr[i] >= bins[8] and phase_long_EEGauto_arr[i] < bins[9]:
# #         pro_eegautos_time_false.append(0.46165922)
# #         pro_eegautos_time.append(0.5)
# #     elif phase_long_EEGauto_arr[i] >= bins[9] and phase_long_EEGauto_arr[i] <= bins[10]:
# #         pro_eegautos_time_false.append(0.457794952)
# #         pro_eegautos_time.append(0.5)
# #     elif phase_long_EEGauto_arr[i] > bins[10] and phase_long_EEGauto_arr[i] < bins[11]:
# #         pro_eegautos_time_false.append(0.031216037)
# #         pro_eegautos_time.append(0)
# #     elif phase_long_EEGauto_arr[i] >= bins[11] and phase_long_EEGauto_arr[i] < bins[12]:
# #         pro_eegautos_time_false.append(0.00072455)
# #         pro_eegautos_time.append(0)
# #     elif phase_long_EEGauto_arr[i] >= bins[12] and phase_long_EEGauto_arr[i] < bins[13]:
# #         pro_eegautos_time_false.append(0.00072455)
# #         pro_eegautos_time.append(0)
# #     elif phase_long_EEGauto_arr[i] >= bins[13] and phase_long_EEGauto_arr[i] < bins[14]:
# #         pro_eegautos_time_false.append(0.001026446)
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
# # print(pro_eegautos_time[4491-2880]);print(pro_eegautos_time[5403-2880]);
# # print(pro_eegautos_time[6715-2880]);print(pro_eegautos_time[8379-2880]);print(pro_eegautos_time[12191-2880]);print(pro_eegautos_time[13651-2880]);print(pro_eegautos_time[17035-2880]);print(pro_eegautos_time[18131-2880]);
#
#
# # # # #### combined probability calculation
# # #### 5 min
# # bins_number = 18
# # bins = np.linspace(-np.pi, np.pi, bins_number + 1)
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
# #         pro_eegautos_time_false.append(0.000181138)
# #         pro_eegautos_time.append(0)
# #     elif phase_long_EEGauto_arr[i] >= bins[6] and phase_long_EEGauto_arr[i] < bins[7]:
# #         pro_eegautos_time_false.append(0.013343799)
# #         pro_eegautos_time.append(0)
# #     elif phase_long_EEGauto_arr[i] >= bins[7] and phase_long_EEGauto_arr[i] < bins[8]:
# #         pro_eegautos_time_false.append(0.105663567)
# #         pro_eegautos_time.append(0)
# #     elif phase_long_EEGauto_arr[i] >= bins[8] and phase_long_EEGauto_arr[i] < bins[9]:
# #         pro_eegautos_time_false.append(0.375437749)
# #         pro_eegautos_time.append(0.5)
# #     elif phase_long_EEGauto_arr[i] >= bins[9] and phase_long_EEGauto_arr[i] <= bins[10]:
# #         pro_eegautos_time_false.append(0.385581452)
# #         pro_eegautos_time.append(0.375)
# #     elif phase_long_EEGauto_arr[i] > bins[10] and phase_long_EEGauto_arr[i] < bins[11]:
# #         pro_eegautos_time_false.append(0.111399589)
# #         pro_eegautos_time.append(0)
# #     elif phase_long_EEGauto_arr[i] >= bins[11] and phase_long_EEGauto_arr[i] < bins[12]:
# #         pro_eegautos_time_false.append(0.007064364)
# #         pro_eegautos_time.append(0.125)
# #     elif phase_long_EEGauto_arr[i] >= bins[12] and phase_long_EEGauto_arr[i] < bins[13]:
# #         pro_eegautos_time_false.append(0.001267963)
# #         pro_eegautos_time.append(0)
# #     elif phase_long_EEGauto_arr[i] >= bins[13] and phase_long_EEGauto_arr[i] < bins[14]:
# #         pro_eegautos_time_false.append(6.03792E-05)
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
# # print(pro_eegautos_time[4491-2880]);print(pro_eegautos_time[5403-2880]);
# # print(pro_eegautos_time[6715-2880]);print(pro_eegautos_time[8379-2880]);print(pro_eegautos_time[12191-2880]);print(pro_eegautos_time[13651-2880]);print(pro_eegautos_time[17035-2880]);print(pro_eegautos_time[18131-2880]);



Pseizureeegvar = 0.0004828;
Pnonseizureeegvar = 0.9995172;

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
#     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined=[]
# for m in range(len(pro_eegvars_time)):
#     P1=Pseizureeegvar*pro_RRIvars_time[m]*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined=[]
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m])
#     Pcombined.append(P1/(P1+P2))

Pcombined=[]
for m in range(len(pro_eegautos_time)):
    P1=Pseizureeegvar*pro_eegautos_time[m]
    P2=Pnonseizureeegvar*(1-pro_eegautos_time_false[m])
    Pcombined.append(P1/(P1+P2))

# Pcombined=[]
# for m in range(len(pro_eegvars_time)):
#     P1=pro_RRIvars_time[m]*Pseizureeegvar
#     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined=[]
# for m in range(len(pro_eegvars_time)):
#     P1=Pseizureeegvar*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))
#
# pyplot.figure(figsize=(12, 5))
# pyplot.plot(window_time_arr, Pcombined)
# pyplot.title('combined probability in VIC0583', fontsize=15)
# pyplot.annotate('', xy=(18.71833, np.max(Pcombined)), xytext=(18.71833, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(22.51833, np.max(Pcombined)), xytext=(22.51833, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(27.985, np.max(Pcombined)), xytext=(27.985, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(34.91833, np.max(Pcombined)), xytext=(34.91833, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(50.801667, np.max(Pcombined)), xytext=(50.801667, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(56.88611, np.max(Pcombined)), xytext=(56.88611, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(70.98472, np.max(Pcombined)), xytext=(70.98472, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(75.551667, np.max(Pcombined)), xytext=(75.551667, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.tight_layout()
# pyplot.xlabel('Time(h)', fontsize=15)
# pyplot.ylabel('seizure probability', fontsize=15)
# pyplot.show()
pro=[]
for item in seizure_timing_index:
    pro.append(float(Pcombined[item-2880]))
    print(Pcombined[item-2880])
print(pro)
Th1=np.min(pro)
print(Th1)


# # t=np.linspace(0+0.00416667,0+0.00416667+0.00416667*(len(Raw_variance_EEG_arr)-1),len(Raw_variance_EEG_arr))
# # a=np.where(t<11.218333+0)
# # t[0:2692]=t[0:2692]-0+12.7816667
# # t[2692:]=t[2692:]-11.218333-0
# # time_feature_arr=[]
# # for i in range(len(t)):
# #     if t[i]>24:
# #         time_feature_arr.append(t[i] - (t[i] // 24) * 24)
# #     else:
# #         time_feature_arr.append(t[i])
# # time_feature_arr=time_feature_arr[2880:19450]
# #
# # bins_number = 18
# # bins = np.linspace(0, 24, bins_number + 1)
# # pro_circadian_time=[]
# # pro_circadian_time_false=[]
# # for i in range(len(time_feature_arr)):
# #     if time_feature_arr[i] >= bins[0] and time_feature_arr[i] <= bins[1]:
# #         pro_circadian_time_false.append(0.046612728)
# #         pro_circadian_time.append(0)
# #     elif time_feature_arr[i] > bins[1] and time_feature_arr[i] < bins[2]:
# #         pro_circadian_time_false.append(0.057964014)
# #         pro_circadian_time.append(0)
# #     elif time_feature_arr[i] >= bins[2] and time_feature_arr[i] < bins[3]:
# #         pro_circadian_time_false.append(0.057964014)
# #         pro_circadian_time.append(0)
# #     elif time_feature_arr[i] >= bins[3] and time_feature_arr[i] < bins[4]:
# #         pro_circadian_time_false.append(0.057964014)
# #         pro_circadian_time.append(0)
# #     elif time_feature_arr[i] >= bins[4] and time_feature_arr[i] < bins[5]:
# #         pro_circadian_time_false.append(0.057964014)
# #         pro_circadian_time.append(0)
# #     elif time_feature_arr[i] >= bins[5] and time_feature_arr[i] <= bins[6]:
# #         pro_circadian_time_false.append(0.057903635)
# #         pro_circadian_time.append(0.125)
# #     elif time_feature_arr[i] > bins[6] and time_feature_arr[i] < bins[7]:
# #         pro_circadian_time_false.append(0.057964014)
# #         pro_circadian_time.append(0)
# #     elif time_feature_arr[i] >= bins[7] and time_feature_arr[i] <= bins[8]:
# #         pro_circadian_time_false.append(0.057964014)
# #         pro_circadian_time.append(0)
# #     elif time_feature_arr[i] > bins[8] and time_feature_arr[i] < bins[9]:
# #         pro_circadian_time_false.append(0.057843256)
# #         pro_circadian_time.append(0.25)
# #     elif time_feature_arr[i] >= bins[9] and time_feature_arr[i] < bins[10]:
# #         pro_circadian_time_false.append(0.057964014)
# #         pro_circadian_time.append(0)
# #     elif time_feature_arr[i] >= bins[10] and time_feature_arr[i] < bins[11]:
# #         pro_circadian_time_false.append(0.057964014)
# #         pro_circadian_time.append(0)
# #     elif time_feature_arr[i] >= bins[11] and time_feature_arr[i] < bins[12]:
# #         pro_circadian_time_false.append(0.057903635)
# #         pro_circadian_time.append(0.125)
# #     elif time_feature_arr[i] >= bins[12] and time_feature_arr[i] < bins[13]:
# #         pro_circadian_time_false.append(0.057843256)
# #         pro_circadian_time.append(0.25)
# #     elif time_feature_arr[i] >= bins[13] and time_feature_arr[i] < bins[14]:
# #         pro_circadian_time_false.append(0.057964014)
# #         pro_circadian_time.append(0)
# #     elif time_feature_arr[i] >= bins[14] and time_feature_arr[i] < bins[15]:
# #         pro_circadian_time_false.append(0.057964014)
# #         pro_circadian_time.append(0)
# #     elif time_feature_arr[i] >= bins[15] and time_feature_arr[i] < bins[16]:
# #         pro_circadian_time_false.append(0.057964014)
# #         pro_circadian_time.append(0)
# #     elif time_feature_arr[i] >= bins[16] and time_feature_arr[i] < bins[17]:
# #         pro_circadian_time_false.append(0.04570704)
# #         pro_circadian_time.append(0.125)
# #     elif time_feature_arr[i] >= bins[17] and time_feature_arr[i] <= bins[18]:
# #         pro_circadian_time_false.append(0.038582297)
# #         pro_circadian_time.append(0.125)
#
# # Pcombined=[]
# # for m in range(len(pro_eegvars_time)):
# #     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_RRIvars_time[m]*pro_RRIautos_time[m]*pro_circadian_time[m]
# #     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
# #     Pcombined.append(P1/(P1+P2))
#
# # Pcombined=[]
# # for m in range(len(pro_eegvars_time)):
# #     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_circadian_time[m]
# #     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_circadian_time_false[m])
# #     Pcombined.append(P1/(P1+P2))
#
# # Pcombined=[]
# # for m in range(len(pro_eegvars_time)):
# #     P1=Pseizureeegvar*pro_RRIvars_time[m]*pro_RRIautos_time[m]*pro_circadian_time[m]
# #     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
# #     Pcombined.append(P1/(P1+P2))
#
# # Pcombined=[]
# # for m in range(len(pro_eegvars_time)):
# #     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_circadian_time[m]
# #     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_circadian_time_false[m])
# #     Pcombined.append(P1/(P1+P2))
#
# # Pcombined=[]
# # for m in range(len(pro_eegautos_time)):
# #     P1=Pseizureeegvar*pro_eegautos_time[m]*pro_circadian_time[m]
# #     P2=Pnonseizureeegvar*(1-pro_eegautos_time_false[m]*pro_circadian_time_false[m])
# #     Pcombined.append(P1/(P1+P2))
#
# # Pcombined=[]
# # for m in range(len(pro_eegvars_time)):
# #     P1=pro_RRIvars_time[m]*Pseizureeegvar*pro_circadian_time[m]
# #     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m]*pro_circadian_time_false[m])
# #     Pcombined.append(P1/(P1+P2))
#
# # Pcombined=[]
# # for m in range(len(pro_eegvars_time)):
# #     P1=Pseizureeegvar*pro_RRIautos_time[m]*pro_circadian_time[m]
# #     P2=Pnonseizureeegvar*(1-pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
# #     Pcombined.append(P1/(P1+P2))
#
#
# # pyplot.figure(figsize=(12, 5))
# # pyplot.plot(window_time_arr, Pcombined)
# # pyplot.title('combined probability in VIC0583', fontsize=15)
# # pyplot.annotate('', xy=(18.71833, np.max(Pcombined)), xytext=(18.71833, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# # pyplot.annotate('', xy=(22.51833, np.max(Pcombined)), xytext=(22.51833, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# # pyplot.annotate('', xy=(27.985, np.max(Pcombined)), xytext=(27.985, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# # pyplot.annotate('', xy=(34.91833, np.max(Pcombined)), xytext=(34.91833, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# # pyplot.annotate('', xy=(50.801667, np.max(Pcombined)), xytext=(50.801667, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# # pyplot.annotate('', xy=(56.88611, np.max(Pcombined)), xytext=(56.88611, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# # pyplot.annotate('', xy=(70.98472, np.max(Pcombined)), xytext=(70.98472, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# # pyplot.annotate('', xy=(75.551667, np.max(Pcombined)), xytext=(75.551667, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# # pyplot.tight_layout()
# # # pyplot.xlim(window_time_arr[0], window_time_arr[-1])
# # pyplot.xlabel('Time(h)', fontsize=15)
# # pyplot.ylabel('seizure probability', fontsize=15)
# # pyplot.show()
# # print(Pcombined); print(len(Pcombined))
# # for item in seizure_timing_index:
# #     print(Pcombined[item-2880])








# ## section 3 froecast
t=np.linspace(0,0+0.00416667*(len(Raw_variance_EEG_arr)-1),len(Raw_variance_EEG_arr))
t_window_arr=t
fore_arr_EEGvars=[]
for k in range(81,82):
    variance_arr = Raw_variance_EEG_arr[2880:(19450+240*k)]
    long_rhythm_var_arr=movingaverage(variance_arr,240*6)
    pyplot.figure(figsize=(6, 3))
    ax = pyplot.subplot(111)
    pyplot.title('EEG variance', fontsize=14)
    pyplot.ylabel('$\mathregular{\u03BCV^2}$', fontsize=14)
    pyplot.xlabel('Time(h)', fontsize=14)
    pyplot.plot(t_window_arr[(240*24+2880):(19450+240*k-3130)], long_rhythm_var_arr[240*24:(30000+240*12)]*10**12,'darkblue' ,label='recorded')
    print(len(long_rhythm_var_arr[240 * 12:]));
    print(len(t_window_arr[(240 * 12 + 2880):(19450 + 240 * k)]))
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC0583/cycles6h_Cz_forecast81hsignal_3hcycle_EEGvar_VIC0583.csv',sep=',',header=None)
forecast_var_EEG= csv_reader.values
forecast_var_EEG_arr=[]
for item in forecast_var_EEG:
    forecast_var_EEG_arr=forecast_var_EEG_arr+list(item*10**12)
t=np.linspace(t_window_arr[19450],t_window_arr[19450]+0.1666667*(len(forecast_var_EEG_arr)-1),len(forecast_var_EEG_arr))
pyplot.plot(t[0:414], forecast_var_EEG_arr[0:414],'k',label='forecasted')
pyplot.legend()
locs, labels = pyplot.yticks([0, 120, 240],fontsize=14)
locs, labels = pyplot.xticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
pyplot.show()

fore_arr_EEGauto=[]
for k in range(81,82):
    auto_arr = Raw_auto_EEG_arr[2880:(19450+240*k)]
    long_rhythm_auto_arr=movingaverage(auto_arr,240*6)
    pyplot.figure(figsize=(6, 3))
    ax = pyplot.subplot(111)
    pyplot.title('EEG autocorrelation', fontsize=14)
    pyplot.xlabel('Time(h)', fontsize=14)
    pyplot.plot(t_window_arr[(240*24+2880):(19450+240*k-3130)], long_rhythm_auto_arr[240*24:(30000+240*12)],'darkblue')
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC0583/cycles6h_Cz_forecast81hsignal_3hcycle_EEGauto_VIC0583.csv',sep=',',header=None)
forecast_auto_EEG= csv_reader.values
forecast_auto_EEG_arr=[]
for item in forecast_auto_EEG:
    forecast_auto_EEG_arr=forecast_auto_EEG_arr+list(item)
t=np.linspace(t_window_arr[19450],t_window_arr[19450]+0.1666667*(len(forecast_auto_EEG_arr)-1),len(forecast_auto_EEG_arr))
pyplot.plot(t[0:414], forecast_auto_EEG_arr[0:414],'k')
# pyplot.legend()
locs, labels = pyplot.yticks([10, 15, 20],fontsize=14)
locs, labels = pyplot.xticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
pyplot.show()

fore_arr_RRIvars=[]
for k in range(81, 82):
    variance_arr = Raw_variance_RRI31_arr[2880:(19450+240*k)]
    long_rhythm_var_arr=movingaverage(variance_arr,240*12)
    pyplot.figure(figsize=(6, 3))
    ax = pyplot.subplot(111)
    pyplot.title('RRI variance', fontsize=14)
    pyplot.ylabel('Second ($\mathregular{s^2}$)', fontsize=14)
    pyplot.xlabel('Time(h)', fontsize=14)
    pyplot.plot(t_window_arr[(240*24+2880):(19450+240*k-3130)], long_rhythm_var_arr[240*24:(30000+240*12)],'darkblue')
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC0583/cycles12h_ch31_forecast81hsignal_3hcycle_RRIvar_VIC0583.csv', sep=',',header=None)
forecast_var_RRI31= csv_reader.values
forecast_var_RRI31_arr=[]
for item in forecast_var_RRI31:
    forecast_var_RRI31_arr=forecast_var_RRI31_arr+list(item)
t=np.linspace(t_window_arr[19450],t_window_arr[19450]+0.1666667*(len(forecast_var_RRI31_arr)-1),len(forecast_var_RRI31_arr))
pyplot.plot(t[0:414], forecast_var_RRI31_arr[0:414],'k')
# pyplot.legend()
locs, labels = pyplot.yticks([0.0008, 0.0013, 0.0018],fontsize=14)
locs, labels = pyplot.xticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
pyplot.ticklabel_format(axis="y", style="sci",scilimits=(0,0))
pyplot.show()

fore_arr_RRIautos=[]
save_data_RRIautos=[]
for k in range(81,82):
    auto_arr = Raw_auto_RRI31_arr[2880:19450+240*k]
    long_rhythm_auto_arr=movingaverage(auto_arr,240*6)
    pyplot.figure(figsize=(6, 3))
    ax = pyplot.subplot(111)
    pyplot.title('RRI autocorrelation', fontsize=14)
    pyplot.xlabel('Time(h)', fontsize=14)
    pyplot.plot(t_window_arr[(240*24+2880):(19450+240*k-3130)], long_rhythm_auto_arr[240*24:(30000+240*12)],'darkblue')
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/VIC0583/cycles6h_forecast81hsignal_3hcycle_RRIauto_VIC0583.csv',sep=',',header=None)
forecast_auto_RRI31= csv_reader.values
forecast_auto_RRI31_arr=[]
for item in forecast_auto_RRI31:
    forecast_auto_RRI31_arr=forecast_auto_RRI31_arr+list(item)
t=np.linspace(t_window_arr[19450],t_window_arr[19450]+0.1666667*(len(forecast_auto_RRI31_arr)-1),len(forecast_auto_RRI31_arr))
pyplot.plot(t[0:414], forecast_auto_RRI31_arr[0:414],'k')
# pyplot.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
locs, labels = pyplot.yticks([2, 3, 4],fontsize=14)
locs, labels = pyplot.xticks(fontsize=14)
pyplot.show()
print(len(forecast_var_EEG_arr));print(len(forecast_auto_EEG_arr));
print(len(forecast_var_RRI31_arr));print(len(forecast_auto_RRI31_arr));



# ### predict, forecast data
var_trans=hilbert(forecast_var_EEG_arr[0:len(forecast_var_EEG_arr)-72])
var_phase=np.angle(var_trans)
rolmean_short_EEGvar=var_phase
var_trans=hilbert(forecast_auto_EEG_arr[0:(len(forecast_auto_EEG_arr)-72)])
var_phase=np.angle(var_trans)
rolmean_short_EEGauto=var_phase
var_trans=hilbert(forecast_var_RRI31_arr[0:(len(forecast_var_RRI31_arr)-72)])
var_phase=np.angle(var_trans)
rolmean_short_RRIvar=var_phase
var_trans=hilbert(forecast_auto_RRI31_arr[0:(len(forecast_auto_RRI31_arr)-72)])
var_phase=np.angle(var_trans)
rolmean_short_RRIauto=var_phase
print(len(rolmean_short_EEGvar))



####  24h 24h 24h 24h
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
#         pro_eegvars_time_false.append(0.123777322)
#         pro_eegvars_time.append(0.125)
#     elif rolmean_short_EEGvar[i] > bins[7] and rolmean_short_EEGvar[i] < bins[8]:
#         pro_eegvars_time_false.append(0.092259389)
#         pro_eegvars_time.append(0.25)
#     elif rolmean_short_EEGvar[i] >= bins[8] and rolmean_short_EEGvar[i] <= bins[9]:
#         pro_eegvars_time_false.append(0.273095037)
#         pro_eegvars_time.append(0.125)
#     elif rolmean_short_EEGvar[i] > bins[9] and rolmean_short_EEGvar[i] < bins[10]:
#         pro_eegvars_time_false.append(0.349414322)
#         pro_eegvars_time.append(0.375)
#     elif rolmean_short_EEGvar[i] >= bins[10] and rolmean_short_EEGvar[i] <= bins[11]:
#         pro_eegvars_time_false.append(0.050054341)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] > bins[11] and rolmean_short_EEGvar[i] < bins[12]:
#         pro_eegvars_time_false.append(0.020468542)
#         pro_eegvars_time.append(0.125)
#     elif rolmean_short_EEGvar[i] > bins[12] and rolmean_short_EEGvar[i] < bins[13]:
#         pro_eegvars_time_false.append(0.033993479)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[13] and rolmean_short_EEGvar[i] < bins[14]:
#         pro_eegvars_time_false.append(0.056937568)
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
#         pro_eegautos_time_false.append(0.199070161)
#         pro_eegautos_time.append(0.25)
#     elif rolmean_short_EEGauto[i] >= bins[8] and rolmean_short_EEGauto[i] < bins[9]:
#         pro_eegautos_time_false.append(0.441673711)
#         pro_eegautos_time.append(0.5)
#     elif rolmean_short_EEGauto[i] >= bins[9] and rolmean_short_EEGauto[i] <= bins[10]:
#         pro_eegautos_time_false.append(0.238618524)
#         pro_eegautos_time.append(0.25)
#     elif rolmean_short_EEGauto[i] > bins[10] and rolmean_short_EEGauto[i] < bins[11]:
#         pro_eegautos_time_false.append(0.055669605)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[11] and rolmean_short_EEGauto[i] < bins[12]:
#         pro_eegautos_time_false.append(0.022219539)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[12] and rolmean_short_EEGauto[i] < bins[13]:
#         pro_eegautos_time_false.append(0.020408163)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[13] and rolmean_short_EEGauto[i] < bins[14]:
#         pro_eegautos_time_false.append(0.022340297)
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
#         pro_RRIvars_time_false.append(0.060801836)
#         pro_RRIvars_time.append(0.125)
#     elif rolmean_short_RRIvar[i] >= bins[7] and rolmean_short_RRIvar[i] <= bins[8]:
#         pro_RRIvars_time_false.append(0.183009298)
#         pro_RRIvars_time.append(0.25)
#     elif rolmean_short_RRIvar[i] > bins[8] and rolmean_short_RRIvar[i] < bins[9]:
#         pro_RRIvars_time_false.append(0.205228837)
#         pro_RRIvars_time.append(0.125)
#     elif rolmean_short_RRIvar[i] >= bins[9] and rolmean_short_RRIvar[i] <= bins[10]:
#         pro_RRIvars_time_false.append(0.420661756)
#         pro_RRIvars_time.append(0.5)
#     elif rolmean_short_RRIvar[i] > bins[10] and rolmean_short_RRIvar[i] <= bins[11]:
#         pro_RRIvars_time_false.append(0.066779374)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] > bins[11] and rolmean_short_RRIvar[i] < bins[12]:
#         pro_RRIvars_time_false.append(0.017449583)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[12] and rolmean_short_RRIvar[i] < bins[13]:
#         pro_RRIvars_time_false.append(0.017208067)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[13] and rolmean_short_RRIvar[i] < bins[14]:
#         pro_RRIvars_time_false.append(0.028861249)
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
#         pro_RRIautos_time_false.append(0.226240792)
#         pro_RRIautos_time.append(0.25)
#     elif rolmean_short_RRIauto[i] > bins[8] and rolmean_short_RRIauto[i] <= bins[9]:
#         pro_RRIautos_time_false.append(0.352433281)
#         pro_RRIautos_time.append(0.5)
#     elif rolmean_short_RRIauto[i] > bins[9] and rolmean_short_RRIauto[i] <= bins[10]:
#         pro_RRIautos_time_false.append(0.295133438)
#         pro_RRIautos_time.append(0.25)
#     elif rolmean_short_RRIauto[i] > bins[10] and rolmean_short_RRIauto[i] < bins[11]:
#         pro_RRIautos_time_false.append(0.056514914)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[11] and rolmean_short_RRIauto[i] < bins[12]:
#         pro_RRIautos_time_false.append(0.021676126)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[12] and rolmean_short_RRIauto[i] < bins[13]:
#         pro_RRIautos_time_false.append(0.019683613)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[13] and rolmean_short_RRIauto[i] < bins[14]:
#         pro_RRIautos_time_false.append(0.028317836)
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


# #### combined probability calculation
# #### 12h 12h 12h 12h
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
#         pro_eegvars_time_false.append(0.034899167)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] > bins[7] and rolmean_short_EEGvar[i] < bins[8]:
#         pro_eegvars_time_false.append(0.248399952)
#         pro_eegvars_time.append(0.375)
#     elif rolmean_short_EEGvar[i] >= bins[8] and rolmean_short_EEGvar[i] <= bins[9]:
#         pro_eegvars_time_false.append(0.26548726)
#         pro_eegvars_time.append(0.5)
#     elif rolmean_short_EEGvar[i] > bins[9] and rolmean_short_EEGvar[i] < bins[10]:
#         pro_eegvars_time_false.append(0.24338848)
#         pro_eegvars_time.append(0.125)
#     elif rolmean_short_EEGvar[i] >= bins[10] and rolmean_short_EEGvar[i] <= bins[11]:
#         pro_eegvars_time_false.append(0.120758363)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] > bins[11] and rolmean_short_EEGvar[i] < bins[12]:
#         pro_eegvars_time_false.append(0.019321338)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] > bins[12] and rolmean_short_EEGvar[i] < bins[13]:
#         pro_eegvars_time_false.append(0.031457553)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[13] and rolmean_short_EEGvar[i] < bins[14]:
#         pro_eegvars_time_false.append(0.036287888)
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
#         pro_eegautos_time_false.append(0.10989011)
#         pro_eegautos_time.append(0.25)
#     elif rolmean_short_EEGauto[i] >= bins[8] and rolmean_short_EEGauto[i] < bins[9]:
#         pro_eegautos_time_false.append(0.436118826)
#         pro_eegautos_time.append(0.375)
#     elif rolmean_short_EEGauto[i] >= bins[9] and rolmean_short_EEGauto[i] <= bins[10]:
#         pro_eegautos_time_false.append(0.39457795)
#         pro_eegautos_time.append(0.375)
#     elif rolmean_short_EEGauto[i] > bins[10] and rolmean_short_EEGauto[i] < bins[11]:
#         pro_eegautos_time_false.append(0.028619732)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[11] and rolmean_short_EEGauto[i] < bins[12]:
#         pro_eegautos_time_false.append(0.010385219)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[12] and rolmean_short_EEGauto[i] < bins[13]:
#         pro_eegautos_time_false.append(0.009358773)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[13] and rolmean_short_EEGauto[i] < bins[14]:
#         pro_eegautos_time_false.append(0.01104939)
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
#         pro_RRIvars_time_false.append(0.052409129)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[7] and rolmean_short_RRIvar[i] <= bins[8]:
#         pro_RRIvars_time_false.append(0.103308779)
#         pro_RRIvars_time.append(0.25)
#     elif rolmean_short_RRIvar[i] > bins[8] and rolmean_short_RRIvar[i] < bins[9]:
#         pro_RRIvars_time_false.append(0.259690859)
#         pro_RRIvars_time.append(0.375)
#     elif rolmean_short_RRIvar[i] >= bins[9] and rolmean_short_RRIvar[i] <= bins[10]:
#         pro_RRIvars_time_false.append(0.524513948)
#         pro_RRIvars_time.append(0.375)
#     elif rolmean_short_RRIvar[i] > bins[10] and rolmean_short_RRIvar[i] <= bins[11]:
#         pro_RRIvars_time_false.append(0.024212052)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] > bins[11] and rolmean_short_RRIvar[i] < bins[12]:
#         pro_RRIvars_time_false.append(0.009419152)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[12] and rolmean_short_RRIvar[i] < bins[13]:
#         pro_RRIvars_time_false.append(0.009358773)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[13] and rolmean_short_RRIvar[i] < bins[14]:
#         pro_RRIvars_time_false.append(0.017087308)
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
#         pro_RRIautos_time_false.append(0.113995894)
#         pro_RRIautos_time.append(0.25)
#     elif rolmean_short_RRIauto[i] > bins[8] and rolmean_short_RRIauto[i] <= bins[9]:
#         pro_RRIautos_time_false.append(0.516121241)
#         pro_RRIautos_time.append(0.5)
#     elif rolmean_short_RRIauto[i] > bins[9] and rolmean_short_RRIauto[i] <= bins[10]:
#         pro_RRIautos_time_false.append(0.301835527)
#         pro_RRIautos_time.append(0.25)
#     elif rolmean_short_RRIauto[i] > bins[10] and rolmean_short_RRIauto[i] < bins[11]:
#         pro_RRIautos_time_false.append(0.031336795)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[11] and rolmean_short_RRIauto[i] < bins[12]:
#         pro_RRIautos_time_false.append(0.011170149)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[12] and rolmean_short_RRIauto[i] < bins[13]:
#         pro_RRIautos_time_false.append(0.010264461)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[13] and rolmean_short_RRIauto[i] < bins[14]:
#         pro_RRIautos_time_false.append(0.015275933)
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

# #### 6h 6h 6h 6h
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
        pro_eegvars_time_false.append(0.004226543)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] >= bins[6] and rolmean_short_EEGvar[i] <= bins[7]:
        pro_eegvars_time_false.append(0.098538824)
        pro_eegvars_time.append(0.125)
    elif rolmean_short_EEGvar[i] > bins[7] and rolmean_short_EEGvar[i] < bins[8]:
        pro_eegvars_time_false.append(0.283057602)
        pro_eegvars_time.append(0.375)
    elif rolmean_short_EEGvar[i] >= bins[8] and rolmean_short_EEGvar[i] <= bins[9]:
        pro_eegvars_time_false.append(0.159461418)
        pro_eegvars_time.append(0.125)
    elif rolmean_short_EEGvar[i] > bins[9] and rolmean_short_EEGvar[i] < bins[10]:
        pro_eegvars_time_false.append(0.143762831)
        pro_eegvars_time.append(0.25)
    elif rolmean_short_EEGvar[i] >= bins[10] and rolmean_short_EEGvar[i] <= bins[11]:
        pro_eegvars_time_false.append(0.174495834)
        pro_eegvars_time.append(0.125)
    elif rolmean_short_EEGvar[i] > bins[11] and rolmean_short_EEGvar[i] < bins[12]:
        pro_eegvars_time_false.append(0.077224973)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] > bins[12] and rolmean_short_EEGvar[i] < bins[13]:
        pro_eegvars_time_false.append(0.034899167)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] >= bins[13] and rolmean_short_EEGvar[i] < bins[14]:
        pro_eegvars_time_false.append(0.02433281)
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
        pro_eegautos_time_false.append(0.042325806)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[8] and rolmean_short_EEGauto[i] < bins[9]:
        pro_eegautos_time_false.append(0.484422171)
        pro_eegautos_time.append(0.625)
    elif rolmean_short_EEGauto[i] >= bins[9] and rolmean_short_EEGauto[i] <= bins[10]:
        pro_eegautos_time_false.append(0.436843376)
        pro_eegautos_time.append(0.375)
    elif rolmean_short_EEGauto[i] > bins[10] and rolmean_short_EEGauto[i] < bins[11]:
        pro_eegautos_time_false.append(0.020830818)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[11] and rolmean_short_EEGauto[i] < bins[12]:
        pro_eegautos_time_false.append(0.00519261)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[12] and rolmean_short_EEGauto[i] < bins[13]:
        pro_eegautos_time_false.append(0.00368313)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[13] and rolmean_short_EEGauto[i] < bins[14]:
        pro_eegautos_time_false.append(0.006702089)
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
        pro_RRIvars_time_false.append(0.000120758)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] >= bins[7] and rolmean_short_RRIvar[i] <= bins[8]:
        pro_RRIvars_time_false.append(0.158797247)
        pro_RRIvars_time.append(0.25)
    elif rolmean_short_RRIvar[i] > bins[8] and rolmean_short_RRIvar[i] < bins[9]:
        pro_RRIvars_time_false.append(0.298575051)
        pro_RRIvars_time.append(0.125)
    elif rolmean_short_RRIvar[i] >= bins[9] and rolmean_short_RRIvar[i] <= bins[10]:
        pro_RRIvars_time_false.append(0.499456587)
        pro_RRIvars_time.append(0.625)
    elif rolmean_short_RRIvar[i] > bins[10] and rolmean_short_RRIvar[i] <= bins[11]:
        pro_RRIvars_time_false.append(0.021978022)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] > bins[11] and rolmean_short_RRIvar[i] < bins[12]:
        pro_RRIvars_time_false.append(0.006339814)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] >= bins[12] and rolmean_short_RRIvar[i] < bins[13]:
        pro_RRIvars_time_false.append(0.006098297)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] >= bins[13] and rolmean_short_RRIvar[i] < bins[14]:
        pro_RRIvars_time_false.append(0.008634223)
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
        pro_RRIautos_time_false.append(0.053556334)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] > bins[8] and rolmean_short_RRIauto[i] <= bins[9]:
        pro_RRIautos_time_false.append(0.526747977)
        pro_RRIautos_time.append(0.875)
    elif rolmean_short_RRIauto[i] > bins[9] and rolmean_short_RRIauto[i] <= bins[10]:
        pro_RRIautos_time_false.append(0.377973675)
        pro_RRIautos_time.append(0.125)
    elif rolmean_short_RRIauto[i] > bins[10] and rolmean_short_RRIauto[i] < bins[11]:
        pro_RRIautos_time_false.append(0.021796884)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[11] and rolmean_short_RRIauto[i] < bins[12]:
        pro_RRIautos_time_false.append(0.006158676)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[12] and rolmean_short_RRIauto[i] < bins[13]:
        pro_RRIautos_time_false.append(0.006219056)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[13] and rolmean_short_RRIauto[i] < bins[14]:
        pro_RRIautos_time_false.append(0.007547398)
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


# # #### 1h
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
# #         pro_eegvars_time_false.append(0.000543413)
# #         pro_eegvars_time.append(0)
# #     elif rolmean_short_EEGvar[i] >= bins[5] and rolmean_short_EEGvar[i] < bins[6]:
# #         pro_eegvars_time_false.append(0.112486415)
# #         pro_eegvars_time.append(0)
# #     elif rolmean_short_EEGvar[i] >= bins[6] and rolmean_short_EEGvar[i] <= bins[7]:
# #         pro_eegvars_time_false.append(0.137241879)
# #         pro_eegvars_time.append(0.125)
# #     elif rolmean_short_EEGvar[i] > bins[7] and rolmean_short_EEGvar[i] < bins[8]:
# #         pro_eegvars_time_false.append(0.107233426)
# #         pro_eegvars_time.append(0.125)
# #     elif rolmean_short_EEGvar[i] >= bins[8] and rolmean_short_EEGvar[i] <= bins[9]:
# #         pro_eegvars_time_false.append(0.141045767)
# #         pro_eegvars_time.append(0.5)
# #     elif rolmean_short_EEGvar[i] > bins[9] and rolmean_short_EEGvar[i] < bins[10]:
# #         pro_eegvars_time_false.append(0.142253351)
# #         pro_eegvars_time.append(0.125)
# #     elif rolmean_short_EEGvar[i] >= bins[10] and rolmean_short_EEGvar[i] <= bins[11]:
# #         pro_eegvars_time_false.append(0.147204444)
# #         pro_eegvars_time.append(0)
# #     elif rolmean_short_EEGvar[i] > bins[11] and rolmean_short_EEGvar[i] < bins[12]:
# #         pro_eegvars_time_false.append(0.085436541)
# #         pro_eegvars_time.append(0)
# #     elif rolmean_short_EEGvar[i] > bins[12] and rolmean_short_EEGvar[i] < bins[13]:
# #         pro_eegvars_time_false.append(0.115505374)
# #         pro_eegvars_time.append(0)
# #     elif rolmean_short_EEGvar[i] >= bins[13] and rolmean_short_EEGvar[i] < bins[14]:
# #         pro_eegvars_time_false.append(0.01104939)
# #         pro_eegvars_time.append(0.125)
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
# #         pro_eegautos_time_false.append(0.004286922)
# #         pro_eegautos_time.append(0)
# #     elif rolmean_short_EEGauto[i] >= bins[7] and rolmean_short_EEGauto[i] < bins[8]:
# #         pro_eegautos_time_false.append(0.042567323)
# #         pro_eegautos_time.append(0)
# #     elif rolmean_short_EEGauto[i] >= bins[8] and rolmean_short_EEGauto[i] < bins[9]:
# #         pro_eegautos_time_false.append(0.46165922)
# #         pro_eegautos_time.append(0.5)
# #     elif rolmean_short_EEGauto[i] >= bins[9] and rolmean_short_EEGauto[i] <= bins[10]:
# #         pro_eegautos_time_false.append(0.457794952)
# #         pro_eegautos_time.append(0.5)
# #     elif rolmean_short_EEGauto[i] > bins[10] and rolmean_short_EEGauto[i] < bins[11]:
# #         pro_eegautos_time_false.append(0.031216037)
# #         pro_eegautos_time.append(0)
# #     elif rolmean_short_EEGauto[i] >= bins[11] and rolmean_short_EEGauto[i] < bins[12]:
# #         pro_eegautos_time_false.append(0.00072455)
# #         pro_eegautos_time.append(0)
# #     elif rolmean_short_EEGauto[i] >= bins[12] and rolmean_short_EEGauto[i] < bins[13]:
# #         pro_eegautos_time_false.append(0.00072455)
# #         pro_eegautos_time.append(0)
# #     elif rolmean_short_EEGauto[i] >= bins[13] and rolmean_short_EEGauto[i] < bins[14]:
# #         pro_eegautos_time_false.append(0.001026446)
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
#
#
# # ##### 5min
# # bins_number = 18
# # bins = np.linspace(-np.pi, np.pi, bins_number + 1)
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
# #         pro_eegautos_time_false.append(0.000181138)
# #         pro_eegautos_time.append(0)
# #     elif rolmean_short_EEGauto[i] >= bins[6] and rolmean_short_EEGauto[i] < bins[7]:
# #         pro_eegautos_time_false.append(0.013343799)
# #         pro_eegautos_time.append(0)
# #     elif rolmean_short_EEGauto[i] >= bins[7] and rolmean_short_EEGauto[i] < bins[8]:
# #         pro_eegautos_time_false.append(0.105663567)
# #         pro_eegautos_time.append(0)
# #     elif rolmean_short_EEGauto[i] >= bins[8] and rolmean_short_EEGauto[i] < bins[9]:
# #         pro_eegautos_time_false.append(0.375437749)
# #         pro_eegautos_time.append(0.5)
# #     elif rolmean_short_EEGauto[i] >= bins[9] and rolmean_short_EEGauto[i] <= bins[10]:
# #         pro_eegautos_time_false.append(0.385581452)
# #         pro_eegautos_time.append(0.375)
# #     elif rolmean_short_EEGauto[i] > bins[10] and rolmean_short_EEGauto[i] < bins[11]:
# #         pro_eegautos_time_false.append(0.111399589)
# #         pro_eegautos_time.append(0)
# #     elif rolmean_short_EEGauto[i] >= bins[11] and rolmean_short_EEGauto[i] < bins[12]:
# #         pro_eegautos_time_false.append(0.007064364)
# #         pro_eegautos_time.append(0.125)
# #     elif rolmean_short_EEGauto[i] >= bins[12] and rolmean_short_EEGauto[i] < bins[13]:
# #         pro_eegautos_time_false.append(0.001267963)
# #         pro_eegautos_time.append(0)
# #     elif rolmean_short_EEGauto[i] >= bins[13] and rolmean_short_EEGauto[i] < bins[14]:
# #         pro_eegautos_time_false.append(6.03792E-05)
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
#



Pseizureeegvar = 0.0004828;
Pnonseizureeegvar = 0.9995172;

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
# # for m in range(475):
#     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_RRIvars_time[m]*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m])
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
#     P1=pro_RRIvars_time[m]*Pseizureeegvar*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m])
#     Pcombined.append(P1/(P1+P2))

Pcombined = []
for m in range(len(pro_eegautos_time)):
    P1=pro_eegautos_time[m]*Pseizureeegvar
    P2=Pnonseizureeegvar*(1-pro_eegautos_time_false[m])
    Pcombined.append(P1/(P1+P2))

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
ax = pyplot.subplot(111)
RRI_timewindow_arr=t[0:414]
print(len(Pcombined))
print(len(RRI_timewindow_arr))
# pyplot.plot(RRI_timewindow_arr[300:],Pcombined[300:])
pyplot.plot(RRI_timewindow_arr,Pcombined)
# pyplot.annotate('',xy=(103.50142,np.max(Pcombined)),xytext=(103.50142,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='k',shrink=0.05))
# pyplot.annotate('',xy=(108.5847,np.max(Pcombined)),xytext=(108.5847,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='k',shrink=0.05))
# pyplot.annotate('',xy=(109.4847,np.max(Pcombined)),xytext=(109.4847,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='k',shrink=0.05))
# pyplot.annotate('',xy=(134.30138,np.max(Pcombined)),xytext=(134.30138,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='k',shrink=0.05))
# pyplot.annotate('',xy=(146.20166,np.max(Pcombined)),xytext=(146.20166,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='k',shrink=0.05))
# pyplot.annotate('',xy=(150,np.max(Pcombined)),xytext=(150,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='k',shrink=0.05))
# pyplot.annotate('',xy=(151.01806,np.max(Pcombined)),xytext=(151.01806,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='k',shrink=0.05))
# pyplot.annotate('',xy=(152.501393,np.max(Pcombined)),xytext=(152.501393,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='k',shrink=0.05))
# pyplot.annotate('',xy=(152.849439,np.max(Pcombined)),xytext=(152.849439,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='k',shrink=0.05))
# pyplot.annotate('',xy=(153.2847,np.max(Pcombined)),xytext=(153.2847,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='k',shrink=0.05))
# pyplot.annotate('',xy=(155.018311,np.max(Pcombined)),xytext=(155.018311,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='k',shrink=0.05))
# pyplot.annotate('',xy=(157.551644,np.max(Pcombined)),xytext=(157.551644,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='k',shrink=0.05))

pyplot.annotate('',xy=(81.951389,np.max(Pcombined)),xytext=(81.951389,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='k',shrink=0.05))
pyplot.annotate('',xy=(101.33667,np.max(Pcombined)),xytext=(101.33667,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='k',shrink=0.05))
pyplot.annotate('',xy=(108.53472,np.max(Pcombined)),xytext=(108.53472,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='k',shrink=0.05))
pyplot.annotate('',xy=(132.30138,np.max(Pcombined)),xytext=(132.30138,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='k',shrink=0.05))
pyplot.annotate('',xy=(146.01833,np.max(Pcombined)),xytext=(146.01833,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='k',shrink=0.05))
pyplot.annotate('',xy=(149.87306,np.max(Pcombined)),xytext=(149.87306,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='k',shrink=0.05))


# pyplot.title('Forecast seizures in VIC0583')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
pyplot.xlabel('Time (hours)',fontsize=16)
pyplot.ylabel('Probability',fontsize=16)
pyplot.ticklabel_format(axis="y", style="sci",scilimits=(0,0))

# locs, labels = pyplot.yticks([0,5E-5,1E-4],fontsize=16)
# locs, labels = pyplot.yticks([0,3E-4,6E-4],fontsize=16)
# locs, labels = pyplot.yticks([0,1E-5,2E-5],fontsize=16)
locs, labels = pyplot.xticks(fontsize=16)
# pyplot.title('EEG variance and autocorrelation forecaster',fontsize=16)
pyplot.title('RRI variance forecaster',fontsize=16)
# pyplot.title('EEG and RRI forecaster',fontsize=16)
pyplot.hlines(Th1, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'r')
pyplot.show()


# Pcombined_X=Pcombined
# Pcombined=split(Pcombined,6)
# print(len(Pcombined))
# index=[]
# for i in range(len(Pcombined)):
#     for item in Pcombined[i]:
#         if item >= Th1:
#             index.append(6*i+0)
# print(RRI_timewindow_arr[index])
# a=np.unique(RRI_timewindow_arr[index])
# print(a); print(len(a))
# time_arr=[81.951389,101.33667,108.53472,132.30138,146.01833,149.87306,]
# k=0
# n_arr=[]
# pretime=[]
# for m in time_arr:
#     for n in a:
#         if m-n<=1 and m-n>=0:
#             k=k+1
#             n_arr.append(n)
#             pretime.append(m - n)
# print(k)
# time_arr=[81.951389,101.33667,108.53472,132.30138,146.01833,149.87306, 103.50142,108.5847,109.4847,109.5014,134.30138,134.55138,146.20166,150,151.01806,152.501393,152.849439,153.2847,155.018311,157.551644]
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
#         if item >= 0.3*Th1:
#             index.append(6*i+0)
# print(RRI_timewindow_arr[index])
# a=np.unique(RRI_timewindow_arr[index])
# print(a); print(len(a))
# time_arr=[81.951389,101.33667,108.53472,132.30138,146.01833,149.87306,]
# k=0
# n_arr=[]
# for m in time_arr:
#     for n in a:
#         if m-n<=1 and m-n>=0:
#             k=k+1
#             n_arr.append(n)
#             pretime.append(m - n)
# print(k)
# time_arr=[81.951389,101.33667,108.53472,132.30138,146.01833,149.87306, 103.50142,108.5847,109.4847,109.5014,134.30138,134.55138,146.20166,150,151.01806,152.501393,152.849439,153.2847,155.018311,157.551644]
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
#         if item >= 0.6*Th1:
#             index.append(6*i+0)
# print(RRI_timewindow_arr[index])
# a=np.unique(RRI_timewindow_arr[index])
# print(a); print(len(a))
# time_arr=[81.951389,101.33667,108.53472,132.30138,146.01833,149.87306,]
# k=0
# n_arr=[]
# for m in time_arr:
#     for n in a:
#         if m-n<=1 and m-n>=0:
#             k=k+1
#             n_arr.append(n)
#             pretime.append(m - n)
# print(k)
# time_arr=[81.951389,101.33667,108.53472,132.30138,146.01833,149.87306, 103.50142,108.5847,109.4847,109.5014,134.30138,134.55138,146.20166,150,151.01806,152.501393,152.849439,153.2847,155.018311,157.551644]
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
#         if item >= 1.2*Th1:
#             index.append(6*i+0)
# print(RRI_timewindow_arr[index])
# a=np.unique(RRI_timewindow_arr[index])
# print(a); print(len(a))
# time_arr=[81.951389,101.33667,108.53472,132.30138,146.01833,149.87306,]
# k=0
# n_arr=[]
# for m in time_arr:
#     for n in a:
#         if m-n<=1 and m-n>=0:
#             k=k+1
#             n_arr.append(n)
#             pretime.append(m - n)
# print(k)
# time_arr=[81.951389,101.33667,108.53472,132.30138,146.01833,149.87306, 103.50142,108.5847,109.4847,109.5014,134.30138,134.55138,146.20166,150,151.01806,152.501393,152.849439,153.2847,155.018311,157.551644]
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
#         if item >= 2*Th1:
#             index.append(6*i+0)
# print(RRI_timewindow_arr[index])
# a=np.unique(RRI_timewindow_arr[index])
# print(a); print(len(a))
# time_arr=[81.951389,101.33667,108.53472,132.30138,146.01833,149.87306,]
# k=0
# n_arr=[]
# for m in time_arr:
#     for n in a:
#         if m-n<=1 and m-n>=0:
#             k=k+1
#             n_arr.append(n)
#             pretime.append(m - n)
# print(k)
# time_arr=[81.951389,101.33667,108.53472,132.30138,146.01833,149.87306, 103.50142,108.5847,109.4847,109.5014,134.30138,134.55138,146.20166,150,151.01806,152.501393,152.849439,153.2847,155.018311,157.551644]
# k=0
# n_arr=[]
# for m in time_arr:
#     for n in a:
#         if m-n<=1 and m-n>=0:
#             k=k+1
#             n_arr.append(n)
# print(k)
# print(pretime)
# print(np.mean(pretime))
#
# Pcombined = split(Pcombined_X, 6)
# print(len(Pcombined))
# time_arr_arr=[]
# AUC_cs_arr=[]
# for i in range(50000):
#     time_arr = np.random.uniform(low=t_window_arr[19450], high=t_window_arr[-1], size=6)
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
# # print(AUC_cs_arr)
# # print(time_arr_arr)
# np.savetxt("C:/Users/wxiong/Documents/PHD/2011.1/VIC0583/chance/AUC_EEGauto_6h_VIC0583_2022.csv", AUC_cs_arr, delimiter=",", fmt='%s')




t1=np.linspace(0,0+0.00416667*(len(Raw_variance_EEG_arr)-1),len(Raw_variance_EEG_arr))
a=np.where(t1<11.218333+0)
t1[0:2692]=t1[0:2692]-0+12.7816667
t1[2692:]=t1[2692:]-11.218333-0
time_feature_arr=[]
for i in range(len(t1)):
    if t1[i]>24:
        time_feature_arr.append(t1[i] - (t1[i] // 24) * 24)
    else:
        time_feature_arr.append(t1[i])

print(len(time_feature_arr))
time_arr=time_feature_arr[19450:]
print(len(time_arr))
new_arr=[]
for j in range(0,414):
    new_arr.append(time_arr[40*j])

bins_number = 18
bins = np.linspace(0, 24, bins_number + 1)
pro_circadian_time=[]
pro_circadian_time_false=[]
for i in range(len(new_arr)):
    if new_arr[i] >= bins[0] and new_arr[i] <= bins[1]:
        pro_circadian_time_false.append(0.046612728)
        pro_circadian_time.append(0)
    elif new_arr[i] > bins[1] and new_arr[i] < bins[2]:
        pro_circadian_time_false.append(0.057964014)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[2] and new_arr[i] < bins[3]:
        pro_circadian_time_false.append(0.057964014)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[3] and new_arr[i] < bins[4]:
        pro_circadian_time_false.append(0.057964014)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[4] and new_arr[i] < bins[5]:
        pro_circadian_time_false.append(0.057964014)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[5] and new_arr[i] <= bins[6]:
        pro_circadian_time_false.append(0.057903635)
        pro_circadian_time.append(0.125)
    elif new_arr[i] > bins[6] and new_arr[i] < bins[7]:
        pro_circadian_time_false.append(0.057964014)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[7] and new_arr[i] <= bins[8]:
        pro_circadian_time_false.append(0.057964014)
        pro_circadian_time.append(0)
    elif new_arr[i] > bins[8] and new_arr[i] < bins[9]:
        pro_circadian_time_false.append(0.057843256)
        pro_circadian_time.append(0.25)
    elif new_arr[i] >= bins[9] and new_arr[i] < bins[10]:
        pro_circadian_time_false.append(0.057964014)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[10] and new_arr[i] < bins[11]:
        pro_circadian_time_false.append(0.057964014)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[11] and new_arr[i] < bins[12]:
        pro_circadian_time_false.append(0.057903635)
        pro_circadian_time.append(0.125)
    elif new_arr[i] >= bins[12] and new_arr[i] < bins[13]:
        pro_circadian_time_false.append(0.057843256)
        pro_circadian_time.append(0.25)
    elif new_arr[i] >= bins[13] and new_arr[i] < bins[14]:
        pro_circadian_time_false.append(0.057964014)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[14] and new_arr[i] < bins[15]:
        pro_circadian_time_false.append(0.057964014)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[15] and new_arr[i] < bins[16]:
        pro_circadian_time_false.append(0.057964014)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[16] and new_arr[i] < bins[17]:
        pro_circadian_time_false.append(0.04570704)
        pro_circadian_time.append(0.125)
    elif new_arr[i] >= bins[17] and new_arr[i] <= bins[18]:
        pro_circadian_time_false.append(0.038582297)
        pro_circadian_time.append(0.125)

# RRI_timewindow_arr=t[0:len(pro_circadian_time)]
# print(RRI_timewindow_arr[-1]-RRI_timewindow_arr[0])
# pyplot.figure(figsize=(8,4))
# pyplot.plot(RRI_timewindow_arr,pro_circadian_time)
# # pyplot.annotate('',xy=(75.551667,np.max(pro_circadian_time)),xytext=(75.551667,np.max(pro_circadian_time)+0.000000000001),arrowprops=dict(facecolor='k',shrink=0.05))
# pyplot.annotate('',xy=(81.951389,np.max(pro_circadian_time)),xytext=(81.951389,np.max(pro_circadian_time)+0.000000000001),arrowprops=dict(facecolor='k',shrink=0.05))
# pyplot.annotate('',xy=(101.33667,np.max(pro_circadian_time)),xytext=(101.33667,np.max(pro_circadian_time)+0.00000000001),arrowprops=dict(facecolor='k',shrink=0.05))
# pyplot.annotate('',xy=(108.53472,np.max(pro_circadian_time)),xytext=(108.53472,np.max(pro_circadian_time)+0.000000000001),arrowprops=dict(facecolor='r',shrink=0.05))
# pyplot.annotate('',xy=(132.30138,np.max(pro_circadian_time)),xytext=(132.30138,np.max(pro_circadian_time)+0.00000000001),arrowprops=dict(facecolor='r',shrink=0.05))
# pyplot.annotate('',xy=(146.01833,np.max(pro_circadian_time)),xytext=(146.01833,np.max(pro_circadian_time)+0.00000000001),arrowprops=dict(facecolor='r',shrink=0.05))
# pyplot.annotate('',xy=(149.87306,np.max(pro_circadian_time)),xytext=(149.87306,np.max(pro_circadian_time)+0.00000000001),arrowprops=dict(facecolor='r',shrink=0.05))
# pyplot.hlines(0.125, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'r')
# # pyplot.hlines(0.11111, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'orange')
# # pyplot.hlines(0.22222, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'orange')
# pyplot.title('Forecast seizures in VIC0583')
# pyplot.xlabel('Time(h)')
# pyplot.ylabel('Seizure probability')
# pyplot.show()

Pcombined=split(pro_circadian_time,6)
print(len(Pcombined))
index=[]
for i in range(len(Pcombined)):
    for item in Pcombined[i]:
        if item >= 0.125:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[81.951389,101.33667,108.53472,132.30138,146.01833,149.87306,]
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
time_arr=[81.951389,101.33667,108.53472,132.30138,146.01833,149.87306, 75.903,77.8849,103.50142,108.5847,109.4847,109.5014,134.30138,134.55138,146.20166,150,151.01806,152.501393,152.849439,153.2847,155.018311,157.551644]
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
        if item >= 0.3*0.125:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[81.951389,101.33667,108.53472,132.30138,146.01833,149.87306,]
k=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k=k+1
            n_arr.append(n)
            pretime.append(m - n)
print(k)
time_arr=[81.951389,101.33667,108.53472,132.30138,146.01833,149.87306, 75.903,77.8849,103.50142,108.5847,109.4847,109.5014,134.30138,134.55138,146.20166,150,151.01806,152.501393,152.849439,153.2847,155.018311,157.551644]
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
        if item >= 0.6*0.125:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[81.951389,101.33667,108.53472,132.30138,146.01833,149.87306,]
k=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k=k+1
            n_arr.append(n)
            pretime.append(m - n)
print(k)
time_arr=[81.951389,101.33667,108.53472,132.30138,146.01833,149.87306, 75.903,77.8849,103.50142,108.5847,109.4847,109.5014,134.30138,134.55138,146.20166,150,151.01806,152.501393,152.849439,153.2847,155.018311,157.551644]
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
        if item >= 1.2*0.125:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[81.951389,101.33667,108.53472,132.30138,146.01833,149.87306,]
k=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k=k+1
            n_arr.append(n)
            pretime.append(m - n)
print(k)
time_arr=[75.551667,81.951389,101.33667,108.53472,132.30138,146.01833,149.87306, 75.903,77.8849,103.50142,108.5847,109.4847,109.5014,134.30138,134.55138,146.20166,150,151.01806,152.501393,152.849439,153.2847,155.018311,157.551644]
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
        if item >= 2*0.125:
            index.append(6*i+0)

print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[81.951389,101.33667,108.53472,132.30138,146.01833,149.87306,]
k=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k=k+1
            n_arr.append(n)
            pretime.append(m - n)
print(k)
time_arr=[81.951389,101.33667,108.53472,132.30138,146.01833,149.87306, 75.903,77.8849,103.50142,108.5847,109.4847,109.5014,134.30138,134.55138,146.20166,150,151.01806,152.501393,152.849439,153.2847,155.018311,157.551644]
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






# # # Pseizureeegvar = 0.0004828;
# # # Pnonseizureeegvar = 0.9995172;
# # #
# # # Pcombined = []
# # # for m in range(len(pro_circadian_time)):
# # #     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_RRIvars_time[m]*pro_RRIautos_time[m]*pro_circadian_time[m]
# # #     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
# # #     Pcombined.append(P1/(P1+P2))
# #
# #
# # # Pcombined = []
# # # for m in range(len(pro_circadian_time)):
# # #     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_circadian_time[m]
# # #     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_circadian_time_false[m])
# # #     Pcombined.append(P1/(P1+P2))
# #
# # # Pcombined = []
# # # for m in range(len(pro_circadian_time)):
# # #     P1=pro_RRIvars_time[m]*Pseizureeegvar*pro_RRIautos_time[m]*pro_circadian_time[m]
# # #     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
# # #     Pcombined.append(P1/(P1+P2))
# #
# # # Pcombined = []
# # # for m in range(len(pro_circadian_time)):
# # #     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_circadian_time[m]
# # #     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_circadian_time_false[m])
# # #     Pcombined.append(P1/(P1+P2))
# #
# # # Pcombined = []
# # # for m in range(len(pro_circadian_time)):
# # #     P1=pro_eegautos_time[m]*Pseizureeegvar*pro_circadian_time[m]
# # #     P2=Pnonseizureeegvar*(1-pro_eegautos_time_false[m]*pro_circadian_time_false[m])
# # #     Pcombined.append(P1/(P1+P2))
# #
# # # Pcombined = []
# # # for m in range(len(pro_circadian_time)):
# # #     P1=Pseizureeegvar*pro_RRIvars_time[m]*pro_circadian_time[m]
# # #     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m]*pro_circadian_time_false[m])
# # #     Pcombined.append(P1/(P1+P2))
# #
# # # Pcombined = []
# # # for m in range(len(pro_circadian_time)):
# # #     P1=Pseizureeegvar*pro_RRIautos_time[m]*pro_circadian_time[m]
# # #     P2=Pnonseizureeegvar*(1-pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
# # #     Pcombined.append(P1/(P1+P2))
# #
# # # pyplot.figure(figsize=(8,4))
# # # RRI_timewindow_arr=t[0:len(pro_circadian_time)]
# # # pyplot.plot(RRI_timewindow_arr,Pcombined)
# # # pyplot.annotate('',xy=(81.951389,np.max(Pcombined)),xytext=(81.951389,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='k',shrink=0.05))
# # # pyplot.annotate('',xy=(101.33667,np.max(Pcombined)),xytext=(101.33667,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='k',shrink=0.05))
# # # pyplot.annotate('',xy=(108.53472,np.max(Pcombined)),xytext=(108.53472,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='r',shrink=0.05))
# # # pyplot.annotate('',xy=(132.30138,np.max(Pcombined)),xytext=(132.30138,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='r',shrink=0.05))
# # # pyplot.annotate('',xy=(146.01833,np.max(Pcombined)),xytext=(146.01833,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='r',shrink=0.05))
# # # pyplot.annotate('',xy=(149.87306,np.max(Pcombined)),xytext=(149.87306,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='r',shrink=0.05))
# # # pyplot.title('Forecast seizures in VIC0583')
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
# # # time_arr=[81.951389,101.33667,108.53472,132.30138,146.01833,149.87306,]
# # # k=0
# # # n_arr=[]
# # # for m in time_arr:
# # #     for n in a:
# # #         if m-n<=1 and m-n>=0:
# # #             k=k+1
# # #             n_arr.append(n)
# # # print(k)
# # # time_arr=[75.551667,81.951389,101.33667,108.53472,132.30138,146.01833,149.87306, 75.903,77.8849,103.50142,108.5847,109.4847,109.5014,134.30138,134.55138,146.20166,150,151.01806,152.501393,152.849439,153.2847,155.018311,157.551644]
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
# # # time_arr=[81.951389,101.33667,108.53472,132.30138,146.01833,149.87306,]
# # # k=0
# # # n_arr=[]
# # # for m in time_arr:
# # #     for n in a:
# # #         if m-n<=1 and m-n>=0:
# # #             k=k+1
# # #             n_arr.append(n)
# # # print(k)
# # # time_arr=[81.951389,101.33667,108.53472,132.30138,146.01833,149.87306, 75.903,77.8849,103.50142,108.5847,109.4847,109.5014,134.30138,134.55138,146.20166,150,151.01806,152.501393,152.849439,153.2847,155.018311,157.551644]
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
# # # time_arr=[81.951389,101.33667,108.53472,132.30138,146.01833,149.87306,]
# # # k=0
# # # n_arr=[]
# # # for m in time_arr:
# # #     for n in a:
# # #         if m-n<=1 and m-n>=0:
# # #             k=k+1
# # #             n_arr.append(n)
# # # print(k)
# # # time_arr=[81.951389,101.33667,108.53472,132.30138,146.01833,149.87306, 75.903,77.8849,103.50142,108.5847,109.4847,109.5014,134.30138,134.55138,146.20166,150,151.01806,152.501393,152.849439,153.2847,155.018311,157.551644]
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
# # # time_arr=[81.951389,101.33667,108.53472,132.30138,146.01833,149.87306,]
# # # k=0
# # # n_arr=[]
# # # for m in time_arr:
# # #     for n in a:
# # #         if m-n<=1 and m-n>=0:
# # #             k=k+1
# # #             n_arr.append(n)
# # # print(k)
# # # time_arr=[81.951389,101.33667,108.53472,132.30138,146.01833,149.87306, 75.903,77.8849,103.50142,108.5847,109.4847,109.5014,134.30138,134.55138,146.20166,150,151.01806,152.501393,152.849439,153.2847,155.018311,157.551644]
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
# # # time_arr=[81.951389,101.33667,108.53472,132.30138,146.01833,149.87306,]
# # # k=0
# # # n_arr=[]
# # # for m in time_arr:
# # #     for n in a:
# # #         if m-n<=1 and m-n>=0:
# # #             k=k+1
# # #             n_arr.append(n)
# # # print(k)
# # # time_arr=[81.951389,101.33667,108.53472,132.30138,146.01833,149.87306, 75.903,77.8849,103.50142,108.5847,109.4847,109.5014,134.30138,134.55138,146.20166,150,151.01806,152.501393,152.849439,153.2847,155.018311,157.551644]
# # # k=0
# # # n_arr=[]
# # # for m in time_arr:
# # #     for n in a:
# # #         if m-n<=1 and m-n>=0:
# # #             k=k+1
# # #             n_arr.append(n)
# # # print(k)