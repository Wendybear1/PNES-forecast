from __future__ import division
import mne
import numpy as np
import scipy.signal
from matplotlib import pyplot
import math
from scipy.fftpack import fft, ifft
from scipy import signal
from scipy.signal import butter, lfilter,iirfilter
from scipy.signal import hilbert
from biosppy.signals import tools
import pandas as pd

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.ar_model import AutoReg
import scipy.stats as stats
from scipy.stats import norm

from statsmodels.tsa.api import SARIMAX
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.arima_model import ARIMA

def movingaverage(values, window_size):
    weights = (np.ones(window_size))/window_size
    a=np.ones(1)
    return lfilter(weights,a,values)




# ###forecast features
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/EEGauto_SA0124_15s_3h.csv',sep=',',header=None)
Raw_variance_EEG= csv_reader.values
Raw_var_EEG_arr=[]
for item in Raw_variance_EEG:
    Raw_var_EEG_arr.append(float(item))

value_arr=[0]
for item in Raw_var_EEG_arr:
    if item<500:
        value_arr.append(item)
    else:
        value_arr.append(value_arr[-1])
Raw_var_EEG_arr=value_arr

fore_arr_EEGvars=[]
save_data_EEGvars=[]
for k in range(27):
    var_arr=Raw_var_EEG_arr[0:(19624+240*3*k)]
    long_rhythm_var_arr=movingaverage(var_arr,240*6)
    long_var_plot = long_rhythm_var_arr[(240*6+240*3*k):(19624+240*3*k)]


    phase_short_EEGvar_arr=long_var_plot
    rolmean_short_EEGvar=phase_short_EEGvar_arr


    target_arr = []
    for i in range(454):
        target_arr.append(rolmean_short_EEGvar[i * 40])
    data = target_arr
    my_order = (1, 1, 1)
    my_seasonal_order = (1, 1, 1, 144)
    model = SARIMAX(data, order=my_order, seasonal_order=my_seasonal_order)
    # model = ARIMA(data, order=my_order)
    model_fit = model.fit()
    fore_arr_EEGvars.append(model_fit.predict(454, 471))
np.savetxt("forecast81hsignal_3hcycle_EEGauto_SA0124_SARIMA.csv", fore_arr_EEGvars, delimiter=",", fmt='%s')




























# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/EEG/EEG_timewindowarr_SA0124_74h_15s.csv')
# eeg_time_var_arr = csv_reader.to_numpy()
#
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/EEG/phase_EEGvar_short_SA0124_74h_15s.csv')
# short_rhythm_var_arr = csv_reader.to_numpy()
#
#
# df=short_rhythm_var_arr
# df_rolling=pd.DataFrame(np.array(short_rhythm_var_arr))
# rolmean_phase_eegvars = df_rolling.rolling(480).mean()
#
# # from statsmodels.tsa.stattools import adfuller
# # result=adfuller(rolmean_phase_eegvars[480:])
# # print(result[0]);print(result[1])
# #
# # pyplot.plot(eeg_time_var_arr,rolmean_phase_eegvars.diff())
# # pyplot.xlabel('Time(h)')
# # pyplot.ylabel('Phases diff')
# # pyplot.show()
# # first_signal=rolmean_phase_eegvars.diff()
# # print(first_signal[480:])
# # pyplot.plot(eeg_time_var_arr,first_signal.diff())
# # pyplot.xlabel('Time(h)')
# # pyplot.ylabel('Phases 2rd diff')
# # pyplot.show()
# # second_signal=first_signal.diff()
# # print(second_signal[481:])
# #
# # from pandas.plotting import autocorrelation_plot
# # autocorrelation_plot(first_signal[480:])
# # pyplot.show()
# # from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
# # plot_acf(first_signal[480:])
# # pyplot.show()
# # ### partial coorelation
# # plot_pacf(first_signal[480:], lags=25)
# # pyplot.show()
#
#
#
#
#
# # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/EEG/phase_EEGauto_short_SA0124_74h_15s.csv')
# # short_rhythm_auto_arr = csv_reader.to_numpy()
# # df=short_rhythm_auto_arr
# # df_rolling=pd.DataFrame(np.array(df))
# # rolmean_phase_eegautos = df_rolling.rolling(480).mean()
# # print(len(rolmean_phase_eegautos))
# #
# # # ## 72h training 1h forecast each time
# # fore_arr_eegvars=[]
# # save_data_eegvars=[]
# #
# # pyplot.figure(figsize=(8,4))
# # for k in range(1):
# #     rolmean_modified=rolmean_phase_eegvars[(479+240*k):(17759+240*k)]
# #     print(rolmean_modified)
# #     rolmean_modified=rolmean_modified.values
# #     print(rolmean_modified)
# #     target_arr=[]
# #     index_arr_eegvar=[]
# #     for i in range(432):
# #         target_arr.append(rolmean_modified[i*40])
# #         index_arr_eegvar.append(479+i*40+240*k)
# #     data=target_arr
# #     print(len(data))
# #     my_order=(2,1,2)
# #     my_seasonal_order=(1,1,1,144)
# #     model=SARIMAX(data, order=my_order,seasonal_order=my_seasonal_order)
# #     # model = ARIMA(data, order=my_order)
# #     model_fit=model.fit()
# #     print(model_fit.summary())
# #     fore_arr_eegvars.append(model_fit.predict(432, 438))
# #     save_data_eegvars.append(model_fit.predict(1, 432))
# # # np.savetxt("forecast48h_eegvar_short_SA0124.csv", fore_arr_eegvars, delimiter=",", fmt='%s')
# # # np.savetxt("forecast48h_eegvar_save_short_SA0124.csv", save_data_eegvars, delimiter=",", fmt='%s')
# # # np.savetxt("forecast48h_eegvar_save_index_short_SA0124.csv", index_arr_eegvar, delimiter=",", fmt='%s')
# # # print(len(fore_arr_eegvars))
# # # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/forecast/forecast_eegvar_short_SA0124.csv')
# # # fore_arr_eegvar=[]
# # # for lines in csv_reader:
# # #     fore_arr_eegvar.append(float(lines))
# # # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/forecast/forecast_eegvar_save_short_SA0124.csv')
# # # fore_arr_eegvar_save=[]
# # # for lines in csv_reader:
# # #     fore_arr_eegvar_save.append(float(lines))
# # # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/forecast/forecast_eegvar_save_index_short_SA0124.csv')
# # # index=csv_reader.to_numpy()
# # # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/forecast/48h/forecast48h_eegvar_short_SA0124.csv')
# # # fore_arr_eegvar=[]
# # # for lines in csv_reader:
# # #     fore_arr_eegvar.append(float(lines))
# # # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/forecast/24h/forecast24h_eegvar_save_short_SA0124.csv')
# # # fore_arr_eegvar_save=[]
# # # for lines in csv_reader:
# # #     fore_arr_eegvar_save.append(float(lines))
# # # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/forecast/24h/forecast24h_eegvar_save_index_short_SA0124.csv')
# # # index=csv_reader.to_numpy()
# # # t=[]
# # # for item in index:
# # #     t.append(int(item))
# # print(fore_arr_eegvars)
# # print(save_data_eegvars)
# # pyplot.figure(figsize=(8, 4))
# # pyplot.plot(eeg_time_var_arr [479:],rolmean_phase_eegvars[479:],label='smoothed phases')
# # pyplot.plot(eeg_time_var_arr [index_arr_eegvar],np.array(save_data_eegvars[0:432]).reshape((432, 1)),c='grey')
# # t=np.linspace(eeg_time_var_arr [-1],eeg_time_var_arr [-1]+0.16666667*(len(fore_arr_eegvars[0])-1),len(fore_arr_eegvars[0]))
# # pyplot.plot(t,fore_arr_eegvars[0],'r',label='forecast phases')
# # pyplot.ylabel('phase')
# # pyplot.title('phases in EEG variance')
# # pyplot.xlabel('time(h)')
# # pyplot.show()
#
#
#
#
#
# # fore_arr_eegautos=[]
# # save_data_eegautos=[]
# # pyplot.figure(figsize=(12,4))
# # for k in range(1):
# #     rolmean_modified=rolmean_phase_eegautos[(479+240*k):(17759+240*k)]
# #     rolmean_modified=rolmean_modified.values
# #     target_arr=[]
# #     index_arr_eegautos=[]
# #     for i in range(432):
# #         target_arr.append(rolmean_modified[i*40])
# #         index_arr_eegautos.append(479+i*40+240*k)
# #     data=target_arr
# #     my_order=(1,1,1)
# #     my_seasonal_order=(1,1,1,144)
# #     model=SARIMAX(data, order=my_order,seasonal_order=my_seasonal_order)
# #     model_fit=model.fit()
# #     yhat=model_fit.predict()
# #     fore_arr_eegautos.append(model_fit.predict(432, 720))
# #     save_data_eegautos.append(model_fit.predict(1, 432))
# # np.savetxt("forecast48h_eegauto_short_SA0124.csv", fore_arr_eegautos, delimiter=",", fmt='%s')
# # np.savetxt("forecast24h_eegauto_save_short_SA0124.csv", save_data_eegautos, delimiter=",", fmt='%s')
# # np.savetxt("forecast24h_eegauto_save_index_short_SA0124.csv", index_arr_eegautos, delimiter=",", fmt='%s')
# # print(len(fore_arr_eegautos))
# # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/forecast/forecast_eegauto_short_SA0124.csv')
# # fore_arr_eegauto=[]
# # for lines in csv_reader:
# #     fore_arr_eegauto.append(float(lines))
# # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/forecast/forecast_eegauto_save_short_SA0124.csv')
# # fore_arr_eegauto_save=[]
# # for lines in csv_reader:
# #     fore_arr_eegauto_save.append(float(lines))
# # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/forecast/forecast_eegauto_save_index_short_SA0124.csv')
# # index=csv_reader.to_numpy()
# # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/forecast/48h/forecast48h_eegauto_short_SA0124.csv')
# # fore_arr_eegauto=[]
# # for lines in csv_reader:
# #     fore_arr_eegauto.append(float(lines))
# # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/forecast/24h/forecast24h_eegauto_save_short_SA0124.csv')
# # fore_arr_eegauto_save=[]
# # for lines in csv_reader:
# #     fore_arr_eegauto_save.append(float(lines))
# # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/forecast/24h/forecast24h_eegauto_save_index_short_SA0124.csv')
# # index=csv_reader.to_numpy()
# # t=[]
# # for item in index:
# #     t.append(int(item))
# # pyplot.figure(figsize=(12, 4))
# # pyplot.plot(eeg_time_var_arr [479:],rolmean_phase_eegautos[479:],c='darkred')
# # # pyplot.plot(eeg_time_var_arr [t],fore_arr_eegauto_save[0:431],c='grey')
# # t=np.linspace(eeg_time_var_arr [-1],eeg_time_var_arr [-1]+0.16666667*(len(fore_arr_eegauto)-1),len(fore_arr_eegauto))
# # pyplot.plot(t,fore_arr_eegauto)
# # pyplot.ylabel('phase')
# # pyplot.title('phases in short cycle EEG auto')
# # pyplot.xlabel('time(h)')
# # pyplot.show()
#
#
#
# # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/ECG/RRI_timewindowarr_SA0124_74h_15s.csv')
# # RRI_time_var_arr = csv_reader.to_numpy()
# # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/ECG/phase_RRI31var_short_SA0124_74h_15s.csv')
# # short_rhythm_RRIvar_arr = csv_reader.to_numpy()
# # df=short_rhythm_RRIvar_arr
# # df_rolling=pd.DataFrame(np.array(df))
# # rolmean_phase_RRIvars = df_rolling.rolling(480).mean()
# # print(len(rolmean_phase_RRIvars))
# # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/ECG/phase_RRI31auto_short_SA0124_74h_15s.csv')
# # short_rhythm_RRIauto_arr = csv_reader.to_numpy()
# # df=short_rhythm_RRIauto_arr
# # df_rolling=pd.DataFrame(np.array(df))
# # rolmean_phase_RRIautos = df_rolling.rolling(480).mean()
# # print(len(rolmean_phase_RRIautos))
# # # # 72h training 1h forecast each time
# # # fore_arr_RRIvars=[]
# # # save_data_RRIvars=[]
# # # pyplot.figure(figsize=(12,4))
# # # for k in range(1):
# # #     rolmean_modified=rolmean_phase_RRIvars[(479+240*k):(17759+240*k)]
# # #     rolmean_modified=rolmean_modified.values
# # #     print(rolmean_modified)
# # #     target_arr=[]
# # #     index_arr_RRIvars=[]
# # #     for i in range(432):
# # #         target_arr.append(rolmean_modified[i*40]) ##10 mins
# # #         index_arr_RRIvars.append(479+i*40+240*k)
# # #     data=target_arr
# # #     my_order=(1,1,1)
# # #     my_seasonal_order=(1,1,1,144)
# # #     model=SARIMAX(data, order=my_order,seasonal_order=my_seasonal_order)
# # #     model_fit=model.fit()
# # #     yhat=model_fit.predict()
# # #     fore_arr_RRIvars.append(model_fit.predict(432, 720))
# # #     save_data_RRIvars.append(model_fit.predict(1, 432))
# # # np.savetxt("forecast48h_RRIvar_short_SA0124.csv", fore_arr_RRIvars, delimiter=",", fmt='%s')
# # # np.savetxt("forecast24h_RRIvar_save_short_SA0124.csv", save_data_RRIvars, delimiter=",", fmt='%s')
# # # np.savetxt("forecast24h_RRIvar_save_index_short_SA0124.csv", index_arr_RRIvars, delimiter=",", fmt='%s')
# # # print(len(fore_arr_RRIvars))
# #
# # # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/forecast/forecast_RRIvar_short_SA0124.csv')
# # # fore_arr_RRIvar=[]
# # # for lines in csv_reader:
# # #     fore_arr_RRIvar.append(float(lines))
# # # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/forecast/forecast_RRIvar_save_short_SA0124.csv')
# # # fore_arr_RRIvar_save=[]
# # # for lines in csv_reader:
# # #     fore_arr_RRIvar_save.append(float(lines))
# # # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/forecast/forecast_RRIvar_save_index_short_SA0124.csv')
# # # index=csv_reader.to_numpy()
# # # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/forecast/48h/forecast48h_RRIvar_short_SA0124.csv')
# # # fore_arr_RRIvar=[]
# # # for lines in csv_reader:
# # #     fore_arr_RRIvar.append(float(lines))
# # # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/forecast/24h/forecast24h_RRIvar_save_short_SA0124.csv')
# # # fore_arr_RRIvar_save=[]
# # # for lines in csv_reader:
# # #     fore_arr_RRIvar_save.append(float(lines))
# # # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/forecast/24h/forecast24h_RRIvar_save_index_short_SA0124.csv')
# # # index=csv_reader.to_numpy()
# # # t=[]
# # # for item in index:
# # #     t.append(int(item))
# # # pyplot.figure(figsize=(12, 4))
# # # pyplot.plot(RRI_time_var_arr [479:],rolmean_phase_RRIvars[479:],c='orange')
# # # pyplot.plot(RRI_time_var_arr [t],fore_arr_RRIvar_save[0:431],c='grey')
# # # t=np.linspace(RRI_time_var_arr[-1],RRI_time_var_arr[-1]+0.16666667*(len(fore_arr_RRIvar)-1),len(fore_arr_RRIvar))
# # # pyplot.plot(t,fore_arr_RRIvar)
# # # pyplot.ylabel('phase')
# # # pyplot.title('phases in short cycle RRI variance')
# # # pyplot.xlabel('time(h)')
# # # pyplot.show()
# #
# #
# #
# #
# # # fore_arr_RRIautos=[]
# # # save_data_RRIautos=[]
# # # pyplot.figure(figsize=(12,4))
# # # for k in range(1):
# # #     rolmean_modified=rolmean_phase_RRIautos[(479+240*k):(17759+240*k)]
# # #     rolmean_modified = rolmean_modified.values
# # #     target_arr=[]
# # #     index_arr_RRIautos=[]
# # #     for i in range(432):
# # #         target_arr.append(rolmean_modified[i*40])
# # #         index_arr_RRIautos.append(479+i*40+240*k)
# # #     data=target_arr
# # #     my_order=(1,1,1)
# # #     my_seasonal_order=(1,1,1,144)
# # #     model=SARIMAX(data, order=my_order,seasonal_order=my_seasonal_order)
# # #     model_fit=model.fit()
# # #     yhat=model_fit.predict()
# # #     fore_arr_RRIautos.append(model_fit.predict(432, 720))
# # #     save_data_RRIautos.append(model_fit.predict(1, 432))
# # # np.savetxt("forecast24h_RRIauto_short_SA0124.csv", fore_arr_RRIautos, delimiter=",", fmt='%s')
# # # np.savetxt("forecast24h_RRIauto_save_short_SA0124.csv", save_data_RRIautos, delimiter=",", fmt='%s')
# # # np.savetxt("forecast24h_RRIauto_save_index_short_SA0124.csv", index_arr_RRIautos, delimiter=",", fmt='%s')
# # # print(len(fore_arr_RRIautos))
# # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/forecast/forecast_RRIauto_short_SA0124.csv')
# # fore_arr_RRIauto=[]
# # for lines in csv_reader:
# #     fore_arr_RRIauto.append(float(lines))
# # # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/forecast/forecast_RRIauto_save_short_SA0124.csv')
# # # fore_arr_RRIauto_save=[]
# # # for lines in csv_reader:
# # #     fore_arr_RRIauto_save.append(float(lines))
# # # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/forecast/forecast_RRIauto_save_index_short_SA0124.csv')
# # # index=csv_reader.to_numpy()
# # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/forecast/24h/forecast24h_RRIauto_short_SA0124.csv')
# # fore_arr_RRIauto=[]
# # for lines in csv_reader:
# #     fore_arr_RRIauto.append(float(lines))
# # # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/forecast/24h/forecast24h_RRIauto_save_short_SA0124.csv')
# # # fore_arr_RRIauto_save=[]
# # # for lines in csv_reader:
# # #     fore_arr_RRIauto_save.append(float(lines))
# # # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/forecast/24h/forecast24h_RRIauto_save_index_short_SA0124.csv')
# # # index=csv_reader.to_numpy()
# # # t=[]
# # # for item in index:
# # #     t.append(int(item))
# # # pyplot.figure(figsize=(12, 4))
# # # pyplot.plot(RRI_time_var_arr[479:],rolmean_phase_RRIautos[479:],c='gold')
# # # pyplot.plot(RRI_time_var_arr[t],fore_arr_RRIauto_save[0:431],c='grey')
# # # t=np.linspace(RRI_time_var_arr[-1],RRI_time_var_arr[-1]+0.16666667*(len(fore_arr_RRIauto)-1),len(fore_arr_RRIauto))
# # # pyplot.plot(t,fore_arr_RRIauto)
# # # pyplot.ylabel('phase')
# # # pyplot.title('phases in short cycle RRI auto')
# # # pyplot.xlabel('time(h)')
# # # pyplot.show()
# #
# #
# #
# #
# # ### forecast probability
# # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/forecast/24h/forecast24h_eegvar_short_SA0124.csv')
# # fore_arr_eegvar=[]
# # for lines in csv_reader:
# #     fore_arr_eegvar.append(float(lines))
# # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/forecast/24h/forecast24h_eegauto_short_SA0124.csv')
# # fore_arr_eegauto=[]
# # for lines in csv_reader:
# #     fore_arr_eegauto.append(float(lines))
# # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/forecast/24h/forecast24h_RRIvar_short_SA0124.csv')
# # fore_arr_RRIvar=[]
# # for lines in csv_reader:
# #     fore_arr_RRIvar.append(float(lines))
# # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/forecast/combine24h_RRIauto.csv')
# # fore_arr_RRIauto=[]
# # for lines in csv_reader:
# #     fore_arr_RRIauto.append(float(lines))
# #
# # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/EEG/EEG_timewindowarr_SA0124_74h_15s.csv')
# # eeg_time_var_arr = csv_reader.to_numpy()
# # t=np.linspace(eeg_time_var_arr [-1],eeg_time_var_arr [-1]+0.16666667*(len(fore_arr_eegvar)-1),len(fore_arr_eegvar))
# #
# #
# # bins_number = 18
# # bins = np.linspace(0, 2*np.pi, bins_number + 1)
# # pro_eegvars_time=[]
# # pro_eegvars_time_false=[]
# # for i in range(len(fore_arr_eegvar)):
# #     if fore_arr_eegvar[i] >= bins[2] and fore_arr_eegvar[i]< bins[3]:
# #         pro_eegvars_time_false.append(0.000462372)
# #         pro_eegvars_time.append(2.8596E-07)
# #     elif fore_arr_eegvar[i] >= bins[3] and fore_arr_eegvar[i] < bins[4]:
# #         pro_eegvars_time_false.append(0.057969911)
# #         pro_eegvars_time.append(3.58523E-05)
# #     elif fore_arr_eegvar[i] >= bins[4] and fore_arr_eegvar[i] < bins[5]:
# #         pro_eegvars_time_false.append(0.045485862)
# #         pro_eegvars_time.append(2.81314E-05)
# #     elif fore_arr_eegvar[i] >= bins[5] and fore_arr_eegvar[i] < bins[6]:
# #         pro_eegvars_time_false.append(0.077216152)
# #         pro_eegvars_time.append(4.77554E-05)
# #     elif fore_arr_eegvar[i] >= bins[6] and fore_arr_eegvar[i] < bins[7]:
# #         pro_eegvars_time_false.append(0.090856131)
# #         pro_eegvars_time.append(5.61912E-05)
# #     elif fore_arr_eegvar[i] >= bins[7] and fore_arr_eegvar[i] < bins[8]:
# #         pro_eegvars_time_false.append(0.119234244)
# #         pro_eegvars_time.append(0.199930111)
# #     elif fore_arr_eegvar[i] >= bins[8] and fore_arr_eegvar[i] < bins[9]:
# #         pro_eegvars_time_false.append(0.141370291)
# #         pro_eegvars_time.append(8.74324E-05)
# #     elif fore_arr_eegvar[i] >= bins[9] and fore_arr_eegvar[i] < bins[10]:
# #         pro_eegvars_time_false.append(0.136631)
# #         pro_eegvars_time.append(0.199930111)
# #     elif fore_arr_eegvar[i] >= bins[10] and fore_arr_eegvar[i] < bins[11]:
# #         pro_eegvars_time_false.append(0.117558124)
# #         pro_eegvars_time.append(7.27054E-05)
# #     elif fore_arr_eegvar[i] >= bins[11] and fore_arr_eegvar[i] < bins[12]:
# #         pro_eegvars_time_false.append(0.10946663)
# #         pro_eegvars_time.append(0.199930111)
# #     elif fore_arr_eegvar[i] >= bins[12] and fore_arr_eegvar[i] < bins[13]:
# #         pro_eegvars_time_false.append(0.017281163)
# #         pro_eegvars_time.append(0.199930111)
# #     elif fore_arr_eegvar[i] >= bins[13] and fore_arr_eegvar[i] < bins[14]:
# #         pro_eegvars_time_false.append(0.00780253)
# #         pro_eegvars_time.append(4.82558E-06)
# #     elif fore_arr_eegvar[i] >= bins[14] and fore_arr_eegvar[i] < bins[15]:
# #         pro_eegvars_time_false.append(0.050745355)
# #         pro_eegvars_time.append(0.199930111)
# #     elif fore_arr_eegvar[i] >= bins[15] and fore_arr_eegvar[i] < bins[16]:
# #         pro_eegvars_time_false.append(0.026355214)
# #         pro_eegvars_time.append(1.62997E-05)
# #     else:
# #         pro_eegvars_time_false.append(0)
# #         pro_eegvars_time.append(0)
# #
# #
# # pro_eegautos_time=[]
# # pro_eegautos_time_false=[]
# # for i in range(len(fore_arr_eegauto)):
# #     if fore_arr_eegauto[i] >= bins[2] and fore_arr_eegauto[i]< bins[3]:
# #         pro_eegautos_time_false.append(0.023270621)
# #         pro_eegautos_time.append(1.37013E-05)
# #     elif fore_arr_eegauto[i] >= bins[3] and fore_arr_eegauto[i] < bins[4]:
# #         pro_eegautos_time_false.append(0.044688853)
# #         pro_eegautos_time.append(2.63119E-05)
# #     elif fore_arr_eegauto[i] >= bins[4] and fore_arr_eegauto[i] < bins[5]:
# #         pro_eegautos_time_false.append(0.074848054)
# #         pro_eegautos_time.append(0.199932183)
# #     elif fore_arr_eegauto[i] >= bins[5] and fore_arr_eegauto[i] < bins[6]:
# #         pro_eegautos_time_false.append(0.092214191)
# #         pro_eegautos_time.append(0.199932183)
# #     elif fore_arr_eegauto[i] >= bins[6] and fore_arr_eegauto[i] < bins[7]:
# #         pro_eegautos_time_false.append(0.10819102)
# #         pro_eegautos_time.append(6.37006E-05)
# #     elif fore_arr_eegauto[i] >= bins[7] and fore_arr_eegauto[i] < bins[8]:
# #         pro_eegautos_time_false.append(0.091403756)
# #         pro_eegautos_time.append(5.38166E-05)
# #     elif fore_arr_eegauto[i] >= bins[8] and fore_arr_eegauto[i] < bins[9]:
# #         pro_eegautos_time_false.append(0.099507952)
# #         pro_eegautos_time.append(5.85882E-05)
# #     elif fore_arr_eegauto[i] >= bins[9] and fore_arr_eegauto[i] < bins[10]:
# #         pro_eegautos_time_false.append(0.070795943)
# #         pro_eegautos_time.append(4.16832E-05)
# #     elif fore_arr_eegauto[i] >= bins[10] and fore_arr_eegauto[i] < bins[11]:
# #         pro_eegautos_time_false.append(0.069059341)
# #         pro_eegautos_time.append(0.199932183)
# #     elif fore_arr_eegauto[i] >= bins[11] and fore_arr_eegauto[i] < bins[12]:
# #         pro_eegautos_time_false.append(0.140144732)
# #         pro_eegautos_time.append(0.199932183)
# #     elif fore_arr_eegauto[i] >= bins[12] and fore_arr_eegauto[i] < bins[13]:
# #         pro_eegautos_time_false.append(0.081099849)
# #         pro_eegautos_time.append(4.77499E-05)
# #     elif fore_arr_eegauto[i] >= bins[13] and fore_arr_eegauto[i] < bins[14]:
# #         pro_eegautos_time_false.append(0.022807524)
# #         pro_eegautos_time.append(1.34286E-05)
# #     elif fore_arr_eegauto[i] >= bins[14] and fore_arr_eegauto[i] < bins[15]:
# #         pro_eegautos_time_false.append(0.001852388)
# #         pro_eegautos_time.append(1.09065E-06)
# #     elif fore_arr_eegauto[i] >= bins[15] and fore_arr_eegauto[i] < bins[16]:
# #         pro_eegautos_time_false.append(0.003646888)
# #         pro_eegautos_time.append(2.14721E-06)
# #     elif fore_arr_eegauto[i] >= bins[16] and fore_arr_eegauto[i] < bins[17]:
# #         pro_eegautos_time_false.append(0.07432707)
# #         pro_eegautos_time.append(0.199932183)
# #     elif fore_arr_eegauto[i] >= bins[17]:
# #         pro_eegautos_time_false.append(0.002141823)
# #         pro_eegautos_time.append(1.26106E-06)
# #     else:
# #         pro_eegautos_time_false.append(0)
# #         pro_eegautos_time.append(0)
# #
# #
# # pro_RRIvars_time=[]
# # pro_RRIvars_time_false=[]
# # for i in range(len(fore_arr_RRIvar)):
# #     if fore_arr_RRIvar[i] >= bins[3] and fore_arr_RRIvar[i]< bins[4]:
# #         pro_RRIvars_time_false.append(0.003762663)
# #         pro_RRIvars_time.append(2.53066E-06)
# #     elif fore_arr_RRIvar[i] >= bins[4] and fore_arr_RRIvar[i] < bins[5]:
# #         pro_RRIvars_time_false.append(0.046946451)
# #         pro_RRIvars_time.append(3.15749E-05)
# #     elif fore_arr_RRIvar[i] >= bins[5] and fore_arr_RRIvar[i] < bins[6]:
# #         pro_RRIvars_time_false.append(0.079884236)
# #         pro_RRIvars_time.append(0.399825689)
# #     elif fore_arr_RRIvar[i] >= bins[6] and fore_arr_RRIvar[i] < bins[7]:
# #         pro_RRIvars_time_false.append(0.119305346)
# #         pro_RRIvars_time.append(8.02415E-05)
# #     elif fore_arr_RRIvar[i] >= bins[7] and fore_arr_RRIvar[i] < bins[8]:
# #         pro_RRIvars_time_false.append(0.07386396)
# #         pro_RRIvars_time.append(4.96789E-05)
# #     elif fore_arr_RRIvar[i] >= bins[8] and fore_arr_RRIvar[i] < bins[9]:
# #         pro_RRIvars_time_false.append(0.091403756)
# #         pro_RRIvars_time.append(6.14756E-05)
# #     elif fore_arr_RRIvar[i] >= bins[9] and fore_arr_RRIvar[i] < bins[10]:
# #         pro_RRIvars_time_false.append(0.131114344)
# #         pro_RRIvars_time.append(0.199912844)
# #     elif fore_arr_RRIvar[i] >= bins[10] and fore_arr_RRIvar[i] < bins[11]:
# #         pro_RRIvars_time_false.append(0.081273511)
# #         pro_RRIvars_time.append(5.46623E-05)
# #     elif fore_arr_RRIvar[i] >= bins[11] and fore_arr_RRIvar[i] < bins[12]:
# #         pro_RRIvars_time_false.append(0.114558625)
# #         pro_RRIvars_time.append(0.199912844)
# #     elif fore_arr_RRIvar[i] >= bins[12] and fore_arr_RRIvar[i] < bins[13]:
# #         pro_RRIvars_time_false.append(0.108885665)
# #         pro_RRIvars_time.append(7.32335E-05)
# #     elif fore_arr_RRIvar[i] >= bins[13] and fore_arr_RRIvar[i] < bins[14]:
# #         pro_RRIvars_time_false.append(0.051403759)
# #         pro_RRIvars_time.append(3.45727E-05)
# #     elif fore_arr_RRIvar[i] >= bins[14] and fore_arr_RRIvar[i] < bins[15]:
# #         pro_RRIvars_time_false.append(0.026338643)
# #         pro_RRIvars_time.append(0.199912844)
# #     elif fore_arr_RRIvar[i] >= bins[15] and fore_arr_RRIvar[i] < bins[16]:
# #         pro_RRIvars_time_false.append(0.049204049)
# #         pro_RRIvars_time.append(3.30933E-05)
# #     elif fore_arr_RRIvar[i] >= bins[16] and fore_arr_RRIvar[i] < bins[17]:
# #         pro_RRIvars_time_false.append(0.022054991)
# #         pro_RRIvars_time.append(1.48336E-05)
# #     else:
# #         pro_RRIvars_time_false.append(0)
# #         pro_RRIvars_time.append(0)
# #
# # pro_RRIautos_time=[]
# # pro_RRIautos_time_false=[]
# # for i in range(len(fore_arr_RRIauto)):
# #     if fore_arr_RRIauto[i] >= bins[1] and fore_arr_RRIauto[i]< bins[2]:
# #         pro_RRIautos_time_false.append(0.002547033)
# #         pro_RRIautos_time.append(4.30422E-06)
# #     elif fore_arr_RRIauto[i] >= bins[2] and fore_arr_RRIauto[i] < bins[3]:
# #         pro_RRIautos_time_false.append(0.075137508)
# #         pro_RRIautos_time.append(0.399517919)
# #     elif fore_arr_RRIauto[i] >= bins[3] and fore_arr_RRIauto[i] < bins[4]:
# #         pro_RRIautos_time_false.append(0.052387836)
# #         pro_RRIautos_time.append(8.853E-05)
# #     elif fore_arr_RRIauto[i] >= bins[4] and fore_arr_RRIauto[i] < bins[5]:
# #         pro_RRIautos_time_false.append(0.082431288)
# #         pro_RRIautos_time.append(0.19975896)
# #     elif fore_arr_RRIauto[i] >= bins[5] and fore_arr_RRIauto[i] < bins[6]:
# #         pro_RRIautos_time_false.append(0.092735153)
# #         pro_RRIautos_time.append(0.000156713)
# #     elif fore_arr_RRIauto[i] >= bins[6] and fore_arr_RRIauto[i] < bins[7]:
# #         pro_RRIautos_time_false.append(0.052329949)
# #         pro_RRIautos_time.append(8.84322E-05)
# #     elif fore_arr_RRIauto[i] >= bins[7] and fore_arr_RRIauto[i] < bins[8]:
# #         pro_RRIautos_time_false.append(0.094355993)
# #         pro_RRIautos_time.append(0.000159452)
# #     elif fore_arr_RRIauto[i] >= bins[8] and fore_arr_RRIauto[i] < bins[9]:
# #         pro_RRIautos_time_false.append(0.082662836)
# #         pro_RRIautos_time.append(0.19975896)
# #     elif fore_arr_RRIauto[i] >= bins[9] and fore_arr_RRIauto[i] < bins[10]:
# #         pro_RRIautos_time_false.append(0.100955123)
# #         pro_RRIautos_time.append(0.000170604)
# #     elif fore_arr_RRIauto[i] >= bins[10] and fore_arr_RRIauto[i] < bins[11]:
# #         pro_RRIautos_time_false.append(0.079189569)
# #         pro_RRIautos_time.append(0.000133822)
# #     elif fore_arr_RRIauto[i] >= bins[11] and fore_arr_RRIauto[i] < bins[12]:
# #         pro_RRIautos_time_false.append(0.114558595)
# #         pro_RRIautos_time.append(0.000193592)
# #     elif fore_arr_RRIauto[i] >= bins[12] and fore_arr_RRIauto[i] < bins[13]:
# #         pro_RRIautos_time_false.append(0.046830697)
# #         pro_RRIautos_time.append(0.19975896)
# #     elif fore_arr_RRIauto[i] >= bins[13] and fore_arr_RRIauto[i] < bins[14]:
# #         pro_RRIautos_time_false.append(0.089261927)
# #         pro_RRIautos_time.append(0.000150843)
# #     elif fore_arr_RRIauto[i] >= bins[14] and fore_arr_RRIauto[i] < bins[15]:
# #         pro_RRIautos_time_false.append(0.007583212)
# #         pro_RRIautos_time.append(1.28148E-05)
# #     elif fore_arr_RRIauto[i] >= bins[15] and fore_arr_RRIauto[i] < bins[16]:
# #         pro_RRIautos_time_false.append(0.004341533)
# #         pro_RRIautos_time.append(7.33674E-06)
# #     elif fore_arr_RRIauto[i] >= bins[16] and fore_arr_RRIauto[i] < bins[17]:
# #         pro_RRIautos_time_false.append(0.022691748)
# #         pro_RRIautos_time.append(3.83467E-05)
# #     else:
# #         pro_RRIautos_time_false.append(0)
# #         pro_RRIautos_time.append(0)
# #
# #
# # Pseizureeegvar=0.000289453; Pseizureeegauto=0.000289445;PseizureRRIvar=0.000289478;PseizureRRIauto=0.000289701;
# # Pnonseizureeegvar=0.999710547;Pnonseizureeegauto=0.99971055;PnonseizureRRIvar=0.999710522;PnonseizureRRIauto=0.999710299;
# # Pcombined=[]
# # for m in range(len(pro_eegvars_time)):
# #     P1=Pseizureeegvar*(pro_eegvars_time[m]*pro_eegautos_time[m]*pro_RRIvars_time[m]*pro_RRIautos_time[m])
# #     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m])
# #     Pcombined.append(P1/(P1+P2))
# # pyplot.figure(figsize=(5,3))
# # pyplot.plot(t,Pcombined,label='combined probability')
# # pyplot.annotate('',xy=(80.90055,np.max(Pcombined)),xytext=(80.90055,np.max(Pcombined)+0.00000000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(92.7538833,np.max(Pcombined)),xytext=(92.7538833,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# #
# # # pyplot.annotate('',xy=(104.916106,np.max(Pcombined)),xytext=(104.916106,np.max(Pcombined)+0.00000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(139.57055,np.max(Pcombined)),xytext=(139.57055,np.max(Pcombined)+0.00000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(152.573328,np.max(Pcombined)),xytext=(152.573328,np.max(Pcombined)+0.00000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(165.256383,np.max(Pcombined)),xytext=(165.256383,np.max(Pcombined)+0.00000001),arrowprops=dict(facecolor='black',shrink=0.05))
# #
# # pyplot.xlabel('time')
# # pyplot.ylabel('seizure probability')
# # pyplot.hlines(1e-07, t[0],t[-1],'r')
# # pyplot.show()
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# # # ## forecast probability by independet idea
# # # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/forecast/24h/forecast24h_eegvar_short_SA0124.csv')
# # # fore_arr_eegvar=[]
# # # for lines in csv_reader:
# # #     fore_arr_eegvar.append(float(lines))
# # # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/forecast/24h/forecast24h_eegauto_short_SA0124.csv')
# # # fore_arr_eegauto=[]
# # # for lines in csv_reader:
# # #     fore_arr_eegauto.append(float(lines))
# # # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/forecast/24h/forecast24h_RRIvar_short_SA0124.csv')
# # # fore_arr_RRIvar=[]
# # # for lines in csv_reader:
# # #     fore_arr_RRIvar.append(float(lines))
# # # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/forecast/24h/forecast24h_RRIauto_short_SA0124.csv')
# # # fore_arr_RRIauto=[]
# # # for lines in csv_reader:
# # #     fore_arr_RRIauto.append(float(lines))
# # # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/EEG/EEG_timewindowarr_SA0124_74h_15s.csv')
# # # eeg_time_var_arr = csv_reader.to_numpy()
# # # t=np.linspace(eeg_time_var_arr [-1],eeg_time_var_arr [-1]+0.16666667*(len(fore_arr_eegvar)-1),len(fore_arr_eegvar))
# # #
# # # bins_number = 18
# # # bins = np.linspace(0, 2*np.pi, bins_number + 1)
# # # pro_eegvars_time=[]
# # # for i in range(len(fore_arr_eegvar)):
# # #     if fore_arr_eegvar[i] >= bins[7] and fore_arr_eegvar[i] < bins[8]:
# # #         pro_eegvars_time.append(0.000484496)
# # #     elif fore_arr_eegvar[i] >= bins[9] and fore_arr_eegvar[i] < bins[10]:
# # #         pro_eegvars_time.append( 0.000422833)
# # #     elif fore_arr_eegvar[i] >= bins[11] and fore_arr_eegvar[i] < bins[12]:
# # #         pro_eegvars_time.append(0.000527704)
# # #     elif fore_arr_eegvar[i] >= bins[12] and fore_arr_eegvar[i] < bins[13]:
# # #         pro_eegvars_time.append(0.003333333)
# # #     elif fore_arr_eegvar[i] >= bins[14] and fore_arr_eegvar[i] < bins[15]:
# # #         pro_eegvars_time.append(0.001137656)
# # #     else:
# # #         pro_eegvars_time.append(1.78788E-07)
# # # pro_eegautos_time=[]
# # # for i in range(len(fore_arr_eegauto)):
# # #     if fore_arr_eegauto[i] >= bins[4] and fore_arr_eegauto[i] < bins[5]:
# # #         pro_eegautos_time.append(0.000772798)
# # #     elif fore_arr_eegauto[i] >= bins[5] and fore_arr_eegauto[i] < bins[6]:
# # #         pro_eegautos_time.append(0.000627353)
# # #     elif fore_arr_eegauto[i] >= bins[10] and fore_arr_eegauto[i] < bins[11]:
# # #         pro_eegautos_time.append(0.000837521)
# # #     elif fore_arr_eegauto[i] >= bins[11] and fore_arr_eegauto[i] < bins[12]:
# # #         pro_eegautos_time.append(0.000412882)
# # #     elif fore_arr_eegauto[i] >= bins[16] and fore_arr_eegauto[i] < bins[17]:
# # #         pro_eegautos_time.append(0.00077821)
# # #     else:
# # #         pro_eegautos_time.append(1.70471E-07)
# # # pro_RRIvars_time=[]
# # # for i in range(len(fore_arr_RRIvar)):
# # #     if fore_arr_RRIvar[i] >= bins[5] and fore_arr_RRIvar[i] < bins[6]:
# # #         pro_RRIvars_time.append(0.001447178)
# # #     elif fore_arr_RRIvar[i] >= bins[9] and fore_arr_RRIvar[i] < bins[10]:
# # #         pro_RRIvars_time.append(0.000441306)
# # #     elif fore_arr_RRIvar[i] >= bins[11] and fore_arr_RRIvar[i] < bins[12]:
# # #         pro_RRIvars_time.append(0.000505051)
# # #     elif fore_arr_RRIvar[i] >= bins[14] and fore_arr_RRIvar[i] < bins[15]:
# # #         pro_RRIvars_time.append(0.002192982)
# # #     else:
# # #         pro_RRIvars_time.append(1.94751E-07)
# # # pro_RRIautos_time=[]
# # # for i in range(len(fore_arr_RRIauto)):
# # #     if fore_arr_RRIauto[i] >= bins[2] and fore_arr_RRIauto[i] < bins[3]:
# # #         pro_RRIautos_time.append(0.001538462)
# # #     elif fore_arr_RRIauto[i] >= bins[4] and fore_arr_RRIauto[i] < bins[5]:
# # #         pro_RRIautos_time.append(0.000701754)
# # #     elif fore_arr_RRIauto[i] >= bins[8] and fore_arr_RRIauto[i] < bins[9]:
# # #         pro_RRIautos_time.append(0.00069979)
# # #     elif fore_arr_RRIauto[i] >= bins[12] and fore_arr_RRIauto[i] < bins[13]:
# # #         pro_RRIautos_time.append(0.001234568)
# # #     else:
# #         pro_RRIautos_time.append(4.89706E-07)
# #
# # print(len(pro_eegvars_time))
# # Pcombined=[]
# # for m in range(len(pro_eegvars_time)):
# #     Pcombined.append(pro_eegvars_time[m]+pro_eegautos_time[m]+pro_RRIvars_time[m]+pro_RRIautos_time[m])
# # print(Pcombined)
# #
# # pyplot.figure(figsize=(6,3))
# # pyplot.plot(t,Pcombined,label='combined probability')
# # pyplot.annotate('',xy=(80.90055,np.max(Pcombined)),xytext=(80.90055,np.max(Pcombined)+0.00000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(92.7538833,np.max(Pcombined)),xytext=(92.7538833,np.max(Pcombined)+0.00000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(104.916106,np.max(Pcombined)),xytext=(104.916106,np.max(Pcombined)+0.00000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # # pyplot.annotate('',xy=(139.57055,np.max(Pcombined)),xytext=(139.57055,np.max(Pcombined)+0.00000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # # pyplot.annotate('',xy=(152.573328,np.max(Pcombined)),xytext=(152.573328,np.max(Pcombined)+0.00000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # # pyplot.annotate('',xy=(165.256383,np.max(Pcombined)),xytext=(165.256383,np.max(Pcombined)+0.00000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.hlines(0.0020752100395338233, t[0],t[-1],'r')
# # pyplot.xlabel('time')
# # pyplot.ylabel('seizure probability')
# # pyplot.show()
# #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# #
#
#
#
#
#
#
# # np.savetxt("forecast_eegvar_short_SA0124.csv", fore_arr_vars, delimiter=",", fmt='%s')
# # print(fore_arr_vars)
# # pyplot.figure(figsize=(12, 4))
# # pyplot.plot(eeg_time_var_arr,rolmean_phase_eegvars,c='r')
# # t=np.linspace(eeg_time_var_arr[-1],eeg_time_var_arr[-1]+0.16666667*(len(fore_arr_vars[0])-1),len(fore_arr_vars[0]))
# # pyplot.plot(t,fore_arr_vars[0])
# # pyplot.xlabel('time(h)')
# # pyplot.ylabel('phase')
# # pyplot.title('phases in short cycle EEG variance')
# # pyplot.show()