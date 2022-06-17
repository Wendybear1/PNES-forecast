import scipy.signal
from scipy.signal import butter, lfilter
from matplotlib import pyplot
import math
from scipy.fftpack import fft, ifft
from scipy import signal
from scipy.signal import butter, lfilter,iirfilter
from scipy.signal import hilbert
from biosppy.signals import tools
import pandas as pd



 ##VIC1012 no pure
# x5=[0,0,1]
# y5=[0,0.28,1]
# x6=[0,1]
# y6=[0,1]
#
# ###### 6 h
# x2=[0,1,1,1,1,1,1]
# y2=[0,0.2,0.2,0.2,0.2,0.2,1]
# x4=[0,0,0,0,0,0,1]
# y4=[0,0.02667,0.02667,0.02667,0.02667,0.02667,1]
# x9=[0,1,1]
# y9=[0,0.37333,1]
# x10=[0,0,1]
# y10=[0,0.12,1]
# x11=[0,1,1,1,1,1,1]
# y11=[0,0.36,0.52,0.52,0.52,0.52,1]
# x12=[0,0,0,0,0,0,1]
# y12=[0,0.02667,0.10667,0.10667,0.10667,0.10667,1]
# x15=[0,1,1,1,1,1,1]
# y15=[0,0.12,0.12,0.386667,0.386667,0.386667,1]
# x16=[0,0,0,0,0,0,1]
# y16=[0,0.013333,0.12,0.12,0.12,0.12,1]
# x17=[0,1,1,1,1,1,1]
# y17=[0,0.946667,0.946667,0.946667,0.946667,0.946667,1]
# x18=[0,0,0,0,0,0,1]
# y18=[0,0.28,0.28,0.28,0.28,0.28,1]
#
#
# from sklearn.metrics import auc
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y5,x5)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y2,x2)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y4,x4)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y7,x7)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y8,x8)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y9,x9)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y10,x10)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y11,x11)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y12,x12)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y13,x13)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y14,x14)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y15,x15)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y16,x16)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y17,x17)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y18,x18)))



# ###TAS0056 no pure
# x5=[0,0,3/4,1]
# y5=[0,0.07407,0.45679,1]
# x6=[0,1]
# y6=[0,1]
#
# # # #### 6h 6h 6h
# # x2=[0,3/4,7/8,7/8,1]
# # y2=[0,0.55556,0.67901,0.82716,1]
# # x4=[0,5/8,3/4,3/4,1]
# # y4=[0,0.37037,0.40741,0.44444,1]
# # x9=[0,3/4,7/8,7/8,1]
# # y9=[0,0.604938,0.728395,0.851852,1]
# # x10=[0,5/8,3/4,3/4,1]
# # y10=[0,0.382716,0.419753,0.45679,1]
# # x13=[0,7/8,1]
# # y13=[0,0.8272,1]
# # x14=[0,0,3/4,1]
# # y14=[0,0.0741,0.4444,1]
# # x15=[0,3/8,7/8,7/8,1]
# # y15=[0,0.50617,0.61728,0.85185,1]
# # x16=[0,0,1/8,3/4,3/4,1]
# # y16=[0,0.04938,0.11111,0.39506,0.45679,1]
# # x17=[0,1/2,7/8,1]
# # y17=[0,0.469136,0.851852,1]
# # x18=[0,0,3/8,3/4,1]
# # y18=[0,0.074074,0.283951,0.45679,1]
#
# # # # # #### 12h 12h 12h
# x2=[0,1/8,3/8,7/8,1]
# y2=[0,0.493827,0.5555556,0.851852,1]
# x4=[0,1/8,1/8,3/8,3/4,1]
# y4=[0,0.271605,0.296296,0.33333,0.44444,1]
# x9=[0,1/8,7/8,1]
# y9=[0,0.506173,0.851852,1]
# x10=[0,1/8,1/2,3/4,1]
# y10=[0,0.296296,0.395062,0.444444,1]
# x13=[0,7/8,7/8,1]
# y13=[0,0.839506,0.851852,1]
# x14=[0,3/8,3/4,3/4,1]
# y14=[0,0.259259,0.444444,0.45679,1]
# x15=[0,1/8,7/8,1]
# y15=[0,0.506173,0.851852,1]
# x16=[0,1/8,3/4,1]
# y16=[0,0.296296,0.444444,1]
# x17=[0,7/8,1]
# y17=[0,0.851852,1]
# x18=[0,0,3/4,1]
# y18=[0,0.074074,0.45679,1]
#
# # # # # #### 1h 1h 1h
# # x9=[0,3/4,7/8,1]
# # y9=[0,0.679012,0.691358,1]
# # x10=[0,3/4,1]
# # y10=[0,0.358025,1]
# # x15=[0,3/4,3/4,7/8,1]
# # y15=[0,0.654321,0.691358,0.728395,1]
# # x16=[0,3/4,3/4,1]
# # y16=[0,0.33333,0.358025,1]
# # x17=[0,7/8,1]
# # y17=[0,0.839506,1]
# # x18=[0,3/4,1]
# # y18=[0,0.45679,1]
#
# # # #### 5 min
# # x2=[0,5/8,5/8,5/8,5/8,1]
# # y2=[0,0.419753,0.54321,0.62963,0.6666667,1]
# # x4=[0,3/8,3/8,3/8,3/8,1]
# # y4=[0,0.185185,0.234568,0.259259,0.283951,1]
# # x9=[0,7/8,7/8,7/8,1]
# # y9=[0,0.740741,0.765432,0.839506,1]
# # x10=[0,1/2,1/2,1/2,1/2,1]
# # y10=[0,0.395062,0.407407,0.419753,0.432099,1]
# # x11=[0,5/8,5/8,5/8,1]
# # y11=[0,0.5308664,0.691358,0.703704,1]
# # x12=[0,0,3/4,1]
# # y12=[0,0.074074,0.444444,1]
# # x15=[0,1/4,7/8,1]
# # y15=[0,0.604938,0.839506,1]
# # x16=[0,1/4,5/8,1]
# # y16=[0,0.320988,0.432099,1]
# # x17=[0,7/8,7/8,1]
# # y17=[0,0.839506,0.851852,1]
# # x18=[0,5/8,5/8,5/8,1]
# # y18=[0,0.358025,0.44444,0.45679,1]
#
# from sklearn.metrics import auc
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y5,x5)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y2,x2)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y4,x4)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y7,x7)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y8,x8)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y9,x9)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y10,x10)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y11,x11)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y12,x12)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y13,x13)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y14,x14)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y15,x15)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y16,x16)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y17,x17)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y18,x18)))





# # ##### TAS0058 no pure
# x5=[0,2/3,1]
# y5=[0,0.613333,1]
# x6=[0,1]
# y6=[0,1]
#
# # # #### 6h 6h 6h
# # x2=[0,2/3,2/3,1]
# # y2=[0,0.36,0.4933333,1]
# # x4=[0,1/2,1/2,1]
# # y4=[0,0.2,0.3866667,1]
# # x11=[0,2/3,2/3,2/3,1]
# # y11=[0,0.32,0.333333,0.4933333,1]
# # x12=[0,1/2,1/2,1]
# # y12=[0,0.18666667,0.38666667,1]
# # x17=[0,2/3,1,1]
# # y17=[0,0.41333,0.853333,1]
# # x18=[0,1/2,2/3,1]
# # y18=[0,0.253333,0.6,1]
#
# # # # # #### 12h 12h 12h
# # x2=[0,0,1]
# # y2=[0,0,1]
# # x4=[0,0,1]
# # y4=[0,0,1]
# # x7=[0,0,1]
# # y7=[0,0,1]
# # x8=[0,0,1]
# # y8=[0,0,1]
# # x11=[0,0,1]
# # y11=[0,0,1]
# # x12=[0,0,1]
# # y12=[0,0,1]
# # x13=[0,1/6,2/3,1]
# # y13=[0,0.253333,0.4933333,1]
# # x14=[0,1/6,1/2,2/3,1]
# # y14=[0,0.173333,0.2,0.306667,1]
# # x17=[0,1/2,1,1]
# # y17=[0,0.4,0.86667,1]
# # x18=[0,1/3,2/3,1]
# # y18=[0,0.226667,0.6,1]
#
#
# # # # # # #### 1h 1h 1h
# x2=[0,1/3,1/3,1/3,1]
# y2=[0,0.12,0.2,0.26667,1]
# x4=[0,1/6,1/3,1/3,1/3,1]
# y4=[0,0.08,0.106667,0.12,0.173333,1]
# x7=[0,1/3,1/3,1/3,1]
# y7=[0,0.186667,0.213333,0.28,1]
# x8=[0,1/3,1/3,1/3,1]
# y8=[0,0.106667,0.12,0.17333,1]
# x11=[0,1/3,1/3,1]
# y11=[0,0.186667,0.493333,1]
# x12=[0,1/6,1/3,1/3,1]
# y12=[0,0.093333,0.213333,0.32,1]
# x13=[0,1/2,2/3,2/3,1]
# y13=[0,0.32,0.453333,0.56,1]
# x14=[0,1/2,1/2,1/2,1]
# y14=[0,0.2,0.253333,0.333333,1]
# x17=[0,1,1]
# y17=[0,0.786667,1]
# x18=[0,1/2,2/3,1]
# y18=[0,0.293333,0.56,1]
#
# # # #### 5 min
# # x17=[0,1,1,1]
# # y17=[0,0.866667,0.88,1]
# # x18=[0,2/3,2/3,1]
# # y18=[0,0.573333,0.6,1]
#
# from sklearn.metrics import auc
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y5,x5)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y2,x2)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y4,x4)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y7,x7)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y8,x8)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y9,x9)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y10,x10)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y11,x11)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y12,x12)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y13,x13)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y14,x14)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y15,x15)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y16,x16)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y17,x17)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y18,x18)))






###### QLD0290 QLD0290 QLD0290 QLD0290 pure
x5=[0,3/5,1]
y5=[0,0.2716,1]
x6=[0,1]
y6=[0,1]


# # # #### 24h 24h 24h
# x2=[0,1/5,2/5,1]
# y2=[0,0.4938,0.8654,1]
# x13=[0,3/5,1,1]
# y13=[0,0.5679,0.9383,1]
# x15=[0,2/5,1]
# y15=[0,0.7654,1]


# # # # #### 12h 12h 12h
# x2=[0,1,1]
# y2=[0,0.9383,1]
# x7=[0,1,1]
# y7=[0,0.9383,1]
# x11=[0,2/5,1,1]
# y11=[0,0.4938,0.9383,1]
# x13=[0,1/5,1,1]
# y13=[0,0.5309, 0.9383,1]
# x17=[0,0,1,1]
# y17=[0,0.5802,0.9383,1]


# # # #### 6h 6h 6h
# x2=[0,0,0,1]
# y2=[0,0.0247,0.037,1]
# x7=[0,1/5,1]
# y7=[0,0.2469,1]
# x11=[0,1/5,1]
# y11=[0,0.284,1]
# x13=[0,1,1,1]
# y13=[0,0.6543, 0.9012,1]
# x17=[0,1/5,1]
# y17=[0,0.1605,1]


# # # #### 1h 1h 1h
# x7=[0,0,0.4,1]
# y7=[0,0.4568,0.5432,1]
# x11=[0,1/5,3/5,1]
# y11=[0,0.4321,0.6173,1]
# x13=[0,0,4/5,4/5,1]
# y13=[0,0.4815,0.7654321,0.8395,1]

#
# # #### 5 min
x13=[0,1,1,1]
y13=[0,0.9136,0.9383,1]



from sklearn.metrics import auc
print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y5,x5)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y2,x2)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y4,x4)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y7,x7)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y8,x8)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y9,x9)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y10,x10)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y11,x11)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y12,x12)))
print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y13,x13)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y14,x14)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y15,x15)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y16,x16)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y17,x17)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y18,x18)))


# # ### VIC0583 VIC0583 VIC0583 VIC0583 pure
# # x5=[0,0,1/6,1]
# # y5=[0,0.1739,0.4348,1]
# # x6=[0,1]
# # y6=[0,1]
# #
# # # # ### 6h 6h 6h
# # # x2=[0,1,1,1,1,1,1]
# # # # y2=[0,0.5652,0.6812,0.7101,0.7101,0.7101,1]
# # # y2=[0,0.5362,0.6667,0.7101,0.7101,0.7101,1]
# # # x4=[0,0,1/6,1/6,1/6,1/6,1]
# # # # y4=[0,0.3188,0.3478,0.3623,0.3913,0.4348,1]
# # # y4=[0,0.2754,0.3043,0.3043,0.3913,0.4348,1]
# # # x7=[0,2/3,1,1,1,1,1]
# # # y7=[0,0.3768,0.6522,0.7101,0.7101,0.7101,1]
# # # x8=[0,0,1/6,1/6,1/6,1/6,1]
# # # y8=[0,0.087,0.2174,0.3188,0.3913,0.4348,1]
# # # x9=[0,1/3,1/3,1,1,1,1]
# # # # y9=[0,0.5507,0.5507,0.797,0.797,0.797,1]
# # # y9=[0,0.4928,0.4928,0.797,0.797,0.797,1]
# # # x10=[0,0,0,0,0,1/6,1]
# # # # y10=[0,0.3188,0.3188,0.3623,0.3623,0.4348,1]
# # # y10=[0,0.2609,0.2609,0.2754,0.2754,0.4348,1]
# # # x11=[0,5/6,5/6,1,1,1,1]
# # # y11=[0,0.4783,0.4783,0.7101,0.7101,0.7101,1]
# # # x12=[0,0,1/6,1/6,1/6,1/6,1]
# # # y12=[0,0.1159,0.3043,0.3333,0.4348,0.4348,1]
# # # x13=[0,0,2/3,1,1,1,1]
# # # y13=[0,0,0.5362,0.7101,0.7101,0.7101,1]
# # # x14=[0,0,1/6,1/6,1/6,1/6,1]
# # # y14=[0,0.1739,0.3043,0.4348,0.4348,0.4348,1]
# #
# #
# # ### 24h 24h 24h 24h
# # x2=[0,5/6,1,1]
# # y2=[0,0.6812,0.7971,1]
# # x4=[0,1/6,1/6,1/6,1/6,1]
# # y4=[0,0.3188,0.3333,0.3913,0.4348,1]
# # x7=[0,5/6,1,1]
# # y7=[0,0.6812,0.7971,1]
# # x8=[0,1/6,1/6,1/6,1/6,1]
# # y8=[0,0.2464,0.3333,0.4058,0.4348,1]
# # x9=[0,1,1,1]
# # y9=[0,0.7681,0.7971,1]
# # x10=[0,1/6,1/6,1/6,1/6,1]
# # y10=[0,0.2754,0.2899,0.4058,0.4348,1]
# # x11=[0,2/3,1,1]
# # y11=[0,0.4783,0.7971,1]
# # x12=[0,1/6,1/6,1]
# # y12=[0,0.3478,0.4348,1]
# # x13=[0,2/3,1,1]
# # y13=[0,0.4928,0.7971,1]
# # x14=[0,0,1/6,1/6,1]
# # y14=[0,0.1014,0.3188,0.4348,1]
# # x15=[0,2/3,1,1]
# # y15=[0,0.4493,0.7971,1]
# # x16=[0,1/6,1/6,1]
# # y16=[0,0.2899,0.4348,1]
# # x17=[0,1/2,1,1]
# # y17=[0,0.5652,0.7971,1]
# # x18=[0,0,0,1/6,1]
# # y18=[0,0.1159,0.3768,0.4348,1]
#
#
# # ### 12h
# # x2=[0,1,1]
# # y2=[0,0.7101,1]
# # x4=[0,1/6,1]
# # y4=[0,0.4058,1]
# # x7=[0,1,1,1]
# # y7=[0,0.4493,0.7101,1]
# # x8=[0,1/6,1/6,1]
# # y8=[0,0.3043,0.4058,1]
# # x9=[0,1/3,1,1]
# # y9=[0,0.4638,0.797,1]
# # x10=[0,0,1/6,1]
# # y10=[0,0.2754,0.4348,1]
# # x11=[0,1,1,1]
# # y11=[0,0.4493,0.7101,1]
# # x12=[0,1/6,1/6,1/6,1/6,1]
# # y12=[0,0.2464,0.3043,0.3333,0.4058,1]
# # x13=[0,1,1]
# # y13=[0,0.7101,1]
# # x14=[0,0,1/6,1]
# # y14=[0,0.1739,0.4348,1]
# # x15=[0,1,1,1]
# # y15=[0,0.4203,0.7971,1]
# # x16=[0,0,1/6,1]
# # y16=[0,0.1739,0.4348,1]
# # x17=[0,1/3,1,1]
# # y17=[0,0.4638,0.7971,1]
# # x18=[0,0,1/6,1]
# # y18=[0,0.2754,0.4348,1]
#
# # ### 1 h
# # x7=[0,1/3,1/2,1]
# # y7=[0,0.2609,0.5072,1]
# # x8=[0,0,0,1]
# # y8=[0,0.2464,0.2754,1]
# # x11=[0,1/2,2/3,1]
# # y11=[0,0.2609,0.5652,1]
# # x12=[0,1/6,1/6,1]
# # y12=[0,0.2464,0.3188,1]
# # x13=[0,2/3,1]
# # y13=[0,0.7101,1]
# # x14=[0,0,0,1]
# # y14=[0,0.1739,0.3623,1]
#
# # ### 5 min
# # x13=[0,5/6,1]
# # y13=[0,0.7536,1]
# # x14=[0,0,1]
# # y14=[0,0.4058,1]
#
# from sklearn.metrics import auc
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y5,x5)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y2,x2)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y4,x4)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y7,x7)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y8,x8)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y9,x9)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y10,x10)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y11,x11)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y12,x12)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y13,x13)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y14,x14)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y15,x15)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y16,x16)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y17,x17)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y18,x18)))


# ### ACT0128 ACT0128 ACT0128 ACT0128 pure
# x5=[0,2/3,1]
# y5=[0,0.25,1]
# x6=[0,1]
# y6=[0,1]
#
# # #### 6h 6h 6h
# # x2=[0,2/3,1,1]
# # y2=[0,0.4167,0.5694,1]
# # # y2=[0,0.3611,0.5694,1]
# # x4=[0,1/3,2/3,1]
# # y4=[0,0.1667,0.1806,1]
# # # y4=[0,0.1389,0.1806,1]
# # x13=[0,1,1]
# # y13=[0,0.569,1]
# # x14=[0,2/3,1]
# # y14=[0,0.181,1]
# # x17=[0,2/3,1,1]
# # y17=[0,0.5556,0.9444,1]
# # # y17=[0,0.472,0.9444,1]
# # x18=[0,1/3,2/3,1]
# # y18=[0,0.2222,0.25,1]
# # # y18=[0,0.167,0.25,1]
#
# # #### 24h 24h 24h
# # x2=[0,0,0,1]
# # y2=[0,0.097222,0.1805556,1]
# # x4=[0,0,1]
# # y4=[0,0.027778,1]
# # x7=[0,0,0,0,0,1]
# # y7=[0,0.18055556,0.22222,0.361111,0.375,1]
# # x8=[0,0,0,1]
# # y8=[0,0.027778,0.0417,1]
# # x9=[0,1/3,2/3,1]
# # y9=[0,0.4305556,0.5694444,1]
# # x10=[0,1/3,1/3,1]
# # y10=[0,0.0972222,0.1527778,1]
# # x11=[0,0,0,1]
# # y11=[0,0.11111,0.375,1]
# # x12=[0,0,1]
# # y12=[0,0.04166667,1]
# # x13=[0,2/3,1,1]
# # y13=[0,0.41666667,0.9444,1]
# # x14=[0,1/3,2/3,1]
# # y14=[0,0.083333,0.25,1]
# # x15=[0,2/3,1]
# # y15=[0,0.5694444,1]
# # x16=[0,1/3,1]
# # y16=[0,0.1527778,1]
# # x17=[0,2/3,1,1]
# # y17=[0,0.5833,0.9444,1]
# # x18=[0,2/3,2/3,1]
# # y18=[0,0.1388889,0.25,1]
#
#
# # x2=[0,0,1/3,1]
# # y2=[0,0.041666,0.1527778,1]
# # x4=[0,0,1]
# # y4=[0,0.0138889,1]
# # x7=[0,0,1/3,1/3,1]
# # y7=[0,0.083333,0.3194444,0.333333,1]
# # x8=[0,0,1]
# # y8=[0,0.027778,1]
# # x9=[0,1/3,2/3,1]
# # y9=[0,0.2916667,0.666667,1]
# # x10=[0,0,0,1]
# # y10=[0,0.04166667,0.1944444,1]
# # x11=[0,0,1/3,1]
# # y11=[0,0.0277778,0.333333,1]
# # x12=[0,0,1]
# # y12=[0,0.02777778,1]
# # x13=[0,1/3,1,1]
# # y13=[0,0.388889,0.944444,1]
# # x14=[0,1/3,2/3,1]
# # y14=[0,0.083333,0.25,1]
# # x15=[0,2/3,1]
# # y15=[0,0.666667,1]
# # x16=[0,0,1]
# # y16=[0,0.1944444,1]
# # x17=[0,2/3,1,1]
# # y17=[0,0.44444,0.9444,1]
# # x18=[0,2/3,2/3,1]
# # y18=[0,0.0972222,0.25,1]
#
#
# # # #### 12h 12h 12h
# # x2=[0,0,1/3,1/3,1]
# # y2=[0,0.1111,0.1667,0.22222,1]
# # x4=[0,0,1]
# # y4=[0,0.0139,1]
# # x7=[0,1/3,2/3,1]
# # y7=[0,0.1667,0.4306,1]
# # x8=[0,1/3,1/3,1]
# # y8=[0,0.0417,0.0833,1]
# # x9=[0,0,2/3,2/3,2/3,1]
# # y9=[0,0.25,0.375,0.4167,0.5278,1]
# # x10=[0,0,1/3,1/3,1]
# # y10=[0,0.0417,0.0833,0.1111,1]
# # x11=[0,2/3,1]
# # y11=[0,0.4306,1]
# # x12=[0,1/3,1]
# # y12=[0,0.0833,1]
# # x13=[0,2/3,1,1]
# # y13=[0,0.5556,0.9444,1]
# # x14=[0,1/3,2/3,1]
# # y14=[0,0.1944,0.25,1]
# # x15=[0,2/3,2/3,1]
# # y15=[0,0.375,0.5278,1]
# # x16=[0,1/3,1/3,1]
# # y16=[0,0.0833,0.111,1]
# # x17=[0,2/3,1,1]
# # y17=[0,0.5833,0.9444,1]
# # x18=[0,1/3,2/3,1]
# # y18=[0,0.0694,0.25,1]
#
# # # #### 1h 1h 1h
# x2=[0,1/3,1]
# y2=[0,0.388889,1]
# x4=[0,1/3,1]
# y4=[0,0.166667,1]
# x7=[0,1/3,1]
# y7=[0,0.388889,1]
# x8=[0,1/3,1]
# y8=[0,0.166667,1]
# x11=[0,1/3,1]
# y11=[0,0.472222,1]
# x12=[0,1/3,1]
# y12=[0,0.166667,1]
# x13=[0,1,1]
# y13=[0,0.777778,1]
# x14=[0,2/3,1]
# y14=[0,0.25,1]
# x17=[0,2/3,1,1]
# y17=[0,0.5694,0.8889,1]
# x18=[0,2/3,1]
# y18=[0,0.25,1]
#
# # # #### 5 min
# # x7=[0,0,0,1]
# # y7=[0,0.2916667,0.388889,1]
# # x8=[0,0,0,1]
# # y8=[0,0.097222,0.1388889,1]
# # x11=[0,0,0,1]
# # y11=[0,0.361111,0.583333,1]
# # x12=[0,0,1]
# # y12=[0,0.15277778,1]
# # x13=[0,1,1,1]
# # y13=[0,0.6944444,0.777778,1]
# # x14=[0,2/3,2/3,1]
# # y14=[0,0.22222,0.25,1]
#
# from sklearn.metrics import auc
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y5,x5)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y2,x2)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y4,x4)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y7,x7)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y8,x8)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y9,x9)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y10,x10)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y11,x11)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y12,x12)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y13,x13)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y14,x14)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y15,x15)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y16,x16)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y17,x17)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y18,x18)))






# # # # QLD1230 QLD1230 QLD1230 QLD1230 pure
# x5=[0,0,1/2,1]
# y5=[0,0.076923,0.282051,1]
# x6=[0,1]
# y6=[0,1]
#
# # # #### 6h 6h 6h
# x2=[0,0,0,0,0,1]
# y2=[0,0.025641,0.038462,0.051282,0.076923,1]
# x4=[0,0,0,1]
# y4=[0,0.012821,0.025641,1]
# x7=[0,0,1,1,1]
# y7=[0,0.038462,0.448718,0.641026,1]
# x8=[0,0,1,1,1]
# y8=[0,0.064103,0.153846,0.16667,1]
# x11=[0,0,1,1]
# y11=[0,0.564103,0.730769,1]
# x12=[0,0,0,1,1]
# y12=[0,0.0641,0.076923,0.179487,1]
# x13=[0,1,1,1]
# y13=[0,0.474359,0.897436,1]
# x14=[0,0,1,1,1]
# y14=[0,0.025641,0.179487,0.269231,1]
# x15=[0,0,0,1]
# y15=[0,0.076923,0.115385,1]
# x16=[0,0,0,1]
# y16=[0,0.025641,0.051282,1]
#
# # #### 24h
# # x17=[0,1,1]
# # y17=[0,0.9487,1]
# # x18=[0,1,1,1]
# # y18=[0,0.2051,0.2821,1]
# #
# # # # #### 12h 12h 12h
# # # x7=[0,0,1,1,1]
# # # y7=[0,0.4231,0.6026,0.8974,1]
# # # x8=[0,0,1,1,1]
# # # y8=[0,0.1667,0.2179,0.2821,1]
# # # x11=[0,0,0,1,1]
# # # y11=[0,0.4231,0.859,0.9231,1]
# # # x12=[0,0,1,1]
# # # y12=[0,0.1667,0.2821,1]
# # # x13=[0,1,1,1]
# # # y13=[0,0.4231,0.9359,1]
# # # x14=[0,0,1,1,1]
# # # y14=[0,0.0769,0.1538,0.2821,1]
# #
# # # # #### 1h 1h 1h
# # # x2=[0,0,0,0,0,1]
# # # y2=[0,0.0641,0.0769,0.1282,0.1538,1]
# # # x4=[0,0,1]
# # # y4=[0,0,1]
# # # x9=[0,0,0,0,0,1]
# # # y9=[0,0.0641,0.0769,0.1282,0.1538,1]
# # # x10=[0,0,1]
# # # y10=[0,0.0128,1]
# # # x11=[0,1,1]
# # # y11=[0,0.859,1]
# # # x12=[0,0,1,1]
# # # y12=[0,0.0641,0.2564,1]
# # # x15=[0,0,0,1]
# # # y15=[0,0.141,0.1795,1]
# # # x16=[0,0,1]
# # # y16=[0,0.0128,1]
# # # x17=[0,1,1,1]
# # # y17=[0,0.5385,0.9103,1]
# # # x18=[0,1,1,1,1]
# # # y18=[0,0.0897,0.1667,0.2692,1]
# #
# # # #### 5 min
# # x15=[0,1,1,1,1]
# # y15=[0,0.1282,0.2051,0.3462,1]
# # x16=[0,1,1]
# # y16=[0,0.0128,1]
# #
# from sklearn.metrics import auc
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y5,x5)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y2,x2)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y4,x4)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y7,x7)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y8,x8)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y9,x9)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y10,x10)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y11,x11)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y12,x12)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y13,x13)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y14,x14)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y15,x15)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y16,x16)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y17,x17)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y18,x18)))




# ## VIC0829 VIC0829 VIC0829 VIC0829 pure
# x5=[0,1/4,1]
# y5=[0,0.2469136,1]
# x6=[0,1]
# y6=[0,1]
#
#
# # # #### 24h 24h 24h
# x2=[0,0,0,0,0,1]
# y2=[0,0.03704,0.07407,0.09877,0.1111,1]
# x4=[0,0,1]
# y4=[0,0,1]
# x7=[0,0,0,0,1]
# y7=[0,0.03704,0.07407,0.1111,1]
# x8=[0,0,1]
# y8=[0,0,1]
# x11=[0,0,0,1]
# y11=[0,0.07407,0.16049,1]
# x12=[0,0,1]
# y12=[0,0.09877,1]
# x13=[0,0,1/4,1]
# y13=[0,0.09877,0.20988,1]
# x14=[0,0,1]
# y14=[0,0.06173,1]
# x17=[0,1/2,1,1]
# y17=[0,0.37037,0.95062,1]
# x18=[0,0,1/4,1]
# y18=[0,0.09877,0.24691,1]


# ##### 12h 12h 12h
# # x2=[0,1/4,1/4,1/4,1]
# # y2=[0,0.11111,0.12346,0.16049,1]
# # x4=[0,1/4,1/4,1]
# # y4=[0,0.02469,0.06173,1]
# x2=[0,0,0,1]
# y2=[0,0.03704,0.04938,1]
# x4=[0,0,1]
# y4=[0,0,1]
#
# x7=[0,0,0,0,1]
# y7=[0,0,0.02469,0.049383,1]
# x8=[0,0,1]
# y8=[0,0,1]
# x11=[0,0,3/4,1]
# y11=[0,0,0.53086,1]
# x12=[0,0,1]
# y12=[0,0.09877,1]
# x13=[0,0,1/4,1/4,1]
# y13=[0,0,0.09877,0.16049,1]
# x14=[0,1/4,1/4,1]
# y14=[0,0.02469,0.06173,1]
# x17=[0,3/4,1,1]
# y17=[0,38/81,77/81,1]
# x18=[0,1/4,1/4,1]
# y18=[0,0.12346,0.24691,1]


# # #### 6h 6h 6h
# x2=[0,0,0,1]
# y2=[0,0.0246914,0.0493827,1]
# # x2=[0,0,1]
# # y2=[0,0.0493827,1]
# x4=[0,0,1]
# y4=[0,0,1]
# x7=[0,0,1]
# y7=[0,0.049383,1]
# x8=[0,0,1]
# y8=[0,0,1]
# x11=[0,0,1]
# y11=[0,0.123457,1]
# x12=[0,0,1]
# y12=[0,0.074074,1]
# x13=[0,1/4,1/2,1]
# y13=[0,0.17284,0.283951,1]
# x14=[0,0,1/4,1]
# y14=[0,0.012346,0.037037,1]
# x17=[0,3/4,1,1]
# y17=[0,0.5061728,0.9506173,1]
# x18=[0,1/4,1/4,1]
# y18=[0,0.1234568,0.2469136,1]
# # x17=[0,1/2,1,1]
# # y17=[0,0.5185185,0.9506173,1]
# # x18=[0,0,1/4,1/4,1]
# # y18=[0,0.0246914,0.1358025,0.2345679,1]


# # #### 1h 1h 1h
# x2=[0,0,1/4,1]
# y2=[0,0.11111,0.16049,1]
# x4=[0,0,1/4,1]
# y4=[0,0.03704,0.04938,1]
# x13=[0,0,1/4,1]
# y13=[0,0.18519,0.24691,1]
# x14=[0,0,1/4,1]
# y14=[0,0.06173,0.07407,1]
# x17=[0,1/2,1]
# y17=[0,0.65432,1]
# x18=[0,1/4,1]
# y18=[0,0.14815,1]

# #### 5 min
# x2=[0,0,1]
# y2=[0,0.03704,1]
# x4=[0,0,1]
# y4=[0,0.01235,1]
# x7=[0,0,1]
# y7=[0,0.049383,1]
# x8=[0,0,1]
# y8=[0,0.01235,1]
# x11=[0,1/4,1]
# y11=[0,0.32099,1]
# x12=[0,0,1]
# y12=[0,0.03704,1]
# x13=[0,1/2,1/2,1]
# y13=[0,0.23457,0.40741,1]
# x14=[0,1/4,1/4,1]
# y14=[0,0.07407,0.08642,1]
# x17=[0,3/4,1]
# y17=[0,0.82716,1]
# x18=[0,1/4,1/4,1]
# y18=[0,0.1234568,0.14815,1]


from sklearn.metrics import auc
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y5,x5)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y2,x2)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y4,x4)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y7,x7)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y8,x8)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y9,x9)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y10,x10)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y11,x11)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y12,x12)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y13,x13)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y14,x14)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y15,x15)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y16,x16)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y17,x17)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y18,x18)))






# # ### QLD1282 QLD1282 QLD1282 QLD1282 pure
# x5=[0,2/3,2/3,1]
# y5=[0,0.2308,0.2692,1]
# x6=[0,1]
# y6=[0,1]
#
#
#
# # # ### 24h 24h 24h 24h
# # x2=[0,0,0,1]
# # y2=[0,0.1975,0.2222,1]
# # # x4=[0,0,1]
# # # y4=[0,0,1]
# # x7=[0,0,0,1]
# # y7=[0,0.1481,0.2222,1]
# # # x8=[0,0,1]
# # # y8=[0,0,1]
# # x11=[0,0,0,1]
# # y11=[0,0.1358,0.2222,1]
# # # x12=[0,0,1]
# # # y12=[0,0,1]
# # x13=[0,1,1]
# # y13=[0,72/81,1]
# # # x14=[0,4/7,4/7,1]
# # # y14=[0,0.2716,0.284,1]
# # x17=[0,1,1]
# # y17=[0,72/81,1]
# # # x18=[0,4/7,4/7,1]
# # # y18=[0,0.2593,0.284,1]
#
# # ### 12h 12h 12h
# # x2=[0,0,0,1]
# # y2=[0,0,0.1605,1]
# # # x4=[0,0,1]
# # # y4=[0,0,1]
# # x7=[0,0,1]
# # y7=[0,0.1605,1]
# # # x8=[0,0,1]
# # # y8=[0,0,1]
# # x11=[0,0,0,1]
# # y11=[0,0.1111,0.1605,1]
# # # x12=[0,0,1]
# # # y12=[0,0,1]
# # x13=[0,1,1]
# # y13=[0,0.8889,1]
# # # x14=[0,4/7,4/7,1]
# # # y14=[0,0.2469,0.284,1]
# # x15=[0,8/9,8/9,1]
# # y15=[0,34/81,61/81,1]
# # # x16=[0,3/7,3/7,3/7,1]
# # # y16=[0,0.1481,0.1728,0.2222,1]
#
#
# # # ### 6h 6h 6h
# # x2=[0,1/9,1/9,1]
# # y2=[0,0.0617,0.1111,1]
# # # x4=[0,0,1]
# # # y4=[0,0.0247,1]
# # x7=[0,0.1111,1]
# # y7=[0,0.1728,1]
# # # x8=[0,0,1]
# # # y8=[0,0.0247,1]
# # x11=[0,0.1111,0.1111,1]
# # y11=[0,0.1235,0.1728,1]
# # # x12=[0,0,1]
# # # y12=[0,0.0247,1]
# # x13=[0,1,1]
# # y13=[0,0.8889,1]
# # # x14=[0,1/7,4/7,4/7,4/7,1]
# # # y14=[0,0.0617,0.2469,0.2592,0.284,1]
# # x15=[0,1,1,1]
# # y15=[0,25/81,35/81,1]
# # # x16=[0,4/7,4/7,1]
# # # y16=[0,0.1358,0.2593,1]
#
#
# # ### 1 h
# # x15=[0,7/9,8/9,1,1]
# # y15=[0,0.3086,0.4444,0.6173,1]
# # # x16=[0,4/7,4/7,4/7,1]
# # # y16=[0,0.1728,0.2099,0.284,1]
#
# # # ### 5 min
# x9=[0,1,1,1,1,1,1]
# y9=[0,53/81,57/81,58/81,59/81,59/81,1]
# # x10=[0,4/7,4/7,1]
# # y10=[0,0.2222,0.2593,1]
# x15=[0,8/9,1,1]
# y15=[0,33/81,60/81,1]
# # x16=[0,3/7,3/7,4/7,1]
# # y16=[0,0.1852,0.1975,0.2593,1]
# x17=[0,8/9,1,1]
# y17=[0,70/81,72/81,1]
# # x18=[0,4/7,4/7,1]
# # y18=[0,0.2593,0.284,1]
#
#
# from sklearn.metrics import auc
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y5,x5)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y2,x2)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y4,x4)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y7,x7)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y8,x8)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y9,x9)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y10,x10)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y11,x11)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y12,x12)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y13,x13)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y14,x14)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y15,x15)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y16,x16)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y17,x17)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y18,x18)))




# ### QLD2307 QLD2307 QLD2307 QLD2307
# x5=[0,0,5/7,1]
# y5=[0,0.0897,0.1667,1]
# x6=[0,1]
# y6=[0,1]
#
# # # ### 6h 6h 6h
# x2=[0,2/7,5/7,1]
# y2=[0,0.6049,0.8025,1]
# # x4=[0,2/7,4/7,4/7,1]
# # y4=[0,0.0617,0.0864,0.1235,1]
# x7=[0,1,1]
# y7=[0,0.8889,1]
# # x8=[0,4/7,1]
# # y8=[0,0.1358,1]
# x9=[0,3/7,6/7,6/7,1]
# y9=[0,0.4568,0.7901,0.8395,1]
# # x10=[0,3/7,5/7,5/7,1]
# # y10=[0,0.0617,0.0741,0.1481,1]
# x11=[0,2/7,1,1]
# y11=[0,10/81,72/81,1]
# # x12=[0,0,4/7,4/7,4/7,1]
# # y12=[0,0.037,0.0741,0.0864,0.1358,1]
# x13=[0,1,1]
# y13=[0,0.9136,1]
# # x14=[0,5/7,1]
# # y14=[0,0.1605,1]
# x15=[0,3/7,6/7,1]
# y15=[0,0.4691,0.8395,1]
# # x16=[0,3/7,5/7,5/7,1]
# # y16=[0,0.0617,0.0864,0.1481,1]
# x17=[0,1,1]
# y17=[0,0.9136,1]
# # x18=[0,5/7,5/7,1]
# # y18=[0,0.0864,0.1605,1]


# ### 24h 24h 24h 24h
# x9=[0,2/7,2/7,2/7,1]
# y9=[0,0.1728,0.284,0.3333,1]
# # x10=[0,1/7,1/7,1]
# # y10=[0,0.0494,0.0617,1]
# x15=[0,2/7,2/7,1]
# y15=[0,0.1728,0.3333,1]
# # x16=[0,1/7,1/7,1]
# # y16=[0,0.0494,0.0617,1]
# x17=[0,2/7,1,1]
# y17=[0,0.5185,0.9136,1]
# # x18=[0,1/7,5/7,5/7,1]
# # y18=[0,0.0247,0.1111,0.1605,1]


# # # ### 12h 12h 12h
# x9=[0,3/7,4/7,5/7,1]
# y9=[0,0.4815,0.5926,0.6049,1]
# # x10=[0,1/7,2/7,4/7,4/7,1]
# # y10=[0,0.0988,0.1111,0.1111,0.1235,1]
# x15=[0,3/7,5/7,1]
# y15=[0,0.4815,0.6049,1]
# # x16=[0,1/7,1/7,4/7,4/7,1]
# # y16=[0,0.037,0.0988,0.1111,0.1235,1]
# x17=[0,1,1]
# y17=[0,0.9136,1]
# # x18=[0,5/7,1]
# # y18=[0,0.1605,1]


### 1 h
# x2=[0,0,1/7,2/7,2/7,1]
# y2=[0,0.0988,0.358,0.4074,0.4444,1]
# x4=[0,0,0,0,0,1]
# y4=[0,0.0247,0.037,0.0741,0.0864,1]
# x11=[0,2/7,4/7,5/7,1]
# y11=[0,0.1975,0.5926,0.679,1]
# x12=[0,0,0,1/7,1]
# y12=[0,0.037,0.1111,0.1358,1]
# x15=[0,4/7,6/7,1,1]
# y15=[0,0.5185,0.7284,0.7407,1]
# x16=[0,3/7,5/7,1]
# y16=[0,0.037,0.1358,1]

# ### 5 min
# x15=[0,2/7,1,1]
# y15=[0,0.4074,0.9012,1]
# x16=[0,0,5/7,5/7,1]
# y16=[0,0.037,0.1111,0.1605,1]


from sklearn.metrics import auc
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y5,x5)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y2,x2)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y4,x4)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y7,x7)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y8,x8)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y9,x9)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y10,x10)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y11,x11)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y12,x12)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y13,x13)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y14,x14)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y15,x15)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y16,x16)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y17,x17)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y18,x18)))




# # ### VIC2835
# x5=[0,1/4,1]
# y5=[0,0.2727,1]
# x6=[0,1]
# y6=[0,1]
#
# # # # ### 6h 6h 6h
# # x2=[0,1/2,1]
# # y2=[0,0.2121,1]
# # x4=[0,1/4,1]
# # y4=[0,0,1]
# #
# # x9=[0,1/2,1]
# # y9=[0,0.2121,1]
# # x10=[0,1/4,1]
# # y10=[0,0.0303,1]
# #
# # x13=[0,1/2,1,1]
# # y13=[0,0.6667,0.8788,1]
# # x14=[0,1/4,1/4,1]
# # y14=[0,0.1212,0.2424,1]
# #
# # x15=[0,1/2,1]
# # y15=[0,0.2121,1]
# # x16=[0,1/4,1]
# # y16=[0,0.0303,1]
# # x17=[0,1,1]
# # y17=[0,0.8788,1]
# # x18=[0,1/4,1]
# # y18=[0,0.2727,1]
#
#
# # # ### 24h 24h 24h 24h
# x2=[0,1/2,1]
# y2=[0,4/33,1]
# x4=[0,1/4,1]
# y4=[0,0.0303,1]
#
# x7=[0,1/2,1/2, 1]
# y7=[0,0.2727, 0.3333,1]
# x8=[0,1/4,1/4,1]
# y8=[0,0.0303,0.0606,1]
# x9=[0,1/2,1]
# y9=[0,0.1515,1]
# x10=[0,1/4,1]
# y10=[0,0.0303,1]
#
# x11=[0,3/4,3/4,1]
# y11=[0,0.3636,0.4242,1]
# x12=[0,1/4,1/4,1]
# y12=[0,0.0606,0.0909,1]
# x13=[0,1/2,1]
# y13=[0,0.5455,1]
# x14=[0,1/4,1]
# y14=[0,0.1212,1]
#
# x15=[0,0,1/2,1]
# y15=[0,0, 8/33,1]
# x16=[0,1/4,1]
# y16=[0,0.0303,1]
# x17=[0,1/2,1]
# y17=[0,0.4848,1]
# x18=[0,1/4,1]
# y18=[0,0.1212,1]


# # # # ### 12h 12h 12h
# # x2=[0,0,1]
# # y2=[0,0.0303,1]
# # x4=[0,0,1]
# # y4=[0,0,1]
# #
# # x7=[0,1/2,1/2,1]
# # y7=[0,0.2424,0.2727,1]
# # x8=[0,0,1]
# # y8=[0,0.0606,1]
# # x9=[0,0,0,1]
# # y9=[0,0.1818,0.2121,1]
# # x10=[0,0,1]
# # y10=[0,0.0303,1]
# # x11=[0,1/2,1,1]
# # y11=[0,0.2424,0.6667,1]
# # x12=[0,0,1/4,1]
# # y12=[0,0.1212,0.2727,1]
# # x13=[0,1/2,1/2,1]
# # y13=[0,0.4545,0.5152,1]
# # x14=[0,0,0,1]
# # y14=[0,0.0606,0.09091,1]
# # x15=[0,0,0,1]
# # y15=[0,0.0909,0.2121,1]
# # x16=[0,0,1]
# # y16=[0,0.0303,1]
# # x17=[0,1/4,1,1]
# # y17=[0,0.5455,0.8788,1]
# # x18=[0,0,0,1/4,1]
# # y18=[0,0.1515,0.1818,0.2727,1]
#
#
# # ### 1 h
# # x2=[0,0,0,0,1]
# # y2=[0,0.1212,0.2424,0.2727,1]
# # x4=[0,0,1]
# # y4=[0,0.0303,1]
# #
# # x9=[0,1/4,1/2,1/2,1]
# # y9=[0,0.2121, 0.303, 0.3939, 1]
# # x10=[0,0,1]
# # y10=[0,0.0606,1]
# #
# # x13=[0,3/4,1]
# # y13=[0,0.7879,1]
# # x14=[0,0,1]
# # y14=[0,0.2121,1]
# #
# # x15=[0,1/4,1/2,1]
# # y15=[0,0.2727,0.5152,1]
# # x16=[0,0,1]
# # y16=[0,0.0909,1]
# # x17=[0,1,1]
# # y17=[0,0.7576,1]
# # x18=[0,1/4,1]
# # y18=[0,0.1515,1]
#
# # ### 5 min
# # x15=[0,3/4,3/4,1]
# # y15=[0,0.303,0.6364,1]
# # x16=[0,1/4,1/4,1]
# # y16=[0,0.0909,0.2121,1]


# from sklearn.metrics import auc
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y5,x5)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y2,x2)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y4,x4)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y7,x7)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y8,x8)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y9,x9)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y10,x10)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y11,x11)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y12,x12)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y13,x13)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y14,x14)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y15,x15)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y16,x16)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y17,x17)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y18,x18)))



# # ### SA1243
# x5=[0,1,1]
# y5=[0,0.2593,1]
# x6=[0,1]
# y6=[0,1]

# # # ### 24h 24h 24h 24h
# x2=[0,0,0, 1/2,1/2, 1/2,1]
# y2=[0,0.08642,0.08642,0.185185,0.185185,0.259259,1]
# # x2=[0,1/2,1/2, 1/2,1]
# # y2=[0,0.185185,0.197531,0.419753,1]
# # x2=[0,0,1/2,1]
# # y2=[0,0.234568,0.506173,1]
# # x4=[0,0,1/2,1/2,1]
# # y4=[0,0.037037,0.061728,0.074074,1]
#
# x7=[0,1/2,1/2,1/2,1]
# y7=[0,0.185185,0.246914,0.259259,1]
# # x8=[0,0,0.5,0.5,0.5,0.5,1]
# # y8=[0,0.049383,0.074074,0.074074, 0.08642,0.08642,1]
#
# x11=[0,0.5,0.5,0.5,0.5,0.5,1]
# y11=[0,0.1605,0.1605,0.4198,0.4198,0.4198,1]
# # x12=[0,0.5,0.5,0.5,0.5,0.5,1]
# # y12=[0,0.074074,0.074074,0.074074,0.08642,0.08642,1]
# x13=[0,0,1/2,1/2,1/2,1/2,1]
# y13=[0,0.1975,0.4691,0.5062,0.5062,0.5062,1]
# # x14=[0,0,0, 0.5,0.5,0.5,1]
# # y14=[0,0.074074,0.074074,0.148148,0.148148,0.148148,1]
#
# x17=[0,1/2,1/2,1,1,1,1]
# y17=[0,0.4321,0.4321,0.9753,0.9753,0.9753,1]
# # x18=[0,0.5,0.5,1,1,1,1]
# # y18=[0,0.1111,0.1111,0.2593,0.2593,0.2593,1]


# # # ### 12h 12h 12h
# # x2=[0,0,0,0.5,0.5,0.5,1]
# # y2=[0,0.0247,0.1481,0.1852,0.1852,0.3704,1]
# # x2=[0,0,0,0,0,0,1]
# # y2=[0,0.0988,0.0988,0.0988,0.1358,0.1605,1]
# x2=[0,0,0,0,0,0,1]
# y2=[0,0.0247,0.0741,0.0741,0.2222,0.3086,1]
# # x4=[0,0,0,0,1]
# # y4=[0,0,0.0617,0.0864,1]
# x7=[0,0,0,0,0,0,1]
# y7=[0,0.0864,0.1605,0.1605,0.3086,0.3086,1]
# # x8=[0,0,0,1]
# # y8=[0,0.0247,0.0864,1]
# # x9=[0,0.5,0.5,0.5,0.5,0.5,1]
# # y9=[0,0.5062,0.5432,0.642,0.642,0.7531,1]
# # x10=[0,0.5,0.5,0.5,1]
# # y10=[0,0.0864,0.1605,0.1975,1]
# x11=[0,1/2,1/2,1/2,1]
# y11=[0,5/81,12/81,30/81,1]
# # x12=[0,0.5,0.5,0.5,1]
# # y12=[0,0.0123,0.0617,0.1235,1]
# x13=[0,0,0,0,1]
# y13=[0,0.2222,0.5309,0.5609,1]
# # x14=[0,0,0,0,1]
# # y14=[0,0.0494,0.1235,0.1481,1]
# # x15=[0,0,0.5,1]
# # y15=[0,0,0.7531,1]
# # x16=[0,0.5,0.5,1]
# # y16=[0,0.1111,0.1975,1]
# x17=[0,0.5,1,1]
# y17=[0,0.642,0.9753,1]
# # x18=[0,0.5,1,1]
# # y18=[0,0.1605,0.2593,1]


# # ### 6h 6h 6h
# x2=[0,0,0,0,0,1]
# y2=[0,0.0741,0.1111,0.1358,0.1605,1]
# # x2=[0,0,0,1/2,1]
# # y2=[0,0.345679,0.358025,0.4444444,1]
# # x2=[0,0,0,1]
# # y2=[0,0.123457,0.481481,1]
# # x4=[0,0,0,1]
# # y4=[0,0.037,0.0494,1]
# x7=[0,0,0,0,1]
# y7=[0,0.1235,0.1852,0.284,1]
# # x8=[0,0,0,1]
# # y8=[0,0.037,0.0988,1]
# x9=[0,0,0.5,0.5,1]
# y9=[0,0.2593,0.4938,0.5556,1]
# # x10=[0,0,0.5,1]
# # y10=[0,0.0864,0.1481,1]
# x11=[0,0,0,1]
# y11=[0,0.1481,0.357,1]
# # x12=[0,0,0,0,1]
# # y12=[0,0.0617,0.0988,0.1358,1]
# x13=[0,0.5,0.5,0.5,1]
# y13=[0,0.1728,0.358,0.5926,1]
# # x14=[0,0.5,0.5,1]
# # y14=[0,0.0617,0.1605,1]
# x15=[0,0.5,1,1]
# y15=[0,0.5679,0.642,1]
# # x16=[0,0,1,1]
# # y16=[0,0.1111,0.1728,1]
# x17=[0,0,0.5,1]
# y17=[0,0,0.8765,1]
# # x18=[0,0,0.5,1]
# # y18=[0,0,0.2222,1]


# ### 1 h
# x2=[0,0,0,0,1]
# y2=[0,0.08642,0.11111,0.148148,1]
# # x4=[0,0,0,0,1]
# # y4=[0,0.024691,0.0370371,0.049383,1]
#
# x11=[0,0,0,0,1]
# y11=[0,0,0.2716,0.3951,1]
# # x12=[0,0,0,0,1]
# # y12=[0,0,0.08642,0.148148,1]
# x15=[0,0,0.5,1]
# y15=[0,0,0.3457,1]
# # x16=[0,0,0.5,1]
# # y16=[0,0,0.0741,1]



# # # ### 5 min
# x2=[0,0,0,0.5,0.5,0.5,1]
# y2=[0,0.037,0.0741,0.1481,0.1728, 0.1728, 1]
# x4=[0,0.5,1]
# y4=[0,0.049383,1]
#
# x9=[0,0,0,0.5,0.5,0.5,1]
# y9=[0,0.2346,0.3086,0.5309,0.5556,0.5556,1]
# x11=[0,0,0.5,1]
# y11=[0,0,0.3086,1]
# x12=[0,0.5,0.5,1]
# y12=[0,0.1111,0.123457,1]
# x15=[0,0,0.5,1]
# y15=[0,0,0.6049,1]
# x16=[0,0.5,1]
# y16=[0,0.1852,1]
# x17=[0,0,0.5,0.5,1]
# y17=[0,0.5679,0.9136,0.9259,1]


# from sklearn.metrics import auc
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y5,x5)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y2,x2)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y4,x4)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y7,x7)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y8,x8)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y9,x9)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y10,x10)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y11,x11)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y12,x12)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y13,x13)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y14,x14)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y15,x15)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y16,x16)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y17,x17)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y18,x18)))



# ### VIC0821
# x5=[0,1/2,1/2,1/2,1]
# y5=[0,0.246377,0.246377,0.246377,1]
# x6=[0,1]
# y6=[0,1]
#
# # # # ### 24h 24h 24h 24h
# x2=[0,1/2,1/2,2/3,1,1,1]
# y2=[0,0.5217,0.7101,0.942,0.942,0.942,1]
# # x2=[0,1/2,1/2, 1/2, 1/2, 1, 1]
# # y2=[0,0.376812,0.681159,0.710145,0.782609,0.942029,1]
# # x4=[0,1]
# # y4=[0,1]
# x7=[0,1/2,1/2,1/2,1/2,1,1]
# y7=[0,0.4348,0.4348,0.7826,0.7826,0.942,1]
# # x8=[0,1]
# # y8=[0,1]
# x9=[0,0,0.5,1,1,1,1]
# y9=[0,0,0.6522,0.7536,0.942,0.942,1]
# # x10=[0,1]
# # y10=[0,1]
# x11=[0,0,0.5,1,1,1,1]
# y11=[0,0,0.5942,0.942,0.942,0.942,1]
# # x12=[0,1]
# # y12=[0,1]
# x13=[0,0.5,0.5,1,1,1,1]
# y13=[0,0.6232,0.6232,0.942,0.942,0.942,1]
# # x14=[0,1]
# # y14=[0,1]
# x15=[0,0,1,1,1,1,1]
# y15=[0,0,0.942,0.942,0.942,0.942,1]
# # x16=[0,1]
# # y16=[0,1]
# x17=[0,0,1/2,1,1,1,1]
# y17=[0,0,0.6522,0.942,0.942,0.942,1]
# # x18=[0,1]
# # y18=[0,1]
#
#
# # # # ### 12h 12h 12h
# # x2=[0,1/2,1/2,1/2,1/2,1/2,1]
# # y2=[0,0.1594,0.1594,0.4783,0.4783,0.4783,1]
# # x4=[0,1]
# # y4=[0,1]
# #
# # x7=[0,0,0.5,0.5,0.5,0.5,1]
# # y7=[0,0,0.1594,0.4638,0.5362,0.5362,1]
# # x8=[0,1]
# # y8=[0,1]
# #
# # x11=[0,0,1/2,1/2,1/2,1/2,1]
# # y11=[0,0,0.1739,0.5362,0.5362,0.5362,1]
# # x12=[0,1]
# # y12=[0,1]
# # x13=[0,0,1,1,1,1,1]
# # y13=[0,0,0.942,0.942,0.942,0.942,1]
# # x14=[0,1]
# # y14=[0,1]
# # x17=[0,0.5,0.5,0.5,0.5,0.5,1]
# # y17=[0,0.6087,0.6087,0.6087,0.6087,0.6087,1]
# # x18=[0,1]
# # y18=[0,1]
#
#
# # # ### 6h 6h 6h
# # x2=[0,0,0,1/2,1/2,1/2,1]
# # y2=[0,0.2899,0.2899,0.3913,0.3913,0.3913,1]
# # x4=[0,1]
# # y4=[0,1]
# #
# # x7=[0,0,0,0.5,0.5,0.5,1]
# # y7=[0,0,0.3478,0.4203,0.4348,0.4348,1]
# # x8=[0,1]
# # y8=[0,1]
# #
# # x11=[0,0,0,0.5,0.5,0.5,1]
# # y11=[0,0,0.3478,0.4348,0.4348,0.4348,1]
# # x12=[0,1]
# # y12=[0,1]
# # x13=[0,0,1,1,1,1,1]
# # y13=[0,0,0.8986,0.942,0.942,0.9421,1]
# # x14=[0,1]
# # y14=[0,1]
# #
# # x17=[0,0.5,1]
# # y17=[0,0.6377,1]
# # x18=[0,1]
# # y18=[0,1]
#
#
# # ### 1 h
# # x2=[0,0,0,0,0,0,1]
# # y2=[0,0.1449,0.2174,0.2174,0.2319,0.2319,1]
# # x4=[0,1]
# # y4=[0,1]
# #
# # x7=[0,0,0,0,0,0,1]
# # y7=[0,0,0.1014,0.2754,0.4638,0.4638,1]
# # x8=[0,1]
# # y8=[0,1]
# # x9=[0,0,0,0.5,0.5,0.5,1]
# # y9=[0,0.5072,0.5072,0.7536,0.7536,0.7536,1]
# # x10=[0,1]
# # y10=[0,1]
# #
# # x11=[0,0,0,1,1,1,1]
# # y11=[0,0,0,0.6957,0.6957,0.6957,1]
# # x12=[0,1]
# # y12=[0,1]
# # x13=[0,0,0,1,1,1,1]
# # y13=[0,0,0.2609,0.7391,0.7391,0.7391,1]
# # x14=[0,1]
# # y14=[0,1]
#
#
#
# # # ### 5 min
# # x2=[0,0,0,0.5,0.5,0.5,1]
# # y2=[0,0,0,0.6812,0.6957,0.6957,1]
# # # x2=[0,0,0,0,0,0,1]
# # # y2=[0,0.029,0.087,0.1014,0.1159,0.1159,1]
# # x4=[0,1]
# # y4=[0,1]
# #
# #
# # x13=[0,0,1,1,1,1,1]
# # y13=[0,0,0.7826,0.9275,0.9275,0.9275,1]
# # x14=[0,1]
# # y14=[0,1]
# #
# # x15=[0,0,0,1,1,1,1]
# # y15=[0,0,0,0.8551,0.8551,0.8551,1]
# # x16=[0,1]
# # y16=[0,1]
#
#
#
from sklearn.metrics import auc
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y5,x5)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y2,x2)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y4,x4)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y7,x7)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y8,x8)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y9,x9)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y10,x10)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y11,x11)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y12,x12)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y13,x13)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y14,x14)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y15,x15)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y16,x16)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y17,x17)))
# # print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y18,x18)))





