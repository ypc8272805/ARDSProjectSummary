import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

result = pd.read_csv('result/result_copy.csv')
result = result * 100
Mnet = result.iloc[:, range(1, 345, 8)]
Mnet_s = result.iloc[:, range(2, 345, 8)]
Log = result.iloc[:, range(3, 345, 8)]
Log_s = result.iloc[:, range(4, 345, 8)]
Ada = result.iloc[:, range(5, 345, 8)]
Ada_s = result.iloc[:, range(6, 345, 8)]
XGB = result.iloc[:, range(7, 345, 8)]
XGB_s = result.iloc[:, range(8, 345, 8)] 

#绘制BER曲线
fig, ax = plt.subplots()
# 绘制MLPC
plt.plot(range(1, 44), Mnet.iloc[11, :].values, 'r-', label=r'MLPC BER mean')
mlpc_upper = np.array(Mnet) + np.array(Mnet_s)
mlpc_lower = np.array(Mnet) - np.array(Mnet_s)

plt.fill_between(range(1, 44), mlpc_lower[11, :], mlpc_upper[11, :], facecolor='grey', alpha=.4,lw=1,linestyle='--'
                 ,edgecolor='black',label=r'BER STD')
##标记BER最小值
temp = list(Mnet.iloc[11, :])
minIndex = temp.index(min(temp)) + 1
plt.plot(minIndex, temp[minIndex - 1], 'go',markersize=10,markeredgecolor='black')
plt.plot((minIndex,minIndex),(temp[minIndex-1],14),'g--',alpha=.5)
plt.plot((minIndex,-1),(temp[minIndex-1],temp[minIndex-1]),'g--',alpha=.5)

##标准差范围的最小特征值个数
upper = Mnet.iloc[11, minIndex - 1] + Mnet_s.iloc[11, minIndex - 1]
lower = Mnet.iloc[11, minIndex - 1] - Mnet_s.iloc[11, minIndex - 1]
Mnet.iloc[11, :] > lower
Mnet.iloc[11, :] < upper
plt.plot(40, temp[39], 'r^',markersize=10,markeredgecolor='black')
plt.plot((40,40),(temp[39],14),'r--',alpha=.5)
plt.plot((40,-1),(temp[39],temp[39]),'r--',alpha=.5)
#坐标轴设置
#plt.xlabel('Number of Features ',size=15)
#plt.ylabel('BER(%)',size=15)
#plt.xlim(0,44)
#plt.ylim(14,33)
#plt.title('')
#plt.legend()

# 绘制Log
plt.plot(range(1, 44), Log.iloc[11, :].values, 'b-', label=r'Log BER mean')
Log_upper = np.array(Log) + np.array(Log_s)
Log_lower = np.array(Log) - np.array(Log_s)
plt.fill_between(range(1, 44), Log_lower[11, :], Log_upper[11, :], color='grey', alpha=.5, lw=1,linestyle='--'
                 ,edgecolor='black')
##标记BER最小值
temp = list(Log.iloc[11, :])
minIndex = temp.index(min(temp)) + 1
plt.plot(minIndex, temp[minIndex - 1], 'go',markersize=10,markeredgecolor='black')
plt.plot((minIndex,minIndex),(temp[minIndex-1],14),'g--',alpha=.5)
plt.plot((minIndex,-1),(temp[minIndex-1],temp[minIndex-1]),'g--',alpha=.5)
##标准差范围的最小特征值个数
upper = Log.iloc[11, minIndex - 1] + Log_s.iloc[11, minIndex - 1]
lower = Log.iloc[11, minIndex - 1] - Log_s.iloc[11, minIndex - 1]
Log.iloc[11, :] > lower
Log.iloc[11, :] < upper
plt.plot(32, temp[31], 'r^',markersize=10,markeredgecolor='black')
plt.plot((32,32),(temp[31],14),'r--',alpha=.5)
plt.plot((32,-1),(temp[31],temp[31]),'r--',alpha=.5)
#plt.xlabel('Number of Features ',size=15)
#plt.ylabel('BER(%)',size=15)
#plt.xlim(0,44)
#plt.ylim(14,33)
#plt.title('')
#plt.legend()

# 绘制Ada
plt.plot(range(1, 44), Ada.iloc[11, :].values, 'g-', label=r'Ada BER mean')
Ada_upper = np.array(Ada) + np.array(Ada_s)
Ada_lower = np.array(Ada) - np.array(Ada_s)
plt.fill_between(range(1, 44), Ada_lower[11, :], Ada_upper[11, :], color='grey', alpha=.5, lw=1,linestyle='--'
                 ,edgecolor='black')
##标记BER最小值
temp = list(Ada.iloc[11, :])
minIndex = temp.index(min(temp)) + 1
plt.plot(minIndex, temp[minIndex - 1], 'go',markersize=10,markeredgecolor='black')
plt.plot((minIndex,minIndex),(temp[minIndex-1],14),'g--',alpha=.5)
plt.plot((minIndex,-1),(temp[minIndex-1],temp[minIndex-1]),'g--',alpha=.5)
##标准差范围的最小特征值个数
upper = Ada.iloc[11, minIndex - 1] + Ada_s.iloc[11, minIndex - 1]
lower = Ada.iloc[11, minIndex - 1] - Ada_s.iloc[11, minIndex - 1]
Ada.iloc[11, :] > lower
Ada.iloc[11, :] < upper
plt.plot(37, temp[36], 'r^',markersize=10,markeredgecolor='black')
plt.plot((37,37),(temp[36],14),'r--',alpha=.5)
plt.plot((37,-1),(temp[36],temp[36]),'r--',alpha=.5)
#坐标轴设置
#plt.xlabel('Number of Features ',size=15)
#plt.ylabel('BER(%)',size=15)
#plt.xlim(0,44)
#plt.ylim(14,33)
#plt.title('')
#plt.legend()


# 绘制XGB
plt.plot(range(1, 44), XGB.iloc[11, :].values, 'k-', label=r'XGB BER mean')
XGB_upper = np.array(XGB) + np.array(XGB_s)
XGB_lower = np.array(XGB) - np.array(XGB_s)
plt.fill_between(range(1, 44), XGB_lower[11, :], XGB_upper[11, :], color='grey', alpha=.5, lw=1,linestyle='--'
                 ,edgecolor='black')
##标记BER最小值
temp = list(XGB.iloc[11, :])
minIndex = temp.index(min(temp)) + 1
plt.plot(minIndex , temp[minIndex-1], 'go',markersize=10,markeredgecolor='black',label='min BER')
plt.plot((minIndex,minIndex),(temp[minIndex-1],14),'g--',alpha=.5)
plt.plot((minIndex,-1),(temp[minIndex-1],temp[minIndex-1]),'g--',alpha=.5)
##标准差范围的最小特征值个数
upper = XGB.iloc[11, minIndex - 1] + XGB_s.iloc[11, minIndex - 1]
lower = XGB.iloc[11, minIndex - 1] - XGB_s.iloc[11, minIndex - 1]
XGB.iloc[11, :] > lower
XGB.iloc[11, :] < upper
plt.plot(19, temp[18], 'r^',markersize=10,markeredgecolor='black',label='min feature number')
plt.plot((19,19),(temp[18],14),'r--',alpha=.5)
plt.plot((19,-1),(temp[18],temp[18]),'r--',alpha=.5)
#坐标轴设置
plt.xlabel('Number of Features ',size=15)
plt.ylabel('BER(%)',size=15)
plt.xlim(-1,44)
plt.ylim(14,33)
plt.title('')
plt.xticks([3],color='red')
plt.xticks([5],color='black')
plt.legend()

fig.set_size_inches(12,9)









#result = pd.read_csv('result.csv')
#result = result
#Mnet = result.iloc[:, range(0, 344, 8)]
#Mnet_s = result.iloc[:, range(1, 344, 8)]
#Log = result.iloc[:, range(2, 344, 8)]
#Log_s = result.iloc[:, range(3, 344, 8)]
#Ada = result.iloc[:, range(4, 344, 8)]
#Ada_s = result.iloc[:, range(5, 344, 8)]
#XGB = result.iloc[:, range(6, 344, 8)]
#XGB_s = result.iloc[:, range(7, 344, 8)]
#fig, ax = plt.subplots()
## 绘制MLPC
#plt.plot(range(1, 44), Mnet.iloc[12, :].values, 'r.-', label=r'MLPC AUC mean')
#mlpc_upper = np.array(Mnet) + np.array(Mnet_s)
#mlpc_lower = np.array(Mnet) - np.array(Mnet_s)
#plt.fill_between(range(1, 44), mlpc_lower[12, :], mlpc_upper[12, :], color='grey', alpha=.5,
#                 label=r'AUC STD')
## 绘制Log
#plt.plot(range(1, 44), Log.iloc[12, :].values, 'b.-', label=r'Log AUC mean')
#Log_upper = np.array(Log) + np.array(Log_s)
#Log_lower = np.array(Log) - np.array(Log_s)
#plt.fill_between(range(1, 44), Log_lower[12, :], Log_upper[12, :], color='grey', alpha=.5, )
#
## 绘制Ada
#plt.plot(range(1, 44), Ada.iloc[12, :].values, 'g.-', label=r'Ada AUC mean')
#Ada_upper = np.array(Ada) + np.array(Ada_s)
#Ada_lower = np.array(Ada) - np.array(Ada_s)
#plt.fill_between(range(1, 44), Ada_lower[12, :], Ada_upper[12, :], color='grey', alpha=.5, )
#
## 绘制XGB
#plt.plot(range(1, 44), XGB.iloc[12, :].values, 'k.-', label=r'XGB AUC mean')
#XGB_upper = np.array(XGB) + np.array(XGB_s)
#XGB_lower = np.array(XGB) - np.array(XGB_s)
#plt.fill_between(range(1, 44), XGB_lower[12, :], XGB_upper[12, :], color='grey', alpha=.5, )
#plt.ylabel('AUC(%)')
#plt.xlabel('Number of feature')
#plt.legend()
