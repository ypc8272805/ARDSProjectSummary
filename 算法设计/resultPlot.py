import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

result = pd.read_csv('result.csv')
result = result * 100
Mnet = result.iloc[:, range(0, 344, 8)]
Mnet_s = result.iloc[:, range(1, 344, 8)]
Log = result.iloc[:, range(2, 344, 8)]
Log_s = result.iloc[:, range(3, 344, 8)]
Ada = result.iloc[:, range(4, 344, 8)]
Ada_s = result.iloc[:, range(5, 344, 8)]
XGB = result.iloc[:, range(6, 344, 8)]
XGB_s = result.iloc[:, range(7, 344, 8)]

fig, ax = plt.subplots()
# 绘制MLPC
plt.plot(range(1, 44), Mnet.iloc[11, :].values, 'r.-', label=r'MLPC BER mean')
mlpc_upper = np.array(Mnet) + np.array(Mnet_s)
mlpc_lower = np.array(Mnet) - np.array(Mnet_s)
plt.fill_between(range(1, 44), mlpc_lower[11, :], mlpc_upper[11, :], color='grey', alpha=.5,
                 label=r'BER STD')
# 绘制Log
plt.plot(range(1, 44), Log.iloc[11, :].values, 'b.-', label=r'Log BER mean')
Log_upper = np.array(Log) + np.array(Log_s)
Log_lower = np.array(Log) - np.array(Log_s)
plt.fill_between(range(1, 44), Log_lower[11, :], Log_upper[11, :], color='grey', alpha=.5, )

# 绘制Ada
plt.plot(range(1, 44), Ada.iloc[11, :].values, 'g.-', label=r'Ada BER mean')
Ada_upper = np.array(Ada) + np.array(Ada_s)
Ada_lower = np.array(Ada) - np.array(Ada_s)
plt.fill_between(range(1, 44), Ada_lower[11, :], Ada_upper[11, :], color='grey', alpha=.5, )

# 绘制XGB
plt.plot(range(1, 44), XGB.iloc[11, :].values, 'k.-', label=r'XGB BER mean')
XGB_upper = np.array(XGB) + np.array(XGB_s)
XGB_lower = np.array(XGB) - np.array(XGB_s)
plt.fill_between(range(1, 44), XGB_lower[11, :], XGB_upper[11, :], color='grey', alpha=.5, )
plt.ylabel('BER(%)')
plt.xlabel('Number of feature')
plt.legend()


result = pd.read_csv('result.csv')
result = result 
Mnet = result.iloc[:, range(0, 344, 8)]
Mnet_s = result.iloc[:, range(1, 344, 8)]
Log = result.iloc[:, range(2, 344, 8)]
Log_s = result.iloc[:, range(3, 344, 8)]
Ada = result.iloc[:, range(4, 344, 8)]
Ada_s = result.iloc[:, range(5, 344, 8)]
XGB = result.iloc[:, range(6, 344, 8)]
XGB_s = result.iloc[:, range(7, 344, 8)]
fig, ax = plt.subplots()
# 绘制MLPC
plt.plot(range(1, 44), Mnet.iloc[12, :].values, 'r.-', label=r'MLPC AUC mean')
mlpc_upper = np.array(Mnet) + np.array(Mnet_s)
mlpc_lower = np.array(Mnet) - np.array(Mnet_s)
plt.fill_between(range(1, 44), mlpc_lower[12, :], mlpc_upper[12, :], color='grey', alpha=.5,
                 label=r'AUC STD')
# 绘制Log
plt.plot(range(1, 44), Log.iloc[12, :].values, 'b.-', label=r'Log AUC mean')
Log_upper = np.array(Log) + np.array(Log_s)
Log_lower = np.array(Log) - np.array(Log_s)
plt.fill_between(range(1, 44), Log_lower[12, :], Log_upper[12, :], color='grey', alpha=.5, )

# 绘制Ada
plt.plot(range(1, 44), Ada.iloc[12, :].values, 'g.-', label=r'Ada AUC mean')
Ada_upper = np.array(Ada) + np.array(Ada_s)
Ada_lower = np.array(Ada) - np.array(Ada_s)
plt.fill_between(range(1, 44), Ada_lower[12, :], Ada_upper[12, :], color='grey', alpha=.5, )

# 绘制XGB
plt.plot(range(1, 44), XGB.iloc[12, :].values, 'k.-', label=r'XGB AUC mean')
XGB_upper = np.array(XGB) + np.array(XGB_s)
XGB_lower = np.array(XGB) - np.array(XGB_s)
plt.fill_between(range(1, 44), XGB_lower[12, :], XGB_upper[12, :], color='grey', alpha=.5, )
plt.ylabel('AUC(%)')
plt.xlabel('Number of feature')
plt.legend()