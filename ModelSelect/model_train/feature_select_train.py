'''
特征值排序 及其 对应结果已经完成，现在三个节点需要验证：
1）全部特征值
2）结果最优特征值
3）最少特征值
神经网络：结果最优 43 最少特征值 40
Log:结果最优 42 最少特征值 37
AdaBoost：结果最优 39 最少特征值 37
XGBoost：结果最优 19 最少特征值 18
'''
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
import algo
import time

# 导入数据
Data = pd.read_csv('C:/Users/zg/OneDrive/ARDSProjectSummary/数据预处理/ARDSoutValue/data/train_data.csv')
train_data = Data.filter(
    regex='spo2_.*|fio2_.*|hr_.*|temp_.*|nbps_.*|nbpm_.*|nbpd_.*|rr_.*|tv_scaler|tv_kg_.*|pip_.*|plap_.*|mv_.*|map_.*|peep_.*|gcs_.*|first_careunit_.*|dbsource_.*|age_.*|ethnicity_class_.*|admission_type_.*|gender_.*|sf_.*|height_first_.*|weight_first_.*|osi_.*|bmi_.*')
train_label = Data.loc[:, ['pfclass_two_300']]

# 导入特征值顺序
scores = pd.read_csv('C:/Users/zg/OneDrive/ARDSProjectSummary/算法设计/scores.csv')
scores.Feature.values
train_data = train_data.loc[:, scores['feature_name'].values]
# 过采样
X_resampled, y_resampled = SMOTE().fit_sample(train_data, train_label.values.ravel())
X = np.array(X_resampled)
y = np.array(y_resampled).ravel()
# 按照这个特征值子集进行训练，每一个特征子集进行十折交叉验证，后面还要绘制box图，所以标准差
featureSelect = [[40, 43, 43], [37, 42, 43], [37, 39, 43], [18, 19, 43]]
Result = pd.DataFrame()  # 用于存储结果
for methodNum in range(4):
    for featureSet in range(3):
        # 选择合适的特征子集
        featureNum = featureSelect[methodNum][featureSet]
        data_pre = X[:, 0:featureNum]
        # 进行交叉验证
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=20)
        if methodNum == 0:
            # 开始MLPC的训练过程
            print('MLPC开始' + time.asctime(time.localtime(time.time())))
            for train_index, test_index in skf.split(data_pre, y):
                X_train, X_test = data_pre[train_index], data_pre[test_index]
                y_train, y_test = y[train_index], y[test_index]

                MLPC_result, MLPC_fpr, MLPC_tpr = algo.MLPC(X_train, y_train, X_test, y_test)
                strName = 'MLPC_' + str(featureNum)
                temp = pd.Series(MLPC_result, name=strName)
                Result = pd.concat([Result, temp], axis=1)
            print('MLPC结束' + time.asctime(time.localtime(time.time())))
        if methodNum == 1:
            # 开始Log的训练过程
            print('LogReg开始' + time.asctime(time.localtime(time.time())))
            for train_index, test_index in skf.split(data_pre, y):
                X_train, X_test = data_pre[train_index], data_pre[test_index]
                y_train, y_test = y[train_index], y[test_index]

                Log_result, Log_fpr, Log_tpr = algo.LogReg(X_train, y_train, X_test, y_test)
                strName = 'Log_' + str(featureNum)
                temp = pd.Series(Log_result, name=strName)
                Result = pd.concat([Result, temp], axis=1)
            print('LogReg结束' + time.asctime(time.localtime(time.time())))
        if methodNum == 2:
            print('AdaBoost开始' + time.asctime(time.localtime(time.time())))
            for train_index, test_index in skf.split(data_pre, y):
                X_train, X_test = data_pre[train_index], data_pre[test_index]
                y_train, y_test = y[train_index], y[test_index]

                Ada_result, Ada_fpr, Ada_tpr = algo.AdaBoost(X_train, y_train, X_test, y_test)
                strName = 'Ada_' + str(featureNum)
                temp = pd.Series(Ada_result, name=strName)
                Result = pd.concat([Result, temp], axis=1)
            print('AdaBoost结束' + time.asctime(time.localtime(time.time())))
        if methodNum == 3:
            print('XGBoost开始' + time.asctime(time.localtime(time.time())))
            for train_index, test_index in skf.split(data_pre, y):
                X_train, X_test = data_pre[train_index], data_pre[test_index]
                y_train, y_test = y[train_index], y[test_index]

                XGB_result, XGB_fpr, XGB_tpr = algo.XGBoost(X_train, y_train, X_test, y_test)
                strName = 'XGBoost_' + str(featureNum)
                temp = pd.Series(XGB_result, name=strName)
                Result = pd.concat([Result, temp], axis=1)  # 将结果存储起来
            print('XGBoost结束' + time.asctime(time.localtime(time.time())))
Result.to_csv('Result.csv')

# 绘制Box图
# 最少特征值
import matplotlib.pyplot as plt

MinFeatureBER = pd.DataFrame()
MinFeatureBER['MLPC'] = list(Result.loc[11, 'MLPC_40'].values)
MinFeatureBER['Log'] = list(Result.loc[11, 'Log_37'].values)
MinFeatureBER['Ada'] = list(Result.loc[11, 'Ada_37'].values)
MinFeatureBER['XGB'] = list(Result.loc[11, 'XGBoost_18'].values)
MinFeatureBER.plot(kind='box')
# 计算所有参数的平均值和标准差
MinReault = pd.DataFrame()
MinReault['MLPC_mean'] = Result['MLPC_40'].mean(axis=1)
MinReault['MLPC_std'] = Result['MLPC_40'].std(axis=1)
MinReault['Log_mean'] = Result['Log_37'].mean(axis=1)
MinReault['Log_std'] = Result['Log_37'].std(axis=1)
MinReault['Ada_mean'] = Result['Ada_37'].mean(axis=1)
MinReault['Ada_std'] = Result['Ada_37'].std(axis=1)
MinReault['XGBoost_mean'] = Result['XGBoost_18'].mean(axis=1)
MinReault['XGBoost_std'] = Result['XGBoost_18'].std(axis=1)

OptFeatureBER = pd.DataFrame()
OptFeatureBER['MLPC'] = list(Result.iloc[11, 10:20].values)
OptFeatureBER['Log'] = list(Result.loc[11, 'Log_42'].values)
OptFeatureBER['Ada'] = list(Result.loc[11, 'Ada_39'].values)
OptFeatureBER['XGB'] = list(Result.loc[11, 'XGBoost_19'].values)
OptFeatureBER.plot.box()
OptReault = pd.DataFrame()
OptReault['MLPC_mean'] = Result.iloc[:, 10:20].mean(axis=1)
OptReault['MLPC_std'] = Result.iloc[:, 10:20].std(axis=1)
OptReault['Log_mean'] = Result['Log_42'].mean(axis=1)
OptReault['Log_std'] = Result['Log_42'].std(axis=1)
OptReault['Ada_mean'] = Result['Ada_39'].mean(axis=1)
OptReault['Ada_std'] = Result['Ada_39'].std(axis=1)
OptReault['XGBoost_mean'] = Result['XGBoost_19'].mean(axis=1)
OptReault['XGBoost_std'] = Result['XGBoost_19'].std(axis=1)

MaxFeatureBER = pd.DataFrame()
MaxFeatureBER['MLPC'] = list(Result.iloc[11, 20:30].values)
MaxFeatureBER['Log'] = list(Result.loc[11, 'Log_43'].values)
MaxFeatureBER['Ada'] = list(Result.loc[11, 'Ada_43'].values)
MaxFeatureBER['XGB'] = list(Result.loc[11, 'XGBoost_43'].values)
MaxFeatureBER.plot.box()
MaxReault = pd.DataFrame()
MaxReault['MLPC_mean'] = Result.iloc[:, 20:30].mean(axis=1)
MaxReault['MLPC_std'] = Result.iloc[:, 20:30].std(axis=1)
MaxReault['Log_mean'] = Result['Log_43'].mean(axis=1)
MaxReault['Log_std'] = Result['Log_43'].std(axis=1)
MaxReault['Ada_mean'] = Result['Ada_43'].mean(axis=1)
MaxReault['Ada_std'] = Result['Ada_43'].std(axis=1)
MaxReault['XGBoost_mean'] = Result['XGBoost_43'].mean(axis=1)
MaxReault['XGBoost_std'] = Result['XGBoost_43'].std(axis=1)

BoxFeatureBER = pd.DataFrame()
BoxFeatureBER['MLPC_min'] = list(Result.loc[11, 'MLPC_40'].values)
BoxFeatureBER['MLPC_opt'] = list(Result.iloc[11, 10:20].values)
BoxFeatureBER['MLPC_max'] = list(Result.iloc[11, 20:30].values)

BoxFeatureBER['Log_min'] = list(Result.loc[11, 'Log_37'].values)
BoxFeatureBER['Log_opt'] = list(Result.loc[11, 'Log_42'].values)
BoxFeatureBER['Log_max'] = list(Result.loc[11, 'Log_43'].values)

BoxFeatureBER['Ada_min'] = list(Result.loc[11, 'Ada_37'].values)
BoxFeatureBER['Ada_opt'] = list(Result.loc[11, 'Ada_39'].values)
BoxFeatureBER['Ada_max'] = list(Result.loc[11, 'Ada_43'].values)

BoxFeatureBER['XGB_min'] = list(Result.loc[11, 'XGBoost_18'].values)
BoxFeatureBER['XGB_opt'] = list(Result.loc[11, 'XGBoost_19'].values)
BoxFeatureBER['XGB_max'] = list(Result.loc[11, 'XGBoost_43'].values)
BoxFeatureBER.plot.box()
