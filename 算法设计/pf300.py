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
# 迭代特征值
sizes = np.shape(X)
s_result = pd.DataFrame()
for i in range(1, (sizes[1] + 1)):
    data_pre = X[:, 0:i]
    Result = pd.DataFrame()  # 用于存储结果
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=20)
    cross_index = 1
    for train_index, test_index in skf.split(data_pre, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # 算法设计
        print('***交叉验证第' + str(cross_index) + '折开始***')
        # XGBoost
        print('XGBoost开始' + time.asctime(time.localtime(time.time())))
        XGB_result, XGB_fpr, XGB_tpr = algo.XGBoost(X_train, y_train, X_test, y_test)
        temp = pd.Series(XGB_result, name='XGBoost')
        Result = pd.concat([Result, temp], axis=1)  # 将结果存储起来
        print('XGBoost结束' + time.asctime(time.localtime(time.time())))
        print('--------------')
        # AdaBoost
        print('AdaBoost开始' + time.asctime(time.localtime(time.time())))
        Ada_result, Ada_fpr, Ada_tpr = algo.AdaBoost(X_train, y_train, X_test, y_test)
        temp = pd.Series(Ada_result, name='AdaBoost')
        Result = pd.concat([Result, temp], axis=1)
        print('AdaBoost结束' + time.asctime(time.localtime(time.time())))
        print('--------------')
        # LogReg
        print('LogReg开始' + time.asctime(time.localtime(time.time())))
        Log_result, Log_fpr, Log_tpr = algo.LogReg(X_train, y_train, X_test, y_test)
        temp = pd.Series(Log_result, name='LogReg')
        Result = pd.concat([Result, temp], axis=1)
        print('LogReg结束' + time.asctime(time.localtime(time.time())))
        print('--------------')
        # MLPC
        print('MLPC开始' + time.asctime(time.localtime(time.time())))
        MLPC_result, MLPC_fpr, MLPC_tpr = algo.MLPC(X_train, y_train, X_test, y_test)
        temp = pd.Series(MLPC_result, name='MLPC')
        Result = pd.concat([Result, temp], axis=1)
        print('MLPC结束' + time.asctime(time.localtime(time.time())))
        cross_index += 1

    MLPC = 'MLPC' + str(i)
    MLPC_s = 'MLPC_s' + str(i)
    Log = 'Log' + str(i)
    Log_s = 'Log_s' + str(i)
    Ada = 'Ada' + str(i)
    Ada_s = 'Ada_s' + str(i)
    XGB = 'XGB' + str(i)
    XGB_s = 'XGB_s' + str(i)
    s_result[MLPC] = Result['MLPC'].mean(axis=1)
    s_result[MLPC_s] = Result['MLPC'].std(axis=1)
    s_result[Log] = Result['LogReg'].mean(axis=1)
    s_result[Log_s] = Result['LogReg'].std(axis=1)
    s_result[Ada] = Result['AdaBoost'].mean(axis=1)
    s_result[Ada_s] = Result['AdaBoost'].std(axis=1)
    s_result[XGB] = Result['XGBoost'].mean(axis=1)
    s_result[XGB_s] = Result['XGBoost'].std(axis=1)
    print('第' + str(i) + '个特征值组合计算结束' + time.asctime(time.localtime(time.time())))
s_result.to_csv('result.csv')