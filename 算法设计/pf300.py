import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
import algo

# 导入数据
Data = pd.read_csv('C:/Users/zg/OneDrive/ARDSProjectSummary/数据预处理/ARDSoutValue/data/train_data.csv',nrows=1000)
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
Result = pd.DataFrame()  # 用于存储结果
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=20)
i = 1
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # 算法设计
    print('***交叉验证第'+ str(i) +'折开始***')
    print('XGBoost开始')
    XGB_result, XGB_fpr, XGB_tpr = algo.XGBoost(X_train, y_train, X_test, y_test)
    temp = pd.Series(XGB_result, name='XGBoost')
    Result = pd.concat([Result, temp], axis=1)  # 将结果存储起来
    print('XGBoost结束')
    print('')
    i += 1
