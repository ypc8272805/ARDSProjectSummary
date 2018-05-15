'''
特征工程
对特征值进行处理
2018.05.14
ypc
'''
import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score

# 读取数据文件
data = pd.read_csv("C:/Users/zg/OneDrive/ARDSProjectSummary/算法设计/数据预处理/newproject/ARDSoutValue/data/imputeData.csv")
# 对哑变量进行处理
dummies_gender = pd.get_dummies(data['gender'], prefix='gender')
dummies_careunit = pd.get_dummies(data['first_careunit'], prefix='first_careunit')
dummies_dbsource = pd.get_dummies(data['dbsource'], prefix='dbsource')
dummies_admission = pd.get_dummies(data['admission_type'], prefix='admission_type')
# 对种族进行处理
data.loc[data['ethnicity'].str.contains('WHITE'), 'ethnicity_class'] = 1
data.loc[data['ethnicity'].str.contains('ASIAN'), 'ethnicity_class'] = 2
data.loc[data['ethnicity'].str.contains('BLACK'), 'ethnicity_class'] = 3
data.loc[data['ethnicity'].str.contains('HISPANIC'), 'ethnicity_class'] = 4
data.loc[data['ethnicity_class'].isnull(), 'ethnicity_class'] = 5
dummies_ethnicity = pd.get_dummies(data['ethnicity_class'], prefix='ethnicity_class')
data = pd.concat([data, dummies_admission, dummies_careunit, dummies_dbsource, dummies_ethnicity, dummies_gender],
                 axis=1)
# 特征组合和计算
data['gcs'] = data['gcsmotor'] + data['gcseyes'] + data['gcsverbal']
data['sf'] = data['spo2'] / (data['fio2'] / 100)
data['osi'] = data['map'] * data['fio2'] / data['spo2']
data['bmi'] = data['weight_first'] / ((data['height_first'] / 100) ** 2)
data['tv_kg'] = data['tv'] / data['weight_first']
data['pf'] = data['pao2'] / (data['fio2'] / 100)

# 归一化处理,将数值变量缩放到0-1范围


min_max_scaler = preprocessing.MinMaxScaler()
data['spo2_scaler'] = min_max_scaler.fit_transform(data.loc[:, ['spo2']])
data['fio2_scaler'] = min_max_scaler.fit_transform(data.loc[:, ['fio2']])
data['hr_scaler'] = min_max_scaler.fit_transform(data.loc[:, ['hr']])
data['temp_scaler'] = min_max_scaler.fit_transform(data.loc[:, ['temp']])
data['nbps_scaler'] = min_max_scaler.fit_transform(data.loc[:, ['nbps']])
data['nbpm_scaler'] = min_max_scaler.fit_transform(data.loc[:, ['nbpm']])
data['nbpd_scaler'] = min_max_scaler.fit_transform(data.loc[:, ['nbpd']])
data['rr_scaler'] = min_max_scaler.fit_transform(data.loc[:, ['rr']])
data['tv_scaler'] = min_max_scaler.fit_transform(data.loc[:, ['tv']])
data['mv_scaler'] = min_max_scaler.fit_transform(data.loc[:, ['mv']])
data['pip_scaler'] = min_max_scaler.fit_transform(data.loc[:, ['pip']])
data['plap_scaler'] = min_max_scaler.fit_transform(data.loc[:, ['plap']])
data['map_scaler'] = min_max_scaler.fit_transform(data.loc[:, ['map']])
data['peep_scaler'] = min_max_scaler.fit_transform(data.loc[:, ['peep']])
data['gcs_scaler'] = min_max_scaler.fit_transform(data.loc[:, ['gcs']])
data['gcs_m_scaler'] = min_max_scaler.fit_transform(data.loc[:, ['gcsmotor']])
data['gcs_v_scaler'] = min_max_scaler.fit_transform(data.loc[:, ['gcsverbal']])
data['gcs_e_scaler'] = min_max_scaler.fit_transform(data.loc[:, ['gcseyes']])
data['age_scaler'] = min_max_scaler.fit_transform(data.loc[:, ['age']])
data['height_first_scaler'] = min_max_scaler.fit_transform(data.loc[:, ['height_first']])
data['weight_first_scaler'] = min_max_scaler.fit_transform(data.loc[:, ['weight_first']])
data['sf_scaler'] = min_max_scaler.fit_transform(data.loc[:, ['sf']])
data['osi_scaler'] = min_max_scaler.fit_transform(data.loc[:, ['osi']])
data['bmi_scaler'] = min_max_scaler.fit_transform(data.loc[:, ['bmi']])
data['tv_kg_scaler'] = min_max_scaler.fit_transform(data.loc[:, ['tv_kg']])

# 处理结局变量PF值
data.loc[data['pf'] > 300, 'pfclass_two_300'] = 1
data.loc[data['pf'] <= 300, 'pfclass_two_300'] = 0

data.loc[data['pf'] > 300, 'pfclass_two_200'] = 1
data.loc[data['pf'] <= 200, 'pfclass_two_200'] = 0

data.loc[data['pf'] > 100, 'pfclass_two_100'] = 1
data.loc[data['pf'] <= 100, 'pfclass_two_100'] = 0

data.loc[data['pf'] > 300, 'pfclass_four'] = 1
data.loc[data['pf'] <= 300, 'pfclass_four'] = 2
data.loc[data['pf'] <= 200, 'pfclass_four'] = 3
data.loc[data['pf'] <= 100, 'pfclass_four'] = 4

train_data = data.filter(
    regex='spo2_.*|fio2_.*|hr_.*|temp_.*|nbps_.*|nbpm_.*|nbpd_.*|rr_.*|tv_scaler|tv_kg_.*|pip_.*|plap_.*|mv_.*|map_.*|peep_.*|gcs_.*|first_careunit_.*|dbsource_.*|age_.*|ethnicity_.*|admission_type_.*|gender_.*|sf_.*|height_first_.*|weight_first_.*|osi_.*|bmi_.*')
train_label = data.loc[:, ['pfclass_two_300', 'pfclass_two_200', 'pfclass_two_100', 'pfclass_four']]

# 先用逻辑回归看看情况
X_train, X_test, y_train, y_test = train_test_split(
    train_data, train_label, test_size=0.3, random_state=42)
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X_train, y_train)
y_pre = clf.predict(X_test)

accuracy_score(y_test, y_pre)
coef = pd.DataFrame({"columns": list(X_train.columns)[:], "coef": list(clf.coef_.T)})
coef['coef'] = coef['coef'].abs()
coef = coef.sort_values(by=['coef'], ascending=False)

data.to_csv('C:/Users/zg/OneDrive/ARDSProjectSummary/算法设计/数据预处理/newproject/ARDSoutValue/data/preData.csv')
train_data.to_csv('C:/Users/zg/OneDrive/ARDSProjectSummary/算法设计/数据预处理/newproject/ARDSoutValue/data/trainData.csv')

import 