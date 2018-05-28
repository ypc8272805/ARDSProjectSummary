# -*- coding: utf-8 -*-

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.model_selection import StratifiedKFold
from xgboost.sklearn import XGBClassifier


Data = pd.read_csv('C:/Users/zg/OneDrive/ARDSProjectSummary/数据预处理/ARDSoutValue/data/train_data.csv')
train_data = Data.filter(
    regex='spo2_.*|fio2_.*|hr_.*|temp_.*|nbps_.*|nbpm_.*|nbpd_.*|rr_.*|tv_scaler|tv_kg_.*|pip_.*|plap_.*|mv_.*|map_.*|peep_.*|gcs_.*|first_careunit_.*|dbsource_.*|age_.*|ethnicity_.*|admission_type_.*|gender_.*|sf_.*|height_first_.*|weight_first_.*|osi_.*|bmi_.*')
train_label = Data.loc[:, ['pfclass_two_300']]

X_resampled, y_resampled = SMOTE().fit_sample(train_data, train_label)
skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=20)
for train_index, test_index in skf.split(X_resampled, y_resampled):
    X_train, X_test = X_resampled[train_index], X_resampled[test_index]
    y_train, y_test = y_resampled[train_index], y_resampled[test_index]
#    clf = XGBClassifier(max_depth=100, learning_rate=0.0001, n_estimators=100, silent=False, objective='binary:logistic',n_jobs=8)
#    clf.fit(X_train, y_train)
#    train_predict = clf.predict(X_train)
#    test_predict = clf.predict(X_test)
#    test_pre_num = clf.predict_proba(X_test)
#    print(roc_auc_score(y_test, test_pre_num[:,1]))
#    fpr, tpr, thresholds = roc_curve(y_test, test_pre_num[:, 1])  # 用于绘制ROC曲线
#    results=result(y_train, y_test, train_predict, test_predict, test_pre_num)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    param = {'','max_depth': 100, 'eta': 0.0001, 'silent': 1, 'objective': 'binary:logistic','tree_method':'exact','lambda':0.5 }
    watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    num_round = 2
    bst = xgb.train(param, dtrain, num_round, watchlist)
    preds = bst.predict(dtest)
    preds_label=[]
    for i in range(len(preds)):
        preds_label.append(round(preds[i]))
    labels = dtest.get_label()
    print(roc_auc_score(labels, preds))
    print('error=%f' % (sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) / float(len(preds))))
