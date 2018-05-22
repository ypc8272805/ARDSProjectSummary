# -*- coding: utf-8 -*-
from numpy import loadtxt
import pandas as pd
from xgboost import XGBClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

Data = pd.read_csv('C:/Users/zg/OneDrive/ARDSProjectSummary/数据预处理/ARDSoutValue/data/train_data.csv',nrows=1000)
X = Data.filter(
    regex='spo2_.*|fio2_.*|hr_.*|temp_.*|nbps_.*|nbpm_.*|nbpd_.*|rr_.*|tv_scaler|tv_kg_.*|pip_.*|plap_.*|mv_.*|map_.*|peep_.*|gcs_.*|first_careunit_.*|dbsource_.*|age_.*|ethnicity_class_.*|admission_type_.*|gender_.*|sf_.*|height_first_.*|weight_first_.*|osi_.*|bmi_.*')
y = Data.loc[:, ['pfclass_two_300']]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=7)

model=XGBClassifier(n_estimators=200)
eval_set=[(X_test,y_test)]
model.fit(X_train,y_train,early_stopping_rounds=20,eval_metric="logloss",eval_set=eval_set,verbose=True)
#print(model.best_iteration)
limit=model.best_iteration

y_pred=model.predict(X_test,ntree_limit=limit)
predictions=[round(value) for value in y_pred]
acc=accuracy_score(y_test,predictions)
print("ACC=%.2f%%"%(acc*100))