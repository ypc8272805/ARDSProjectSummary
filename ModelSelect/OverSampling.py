import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

# 加载数据
Data = pd.read_csv('C:/Users/zg/OneDrive/ARDSProjectSummary/数据预处理/ARDSoutValue/data/preData.csv')
train_data = Data.filter(
    regex='spo2_.*|fio2_.*|hr_.*|temp_.*|nbps_.*|nbpm_.*|nbpd_.*|rr_.*|tv_scaler|tv_kg_.*|pip_.*|plap_.*|mv_.*|map_.*|peep_.*|gcs_.*|first_careunit_.*|dbsource_.*|age_.*|ethnicity_class_.*|admission_type_.*|gender_.*|sf_.*|height_first_.*|weight_first_.*|osi_.*|bmi_.*')
train_label = Data.loc[:, ['pfclass_two_300']]

# 利用SMOTE方法对数据进行过采样
X_resampled, y_resampled = SMOTE().fit_sample(train_data, train_label)
# 进行十折交叉验证
skf = StratifiedKFold(n_splits=10)
for train_index, test_index in skf.split(X_resampled, y_resampled):
    X_train, X_test = X_resampled[train_index], X_resampled[test_index]
    y_train, y_test = y_resampled[train_index], y_resampled[test_index]
    clf = LogisticRegression(penalty='l2', solver='sag', max_iter=5000, n_jobs=8, random_state=20)
    clf.fit(X_train, y_train)
    y_pro = clf.predict_proba(X_test)
    y_pre = clf.predict(X_test)
    print(roc_auc_score(y_test, y_pro[:, 1]))
    print(accuracy_score(y_test, y_pre))
    