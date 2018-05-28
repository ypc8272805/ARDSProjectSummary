'''
使用filter和wrapper两种方式进行特征值选择
比较两种方法的性能
wrapper：使用 Boruta all-relevant feature selection method
filter：选择2-3种评分标准，将几种评分结果进行融合，对特征值进行排序，从而选择合适的特征值
'''
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.model_selection import StratifiedKFold
from skfeature.function.similarity_based import reliefF
from skfeature.function.statistical_based import chi_square
from skfeature.function.information_theoretical_based import MIFS, MRMR
from skfeature.function.wrapper import decision_tree_backward, decision_tree_forward
from sklearn.utils import shuffle
import sklearn.preprocessing as preprocessing
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

Data = pd.read_csv('C:/Users/zg/OneDrive/ARDSProjectSummary/数据预处理/ARDSoutValue/data/train_data.csv')
train_data = Data.filter(
    regex='spo2_.*|fio2_.*|hr_.*|temp_.*|nbps_.*|nbpm_.*|nbpd_.*|rr_.*|tv_scaler|tv_kg_.*|pip_.*|plap_.*|mv_.*|map_.*|peep_.*|gcs_.*|first_careunit_.*|dbsource_.*|age_.*|ethnicity_class_.*|admission_type_.*|gender_.*|sf_.*|height_first_.*|weight_first_.*|osi_.*|bmi_.*')
train_label = Data.loc[:, ['pfclass_two_300']]
n_samples, n_features = train_data.shape
# 随机采样1000个样本用于计算

X = np.array(train_data)
y = np.array(train_label)

X_relief, y_relief = shuffle(X, y, n_samples=10000, random_state=0)
'''
Filter
方法：
Distance:RelieF
Dependence:Chi-squared
Information:MIFS (Mutual Information Feature 
'''
# Relief 和 Chi 都是给出每个特征值的一个score,MIFS稍有不同，电脑是第二行也可以当作一个分数，将这三种分数都归一化为0-1之间的数值，求平均
RelieF_score = reliefF.reliefF(X_relief, y_relief[:, 0], k=n_features)  # RelieF
Chi = chi_square.chi_square(X, y[:, 0])
# 返回值，第一行为特征值排序后的结果，第二行为目标函数，第三行是自变量与相应变量之间的互信息
Mifs = MIFS.mifs(X_relief, y_relief[:, 0], n_selected_features=n_features)

'''
使用mean method 进行选择融合
'''
scores = pd.DataFrame({'Feature': list(Mifs[0]), 'MIFS': list(Mifs[1])})
scores = scores.sort_values(by=['Feature'])
scores['Relief'] = RelieF_score
scores['Chi'] = Chi
# 归一化
min_max_scaler = preprocessing.MinMaxScaler()
scores['MIFS_scaler'] = min_max_scaler.fit_transform(scores.loc[:, ['MIFS']])
scores['Relief_scaler'] = min_max_scaler.fit_transform(scores.loc[:, ['Relief']])
scores['Chi_scaler'] = min_max_scaler.fit_transform(scores.loc[:, ['Chi']])
scores['mean'] = (scores['MIFS_scaler'] + scores['Relief_scaler'] + scores['Chi_scaler']) / 3
scores['feature_name'] = train_data.columns
scores = scores.sort_values(by='mean', ascending=False)

'''
使用文献中的排序方法来进行选择融合
'''
rank_Relief = pd.DataFrame(
    {'Relief_index': range(43), 'Relief_scores': RelieF_score, 'feature_name': train_data.columns})
rank_Chi = pd.DataFrame({'Chi_index': range(43), 'Chi_scores': Chi, 'feature_name': train_data.columns})
rank_MIFS = pd.DataFrame(
    {'MIFS_index': list(Mifs[0]), 'MIFS_scores': list(Mifs[1]), 'feature_name': train_data.columns})

rank_Relief = rank_Relief.sort_values(by='Relief_scores', ascending=False)
rank_Chi = rank_Chi.sort_values(by='Chi_scores', ascending=False)
rank_MIFS = rank_MIFS.sort_values(by='MIFS_scores', ascending=False)

rank = pd.DataFrame({'Relief': list(rank_Relief['Relief_index']), 'Chi': list(rank_Chi['Chi_index']),
                     'MIFS': list(rank_MIFS['MIFS_index'])})
# 感觉这个方法有问题，先暂时不适用了
'''
forward
'''
# idx = decision_tree_forward.decision_tree_forward(X_relief, y_relief[:,0],n_features)
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)
feat_selector.fit(X, y)
feat_selector.ranking_
