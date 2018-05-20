import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from scipy.stats import pearsonr


# 计算某个机器学习算法的结果，包括训练集、验证集结果
def result(y_train, y_test, train_predict, test_predict, fpr, tpr):
    result = []
    # confusion_m_train=confusion_matrix(y_train,train_predict)
    tn, fp, fn, tp = confusion_matrix(y_train, train_predict).ravel()
    result.append(tp / (tp + fn))  # 灵敏度
    result.append(tn / (tn + fp))  # 特异性
    result.append(tp / (tp + fp))  # PPV精确性和准确率是不一样的
    result.append(tn / (tn + fn))  # NPV
    result.append(accuracy_score(y_train, train_predict))  # 准确率
    # result.append(roc_auc_score(y_train, train_predict))#AUC
    # 测试集混淆矩阵，并计算敏感性、特异性、准确率
    test_tn, test_fp, test_fn, test_tp = confusion_matrix(y_test, test_predict).ravel()
    # confusion_m_test=confusion_matrix(y_test,test_predict)
    result.append(test_tp / (test_tp + test_fn))
    result.append(test_tn / (test_tn + test_fp))
    result.append(test_tp / (test_tp + test_fp))
    result.append(test_tn / (test_tn + test_fn))
    result.append(accuracy_score(y_test, test_predict))
    result.append(auc(fpr, tpr))
    return result


def AdaBoost(x_train, y_train, x_test, y_test):
    '''
    函数名称：MLPC
    函数入参：训练集和测试集样本，x与y分开传入
    返回参数：返回的是模型训练结果，和绘制ROC曲线需要的参数
    描述：AdaBoost是一种集成学习算法，应该是一种串行的算法，之前小组内对这种算法有过
    深入的讨论，这个算法表比较稳定，结果也很不错
    '''
    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(x_train, y_train)
    train_pre = clf.predict(x_train)
    test_pre = clf.predict(x_test)
    # 计算测试集的AUC值
    test_pre_num = clf.predict_proba(x_test)
    fpr, tpr, thresholds = roc_curve(y_test, test_pre_num[:, 1])
    # 计算结果 训练集和测试集
    results = result(y_train, y_test, train_pre, test_pre, fpr, tpr)
    return results, fpr, tpr
