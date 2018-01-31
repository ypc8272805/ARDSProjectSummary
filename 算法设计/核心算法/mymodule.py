# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 10:46:12 2017
这里存储了我平时要用的一些函数
@author: ypc
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix,accuracy_score
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
#特征提取计算函数
def rank_to_dict(ranks, names, order=1):  
    minmax = MinMaxScaler()  
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]  
    ranks = map(lambda x: round(x, 4), ranks)  
    return dict(zip(names, ranks )) 

def select_feature(X,Y,names):
    i=0
    ranks = {} 
    skf = StratifiedKFold(n_splits=10)
    for train_index, test_index in skf.split(X,Y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        ros=RandomOverSampler(random_state=50) 
        x_res,y_res=ros.fit_sample(x_train,y_train)
        rf = RandomForestClassifier(n_estimators=20, max_depth=4)  
        rf.fit(x_res, y_res)
        i+=1
        ranks[i]=rank_to_dict(np.abs(rf.feature_importances_),names)
    result=pd.DataFrame(ranks)
    result['mean']=result.mean(axis=1)
    temp=result.sort_values(by = 'mean',axis = 0,ascending = False)  
    return temp[temp['mean']>0.01] 

def select_feature_R(X,Y,names):
    #这个返回的是全部排序结果
    i=0
    ranks = {} 
    skf = StratifiedKFold(n_splits=10)
    for train_index, test_index in skf.split(X,Y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        ros=RandomOverSampler(random_state=50) 
        x_res,y_res=ros.fit_sample(x_train,y_train)
        rf = RandomForestClassifier(n_estimators=20, max_depth=4)  
        rf.fit(x_res, y_res)
        i+=1
        ranks[i]=rank_to_dict(np.abs(rf.feature_importances_),names)
    result=pd.DataFrame(ranks)
    result['mean']=result.mean(axis=1)
    temp=result.sort_values(by = 'mean',axis = 0,ascending = False)  
    
    return temp['mean']

def select_feature_P(X,Y,names):
    ranks = {} 
    size=np.shape(X)
    result=[]
    for i in range(0,size[1]):
        temp=pearsonr(X[:,i],Y)
        result.append(temp[0])
    ranks["pearson"]=rank_to_dict(np.abs(result),names)
    result=pd.DataFrame(ranks)
    return result
#计算某个机器学习算法的结果，包括训练集、验证集结果
def result(y_train,y_test,train_predict,test_predict,fpr,tpr):
    result=[]
    #confusion_m_train=confusion_matrix(y_train,train_predict)
    tn, fp, fn, tp = confusion_matrix(y_train,train_predict).ravel()
    result.append(tp/(tp+fn))#灵敏度
    result.append(tn/(tn+fp))#特异性
    result.append(tp/(tp+fp))#PPV精确性和准确率是不一样的 
    result.append(tn/(tn+fn))#NPV
    result.append(accuracy_score(y_train,train_predict))#准确率
    #result.append(roc_auc_score(y_train, train_predict))#AUC
    #测试集混淆矩阵，并计算敏感性、特异性、准确率
    test_tn, test_fp, test_fn, test_tp = confusion_matrix(y_test,test_predict).ravel()
    #confusion_m_test=confusion_matrix(y_test,test_predict)
    result.append(test_tp/(test_tp+test_fn))
    result.append(test_tn/(test_tn+test_fp))
    result.append(test_tp/(test_tp+test_fp))
    result.append(test_tn/(test_tn+test_fn))
    result.append(accuracy_score(y_test,test_predict))
    result.append(auc(fpr,tpr))
    return result
'''
机器学习算法模型，每一个模型都打包，返回结果
'''
def MLPC(x_train,y_train,x_test,y_test):
    '''
    函数名称：MLPC
    函数入参：训练集和测试集样本，x与y分开传入
    返回参数：返回的是模型训练结果，和绘制ROC曲线需要的参数
    描述：这个函数主要是封装了神经网络的相关模块
    '''
    clf=MLPClassifier(hidden_layer_sizes=(20,),activation='tanh',solver='adam',
                      batch_size=50,max_iter=2000,shuffle=True)
    clf.fit(x_train,y_train)
    train_pre=clf.predict(x_train)
    test_pre=clf.predict(x_test)
    #计算测试集的AUC值
    test_pre_num=clf.predict_proba(x_test)
    fpr, tpr, thresholds= roc_curve(y_test, test_pre_num[:, 1])
    #计算结果 训练集和测试集
    results=result(y_train,y_test,train_pre,test_pre,fpr,tpr)
    return results,fpr,tpr

def Bagging(x_train,y_train,x_test,y_test):
    '''
    函数名称：MLPC
    函数入参：训练集和测试集样本，x与y分开传入
    返回参数：返回的是模型训练结果，和绘制ROC曲线需要的参数
    描述：Bagging是一种集成学习算法，之前我也用过，随机抽样，后使用投票法
    '''
    bagging = BaggingClassifier(KNeighborsClassifier())
    bagging.fit(x_train,y_train)
    train_pre=bagging.predict(x_train)
    test_pre=bagging.predict(x_test)
    test_pre_num=bagging.predict_proba(x_test)
    fpr, tpr, thresholds= roc_curve(y_test, test_pre_num[:, 1])
    results=result(y_train,y_test,train_pre,test_pre,fpr,tpr)
    return results,fpr,tpr

def RandomFor(x_train,y_train,x_test,y_test):
    '''
    函数名称：MLPC
    函数入参：训练集和测试集样本，x与y分开传入
    返回参数：返回的是模型训练结果，和绘制ROC曲线需要的参数
    描述：随机森林分类模型，基本没有调优，感觉这个算法对于我这个数据不是很合适，存在
    很多问题，由于时间问题，没有时间去处理，训练集和测试集之间的差异非常大，训练集上
    结果很好，但是验证集结果很差，导致结果无法使用，所以在后面，并没有使用这个模型
    '''
    clf = RandomForestClassifier(n_estimators=10, max_depth=None,
                                 min_samples_split=2, random_state=0)
    clf.fit(x_train,y_train)
    train_pre=clf.predict(x_train)
    test_pre=clf.predict(x_test)
    #计算测试集的AUC值
    test_pre_num=clf.predict_proba(x_test)
    fpr, tpr, thresholds= roc_curve(y_test, test_pre_num[:, 1])
    #计算结果 训练集和测试集
    results=result(y_train,y_test,train_pre,test_pre,fpr,tpr)
    return results,fpr,tpr

def AdaBoost(x_train,y_train,x_test,y_test):
    '''
    函数名称：MLPC
    函数入参：训练集和测试集样本，x与y分开传入
    返回参数：返回的是模型训练结果，和绘制ROC曲线需要的参数
    描述：AdaBoost是一种集成学习算法，应该是一种串行的算法，之前小组内对这种算法有过
    深入的讨论，这个算法表比较稳定，结果也很不错
    '''
    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(x_train,y_train)
    train_pre=clf.predict(x_train)
    test_pre=clf.predict(x_test)
    #计算测试集的AUC值
    test_pre_num=clf.predict_proba(x_test)
    fpr, tpr, thresholds= roc_curve(y_test, test_pre_num[:, 1])
    #计算结果 训练集和测试集
    results=result(y_train,y_test,train_pre,test_pre,fpr,tpr)
    return results,fpr,tpr

def DecTreeC(x_train,y_train,x_test,y_test):
    '''
    函数名称：MLPC
    函数入参：训练集和测试集样本，x与y分开传入
    返回参数：返回的是模型训练结果，和绘制ROC曲线需要的参数
    描述：决策树我了解的比较少，对其原理、和如何对参数调优没有过多涉及，所以在后续试
    验中，我也没有使用这个模型
    '''
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(x_train,y_train)
    train_pre=clf.predict(x_train)
    test_pre=clf.predict(x_test)
    #计算测试集的AUC值
    test_pre_num=clf.predict_proba(x_test)
    fpr, tpr, thresholds= roc_curve(y_test, test_pre_num[:, 1])
    #计算结果 训练集和测试集
    results=result(y_train,y_test,train_pre,test_pre,fpr,tpr)
    return results,fpr,tpr

def LogReg(x_train,y_train,x_test,y_test):
    '''
    函数名称：MLPC
    函数入参：训练集和测试集样本，x与y分开传入
    返回参数：返回的是模型训练结果，和绘制ROC曲线需要的参数
    描述：逻辑回归算法
    '''
    clf = LogisticRegression()
    clf.fit(x_train,y_train)
    train_pre=clf.predict(x_train)
    test_pre=clf.predict(x_test)
    #计算测试集的AUC值
    test_pre_num=clf.predict_proba(x_test)
    fpr, tpr, thresholds= roc_curve(y_test, test_pre_num[:, 1])
    #计算结果 训练集和测试集
    results=result(y_train,y_test,train_pre,test_pre,fpr,tpr)
    return results,fpr,tpr

def KN(x_train,y_train,x_test,y_test):
    '''
    函数名称：MLPC
    函数入参：训练集和测试集样本，x与y分开传入
    返回参数：返回的是模型训练结果，和绘制ROC曲线需要的参数
    描述：
    '''
    clf = KNeighborsClassifier()
    clf.fit(x_train,y_train)
    train_pre=clf.predict(x_train)
    test_pre=clf.predict(x_test)
    #计算测试集的AUC值
    test_pre_num=clf.predict_proba(x_test)
    fpr, tpr, thresholds= roc_curve(y_test, test_pre_num[:, 1])
    #计算结果 训练集和测试集
    results=result(y_train,y_test,train_pre,test_pre,fpr,tpr)
    return results,fpr,tpr

def SVM(x_train,y_train,x_test,y_test):
    '''
    函数名称：MLPC
    函数入参：训练集和测试集样本，x与y分开传入
    返回参数：返回的是模型训练结果，和绘制ROC曲线需要的参数
    描述：SVM算法，这个算法想要用好，感觉必须要对参数进行调调节，寻找合适的参数，但是
    这个过程非常耗时，切SVM在进行训练的过程中也很耗时，同时其结果也不一定优于其他模型
    '''
    clf = SVC(kernel='linear', probability=True,gamma=1,C=1)
    clf.fit(x_train,y_train)
    train_pre=clf.predict(x_train)
    test_pre=clf.predict(x_test)
    #计算测试集的AUC值
    test_pre_num=clf.predict_proba(x_test)
    fpr, tpr, thresholds= roc_curve(y_test, test_pre_num[:, 1])
    #计算结果 训练集和测试集
    results=result(y_train,y_test,train_pre,test_pre,fpr,tpr)
    return results,fpr,tpr

def plot_roc(tprs,mean_fpr,aucs,names,colors):
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)#这里的平均AUC和 10折直接计算平均值是否有不同
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color=colors,
             label=names+'(AUC=%0.4f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=1.5, alpha=0.8)
def roc_need(tprs,aucs,mean_fpr,fpr,tpr):
    tprs.append(interp(mean_fpr,fpr,tpr))
    tprs[-1][0]=0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc) 