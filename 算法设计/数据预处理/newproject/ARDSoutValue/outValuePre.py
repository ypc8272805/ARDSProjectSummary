# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

# 直接从数据库中提取相关数据
engine = create_engine('postgresql://postgres@localhost:5432/mimic', echo=True)
sql = 'select * from mimiciii.secondData'
data = pd.read_sql_query(sql, con=engine, index_col=None, coerce_float=True, params=None, parse_dates=None,
                         chunksize=None)
# 这里主要是提出需要的数据进行处理
tempData = data.loc[:, ['subject_id', 'hadm_id', 'icustay_id', 'pao2', 'spo2', 'fio2',
                        'hr', 'temp', 'nbps', 'nbpd', 'nbpm', 'abps', 'abpd', 'abpm', 'rr', 'tv', 'mv', 'pip', 'plap',
                        'map', 'peep', 'gcseyes', 'gcsmotor', 'gcsverbal', 'height_first', 'weight_first', 'age']]
# SpO2异常值处理,默认低于50的SpO2 全部排除
tempData.loc[tempData['spo2'] <= 50, 'spo2'] = np.nan
# FiO2没有异常值，但是有一定比例的缺失值，只需要补全缺失值即可
# hr心率 大于200 小于30 都认为是异常值，
tempData.loc[tempData['hr'] <= 30, 'hr'] = np.nan
tempData.loc[tempData['hr'] >= 200, 'hr'] = np.nan
# temp 体温，小于30度的情况认为是异常值
tempData.loc[tempData['temp'] <= 30, 'temp'] = np.nan
'''
处理血压存在的问题，稍有些麻烦
先处理无创血压：
nbps nbpm nbpd
'''
# 先处理大小逻辑关系
tempData.loc[tempData['nbps'] <= 30, 'nbps'] = np.nan
tempData.loc[tempData['nbpm'] <= 25, 'nbpm'] = np.nan
tempData.loc[tempData['nbpd'] <= 20, 'nbpd'] = np.nan
tempData.loc[tempData['nbps'] < tempData['nbpm'], ['nbps', 'nbpm', 'nbpm']] = np.nan
tempData.loc[tempData['nbps'] < tempData['nbpd'], ['nbps', 'nbpm', 'nbpm']] = np.nan
tempData.loc[tempData['nbpm'] < tempData['nbpd'], ['nbps', 'nbpm', 'nbpm']] = np.nan

# 异常值暂时不处理，应为不知道异常值的范围，如果血压太小 或者 太大还是需要限制以下

'''
处理血压存在的问题，稍有些麻烦
有创血压：
bps abpm abpd
'''
tempData.loc[tempData['abps'] <= 30, 'abps'] = np.nan
tempData.loc[tempData['abpm'] <= 25, 'abpm'] = np.nan
tempData.loc[tempData['abpd'] <= 20, 'abpd'] = np.nan
tempData.loc[tempData['abps'] < tempData['abpm'], ['abps', 'abpm', 'abpm']] = np.nan
tempData.loc[tempData['abps'] < tempData['abpd'], ['abps', 'abpm', 'abpm']] = np.nan
tempData.loc[tempData['abpm'] < tempData['abpd'], ['abps', 'abpm', 'abpm']] = np.nan

# 异常值暂时不处理，应为不知道异常值的范围，如果血压太小 或者 太大还是需要限制以下

# 处理呼吸频率 RR
# 呼吸频率有很多是0，可以加入一个类别区分一下，而不是直接删除，RR=0 --0 RR 不等于0 --1 ，
# 甚至可以对RR进行阶段性的划分，转化为哑变量
# 没有最大异常值
# 通过观察发现，RR=0 应该是异常值，应为同时记录了患者的TV MV 等多个机械通气参数
tempData.loc[tempData['rr'] == 0, 'rr'] = np.nan

# TV 潮气量处理
tempData.loc[tempData['tv'] >= 1500, 'tv'] = np.nan

# MV 分钟通气量
tempData.loc[tempData['mv'] >= 40, 'mv'] = np.nan

'''
呼吸机压力四种参数的关系
pip>plap>map>peep
'''
tempData.loc[tempData['peep'] > 30, 'peep'] = np.nan
tempData.loc[tempData['map'] > 50, 'map'] = np.nan
tempData.loc[tempData['pip'] < tempData['plap'], 'pip'] = np.nan
tempData.loc[tempData['pip'] < tempData['map'], 'pip'] = np.nan
tempData.loc[tempData['plap'] < tempData['map'], 'plap'] = np.nan
tempData.loc[tempData['plap'] < tempData['peep'], 'plap'] = np.nan
tempData.loc[tempData['map'] < tempData['peep'], 'map'] = np.nan

# 身高、体重、年龄
tempData.loc[tempData['height_first'] > 220, 'height_first'] = np.nan
tempData.loc[tempData['height_first'] < 100, 'height_first'] = np.nan

tempData.loc[tempData['weight_first'] > 300, 'weight_first'] = np.nan
tempData.loc[tempData['weight_first'] < 30, 'weight_first'] = np.nan

tempData.loc[tempData['age'] > 300, 'age'] = 91.4

'''
对缺失值进行处理
使用 knnimpute 工具包
'''
# from knnimpute import (knn_impute_few_observed, knn_impute_optimistic, knn_impute_reference,
#                       knn_impute_with_argpartition, knn_initialize)
#
# knnData = tempData.loc[0:20000,
#          ['pao2', 'spo2', 'fio2', 'hr', 'temp', 'nbps', 'nbpd', 'nbpm', 'abps', 'abpd', 'abpm', 'rr', 'tv', 'mv',
#           'pip', 'plap', 'map', 'peep', 'gcseyes', 'gcsmotor', 'gcsverbal', 'height_first', 'weight_first', 'age']]
# knnDataArray = np.array(knnData)
# markData = np.array(knnData.isnull())

# ypc=knn_impute_optimistic(knnDataArray, markData, k=3,verbose=True)
# ypc=knn_impute_few_observed(knnDataArray, markData, k=3)

'''
对于血压这些缺失数据，我的处理方法和思路
我利用随机森林和knn 两种方法进行插值
1. 使用随机森林利用患者的基本信息 + 有创血压--》预测无创血压 ，但是假如患者的有创血压也确实，此时，这部分数据无法使用这个方法
2. 基于以上的处理结果，我们在使用knn来预测剩下的缺失值
'''
from sklearn.ensemble import RandomForestRegressor


def nbpRandom(tempData, nbp, abp):
    abpNotNull = tempData[tempData[abp].notnull()]  # 后续只使用这一部分数据
    # 把abpNotNull分为nbpNull和nbpNotNull两部分
    nbpNull = abpNotNull[abpNotNull[nbp].isnull()]  # 测试集
    nbpNotNull = abpNotNull[abpNotNull[nbp].notnull()]  # 训练集
    X = nbpNotNull.loc[:, [abp, 'hr', 'rr', 'age', 'subject_id']]
    X = X.fillna(X.mean())
    y = nbpNotNull[nbp]
    regr = RandomForestRegressor(max_depth=2, random_state=0)
    regr.fit(X, y)
    X_pre = nbpNull.loc[:, [abp, 'hr', 'rr', 'age', 'subject_id']]
    X_pre = X_pre.fillna(X_pre.mean())
    nbpPre = regr.predict(X_pre)
    abpNotNull.loc[abpNotNull[nbp].isnull(), nbp] = nbpPre
    tempData.loc[tempData[abp].notnull(), nbp] = abpNotNull[nbp]
    return tempData


tempData = nbpRandom(tempData, nbp='nbps', abp='abps')
tempData = nbpRandom(tempData, nbp='nbpm', abp='abpm')
tempData = nbpRandom(tempData, nbp='nbpd', abp='abpd')
#  temp = tempData.pip.value_counts()
# temp.index.values
# plt.bar(temp.index.values, list(temp))
# plt.show()
# naData = np.array(temp)
# missMark = np.array(temp.isnull())
