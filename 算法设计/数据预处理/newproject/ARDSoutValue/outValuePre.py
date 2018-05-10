# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from knnimpute import (knn_impute_few_observed, knn_impute_optimistic, knn_impute_reference,
                       knn_impute_with_argpartition, knn_initialize)

# 直接从数据库中提取相关数据
engine = create_engine('postgresql://postgres@localhost:5432/mimic', echo=True)
sql = 'select * from mimiciii.secondData'
data = pd.read_sql_query(sql, con=engine, index_col=None, coerce_float=True, params=None, parse_dates=None,
                         chunksize=None)
# 这里主要是提出需要的数据进行处理
tempData = data.loc[:, ['subject_id', 'hadm_id', 'icustay_id', 'pao2', 'spo2', 'fio2',
                        'hr', 'temp', 'nbps', 'nbpd', 'nbpm', 'abps', 'abpd', 'abpm', 'rr', 'tv', 'mv', 'pip', 'plap',
                        'map', 'peep', 'gcseyes', 'gcsmotor', 'gcsverbal', 'height_first', 'weight_first']]
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
tempData.loc[tempData['nbps'] < tempData['nbpm'], ['nbps', 'nbpm', 'nbpm']] = np.nan
tempData.loc[tempData['nbps'] < tempData['nbpd'], ['nbps', 'nbpm', 'nbpm']] = np.nan


# temp = tempData.temp.value_counts()
# temp.index.values
# plt.bar(temp.index.values, list(temp))
# plt.show()
# naData = np.array(temp)
# missMark = np.array(temp.isnull())
