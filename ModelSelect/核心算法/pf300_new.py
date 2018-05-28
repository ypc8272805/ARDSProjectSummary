# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 16:33:00 2018
这一程序就是整个算法的核心，其中包括数据预处理、特征值选择、交叉验证所有过程
调用了外部函数包mymodel.py
@author: ypc
"""
import numpy as np
import pandas as pd
import time
from pandas import DataFrame
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler 
import mymodule as my
import matplotlib.pyplot as plt

#读取数据
mimicdata = pd.read_csv('D:/mimicdata/data/nomissdata_pf.csv')
x=mimicdata.loc[:,['age_new','bmi','gender_num','eth_num','spo2','fio2','hr','rr','temp','m_nbps','m_nbpd','m_nbpm'
                   ,'tv','mv','tv_w','m_pip','m_plap','map','m_peep','sf','osi']]
names=x.columns
Y=np.array(mimicdata.loc[:,'two_class_300'])
#数据归一化

max_abs_scaler=MaxAbsScaler()
xs=max_abs_scaler.fit_transform(x)
#特征值排序
feature_R=my.select_feature_R(xs,Y,names)
feature_P=my.select_feature_P(xs,Y,names)
feature_P['R_mean']=feature_R
feature_P['mean']=feature_P.mean(axis=1)
feature_P=feature_P.sort_values(by = 'mean',axis = 0,ascending = False)  
#准备数据，训练模型
X=np.array(x[list(feature_P.index)])
max_abs_scaler=MaxAbsScaler()
X=max_abs_scaler.fit_transform(X)
size=np.shape(X)
names=[]
s_result=DataFrame()
for i in range(1,(size[1]+1)):
    data_fre=X[:,0:i]
    skf = StratifiedKFold(n_splits=5)
    
    Result=pd.DataFrame()#用于存储结果
     
    for train_index, test_index in skf.split(data_fre, Y):
        x_train, x_test =data_fre[train_index], data_fre[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        ros=RandomOverSampler(random_state=50)    #过采样 
        x_res,y_res=ros.fit_sample(x_train,y_train)
        
        MLPCResult,M_fpr,M_tpr=my.MLPC(x_res,y_res,x_test,y_test)#神经网络，打包起来
        temp=pd.Series(MLPCResult,name='MLPC')
        Result=pd.concat([Result,temp],axis=1)#将结果存储起来
        
        BaggingRes,B_fpr,B_tpr=my.Bagging(x_res,y_res,x_test,y_test)
        temp=pd.Series(BaggingRes,name='Bag')
        Result=pd.concat([Result,temp],axis=1)
        
        AdaRes,A_fpr,A_tpr=my.AdaBoost(x_res,y_res,x_test,y_test)
        temp=pd.Series(AdaRes,name='Ada')
        Result=pd.concat([Result,temp],axis=1)
        
        LogRes,L_fpr,L_tpr=my.LogReg(x_res,y_res,x_test,y_test)
        temp=pd.Series(LogRes,name='Log')
        Result=pd.concat([Result,temp],axis=1)
    
    MLPC='MLPC'+str(i)
    Bag='Bag'+str(i)
    Ada='Ada'+str(i)
    Log='Log'+str(i)
    s_result[MLPC]= Result['MLPC'].mean(axis=1)   
    s_result[Bag]= Result['Bag'].mean(axis=1) 
    s_result[Ada]= Result['Ada'].mean(axis=1)   
    s_result[Log]= Result['Log'].mean(axis=1)   
    print(i)
    print(time.asctime(time.localtime(time.time())))

#对特征值选择绘制直方图
fea_num=21
pearson_num=feature_P.values[:,:1]
R_num=feature_P.values[:,1:2]

fig=plt.figure()
ax1=fig.add_subplot(2,1,1)
index=np.arange(fea_num)
bar_width=0.3
first=ax1.bar(index,pearson_num,bar_width,
             color='#7986CB',label='Pearson相关系数')
second=ax1.bar(index+bar_width,R_num,bar_width,
             color='#1A237E',label='Random Forest重要性评分')

ax1.set_ylabel('重要性评分',fontsize=13)
ax1.set_title('A-特征值选择',fontsize=15)
ax1.set_xticks(index)
ax1.set_xlabel('特征值',fontsize=13)
ax1.set_yticks((0,0.5,1))
ax1.set_xticklabels(('S/F','SpO2','FiO2','OSI','PEEP','MAP','PLAP','PIP','MV','RR','BMI',
                     'TV/kg','TEMP','Gender','HR','NBPM','NBPD','NBPS','Age','TV','Eth'),fontsize=13) 
ax1.legend()

ax2=fig.add_subplot(2,1,2)
mean_num=feature_P.values[:,2:3]
third=ax2.bar(index,mean_num,bar_width,
             color='#3949AB')
ax2.set_title('B-特征值重要性综合评分',fontsize=15)
ax2.set_xlabel('特征值',fontsize=13)
ax2.set_ylabel('重要性评分',fontsize=13)

ax2.set_xticks(index)
ax2.set_yticks((0,0.5,1))
ax2.set_xticklabels(('S/F','SpO2','FiO2','OSI','PEEP','MAP','PLAP','PIP','MV','RR','BMI',
                     'TV/kg','TEMP','Gender','HR','NBPM','NBPD','NBPS','Age','TV','Eth'),fontsize=13) 

fig.tight_layout()
#plt.subplots_adjust(wspace=0,hspace=0)
plt.show()

#绘制每种算法的敏感性、特异性、准确率、AUC
index=range(3,84,4)
mlpc=s_result.iloc[:,index]*100
fig2,ax=plt.subplots()
t=range(1,22)
data1=mlpc.iloc[9,:]#准确率
data2=mlpc.iloc[5,:]
data3=mlpc.iloc[6,:]
data4=mlpc.iloc[10,:]
ax.plot(t,data1,color='#F50057',linestyle='-',label='准确率')
ax.plot(t,data2,color='#1E88E5',linestyle='-.',label='敏感性')
ax.plot(t,data3,color='#0D47A1',linestyle=':',label='特异性')
ax.plot(t,data4,color='#388E3C',linestyle='--',label='AUC')        
#ax.plot([19,19],[62,88],color='#000000',linestyle='--',label='Best')
ax.annotate('BEST',xy=(20,data1[20]+1),
            xytext=(19.5,data1[20]+5),
            arrowprops=dict(facecolor='black'),
            horizontalalignment='left',verticalalignment='top')
ax.set_yticks((60,70,80,90))
ax.set_xticks(np.arange(1,21,3))
ax.set_xlabel('特征值个数') 
ax.set_ylabel('结果(%)')
ax.set_title('不同特征子集下AdaBoost结果')
ax.legend(loc=1)
fig2.tight_layout()
plt.show()