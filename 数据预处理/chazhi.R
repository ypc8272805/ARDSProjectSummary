#数据插值工作
#插值的对象主要是数据确实严重的数据：使用有创血压补全无创血压，对一些呼吸参数进行补全。
#使用的方法：MICE包对数据进行补全，这个补全数据，不是说仅通过自身全是数据去补全，而是寻找与缺失数据相关性较高的数据来补全，
#这样的缺失值处理更加可靠

library(lattice)
library(mice)
#前期已经使用的相关函数对所有变量进行了分析，当然从变量的实际意义来说，也应该是使用有创舒张压补全无创舒张压
#这里使用的method方法，选择这个是一个线性预测模型，他的运算速度比较快，也是我们需要使用的一种模型
imp<-mice(value_pf[c("nbps","abps")],m=5,seed=123,method = 'norm.predict')
fit<-with(imp,lm(nbps~abps))
pooled<-pool(fit)
summary(pooled)
#这个插值方法，会产生物种不同的插值结果，最终我们是使用这5组的平均值作为最后的结果
new_data1<-complete(imp,action = 1)
new_data2<-complete(imp,action = 2)
new_data3<-com?lete(imp,action = 3)
new_data4<-complete(imp,action = 4)
new_data5<-complete(imp,action = 5)
#将结果直接保存在原有数据集中，列明发生了变化
value_pf$m_nbps<-ceiling((new_data1$nbps+new_data2$nbps+new_data3$nbps+new_data4$nbps+new_data5$nbps)/5)

imp<-mice(value_pf[c("nbpd","abpd")],m=5,seed=123,method='norm.predict')
fit<-with(imp,lm(nbpd~abpd))
pooled<-pool(fit)
summary(pooled)
new_data1<-complete(imp,action = 1)
new_data2<-complete(imp,action = 2)
new_data3<-complete(imp,action = 3)
new_data4<-complete(imp,action = 4)
new_data5<-complete(imp,action = 5)
value_pf$m_nbpd<-ceiling((new_data1$nbpd+new_data2$nbpd+new_data3$nbpd+new_data4$nbpd+new_data5$nbpd)/5)

imp<-mice(value_pf[c("nbpm","abpm")],m=5,seed=123,method = 'norm.predict')
fit<-with(imp,lm(nbpm~abpm))
pooled<-pool(fit)
summary(pooled)
new_data1<-complete(imp,action = 1)
new_data2<-complete(imp,action = 2)
new_data3<-complete(imp,action = 3)
new_data4<-complete(imp,action = 4)
new_data5<-complete(imp,action = 5)
value_pf$m_nbpm<-ceiling((new_data1$nbpm+new_data2$nbpm+new_data3$nbpm+new_data4$nbpm+new_data5$nbpm)/5)

imp<-mice(value_pf[c("pip","plap")],m=5,seed=123,method = 'norm.predict')
fit<-with(imp,lm(pip~plap))
pooled<-pool(fit)
summary(pooled)
new_data1<-complete(imp,action = 1)
new_data2<-complete(imp,action = 2)
new_data3<-complete(imp,action = 3)
new_data4<-complete(imp,action = 4)
new_data5<-complete(imp,action = 5)
value_pf$m_pip<-ceiling((new_data1$pip+new_data2$pip+new_data3$pip+new_data4$pip+new_data5$pip)/5)
value_pf$m_plap<-ceiling((new_data1$plap+new_data2$plap+new_data3$plap+new_data4$plap+new_data5$plap)/5)

imp<-mice(value_pf[c("map","peep")],m=5,seed=123,method = 'norm.predict')
fit<-with(imp,lm(map~peep))
pooled<-pool(fit)
summary(pooled)
new_data1<-complete(imp,action = 1)
new_data2<-complete(imp,action = 2)
new_data3<-complete(imp,?ction = 3)
new_data4<-complete(imp,action = 4)
new_data5<-complete(imp,action = 5)
value_pf$m_peep<-ceiling((new_data1$peep+new_data2$peep+new_data3$peep+new_data4$peep+new_data5$peep)/5)
value_pf$logsf<-log(value_pf$sf) 
value_pf$tv_w<-value_pf$tv/value_pf$weight_first
newdata=value_pf[c("subject_id","hadm_id","icustay_id","spo2","pao2","fio2","hr","temp","m_nbps","m_nbpd","m_nbpm","rr","tv","tv_w","m_pip","m_plap","mv","map",
                       "m_peep","gender_num","eth_num","age_new","sf","logsf","bm?","osi","pf","oi","two_class_100","two_class_200","two_class_300","four_class","three_class")]
ypc=value_pf[c("subject_id","hadm_id","icustay_id","spo2","pao2","fio2","hr","temp","nbps","nbpd","nbpm","rr","tv","pip","plap","mv","map",
                   "peep","gender_num","eth_num","age_new","sf","bmi","osi","pf","oi","two_class_100","two_class_200","two_class_300","four_class","three_class")]
ypc1=na.omit(ypc)
ypc2=cor(ypc1)
#将数据存储起来
#nomissdata=na.omit(newdata)
#write.csv(nomissdata,file = 'D:\\mimicdata\\data\\nomissdata_pf.csv')

