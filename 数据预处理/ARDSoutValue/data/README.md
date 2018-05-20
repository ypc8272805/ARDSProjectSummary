# imputeData.csv
* 去除异常值 补全缺失值的数据集
* 数据来源：数据库中的secondData中的数据
* outValuePre.py最终输出结果
* 处理方法：<br>1）处理异常值<br>2）使用随机森林模型来预测血压、气道压力等缺失比较严重的数据，但是这个文件中任然有缺失值<br>3)利用knn对剩余缺失值进行处理
 'subject_id', 'hadm_id', 'icustay_id', 'charttime',
 'pao2', 'spo2', 'fio2', 'hr', 'temp', 'nbps', 'nbpd', 'nbpm', 'abps',
 'abpd', 'abpm', 'rr', 'tv', 'pip', 'plap', 'mv', 'map', 'peep',
 'gcsmotor', 'gcsverbal', 'gcseyes', 'first_careunit', 'last_careunit',
 'dbsource', 'age', 'ethnicity', 'admission_type', 'gender',
 'height_first', 'height_min', 'height_max', 'weight_first',
 'weight_min', 'weight_max'

---
# preData.csv train_data.csv test_data.csv
* 特征工程 处理哑变量 特征组合变换
* 数据来源：imputeData.csv
* featurePre.py最终输出结果
* 处理方法：<br>1)主要是处理哑变量、特征组合、处理结局变量、归一化
<br>2)将患者分为训练集和测试集，患者随机化处理的文件是random_subject_id.csv文件，我们选择前6601名患者为训练集，剩余的2102名患者为测试集
