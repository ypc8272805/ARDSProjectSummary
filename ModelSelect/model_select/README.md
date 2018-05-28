# 算法简介
本目录下，收录了关于ARDS疾病辨识的所有相关算法，其中
* `核心算法`中的内容为之前的实验算法
* `featureSelect.py`是特征值排序的文件
* `OverSampling.py`是关于过采样的实验代码
* `test_*.py xgboost_early_stop.py`都是一些实验代码
* `algo.py,pf300.py`是具体的辨识算法的模型和特征选择过程
---
## 特征选择
特征值选择的方法，大致分为三种类型，即Filter 、Wrapper、Embedded三种方式，本研究在Filter的基础上做了一些调整。
<br>Filter特征选择方法主要是研究特征值于结局变量（或预测结果）之间的关系，就这一方法计算简单、使用灵活，在特征值选择中较
为常用。目前比较常用的Filter特征值选择方法众多，但是一般分为三大类。
* **Distance**:Relief,Euclidean distanc...
* **Dependence**:Pearson Chi-squared ...
* **Information**:Mutual Information  Information Gain ...<br>
当然，除了上面提到的方法，还有很多，如果想了解更多可以访问这里：[link to FeatureSelect@ASU](featureselection.asu.edu)<br>
---
### Feature Select method 
我选用了Rlief、Chi-squared、Mutual Information三种方法，从三个方面综合评估了特征值重要性的相关信息，再利用Filter Rank Aggregation方法，将三种结果融合，得到最终的特征值排序结果。<br>
如果想要了解每种算法的具体原理，可以自行查阅相关资料，后期有时间我也会对这部分内容进行补充。<br>
特征值排序结果保存在了`scores.csv`中，方便后续算法调用排序结果。

## 辨识算法
本研究使用四种常用的机器学习算法：
* 神经网络
* 逻辑回归
* AdaBoost
* XGBoost<br>
其中，前三种方法使用的是scikit-learn进行实现。具体的算法在`algo.py`中。
## 模型评估
利用交叉验证方法，评估者四种方法在不同特征子集下的辨识效果，具体实现在`pf300.py`中。
<br>最终计算结果存储在`result.csv`中，结果中存储了每种算法对应的不同特征子集的结果，用于后续分析。
## 性能指标
SEN SPE ACC BER AUC ROC曲线 不同特征子集下的BER AUC曲线变化情况。使用适当的方法选择合适的特征子集和算法。具体实现在`algo.result()`中
## 特征选择的结果
根据`result.csv`中的结果，我们可以找到每种算法的最优特征子集，具体程序在`resultPlot.py`中。
![Feature Select Result](https://github.com/ypc8272805/ARDSProjectSummary/blob/master/ModelSelect/model_select/result/featureSelect2.png)
<br>从图中，我们也可以看出每种算法，在不同特征子集下的BER变化趋势。