# 模型训练
---
正对每种算法，都会有一个最小特征子集 最优特征子集 和全部特征子集
针对每一种情况，进行一次十折交叉验证，获得实验结果，最终结果保存在Result.csv中，![BER](https://github.com/ypc8272805/ARDSProjectSummary/blob/master/ModelSelect/model_train/Figure_1.png)
<br>图中所示为四种方法在三种不同特征子集下的BER变化情况，通过横向比较，可以看出不同模型之间存在较大差异，相同模型的不同特征子集BER差异较小，
整体来看，还是XGBoost的效果好。