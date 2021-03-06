# 基于无创参数辨识ARDS疾病严重程度的算法研究

---

## 目录
- 背景介绍
- 数据来源
- 患者选择
- 数据提取
- 特征值选择
- Model Select
- 实验结果
- 总结

---
## 背景介绍
**急性呼吸窘迫综合征**是一种严重威胁人类生命健康的疾病，具有起病急、病死率高等特点。目前这种疾病的主要诊断和疾病严重程度分级标准依赖于**血气分析**结果，计算患者的氧合指数(PaO2/FiO2),但是血气分析是有创操作，且不能连续监测病情的发展。针对以上问题，本文基于患者的多种无创生理参数，使用**神经网络、逻辑回归、AdaBoost、Bagging**监督学习算法，结合特征值选择技术，研究无创参数下辨识ARDS疾病的算法模型，为医务人员提供辅助诊断决策。

---
## 数据来源
本项目使用的所有临床数据均来自开源数据库**MIMIC-III**，可以访问[这里](https://mimic.physionet.org/)了解更多关于MIMIC数据库的相关信息。MIMIC-III（Medical Information Mart for Intensive Care，V1.4）是由麻省理工学院（MIT）和贝斯以色列女执事医疗中心（BIDMC）共同开发的一个大型的单中心的重症监护医学信息数据库，目前可免费申请使用。截至目前数据库中包含了2001年至2012年期间BIDMC重症监护病房的上万名患者的相关诊疗信息数据。数据库包括人口统计学信息(去隐私化的信息)、床旁生命体征监测数据（包括波形数据）、实验室化验数据、患者用药信息、护理人员日志信息、影像学检查报告、患者死亡率等信息。目前数据库中记录了患者46520名（61532次ICU记录），其中成年患者38597名（49785次ICU记录）。MIMIC-III数据库也存在一些问题，例如：
- 是一个单中心的数据库，同时由于数据量庞大，时间跨度超过10年，很多数据不是很准确，所以使用者还是要非常认证去筛选数据，当然有如此一个多的ICU患者信息让我们去研究，还是值得庆幸的；
- 当然MIMIC-III不是唯一的ICU重症患者数据库，大家也可以选择E-ICU数据库，由于我所在的实验室前期使用的时MIMIC，所以暂时使用这个做相关数据挖掘的研究，后续会使用e-ICU进行相关研究，你可以在[这里](http://eicu-crd.mit.edu/)了解有关eICU的更多信息；
- 如果对于ARDS疾病而言，还有很多的专业数据库，其提供其他研究人员的实验数据，这些实验设计都是研究人员根据自己的需要定义的，虽然有一定的局限性，但是所有数据都有高可靠性，这一点就足够让我们去尝试一下。

---
## 患者选择
在确定了我要使用的临床数据库后，着手开始选择我们需要研究的对象，**这一部分的工作需要大量的文献调研，同时一定要再三核对，一旦入组患者选择出现问题，后面你的很多工作都要推倒重来**。我们根据研究需要制定了以下的患者入组标准：
1. 患者年龄大于16周岁；
2. 在ICU停留时间超过48小时：*如果停留时间太短，这部分患者的病情很难判断，也会对整体数据造成不必要的干扰；*
3. 只选择第一次进入ICU的患者数据：*如果一个病人重复多次进入ICU，则有可能病人病情较为复杂，有可能会对实验结果造成影响，所以在本文中，只使用每个患者第一次进入ICU的数据；*
4. 在ICU期间进行过胸部影像学检查；
5. 在ICU期间进行了机械通气的患者：*有创通气和无创通气患者都包括在内；*
6. 第一天PF有小于300的情况：*第一天的pf最小值小于300 就认为符合条件。*

在具体操作过程中，要把每一个条件具体化为代码，最终从数据苦衷选择合适的患者，这一部分的具体实现可以通过以下目录中查看获取：
```
.../数据提取/1患者筛选工作
```
这里详细介绍了我是如何选择合适的病人的完整SQL代码。

---
## 数据提取
这一部分工作比较耗时，具体工作内容就是，通过选择好的病人，提取这些患者的生理参数，并整理数据，用于后期的数据分析，要熟悉数据库的表结构，了解可能用到的所有患者数据，当这些准备工作都做完之后，才开始数据提取工作，否则你会哭的。这里不用担心浪费时间，浪费的时间都是有价值的，为后面节约很多时间。


数据提取涉及到数据库的多个表和后期MIMIC团队在github中共享的视图，要合理使用这些试图，可以帮你省去很多时间，建议将github上的所有视图全部跑一边，不要嫌浪费时间，最多也就花一天时间可以跑完所有视图。这些视图不仅会帮助你快速提取数据，还能从别人的视图语言中学到很多有用的语法表达。

当然具体我怎么提取数据的，这些代码如何实现的，其实我觉得应该帮不上你什么忙，因为这都是针对我自己的问题写的代码，不一定有通用性，且必须明白我的目的才能看懂这些代码，写在这里只是方便我自己后期查看。

```
.../数据提取/2chartevents数据提取
.../数据提取/3labevents数据提取
.../数据提取/4数据合并数据匹配
```

### 1、异常值处理

MIMIC-III中的数据质量还是存在一些问题，质量不是很高，很多不符合生理情况的数值，这些问题，都要对每种生理参数结合患者的具体信息进
行分析，才能确定异常值范围。<br>
```angular2html
../数据预处理/ARDSoutValue/datapre.ipynb
../数据预处理/ARDSoutValue/outValuePre.py
```
以上文件中详细描述了如何处理异常值的方法，并对缺失值进行处理。

### 2、缺失值处理
```angular2html
../数据预处理/ARDSoutValue/outValuePre.py
```

## 特征值排序
`ModelSelect/model_select/featureSelection.py`
## 模型选择
`ModelSelect/model_select/pf300.py`
## 模型训练
`ModelSelect/model_train/feature_select_train.py`
## 预测