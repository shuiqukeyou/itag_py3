## 介绍
论文**An Integral Tag Recommendation Model for Textual Content**的模型python3实现


官方有一个[python2版本的实现](https://github.com/SoftWiser-group/iTag )

在python3下有一堆BUG，且不能支持高版本的CUDA，使用python进行了重构并且复现了论文结果，调整了代码结构，另外注释拉满

## 需求
- Numpy >= 1.14.3 
- Tensorflow >= 1.8.0
- Keras >= 2.1.6

## 说明
实际上对官方代码进行了一些修改:
- tag的序号编码方式等
- 文件结构
- 变量名
- 一些需要手工设置的变量改为了自动方式

使用同样的数据集跑出来的结果和论文的基本接近，可以认为是实现了该模型

## 文件结构

- main.py：主文件
- dataload.py：数据加载函数文件。官方原版运行前需要手动运行dataload.py文件，改版后如无特殊情况可以不需要手动运行
- congfig.py：配置文件，保留了官方项目中声明但未使用的变量。官方原版无此文件，参数在main.py中更改
- function.py：函数文件，存放成本函数、精确、召回等函数。官方原版分散在各处
- layer.py：层对象文件，定义了attention层和mask层。官方原版还有几个别的层

- shared.txt：文本、tag共现词汇对，需要预处理时生成（未提供）
- data.npz：数据文件（未提供）