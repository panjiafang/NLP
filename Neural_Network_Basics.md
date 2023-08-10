# 神经网络基础

## 1. 神经网络的基本组成

Neural Network: 神经网络，全称为人工神经网络（Artificial Neural Network, ANN）

#### Neuron: 神经元

通过生物神经网络的神经元演变出人工神经元

由多个输入、一个输出、参数w和b、激活函数f构成

举例：
![Neuron](/images/neuron.png)

权重向量w*输入向量x+偏置b，通过激活函数f，得到输出y

#### 单层神经网络(Single Layer Neural Network)

以下是一个包含3个神经元的单层神经网络

![Single Layer Neural Network](/images/single-layer-neural-network.png)

计算方式如下：

![Single Layer Neural Network Calculation](/images/single-layer-neural-network-matrix.png)

#### 多层神经网络(Multi Layer Neural Network)

![Multi Layer Neural Network](/images/multi-layer-neural-network.png)

前向计算

中间层称为隐层

![Feedforward Computation](/images/feedforward-computation.png)

#### 非线性的激活函数(Non-Linearity Activation Function)

常见的有

- sigmoid 一个负无穷到正无穷的输入转化为0-1之间的数

- tanh 一个负无穷到正无穷的输入转化为-1到1之间的数

- ReLU 输入为正数，输出仍为正数，输入为负数，输出为0

![Activation Function](/images/functions.png)

#### 输出层(Output Layer)

主要用于验证神经网络的输出是否符合预期

常用的有

- softmax 用于多分类问题，输出为各个类别的概率

- sigmoid 用于二分类问题，输出为0-1之间的数


## 2. 神经网络的训练

TODO 对应[课程P13 2-4 如何训练神经网络](https://www.bilibili.com/video/BV1UG411p7zv?p=13)

训练神经网络的目的是为了减小损失函数的值。损失函数的值越小，神经网络的输出越接近真实值

#### 交叉熵(Cross Entropy)

交叉熵可以作为损失函数，用于衡量训练后的神经网络输出的概率分布与真实概率分布之间的差异性。

#### 梯度下降(Gradient Descent)

梯度下降是一种优化算法，用于最小化损失函数的值。

#### 反向传播算法(Backpropagation)

反向传播算法是一种高效计算梯度的方法，深度学习框架（TensorFlow、PyTorch）都有使用

## 3. Word2Vec

Word2Vec是一种将词语转化为向量的技术，Word2Vec的输入是一个词语序列，输出是词语的向量表示

#### Word2Vec的两种模型

- CBOW(Continuous Bag-of-Words)模型: 通过上下文推理出目标词
  
- Skip-Gram模型: 通过目标词推理出上下文

![Word2Vec Types](/images/word2vec-types.png)

#### 滑动窗口(Sliding Window)

#### 负采样(Negative Sampling)

## 4. 循环神经网络(Recurrent Neural Network, RNN) 

处理序列数据会产生顺序记忆

#### 顺序记忆(Sequential Memory)

一种更容易识别序列数据的记忆方式

RNN递归的更新顺序记忆以此来对序列数据进行建模

#### RNN的结构

![RNN Structure](/images/rnn.png)

#### RNN单元

![RNN Cell](/images/rnn-cell.png)

#### RNN模型实例

![RNN Model ](/images/rnn-model-p.png)

#### RNN场景

- 序列标注

- 序列预测

- 图片描述

- 文本分类

#### RNN优缺点

- 优点

    - 可以处理任意长度的数据

    - 参数共享

    - 模型大小不会随着序列长度增加而增加

- 缺点

    - 顺序计算比较耗时
  
    - 难以捕捉长期依赖关系

#### RNN梯度问题

- 梯度消失和梯度爆炸

#### RNN变种

- 门控循环单元(Gated Recurrent Unit, GRU)

- 长短期记忆网络(Long Short-Term Memory Network, LSTM) 


## 5. 门控循环单元(Gated Recurrent Unit, GRU)

通过门控机制，决定哪些信息可以传入进来，哪些信息不能往下传出去

#### 更新门和重置门

## 6. 长短期记忆网络(Long Short-Term Memory Network, LSTM)

增加了Cell State，用于存储长期记忆

#### 遗忘门和输入门

## 7. 双向RNN(Bidirectional RNN)

![Bidirectional RNN](/images/bi-rnn.png)

## 8. 卷积神经网络(Convolutional Neural Network, CNN)

一般用于计算机视觉领域(Computer Vision, CV)，后面用在了NLP领域处理如：情感分类、关系分类等问题

归功于CNN在提取局部和位置不变的特征方面的优势
