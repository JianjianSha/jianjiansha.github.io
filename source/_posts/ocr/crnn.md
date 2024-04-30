---
title: CRNN 论文解读
date: 2024-04-07 14:37:33
tags: ocr
---

论文：[An End-to-End Trainable Neural Network for Image-based Sequence
Recognition and Its Application to Scene Text Recognition](https://arxiv.org/abs/1507.05717)

本文是一个序列目标识别网络。

# 1. 简介

本文提出 CRNN 模型，一种深度卷积神经网络 DCNN 与循环神经网络 RNN 的结合，用于识别图像中的序列目标（例如文本），模型优点：

1. 直接从序列标签中学习（例如单词），不需要详细的标签（例如字符）
2. 与 DCNN 一样直接从图像数据中学习特征
3. 与 RNN 一样可以生成序列标签
4. 序列长度不受限制
5. 在场景文本识别中比之前的工作效果更好
6. 比标准 DCNN 的参数更少

# 2. 网络架构

网络架构如图 1 所示，

![](/images/ocr/crnn_1.png)

<center>图 1</center>

整个网络包含三个部分：

1. 卷积层
2. 循环层
3. transcription layer

## 2.1 特征提取

图像需要 rescale 到相同高度，然后喂给 CNN。相同的高度可以保证提取的特征维度相同，图像的宽度可以不同（但是同一个 batch 中的宽度还是要相同的），不同的宽度得到不同长度的特征序列。CNN 输出特征之后，从左到右按列提取特征向量，得到特征序列，如图 2，

![](/images/ocr/crnn_2.png)

<center>图 2</center>

## 2.2 序列标签

使用双向循环神经网络，对输入特征序列 $x _ t$，输出为标签预测分布 $y _ t$ 。使用 RNN 有三点原因：

1. 可以很好地捕获序列中的上下文信息，使得序列目标识别更加稳定

    以场景文本识别为例，宽字符需要更多数量的特征向量。另外，一些模棱两可的字符通过观察上下文更容易区分，例如区分 `il` 两个字符

2. RNN 可以将误差梯度反向传播到输入，也就是 CNN 层，使得我们可以将 CNN 和 RNN 统一到一个网络中训练

3. RNN 可以处理任意长度的序列

一个 RNN 单元可以表示为，

$$h _ t = g(x _ t, h _ {t-1})$$

其中，$x _ t$ 是当前时刻 $t$ 的输入，$h _ {t-1}$ 是上一个时刻的内部隐层状态，输出是当前时刻的状态 $h _ t$ 。

传统 RNN 会出现梯度消失的问题，可以使用 LSTM 解决这个问题。为了能利用两侧的上下文信息，我们堆叠两个 LSTM 层，一个前向一个后向，得到双向 LSTM 层，然后可以堆叠多个这样的双向 LSTM 层，得到深度双向 LSTM，如图 3 (b)

![](/images/ocr/crnn_3.png)

<center>图 3</center>

## 2.3 Transcription

Transcription 层用于将 RNN 输出的 label 预测转为 label 序列，RNN 输出的 label 预测是每个时刻的 label 概率分布，如果单独对每个时刻根据 argmax 预测 label 值，显然不妥，因为没有将 label 序列作为一个整体考虑。

### 2.3.1 标签序列概率

使用连接性时序分类 CTC 。记输入为预测序列 $\mathbf y =y_1,\ldots,y_T$ ，其中 $y _ t \in \mathbb R ^ {|L'|}$ 是时刻 $t$ 的标签概率分布，$L' = L \cup \lbrace - \rbrace$ 是所有的标签集合（例如英文中的所有字符），以及一个空白字符，使用 $-$ 表示，因为 gt 标签序列长度 $\le T$，可能存在一些 $y _ t$ ，其 gt label 为空白符 $-$ 。

一个序列到序列的映射 $B$ 其输入为 $\pi \in L ^ {'T}$，输出为 $\mathbf l$，$B$ 做的事情是：1. 移除连续重复的标签，2. 移除空白符。

例如 $B$ 的一个输入为 `--hh-e-l-ll-oo-`，输出为 `hello` 。

$B$ 对应的条件概率为

$$\begin{aligned}p(\mathbf l|\mathbf y) & = \sum _ {\pi:B(\pi)=\mathbf l} p(\pi | \mathbf y)
\\
p(\pi|\mathbf y) &= \prod _ {t=1}^T y _ {\pi _ t} ^ t 
\end{aligned}\tag{1}
$$

其中 $y _ {\pi _ t} ^ t$ 是 $t$ 时刻的预测概率向量中 $\pi _ t$ 对应的概率值。

### 2.3.2 无词典转录

此模式下，根据 (1) 式，具有最大概率 $p(\mathbf l|\mathbf y)$ 的序列 $\mathbf l ^ *$ 就是最终的预测序列。可以使用下式近似得到

$$\mathbf l ^ * \approx B(\arg \max _ {\pi} p(\pi | \mathbf y)) \tag{2}$$

也就是说，每个时刻单独预测。

### 2.3.3 基于词典的转录

此模式下，有一个词典 $\mathcal D$，标签序列识别为词典中具有最大条件概率 $p(\mathbf l|\mathbf y)$ 的词条，即

$$\mathbf l ^ * = \arg \max _ {\mathbf l \in \mathcal D} p(\mathbf l|\mathbf y)$$

对于大词典，使用穷举法搜索将会非常耗时。为解决此问题，我们观察到在无词典转录模式下，使用 (2) 式计算的近似结果通常接近 gt（即，编辑距离较小），于是我们可以将搜索限制到最近邻候选集 $\mathcal N _ {\delta}(\mathbf l')$，其中 $\delta$ 是最大编辑距离，$\mathbf l'$ 是从 $\mathbf y$ 根据无词典转录得到的序列，于是最终预测序列为

$$\mathbf l ^ * = \arg \max _ {\mathbf l \in \mathcal N _ {\delta}(\mathbf l')} p(\mathbf l | \mathbf y)
\\
\mathbf l' = B(\arg \max _ {\pi} p(\pi | \mathbf y))$$

得到 $\mathbf l'$ 之后，候选集 $\mathcal N _ {\delta}(\mathbf l')$ 可以根据 BK-树 数据结构高效地获取。

## 2.4 网络训练

记训练集为 $\mathcal X = \lbrace I _ i , \mathbf l _ i \rbrace _ i$ （对于 ocr 任务，需要先检测，后识别，本文主要是识别）。目标函数为

$$\mathcal L = -\sum _ {\mathbf I _ i, \mathbf l _ i \in \mathcal X} \log p(\mathbf l _ i |\mathbf y _ i)$$