---
title: 基于卷积网络的中文分词
date: 2022-02-22 14:20:29
tags: NLP
summary: 总结若干基于深度学习方法的中文分词
img: /images/nlp/convcws_0.png
mathjax: true
categories: 自然语言处理
---

# 1. ConvCWS

论文：[Convolutional Neural Network with Word Embeddings for Chinese Word Segmentation](https://arxiv.org/abs/1711.04411)

将中文分词看作 sequence labaling 任务，例如

```
S   S   B   E  B   M   E   S
我  有  一  台  计  算  机  。
```

## 1.1 网络框架

### 1.1.1 卷积网络

1. embedding

    一个句子中各字符通过 embedding，即需要一个映射表 $M_c \in \mathbb R^{V \times d}$，其中 $V$ 为字符集大小，$d$ 为 embedding 维度，假设一个句子长度为 $L$，那么 embedding 后得到 $X \in \mathbb R^{L \times d}$。

2. 堆叠 conv layer

    如图 1 所示，

    ![](/images/nlp/convcws_1.png)

    图 1.

    conv layer 的输入 channel 为 $N$，输出 channel 为 $M$，输入序列长度为 $L$，kernel size 为 $k$（一维，与图像处理中的二维 kernel 不同），图 1 所示 conv layer 的变换如下，

    $$F(X)=(X\star W+b) \otimes \sigma(X\star V+c) \tag{1}$$

    两组 filters，第一组 $W \in \mathbb R^{k \times N \times M}, \ b \in \mathbb R^M$，第二组 $V \in \mathbb R^{k \times N \times M}, \ c \in \mathbb R^M$，输入 $X \in \mathbb R^{L \times N}$，$\sigma$ 为 sigmoid 函数，$\otimes$ 表示按位乘。输出 $F(X) \in \mathbb R^{L \times M}$，即需要对 $X$ 进行 padding（图 1 中未进行 padding）。

    没有 pooling 层，持续堆叠 conv layer 以获得长距信息。

3. 线性变换

    在 conv layers 最上层，增加一个 linear layer，将 conv  block 的最终输出（$\in \mathbb R^{L \times M}$）转变为非归一化分类得分 $E \in \mathbb R^{L \times C}$ ，其中 $C$ 为分类数量。


### 1.1.2 CRF
考虑使用 CRF 来预测 label 序列。

定义 label 序列 $y =(y_1,\ldots, y_L)$ 的得分为

$$s(S, y)=\sum_{i=1}^L E_{i,y_i}+\sum_{i=1}^{L-1} T_{y_i, y_{i+1}} \tag{2}$$

其中 $S=(c_1,\ldots, c_L)$ 表示字符序列（一个句子），$E$ 是前面卷积网络的输出得分，$T\in \mathbb R^{C\times C}$ 是状态转移矩阵（非归一化概率）。

label 序列的后验概率为

$$p(y|S)=\frac {\exp s(S,y)} {\sum_{y'} \exp s(S,y')} \tag{3}$$

> 增加一个 layer，输入为上一节 convnet 的输出得分 $E$，输出为后验概率

**损失**：负对数似然

$$\mathcal L(S,y)=-\log p(y|S) \tag{4}$$


**训练**阶段，采用反向传播，更新得到所有 layer 的参数，包括卷积参数和 CRF 状态转移矩阵。

**测试**阶段，求

$$\max_y p(y|S)=\max_y p(S,y)$$

卷积网络依然是执行一次，得到输出得分 $E$ 后，采用 Viterbi 算法：

迭代公式为 

$$\begin{aligned}\delta_t(j)&=\max_{i \in 1, \ldots, C} \Psi(j,i,x_t) \delta_{t-1} (i)
\\&=\max_{i \in 1, \ldots, C} p(j|i)p(x_t|j) \delta_{t-1}(i)
\\&=\max_{i \in 1, \ldots, C} \exp (T_{i,j}+E_{t,j})
\end{aligned}$$

同时还得到

$$y_{t-1}^{\star}(j)=\arg \max_{y_{t-1}} \delta_t(j)$$

初始条件为 $\delta_1(j)=\Psi(j,y_0,x_1)=p(x_t|j)=E_{1,j}$ （一共 $C$ 个初始条件）。

> 注：上面 (2) 与标准 CRF 不同，没有考虑 $y_0\rightarrow y_1$ 的状态转移概率，故初始条件中省去了因子 $T_{0,j}$

# 1.2 词嵌入

利用词嵌入 word embeddings，即设计一个基于词而非字符的模型。

基于词的模型可以利用 char-level 和 word-level 的信息。

设计如下词特征：

$$
\begin{array}{c|l}
\hline
Length       &     Features
\\
\hline
1    &   c_i\\\hline
2     &     c_{i-1}c_i & c_ic_{i+1}\\\hline
3       &  c_{i-2}c_{i-1}c_i & c_{i-1}c_ic_{i+1} & c_ic_{i+1}c_{i+2}\\\hline
4 & c_{i-3}c_{i-2}c_{i-1}c_i & c_{i-2}c_{i-1}c_ic_{i+1} \\
& c_{i-1}c_ic_{i+1}c_{i+2} & c_ic_{i+1}c_{i+2}c_{i+3}\\
\hline
\end{array}
$$

<center>表 1.</center> 


字符 $c_i$ 的最终 embedding 为 `11` 个 embedding 的 concatenation（`1` 个字符 embedding 和 `10` 个词 embedding）：

$$\begin{aligned}R(c_i)=&M_c[c_i] \oplus \\
&M_w[c_i] \oplus M_w[c_{i-1}c_i] \oplus \cdots \oplus \\
&M_w[c_ic_{i+1}c_{c+2}c_{i+3}]
\end{aligned} \tag{5}$$

由于中文词长度通常不超过 `4`，故上表中我们最大考虑 `4` 个字符的词特征。记字符集 size 为 $V$，那么特征空间 size 为 $O(V^4)$，显然这是非常耗费内存和计算资源的，一种缓解办法是：


整个过程如下：
1. 训练出一个 teacher 模型（例如前面基于字符的 CWS）
2. 使用 teacher CWS 对 unlabel 数据集 $\mathcal D_{un}$ 进行分词
3. 根据第 `2` 步的结果建立词集 $V_{word}$，其中低频(<5)词均用 UNK 进行替代
4. 使用 `word2vec` 工具训练词 embedding
5. 训练 student 分词模型



# 1.3 实验

bench-mark 数据集选用 PKU 和 MSR。unlabel 数据来自 Sogou 新闻。

**Dropout**

所有的 conv layer 和 embedding layer 均用上 dropout，dropout rate 固定为 `0.2`。（embedding layer 的输出就是 char embedding，对这个 embedding vector 进行 dropout，即向量中每个 element 按 `0.2` 的概率置 0）

**超参数**

两个数据集的实验中使用相同的超参数，如表 2 所示，

|超参数|value|
|--|--|
|char embedding 维度|200|
|word embedding 维度|50|
|conv layer 数量|5|
|conv layer 输出 channel|200|
|kernel size|3|
|dropout rate|0.2|

表 2

**预训练**

使用 word2vec 在 unlabel 数据上预训练得到 char embedding 和 word embedding，然后这些 embedding 在监督学习中被反向传播微调。具体地：

1. 基于字符的 CWS 模型

    将预训练 char embedding 作为 embedding layer 的 weight 初始值，对于新字符，使用一种合适的方法来初始化 embedding（例如高斯分布？）

2. 基于词的 CWS 模型

    将 char embedding 和 word embedding 根据 (5) 式组合，然后作为 embedding layer 的 weight 初始值，新字符的 embedding 使用一种合适的方法来初始化 embedding（例如高斯分布？）

**优化：** 使用 Adam 优化方法，batch size 为 100，训练不超过 100 个 epoch。对所有 conv layer 使用 [Weight 归一化](https://arxiv.org/abs/1602.07868) 以加速训练过程，Weight norm 也可以参考[这篇文章](/2021/03/08/dl/norm) 。

