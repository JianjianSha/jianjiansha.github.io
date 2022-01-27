---
title: Connectionist Temporal Classification (1)
date: 2021-12-04 16:02:36
tags: deep learning
mathjax: true
---

隐马尔可夫模型中，可观测序列 $X$ 与隐状态序列 $Y$ 的长度是相等的，但是在语音识别中，语音序列被转为文字序列，这两者长度可能不相等，也就是说，语音片与文字无法一一对应。这就需要寻找其他解决方法。
<!--more-->

# 简介

我们将“对数据序列打标签”看作是 **时序分类**，通常可采用 RNN 网络实现，这称为 connectionist temporal classification（CTC），所适用的任务包括 语音识别，图片中的文字识别等。


记输入序列为 $X=[x_1,\ldots, x_T]$，输出序列为 $Y=[y_1,\ldots,y_U]$，（通常 $T > U$）。我们的任务是找到一个准确的 $X \rightarrow Y$ 的映射，但是可以想象这其中的难点：
1. $X, \ Y$ 长度可变
2. $X, \ Y$ 长度之比可变
3. $X, \ Y$ 序列元素之间没有明确关系。

## 损失函数
采用负对数似然函数。给定一个输入 $X$ 以及对应的输出 $Y$，我们的目标是求 $p(Y|X)$ 的最大化。

## 推断
给定输入 $X$，求最有可能的输出 $Y$，即

$$Y^{\star}=\arg \max_Y p(Y|X)$$

# 算法

## 对齐方式
通常，输入序列的一个或多个元素用于生成一个输出元素，是多对一的关系，所以输出序列长度小于输入序列长度。例如一个输入序列长度为 6，输出序列长度为 3，如下
```
x1 x2 x3 x4 x5 x6       input(X)
c  c  a  a  a  t        alignment
 c  |    a   | t        output(Y)
```
通过将连续相邻的重复的输出元素进行合并（去重），可以得到输出序列为 `cat`。

但是这种做法也存在以下两个问题：
1. 有时候输入元素不一定必须要对应一个输出元素。例如语音识别中，语音之中存在沉默片段，这部分不应对应任何一个文字（或单词）。
2. 将相邻的重复元素去重，会使得例如 “hello” 变成 “helo”，而无法得到想要的 “hello”。

为了解决这两个问题，CTC 引入了一个新的 token $\epsilon$ 表示空白（blank token），这个 $\epsilon$ 最终从输出序列中移除。于是，对于一段语音，可以通过如下方式进行识别，

|h|h|e|$\epsilon$|$\epsilon$|l|l|l|$\epsilon$|l|l|o| 操作|
|--|--|--|--|--|--|--|--|--|--|--|--|--|
|h||e|$\epsilon$||l|||$\epsilon$|l||o|合并相邻重复的元素|
|h||e|||l||||l||o|移除 $\epsilon$|

最后就得到了 `hello` 这个输出序列。

## 损失函数
我们采用 RNN 网络结构，记输入序列长度为 $T$，那么 RNN 一共有 $T$ 个输出，每个时刻 $t$ 的输出 $z_t$ 为一个长度为 $L+1$ 的得分向量，其中 $L$ 为输出字符集大小，$+1$ 是由于我们增加了一个 $\epsilon$。RNN 的输出序列记为 $A$，那么通过上述的合并相邻重复和去掉 $\epsilon$ 这个操作，可得到最终的输出序列 $Y$，记这个操作为 $B$，即 $Y=B(A)$，显然 $A$ 与 $Y$ 是多对一的关系，即 $A \in B^{-1}(Y)$，那么似然函数

$$p(Y|X)=\sum_{A \in B^{-1}(Y)} \prod_{t=1}^T p_t(a_t|X)$$

对于一个输出 $Y$，有很多个 $Z$，故如果想暴力计算，那么计算量会非常大，考虑动态规划方法。

空白 token  $\epsilon$ 可出现在输出序列 $Y$ 的任意位置，那么不妨我们将每两个 $y_i , \ i \in [U]$ 之间均插入 $\epsilon$，以及最起始位置和最后位置也插入  $\epsilon$，那么序列变为

$$Z=[\epsilon, y_1,\epsilon, y_2,\ldots, \epsilon, y_U, \epsilon]$$

仍以语音识别为例，如果语音开头有沉默，那么 RNN 输出序列 $A$ 以 $\epsilon$，否则以 $y_1$ 开头。具体而言，将语音等分为 $T$ 个语音片，即 $A=[a_1,a_2,\ldots,a_T]$，第 1 个语音片为沉默（无声音），那么对应 $\epsilon$，否则对应 $y_1$，第 2 个语音片则需要根据上一个语音片的情况进行讨论，如下表所示，


$$\begin{array}{c|c|c}
a_1 & a_1 & (a_1,a_2)
\\
\hline
\\
a_1=y_1 & \begin{aligned}a_2&=\epsilon \\ a_2&=y_1 \\ a_2&=y_2 (y_2\neq y_1)\end{aligned} & \begin{aligned} &(y_1, \epsilon) \\ &(y_1,y_1) \\ &(y_1,y_2)\end{aligned}
\\
\hline
\\
a_1=\epsilon & \begin{aligned}a_2&=\epsilon \\ a_2&=y_1\end{aligned} & \begin{aligned} &(\epsilon, \epsilon) \\ &(\epsilon,y_1)\end{aligned}
\end{array}$$

注意，$a_1=y_1, \ a_2=y_2$ 时，必须保证 $y_1\neq y_2$ 这个条件成立，否则 $B(a_1,a_2)=B(y_1,y_1)=(y_1)$，即，经过合并后，就少了一个 $y_1$。

我们讨论更一般的情况。

输出 $Y$ 的长度为 $U$，$Z$ 的长度为 $2U+1$，另外 $U \le T$，如果 ：

1. $2U+1 <= T$，这表示需要重复 $Z$ 中的部分元素，共重复 $T-(2U+1)$ 次。
2. $2U+1 > T$，这表示需要跳过 $Z$ 中的部分 $\epsilon$ ，共跳过 $2U+1-T$ 次。


为了使用动态规划方法，定义 $\alpha_{s,t}$ 表示到达 $t$ 时刻止，预测为 $Z_{1:s}$ 的概率，

$$\alpha_{s,t}=P\{C(A_{1:t}) = Z_{1:s}\}$$

其中 $C$ 表示合并连续重复的元素操作。

对于 $Z$ 中连续的三个元素 $z_{s-2}, z_{s-1},z_s$，有两种情况：

**1. 无法跳过 $z_{s-1}$**，即，必须有一个 RNN 输出 $a_j$ 对齐到 $z_{s-1}$。

第一个原因是 $z_{s-2}=z_s \ \neq \epsilon$，如果跳过预测 $z_{s-1}$，那么有一个 $y_i$ 值被合并，导致 $Y$ 长度减小了 1。 第二个原因是 $z_s=\epsilon$，由于 $\epsilon$ 与 $y_i$ 是间隔的，说明 $z_{s-1}=y_i$，此时若跳过预测 $z_{s-1}$，那么就少了一个 $y_i$，导致 $Y$ 长度减小了 1 。

如图1 所示，

![](images/dl/CTC1.png)

<center>图 1 （来源于 ref 1）</center>

也就是说 $\alpha_{s,t}$ 可由 $\alpha_{s,t-1}$ 转化而来，即 $a_t=a_{t-1}$，也可由 $\alpha_{s-1,t-1}$ 转化而来，即 $a_t \neq a_{t-1}$，只能是这两种之一。

转化公式为

$$\alpha_{s,t}=(\alpha_{s-1,t-1}+\alpha_{s,t-1}) \cdot p_t(z_s|X)$$

**2. 允许跳过 $z_{s-1}$**，允许跳过，即，可以跳过，也可以不跳过。
如图 2 所示，

![](images/dl/CTC2.png)

<center>图 2 （来源于 ref 1）</center>

可见，只有 $z_{s-1}=\epsilon$，且 $z_{s-2}\neq z_s$ 时才允许跳过预测 $z_{s-1}$。此时，$\alpha_{s,t}$ 可由 $\alpha_{s-2,t-1}$ 或 $\alpha_{s-1,t-1}$ 或者 $\alpha_{s,t-1}$ 转化而来。

$$\alpha_{s,t}=(\alpha_{s-2,t-1}+\alpha_{s-1,t-1}+\alpha_{s,t-1})\cdot p_t(z_s|X)$$

**综上**

对于连续的三个元素 $z_{s-2},z_{s-1},z_s$，只有以下三种可能的情形： 
```
z(s-2)    z(s-1)     z(s)
a         epsilon    a
epsilon   a          epsilon
a         epsilon    b
```

例如，一个输入序列长度为 6，输出序列 $Y=[a,b]$，那么所有可能的预测路径如下图，

![](images/dl/CTC3.png)

<center>图 3 （来源于 ref 1）</center>

从图3可知，只有两个有效的起始节点，以及两个有效的终止节点。

现在，就可以快速地计算似然函数了，损失函数采用负对数似然，对于一个训练集 $\mathcal D$，损失函数为 

$$\sum_{(X,Y) \in \mathcal D} -\log p(Y|X)$$

接下来就是计算梯度并训练模型了。
将 CTC 损失函数对每个时刻的输出概率求导。


推断一节在下一篇讨论。


# 参考
1. [Sequence Modeling With CTCT](https://distill.pub/2017/ctc/)