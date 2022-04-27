---
title: lstm
date: 2021-11-12 16:16:11
tags: RNN
mathjax: true
img: /images/ml/lstm3.png
categories: 循环神经网络
---
Long short term memory
<!--more-->

# 1. RNN

RNN 适合处理时序输入，例如机器翻译、语言识别等。RNN 结构如下图，

![](/images/ml/lstm1.png)

<center>图 1. RNN结构</center>

但是 RNN 存在 long-term 依赖的问题，根据图 1

$$\begin{aligned}h^{(t+1)}&=Wh^{(t)} + U x^{(t+1)}
\\ &= W(W h^{(t-1)}+Ux^{(t)})+ U x^{(t+1)}
\\&=\cdots
\\ &=W^{t+1}h^{(0)}+W^tU x^{(1)}+\cdots + WU x^{(t)}+Ux^{(t+1)}
\end{aligned}$$

易知，从 $0$ 时刻开始，经过 $t$ 个 step 后，矩阵系数的幂最高达到 $W^t$，为了方便讨论，假设 $W$ 可以特征分解为 $W=V \text {diag}(\lambda) V^{-1}$，那么 

$$W^t = (V \text {diag}(\lambda) V^{-1})^t = V \text{diag}(\lambda)^t V^{-1}$$

（注：为了方便，本文中向量不采用粗体字符，但是根据上下文，不难理解它们表示的是向量）

如果特征值 $\lambda_i$ 不在 `1` 或者 `-1` 附近，那么 $\lambda_i^t$ 将会特别大或者特别小（只关心其绝对值，因为正负号只影响梯度方向）。如果特别大，则会导致“梯度爆炸”，训练过程极不稳定；如果特别小，导致“梯度消失”，那么无法确定参数的更新方向，从而无法正确的学习更新。

# 2. LSTM
全称：long short term memory networks

LSTM 是一种可以学习 long-term 依赖的 RNN 结构。

先给出标准 RNN 的结构图，激活函数用 `tanh` 为例，如图 2

![](/images/ml/lstm2.png)

<center>图 2. 标准 RNN 结构</center>

数学表示为：

$$a^{(t)} = b+W h^{(t-1)}+Ux^{(t)}$$

$$h^{(t)} = \tanh(a^{(t)})$$

$$o^{(t)} = c + V h^{(t)}$$
$$\hat y^{(t)} = \text{softmax} (o^{(t)})$$

图 2 中，模型输出未画出，但这没关系，不影响我们理解 lstm 。

LSTM 与标准 RNN 有大体相似的连接结构，但是方框中的部分不同，如图 3

![](/images/ml/lstm3.png)

<center>图 3 lstm 结构</center>

图 3 下方的 5 种操作，其中 `concatenate` 是将两个输入向量 合并起来 变成一个长的向量（而非相加），例如在图 2 中，隐层输出和输入两个向量（看作列向量）合并为 

$$x'=\begin{bmatrix}h^{(t-1)} \\ x^{(t)}\end{bmatrix}$$

那么对应的参数矩阵也需要合并为 $W'=[W, U]$，最终的操作结果则为

$$W'x'=[W, U]\begin{bmatrix}h^{(t-1)} \\ x^{(t)}\end{bmatrix}=W h^{(t-1)}+Ux^{(t)}$$


__总结各变量 shape：__

$x^{(t)} \in \mathbb R^{d}$，输入特征维度为 $d$； $U \in \mathbb R^{n \times d}$

$h^{(t)} \in \mathbb R^{n}$，隐层输出维度为 $n$； $W \in \mathbb R^{n \times n}$

$o^{(t)} \in \mathbb R^{c}$，输出层维度为 $c$； $V \in \mathbb R^{c \times n}$

## 2.1 核心思想

LSTM 的关键是引入了 cell 状态，即图 3 中的上面那条水平线，cell state 相当于一个传送带，在整个链中，从头到尾，中间仅受到一点点小影响：门（gate）对 cell 信息流的控制。

如图 4，

![](/images/ml/lstm4.png)
<center>图 4. 遗忘门（gate）结构</center>

这个门由 一个 sigmoid 激活函数以及一个按位相乘操作组成，sigmoid 函数位于 $(0,1)$ 之间，越接近 0 则表示不让 cell 中的信息通过，越接近 1 表示尽量让 cell 中信息通过。

## 2.2 摸清 LSTM 脉络

LSTM 中，链接起来的方框称为 `cell`，每个 `cell` 包含几个重要的变量：

1. 输入 $x$
2. state $C$（或者用 $s$ 表示）
3. 输出 $h$ （隐层输出）

控制信息流可以使用 leaky unit，即 `running_mean` 来实现，$\mu_t=(1-\alpha) \mu_{t-1} + \alpha x_t$，当 $\alpha$ 越大，表示遗忘很快，$\alpha$ 时，表示记忆很好，能记住较长时间之前的信息。但是 $\alpha$ 是固定的，不够灵活，在 LSTM 中采用 遗忘门 来控制信息流的传递，遗忘门的输出由 `cell` 状态，输入，和输出三者共同动态的决定。

### 2.2.1 遗忘门
如图 5 （a），

![](/images/ml/lstm5.png)
<center>图 5</center>

遗忘门的数学表示为

$$f_t=\sigma (W_f \cdot [h_{t-1}, x_t] + b_f)$$
其中 上一时刻 `cell` 的输出 $h_{t-1}$ 与本时刻的输入 $x_t$ 进行 concatenate，$W_f$ 和 $b_f$ 为遗忘门的权重和偏置参数，$\sigma$ 表示 sigmoid 激活。

### 2.2.2 输入门

前面遗忘门控制 cell state 中，哪些信息是需要保留的，哪些信息是需要遗忘的。现在我们需要确定，新信息如何加到 cell state。


如图 5(b) 所示，右侧的 tanh layer 用于生成 $\hat C_t$，它是由新信息生成，并将叠加到 cell state 上，也就是说，$\hat C_t$ 可以看作是 cell state 的更新量，但是这个更新量还不能直接加到 cell state 上，需要一个输入门控制这个更新量或者说是对更新量做一个 scale 操作，（输入门结构与遗忘门类似，只是参数不同）。数学表示为

$$i_t=\sigma(W_i \cdot [h_{t-1}, x_t]+b_i)$$

$$\hat C_t = \tanh (W_C \cdot [h_{t-1}, x_t] + b_C)$$

**总结：**

旧的 cell state $C_{t-1}$ 按位乘以遗忘门输出 $f_t$，以便遗忘掉过去的一些无用信息，然后再按位加上更新量 $i_t \star \hat C_t$，其中 $\star$ 表示按位乘。如图 5 (c) 所示，更新 cell state 的数学表示为

$$C_t = f_t \star C_{t-1} + i_t \star \hat C_t$$

### 2.2.3 输出门

现在来确定 cell 的输出。首先与 普通 RNN 一样，使用一个输出门，输出门的输入为 本时刻的输入 $x_t$ 和 上一时刻 cell 的输出 $h_{t-1}$，进行 concatenate 然后经过仿射变换后经 sigmoid 激活函数，就得到输出门的 output。

但是输出门的 output 还不能作为 cell 的输出。将 cell state 经 tanh 压缩到 `(-1,1)` 之间，然后按位乘以输出门的 output 得到 cell 的输出，显然，此举可看作是对输出进行 scale。

数学表示为

$$o_t=\sigma(W_o [h_{t-1},x_t]+b_o)$$
$$h_t=o_t \star \tanh (C_t)$$


## 2.3 LSTM 变体

上述 LSTM 结构是一个标准形式，实际上还有很多变体，在很多论文中都对标准 LSTM 进行了一些微小改变。

**第一个变体** 结构如下 图 6(a)，

![](/images/ml/lstm6.png)

<center>图 6. LSTM 变体</center>

图 6 (a) 中增加了一个 窥视孔 连接，即，将 cell state 连接到 gates 上，其中 输入门和遗忘门连接的是上一时刻的 cell state，而输出门连接的是本时刻的 cell state，于是

$$f_t = \sigma (W_f \cdot [C_{t-1}, h_{t-1}, x_t]+b_f)$$

$$i_i = \sigma(W_i \cdot [C_{t-1}, h_{t-1}, x_t] + b_i)$$

$$o_t = \sigma(W_o \cdot [C_t, h_{t-1}, x_t]+b_o)$$

**第二个变体** 结构如图 6(b) 所示，将遗忘门和输入门耦合起来，将 $1-f_t$ 作为输入门，而不是使用独立的输入门，这样做的思想是：$f_t$ 中用于记忆旧信息的部分，相应的 输入门 $1-f_t$ 不增加新输入信息，而 $f_t$ 中用于遗忘的旧信息的部分，输入门 $1-f_t$ 将会增加新信息进来到 cell state，数学表示为

$$C_t = f_t \star C_{t-1} + (1-f_t) \star \hat C_t$$

**第三个变体** 结构的改变更加明显，如图 6 (c) 所示，将 cell state 与 hidden state 合并，统一用 hidden state $h_t$ 表示。

数学表示为，

$$z_t = \sigma(W_z \cdot [h_{t-1},x_t])$$
$$r_t = \sigma(W_r \cdot [h_{t-1},x_t])$$
$$\hat h_t = \tanh(W \cdot [r_t \star h_{t-1}, x_t])$$
$$h_t = (1-z_t) \star h_{t-1} + z_t \star \hat h_t$$


## 2.4 Bi-LSTM
如图 7，

![](/images/ml/lstm7.png)

<center>图 7 双层 LSTM 结构。图源 ref 2</center>

第一层从左到右，得到表征向量，例如 $h_1^{(t)}$，第二层从右到左，得到表征向量 $h_2^{(t)}$，两层向量可以 concatenate 或者 element-wise sum 得到第三次向量 $h^{(t)}$，然后使用全连接层得到 $o^{(t)}$，然后使用 softmax 得到 $y^{(t)}$ 。



ref

1. [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

2. [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/pdf/1508.01991.pdf)