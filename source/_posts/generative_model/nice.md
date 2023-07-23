---
title: 'NICE:非线性独立成分估计'
date: 2022-08-06 15:14:28
tags: flow
mathjax: true
---

论文：[NICE: NON-LINEAR INDEPENDENT COMPONENTS ESTIMATION](https://arxiv.org/abs/1410.8516v6)

# 1. Introduction

如何捕获未知的复杂的数据分布？考虑一个变换 

$$h=f(x) \tag{1}$$

使得变换后的分布可按维度因式分解，即各维度相互独立

$$p_H(h)=\prod_d p_{H_d}(h_d) \tag{2}$$

根据概率分布的变换公式有

$$p_X(x)=p_H(f(x)) |\det \frac {\partial f(x)} {\partial x}| \tag{3}$$

其中 $\frac {\partial f(x)}{\partial x}$ 是 Jacobian 矩阵。如果我们能找到一个变换 $f(x)$，使得 Jacobian 矩阵的行列式容易计算出来，且逆变换 $f^{-1}$ 也容易得到，那么我们就可以按如下方式采样来自 $p_X(x)$ 中的样本：

$$h \sim p_H(h) \\\\ x=f^{-1}(h) \tag{4}$$

故关键点是：

1. 容易计算的 Jacobian 矩阵行列式
2. 容易获取的逆变换 $f^{-1}(h)$


## 1.1 解决思路

将数据 $x$ 分割为两部分 $(x_1, x_2)$，然后作如下变换：

$$\begin{aligned} y_1&=x_1 \\\\ y_2&=x_2+m(x_1) \end{aligned}\tag{5}$$

其中 $m$ 是由神经网络构成的一个复杂的变换函数。这个变换 $y=f(x)$ 的逆变换很容易求得，

$$\begin{aligned} x_1&=y_1 \\\\ x_2&=y_2-m(y_1) \end{aligned} \tag{6}$$

Jacobian 矩阵为 

$$\frac {\partial y}{\partial x}= \begin{bmatrix}I_1 & 0 \\ \frac {\partial y_2}{\partial x_1} & I_2 \end{bmatrix} \tag{7}$$

显然这是个下三角矩阵，且对角线全为 `1`，所以行列式为 `1`。

下面我们考虑具体问题。

假设数据分布 $p_{\theta}$，$\theta$ 是分布参数， 数据集 $\mathcal D$ 数量为 $N$，每个数据 $x \in \mathbb R^D$ 。

使用最大概率似然估计，对 (3) 式求对数，

$$\log p_X(x) = \log p_H(f(x)) + \log (|\det (\frac {\partial f(x)}{\partial x})|) \tag{8}$$

其中 $p_H(h)$ 是一个先验分布，如果分布可以因式分解（例如使用标准各向同性高斯分布），那么 (8) 式变为

$$\log p_X(x) = \sum_{d=1}^D \log p_{H_d}(f_d(x)) + \log (|\det (\frac {\partial f(x)}{\partial x})|) \tag{9}$$

如果类比 自编码器，那么我们可以将 $f$ 看作编码器，$f^{-1}$ 看作解码器。

# 2. 框架

如果模型框架使用 $L$ 个 layer，每个 layer 就是一个变换，那么 $f = f_L \circ \cdots f_2 \circ f_1$，根据 “矩阵乘积的行列式等于行列式的乘积” 这一定理，容易求得多 layer 组合后的 Jacobian 矩阵的行列式。

我们先考虑单个 layer 的变换。

## 2.1 Coupling Layer

**一般 Coupling Layer**

数据 $x \in \mathbb R^D$，$I_1, I_2$ 是对序列 $[1,2,\ldots, D]$ 的一个分割，其中 $I_1$ 长度为 $d=|I_1|$，函数 $m$ 是作用于 $\mathbb R^d$ 上的一个函数，定义

$$\begin{aligned} y_{I_1} &= x_{I_1} \\\\ y_{I_2} &= g(x_{I_2}; m(x_{I_1}) \end{aligned} \tag{10}$$

其中函数 $g$ 在给定第二个参数时，对第一个参数可逆。此时 Jacobian 矩阵为

$$\frac {\partial y}{\partial x}= \begin{bmatrix}I_d & 0 \\\\ \frac {\partial y_{I_2}}{\partial x_{I_1}} & \frac {\partial y_{I_2}}{\partial x_{I_2}} \end{bmatrix} $$

其中 $I_d$ 为 d 阶单位矩阵，于是 $\det \frac {\partial y}{\partial x}=\det \frac {\partial y_{I_2}} {\partial x_{I_2}}$，且逆变换为

$$\begin{aligned} x_{I_1} &= y_{I_1} \\\\ x_{I_2} &= g^{-1} (y_{I_2}; m(y_{I_1})) \end{aligned}\tag{11}$$

称上述变换为一个 coupling layer，其中 耦合函数为 $m$ 。

**加性 coupling layer**

$g(a;b)=a+b$

代入上式到 (11) 式得，

$$x_{I_2}=y_{I_2} - m(y_{I_1})\tag{11}$$

此时跟 `1.1` 小节相同，Jacobian 矩阵行列式为 `1` 。

当然也可以选择 乘性 coupling layer $g(a;b)=a \odot b, b \ne 0$ 或者仿射函数 $g(a;b)=a \odot b_1 +b_2, b_1\ne 0$ 。

**组合 coupling layers**

使用多个 coupling layer 进行组合以获得一个更加复杂的变换。

由于 $y_{I_1} = x_{I_1}$ ，单个 coupling layer 会使得 $x_{I_1}$ 保持不变，我们需要在相邻的两个 layers 中交换这两个部分，使得两个 coupling layers 的组合可以对每个维度进行修改，即

$$y_{I_1}^{(1)}=x_{I_1}, \quad y_{I_2}^{(1)}=x_{I_2}+m^{(1)}(x_{I_1})  \tag{12-1}$$

$$y_{I_1}^{(2)}=y_{I_1}^{(1)}+m^{(2)}(y_{I_2}^{(1)}), \quad y_{I_2}^{(2)}=y_{I_2}^{(1)} \tag{12-2}$$

上标指示 coupling layer 序号。

显然，上面这两个 layers 的变换将输入的两部分 （输入分为 d 维和 D-d 维两部分）进行了交换。我们看第二个 layer 的变换，其逆变换容易求得，再看其 Jacobian 矩阵为

$$\frac {\partial y^{(2)}}{\partial y^{(1)}}= \begin{bmatrix}I_d & \frac {\partial y_{I_2}}{\partial x_{I_2}} \\\\ 0 & I_{D-d}  \end{bmatrix} $$

易知，Jacobian 矩阵行列式依然为 `1`。


实际上可以对维度进行随机打乱，或者反转维度顺序，然后再重新分割为 $I_1$， $I_2$ 两部分，这一就可以一直用 (12-1) 式进行耦合。但是需要注意，此时需要记住打乱后的维度顺序，这样生成样本时才能恢复正确的维度顺序。

## 2.2 尺度伸缩

单个加性 coupling layer 的 Jacobian 矩阵行列式为 1，那么组合后的 Jacobian 矩阵行列式依然为 1，即 volume preserving，为了灵活性以及增大模型的泛化能力，在网络层顶端增加一个对角矩阵变换 $S$ 作为最后一层 layer，对每个输出乘以系数 $S_{ii}$，这使得我们可以对某些维度赋予更大的权重，而其他维度权重则较小。

基于 (9) 式进行改写，于是对数似然变为，

$$\log p_X(x) = \sum_{d=1}^D \log p_{H_d}(S_{dd} \cdot f_d(x))+ \log |S_{dd}| \tag{13}$$

这里 coupling layers 组合变换的 Jacobian 矩阵行列式 为 1， 所以取对数后为 0。最后一层尺度变换层 $h = S h^{(L)}$，其对应的 Jacobian 矩阵为

$$\frac {\partial h}{\partial h^{(L)}}=S$$

由于 $S$ 是对角矩阵，其行列式为对角线元素之和，所以取对数后，就是 (13) 式最后一项。

**一个例子**

先验分布使用各向同性的标准高斯分布（这里随机变量就是尺度变换后的输出），

$$p_H(h)=\frac 1 {(2\pi)^{D/2}} \exp (-\frac 1 2 h^{\top} h) \tag{14}$$

(14) 式代入 (13) 式，于是 

$$\log p_X(x) = -\frac D 2 \log 2\pi - \frac 1 2( S h^{(L)})^{\top}(S h^{(L)}) + \sum_{d=1}^D \log S_{dd} \tag{15}$$

于是，我们可以写出分布 $p_X(x)$ 

$$p_X(x)=\frac 1 {(2\pi)^{D/2} |S|^{-1}} \exp(-\frac 1 2  f(x)^{\top} S^{\top} S f(x)) \tag{16}$$

即协方差矩阵为 $\Sigma^{-1}=S^2$ 。

当 $S_{dd} \rightarrow +\infty$ 时，对应 `d-th` 维度的方差为 0，这表示数据 $x$ 在该维度的流形坍缩为一个点。

损失函数为 (15) 式。对于 mnist 这种简单的数据集，$m$ 函数使用非线性激活的 MLP。



实例代码：

https://github.com/bojone/flow/blob/master/nice.py

步骤总结：

1. 加载 mnist，batch shape 为 `(batch_size, 784)`
2. 输入数据上加噪声，$x=x+\epsilon, \epsilon \sim \mathcal U(-0.01,0)$，因为数据原本无法充满 784 维，增加噪声后可以有效防止过拟合。
3. 


|Layer|function|
|--|--|
|shuffle|数据维度反转，`x=x[::-1]`|
|split|`x1=x[:392], x2=x[392:]`|
|basic_model|前面的 `m` 函数：`fc1k-fc1k-fc1k-fc1k-fc1k-fc392`|
|couple|`x2=x2+m(x1)`|
|concat|`x=np.hstack((x1,x2))`|
表 1. 一个 block 的结构，其中 $ x,  x_1,  x_2$ 表示向量（单个样本），维度分别为 $794, 392, 392$。部分功能说明使用了 numpy 代码，而实际上应该替换为 Keras 或 Pytorch。`fc1k` 表示输出 channel 为 1000 的全连接层，同理，`fc392` 表示输出 channel 为 392 的全连接层。每个全连接层后接一个 relu 激活层。

|Layer| Output size|function|
|--|--|--|
|noising|(B,784)|$x=x+\epsilon, \epsilon \sim \mathcal U(-0.01,0)$|
|block 1| (B,784)|
|block 2| (B,784)|
|block 3| (B,784)|
|block 4| (B,784)|
|scale|(B,784)|$h=\exp{ w} \odot  h^{(L)}$|

表 2. 整个网络框架。其中最后一层是 scale，尺度变换层，其权重向量为 $\mathbf w$，由于尺度变换需要限制 $\mathbf w_i> 0$，所以取权重的对数，使得权重范围为整个实数域，即 $\mathbf w$ 表示权重的对数，于是变换公式为 $ h=\exp{ w} \odot  h^{(L)}$。

损失函数：根据 (15) 式，我们希望最大化这个对数概率，或者最小化负对数概率，忽略常数项，那么目标函数为

$$\min_{\theta} \  \frac 1 2( S h^{(L)})^{\top}(S h^{(L)}) - \sum_{d=1}^D \log S_{dd}$$

其中 $\theta$ 表示网络的所有参数。

**生成过程**

前面的 **一个例子** 中，使用各项同性的高斯分布作为尺度变换后的输出的概率分布，那么生成过程就是反过来，从这个各项同性的高斯分布 (14) 式 中采样 

$$ h \sim p_H(h)$$

将上面的过程看作是前向过程，这里的生成过程是反向过程，那么显然反向过程就是反过来操作：

1. 逆尺度变换 $h^{(L)}=h \odot \exp(-w)$
2. 逆 block4 -> 逆 block3 -> 逆 block2 -> 逆 block1


前向过程中开始的加噪操作，由于噪声水平非常小，所以反向过程中不进行相应的逆操作，即第 `2` 步输出就是样本图像，其像素值范围为 $[0,1]$ ，乘上 255 然后取整，若超出范围 $[0,255]$ 则对其进行 clip。