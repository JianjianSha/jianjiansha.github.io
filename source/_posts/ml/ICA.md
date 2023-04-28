---
title: 独立成分分析
date: 2022-07-14 15:32:57
tags: machine learning
mathjax: true
---

# 1. 问题描述

通常使用“鸡尾酒会问题”来描述 ICA。假设有两个人对话，如图 1 中的蓝色和红色的人，每个人旁边有一个麦克风，分别为紫色和粉色，那么每个麦克风接收到的语音均为蓝色和红色人语音的混合，即一共接收到两个混合音频文件，问题是如何从这两份个音频文件中区分出这两个语音？

![](/images/ml/ICA_1.png)
图 1，图源 [<sup>1</sup>](#refer-anchor-1)

ICA 用于将一组向量转变为最大可能的独立向量组。在上面的例子中，ICA 将讲个混合音频转换为两个非混合的表示两个说话者的语音，过程如图 2，

![](/images/ml/ICA_2.png)
图 2，图源 [<sup>1</sup>](#refer-anchor-1)

注意，输入输出的数量相等，由于输出各分量相互独立，故不像 PCA （主成分分析）那样丢弃部分次要成分。

# 2. 解决思路

先来严格定义 ICA，假设从 $n$ 个独立成分 $s_1,\ldots, s_n$ 中观测到 $n$ 个线性混合信号 $x_1,\ldots, x_n$，那么


$$x_i=\sum_{j=1}^n a_{ij} s_j, \ i=1,\ldots, n \tag{1}$$

用矩阵表示则是 

$$\mathbf x=A \mathbf s \tag{2}$$

称 $A$ 为混合矩阵。

通过将 $A$ 的列 $\mathbf a_i$ 除以某个标量，总是可以取消源 $s_i$ 的相同标量乘数，而 $A$ 和 $\mathbf s$ 均未知，所以不妨假设独立成分的方差均为 1，即 $\mathbb V[s_i]=1$ 。

我们也可以假设观测变量和独立成分的均值都为 0，如果不是的话，可以通过对观测变量中心化，使得模型为 零均值模型。

## 2.1 例子说明

假设两个信号源独立分量 $s_1,s_2$ 均服从以下分布

$$p(s_i)=\begin{cases} \frac 1 {2\sqrt 3} & |s_i| \le \sqrt 3 \\ 0 & o.w.\end{cases} \tag{3}$$

混合矩阵取 

$$A=\begin{bmatrix} 2 & 3 \\ 2 & 1\end{bmatrix} \tag{4}$$

得到观测信号 $x_1, x_2$，如图 3，左侧表示源信号，右侧表示观测信号，

![](/images/ml/ICA_3.png)
图 3，图源 [<sup>2</sup>](#refer-anchor-2)

观测数据是一个平行四边形上的均匀分布，注意，在这个平行四边形的左下和右上两个角点出，如果知道了 $x_1$，那么 $x_2$ 也就确定了，反之同样成立，故 $x_1, \ x_2$ 不再相互独立。但是对于左图则不一样，无论是知道了 $s_1$ 还是 $s_2$ 的值均无法知道另一个变量的值。

也就是说，混合矩阵使得原本独立的一组成分 $\mathbf s$  变成了不独立的一组成分 $\mathbf x$，所以，估计 ICA 模型变成了利用 $\mathbf x$ 中的信息来估计混合矩阵 $A$。

从图 3 中可以直观的知道：平行四边形的边正好是混合矩阵 $A$ 的列向量的方向。所以对于完全均匀分布的源信号成分而言，估计 $\mathbf x$ 的联合概率密度 $p(\mathbf x)$，然后定义边缘来估计 ICA 模型。

## 2.1 假设条件
ICA 中有两个假设：

1. 要恢复的隐式成分必须是统计独立

    统计独立指 $p(x,y)=p(x)p(y)$

2. 成分是非高斯型分布

    ICA 利用 “非高斯度” 来恢复原来的独立成分。非高斯度 度量了一个分布与高斯分布的差距。

第一点是 ICA 的出发点，如果成分都不独立，那就没必要使用 ICA。

**第二点假设**

如果 $s_i, i=1,\ldots,n$ 是高斯变量，

假定 $A$ 是正交矩阵，那么 $x_i, i=1,\ldots, n$ 也是高斯变量，且互相独立，事实上此时 $p(x_1,\ldots, x_n)$ 与 $p(s_1,\ldots, s_n)$ 完成相同，无法从 $\mathbf x$ 的分布中估计 ICA。

当然 $A$ 可以不是正交矩阵，但是需要是满秩矩阵，否则 $\mathbf x$ 线性相关，其中至少一个 $x_i$ 可以用其他分量 $\mathbf x_{-i}$ 线性表示。

## 2.2 ICA 估计原理

假设所有的独立成分有相同的分布。现在我们根据观测变量 $x_1,\ldots, x_n$ 来估计某个源变量，

$$y=\mathbf w^{\top} \mathbf x \tag{5}$$

其中 $\mathbf w$ 是待求的参数。根据 (2) 式 $\mathbf s= A^{-1} \mathbf x$，所以如果 $\mathbf w^{\top}$ 是 $A^{-1}$ 的某一行，那么 $y$ 就是一个独立成分。

定义 $\mathbf z = A^{\top} \mathbf w$，那么 (5) 式变为

$$y=\mathbf w^{\top} \mathbf x = \mathbf w^{\top} A \mathbf s = \mathbf z^{\top} \mathbf s \tag{6}$$

所以 $y$ 是 $s_1,\ldots, s_n$ 的线性组合。根据中心极限定理，独立随机变量的和比原始任何变量更接近高斯分布。所以 $\mathbf z^{\top} \mathbf s$ 比 $s_i$ 更接近高斯分布，当且仅当 $\mathbf z$ 只有一个元素非零时（例如仅 $z_i\ne 0$，此时 $y=z_i s_i$），$\mathbf z^{\top}\mathbf s$ 变成最小高斯（即最远离高斯），根据 (6) 式 $\mathbf w^{\top} \mathbf x=\mathbf z^{\top} \mathbf s$，故 $\mathbf w^{\top} \mathbf x$ 最远离高斯，或者称非高斯性最大，于是得到结论：最大化 $\mathbf w^{\top} \mathbf x$ 的非高斯性，求得 $\mathbf w$ ，从而得到一个独立成分。

## 2.3 非高斯性度量

需要有对非高斯性的度量，然后才能尝试最大化 $\mathbf w^{\top} \mathbf x$ 的非高斯性。

为了简化，假设 $y$ 的均值为 0，方差为 1。实际上，通过 ICA 算法的预处理，可以使这个假设成真。

### 2.3.1 峰度（Kurtosis）

$y$ 峰度的定义如下，

$$\text{kurt}(y)=\mathbb E[y^4] - 3 \mathbb E[y ^ 2] ^3 \tag{7}$$

根据假设，$y$ 方差为 1，那么 (7) 式右侧简化为 $\mathbb E[y^4]-3$。

如果 $y$ 是高斯变量，那么 $\text{kurt}(y)=3 \mathbb E[y ^ 2] ^2-3=0$ 。其他随机变量，峰度可正可负。

峰度具有线性性质：

1. $\text{kurt}(x_1+x_2)= \text{kurt}(x_1)+\text{kurt}(x_2)$
2. $\text{kurt}(ax_1)=a^4 \text{kurt}(x_1)$

**峰度的例子**

考虑二维模型 $\mathbf x = A \mathbf s$，假设独立成分的峰度 $\text{kurt}(s_1), \ \text{kurt}(s_2)$ 均不为 0，之前假设了随机变量方差为 1，那么根据 (6) 式 $y=z_1 s_1+z_2 s_2$ 和峰度的线性性质可知

$$\text{kurt} (y)=\text{kurt}(z_1 s_1)+ \text{kurt}(z_2 s_2)=z_1^4 \text{kurt}(s_1) + z_2^4 \text{kurt}(s_2)$$

另一方面，我们有办法使 $y$ 的方差为 1，前文假设了独立成分的均值为 $0$，那么有 

$$\begin{aligned}\mathbb E[y^2]&=\mathbb E[z_1^2 s_1^2 +z_2^2 s_2^2 + 2z_1 z_2 s_1 s_2]
\\\\ &=z_1^2 + z_2^2 + 2z_1 z_2 \mathbb E[s_1 s_2]
\\\\ &= z_1^2 + z_2^2=1
\end{aligned}$$

问题转为：

$$\max_{\mathbf z} \ z_1^4 \text{kurt}(s_1) + z_2^4 \text{kurt}(s_2) \quad s.t. \ ||\mathbf z||=1$$

不妨考虑峰度均为正的情况，$\text{kurt}(s_1), \ \text{kurt}(s_2) > 0$，那么当 

1. $\text{kurt}(s_1)\ge \ \text{kurt}(s_2)$ 时，$z_1=\pm 1, z_2=0$ 为最优解
2. $\text{kurt}(s_1) < \ \text{kurt}(s_2)$ 时，$z_1=0, z_2=\pm 1$ 为最优解

实际上，我们可以从权值向量 $\mathbf{w}$ 开始，基于混合向量 $\mathbf x$ 的样本 $x_1,\ldots,x_n$ 计算方向，即峰度增长最快的方向（如果峰度为正）或减少最快（如果峰度为负）的方向，并使用梯度方法或其中一个扩展来寻找新的矢量w。这个例子可以一般化到任意维度来表明峰度可以作为ICA问题的优化准则。

但是，峰度在实际应用中也存在一些缺陷，主要原因是峰度对异常值非常敏感。其值可能仅取决于分布尾部的少数观测值，这可能是错误的或不相关的观测。 换句话说，峰度是一种不稳定的非高斯性度量。接下来，介绍另一种比峰度优越的非高斯性度量方法：负熵。

### 2.3.2 负熵（Negentropy）

信息论中有个基本的结论：在所有方差相等的随机变量中，高斯变量的熵最大。这意味着熵可以作为非高斯性的一种度量，为了获得对于高斯变量为零并且总是非负的非高斯性度量，通常使用负熵，定义如下：

$$J(\mathbf y)=H(\mathbf y_{gauss})-H(\mathbf y) \tag{8}$$

$\mathbf y_{gauss}$ 是与 $\mathbf y$ 有相同协方差矩阵的高斯随机变量。负熵对于可逆线性变换是不变的。使用负熵作为非高斯性度量的优势是：统计理论能够很好的证明它。实际上，对统计性质而言，负熵在某种意义上是对非高斯性的最优估计。但是负熵的计算非常复杂，用定义估计负熵需要估计概率密度函数。因此我们可以寻找负熵的近似。

**负熵的近似**

像上面提到的，负熵的计算非常困难，因此对比函数仍然是理论函数。所以在实际应用中经常会使用一些负熵的近似，接下来介绍具有不错性质的负熵的近似。

传统的负熵近似是使用高阶矩，像下面这种形式：

$$J(y) \approx \frac 1 {12} \mathbb E[y ^3] ^2 + \frac 1 {48} kurt(y)^2 \tag{9}$$

假设y是零均值方差为1的随机变量。然而这种近似的有效性可能是有限的，这些近似受到了峰度的非鲁棒性的影响。为了避免这种问题，一种新的负熵的近似被提出，这种近似是基于最大熵原理的，其近似如下：

$$J(y) \approx \sum_{i=1}^p k_i \{\mathbb E[G_i(y)] - \mathbb E[G_i(v)]\}^2 \tag{10}$$

(没有仔细读完，后续内容可参考 [[2]](#refer-anchor-2) )

# REF


<div id="refer-anchor-1"></div>

- [1] [Independent Component Analysis (ICA)](https://towardsdatascience.com/independent-component-analysis-ica-a3eba0ccec35)

<div id='refer-anchor-2'></div>

- [2] [详解独立成分分析](https://blog.csdn.net/Tonywu2018/article/details/91490158)