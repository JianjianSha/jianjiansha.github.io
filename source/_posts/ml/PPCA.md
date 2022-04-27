---
title: PCA的概率模型
date: 2022-02-16 18:31:17
tags: machine learning
---

本文从概率角度理解 PCA。
<!--more-->

# 1. 生成过程和概率模型

假设有隐变量 $\mathbf z \in \mathbb R^M$，其先验分布为标准正态分布 $p(\mathbf z)=\mathcal N(\mathbf 0, I)$。可观察变量 $\mathbf x$ 与隐变量 $\mathbf z$ 满足一个线性关系

$$\mathbf x=B\mathbf z+\boldsymbol \mu + \boldsymbol \epsilon \in \mathbb R^D \tag{1}$$

其中 $\boldsymbol \epsilon \sim \mathcal N(\mathbf 0, \sigma^2I)$ 是一个高斯噪声。转换矩阵 $B \in \mathbb R^{D \times M}$，$\boldsymbol \mu \in \mathbb R^D$ 是一个偏移（实现仿射变换），$\mathbf x$ 的概率分布为

$$p(\mathbf x|\mathbf z, B, \boldsymbol \mu, \sigma^2)=\mathcal N(\mathbf x|B\mathbf z+\boldsymbol \mu, \sigma^2I) \tag{2}$$

数据生成过程/步骤：
1. 随机生成 $\mathbf z_n$
2. 根据 $\mathbf z_n$ 随机生成数据 $\mathbf x_n \sim p(\mathbf x|\mathbf z_n, B, \boldsymbol \mu, \sigma^2)$

于是有

$$p(\mathbf x,\mathbf z|B, \boldsymbol \mu, \sigma^2)=p(\mathbf x|\mathbf z, B, \boldsymbol \mu, \sigma^2) p(\mathbf z) \tag{3}$$

$\mathbf z$ 就是数据 $\mathbf x$ 在低维空间的表示，根据数据集我们可以学习得到 $\mathbf z$ 的分布，每次随机生成 $\mathbf z$ 后，使用 (1) 式将 $\mathbf z$ 恢复到原空间中。

## 1.1 似然和联合分布
数据的似然概率为 

$$p(\mathbf x|B,\boldsymbol \mu, \sigma^2)=\int p(\mathbf x|\mathbf z, B, \boldsymbol \mu, \sigma^2) p(\mathbf z) d \mathbf z
\\=\int \mathcal N(\mathbf x|B\mathbf z+\boldsymbol \mu, \sigma^2I)\mathcal N(\mathbf z|\mathbf 0,I)d\mathbf z \tag{4}$$

根据期望的线性性质得数据的期望

$$\mathbb E_{\mathbf x}[\mathbf x]=\mathbb E_{\mathbf z}[B\mathbf z+\boldsymbol \mu] + \mathbb E_{\boldsymbol \epsilon}[\boldsymbol \epsilon]=\boldsymbol \mu$$

协方差矩阵为

$$\mathbb V[\mathbf x]=\mathbb V_{\mathbf z}[B\mathbf z+\boldsymbol \mu] + \mathbb V_{\boldsymbol \epsilon}[\boldsymbol \epsilon]=\mathbb V_{\mathbf z}[B\mathbf z] + \sigma^2I
\\=B \mathbb V_{\mathbf z}[\mathbf z] B^{\top} + \sigma^2I=BB^{\top}+\sigma^2I$$

上式推导中，第一个等式成立是因为 $\mathbf z$ 与 $\boldsymbol \epsilon$ 独立，第二个等式成立是因为 $\boldsymbol \mu$ 是常量。


根据概率论相关知识，高斯随机变量 $\mathbf z$ 的线性变换仍然是高斯分布，而 $\mathbf x$ 的期望和方差由上面两式给出，故 $\mathbf x$ 的分布为

$$p(\mathbf x|B,\boldsymbol \mu, \sigma^2)=\mathcal N(\mathbf x|\boldsymbol \mu, BB^{\top}+\sigma^2I) \tag{5}$$

而 $\mathbf x$ 与 $\mathbf z$ 之间的交叉协方差为

$$\begin{aligned}\text{Cov}[\mathbf x,\mathbf z]&=\text{Cov}[B\mathbf z+\boldsymbol \mu + \boldsymbol \epsilon, \mathbf z]
\\&=\text{Cov}[B\mathbf z, \mathbf z] + \text{Cov}[\boldsymbol \mu, \mathbf z]+ \text{Cov}[\boldsymbol \epsilon, \mathbf z]
\\&=B \text{Cov}[\mathbf z,\mathbf z]
\\&=B
\end{aligned}$$

$\mathbf x$ 和 $\mathbf z$ 的联合概率也是高斯型，故联合概率为

$$p(\mathbf x,\mathbf z|B,\boldsymbol \mu, \sigma^2)=\mathcal N \left(\begin{bmatrix}\mathbf x \\ \mathbf z\end{bmatrix}| \begin{bmatrix}\boldsymbol \mu \\ \mathbf 0\end{bmatrix}, \begin{bmatrix}BB^{\top}+\sigma^2I & B \\ B^{\top} & I\end{bmatrix}\right) \tag{6}$$

利用 (6) 式的联合概率分布，可以计算 $\mathbf z$ 后验分布，显然这也是一个高斯型分布，

$$p(\mathbf z|\mathbf x)=\mathcal N(\mathbf z|\mathbf m, C)$$

根据高斯型联合概率分布的条件分布（参考这里关于 [高斯分布]() 的介绍），可知

$$\mathbf m=B^{\top}(BB^{\top}+\sigma^2I)^{-1}(\mathbf x-\boldsymbol \mu) \tag{7}$$

$$C=I-B^{\top}(BB^{\top}+\sigma^2I)^{-1}B \tag{8}$$

得到隐变量 $\mathbf z$ 的后验概率分布之后：
1. 对于一个新的数据 $\mathbf x^{\star}$，计算 $p(\mathbf z^{\star}|\mathbf x^{\star})$ 的概率分布
2. 根据这个后验分布随机生成 $\mathbf z^{\star}$
3. 重建数据 $\tilde {\mathbf x}^{\star}~p(\mathbf x|\mathbf z^{\star},B,\boldsymbol \mu, \sigma^2)$，其中 $B,\boldsymbol \mu,\sigma^2$ 为模型参数，在对数据集做 MLE 或 MAP 时学习得到

## 1.2 MLE

使用 MLE 或 MAP 进行参数估计，似然函数使用 (5) 式。负对数似然为

$$\mathcal L=-\log P(X)=-\sum_{n=1}^N \log p(\mathbf x_n)\propto \sum_{n=1}^N (\mathbf x_n-\mathbb E[\mathbf x])^{\top}\Sigma^{-1}(\mathbf x_n-\mathbb E[\mathbf x])$$

其中 $X$ 表示整个数据集。$\mathbb E[\mathbf x]$ 表示随机变量 $\mathbf x$ 的真实期望，$\Sigma$ 是其真实协方差矩阵，根据 (5) 式得

$$\mathbb E[\mathbf x]=\boldsymbol \mu
\\ \Sigma=BB^{\top}+\sigma^2I$$

$\mathcal L$ 对 $\boldsymbol \mu$ 求梯度，

$$\nabla {\mathcal L}_{\boldsymbol \mu}=\sum_{n=1}^N (\mathbf x_n-\boldsymbol \mu)\Sigma^{-1}=\mathbf 0$$

解得，

$$\boldsymbol \mu_{ML}=\frac 1 N \sum_{n=1}\mathbf x_n \tag{9}$$

同理对 $B$ 和 $\sigma^2$ 求梯度并等于零，解得

$$B_{ML}=T(\Lambda-\sigma^2I)^{1/2}R \tag{10}$$

$$\sigma^2=\frac 1 {D-M} \sum_{j=M+1}^D \lambda_j \tag{11}$$

其中 $T \in \mathbb R^{D\times M}$ 包含了数据协方差矩阵的前 $M$ 个特征向量（特征向量是 $D$ 维），$\Lambda=\text{diag}(\lambda_1,\ldots, \lambda_M)$ 是数据协方差矩阵的前 $M$ 个特征值（如果几何重数 $>1$ 则可重复），$R \in \mathbb R^{M\times M}$ 是任意正交矩阵。

## 1.3 MLE 结果分析

### 1.3.1 期望

随机变量 $\mathbf x$ 的概率分布的期望 $\boldsymbol \mu$ 就是样本均值


### 1.3.2 协方差

根据 (9) 式，噪声方差 $\sigma^2$ 为数据映射到 “ $M$ 阶主子空间的正交补空间 ” 后方差的平均，即数据协方差矩阵对应到补空间的特征值之和。

对 $\sigma^2=0$ 的情况进行验证。

根据 (7) 式进行数据降维

$$\mathbf z \approx B^{\top}(BB^{\top}+\sigma^2I)^{-1}(\mathbf x-\boldsymbol \mu)=B^{\top}(BB^{\top})^{-1}(\mathbf x-\boldsymbol \mu)$$

根据 上式以及 (2) 式，数据恢复为

$$\tilde {\mathbf x} \approx B\mathbf z+ \boldsymbol \mu \approx \mathbf x$$

也就是说，对于原始数据 $\mathbf x$:

1. 使用 PCA （数据协方差矩阵特征分解）恢复的数据记为 $\tilde {\mathbf x}'$
2. 如果不引入噪声，通过两步随机生成（先生成 $\mathbf z^{\star}$，然后生成 $\tilde {\mathbf x}$）， 那么 $p(\tilde {\mathbf x}=\tilde {\mathbf x}')$ 的概率密度最大。