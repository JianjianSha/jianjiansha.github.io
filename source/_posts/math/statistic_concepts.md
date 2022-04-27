---
title: 概率统计基本概念
date: 2022-02-12 13:54:51
tags: 
    - math
    - machine learning
mathjax: true
---
总结概率统计的一些知识点。
<!--more-->
# 1. 统计

## 1.1 协方差

两个随机变量 $X,Y \in \mathbb R$ 的协方差定义为

$$\begin{aligned}\text {Cov}[X,Y]&:=\mathbb E[(X-\mathbb E[X])(Y-\mathbb E[Y])]
\\&=\mathbb E[XY-X\mathbb E[Y]-\mathbb E[X]Y-\mathbb E[X]\mathbb E[Y]]
\\&=\mathbb E[XY]-\mathbb E[X]\mathbb E[Y]
\end{aligned}$$

给定联合概率分布 $P(X,Y)$，协方差 $\text {Cov}[X,Y]$ 可以计算出来，是一个标量。特别地，$\mathbb V[X]:=\text {Cov}[X,X]=\mathbb E[X^2]-\mathbb E^2[X]$ 称作 $X$ 的方差。

**多维随机变量的协方差**

对于随机向量 $X \in \mathbb R^D$ 和 $Y \in \mathbb R^E$，协方差矩阵为

$$\text {Cov}[X,Y]=\mathbb E[XY^{\top}]-\mathbb E[X] \mathbb E[Y]^{\top} = \text {Cov}[Y, X]^{\top} \in \mathbb R^{D \times E}$$

**协方差矩阵**

注意，协方差矩阵指的是多维随机变量 $X \in \mathbb R^D$ 的方差，
$$\mathbb V[X]=\mathbb E[XX^{\top}]-\mathbb E[X] \mathbb E[X]^{\top}$$

协方差矩阵是对阵半正定矩阵。

**一般机器学习应用中，我们会假设协方差矩阵就是对称正定的，这样会更方便我们处理问题。事实上，协方差矩阵是对称的，且对角线元素值为正（一维随机变量的方差为正），而实际问题中的协方差矩阵也是满秩的（没有哪个列是其他列的线性组合），所以这种假设是合理的。**

性质：（$X,Y \in \mathbb R^D$）
1. $\mathbb E[X \pm Y]=\mathbb E[X] \pm \mathbb E[Y]$
2. $\mathbb V[X\pm Y]=\mathbb V[X]+\mathbb V[Y] \pm \text{Cov}[X,Y]\pm \text{Cov}[Y,X]$
3. $X,Y$ 不相关 $\Leftrightarrow \text{Cov}[X,Y]=\mathbf 0$
4. $X,Y$ 独立 $\Rightarrow \text{Cov}[X,Y]=\mathbf 0$，反之不一定
5. $$\text{Cov}[X+Y,Z]=E[(X+Y)Z^{\top}]-E[X+Y]E[Z]^{\top}
\\=E[XZ^{\top}]+E[YZ^{\top}]-E[X]E[Z]^{\top}-E[Y]E[Z]^{\top}
\\=\text{Cov}[X,Z]+\text{Cov}[Y,Z]$$
6. $\text{Cov}[A, X]=E[AX^{\top}]-E[A]E[X]^{\top}=AE[X]^{\top}-AE[X]^{\top}=\mathbf 0_{D\times E}$

    其中 $A \in \mathbb R^D$，为常向量，$X \in \mathbb R^E$ 为随机向量。

**相关系数**

两个一维随机变量的相关稀疏 (Correlation) 为
$$\text {corr}[X,Y]=\frac {\text {Cov}[X,Y]} {\sqrt {\mathbb V[X] \mathbb V[Y]}} \in [-1, 1]$$

相关系数矩阵：标准化随机变量（多维）$X/\mathbb V^{1/2}[X]$ 的协方差矩阵。这里 $\mathbb V^{1/2}[X]$ 是 $X$ 的标准差向量，$X/\mathbb V^{1/2}[X]$ 是 element-wise 相除。


## 1.2 经验期望和协方差

前面的期望和协方差均是针对真实分布的统计量，然而机器学习中，我们的数据集大小有限，所以无法得到真实的统计量，而是经验统计量。

经验期望：
$$\overline {\mathbf x}:=\frac 1 N \sum_{n=1}^N \mathbf x_n$$

其中，$N$ 表示数据集大小，$\mathbf x_n \in \mathbb R^D$。

经验协方差矩阵：
$$\Sigma:=\frac 1 N \sum_{n=1}^N (\mathbf x_n - \overline {\mathbf x})(\mathbf x_n - \overline {\mathbf x})^{\top}$$

## 1.3 高斯分布

**高斯混合**

两个一维高斯分布的混合，
$$p(x)=\alpha p_1(x) + (1-\alpha) p_2(x), \quad \alpha \in [0, 1]$$

其中，$X_i \sim (\mu_i, \sigma_i^2),\ i =1,2$， 那么，根据期望的线性映射的性质有

$$\mathbb E[X]=\alpha \mu_1 + (1-\alpha)\mu_2$$
另外，
$$\begin{aligned} \mathbb E[X^2]&=\int x^2 p(x)dx
\\&=\int \alpha x^2 p_1(x)+(1-\alpha)x^2 p_2(x) dx
\\&=\alpha \mathbb E_1[X^2] + (1-\alpha) \mathbb E_2[X^2]
\\&=\alpha(\mu_1^2+\sigma_1^2) + (1-\alpha)(\mu_2^2+\sigma_2^2)
\end{aligned}$$
于是方差为

$$\mathbb V[X]=\mathbb E[X^2]-\mathbb E^2[X]=\alpha(\mu_1^2+\sigma_1^2) + (1-\alpha)(\mu_2^2+\sigma_2^2)-[\alpha \mu_1 + (1-\alpha)\mu_2]^2$$

**多维高斯分布**

服从标准高斯分布的多维随机变量，经过线性映射，可以得到非标准高斯分布。例如 $X \in \mathbb R^D$ 满足 $X \sim \mathcal N(\mathbf 0, I)$，那么 $\mathbf y=A\mathbf x + \boldsymbol \mu \in \mathbb R^D$ 服从 $\mathcal N(\boldsymbol \mu, \Sigma)$ 分布，其中 $\Sigma=AA^{\top}$。故要得到一个 $\mathcal N(\boldsymbol \mu, \Sigma)$ 分布，将协方差矩阵 $\Sigma$ 做 CholesKy 分解，得到矩阵 $A$，然后做线性变换 $\mathbf y=A\mathbf x + \boldsymbol \mu$ 即可。

**边际和条件型高斯分布**

考虑两个多维随机变量 $\mathbf x$ 和 $\mathbf y$，其维度可能不等，联合概率分布为

$$p(\mathbf x, \mathbf y)=\mathcal N(\begin{bmatrix} \boldsymbol \mu_x \\ \boldsymbol \mu_y\end{bmatrix}, \begin{bmatrix} \Sigma_{xx} & \Sigma_{xy} \\ \Sigma_{yx} & \Sigma_{yy}\end{bmatrix})$$

那么条件概率分布也是高斯型，

$$p(\mathbf x| \mathbf y)=\mathcal N(\boldsymbol \mu_{x|y}, \Sigma_{x|y})$$
其中

$$\boldsymbol \mu_{x|y}=\boldsymbol \mu_x+\Sigma_{xy}\Sigma_{yy}^{-1}(\mathbf y-\boldsymbol \mu_y)$$
$$\Sigma_{x|y}=\Sigma_{xx}-\Sigma_{xy}\Sigma_{yy}^{-1}\Sigma_{yx}$$

边际分布也是高斯型，

$$p(\mathbf x)=\int p(\mathbf x,\mathbf y) d\mathbf y=\mathcal N(\mathbf x|\boldsymbol \mu_x, \Sigma_{xx})$$

# 2. 指数分布

**Bernoulli 分布**

$$p(x|\mu)=\mu^x(1-\mu)^{1-x}$$
$$\mathbb E[X]=\mu$$
$$\mathbb V[X]=\mu(1-\mu)$$

**二项分布**

$$p(X=m|N,\mu)=\begin{pmatrix}N \\ m\end{pmatrix}\mu^m (1-\mu)^{N-m}$$
$$\mathbb E[X]=N \mu$$
$$\mathbb V[X]=N\mu(1-\mu)$$

二项分布可看作 $N$ 个独立的 Bernoulli 分布。


**Beta 分布**

Beta 分布中的随机变量范围为 $\mu \in [0, 1]$，

$$\text{Beta}(\alpha, \beta) = p(\mu|\alpha, \beta)=\frac {\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} \mu^{\alpha-1} (1-\mu)^{\beta-1}$$
$$\mathbb E[\mu]=\frac {\alpha}{\alpha+\beta}$$
$$\mathbb V[\mu]=\frac {\alpha \beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$$
其中，$\alpha, \beta > 0$ 。Gamma 函数（连续情况的阶乘）满足

$$\Gamma(t)=\int_0^{\infty} x^{t-1} \exp (-x) dx, \ t > 0$$
$$\Gamma(t+1)=t\Gamma(t)$$

由于 $\mu \in [0, 1]$，我们可以将 $\mu$ 看作是某个 Bernoulli/Binomial 分布的参数，即 Bernoulli 分布的参数 $\mu$ 不再是一个确定不变的量，而是一个随机变量。

## 2.1 共轭

贝叶斯定理：
$$p(x|y)=\frac {p(x)p(y|x)} {p(y)}$$

其中 $x$ 可以是某隐变量（不可观察，例如 HMM 中的状态），而 $y$ 是可以直接观察的量。那么在给定观察 $y$ 的条件下，隐变量的后验分布 $p(x|y) \propto p(x) p(y|x)$，即正比于先验与似然的乘积。

如果后验分布与先验分布类型相同，那么称先验共轭于似然函数，即**如果 $p(x)$ 与 $p(y|x)$ 共轭** ，
那么 $p(x|y)$ 与 $p(x)$ 分布类型相同。


**Beta-Binomial共轭**

考虑一个二项分布，其参数 $\mu$ 未知，看作一个随机变量，对于 $\mu$，我们已知其先验，遵从 Beta 分布，

$$p(\mu|\alpha, \beta)=\frac {\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} \mu^{\alpha-1} (1-\mu)^{\beta-1}$$

现在执行 $N$ 次抛硬币试验，正面向上的次数为可观测量 $y$，且观测到 $y=h$，那么 $\mu$ 的后验概率为，

$$\begin{aligned}p(\mu|y=h,N,\alpha,\beta) & \propto p(\mu|\alpha,\beta) p(y=h|\mu)
\\& \propto \mu^h(1-\mu)^{N-h}\mu^{\alpha-1} (1-\mu)^{\beta-1}
\\ &= \mu^{h+\alpha-1} (1-\mu)^{N-h+\beta-1}
\\& \propto \text{Beta}(h+\alpha, N-h+\beta)
\end{aligned}$$
即，后验与先验分布类型相同，均为 Beta 分布。


|似然|共轭先验|后验|
|--|--|--|
|Bernoulli|Beta|Beta|
|Binomial|Beta|Beta|
|Gaussian|Gaussian/inverse Gamma|Gaussian/inverse Gamma|
|Gaussian|Gaussian/inverse Wishart|Gaussian/inverse Wishart|
|Multinomial|Dirichlet|Dirichlet|


## 2.2 充分统计量

包含了足够的可以表达分布信息的统计量。

## 2.3 指数家族

满足如下形式的分布

$$p(\mathbf x|\boldsymbol \theta)=h(\mathbf x) \exp(\boldsymbol \theta^{\top} \boldsymbol \phi(\mathbf x)-A(\boldsymbol \theta))$$

