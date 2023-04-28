---
title: 非归一化统计模型的Score Matching估计
date: 2022-07-09 10:09:27
tags: 
    - scored based model
    - generative model
mathjax: true
---


论文：[Estimation of Non-Normalized Statistical Models by Score Matching](https://www.cs.helsinki.fi/u/ahyvarin/papers/JMLR05.pdf)

# 1. 问题提出
很多时候，我们只能给出模型的非归一化概率密度，也就是说，归一化常量未知且难以计算。

随机变量 $\mathbf x \in \mathbb R^n$ 的概率密度记为 $p_{\mathbf x}(\cdot)$，我们使用一个模型来近似数据真实的概率密度 $p_{\mathbf x}(\cdot)$，模型的概率密度 $p(\cdot; \theta)$，其中 $\theta$ 为参数，于是，任务就变成了根据观测数据 $\mathbf x$ 来估计参数 $\theta$。

我们考虑的是具有如下形式的概率密度函数，

$$p(\xi; \theta)=\frac 1 {Z(\theta)} q(\xi; \theta) \tag{1}$$

我们知道 $q$ 的解析表达式，但是归一化常数 $Z$ 却难以计算出解析式，

$$Z(\theta)=\int _{\xi \in \mathbb R^n} q(\xi; \theta) d\xi \tag{2}$$


$\xi \in \mathbb R^n$ 是与 $\mathbf x$ 一样，表示随机变量，$\mathbf x$ 根据上下文 $\mathbf x$ 有时还用来指真实数据（对应的概率），以便与模型概率区分开来。

通常使用马尔可夫蒙特卡洛 MCMC 来估计非归一化模型，但是这种方法非常 slow 且使用了近似导致效果也 poor。在马尔可夫随机场中常常碰到非归一化模型。

本文作者团队提出一种简单的非归一化模型估计方法：最小化数据 $\mathbf x$ 的得分函数与模型的得分函数之间的平方差的期望值。这里得分函数指 $\nabla_{\mathbf x} \log p(\mathbf x)$

# 2. Score Matching

如何估计非归一化模型，作者提出：最小化 真实数据的得分函数与模型的得分函数的平方差的期望，

$$\min_{\theta} \mathbb E_{\mathbf x} [||\psi(\xi; \theta)-\psi_{\mathbf x}(\xi)||^2] \tag{3}$$

其中 **得分函数** 指对数概率的梯度，

$$\psi(\xi;\theta)=\begin{pmatrix}\frac {\partial \log p(\xi;\theta)}{\partial \xi_1} \\ \vdots \\ \frac {\partial \log p(\xi;\theta)}{\partial \xi_n} \end{pmatrix}=\begin{pmatrix} \psi_1(\xi;\theta) \\ \vdots \\\psi_n(\xi;\theta) \end{pmatrix}=\nabla_{\xi} \log p(\xi;\theta) \tag{4}$$

由于 $Z(\theta)$ 是与 $\xi$ 无关的常数，所以

$$\psi(\xi;\theta)=\nabla_{\xi} \log q(\xi;\theta) \tag{5}$$

将损失函数展开为，

$$J(\theta)=\frac 1 2 \int_{\xi \in \mathbb R^n} p_{\mathbf x}(\xi) \|\psi(\xi;\theta)-\psi_{\mathbf x}(\xi)\|^2 d\xi \tag{6}$$

增加了一个 $\frac 1 2$ 因子，是为了求导计算方便。于是参数估计为

$$\hat \theta=\arg \min_{\theta} J(\theta)$$

小结：以前是令模型概率密度逼近真实概率密度，现在是令模型概率密度的导数逼近真实概率密度的导数，也就是这里所说的得分函数的逼近。

不过 (6) 式看着还是要根据观察数据计算数据得分函数 $\psi_{\mathbf x}$，实际上不需要，如下定理所述。

## 2.1 定理 1

假设模型得分函数 $\psi(\xi;\theta) 和数据概率密度分布 p_{\mathbf x}(\xi)$可微，期望 $\mathbb E_{\mathbf x}[\|\psi(\mathbf x;\theta)\|^2]$ 和 $\mathbb E_{\mathbf x} [\|\psi_{\mathbf x}(\mathbf x)\|^2]$ 对任意 $\theta$ 均有限，且对任意 $\theta$ 在 $\|\xi\| \rightarrow \infty$ 时有  $p_{\mathbf x}(\xi) \psi(\xi;\theta) \rightarrow 0$，那么目标函数 (6) 式变为 

$$J(\theta)=\int_{\xi \in \mathbb R^n} p_{\mathbf x}(\xi) \sum_{i=1}^n \left [\partial_i \psi_i(\xi;\theta) + \frac 1 2 \psi_i(\xi;\theta)^2\right] d\xi + const \tag{7}$$

其中 $const$ 是与 $\theta$ 无关的常数，且

$$\psi_i(\xi;\theta)=\frac {\partial \log q(\xi;\theta)}{\partial \xi_i}$$

$$\partial_i \psi_i(\xi;\theta)=\frac {\partial \psi_i(\xi;\theta)}{\partial \xi_i}=\frac {\partial ^ 2 \log q(\xi;\theta)}{\partial \xi_i ^ 2}$$




<details>
<summary>证明过程</summary>

根据 (6) 式，

$$J(\theta)=\int p_{\mathbf x}(\xi)\left[\frac 1 2 \|\psi(\xi;\theta)\|^2+\frac 1 2 \|\psi_{\mathbf x}(\xi)\|^2-\psi_{\mathbf x}(\xi)^{\top}\psi(\xi;\theta)\right] d\xi$$

上式第二项与 $\theta$ 无关，看作常数项。第一项

$$\|\psi(\xi;\theta)\|^2=\sum_{i=1}^n \psi_i(\xi;\theta)$$

这正是 (7) 式中方括号中的第二项。

重点看第三项，

$$\begin{aligned}-\int p_{\mathbf x}(\xi) \psi_{\mathbf x}(\xi)^{\top}\psi(\xi;\theta)d\xi&=-\int p_{\mathbf x}(\xi) \sum_i \psi_{\mathbf x,i}(\xi) \psi_i(\xi;\theta)d\xi
\\&=-\sum_i \int  p_{\mathbf x}(\xi) \frac {\partial \log p_{\mathbf x}(\xi)}{\partial \xi_i} \psi_i(\xi;\theta) d\xi
\\&=-\sum_i \int \frac {p_{\mathbf x}(\xi)} {p_{\mathbf x}(\xi)} \frac {\partial p_{\mathbf x}(\xi)}{\partial \xi_i} \psi_i(\xi;\theta) d\xi
\\&=-\sum_i \int \frac {\partial p_{\mathbf x}(\xi)}{\partial \xi_i} \psi_i(\xi;\theta) d\xi
\end{aligned}$$

如果我们能证明 

$$\int \frac {\partial p_{\mathbf x}(\xi)}{\partial \xi_i} \psi_i(\xi;\theta) d\xi=-\int \frac {\partial \psi_i(\xi)}{\partial \xi_i} p_{\mathbf x}(\xi)d\xi \tag{8}$$

那么就能得到 (7) 式。

根据乘法求导公司，$(uv)'=u'v+uv'$，得到分部积分 

$$\int u(x) v'(x) dx=\int [u(x)v(x)]'dx -\int u'(x)v(x)dx$$

那么对于 (8) 式，则有

$$\int \frac {\partial p_{\mathbf x}(\xi)}{\partial \xi_i} \psi_i(\xi;\theta) d\xi=\int_{} [p_{\mathbf x}(\xi) \psi_i(\xi;\theta)]' d \xi -\int \frac {\partial \psi_i(\xi)}{\partial \xi_i} p_{\mathbf x}(\xi)d\xi$$

积分范围为 $\xi \in \mathbb R^n$ 。对于一维数据，上式等价于

$$\int \frac {\partial p_{\mathbf x}(\xi)}{\partial \xi_i} \psi_i(\xi;\theta) d\xi=\left . [p_{\mathbf x}(\xi) \psi_i(\xi;\theta)] \right| _{\xi=-\infty}^{\xi = +\infty} -\int \frac {\partial \psi_i(\xi)}{\partial \xi_i} p_{\mathbf x}(\xi)d\xi$$

根据定理 1 的条件：对任意 $\theta$ 在 $\|\xi\| \rightarrow \infty$ 时有  $p_{\mathbf x}(\xi) \psi(\xi;\theta) \rightarrow 0$，那么上式等号右端第一项为 0，于是 (8) 式得证。

对于高维情况，下面这个 引理 1 用来证明 (8) 式。

**引理 1**
假设 $f, g$ 可微，那么

$$\lim_{a\rightarrow \infty, b\rightarrow -\infty} f(a,\xi_2,\ldots,\xi_n)g(b,\xi_2,\ldots,\xi_n)-f(b,\xi_2,\ldots,\xi_n)g(b,\xi_2,\ldots,\xi_n)
\\=\int_{-\infty}^{\infty} f(\xi)\frac {\partial g(\xi)}{\partial \xi_1} d\xi_1+\int_{-\infty}^{\infty} g(\xi)\frac {\partial f(\xi)}{\partial \xi_1} d\xi_1 \tag{9}$$

上式为了简单起见，仅使用了 $\xi_1$，实际上对 $\xi_i$ 均成立。

证明：

考虑 $i=1$ 的情况，

根据 

$$\frac {\partial f(\xi)g(\xi)}{\partial \xi_1}=f(\xi) \frac {\partial g(\xi)}{\partial \xi_1}+g(\xi)\frac {\partial f(\xi)}{\partial \xi_1}$$

对上式两边积分，

$$\int_{-\infty}^{\infty}\frac {\partial f(\xi)g(\xi)}{\partial \xi_1}d \xi_1=\int_{-\infty}^{\infty}f(\xi) \frac {\partial g(\xi)}{\partial \xi_1} d\xi_1+\int_{-\infty}^{\infty}g(\xi)\frac {\partial f(\xi)}{\partial \xi_1} d\xi_1$$

上式等号左边与 (9) 式等号左边相等。故 引理 1 得证。

下面来证明 (8) 式。

$$\int \frac {\partial p_{\mathbf x}(\xi)}{\partial \xi_1} \psi_1(\xi;\theta) d\xi=\int \left[\int \frac {\partial p_{\mathbf x}(\xi)}{\partial \xi_1} \psi_1(\xi;\theta) d\xi_1 \right] d\xi_{-1}
\\=\int \left[\lim_{a\rightarrow \infty, b\rightarrow -\infty} [p_{\mathbf x}(a,\xi_2,\ldots,\xi_n)\psi_1(a,\xi_2,\ldots, \xi_n;\theta)-p_{\mathbf x}(b,\xi_2,\ldots,\xi_n)\psi_1(b,\xi_2,\ldots, \xi_n;\theta)]-\int \frac {\partial p_{\mathbf x}(\xi)}{\partial \xi_1} \psi_1(\xi;\theta) d\xi_1 \right]d\xi_{-1}$$

根据定理 1 的假设条件：对任意 $\theta$ 在 $\|\xi\| \rightarrow \infty$ 时有  $p_{\mathbf x}(\xi) \psi(\xi;\theta) \rightarrow 0$，于是上式积分内第一项 $\lim [\cdot]$ 为 0，于是上式变为

$$\int \frac {\partial p_{\mathbf x}(\xi)}{\partial \xi_1} \psi_1(\xi;\theta) d\xi=-\int \frac {\partial p_{\mathbf x}(\xi)}{\partial \xi_1} \psi_1(\xi;\theta) d\xi_1$$

对于其他 $i$ ，上述推导过程均成立，于是 (8) 式得证，定理 1 成立。

</details>


假设有 $T$ 个观测序列，记为 $\mathbf x_1, \ldots, \mathbf x_T$，那么损失函数 (7) 近似为

$$\tilde J(\theta)=\frac 1 T \sum_{t=1}^T \sum_{i=1}^n \left[\partial_i \psi_i(\xi;\theta) + \frac 1 2 \psi_i(\xi;\theta)^2\right] + const \tag{9}$$

如果我们有非归一化密度函数 $q$ 的解析式，那么可以直接通过求导得到得分函数 (5) 式，带入损失函数 (9) 式，求得损失函数的解析解，最后可求最优解 $\theta^*$ 。

## 2.2 定理 2

假设 $\mathbf x$ 的概率密度函数服从模型 $p_{\mathbf x}(\cdot)=p(\cdot;\theta^*)$，其中 $\theta^*$ 是模型参数值。假设没有其他模型参数使得模型 pdf 等于 $p(\cdot;\theta^*)$，并且非归一化模型 $q(\xi;\theta) > 0$ 对所有的 $\xi, \theta$ 均成立，那么

$$J(\theta)=0 \Leftrightarrow \theta=\theta^*$$

上式中 $\theta$ 是损失函数 $J=0$ 的解。

定理 2 表示，使用得分函数的平均差的期望作为损失是有效的，以此损失函数训练出的模型可以很好的逼近真实的的数据概率分布。

**充分性证明** $J(\theta)=0 \Rightarrow \theta=\theta^*$

根据题目假设，$q>0$，这意味着 $\forall \xi , \ p_{\mathbf x}(\xi) > 0$，再加上 $J(\theta)=0$，所以根据 (6) 式，有 $\psi_{\mathbf x}(\xi)=\psi(\cdot;\theta)$ 对任意 $\xi$ 均成立，再根据 $\psi$ 的定义，$\psi(\cdot)$ 是 $\log p(\cdot)$ 的导数，也就是说两个函数的导数处处相等，那么这俩函数必然相差一个常量，即 $\log p_{\mathbf x}(\cdot)= \log p(\cdot;\theta) + c$，其中 c 是某个常量，于是 $p_{\mathbf x}(\cdot)=e^c \cdot p(\cdot;\theta)$，又由于 $p$ 是概率密度函数，积分等于 1，那么必然 $e^c=1$，故 $p_{\mathbf x}(\cdot)=p(\cdot;\theta)$，题干中假设了数据分布 $p_{\mathbf x}(\cdot)=p(\cdot;\theta^*)$，且没有其他模型参数使得模型 pdf 等于 $p(\cdot; \theta^*)$，那么一定有 $\theta=\theta^*$ 。

**必要性证明** $J(\theta)=0 \Leftarrow \theta=\theta^*$ 。太简单了，略

# 3. 例子

本节我们给三个例子说明 score matching 如何工作。

## 3.1 多元高斯分布

首先以一个非常简单的例子说明。

考虑一个多元高斯分布

$$p(\mathbf x; M, \mu)=\frac 1 {Z(M,\mu)} \exp (-\frac 1 2 (\mathbf x-\mu)^{\top} M (\mathbf x-\mu)) \tag{10}$$

其中 $M^{-1}$ 表示协方差矩阵，由于协方差矩阵是对称正定，所以 $M$ 也是对称正定。

这里 $Z(M,\mu)$ 是已知的，但是为了说明，这里假设 $Z(M, \mu)$ 不容易计算解析式。

现在我们要根据观测数据 $\mathbf x_{1:T}$ 来估计模型参数 $\theta=(\mu, M)$ 。

非归一化概率密度为

$$q(\mathbf x)=\exp (-\frac 1 2 (\mathbf x-\mu)^{\top} M (\mathbf x-\mu))$$

于是模型得分函数为 

$$\psi(\mathbf x; M, \mu)=-M(\mathbf x - \mu)$$

且二阶偏导为

$$\partial_i \psi_i(\mathbf x;M, \mu)=\frac {\partial \psi _ i }{\partial \mathbf x _i}=-m_{ii}$$

根据 (9) 式，(9) 式中 $\sum_{i=1}^n$ 放入方括号内部，那么第一项则变为 $-\sum_{i=1}^n m_{ii}$，第二项则是

$$\frac 1 2 \sum_{i=1}^n \psi_i^2=\frac 1 2 \psi^{\top} \psi$$

于是 (9) 式损失函数变为

$$\tilde J(M,\mu)=\frac 1 T \sum_{t=1}^T \left[\frac 1 2 (\mathbf x_t - \mu)^{\top} MM (\mathbf x_t-\mu)- \sum_i m_{ii}\right] \tag{11}$$

(11) 式方括号中第一项本来是 $M^{\top}M$，由于 $M$ 对称，所以 $M^{\top}M=MM$ 。

对 $J$ 求关于参数 $\mu$ 的导数，

$$\nabla_{\mu} \tilde J=\frac 1 T \sum_{t=1}^T MM (\mathbf x_t-\mu)=0$$

解得 

$$\mu=\frac 1 T \sum_{t=1}^T \mathbf x$$


对 $J$ 求关于参数 $M$ 的导数，

$$\frac {d (\mathbf x_t - \mu)^{\top} MM (\mathbf x_t-\mu)}{dM}=M(\mathbf x_t-\mu) \frac {d M(\mathbf x_t-\mu)}{dM}=M(\mathbf x_t-\mu) (\mathbf x_t-\mu)^{\top}$$

$$\frac {d \sum_{i=1}^n m_{ii}} {dM}=I$$

上面二式都是标量对矩阵的求导，导数也是矩阵，第二式很好理解，只有针对矩阵对角线位置元素求导为 1，其他位置显然都为 0 。对于第一式，记 $\mathbf x -\mu = \mathbf a$，那么分子为

$$\begin{aligned}f &=(\mathbf x_t - \mu)^{\top} M^{\top} M (\mathbf x_t-\mu)
\\\\ &=\sum_k \left[\sum_n \left(\sum_m a_m \cdot M_{m,:}^{\top} \right)_n M_{n,:} \right ]_k a_k
\\\\ &=\sum_k \sum_n \sum_m a_k a_m M_{mn}^{\top} M_{nk}
\\\\ &=\sum_k \sum_n \sum_m a_k a_m M_{nm} M_{nk}
\end{aligned}$$

求导，

$$\begin{aligned}\frac {\partial f}{\partial M_{ij}} &=\underbrace {\sum_k a_k a_j M_{ik}} _ {n=i,m=j} + \underbrace {\sum_m a_j a_m M_{im}} _ {n=i,k=j}
\\\\ &=M_{i,:} (\mathbf a \mathbf a^{\top})_{:,j} +  M_{i,:}(\mathbf a \mathbf a^{\top}) _ {:,j}
\\\\ &=(2M\mathbf a \mathbf a^{\top}) _ {ij}^ {\top}
\end{aligned}$$

于是 

$$\frac {\partial f}{\partial M}=2 M \mathbf a \mathbf a^{\top}$$

$$\nabla_M \tilde J= -I + M  \frac 1 {T} \sum_{t=1}^T  \mathbf a \mathbf a^{\top} =0$$

解得（注意 $M^{-1}$ 才是协方差矩阵）

$$M^{-1}=\frac 1 T \sum_{t=1}^T (\mathbf x_t-\mu) (\mathbf x_t-\mu)^{\top}$$

可见，基于 score matching 的模型估计与基于最大似然的模型估计结果一样。



