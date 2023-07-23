---
title: 高斯混合模型
date: 2021-11-13 13:53:13
tags: machine learning
p: ml/GMM
mathjax: true
---
机器学习中，我们常用某种方式去表征数据，例如选取合适的模型。单一模型的表征能力很弱，故可以采用混合模型，例如混合高斯分布。
<!--more-->

混合模型是由多个单个模型凸组合而成，例如 $K$ 个简单分布凸组合成分布

$$p(\mathbf x) = \sum_{k=1}^ K \pi_k p_k(\mathbf x) \tag{1}$$

$$0 \le \pi_k \le 1, \quad \sum_{k=1}^ K \pi_k = 1$$

本文主要讨论高斯混合模型 (GMM)，即，每个简单分布都是高斯分布。

# GMM

K 个高斯分布的线性组合如下，

$$p(\mathbf x|\theta)=\sum_{k=1}^ K \pi_k \mathcal N(\mathbf x|\mu_k, \Sigma_k) \tag{2}$$

$$0 \le \pi_k \le 1, \quad \sum_{k=1}^ K \pi_k = 1$$

定义参数 $\theta := \{\mu_k, \Sigma_k, \pi_k: k = 1,\ldots, K\}$

## 参数学习

根据最大似然学习参数

记数据集为 $\mathcal X = \{\mathbf x_1, \cdots, \mathbf x_N\}$ 。似然函数为

$$p(\mathcal X|\theta)=\prod_{n=1}^ N p(\mathbf x_n |\theta) \tag{3}$$

$$p(\mathbf x_n|\theta)=\sum_{k=1}^ K \pi_k \mathcal N(\mathbf x_n |\mu_k,\Sigma_k) \tag{4}$$

对数似然为

$$\log p(\mathcal X|\theta)=\underbrace {\sum_{n=1}^ N \log \sum_{k=1}^ K \pi_k \mathcal N(\mathbf x_n |\mu_k,\Sigma_k)}_{=:\mathcal L} \tag{5}$$

考虑单个高斯分布（即，非高斯混合），那么单个样本的似然函数为

$$\log \mathcal N(\mathbf x|\mu,\Sigma)=-\frac D 2 \log(2\pi)-\frac 1 2 \log \det(\Sigma) - \frac 1 2 (\mathbf x-\mu) ^  {\top}\Sigma ^  {-1}(\mathbf x-\mu)$$

此时我们可以求得关于参数 $\mu$ 和 $\Sigma$ 的最大似然估计的解析解（求数据集的似然最大值，对参数分别求偏导，并令偏导为 0 ，可得最大似然的解析解）。但是 (5) 中，`log` 位于 $\sum_{k=1}^ K$ 外部，使得偏导表达式复杂。


我们依然使用求偏导并令偏导为 0 的方法进行求解，

$$\frac {\partial \mathcal L} {\partial \mu_k} =\mathbf 0 \Leftrightarrow \sum_{n=1}^ N \frac {\partial \log p(\mathbf x_n |\theta)} {\partial \mu_k} = \mathbf 0$$

$$\frac {\partial \mathcal L} {\partial \Sigma_k} =\mathbf 0 \Leftrightarrow \sum_{n=1}^ N \frac {\partial \log p(\mathbf x_n |\theta)} {\partial \Sigma_k} = \mathbf 0$$

$$\frac {\partial \mathcal L} {\partial \pi_k} =\mathbf 0 \Leftrightarrow \sum_{n=1}^ N \frac {\partial \log p(\mathbf x_n |\theta)} {\partial \pi_k} = \mathbf 0$$

此外，log 函数的求导规则为

$$\frac {\log p(\mathbf x_n|\theta)} {\partial \theta}=\frac 1 {p(\mathbf x_n |\theta)}\cdot \frac {\partial p(\mathbf x_n |\theta)} {\partial \theta}$$

## Responsibility

定义GMM 中第 $k$ 个高斯分量对第 $n$ 个数组点的 responsibility 为

$$r_{nk}=\frac {\pi_k \mathcal N(\mathbf x_n|\mu_k, \Sigma_k)} {\sum_{j=1}^ K \pi_j \mathcal N(\mathbf x_n|\mu_j,\Sigma_k)} \tag{6}$$

或者说是第 $n$ 个数据点属于第 $k$ 个高斯分量的概率（数据点由这个高斯分量产生的概率，这个概率不是真实的概率，而是基于最大似然估计）。

$$r_{nk}=p(z_k=1|\mathbf x_n)=\frac {p(\mathbf x_n, z_k=1)}{p(\mathbf x_n)}$$

可见 $r_{nk}$ 实际上是 $\mathbf z$ 的后验概率 $p(\mathbf z|\mathbf x_n)$ 。

$\mathbf r_n :=[r_{n1},\ldots,r_{nK}]^ {\top}$ 为一个归一化的概率向量。

将 $K$ 个高斯分量对 $N$ 个数据点的 responsibilities 写成矩阵形式 $R \in \mathbb R^ {N \times K}$，那么矩阵中第 $n$ 行表示数据 $\mathbf x_n$ 来自各个高斯分量的概率，这是一个归一化向量，第 $k$ 列表示高斯分量 $\mathcal N(\mu_k,\Sigma_k)$ 对所有数据的 responsibilities（注意这不是归一化的向量）。

基于 responsibilities，我们可以对模型参数 $\theta$ 进行更新，而 responsibilities 又依赖于模型参数，所以对 $\theta$ 更新时，需要固定 responsibilities，然后更新 $\theta$，然后再计算新的 responsibilities，然后再更新 $\theta$，如此迭代更新下去，直到达到一个预设的最大迭代次数，或者参数的变化量（例如 F2 范数的变化量）小于一个预设的阈值。

## 更新 Mean

均值（期望）参数 $\mu_k$ 的更新为

$$\mu_k^ {new}=\frac {\sum_{n=1}^ N r_{nk}\mathbf x_n} {\sum_{n=1}^ N r_{nk}} \tag{7}$$

证：

计算单个数据的概率对参数 $\mu_k$ 的梯度，
$$\begin{aligned} \frac {\partial p(\mathbf x_n |\theta)}{\partial \mu_k}&=\sum_{j=1}^ K \pi_j \frac {\partial \mathcal N(\mathbf x_n|\mu_j,\Sigma_j)} {\partial \mu_k}=\pi_k \frac {\partial \mathcal N(\mathbf x_n|\mu_k,\Sigma_k)} {\partial \mu_k}
\\&=\pi_k (\mathbf x_n-\mu_k)^ {\top} \Sigma_k^ {-1} \mathcal N(\mathbf x_n|\mu_k,\Sigma_k)
\end{aligned}$$

于是，数据集的对数似然对 $\mu_k$ 的梯度为

$$\begin{aligned} \frac {\partial \mathcal L} {\partial \mu_k}&=\sum_{n=1}^ N \frac {\partial \log p(\mathbf x_n |\theta)}{\partial \mu_k}=\sum_{n=1} ^  N \frac 1 {p(\mathbf x_n |\theta)} \frac {\partial p(\mathbf x_n |\theta)}{\partial \mu_k}
\\\\ &=\sum_{n=1} ^  N (\mathbf x_n-\mu_k)^ {\top} \Sigma_k ^  {-1} \underbrace {\frac {\pi_k \mathcal N(\mathbf x_n |\mu_k,\Sigma_k)} {\sum_ {j=1} ^  K \pi_j \mathcal N(\mathbf x _ n |\mu_j,\Sigma_j)}} _ {=r_ {nk}}
\\\\ &=\sum_{n=1} ^  N r _ {nk}(\mathbf x_n - \mu_k) ^  {\top} \Sigma_k ^  {-1}
\end{aligned}$$

注意上式中间一行的推导中， $r_{nk}$ 这一项中的参数 $\theta$ 使用的是上一轮的值，参数 $\theta=\{(\pi_k, \mu_k, \Sigma_k)|k=1,\ldots, K\}$

 令上式这个梯度为零，得

$$\sum_{n=1}^ N r_{nk} (\mathbf x_n-\mu_k)^ {\top} \Sigma_k^ {-1}=\mathbf 0$$

上式两端右乘 $\Sigma_k$，得

$$\sum_{n=1}^ N r_{nk} \mathbf x_n = \sum_{n=1}^ N r_{nk} \mu_k \Leftrightarrow \mu_k^ {new}=\frac {\sum_{n=1}^ N r_{nk}\mathbf x_n} {\sum_{n=1}^ N r_{nk}}=\frac 1 {N_k} \sum_{n=1} r_{nk}\mathbf x_n$$

其中 $N_k :=\sum_{n=1}^ N r_{nk}$ 就是上面我们所说的 responsibilites 矩阵的 第 $k$ 列的和，表示第 $k$ 个高斯分量对所有数据的 responsibilities 之和。

(7) 可以看作是所有数据在分布 
$$\mathbf r_k := [r_{1k},\cdots, r_{Nk}]/N_k \tag{8}$$

下的期望，

$$\mu_k \leftarrow \mathbb E_{\mathbf r_k} [\mathcal X]$$

（类比，数据 $1,2,\ldots, 6$ 在 $[1/6,1/6,\cdots,1/6]$ 分布下的期望）

## 更新协方差

协方差矩阵 $\Sigma_k$ 的更新为

$$\Sigma_k^ {new} = \frac 1 {N_k} \sum_{n=1}^ N r_{nk} (\mathbf x_n-\mu_k)(\mathbf x_n-\mu_k)^ {\top} \tag{9}$$

从形式上看，(9) 式可以看作是所有数据在分布 (8) 分布下的二阶中心矩。下面来证明 (9) 式。

证：

计算数据集的对数似然对 $\Sigma_k$ 的梯度，

$$\frac {\partial \mathcal L}{\partial \Sigma_k}= \sum_{n=1}^ N \frac {\partial \log p(\mathbf x_n|\theta)} {\partial \Sigma_k}=\sum_{n=1}^ N \frac 1 {p(\mathbf x_n|\theta)} \frac {\partial p(\mathbf x_n|\theta)}{\partial \Sigma_k} \tag{10}$$

$p(\mathbf x_n|\theta)$ 由 (4) 式给出，故只要计算

$$\begin{aligned} \frac {\partial p(\mathbf x_n|\theta)} {\partial \Sigma_k}&=\frac {\partial}{\partial \Sigma_k}\left(\pi_k (2\pi)^  {-D/2} \det(\Sigma_k)^  {-1/2} \exp (-\frac 1 2 (\mathbf x_n-\mu_k)^  {\top} \Sigma_k^  {-1} (\mathbf x_n-\mu_k))\right)
\\\\ &=\pi_k (2\pi)^  {-D/2} [\left(\frac {\partial}{\partial \Sigma_k}\det(\Sigma_k)^  {-1/2}\right)\exp(-\frac 1 2 (\mathbf x_n-\mu_k)^  {\top}\Sigma_k^  {-1}(\mathbf x_n-\mu_k))
\\\\ & \quad +\det(\Sigma_k)^  {-1/2} \frac {\partial}{\partial \Sigma_k} \exp(-\frac 1 2 (\mathbf x_n-\mu_k)^  {\top}\Sigma_k^  {-1}(\mathbf x_n-\mu_k))
]
\end{aligned} \tag{11}$$

根据矩阵的求导规则可知，

$$\frac {\partial}{\partial \Sigma_k} \det(\Sigma_k)^  {-1/2}=-\frac 1 2 \det(\Sigma_k)^  {-1/2} \Sigma_k^  {-1}$$

$$\frac {\partial} {\partial \Sigma_k}(\mathbf x_n-\mu_k)^  {\top}\Sigma_k^  {-1}(\mathbf x_n-\mu_k) = -\Sigma_k^  {-1}(\mathbf x_n-\mu_k)(\mathbf x_n-\mu_k)^  {\top}\Sigma_k^  {-1}$$

将上面两式 代入 (11)式 得

$$\begin{aligned} \frac {\partial p(\mathbf x_n|\theta)}{\partial \Sigma_k}&=\pi_k (2\pi)^  {-D/2} [-\frac 1 2 \det(\Sigma_k)^  {-1/2} \Sigma_k^  {-1} \exp(-\frac 1 2 (\mathbf x_n-\mu_k)^  {\top}\Sigma_k^  {-1}(\mathbf x_n-\mu_k)) 
\\\\ & \quad +\frac 1 2 \det(\Sigma_k)^  {-1/2} \Sigma_k^  {-1}(\mathbf x_n-\mu_k)(\mathbf x_n-\mu_k)^  {\top}\Sigma_k^  {-1} \exp(-\frac 1 2 (\mathbf x_n-\mu_k)^  {\top}\Sigma_k^  {-1}(\mathbf x_n-\mu_k))]
\\\\& = \pi_k \mathcal N(\mathbf x_n|\mu_k,\Sigma_k)[-\frac 1 2 \Sigma_k^  {-1}+\frac 1 2\Sigma_k^  {-1}(\mathbf x_n-\mu_k)(\mathbf x_n-\mu_k)^  {\top}\Sigma_k^  {-1}]
\\\\&=\pi_k \mathcal N(\mathbf x_n|\mu_k,\Sigma_k)\cdot [-\frac 1 2 (\Sigma_k^  {-1}-\Sigma_k^  {-1}(\mathbf x_n-\mu_k)(\mathbf x_n-\mu_k)^  {\top}\Sigma_k^  {-1})]
\end{aligned}$$

将上式和 (4) 式代入 (10) 式，得

$$\begin{aligned}\frac {\partial \mathcal L} {\partial \Sigma_k}&= \sum_ {n=1} ^  N \underbrace {\frac {\pi_k \mathcal N(\mathbf x_n |\mu_k,\Sigma_k)} {\sum_{j=1}^  K \pi_j \mathcal N(\mathbf x_n |\mu_j,\Sigma_j)}}_ {=r_{nk}} \cdot  [-\frac 1 2 (\Sigma_k^  {-1}-\Sigma_k^  {-1}(\mathbf x_n -\mu_k)(\mathbf x_n-\mu_k)^  {\top}\Sigma_k^  {-1})]
\\\\&=-\frac 1 2 \sum_ {n=1}^  N r_{nk} (\Sigma_k^  {-1}-\Sigma_k^  {-1}(\mathbf x_n-\mu_k)(\mathbf x_n -\mu_k)^  {\top}\Sigma_k^  {-1})
\\\\&=-\frac 1 2 \Sigma_ k^  {-1} \underbrace{\sum_{n=1}^  N r_{nk}}_{=N_k} + \frac 1 2 \Sigma_k^  {-1}\left(\sum  _ {n=1}^  N r _ {nk}(\mathbf x_n -\mu_k)(\mathbf x_n-\mu_k)^  {\top}\right)\Sigma _ k^  {-1}
\end{aligned}$$

令上式这个梯度为零，得

$$N_k\Sigma_k^  {-1}=\Sigma_k^  {-1}\left(\sum_{n=1}^  N r_{nk}(\mathbf x_n-\mu_k)(\mathbf x_n-\mu_k)^  {\top}\right)\Sigma_k^ {-1}$$

上式两边右乘 $\Sigma_k^ {-1}$，得

$$N_k \mathbf I = \left(\sum_{n=1}^ N r_{nk}(\mathbf x_n-\mu_k)(\mathbf x_n-\mu_k)^ {\top}\right)\Sigma_k^ {-1}$$

然后左乘 $\Sigma_k$ 得到 (9) 。

## 更新混合权重

混合模型的权重参数 $\pi_k$ 的更新为

$$\pi_k^ {new} = \frac {N_k} N \tag{12}$$

证：

由于 $\sum_{k=1}^ N \pi_k=1$ 这个约束条件，我们采用拉格朗日乘子法，

$$\begin{aligned}L &=\mathcal L + \lambda \left(\sum_{k=1}^ K \pi_k-1\right)
\\\\&= \sum_{n=1}^ N \log \sum_{k=1}^ K \pi_k \mathcal N(\mathbf x_n|\mu_k,\Sigma_k) + \lambda \left(\sum_{k=1}^ K \pi_k-1\right)
\end{aligned}$$

求梯度，由于 $\pi_k$ 不存在于 $\mathcal N$ 中，故梯度非常容易计算，

$$\begin{aligned}\frac {\partial L}{\partial \pi_k}&=\sum_{n=1}^ N \frac {\mathcal N(\mathbf x_n|\mu_k,\Sigma_k)}{\sum_{j=1}^ K \pi_j \mathcal N(\mathbf x_n|\mu_j, \Sigma_j)} + \lambda
\\\\&= \frac 1 {\pi_k} \underbrace{ \sum_{n=1}^ N \frac {\pi_k \mathcal N(\mathbf x_n|\mu_k,\Sigma_k)}{\sum_{j=1}^ K \pi_j \mathcal N(\mathbf x_n|\mu_j, \Sigma_j)}}_{=N_k} + \lambda = \frac {N_k} {\pi_k}+\lambda
\end{aligned}$$

$$\frac {\partial L}{\partial \lambda}=\sum_{k=1}^ K \pi_k -1$$


令上面两个梯度均为零，得

$$\pi_k = - \frac {N_k}{\lambda}$$

$$\sum_{k=1}^ K \pi_k -1 = \sum_{k=1}^ K \left(-\frac {N_k} {\lambda}\right)-1=0 \Leftrightarrow \lambda = -\sum_{k=1}^ K N_k=-N$$

于是

$$\pi_k^ {new} = \frac {N_k} N$$

其中 $N$ 为数据集大小。

# EM 算法总结

为何要引入 responsibilities？为何要使用 EM 算法求解？

从上面的的偏导的计算中不难发现，令偏导位0 得到一方程组 $\frac {\partial \mathcal L}{\partial \theta}=\mathbf 0$，但是这个方程组无法得到参数的解析解，故而考虑迭代方法一步一步逼近真实解，这就是 EM 算法。为了使得计算得以进行，在迭代的过程中，我们将方程组的一些项（这些项中包含了参数 $\theta$）固定，并使用旧的参数 $\theta^ t$ 计算出来（E step），然后求解方程组，得到新的参数数值 $\theta^ {t+1}$（M step）。重复迭代过程即可。


**EM 算法迭代步骤：**

1. 初始化 $\ \pi_k, \mu_k, \Sigma_k$，$k=1,\ldots, K$ 。例如可以初始化为 $\pi_k=1/K$，$\mu_k \sim \mathcal N(\mathbf 0, \sigma^ 2 I)$，$\Sigma_k=\mathbf I$ 。
2. E-step。 计算每个分量对每个数据 $\mathbf x_n$ 的 responsibility，(6) 式。
3. M-step。 更新参数的值。(7) ，(9) ， (12) 式。


# 隐变量视角

可以将 GMM 看作是一个具有隐变量 $z$ 的模型。

生成过程：

数据点 $\mathbf x$ 是由 GMM 中 K 个概率中的某个确定的概率生成，记 $z_k \in \{0,1\}$ 表示是否选择第 k 个概率，如果 $z_k=1$ 表示选择第 k 个概率，然后生成 $\mathbf x$，故

$$p(\mathbf x|z_k=1)=\mathcal N(\mathbf x|\mu_k, \Sigma_k)$$

定义 $\mathbf z=[z_1,\ldots,z_K]^ {\top}$ 为随机向量，它是一个 one-hot 向量，显然有

$$p(\mathbf z)=\pi, \quad \sum_{k=1}^ K \pi_k=1$$

单个样本的最大似然为

$$p(\mathbf x|\theta) = \sum_{\mathbf z} p(\mathbf x|\theta, \mathbf z)\cdot p(\mathbf z|\theta)=\sum_{k=1}^ K \pi_k \mathcal N(\mathbf x|\mu_k,\Sigma_k)$$

由于 $\mathbf z$ 是 one-hot vector，故一共有 $K$ 种取值。

## 后验概率分布
上面所讨论的 responsibility 实际上就是隐变量 $\mathbf z$ 的后验概率，

$$\begin{aligned}p(z_k=1|\mathbf x_n)&=\frac {p(z_k=1)p(\mathbf x_n|z_k=1)}{p(\mathbf x)}
\\\\&=\frac {\pi_k \mathcal N(\mathbf x_n|\mu_k,\Sigma_k)}{\sum_{j=1}^ K \pi_j \mathcal N(\mathbf x_n|\mu_j,\Sigma_j)}
\\\\&=r_{nk}
\end{aligned}$$

## EM 算法回顾

从隐变量视角回顾 EM 算法，发现 **E-step 实际上就是计算 $\mathbf z$ 的后验概率 $p(\mathbf z|\mathbf x, \theta^ {(t)})$**，

**M-step 是求最大化**

$$\begin{aligned}\max_{\theta} \ Q(\theta|\theta^ {(t)})&=\mathbb E_{\mathbf z|\mathbf x, \theta^ {(t)}} [\log p(\mathbf x| \mathbf z;\theta)]
\\\\&=\int [\log p(\mathbf x| \mathbf z;\theta)] \cdot p(\mathbf z|\mathbf x, \theta^ {(t)}) \ d \mathbf z
\end{aligned} \tag{13}$$

的参数 $\theta$ 。

对于离散型隐变量 $\mathbf z$ 而言，上式可写为

$$Q(\theta|\theta^ {(t)})=\sum_{\mathbf z} [\log p(\mathbf x| \mathbf z;\theta)] \cdot p(\mathbf z|\mathbf x, \theta^ {(t)}) \tag{14}$$


**证：**

我们验证对 $\mu_k$ 的梯度是否与上面的推导一致，对 $\Sigma_k$ 的梯度验证完全类似。

对 $\mu_k$ 求梯度，

$$\begin{aligned} \frac {\partial Q}{\partial \mu_k}&=\frac {\partial} {\partial \mu_k} \log p(\mathbf x_n|z_k=1;\theta) p(z_k=1|\mathbf x_n, \theta^ {(t)})
\\\\&=\frac {\partial} {\partial \mu_k} \log \mathcal N(\mathbf x_n|\mu_k,\Sigma_k) \cdot r_{nk}
\\\\&=\frac 1 {\mathcal N(\mathbf x_n|\mu_k,\Sigma_k)} \frac {\partial \mathcal N(\mathbf x_n|\mu_k,\Sigma_k)} {\partial \mu_k} \cdot r_{nk}
\\\\&=\frac 1 {\mathcal N(\mathbf x_n|\mu_k,\Sigma_k)} (\mathbf x_n-\mu_k)^ {\top} \Sigma_k^ {-1} \mathcal N(\mathbf x_n|\mu_k,\Sigma_k) \cdot r_{nk}
\\\\&= (\mathbf x_n-\mu_k)^ {\top} \Sigma_k^ {-1} \cdot r_{nk}
\end{aligned}$$

这是单个数据点的对数最大似然 对 $\mu_k$ 的梯度，如果扩展到整个数据集，那么梯度为

$$\frac {\sum_{n=1}^ N \partial Q} {\partial \mu_k}=\sum_{n=1}^ N  (\mathbf x_n-\mu_k)^ {\top} \Sigma_k^ {-1} \cdot r_{nk}$$

这与 (7) 式的证明中所推导的梯度 $\partial \mathcal L / \partial \mu_k$ 完全一样，令上式为 $\mathbf 0$ 求得向量 $\mu_k$ 的最优解即 (7) 式。


根据后验概率 $p(\mathbf z|\mathbf x, \theta^ {(t)})$，那么迭代更新 $\mathbf z$ 的先验分布如下，

$$p(z_k=1)=\sum_{\mathbf x} p(z_k=1,\mathbf x| \theta^ {(t)}) = \sum_{\mathbf x} p(\mathbf z|\mathbf x, \theta^ {(t)}) p(\mathbf x)=\frac 1 N \sum_{n=1}^ N r_{nk}=\frac {N_k} N$$

### 解释

写出样本 $\mathbf x$ 的对数似然，

$$\begin{aligned}\log p(\mathbf x;\theta)&=\log \sum_{\mathbf z} p(\mathbf z;\theta) p(\mathbf x| \mathbf z;\theta)
\\\\& \ge \sum_{\mathbf z} p(\mathbf z;\theta) \log p(\mathbf x|\mathbf z;\theta)
\\\\&=\mathbb E_{p(\mathbf z;\theta)}[\log p(\mathbf x|\mathbf z;\theta)]
\end{aligned} \tag{15}$$

根据 Jensen 不等式得到上面推导中的不等关系。

在 $t$ timestep 迭代更新时，已知此时的参数值为 $\theta^ t$，

**E step：**


当 $p(\mathbf x|\mathbf z;\theta^ {t})=c$ 为一常数时，不等式中等号成立，此时 

$$p(\mathbf x; \theta^ {t})=\sum_{\mathbf z}p(\mathbf x,\mathbf z;\theta^ {t})=\sum_{\mathbf z}c \cdot p(\mathbf z;\theta^ {t})=c$$

于是 

$$p(\mathbf z; \theta^ {t})=\frac {p(\mathbf z| \mathbf x;\theta^ {t}) p(\mathbf x;\theta^ {t})}{p(\mathbf x|\mathbf z; \theta^ {t})}=p(\mathbf z| \mathbf x;\theta^ {t})$$


**M step：**

于是，我们根据 $\theta^ t$ 计算出 $\mathbf z$ 的后验概率 $p(\mathbf z| \mathbf x;\theta^ {t})$，作为 $p(\mathbf z; \theta)$ 的值代入 (15) 式，求

$$\max_{\theta^ {t+1}} \ \mathbb E_{p(\mathbf z;\theta^ t)}[\log p(\mathbf x|\mathbf z;\theta^ {t+1})]$$

上式这个期望就是 (13) 式。

事实上由于此时 $p(\mathbf z; \theta)=p(\mathbf z| \mathbf x;\theta^ {t})$，(13) 式等价于 最大化

$$\begin{aligned} \max_{\theta} Q(\theta|\theta^ {(t)})&=\max_{\theta} \mathbb E_{\mathbf z|\mathbf x, \theta^ {(t)}} [\log p(\mathbf x| \mathbf z;\theta)]
\\\\&=\max_{\theta} \int [\log p(\mathbf x,\mathbf z;\theta)-\log p(\mathbf z| \mathbf x;\theta^ {t})] \cdot p(\mathbf z|\mathbf x, \theta^ {(t)}) \ d \mathbf z
\\\\&=\max_{\theta} \mathbb E_{\mathbf z|\mathbf x, \theta^ {(t)}} [\log p(\mathbf x, \mathbf z;\theta)]
\end{aligned}$$

**可行性研究**

再次强调，由于无法计算解析解，所以优化时，分别先后计算 $p(\mathbf z;\theta)$ 和 (15) 式右端期望的最优解。这种迭代方式是可行的，证明参见 [EM 算法](/ml/2021/11/01/em) 2.2 节内容。