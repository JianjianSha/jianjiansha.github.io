---
title: EM 算法
date: 2021-11-01 18:20:46
tags: machine learning
p: ml/em
mathjax: true
---

EM （Expection-Maximization）算法，是一种迭代算法，求期望和求最大交替迭代，从而更新我们所关心的变量，直到收敛（更新足够小）。EM 算法通常用于含有隐变量的概率模型参数的极大似然估计。
<!--more-->

在 [高斯混合模型](2021/11/13/ml/GMM)一文中，讲到迭代更新模型参数，这正是 EM 算法

1. E 步：估计 responsibility  $r_{nk}$ 值（数据 $\mathbf x_n$ 来自第 $k$ 个分量的后验概率）。
2. M 步：使用 $r_{nk}$ 的值更新参数 $\theta$ 。

注：需要给出初始化参数 $\theta_0$。




# 1. 例子
（三硬币模型）：假设有 A,B,C 三枚硬币，正面出现的概率分布为 $\pi, p, q$。定义一次试验为：先抛 A，如出现正面，那么选择 B，否则选择 C，然后再抛所选的硬币（B 或 C），正面向上记为 1，反面向上记为 0 。独立重复试验 10 次，观测结果为 
```
1,1,0,1,0,0,1,0,1,1
```

选择的是 B 还是 C 是不可知的 ，这是隐变量记为 $z$，$z=1$ 表示选择 B，$z=0$ 表示选择 C 。观测结果变量记为 $x$，参数记为 $\theta=(\pi, p, q)$，根据以上试验的观测结果估计参数值。

一次试验观测结果为 x，概率为

$$\begin{aligned}P(x;\theta)&=\sum_z P(z;\theta) P(x|z;\theta)
\\&=\pi p^x(1-p)^{1-x}+(1-\pi)q^x(1-q)^{1-x}
\end{aligned}$$

那么对于所有的试验结果 $X$，概率为

$$\begin{aligned}P(X;\theta)&=\sum_Z P(Z;\theta) P(X|Z;\theta)
\\&=\prod_{i=1}^n [\pi p^{x_i}(1-p)^{1-x_i}+(1-\pi)q^{x_i}(1-q)^{1-x_i}]
\end{aligned}$$

极大似然估计为

$$\hat \theta =\arg \max_{\theta} \log P(X;\theta)$$

显然上式没有解析解，故想到通过迭代方法求数值解，于是 EM 算法就大显身手了。

既然是迭代，必然要设置一个初始值，记初值 $\theta_0=(\pi_0, p_0, q_0)$ 。

当第 $t$ 次迭代后，参数为 $\theta_t=(\pi_t, p_t, q_t)$，那么第 $t+1$ 次迭代时，

**E step：** 

计算观测数据 $x_j$ 来自硬币 B ($z=1$) 的概率，

$$\begin{aligned} r_{j1}&=P(z=1|X=x_j)
\\&=\frac {P(X=x_j,z=1)}{P(X=x_j)}
\\&=\frac {P(X=x_j,z=1)}{P(X=x_j,z=1)+P(X=x_j,z=0)}
\\&=\frac {\pi_t p_t^{x_j}(1-p_t)^{1-x_j}}{\pi_t p_t^{x_j}(1-p_t)^{1-x_j}+(1-\pi_t) q_t^{x_j}(1-q_t)^{1-x_j}}\end{aligned}$$

其中 $x_j$ 表示第 $j$ 次试验的观测结果。易得

$$r_{j0}=P(z=0|X=x_j)=1-r_{j1}$$

**M step：** 

计算模型的新估值

首先计算 $N_1=\sum_{j=1}^n r_{j1}$，然后根据 [高斯混合模型](/2021/11/13/ml/GMM) 中的 (7), (9), (12) 式，得

权重 

$$\pi_{t+1}=\frac 1 n \sum_{j=1}^n r_{j1}$$

两个分布的期望值为

$$p_{t+1}=\frac 1 {N_1} \sum_{j=1}^n r_{j1} x_j$$

$$q_{t+1}=\frac 1 {N_0} \sum_{j=1}^n r_{j0} x_j$$

# 2. 理论

观测变量记为 $x$，隐变量记为 $z$，模型参数记为 $\theta$。$x$ 的概率为

$$p(x;\theta)=\sum_z p(z;\theta) p(x|z;\theta)=\sum_z p(x,z;\theta)$$

对于一组观测数据 $x_1, \ldots, x_n$，其对数似然为 

$$l=\sum_{i=1}^n \log p(x_i; \theta)=\sum_{i=1}^n \log \sum_{z_i} p(x_i,z_i; \theta) \tag{1}$$

若没有隐变量 $z$，我们可以根据最大似然来估计参数 $\theta$，通常是对参数求导令导数为 0 进行求解；但是有了隐变量 $z$ 后，显然对 (1) 式求导令导数为 0 的等式很难求解。将 (1) 式进行如下处理：

$$\begin{aligned} \sum_i \log p(x_i;\theta) &= \sum_i \log \sum_{z_i} p(x_i,z_i; \theta)
\\&=\sum_i \log \sum_{z_i} Q(z_i) \frac {P(x_i,z_i;\theta)}{Q(z_i)}
\\& \ge \sum_i \sum_{z_i} Q(z_i) \log \frac {P(x_i,z_i;\theta)}{Q(z_i)}
\end{aligned} \tag{2}$$

其中 $Q(z)$ 是 $z$ 的概率分布，满足 $\sum_z Q(z)=1$。

上式不等式的推导使用了 Jensen 不等式：对于凹函数 $f$ 满足

$$f(\sum_j w_j x_j) \ge \sum_j w_j f(x_j)  \quad s.t. \ \sum_j w_j=1, \ w_j \ge 0 \tag{3}$$

于是要求最大似然，我们只要求其 lower bound，并令这个 lower bound 最大，即求  $\sum_i \sum_z Q(z) \log \frac {P(x_i,z;\theta)}{Q(z)}$ 的最大值。对某个观测数据 $x$，

$$\sum_z Q(z) \log \frac {P(x,z;\theta)}{Q(z)}=\mathbb E_{Q(z)}\left[\log \frac {P(x,z;\theta)} {Q(z)}\right]$$


$$\log p(x;\theta) =\log \sum_z Q(z) \frac {P(x,z;\theta)}{Q(z)} \ge \mathbb E_{Q(z)}\left[\log \frac {P(x,z;\theta)} {Q(z)}\right] \tag{4}$$

但是 (4) 式 RHS 的最大值不一定就是 LHS 的最大值，但是可以通过迭代的方式求得，如下图所示，

![](/images/ml/em_1.png)

<center>图 1. 迭代方式求解优化问题

[来源](https://blog.csdn.net/v_JULY_v/article/details/81708386)

</center>

图 1 说明：

1. t 时刻参数 $\theta^t$，固定此参数，求 $Q(z)$ 使得 (4) 式等号成立，即图中 $Q(z)$ 分布从绿色变成蓝色。
2. 固定 $Q(z)$，求 $\theta^{t+1}$，使得 $\sum_i \sum_{z_i} Q(z_i) \log \frac {P(x_i,z_i;\theta)}{Q(z_i)}$ 最大。

## 2.1 EM 算法

### 2.1.1 E step

对于第 `1` 步，要求 (4) 式等号成立时的 $Q(z)$。由于 log 函数是严格上凹的，当且仅当 $x_j, j=1,2,\ldots$ 均相等时，Jensen 不等式中等号成立，不妨令其均等于 $c$，对比 (2) 和 (4) 式，可知

$$\frac {P(x,z;\theta)}{Q(z)}=c$$

于是 

$$P(x;\theta)=\sum_z P(x,z;\theta)=\sum_z Q(z) \cdot c=1 \cdot c=c$$

综合上两式，可得

$$Q(z)=\frac {P(x,z;\theta)} c=\frac {P(x,z;\theta)} {P(x;\theta}=P(z|x;\theta) \tag{5}$$

即，__当 $Q(z)$ 是 $z$ 的后验分布 $P(z|x;\theta)$ 时，(4) 式等号成立__。

__这就是 E 步，求后验分布 $p(z|x;\theta^t)$__。

考虑到问题中是一组观测数据，各观测数据之间独立（n 个独立重复试验），所以实际上我们求 $Q(z_i)=p(z_i|x_i;\theta^t)$，注意这里 $x_i, \ z_i$ 表示第 `i` 次试验中的随机变量。这个 $p(z_i|x_i; \theta^t)$ 就是前面 [GMM](/2021/11/13/ml/GMM) 中所说的 Responsibility 矩阵的第 `i` 行：观测数据 $x_i$ 来自 $z_i$ 各个值的概率。

### 2.1.2 M step

对于第 `2` 步，

$$\max_{\theta} \ l=\max_{\theta} \sum_i \sum_{z_i} Q(z_i) \log \frac {P(x_i,z_i;\theta)}{Q(z_i)}= \max_{\theta} \sum_i \sum_{z_i} Q(z_i) \log P(x_i,z_i;\theta) \tag{6}$$

其中 $Q(z_i)$ 与参数 $\theta$ 无关，因为已经使用已知值 $\theta^t$ 求解出来，而待求参数 $\theta$ 仅存在于 $P(x_i,z_i;\theta)$ 中。

目标 $l$ 对参数 $\theta$ 求梯度并令梯度为 0，求得 $\theta^{t+1}$。

## 2.2 EM 算法的可行性

如果确保 EM 算法收敛到最大似然估计？

假设 EM 第 t 次和第 t+1 次的迭代得到参数为 $\theta^t$ 和 $\theta^{t+1}$，那么如果证明了似然函数 $l(\theta^{t+1}) \ge l(\theta^t)$，那么表示 EM 算法一直朝着正确的方向进行迭代，当迭代的似然函数值保持不变，或者变化小于一定阈值，那么迭代结束。

我们使用归纳法证明。

在得到 $\theta^t$ （初始时给定 $\theta^0$ 的值，即参数初始值）之后，根据 E step，得到 

$$Q^t(z_i)=p(z_i|x_i;\theta^t)$$

E step 保证了给定 $\theta^t$ 值，Jensen 不等式中等号成立，根据 (2) 式，即

$$l(\theta^t)=\sum_i \log \sum_{z_i} Q^t(z_i) \frac {p(x_i,z_i;\theta^t)}{Q(z_i)} = \sum_i \sum_{z_i} Q^t(z_i) \log \frac {p(x_i,z_i;\theta^t)}{Q^t(z_i)}$$

然后 M step，固定 $Q^t(z_i)$ 不变，在这个时候，对 (6) 式关于 $\theta$ 求导得到目标最大值，从而得到最优参数值 $\theta^{t+1}$ 。

在 $\theta^t$ 处，Jensen 不等式中等号成立，但是在 $\theta^{t+1}$ 处，Jensen 不等式中的等号不一定成立，故

$$\begin{aligned}l(\theta^{t+1}) & \ge \sum_i \sum_{z_i} Q^t(z_i) \log \frac {p(x_i,z_i;\theta^{t+1})}{Q^t(z_i)} 
\\ &\ge \sum_i \sum_{z_i} Q^t(z_i) \log \frac {p(x_i,z_i;\theta^t)}{Q^t(z_i)}
\\&= l(\theta^t)
\end{aligned} \tag{7}$$

在 t+2 时刻，固定 $\theta^{t+1}$ 不变，求解 $Q^{t+1}(z_i)$ ，使得 (7) 式 的第一个不等式中等号成立。

# 3. 例子回顾

首先我们先回顾上面第一节的那个例子。E step 中计算的 

$$r_{j1}=P(z=1|X=x_j)
\\r_{j0}=P(z=0|X=x_j)$$

其实就是 $Q(z_j)$，这里 $z_j \in \{1,0\}, \forall j = 1,2,\ldots$，因为 E step 中，固定参数后，$Q(z_j)$ 最优解就是 $P(z_j|x_j)=r_{jz_j}$。

M step 中固定 $Q(z)$ 然后计算参数 $\theta_{t+1}=(\pi_{t+1}, p_{t+1}, q_{t+1})$，为了一致，这里参数使用下标表示 time step。

根据 (6) 式，对参数求偏导，

$$\begin{aligned} &\frac {\partial \sum_i \sum_{z_i} Q(z_i) \log P(x_i,z_i;\theta)}{\partial\pi}
\\=& \sum_i \sum_{z_i} Q(z_i) \frac 1 {P(x_i,z_i;\theta)} \frac {\partial P(x_i,z_i;\theta)} {\partial \pi}
\\& \downarrow 展开 {z_i \in \{0,1\}}
\\=&\sum_i r_{i1} \frac 1 {P(x_i,1;\theta)} \frac {\partial \pi p^{x_i}(1-p)^{1-x_i}} {\partial \pi}+r_{i0} \frac 1 {P(x_i,0;\theta)} \frac {\partial (1-\pi) q^{x_i}(1-q)^{1-x_i}} {\partial \pi}
\\=& \sum_i r_{i1} \frac 1 {\pi} - r_{i0} \frac 1 {1-\pi}
\\=& \sum_i r_{i1} \frac 1 {\pi} - (1-r_{i1}) \frac 1 {1-\pi}
\\=& \sum_i [r_{i1}\frac 1 {\pi(1-\pi)}- \frac 1 {1-\pi}]
\\=&0 \ \color{silver} {(\# 令偏导等于 0 \#)}
\end{aligned}$$

得到 

$$\pi= \frac 1 n \sum_i r_{i1}$$

类似的，分别对 $p, \ q$ 求导，将上面推导的第三行进行改写，$\partial \pi$ 改写为 $\partial p$ 和 $\partial q$，

$$\begin{aligned} &\frac {\partial \sum_i \sum_{z_i} Q(z_i) \log P(x_i,z_i;\theta)}{\partial p}
\\=& \sum_i \sum_{z_i} Q(z_i) \frac 1 {P(x_i,z_i;\theta)} \frac {\partial P(x_i,z_i;\theta)} {\partial p}
\\=&\sum_i r_{i1} \frac 1 {P(x_i,1;\theta)} \frac {\partial \pi p^{x_i}(1-p)^{1-x_i}} {\partial p}+r_{i0} \frac 1 {P(x_i,0;\theta)} \underbrace {\frac {\partial (1-\pi) q^{x_i}(1-q)^{1-x_i}} {\partial p}}_{分子与 p 无关，导数为 0}
\\=& \sum_i r_{i1} \frac 1 {P(x_i,1;\theta)} \pi [x_i p^{x_i-1} (1-p)^{1-x_i}-(1-x_i)p^{x_i}(1-p)^{-x_i}]
\\=& \sum_i r_{i1} \frac 1 {P(x_i,1;\theta)} [\frac {x_i} p P(x_i,1;\theta)-\frac {1-x_i}{1-p} P(x_i,1;\theta)]
\\=& \sum_i r_{i1} (\frac {x_i} p- \frac {1-x_i}{1-p})
\\=&0 \ \color{silver} {(\# 令偏导等于 0 \#)}
\end{aligned}$$

解得

$$p=\frac {\sum_i r_{i1}x_i}{\sum_i r_{i1}}$$

q 的求解过程与 p 相似，略。

对于高斯混合模型 ，理解透彻了这篇文章中 E, M 步骤所求的目标，容易验证与 [GMM](/2021/11/13/ml/GMM) 这篇文章中的 (7), (9), (12) 式的结果也是相同的，并且理解了为什么想到去计算 responsibilities，以及通过 responsibilities 再计算新的参数 $\theta$，这样一直迭代下去，可以逼近最终要求的后验分布。