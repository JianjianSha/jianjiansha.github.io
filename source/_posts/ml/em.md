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




# 例子
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

当第 $i$ 次迭代后，参数为 $\theta_i=(\pi_i, p_i, q_i)$，那么第 $i+1$ 次迭代时，

**E step：** 

计算观测数据 $x_j$ 来自硬币 B ($z=1$) 的概率，

$$\begin{aligned} r_{j1}&=P(z=1|X=x_j)
\\&=\frac {P(X=x_j,z=1)}{P(X=x_j)}
\\&=\frac {P(X=x_j,z=1)}{P(X=x_j,z=1)+P(X=x_j,z=0)}
\\&=\frac {\pi_i p_i^{x_j}(1-p_i)^{1-x_j}}{\pi_i p_i^{x_j}(1-p_i)^{1-x_j}+(1-\pi_i) q_i^{x_j}(1-q_i)^{1-x_j}}\end{aligned}$$

其中 $x_j$ 表示第 $j$ 次试验的观测结果。易得

$$r_{j0}=P(z=0|X=x_j)=1-r_{j1}$$

**M step：** 

计算模型的新估值

首先计算 $N_1=\sum_{j=1}^n r_{j1}$，然后根据 [高斯混合模型](2021/11/13/ml/GMM) 中的 {eq}`GMM7`， {eq}`GMM9`，{eq}`GMM11` 式，得

权重 

$$\pi_{i+1}=\frac 1 n \sum_{j=1}^n r_{j1}$$

两个分布的期望值为

$$p_{i+1}=\frac 1 {N_1} \sum_{j=1}^n r_{j1} x_j$$

$$q_{i+1}=\frac 1 {N_0} \sum_{j=1}^n r_{j0} x_j$$

# 