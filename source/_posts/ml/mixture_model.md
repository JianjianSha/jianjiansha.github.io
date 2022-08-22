---
title: 混合模型
date: 2022-05-16 17:52:30
tags: machine learning
summary: 多种模型的混合
---

# 1. 混合模型
混合模型中，我们假设数据 $y$ 可以从 $K$ 个生成过程其中之一生成得到，每个生成过程有各自的参数，分布为 $\pi_k (y|\alpha_k)$。

## 1.1 混合似然

令 $z \in \{1,...,K\}$ 表示生成数据 $y$ 所用的生成过程，基于 $z$ 的条件似然为

$$\pi(y|\boldsymbol \alpha, z)=\pi_z(y|\alpha_z)$$

其中 $\boldsymbol \alpha=(\alpha_1,\ldots, \alpha_K)$。

记每个生成过程发生（被选中）的概率为 $\theta_k$，那么有

$$\boldsymbol \theta=(\theta_1, \ldots, \theta_K), \ 0 \le \theta_k \le 1, \ \sum_{k=1}^K \theta_k =1$$

于是选择生成过程可以看做是多形式分布，

$$\pi(z|\boldsymbol \theta)=\theta_z$$

联合似然函数为

$$\pi(y,z|\boldsymbol \alpha, \boldsymbol \theta)=\pi(y|\boldsymbol \alpha, z) \pi(z|\boldsymbol \theta)=\pi_z(y|\alpha_z) \theta_z$$

$y$ 的边缘分布为

$$\begin{aligned}\pi(y|\boldsymbol \alpha, \boldsymbol \theta) &= \sum_z \pi(y,z|\boldsymbol \alpha, \boldsymbol \theta)
\\&=\sum_z \pi_z(y|\alpha_z) \theta_z
\\&=\sum_{k=1}^K \theta_k \pi_k(y|\alpha_k)
\end{aligned}$$

## 1.2 贝叶斯混合后验

我们需要参数的先验。

假设选择生成过程即多项式分布的参数和生成过程的参数独立，那么参数变量的先验为

$$\pi(\boldsymbol \alpha, \boldsymbol \theta)=\pi(\boldsymbol \alpha) \pi(\boldsymbol \theta)$$

那么基于某个观测数据 $y$，参数的后验为

$$\begin{aligned}\pi(\boldsymbol \alpha, \boldsymbol \theta|y)&=\frac {\pi(\boldsymbol \alpha, \boldsymbol \theta, y)}{\pi(y)} \propto \pi(\boldsymbol \alpha, \boldsymbol \theta, y)
\\&= \pi(\boldsymbol \alpha, \boldsymbol \theta)\pi(y|\boldsymbol \alpha, \boldsymbol \theta)
\\&=\pi(\boldsymbol \alpha) \pi(\boldsymbol \theta) \sum_{k=1}^K \theta_k \pi_k(y|\alpha_k)
\end{aligned}$$

类似地，基于整个观测数据集 $\mathbf y$ 参数的后验为

$$\pi(\boldsymbol \alpha, \boldsymbol \theta|\mathbf y)=\pi(\boldsymbol \alpha) \pi(\boldsymbol \theta)\prod_{n=1}^N \sum_{k=1}^K \theta_k \pi_k(y_n|\alpha_k)$$


# 2. ref
https://mc-stan.org/users/documentation/case-studies/identifying_mixture_models.html