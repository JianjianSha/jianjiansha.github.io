---
title: 高斯分布的似然估计
date: 2022-05-21 13:53:41
tags: machine learning
mathjax: true
---

# 1 理论

随机变量 $\mathbf x$ 服从高斯分布 $\mathcal N(\mu, \Sigma)$，单个样本的对数似然为

$$l= \log \mathcal N(\mathbf x|\mu,\Sigma)=-\frac D 2 \log(2\pi)-\frac 1 2 \log \det(\Sigma) - \frac 1 2 (\mathbf x-\mu)^{\top}\Sigma^{-1}(\mathbf x-\mu) \tag{1}$$

整个数据集 $\mathcal X$ 的对数似然为

$$\mathcal L = \sum_i \log \mathcal N(\mathbf x|\mu,\Sigma)$$

## 1.1 参数估计

对参数求偏导并令偏导为 0，那么

$$\frac {\partial \mathcal L} {\partial \mu}=\sum_i \frac {\partial \mathcal l_i} {\partial \mu}=\sum_i - \Sigma^{-1}(\mathbf x_i-\mu) \stackrel{\Delta}= \mathbf 0$$

两边左乘 $\Sigma$，解得

$$\mu=\frac 1 n \sum_i \mathbf x_i \tag{2}$$

另外一个偏导，

$$\frac {\partial \mathcal L} {\partial \Sigma}=\sum_i -\Sigma^{-1} + \frac 1 2 (\Sigma^{-1} \circ I) + \frac 1 2 \Sigma^{-\top}(\mathbf x_i-\mu)(\mathbf x_i-\mu)^{\top}\Sigma^{-\top} \stackrel{\Delta}=\mathbf 0$$


其中 $I$ 是单位矩阵， $\Sigma^{-1} \circ I$ 是 Hadamard 乘积，也就是矩阵按位（elementwise）相乘，结果就是矩阵对角化 $diag(\Sigma^{-1})$ ，由于 $\Sigma$ 协方差矩阵是 **对称矩阵**，故 $\Sigma^{-\top}=\Sigma^{-1}$，于是上式两边先后左乘 $\Sigma$ 和右乘 $\Sigma$，得到

$$\sum_i -\Sigma + \frac 1 2 \Sigma + \frac 1 2 (\mathbf x_i-\mu)(\mathbf x_i-\mu)^{\top} = \mathbf 0$$

上式中，因为 $\Sigma(\Sigma^{-1} \circ I)=\Sigma [diag(\Sigma^{-1})]=I$，解上式得 

$$\Sigma=\frac 1 n \sum_i (\mathbf x_i-\mu)(\mathbf x_i-\mu)^{\top} \tag{3}$$