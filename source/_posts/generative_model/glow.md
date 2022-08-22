---
title: Glow
date: 2022-08-20 16:19:01
tags: flow
mathjax: true
---

# 1. 背景

假设 $\mathbf x \sim p^{\star}(\mathbf x)$ 分布未知。观察到 i.i.d 的数据集 $\mathcal D$，模型分布 $p_{\theta}(\mathbf x)$ ，那么数据集负对数似然为

$$L(\mathcal D)=\frac 1 N \sum_{i=1}^N - \log p_{\theta}(\mathbf x^{(i)}) \tag{1}$$

基于 flow 的生成过程可描述为

$$\mathbf z \sim p_{\theta}(\mathbf z) \tag{2}$$

$$\mathbf x = \mathbf g_{\theta}(\mathbf z) \tag{3}$$

其中隐变量 $\mathbf z$ 的先验分布 $p_{\theta}(\mathbf z)$ 比较简单，例如 $p_{\theta}(\mathbf z) = \mathcal N(\mathbf z; 0, I)$ 。函数 $\mathbf g_{\theta}$ 可逆，逆变换为 $\mathbf z = \mathbf f_{\theta}(\mathbf x)=\mathbf g_{\theta}^{-1}(\mathbf x)$ 。

函数 $\mathbf f$ 可以是多个函数的组合 $\mathbf f = \mathbf f_1 \circ \mathbf f_2 \circ \cdots \circ \mathbf f_K$，这也是 flow 得名由来，变量流为

$$\mathbf x \leftrightarrow \mathbf h_1 \cdots \leftrightarrow \mathbf z \tag{4}$$

根据变量变换公式，可知其概率关系满足

$$\begin{aligned}\log  p_{\theta}(\mathbf x) &= \log p_{\theta}(\mathbf z) + \log |\det (d \mathbf z / d\mathbf x)|
\\ &= \log p_{\theta}(\mathbf z) + \sum_{i=1}^K \log |\det (d \mathbf h_i / d\mathbf h_{i-1})|
\end{aligned} \tag{5}$$

其中 $\mathbf x = \mathbf h_0, \ \mathbf z = \mathbf h_L$ 。 上式推导用到了矩阵乘积的行列式等于行列式的乘积这一定理。

