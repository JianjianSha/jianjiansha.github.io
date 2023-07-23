---
title: Importance Weighted Autoencoders
date: 2022-08-23 17:22:57
tags: 
    - vae
    - generative model
mathjax: true
---

论文：[Importance Weighted Autoencoders](https://arxiv.org/abs/1509.00519)

VAE 对后验概率 $p_{\theta}(z|x)$（以及它的变分近似 $q(z|x)$）做了强假设，例如后验分布可按维度因式分解，即后验分布的协方差矩阵是对角矩阵。VAE 的目标函数会导致过于简单的表征，从而不能充分利用网络容量。本文作者实现了 importance weighted autoencoder (IWAE)，在 VAE 的基础上使用了严格紧密的对数似然下限。

为了表示简单，本文的 $x, h, z$ 均表示向量，而不用加粗字母表示向量，有时候标量也使用不加粗字母表示，这可以从上下文区分。

# 1. 背景

VAE 中生成过程（解码）为

$$p(x|\theta)=\sum_{h^1, \ldots, h^L} p(h^L|\theta)p(h ^ {L-1}|h ^ L,\theta) \cdots p(x|h ^ 1,\theta) \tag{1}$$

其中 $\theta$ 表示网络参数，$h^1,\ldots, h^L$ 表示中间层变量或隐变量。定义 $h^0 \stackrel{\Delta}= x$ 。$p(h^l|h^{l+1})$ 表示一个复杂的非线性变换。

识别（推断/编码）网络类似的表示为，

$$q(h|x)=q(h ^ 1|x)q(h ^ 2|h ^ 1)\cdots q(h ^ L|h ^ {L-1}) \tag{2}$$

定义 $h \stackrel{\Delta}= h^L$。先验概率固定为一个简单的分布，如 $h \sim \mathcal N(0, I)$ 。

条件分布 $p(h ^ l|h ^ {l+1})$ 和 $q(h^l|h^{l-1})$ 均为协方差为对阵矩阵的高斯分布，对角线元素和期望向量均由网络计算得到。

对于实数观测值（$x$），$p(x|h ^ 1)$ 也可以定义为与 $p(h ^ l|h ^ {l+1})$ 类似的高斯分布；对于二分类观测值（$x$），$p(x|h ^ 1)$  定义为伯努利分布，其期望值由网络计算得到。

**VAE 的目标函数是最大化 ELBO $\mathcal L(x)$**，

$$\log p(x) = \log \mathbb E_{q(h|x)} \left[\frac {p(x,h)}{q(h|x)}\right] \ge \mathbb E_{q(h|x)} \left[\log \frac {p(x,h)}{q(h|x)}\right] \stackrel{\Delta}= \mathcal L(x) \tag{3}$$

证明：

$$ \log \mathbb E_{q(h|x)} \left[\frac {p(x,h)}{q(h|x)}\right] =\log \left( \int_h q(h|x) \frac {p(x,h)}{q(h|x)} dh \right)=\log  p(x)$$

根据 Jensen 不等式，以及 log 是上凹函数，故

$$\log \mathbb E_{q(h|x)} \left[\frac {p(x,h)}{q(h|x)}\right] \ge \mathbb E_{q(h|x)} \left[\log \frac {p(x,h)}{q(h|x)}\right]$$

证毕。

**结论**：

$$\mathcal L(x) = \log p(x) - D_{KL}(q(h|x)||p(h|x)) \tag{4}$$

证明：

根据定义进行推导，

$$\begin{aligned}\mathcal L(x) &=\mathbb E_{q(h|x)} \left[\log \frac {p(x,h)}{q(h|x)}\right] 
\\\\ &= \mathbb E_{q(h|x)} \left[\log \frac {p(h|x)}{q(h|x)} + \log p(x)\right] 
\\\\ &= \mathbb E_{q(h|x)} [\log p(x)] - D_{KL}(q(h|x)||p(h|x))
\\\\ &= \log p(x) - D_{KL}(q(h|x)||p(h|x))
\end{aligned}$$

给定一个 $x$，其概率密度为 $\log p(x)$，那么对条件分布 $q(h|x)$ 而言，$x$ 和 $\log p(x)$ 均可视作常量，那么 $\mathbb E_{q(h|x)} [\log p(x)] = \log p(x)$ 。证毕。


本文的识别（编码）网络中，$q(h ^ l|h ^ {l-1}, \theta)$ 形式为 $\mathcal N(h ^ l|\mu(h ^ {l-1},\theta), \Sigma(h ^ {l-1}, \theta))$，期望和协方差矩阵是前一个 layer 的输出 $h^{l-1}$ 和网络参数 $\theta$ 的函数。

根据重参数技巧，$\epsilon ^ l \sim \mathcal N(0,I)$，可以得到后一个 layer 的输出为

$$h^l(\epsilon ^ l, h ^ {l-1}, \theta)=\Sigma(h ^ {l-1}, \theta) ^ {1/2} \epsilon ^ l + \mu(h ^ {l-1}, \theta) \tag{5}$$

计算每个 $h ^ l$ 时，均使用重参数技巧采样一个辅助随机变量 $\epsilon ^ l$，那么最终的编码值 $h(\epsilon, x, \theta)$ 根据 (5) 是递归得到，这里 $\epsilon=(\epsilon ^ 1,\ldots, \epsilon ^ L)$ 。

那么 ELBO 的梯度可计算为

$$\begin{aligned}\nabla_{\theta} \mathbb E_{q(h|x,\theta)} \left [\log \frac {p(x,h|\theta)}{q(h|x,\theta)} \right]&= \nabla_{\theta} \mathbb E_{\epsilon \sim \mathcal N^L(0, I)} \left [\log \frac {p(x,h(\epsilon,x, \theta)|\theta)}{q(h(\epsilon, x,\theta)|x, \theta)}\right]
\\\\ &=\mathbb E_{\epsilon \sim \mathcal N^L(0, I)} \left[\nabla_{\theta} \log \frac {p(x,h(\epsilon,x, \theta)|\theta)}{q(h(\epsilon, x,\theta)|x, \theta)}\right]
\end{aligned} \tag{6}$$

# 2. IWAE

作者认为，识别网络的输出样本 $h$ 中，有 20% 处于高后验概率区域，已经足够，可以使得识别网络执行非常准确的推断。将标准降到这个程度，会使得生成网络的空间更加灵活。作者提到的算法模型称为 IWAE。

IWAE 的网络框架与 VAE 相同，由识别网络和生成网络构成，学习的目标是最大化一个 $\log p(x)$ 的下限，这个下限与 VAE 中的不同，如下所示，对应对数似然的 k 样本重要性权重估计，

$$\mathcal L_k(x)=\mathbb E_{h_1,\ldots, h_k \sim q(h|x)} \left[\log \frac 1 k \sum_{i=1}^k \frac {p(x,h_i)}{q(h_i|x)}\right] \tag{7}$$

其中，$h_1,\ldots, h_k$ 是从识别网络中独立的采样 k 次的结果。记 $w_i = p(x,h_i)/q(h_i|x)$ 。

(7) 式就是边缘概率分布 $\log p(x)$ 的下限，即

$$\mathcal L_k = \mathbb E \left[\log \frac 1 k \sum_{i=1}^k w_i \right] \le \log \mathbb E \left[\frac 1 k \sum_{i=1}^k w_i\right] = \log p(x) \tag{8}$$

记住，上式是针对 $q(h|x)$ 求期望。

当 $k=1$ 时，就是标准 VAE 的目标函数。

当 $k>1$ 时，使用更多的样本可以提高下限的紧密程度，即

$$\log p(x) \ge \mathcal L_{k+1} \ge \mathcal L_k \tag{9}$$

当 $p(h,x)/q(h|x)$ 有界时，那么 $k \rightarrow \infty$ 时 $\mathcal L_k \rightarrow \log p(x)$ 。

证明过程见论文附录，这里不赘述。

## 2.1 训练过程

使用 (7) 式计算 $\mathcal L_k$ 。与 VAE 一样，使用重参数技巧，

$$\begin{aligned}\nabla_{\theta} \mathcal L_k(x)&=\nabla_{\theta} \mathbb E_{h_1,\ldots, h_k} \left[\log \frac 1 k \sum_{i=1}^k w_i \right]
\\\\ &=\mathbb E_{\epsilon_1,\ldots, \epsilon_k} \left [\nabla_{\theta} \log \frac 1 k \sum_{i=1}^k w(x, h(x,\epsilon_i,\theta), \theta)\right]
\\\\ &=\mathbb E_{\epsilon_1,\ldots, \epsilon_k} \left [\sum_{i=1}^k \tilde w_i \nabla_{\theta} \log w(x, h(x,\epsilon_i,\theta), \theta)\right]
\end{aligned} \tag{10}$$


其中 $\tilde w_i = w_i/\sum_{i=1}^k w_i$ 是归一化权重。

当 $k=1$ 时，$\tilde w_1=1$， 对应 VAE 的更新规则。

