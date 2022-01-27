---
title: Cross Entropy Loss
p: dl/x_ent_loss
date: 2021-01-12 10:35:40
tags: deep learning
mathjax: true
---

深度学习中有很多损失计算方式，不同的损失适合不同的问题，所以有必要对损失进行归类总结一下。
<!-- more -->
# 分类问题
考虑以下两种分类问题。
## 多分类
多分类（二分类可以看作是其特殊的一种情况）指每个样本属于`C` 个分类中的一个，预测值通常是一个长度为 `C` 的向量，ground truth 为 one-hot 向量。
## 多标签分类
共 `C` 种分类，每个样本可以属于其中一种或多种分类，网络输出依然是 `C` 长度的向量，ground truth 向量元素值为 `0` 或 `1`，且可以有多个 `1`。

# 激活函数
对于分类问题，网络最后一层的输出为长度为`C` 的向量（通常使用全连接），其元素值的范围为实数域，所以需要一个激活层，从而方便损失计算（以及梯度反向传播计算）。激活层有以下几种：
## Sigmoid
对向量的每个元素（神经元）使用 Sigmoid 函数，使得输出向量位于 `(0,1)` 之间，此时向量中最大元素对应的就是样本的预测分类。

Sigmoid 常用于二分类问题，此时标记为正类/负类，ground truth 为 `1/0`，网络最后一层可以使用单个神经元，神经元的输出经过 Sigmoid 后处于范围 `(0,1)`，如果值大于等于 0.5，那么就属于正类，否则属于负类。
## Softmax
Softmax 除了可以将神经元输出压缩到 `(0,1)` 之间，还使各元素值和为 1。Softmax 通常用于多分类。

# 分类损失
# Cross-Entropy Loss
单个样本的交叉熵损失计算公式为，
$$CE=-\sum_{i=1}^C y_i \log (x_i)$$
其中 $y_i$ 为 ground truth 向量中的第`i`个元素值，$x_i$ 为预测向量中第 `i` 个值，从这里也可以看出，正是有了上述激活层，才使得 $\log(x_i)$ 这一项有意义。多分类问题中，由于 GT 是 one-hot 向量，所以只有 $i=c$ 这项保留下来，其他项均为零。

二分类问题中，GT: $y=1$ 表示正，$y=0$ 表示负，最后一层输出为单个值（经过 Sigmoid 激活）`x`，所以单个样本的交叉熵损失为 
$$CE=-y \log (x) - (1-y) \log(1-x)$$

这跟负对数似然是一样的。

多标签分类问题中，假设一共有 `C` 种分类，对于单个样本而言，需要做 `C` 次二分类预测，交叉熵损失为
$$CE=-\sum_{i=1}^C y_i \log(x_i) - (1-y_i) \log(1-x_i)$$


## Balanced Cross Entropy
一种常见的解决分类不均衡的方法是引入一个权重因子 $\alpha \in [0,1]$，定义 
$$\alpha_t=\begin{cases} \alpha & y=1 \\ 1-\alpha & y=0 \end{cases}$$

考虑二分类问题，为了表示简便，定义真实分类对应的预测值为
$$x_t=\begin{cases} x & y=1 \\ 1-x & y=0 \end{cases}$$

$\alpha$ 均衡交叉熵损失为
$$CE=-\alpha_t \log(x_t)$$
通常，取 $\alpha$ 为类别的频率的倒数，这样就增加了低频类别的贡献，降低了高频类别的贡献。也可以将 $\alpha$ 看作超参，并使用 cross validation 获取一个较好的值。

## Focal Loss
单个样本的 Focal loss 为，
$$FL=-(1-x_t) ^{\gamma} \log(x_t)$$
展开则为
$$FL=-y(1-x)^{\gamma} \log(x) -(1-y)x^{\gamma} \log(1-x)$$

其中，$\gamma \ge 0$。


相比于交叉熵损失，Focal loss 增加了一个 scale 因子 $(1-x_t)^{\gamma}$，当 $x_t$ 越大，表明分类越是正确，越是应该降低其对损失的贡献，所以这个 scale 因子动态降低了那些 easy 样本对损失的贡献，从而使模型更专注于处理 hard 样本。

类似地，可以对 Focal loss 进行 $\alpha$ 均衡以处理类别不均衡的问题，此时 Focal loss 变体为
$$FL=-\alpha_t (1-x_t)^{\gamma} \log (x_t)$$
