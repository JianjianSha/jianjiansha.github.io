---
title: loss
date: 2019-07-16 17:32:26
tags: CV
mathjax: true
---
总结一些常见的损失（虽然我把本文归类到 CV，但实际上这些损失函数并不仅仅用于 CV 中，只是目前我只关注 CV 而已）
<!-- more -->
# Cross-Entropy Loss
交叉熵损失常用于分类任务中，比如共有 C 中可能的分类，（softmax 之后的）预测向量为 $P=(p_1,...,p_C)$，其中 $p_i$ 表示分类为 i 的概率，且有 $\sum_i^C p_i=1$，目标真实分类为 c，那么 gt target 为 $T=(t_1,...,t_C)$，其中
$$t_i=\begin{cases} 1 & i=c \\\\ 0 & i\ne c \end{cases}$$
于是交叉熵损失为
$$CE=-\sum_{i=1}^C t_i \log p_i$$

## Binary Cross-Entropy Loss
特别地，当分类数量 C=2 时，目标为正的预测概率为 p，真实分类为 t，$t \in \{0,1\}$，
$$CE=-t \log p - (1-t) \log (1-p)$$
为方便起见，记
$$p_t=\begin{cases} p & t=1 \\\\ 1-p & t=0 \end{cases}$$
于是，
$$ CE=-\log p_t $$

## Balanced Cross-Entropy Loss
如果样本分类分布不均（long-tail distribution），即少数分类的占据了绝大多数样本，而其他分类的样本数量则非常少，比如二分类中，分类为 1 的样本很少而分类为 0 的样本很多，那么从分类为 1 的样本中学习到的信息就有限，或者说分类为 1 的样本对损失贡献较小从而对优化过程作用较弱，故引入权重因子，t=1 具有权重 $\alpha$，t=0 具有权重 $1-\alpha$，$\alpha \in [0,1]$。实际操作中，设置 $\alpha$ 反比例于分类样本频次，或将 $\alpha$ 作为超参数通过交叉验证设置其值（RetinaNet 中设置为 0.25）。于是平衡交叉熵损失为，
$$CE=-\alpha_t \log p_t$$

## Focal Loss
虽然 balanced cross-entropy loss 中 $\alpha$ 平衡了正负样本，但是并没有区分简单样本和困难样本，我们知道 $p_t \gg 0.5$ 属于简单样本，当简单样本数量很多时，其贡献的总损失不容忽视，显然，我们更应该重视困难样本，因为从困难样本中更能学习到有用（对模型至关重要的）信息，所以，降低简单样本的损失权重，比如这里的 Focal loss，
$$FL=-(1-p_t)^{\gamma} \log p_t \ , \ \gamma \ge 0$$
其中 $(1-p_t)^{\gamma}$ 称为调节因子。

Focal loss 的性质：
1. $p_t$ 较小，表示误分类，困难样本，此时 $(1-p_t)^{\gamma}$ 相对较大
2. $p_t$ 较大，表示分类正确，简单样本，此时 $(1-p_t)^{\gamma}$ 相对较小

# MSE
均方误差为
$$MSE = \frac 1 n \sum_{i=1}^n (Y_i-\hat Y_i)^2$$
表示 n 个样本的 L2 范数误差的平均，其中 $Y_i, \hat Y_i$ 分别表示第 i 个样本的真实值和预测值。

## L2 Loss
$$L_2=(Y_i-\hat Y_i)^2$$
缺点：当 $|Y_i-\hat Y_i|>1$ 时，误差会被放大很多，导致模型训练不稳定。
## L1 Loss
$$L_1=|Y_i-\hat Y_i|$$
缺点：当 $|Y_i-\hat Y_i|<1$ 时，梯度（的绝对值）不变，导致优化过程出现震荡。
## Smooth L1 Loss
结合以上两点，得到 Smooth L1 损失，
$$L=smooth_{L_1}(Y_i-\hat Y_i)
\\\\ smooth_{L_1}(x)=\begin{cases} 0.5 x^2 & |x|<1
\\\\ |x|-0.5 & otherwise \end{cases}$$

## Regularized Loss
机器学习中，为防止过拟合加入正则项损失，通常是参数的 L1 范数或 L2 范数，略。

