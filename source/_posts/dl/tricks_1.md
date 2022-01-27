---
title: Deep Learning Tricks1
p: dl/tricks1
date: 2021-01-08 14:32:52
tags: deep learning
mathjax: true
---
深度学习中有很多训练技巧，这里做一些总结。
<!-- more -->
# BatchNorm
## BatchNorm 的作用：
1. 防止过拟合。每个样本均经过批归一化后，防止出现离群点，从而导致过拟合。
2. 加快收敛。梯度下降过程中，每一层参数不断变化，导致输出结果的分布也在不断变化，后层网络就需要不停的适应这种变化。通常称这种变化为 `Internal Covariate Shift`，当数据流经深度网络的某层时，这层的计算结果可能会使得数据变的更大，或这更小，使得后面的网络层学习变得困难。使用 BN 后，BN 应用在每层计算函数之后，激活函数之前，使得数据分布集中在中间位置，这样再应用激活函数才有意义，并且 BN 之后的数据集中在激活函数的中间位置，此位置的梯度较大，从而有效防止了梯度弥散，并加快收敛速度。
3. 防止梯度弥散。

## BatchNorm 的计算公式
假设批大小为 `m`，足够小的数 $\epsilon$，

$\mu=\frac 1 m \sum_{i=1}^m x_i$

$\sigma^2=\frac 1 m \sum_{i=1}^m (x_i - \mu)^2$

$\hat x_i=\frac {x_i -\mu} {\sqrt{\sigma^2+\epsilon}}$

$y_i=\gamma \hat x_i+\beta$

上面最后一步为 `scale and shift`操作，其中参数 $\gamma, \beta$ 是 BN 层需要学习的参数，因为对数据做了归一化处理，使得难以学习到输入数据的特征，所以引入可训练参数 $\gamma, \ \beta$，这样既实现了归一化，又增加参数以可以学习输入数据的特征。

## BatchNorm 适用范围
BatchNorm 降低了数据之间的绝对差异，考虑相对差异（归一化带来的），所以如果某任务中图像的绝对差异很重要，那么其实不适合使用 BatchNorm。