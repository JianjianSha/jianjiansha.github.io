---
title: pytorch 中的各种 norm 总结
date: 2023-04-19 09:57:01
tags: PyTorch
mathjax: true
summary: 总结 pytorch 的所有 norm
---

几个 norm 的操作图示，

![](/images/pytorch/norm_1.png)

<center>图 1. 图源：《Group Normalization》论文</center>


## BatchNorm
批归一化是针对一个 mini-batch 内的数据进行归一化。首先给出归一化公式：
$$y=\frac {x-E[x]} {\sqrt{V[x]+\epsilon}} * \gamma + \beta$$

批归一化过程为：
BatchNorm Layer 的输入（mini-batch）为 $\mathcal B=\{x_{1...m}\}$ （单个通道，例如 [B,C,H,W] 的 batch 数据，那么取 size 为 $[B, H, W]$ 的数据），可学习参数为 $\gamma, \beta$。计算 mini-batch 的均值，方差
$$\mu_{\mathcal B} = \frac 1 m \sum_{i=1}^m x_i, \quad \sigma_{\mathcal B}^2=\frac 1 m \sum_{i=1}^m(x_i - \mu_{\mathcal B})^2$$
然后计算归一化后的值
$$\hat x_i = \frac {x_i - \mu_{\mathcal B}} {\sqrt {\sigma_{\mathcal B}^2+ \epsilon}}$$
最后进行 scale 和 shift，
$$y_i=\hat x_i \cdot \gamma + \beta$$



__小结：__ 沿着 batch 方向进行归一化

## LayerNorm
Layer 归一化是针对某个数据（样本）内部进行归一化，假设某个数据样本到达 LayerNorm 层为 $x$，无论 $x$ 是多少维的 tensor，均可以看作是 1D vector，即 $x=(x_1,...x_H)$，$H$ 是 LayerNorm 层的单元数（也是 $x$ 的特征数），于是 LayerNorm 过程为
$$\mu=\frac 1 H \sum_{i=1}^H x_i, \quad \sigma^2=\frac 1 H \sum_{i=1}^H (x_i-\mu)^2$$
于是 LayerNorm 后的值为
$$y=\frac {x-\mu} {\sqrt {\sigma^2+\epsilon}} \cdot \gamma + \beta$$

__小结：__ 

1. 沿着特征方向进行归一化（特征包含了除 batch 维度外的其他所有维度）
2. 输入 $(B, C, H, W)$，那么将单个样本 $(C, H, W)$ 做归一化

有了前面的归一化介绍，我们知道归一化过程都很类似，区别在于如何计算 $\mu, \sigma$，或者说沿着什么方向进行归一化。

## InstanceNorm
对于每个样例的每个 channel 分别计算 $\mu, \sigma$。假设输入为 $(B,C,H,W)$，那么沿着 $(H,W)$ 方向做归一化。

## GroupNorm
GroupNorm 是选择一组 channels 进行归一化，所以是介于 InstanceNorm（单个channel）和 LayerNorm （全部 channels）之间的。

输入 $(B, C, H, W)$，其中 $C=g \cdot c$，$g$ 为分组数量，那么对 $(c, H, W)$ 做归一化。


调用示例（来自 pytorch 官网）：

```python
>>> input = torch.randn(20, 6, 10, 10)
>>> # Separate 6 channels into 3 groups
>>> m = nn.GroupNorm(3, 6)
>>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
>>> m = nn.GroupNorm(6, 6)
>>> # Put all 6 channels into a single group (equivalent with LayerNorm)
>>> m = nn.GroupNorm(1, 6)
>>> # Activating the module
>>> output = m(input)
```