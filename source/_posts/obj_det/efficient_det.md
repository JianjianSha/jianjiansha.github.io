---
title: EfficientDet 论文总结
date: 2022-03-04 17:31:39
tags: object detection
categories: 目标检测
summary: EfficentDet 论文分析与总结
mathjax: true
---

论文：[EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070)

# 1. 简介

之前的目标检测模型要么高效（例如 one-stage 和 anchor-free），要么高准确率，但是二者无法兼得。EfficientDet 旨在寻求一种可 scalable 目标检测框架，使得能同时兼顾高效和高准确率。

论文从 backbone，特征融合以及 分类/box network 等角度重新审视，并总结两点结论：

1. 高效的多尺度特征融合

    本文指出，FPN 自从问世被广泛应用，然而只是简单地将不同尺度的特征相加进行融合，这种融合不太妥。为了更好地的融合不同尺度的特征，本文提出一种高效的带权重双向 FPN，简称 BiFPN，这里的权重参数是可学习的。

2. 模型缩放

    对网络（包括 backbone，feature networ，以及 box/class network）的 `resolution/depth/width` 进行缩放。 （模型缩放的思想在之前已经被提出）

backbone 采用图像分类网络 EfficentNets，然后结合 BiFPN 以及 model scaling，这就是 EfficientDet 的主要思路。

> EfficientDet 是 one-stage detector。

# 2. BiFPN

## 2.1 跨尺度融合
如何融合不同尺度的特征？

如图 1，不同的模型其融合方式也不同。

![](/images/obj_det/efficientdet_1.png)
<center>图 1. 几种网络的特征融合方式</center>

记 $P_i ^ {in}, \ P_i ^ {out}$ 分别为第 `i` 层融合前后的特征，那么对于 FPN，有

$$\begin{aligned}P_7 ^ {out}&=Cov(P_7 ^ {in})
\\\\ P_6 ^ {out}&=Cov(P_6 ^ {in}+Resize(P_7 ^ {in}))
\\\\ \ldots
\\\\ P_3 ^ {out}&=Cov(P_3 ^ {in}+Resize(P_4 ^ {in}))
\end{aligned}$$

FPN 的融合信息流是单向的，即从上到下，PANET 则实现了融合信息的双向流动。NAS-FPN 则进一步地使用 neural architecture search（NAS）搜索出较好的跨尺度信息融合方式，不过 NAS 对硬件要求（GPU）较高，而且搜索出来的网络结果不规则，难以对其进行解释或者调整。

作者比较发现：PANET 有更高的准确率，然而其参数量和计算量也更高，于是提出几点优化：

1. 去掉只有一个输入的节点。

    PANET 中，去掉 `P7` 这一层的中间节点，以及 `P3` 层的最右端节点。这么做的原因很简单：这里是进行特征融合的网络，如果一个节点的输入没有进行特征融合，那么没有存在的必要。

2. 增加一条 edge，从原始节点到特征融合的输出节点，目的是为了融合更多特征。

3. 多次进行这种融合 （top-down & bottom-up），即多次堆叠这种融合，以实现更高水平的特征融合。

## 2.2 带权特征融合

用于融合的特征具有不同的尺度，其对融合的贡献不应该相同，于是作者提出带权重的特征融合，权重是可学习的参数。

考虑一下三种带权融合：

**Unbounded fusion**

$$O=\sum_i w_i \cdot I_i$$

其中 $w_i$ 可以是一个 scalar，表示一个特征一个权重，也可以是一个 vector，表示特征的各通道的权重，也可以是一个 tensor，表示特征的每个 pixel 对应一个权重。

**Softmax-based fusion**

$$O=\sum_i \frac {e ^ {w_i}}{\sum_j e ^ {w_j}} \cdot I_i$$

每个特征 $I_i$ 对应一个参数 $w_i$，使用 softmax 得到归一化权重，从而避免了 Unbounded fusion 会造成训练不稳定的缺点，但是 Softmax-based fusion 也有缺点，即降低 GPU 的计算速度。

**Fast normalized fusion**