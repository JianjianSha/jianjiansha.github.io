---
title: 人脸反欺骗 CDCN 论文解读
date: 2024-04-24 14:01:06
tags: face anti-spoofing
---

论文：[Searching Central Difference Convolutional Networks for Face Anti-Spoofing](https://arxiv.org/abs/2003.04092)

源码：[ZitongYu/CDCN](https://github.com/ZitongYu/CDCN)

# 1. 简介

提出基于 CDC（central difference convolution）的 FAS（人脸反欺骗）模型称为 CDCN，此模型是 frame level，即输入是单张图片（而非时间序列的多帧），模型使用 CDC 代替普通卷积，更能获取内在的人脸的细节模式。

使用 NAS（neural architecture search）搜索一个特别设计的 CDC 搜索空间，发现一个更强力的网络结构，记为 CDCN++。CDCN++ 集成了 MAFM（multiscale attention fusion module），提升性能。

# 2. 方法

## 2.1 CDC

普通卷积分为两步：采样和聚合，前者是采样当前中心的感受野 region，后者是进行加权求和。CDC 也是类似的两步，只是在聚合这一步与普通卷积不同，CDC 聚合面向中心的梯度，如图 1，

![]()

<center>图 1</center>

整个过程可表示为，

$$y(p _ 0) = \sum _ {p _ n \in \mathcal R} w(p _ n) \cdot [x(p _ 0 + p _ n) - x(p _ 0)] \tag {1}$$

其中 $p _ 0$ 是当前中心点，$y(p _ 0)$ 是输出值，$\mathcal R$ 是感受野区域，$p _ n$ 感受野中与中心 $p _ 0$ 的坐标偏差，例如 $3 \times 3$ 感受野为 $R=\{(-1,-1), (-1, 0) ,\ldots, (0, 1), (1, 1)\}$

对于人脸反欺骗任务，图像强度信息和梯度信息都很重要，所以需要联合普通卷积和 CDC，

$$y(p _ 0) = \theta \cdot \sum _ {p _ n \in \mathcal R} w(p _ n) \cdot [x(p _ 0 + p _ n) - x(p _ 0)] + (1 - \theta) \cdot \sum _ {p _ n \in \mathcal R} w(p _ n) \cdot x(p _ 0 + p _ n) \tag{2}$$

其中 $\theta \in [0, 1]$ 是平衡因子。

将 (2) 式整理为

$$y(p _ 0) = \sum _ {p _ n \in \mathcal R} w(p _ n) \cdot x(p _ 0 + p _ n) + \theta \cdot \left[ - x(p _ 0) \cdot \sum _ {p _ n \in \mathcal R} w(p _ n)\right] \tag{3}$$

(3) 式中第一项为普通卷积，第二项是对卷积核求和然后与输入值相乘，所以 pytorch 或 tensorflow 均很好实现。

## 2.2 CDCN

使用 [depthnet](https://arxiv.org/abs/1803.11097) 作为我们的基线，然后引入 CDC 构成 CDCN 网络。注意当 $\theta=0$ 时 CDCN 就是 depthnet。

CDCN 结构如表 1 所示

![]()
<center>表 1</center>

