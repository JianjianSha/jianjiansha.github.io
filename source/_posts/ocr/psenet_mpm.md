---
title: 基于 PSENet 的场景文本检测
date: 2024-04-02 09:18:10
tags: OCR
---

# 1. 简介

场景文本检测算法大概可分为三类：

1. 基于回归，实时目标检测
2. 基于角点检测，检测 text box 的角点
3. 基于分割，像素级别的预测，更容易检测弯曲的文本，缺点是难以区分相邻的或者有重叠的文本

本文基于 PSENet 模型，提出混合池化模块（Mixed Pooling Module, MPM），考虑了三种 kernel shape：`NxN`，`1xN` 和 `Nx1` ，目的是更准确地定位场景文本区域。

# 2. 方法

基于 PSENet 检测场景文本，设计了 MPM，优化了骨干网络，流程如图1，

![](/images/ocr/psenet_based_1.png)

原生 PSENet 网络结构和本文优化后的网络结构如图2，

![](/images/ocr/psenet_based_2.png)

原生 PSENet 使用了 ResNet 抽取图像特征，本文优化的网络结合了 ResNeXt 和 Res2Net，增强特征提取，并且将 MPM 嵌入到网络中以获取不同位置的长距和短距之间的关联。

## 2.1 MPM

如图 3 所示是 MPM 的结构，

![](/images/ocr/psenet_based_3.png)

MPM 中不同的池化核 size 用于适配文本特征，同时避免由于规则池化核 `NxN` 引入的噪声影响。

一维池化有效地增强水平方向或垂直方向的感受野，在高语义级别上提高长距依赖，而方形池化核则可以捕获大范围的上下文信息。

图 3 中，`AdaptiveAvgPool` 是自适应均值池化，输出 spatial size 分别是 `18x18`，`10x10`，`1xh` 和 `wx1`，然后通过插值上采样得到 `hxw` 的 spatial size 。

## 2.2 优化骨干网络

如图 4 所示，

![](/images/ocr/psenet_based_4.png)

将 ResNet 改为 ResNeXt 和 Res2Net 的结合。

ResNeXt 是分组卷积，由 ResNet 和 Inception 构成，有效地降低了参数量。

Res2Net 增大了感受野，提高了不同尺寸特征提取的能力。

具体而言，经 `1x1` 卷积之后，特征 map 被分为 `s` 组，每一组特征的 channel 均为输入channel 的 `1/s`，spatial size 均相等，记一组特征为 $x _ i, i \in [1,s]$，除了 $x _ 1$，每个 $x _ i$ 均有一个 `3x3` 的卷积，卷积核记为 $K _ i$，卷积输出记为 $Y _ i$，卷积输入为 $x _ i$ 与 $Y _ i - 1$ 之和，

$$Y _ i = \begin{cases} x _ i & i=1 
\\ K _ i (x _ i) & i=2 
\\ K _ i (x _ i + Y _ {i-1} - 1) & 2 < i \le s \end{cases} \tag{1}$$