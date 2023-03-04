---
title: CSPNet 论文解读
date: 2022-04-12 16:03:53
tags: object detection
mathjax: true
summary: 重新审视网络框架并缓解计算量巨大的问题
---


# 1. 简介

模型为了强大而变得更宽更深，但是同时也带来了很大的计算负担。深度分离卷积是常用的降低计算量的一种方法，然而这种方法并不兼容当前的工业 IC 设计。

**回顾深度分离卷积**

深度分离卷积 = Depthwise Convolution + Pointwise Convolution

Depthwise Conv: $(B, C_1, H, W) \stackrel{(1, C_1, k, k)}\longrightarrow (B, C_1, H, W)$

Pointwise Conv: $(B, C_1, H, W) \stackrel{(C_1, C_2, 1, 1)}\longrightarrow (B,C_2, H, W)$

卷积参数量为 $C_1 \times (k^2 + C_2)$

而普通的卷积 $(B, C_1, H, W) \stackrel{(C_1, C_2, k, k)}\longrightarrow (B, C_2, H, W)$ 参数量为 $C_1 \times C_2 \times k^2$ 。


研究团队介绍了 Cross Stage Partial Network（CSPNet），目的是为了降低计算量的同时能获得丰富的梯度，实现方式是：将 base layer 的特征分为两个部分，然后经过一个 cross-stage 层级后再进行融合。CSPNet 处理以下三个问题：
1. 加强 CNN 的学习能力

    现有的 CNN 在轻量化后准确性也下降了。故研究团队希望可以增强 CNN 的学习能力，使得轻量化后依然维持足够高的准确率。CSPNet 可以应用于 ResNet，ResNeXt 以及 DenseNet 等。

2. 去除计算瓶颈

    给 CNN 中每个 layer 分配相当的计算量。

3. 降低内存消耗

# 2. 方法

## 2.1 CSPNet

图 1 是 DenseNet 的结构（仅显示了一个 stage），

![](/images/obj_det/CSPNet_1.png)

<center>图 1. DenseNet 的 one-stage</center>

DenseNet 中每个 stage 包含一个 dense block 和一个 transition layer，dense block 由 k 个 dense layers 组成。第 `i` 个 dense layer 的输出与其输入进行 concatenate，作为第 `i+1` 个 dense layer 的输入，数学表示如下，

$$\mathbf x_1=\mathbf w_1\star \mathbf x_0
\\\\ \mathbf x_2=\mathbf w_2 \star [\mathbf x_0, \mathbf x_1]
\\\\ \vdots
\\\\ \mathbf x_k=\mathbf w_k \star [\mathbf x_0, \ldots,\mathbf x_{k-1}]$$


反向传播梯度时，参数更新写为

$$\mathbf w_1'=f(\mathbf w_1, \mathbf g_0)
\\\\ \mathbf w_2'=f(\mathbf w_2, \mathbf g_0, \mathbf g_1)
\\\\ \vdots
\\\\ \mathbf w_k'=f(\mathbf w_k, \mathbf g_0, \ldots, \mathbf g_{k-1})$$

其中 $\mathbf g_i$ 为第 `i` 个 dense layer 的梯度值。作者给出结论：大量的梯度信息在更新参数时被重复使用，不同的 dense layer 重复学习相同的梯度信息。

注意：这里的 $\mathbf g_i, \ i=0,1,\ldots, k-1$ 全部由 `k-th` layer 生成的梯度，这些梯度分别反向传播到 `i-th` layer。

## 2.2 CSPDenseNet
图 2 是在 DenseNet 上使用 CSP 结构，

![](/images/obj_det/CSPNet_2.png)

<center>图 2. CSPDenseNet</center>

CSPDenseNet 的一个 stage 是由一个分部 dense block 和一个分部 transition layer 组成。在分部 dense block 中，输入沿通道切分为两部分 $x_0=[x_0', x_0^{"}]$，其中前者 $x_0'$ 直接链接到这个 stage 的尾部，后者  $x_0^{"}$ 则穿过 dense block 和 transition layer，然后与  $x_0'$ 进行 concatenate 并作为下一 stage 的输入。

分部 transition layer 中的执行步骤：

1. dense block 的输出 $[x_0^{"} ,x_1,\ldots, x_k]$ 经过一个转移层
2. 第 `1` 步中转移层的输出 $x_T$ 与 $x_0'$ 进行 concatenate，然后再经过一个转移层，输出 $x_U$

$$\frac {\partial \mathbf w_{k\_i} \star \mathbf x_i}{\partial \mathbf x_i}$$

