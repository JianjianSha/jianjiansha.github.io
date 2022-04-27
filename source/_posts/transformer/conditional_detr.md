---
title: Conditional DETR 解读
date: 2022-04-22 16:27:24
tags: transformer
mathjax: true
summary: 引入 conditional spatial queries 提高模型定位性能以及加快训练过程
---

论文：[Conditional DETR for Fast Training Convergence](https://arxiv.org/abs/2108.06152)
源码：[Atten4Vis/ConditionalDETR](https://github.com/Atten4Vis/ConditionalDETR)

# 1. 简介

DETR 将 transformer 引入目标检测任务中，实现一种 anchor free 且无需 post-processing（例如 NMS），但是 DETR 的缺点是训练太慢。

作者通过研究发现 DETR decoder 的 cross-attention（第二个 attention） 的权重很难准确的定位到目标 box 的 extremities（上下左右），如图 1，

![](/images/transformer/conditional_detr1.png)

<center>图 1. cross-attention 的 attention weight maps</center>

图 1 中：

第一行是 conditional DETR-R50 训练 50 次后的 attention 权重 map， `spatial keys` 与 `spatial queries` 之间的点积 $\mathbf p_q^{\top} \cdot \mathbf p_k$ 的 soft-max 归一化

第二行是原生 DETR-50 训练 50 次后的 attention 权重 map，`query` 与 `spatial key` 之间的点积 $(\mathbf o_q + \mathbf c_q)^{\top} \mathbf p_k$ 的 soft-max 归一化。这里，$\mathbf o_q$ 是可学习的 object queries。

由于 $\mathbf p_q$ 在本文 conditional DETR 中由上一层的 `content embedding` 学习得到，而原生 DETR 中 $\mathbf o_q$ 没有包含 content 信息，故原生 DETR 中使用 $(\mathbf o_q + \mathbf c_q)$ 即 `query` 来对应 conditional DETR 中的 `spatial query`。（如果对部分变量不太了解，可以阅读完下文后再回头看看。）

第三行是原生 DETR-50 训练 500 次后的 attention 权重 map。

可见原生 DETR 在训练 50 时，目标 box 的左右边缘并不能被 attention weight 很好的表征。


# 2. Conditional DETR

图 2 是 conditional DETR 中一个 decoder layer 的结构图，

![](/images/transformer/conditional_detr2.png)

<center>图 2</center>

`layer`：Decoder 是由若干个 layer 堆叠（stack）组成。每个 layer 包含三个部分：
    - `self-attention`
    - `cross-attention`
    - `FFN`

## 2.1 cross-attention
图 2 中 `cross-attention` 的 query 和 key 均不采用相加，而是 concatenate 操作。

query = [content query, spatial query]
key   = [content key, spatial key]
value = [content value]

变量说明：

1. `content query`($\mathbf c_q$) ：layer 中 `self-attention` 的输出。 
2. `spatial query`($\mathbf p_q$) ：原生 DETR 中采用一个可学习的 object query，在 conditional DETR 中，这个变量由两个因素决定：
    - decoder embedding($\mathbf f$)
    - reference point($\mathbf s$)
3. `content key`($\mathbf c_k$) ：Encoder 的输出
4. `spatial key`($\mathbf p_k$) ：(sine) Position Embedding
5. `content value`：Encoder 的输出
6. `decoder embedding`($\mathbf f$) ：decoder layer 的输出
7. `content embedding`: decoder layer 的一个（content）输入

cross-attention weight 为

$$\mathbf c_q^{\top} \mathbf c_k+\mathbf p_q^{\top} \mathbf p_k$$

将第一项看作是 content attention，第二项为 spatial attention。

`decoder embedding` 包含了目标 box 与参考点 reference point 之间的位移信息，首先将 `decoder embedding` 映射，然后与参考点相加，得到非归一化位置，那么预测 box 为

$$\mathbf b = \text{sigmoid}(FFN(\mathbf f)+[\mathbf s^{\top} \ 0 \ 0]^{\top})$$

其中 $\mathbf b = [b_{cx}, b_{cy}, b_w, b_h]^{\top}$

`decoder embedding` 还包含了四个 extremities 所含区域信息，以此用来预测分类得分，

$$\mathbf e=FFN(\mathbf f)$$

## 2.2 Conditional spatial query

cross-attention 的 `spatial query` $\mathbf p_q$ 由两个因素决定：

$$(\mathbf s,\mathbf f) \rightarrow \mathbf p_q$$

先将 reference point $\mathbf s$ 从 2d 映射到 256d，

$$\mathbf p_s = \text{sinusoidal(sigmoid}(\mathbf s))$$

然后将 $\mathbf f$ 经过一个 FFN 变换，这个 FFN 为：`FC+ReLU+FC`

$$T = FFN(\mathbf f)$$

这里将 256d 的向量 $\lambda_q$ 转为对角矩阵 $T$，这样做矩阵相乘，即可得到向量内积，且计算效率更高，

$$\mathbf p_q = T \mathbf p_s = \lambda_q \cdot \mathbf p_s$$

## 2.3 reference point

原生 DETR 中，$\mathbf s = [0 \ 0]^{\top}$。

conditional DETR 中，作者研究了两种生成 reference point $\mathbf s$ 的方法：

1. 将 $\mathbf s$ 看作是可学习的参数
2. 使用 object queries 预测得到，

    $$\mathbf s=FFN(\mathbf o_q)$$

    其中 FFN 为 `FC+ReLU+FC`。

## 2.4 Loss

根据原生 DETR 一样，使用预测和 gt 之间的二分类匹配（匈牙利算法），然后根据匹配结果计算用于反向传播的损失函数。

损失函数与 deformable DETR 中一样：相同的匹配损失函数，相同的 $N=300$ object queries 的损失函数 (L1+GIoU)，相同的 trade-off 参数（各类型损失的平衡因子）。分类损失使用 focal loss。