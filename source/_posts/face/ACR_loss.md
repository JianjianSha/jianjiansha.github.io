---
title: 'ACR Loss: Adaptive Coordinate-based Regression Loss for Face Alignment'
date: 2023-08-22 11:01:39
tags: face alignment
mathjax: true
---

论文：[ACR Loss: Adaptive Coordinate-based Regression Loss for Face Alignment](https://arXiv.org/abs/2203.15835)

# 1. 介绍

本文介绍了一种新的 loss，使得 coordinate-based regression 的 landmark 检测更加准确。

## 1.1 现有损失的不足

现有的损失如，计算预测 landmark points 与 gt points 的 L1 和 L2 值。

L2 损失为 $y = x ^ 2$，梯度幅度为 $|x|$，对于一个 batch 数据，优化 $M \times b$（$M, \ b$ 分别为 landmark 点数和 batch size）个 landmark 点数，那么误差大的 landmark points 更加被注意到，从而得到优化，而小的误差的点则容易被忽视，得不到充分优化。

L1 损失为 $y = |x|$，梯度幅度为 $1$，虽然避免了 L2 损失中的问题，但是梯度服务较小，尤其是训练初期，导致训练步数很多。

## 1.2 ACR loss

ACR loss 定义： $y = \log (|x| ^ {2 - \Phi}+1)$， 其中 $\Phi \in [0, 1]$ 。

对于具有挑战性的点，设置较大的 $\Phi$ 值，对于不太具有挑战性的点，设置较小的 $\Phi$ 值，通过调节 $\Phi$ 值的大小，控制 loss 。如图 1，$\Phi$ 越大，ACR loss 越大；反之则越小。

![](/images/face/ACR_loss_1.png)
<center>图 1.</center>

需要制定一个指标衡量一个 landmark point 的挑战性，从而确定 $\Phi$ 的值。

对于一个 image，使用一个 M-dim 的向量表示一个 Face。根据 ASM（Active Shape Model），有以下关系，

$$\text{Face} _ {M \times 1} \approx \text{Mean\_Face} _ {M \times 1} + \mathbf V _ {M \times M} \mathbf b _ {M \times 1} \tag{1}$$

其中 $M$ 是 landmark pionts 数量，$\text{Face} _ {M \times 1}$ 是 image 中所有点的 $(x, y)$ 均值？

$\text {Mean\_Face} _ {M \times 1}$ 是对整个训练集的 $\text{Face} _ {M \times 1}$ 求均值，$\mathbf V =\lbrace v _ 1, \ldots, v _ M \rbrace$ 是训练集中随机向量的协方差矩阵的 $M$ 个特征向量。

单个 image 中 $M$ 个点的 $(x, y)$ 均值构成一个 M-dim 的向量，将其看作是随机向量，那么其期望就是 $\text {Mean\_Face} _ {M \times 1}$，其协方差矩阵 size 为 $M \times M$ ，且是实对称矩阵，所以其特征向量之间正交（且归一化）。

根据 (1) 式可以计算出

$$\mathbf b _ {M \times 1}=\mathbf V _ {k \times M} ^ {\top} (\text{Face} _ {M \times 1} - \text{Mean\_Face} _ {M \times 1}) \tag{2}$$

根据 (2) 式，显然 $b _ i=v _ i ^ {\top}(\text{Face} _ {M \times 1} - \text{Mean\_Face} _ {M \times 1})$，即，$\mathbf b$ 的第 `i` 个元素值与第 `i` 个特征向量相关。

定义 Smooth_Face 如下，

$$\text{Smooth\_Face} _ {M \times 1} = \text{Mean\_Face} _ {M \times 1} + \mathbf V _ {M \times k} \mathbf b _ {k \times 1} \tag{3}$$

其中 $k \in [0, M]$ ，所以 $\text{Smooth\_Face} _ {M \times 1}$ 其实是 $\text{Face} _ {M \times 1}$ 的近似。

选定 $k$ 值之后，那么可以计算每个 landmark point 的 $\Phi$ 值，并以此值表征 landmark point 的挑战性。定义 

$$\Phi _ {i,m} = \frac {|\text{Smooth\_Face} _ {i,m}-\text{Face} _ {i,m}|}{\max (|\text{Smooth\_Face} _ {i,q}-\text{Face} _ {i,q}|), \ \forall q \in [1,M]} \tag{4}$$

其中下标 $m$ 表示 landmark point index，$i$ 表示训练集中 image index。

ACR loss 按如下式定义，

$$\Delta _ {i,m} = |\text{Face} _ {i,m} - \text{Pr\_Face} _ {i,m}| \tag{5}$$

$$\text{loss\_face} _ {i,m} = \begin{cases} \lambda \log (1+\Delta _ {i,m} ^ {2-\Phi _ {i,m}}) & \Delta _ {i,m} \le 1 \\\\ \Delta _ {i,m} ^ 2 +C & \Delta _ {i,m} > 1 \end{cases} \tag{6}$$

$$L _ {ACR} = \frac 1 {MN} \sum _ {i=1} ^ N \sum _ {m=1} ^ M \text{loss\_face} _ {i,m} \tag{7}$$

其中 $N$ 是训练集大小，$\text{Pr\_Face} _ {i,m}$ 是模型预测输出。$C = \Phi _ {i,m} \log 2 - 1$ 。$\lambda$ 作为超参数，用于调节 ACR loss 曲率，如图 1 。

**小结**

$\Phi$ 值表明了 points 的挑战性测度。

对于较低挑战性 points，$\Phi _ {i,m}$ 接近 0，相应的损失幅度以及梯度都随着误差 $\Delta _ {i,m}$ 减小而减小。

对于具有挑战性的 points，$\Phi _ {i,m}$ 接近 1，损失梯度随着误差减小而增大 。

最为对比，L2 损失则仅仅依赖误差，L2 损失对小误差相对大误差并不敏感 ，因为损失梯度随着误差减小而减小，这对于具有挑战性的 points 而言，很难被模型学习到。

**$k$ 值选择策略**

$k$ 值决定了 smooth face 与 gt face 之间的相似度，$k$ 越小，那么 smooth face 越接近 mean face，$k$ 越大，那么 smooth face 越接近 gt face 。

随着训练的进行，模型预测准确性提升，需要增大 $k$ 使得 smooth face 接近 gt face，这样 $\Phi _ {i,m}$ 更能表征具有挑战性的 points 。

根据经验，

1. epochs 0 ~ 15 期间，选择 $k = 80\% M$
2. epochs 16 ~ 30 期间，选择 $k=85\% M$
3. epochs 31 ~ 70 期间，选择 $k=90\% M$
4. epochs 71 ~ 100 期间，选择 $k=95\% M$
5. epochs 101 ~ 150 期间，选择 $k=97\% M$

$k$ 不能等于 $M$，否则 $\Phi=NaN$