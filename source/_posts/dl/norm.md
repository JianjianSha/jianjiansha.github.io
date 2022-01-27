---
title: Normalization
date: 2021-03-08 11:26:46
tags: deep learning
---

# Batch Norm
对 channel 之外的所有维度做归一化，即在 `(B,H,W)` 上做归一化，每个 channel 独立进行，
$$y=\frac {x-E[x]} {\sqrt{Var[x]+\epsilon}} \cdot \gamma + \beta$$
作用：
1. 防止过拟合。单个样本的输出依赖于整个 mini-batch，防止对某个样本过拟合
2. 加快收敛。梯度下降过程中，每一层的 W 和 b 都会不断变化，导致输出结果分布也不断变化，后层网络需要不停地适应这种变化，而 BN 可使得每一层输入分布近似不变。
3. 防止梯度消失。以 sigmoid 激活函数为例，经过 BN 使得输出在中心附近，梯度较大。

# Layer Norm
Layer Norm 对每个样本进行归一化，即在 `(C,H,W)` 上做归一化，每个样本独立进行。

# Instance Norm
Instance Norm 对每个样本的每个 channel 进行归一化，即在 `(H,W)` 上做归一化。

# Group Norm
与 Layer Norm 类似，不同的是 Group Norm 将 `(C,H,W)` 在 channel 上分组，假设分为 `G` 组，那么在 `(C/G, H, W)` 上做归一化。

