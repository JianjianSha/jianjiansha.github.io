---
title: Normalization
date: 2021-03-08 11:26:46
tags: deep learning
categories: 深度学习
p: dl/norm
---

# 1. Batch Norm
对 channel 之外的所有维度做归一化，即在 `(B,H,W)` 上做归一化，每个 channel 独立进行，
$$y=\frac {x-E[x]} {\sqrt{Var[x]+\epsilon}} \cdot \gamma + \beta$$

其中 $\gamma, \beta$ 是需要学习的参数。

作用：
1. 防止过拟合。单个样本的输出依赖于整个 mini-batch，防止对某个样本过拟合
2. 加快收敛。梯度下降过程中，每一层的 W 和 b 都会不断变化，导致输出结果分布也不断变化，后层网络需要不停地适应这种变化，而 BN 可使得每一层输入分布近似不变。
3. 防止梯度消失。以 sigmoid 激活函数为例，经过 BN 使得输出在中心附近，梯度较大。

# 2. Layer Norm
Layer Norm 对每个样本进行归一化，即在 `(C,H,W)` 上做归一化，每个样本独立进行。

# 3. Instance Norm
Instance Norm 对每个样本的每个 channel 进行归一化，即在 `(H,W)` 上做归一化。

# 4. Group Norm
与 Layer Norm 类似，不同的是 Group Norm 将 `(C,H,W)` 在 channel 上分组，假设分为 `G` 组，那么在 `(C/G, H, W)` 上做归一化。

# 5. Weight Norm

论文：[Weight normalization: A simple reparameterization to accelerate training of deep neural networks](https://arxiv.org/abs/1602.07868)

神经网络的一个节点计算为

$$y=\phi(\mathbf w\cdot \mathbf x+b) \tag{5-1}$$

其中参数 $\mathbf w$ 可解耦为一个标量和一个方向向量，

$$\mathbf w=\frac g {||\mathbf v||} \mathbf v \tag{5-2}$$

使得 $\mathbf w$ 的欧氏范数等于 $g$，与 $\mathbf v$ 无关，仅方向与 $\mathbf v$ 相关，给了 $\mathbf v$ 更多的自由度。

损失关于 $g$ 的梯度为

$$\nabla_g L=\frac {\nabla_{\mathbf w}L \cdot \mathbf v}{||\mathbf v||} \tag{3}$$

由于 

$$\mathbf w = g (\mathbf v^{\top}\mathbf v)^{-1/2} \mathbf v$$

故有

$$\begin{aligned}\frac {\mathbf w}{d\mathbf v}&=g(\mathbf v^{\top}\mathbf v)^{-1/2}\frac {d\mathbf v}{d\mathbf v} +\mathbf v \frac {d(g(\mathbf v^{\top}\mathbf v)^{-1/2})}{d\mathbf v}
\\&=g(\mathbf v^{\top}\mathbf v)^{-1/2}I+\mathbf v(-g (\mathbf v^{\top}\mathbf v)^{-3/2} \mathbf v^{\top})
\\&=\frac g {||\mathbf v||} I - \frac g {||\mathbf v||^3} \mathbf v \mathbf v^{\top}
\end{aligned}$$

损失关于 $\mathbf v$ 的梯度为

$$\begin{aligned}\nabla_{\mathbf v}L&=\nabla_{\mathbf w}L \left(\frac g {||\mathbf v||}- g \mathbf v \mathbf v^{\top}\right)
\\&=\frac g {||\mathbf v||}\nabla_{\mathbf w}L- \frac g {||\mathbf v||^3}(\nabla_{\mathbf w}L )\mathbf v \mathbf v^{\top}
\\&=\frac g {||\mathbf v||}\nabla_{\mathbf w}L-\frac {g\nabla_g L} {||\mathbf v||^2} \mathbf v
\end{aligned}$$

实现时不直接更新 $g$，因为 $g \ge 0$ 这个必要条件的限制，我们改用 $g$ 的 log-scale 即 $s=\log g$ 进行更新，这样 $s \in \mathbb R$ 的更新更加直接。

