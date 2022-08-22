---
title: 一种用于密度和得分估计的可扩展方法
date: 2022-07-09 16:22:28
tags:
    - scored based model
    - generative model
mathjax: true
---

论文：[Sliced Score Matching: A Scalable Approach to Density and Score Estimation](https://arxiv.org/abs/1905.07088)

[Score Matching](/2022/07/09/diffusion_model/score_match) 方法用于学习非归一化统计模型（例如 energy-based 模型）的概率密度，核心思想是最小化模型 score 和真实数据 score 距离（score：对数概率密度的导数），优点是可以不用考虑无法求解析式的归一化因子，缺点是要计算模型对数概率密度的 Hessian 矩阵（对参数的二阶导矩阵）的对角线元素之和，也就是 Hessian 矩阵的 trace，在遇到复杂模型时，Hessian 矩阵的 trace 计算代价较大。

为此，作者提出 sliced score matching，可以将 score matching 方法扩展导深度非归一化模型和高维数据上。

# 1. 背景

观测数据为 $\mathbf x_{1:N}$，数据真实分布为 $p_d(\mathbf x)$，我们的任务是学习非归一化概率密度 $\tilde p_m(\mathbf x;\theta)$，其中 $\theta$ 是模型参数，配分函数记为 $Z_{\theta}$，这个归一化项难以计算出解析式，记模型归一化后的概率密度为

$$p_m(\mathbf x;\theta)=\frac {\tilde p_m(\mathbf x;\theta)}{Z_{\theta}}, \quad Z_{\theta}=\int \tilde p_m(\mathbf x; \theta) d\mathbf x \tag{1}$$

为了表示方便，记 $p_m, \ p_d$ 的得分函数分别为 $\mathbf s_m(\mathbf x;\theta) \stackrel{\Delta}= \nabla_{\mathbf x} \log p_m(\mathbf x;\theta)$，$\mathbf s_d(\mathbf x) \stackrel{\Delta}= \nabla_{\mathbf x} \log p_d(\mathbf x)$ 。

下面进行一个简单的回顾。

## 1.1 Score Matching

常见的损失函数如 负对数似然 $-\mathbb E_{p_d} [\log p_m]$ 或者 KL 散度 $KL(p_d||p_m)=\mathbb E_{p_d} [\log p_d - \log p_m]$ 均面临理配分函数 $Z_{\theta}$ 难以处理的问题。

使用得分函数的距离差来表示损失，

$$L(\theta) \stackrel{\Delta}= \frac 1 2 \mathbb E_{p_d} [\|\mathbf s_m(\mathbf x;\theta) - \mathbf s_d(\mathbf x) \|^2] \tag{2}$$


由于得分函数 

$$\mathbf s_m(\mathbf x;\theta) \stackrel{\Delta}= \nabla_{\mathbf x} \log p_m(\mathbf x;\theta)=\nabla_{\mathbf x} \log \tilde p_m(\mathbf x; \theta)$$

其中 $\log Z_{\theta}$ 与 $\mathbf x$ 无关，所以导数为 0，于是 (2) 式中不含有配分函数 $Z_{\theta}$ 。

根据论文 [Score Matching](/2022/07/09/diffusion_model/score_match) 中的定理 1，(2) 式可变为 $L(\theta)=J(\theta)+C$，其中 $C$ 是与参数 $\theta$ 无关的项，且

$$J(\theta) \stackrel{\Delta}= \mathbb E_{p_d} \left [tr (\nabla_{\mathbf x} \mathbf s_m(\mathbf x;\theta)) + \frac 1 2 \|\mathbf s_m(\mathbf x;\theta)\|^2 \right] \tag{3}$$

$$\nabla_{\mathbf x} \mathbf s_m(\mathbf x;\theta)=\nabla_{\mathbf x}^2 \log \tilde p_m(\mathbf x;\theta)$$

那么根据观测数据得到目标损失为

$$\hat J(\theta;\mathbf x_{1:N}) \stackrel{\Delta}= \frac 1 N \sum_{i=1}^N \left[tr (\nabla_{\mathbf x} \mathbf s_m(\mathbf x_i;\theta)) + \frac 1 2 \|\mathbf s_m(\mathbf x_i;\theta)\|^2\right] \tag{4}$$

其中 $\mathbf s_m(\mathbf x_i;\theta)=\nabla_{\mathbf x} \log \tilde p_m(\mathbf x;\theta)$ 在 $\mathbf x=\mathbf x_i$ 处的值，$\nabla_{\mathbf x} \mathbf s_m(\mathbf x_i;\theta)=\nabla_{\mathbf x} \mathbf s_m(\mathbf x;\theta)$ 在 $\mathbf x=\mathbf x_i$ 处的值。


## 1.2 用于隐式分布

score matching 除了可用于非归一化模型，也可以用于估计隐式分布，即概率分布没有解析式，但是可以对其进行采样，例如 GAN 生成器的输出就是一个隐式分布。

对于一个隐式概率密度 $q_{\theta}(\mathbf x)$，得分函数 $\mathbf s_q(\mathbf x)=\nabla_{\mathbf x} q_{\theta}(\mathbf x)$ 。假设 $\mathbf x \sim q_{\theta}(\theta)$ 可以重参数为 $\mathbf x=g_{\theta}(\epsilon)$，其中 $\epsilon$ 是一个简单与 $\theta$ 无关的随机变量，例如标准高斯分布变量，$g_{\theta}(\cdot)$ 是一个确定性映射（不含随机过程）。


# 2. 使用 Sliced Score Matching 的密度估计

## 2.1 Sliced Score Matching

考虑导数据维度较高，将 $\mathbf s_d(\mathbf x), \ \mathbf s_m(\mathbf x;\theta)$ 投影到某个随机方向 $\mathbf v$ 上，然后比较投影之差。

于是 (2) 式变为

$$L(\theta; p_{\mathbf v}) \stackrel{\Delta}= \frac 1 2 \mathbb E_{p_{\mathbf v}} \mathbb E_{p_d} [(\mathbf v^{\top} \mathbf s_m(\mathbf x;\theta) - \mathbf v^{\top} \mathbf s_d(\mathbf x))^2] \tag{5}$$

这里 $\mathbf v \sim p_{\mathbf v}$ 是与 $\mathbf x$ 独立的随机变量，并且我们要求 $\mathbb E_{p_{\mathbf v}}[ \mathbf v \mathbf v^{\top}]\succ 0$，且 $\mathbb E_{p_{\mathbf v}}[\|\mathbf v\|^2] < \infty$ ，前者表示 $\mathbf v=\mathbf 0$ 概率（不是概率密度）为 0，后者表示 $\mathbf v$ 范数为有限值的概率为 1。满足这个要求的分布有很多，例如 多元标准整体分布 $\mathcal N(\mathbf 0, I_D)$，多元 Rademacher 分布（$\{\pm 1\}^D$ 上的均匀分布），超球面上 $\mathbb S^{D-1}$ 上的均匀分布，其中 $D$ 是随机变量 $\mathbf x$ 的维度。

**定理 1**

假设 1（得分函数的正则）：模型得分函数 $\mathbf s_m(\mathbf x)$ 和数据得分函数 $\mathbf s_d(\mathbf x)$ 均可微，且满足 $\mathbb E_{p_d}[\|\mathbf s_m(\mathbf x)\|^2]< \infty$，$\mathbb E_{p_d}[\|\mathbf s_d(\mathbf x)\|^2]< \infty$

假设 2 （投影向量的正则）；$\mathbb E_{p_{\mathbf v}}[ \mathbf v \mathbf v^{\top}]\succ 0$，且 $\mathbb E_{p_{\mathbf v}}[\|\mathbf v\|^2] < \infty$

假设 3 （边界条件）： $\forall \theta \in \Theta$， $\lim_{\|\mathbf x\|\rightarrow \infty} \mathbf s_m(\mathbf x; \theta) p_d(\mathbf x)=0$

如果满足以上三个假设条件，那么

$$L(\theta;p_{\mathbf v}) \stackrel{\Delta}=\frac 1 2 \mathbb E_{p_{\mathbf v}} \mathbb E_{p_d} [(\mathbf v^{\top} \mathbf s_m(\mathbf x;\theta) - \mathbf v^{\top} \mathbf s_d(\mathbf x))^2]=J(\theta)+C \tag{6}$$

其中 

$$J(\theta)=\mathbb E_{p_{\mathbf v}} \mathbb E_{p_d} [\mathbf v^{\top} \nabla_{\mathbf x} \mathbf s_m (\mathbf x;\theta) \mathbf v + \frac 1 2 (\mathbf v^{\top} \mathbf s_m(\mathbf x;\theta))^2]+C \tag{7}$$

其中 $C$ 是与参数 $\theta$ 无关的常量。

证明与论文 [Score Matching](/2022/07/09/diffusion_model/score_match) 中的定理 1 的证明过程类似，可以查看论文附录，这里不再赘述。


实际应用中，给定观测数据集 $\mathbf x_{1:N}$，每个数据均从 $p_{\mathbf v}$ 独立地采样 $M$ 的投影向量，所有的投影向量记为 $\{\mathbf v_{ij} \}_{1\le i \le N, 1\le j \le M}$，简记为 $\mathbf v_{11:NM}$，那么 (6) 式变为

$$\hat J(\theta; \mathbf x_{1:N}, \mathbf v_{11:NM})=\frac 1 N \frac 1 M \sum_{i=1}^N \sum_{j=1}^M \mathbf v_{ij}^{\top} \nabla_{\mathbf x} \mathbf s_m(\mathbf x_i;\theta) \mathbf v_{ij} + \frac 1 2 (\mathbf v_{ij}^{\top}\mathbf s_m(\mathbf x_i; \theta))^2 \tag{8}$$

### 2.1.1 variance reduction

当 $p_{\mathbf v}$ 是多元标准高斯分布或者 多元 Rademacher 分布，那么

$$\begin{aligned}\mathbb E_{p_{\mathbf v}} [(\mathbf v^{\top} \mathbf s_m(\mathbf x;\theta))^2]&=\mathbf s_m^{\top}(\mathbf x;\theta) \mathbb E_{p_{\mathbf v}} [\mathbf v \mathbf v^{\top}] \mathbf s_m(\mathbf x;\theta) \\&= \mathbf s_m^{\top}(\mathbf x;\theta) (\mathbb V[\mathbf v]+ \mathbb E[\mathbf v] \mathbb E^{\top}[\mathbf v]) \mathbf s_m(\mathbf x;\theta)\\&= \mathbf s_m^{\top}(\mathbf x;\theta) \mathbf s_m(\mathbf x;\theta) \end{aligned}\tag{9}$$

于是 (8) 式变为 

$$\hat J_{vr}(\theta; \mathbf x_{1:N}, \mathbf v_{11:NM})=\frac 1 N \frac 1 M \sum_{i=1}^N \sum_{j=1}^M \mathbf v_{ij}^{\top} \nabla_{\mathbf x} \mathbf s_m(\mathbf x_i;\theta) \mathbf v_{ij} + \frac 1 2 \|\mathbf s_m(\mathbf x_i;\theta)\|^2 \tag{10}$$

称 $\hat J_{vr}$ 为 sliced score matching with variance reduction (SSM-VR) 。


### 2.1.2 计算复杂度分析

对于 sliced score matching (SSM)，二阶导 $\mathbf v^{\top} \nabla_{\mathbf x} \mathbf s_m(\mathbf x;\theta) \mathbf v$ 相较于 $tr(\nabla_{\mathbf x} \mathbf s_m(\mathbf x;\theta))$ 计算复杂度大大降低。这是因为，使用支持自动高阶求导系统（如 PyTorch），对每个 $\mathbf v$ 只需要计算两次求导操作以便得到二阶导，第一次是求一阶导得到 $\mathbf s_m(\mathbf x;\theta)$，这与 $\mathbf v$ 无关，第二次是求导则利用了 $\mathbf v$，即计算一个标量对向量的导数，如算法 1 所示，所以每个数据 $\mathbf x_i$，求导操作只需要 $1+M$ 次求导操作。如果直接计算 $tr(\nabla_{\mathbf x} \mathbf s_m(\mathbf x;\theta))$，那么需要 $1+D$ 次求导操作，因为二阶导矩阵 $\nabla_{\mathbf x} \mathbf s_m(\mathbf x;\theta)$ 的对角线元素需要单独计算，如算法 2 。

---
**算法 1** Sliced Score Matching
**输入：** $\tilde p_m(\cdot;\theta), \ \mathbf x, \ \mathbf v$
**输出：** $\hat J(\theta, \mathbf x, \mathbf v)$

1. 计算得分函数（一阶导）： $\mathbf s_m(\mathbf x;\theta) \leftarrow \text{grad}(\log \tilde p_m(\mathbf x;\theta), \mathbf x)$

2. $\mathbf v^{\top} \nabla_{\mathbf x} \mathbf s_m(\mathbf x;\theta) \leftarrow \text{grad}(\mathbf v^{\top} \mathbf s_m(\mathbf x;\theta), \mathbf x)$

3. $J \leftarrow \frac 1 2 (\mathbf v^{\top} \mathbf s_m(\mathbf x;\theta))^2$

4. $J \leftarrow J+ \mathbf v^{\top} \nabla_{\mathbf x} \mathbf s_m(\mathbf x;\theta) \mathbf v$

5. **return** $J$

---

算法 1 中，第 `1` 步对某个数据 $\mathbf x$ 只需要计算 1 次，第 `2` 步对数据 $\mathbf x$ 需要计算 $M$ 次，因为有 $M$ 个投影向量，故针对单个数据 $\mathbf x$一共计算 $1+M$ 次求导。

---
**算法 2** Score Matching

**输入：** $\tilde p_m(\cdot;\theta), \ \mathbf x$
**输出：** $\hat J(\theta,\mathbf x)$

1. 计算得分函数（一阶导）： $\mathbf s_m(\mathbf x;\theta) \leftarrow \text{grad}(\log \tilde p_m(\mathbf x;\theta), \mathbf x)$

2. $J \leftarrow \frac 1 2 \|\mathbf s_m(\mathbf x;\theta)\|^2$

3. **for** $d = 1,\ldots, D$ **do**
4.  &emsp; $(\nabla_{\mathbf x} \mathbf s_m(\mathbf x;\theta))_d \leftarrow \text{grad}((\mathbf s_m(\mathbf x;\theta))_d, \mathbf x)_d$
5. &emsp; $J \leftarrow J + (\nabla_{\mathbf x} \mathbf s_m(\mathbf x;\theta))_d$
6. **end for**

7. **return** $J$

---

算法 2 中，第 `1` 步对某个数据 $\mathbf x$ 计算 1 次，第 `4` 步对数据 $\mathbf x$ 需要计算 $D$ 次，因为 $\mathbf s_m(\mathbf x, \theta)$ 和 $\mathbf x$ 均为向量，Pytorch 不能计算 向量对向量的导数，分为 $D$ 次计算，$d \mathbf s_m(\mathbf x;\theta)_d/ d\mathbf x$ 是一个 $D$ 维向量，由于要求 Hessian 矩阵的 trace，需要计算其对角线元素，取向量 $d \mathbf s_m(\mathbf x;\theta)_d/ d\mathbf x$ 的第 `d` 个元素，即 第 `4` 步。故针对单个数据 $\mathbf x$ 一共需要求导 $1+D$ 次。


如果 $D \gg M$，那么使用 sliced score matching 可以大大降低计算复杂度。实际应用中，可以调整 $M$ 值以便在计算复杂度和目标值 $J$ 的方差之间取得一个平衡。作者实验中发现 $M=1$ 效果不错。

## 2.2 Sliced Score Estimation

作者提出使用神经网络模型 $\mathbf h(\mathbf x;\theta): \mathbb R^D \rightarrow \mathbb R^D$ 来模拟得分函数 $\nabla_{\mathbf x} \log q(\mathbf x)$。数据集 $\mathbf x_{1:N} \stackrel{i.i.d}\sim q(\mathbf x)$，这里使用 $q(\mathbf x)$ 是为了与前面非归一化 $\tilde p(\mathbf x)$ 区分。根据 (8) 式可知，需要最小化目标函数

$$\frac 1 N \frac 1 M \sum_{i=1}^N \sum_{j=1}^M \mathbf v_{ij}^{\top} \nabla_{\mathbf x} \mathbf h(\mathbf x_i; \theta) \mathbf v_{ij} + \frac 1 2 (\mathbf v_{ij}^{\top} \mathbf h(\mathbf x_i;\theta))^2 \tag{11}$$

得到最优参数值记为 $\hat \theta$，于是 $\mathbf h(\mathbf x;\hat \theta)$ 可用于估计 $\nabla_{\mathbf x} \log q(\mathbf x)$ 。


