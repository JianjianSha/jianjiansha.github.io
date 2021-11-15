---
title: 隐马尔可夫模型的参数学习
date: 2021-11-01 14:53:22
tags: machine learning
mathjax: true
p: ml/hmm_learn
---
本文根据最大似然法则求模型参数：状态转移矩阵 $A$ 和发射矩阵 $B$（初始状态概率 $\pi$ 被包含在 $A$ 中，即 $\pi=A_{1,2:}$，下标从 1 开始计算）。
<!--more-->
# 监督学习
所谓有监督，指对于观测序列，有对应的（人工标注）状态序列。记某个样本为 $(\mathbf x, \mathbf z)$，包括观测序列 $\mathbf x$ 和状态序列 $\mathbf z$，那么其概率为

$$\begin{aligned}P(\mathbf x, \mathbf z;A, B)&=\prod_{t=1}^T P(z_t|z_{t-1};A)P(x_t|z_t;B)
\\&=\prod_{t=1}^T A_{z_{t-1}z_t} B_{z_tx_t}
\end{aligned}$$

取对数，

$$\log P = \sum_{i=1}^{|S|} \sum_{j=1}^{|S|} \sum_{t=1}^T \mathbb I[z_{t-1}=s_i \land z_t =s_j]\log A_{ij}+\sum_{j=1}^{|S|}\sum_{k=1}^{|V|}\sum_{t=1}^T \mathbb I[z_t=s_j \land x_t=v_k]\log B_{jk}$$


对训练集 $\mathcal D =(\mathbf x^{(1)},\mathbf z^{(1)}), \cdots, (\mathbf x^{(m)}, \mathbf z^{(m)})$，对数似然函数为

$$L=\sum_{l=1}^m \sum_{i=1}^{|S|} \sum_{j=1}^{|S|} \sum_{t=1}^{T_l} \mathbb I[z_{t-1}^{(l)}=s_i \land z_t^{(l)} =s_j]\log A_{ij}
\\+\sum_{l=1}^m \sum_{j=1}^{|S|}\sum_{k=1}^{|V|}\sum_{t=1}^{T_l} \mathbb I[z_t^{(l)}=s_j \land x_t^{(l)}=v_k]\log B_{jk} \tag{1}$$

约束条件为 

$$\sum_{j=1}^{|S|} A_{ij}=1, \quad i=1,\cdots, |S|
\\ A_{ij} \ge 0, \quad i,j=1,\cdots, |S|
\\ \sum_{k}^{|V|} B_{jk}=1, \quad j=1,\cdots, |S|
\\ B_{jk} \ge 0, \quad j=1,\cdots, |S|, k=1,\cdots, |V|$$


使用拉格朗日乘子法，拉格朗日函数为

$$\mathcal L = L+ \sum_{i=1}^{|S|} \alpha_i \left(1-\sum_{j=1}^{|S|} A_{ij}\right)+\sum_{j=1}^{|S|} \beta_j \left(1-\sum_{k}^{|V|} B_{jk}\right)$$

对各参数求偏导，令其为 0，

$$\frac {\partial \mathcal L} {\partial A_{ij}}=\frac 1 {A_{ij}} \sum_{l=1}^m \sum_{t=1}^{T_l}\mathbb I[z_{t-1}^{(l)}=s_i \land z_t^{(l)} =s_j]-\alpha_i=0$$

$$\Rightarrow A_{ij}=\frac 1 {\alpha_i} \sum_{l=1}^m \sum_{t=1}^{T_l}\mathbb I[z_{t-1}^{(l)}=s_i \land z_t^{(l)} =s_j]$$

$$\frac {\partial \mathcal L} {\partial \alpha_i}=1-\sum_{j=1}^{|S|} A_{ij}=0
\\
\Rightarrow  1-\sum_{j=1}^{|S|}\frac 1 {\alpha_i} \sum_{l=1}^m \sum_{t=1}^{T_l}\mathbb I[z_{t-1}^{(l)}=s_i \land z_t^{(l)} =s_j]=0
\\ \Rightarrow  \alpha_i=\sum_{j=1}^{|S|}\sum_{l=1}^m \sum_{t=1}^{T_l}\mathbb I[z_{t-1}^{(l)}=s_i \land z_t^{(l)} =s_j]=\sum_{l=1}^m \sum_{t=1}^{T_l}\mathbb I[z_{t-1}^{(l)}=s_i]$$

于是，

$$A_{ij}=\frac {\sum_{l=1}^m \sum_{t=1}^{T_l}\mathbb I[z_{t-1}^{(l)}=s_i \land z_t^{(l)} =s_j]}{\sum_{l=1}^m \sum_{t=1}^{T_l}\mathbb I[z_{t-1}^{(l)}=s_i]} \tag{2}$$

类似地推导可得

$$B_{jk}=\frac {\sum_{l=1}^m \sum_{t=1}^{T_l} \mathbb I[z_t^{(l)}=s_j \land x_t^{(l)}=v_k]}{\sum_{l=1}^m \sum_{t=1}^{T_l} \mathbb I[z_t^{(l)}=s_j]} \tag{3}$$

为了防止除零错误，即 $z_{t-1}^{(l)}=s_i$ 仅出现一次，可以采用 $+1$ 处理，例如，额外让每组 $z_{t-1}^{(l)}=s_i \land z_t^{(l)} =s_j$ 出现一次，那么 $z_{t-1}^{(l)}=s_i$ 出现 $|S|$ 次，此时 $A_{ij}=1/|S|$，这表示从状态 $i$ 等概率地转移到任意下一状态，同样类似地，$B_{jk}=1/|V|$ 。

# 非监督学习

