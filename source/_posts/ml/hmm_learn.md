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

由于这里存在隐变量 $\mathbf z$ （状态），根据 [GMM](2021/11/13/ml/GMM) 中的 EM 算法进行迭代，步骤如下：

---

<center>使用 EM 算法进行 HMM 学习</center>

初始化模型参数 $A^{(0)}, \ B^{(0)}$

**for t=1,2,...**

1. E-step，计算状态后验概率
    $$r(\mathbf z)=p(\mathbf z|\mathbf x;A^{(t-1)},B^{(t-1)})$$

2. M-step，求以下最大值优化问题
    $$A^{(t)},B^{(t)}=\arg \max_{A,B} \sum_{\mathbf z} r(\mathbf z) \log \frac {P(\mathbf x, \mathbf z;A, B)} {r(\mathbf z)}$$
    $$\sum_{j=1}^{|S|} A_{ij}=1, \ i=1,\ldots,|S|; A_{ij}\ge 0, i,j=1,\ldots |S|$$
    $$\sum_{k=1}^{|V|}B_{ik}=1, \ i=1,\ldots,|S|; B_{ik} \ge 0, i=1,\ldots, |S|, k = 1,\ldots,|V|$$

（也可以通过计算 $A, B$ 的变化值 (例如矩阵的 F 范数)，当变化值小于某一预设阈值时，停止收敛。）

---

上述算法的一个缺点是 $\mathbf z$ （状态序列） 有 $|S|^T$ 个取值（其中 $T$ 为观测序列长度，$|S|$ 为状态集数量），这个值很容易过大，导致计算耗时。考虑使用前向或后向算法，即，通过迭代的形式计算最值（动态优化）。

首先重写目标函数，

$$\begin{aligned}
A,B &=\arg\max_{A,B} \sum_{\mathbf z} r(\mathbf z) \log \frac {P(\mathbf x, \mathbf z;A, B)} {r(\mathbf z)}
\\& = \arg\max_{A,B} \sum_{\mathbf z} r(\mathbf z) \log P(\mathbf x, \mathbf z;A, B)
\\&=\arg\max_{A,B} \sum_{\mathbf z} r(\mathbf z) \log \left(\prod_{t=1}^T p(x_t|z_t;B)\right)\left(\prod_{t=1}^T p(z_t|z_{t-1};A)\right)
\\&=\arg\max_{A,B} \sum_{\mathbf z} r(\mathbf z) \left(\sum_{t=1}^T (\log B_{z_t,x_t}+\log A_{z_{t-1},z_t})\right)
\\&=\arg\max_{A,B} \sum_{\mathbf z} r(\mathbf z) \left(\sum_{j=1}^{|S|} \sum_{k=1}^{|V|}\sum_{t=1}^T \mathbb I[z_t=s_j \land x_t=v_k] \log B_{jk}\right) 

\\& \quad + \left(\sum_{i=1}^{|S|} \sum_{j=1}^{|S|}\sum_{t=1}^T \mathbb I[z_{t-1}=s_i \land z_t=s_j] \log A_{ij} \right)
\end{aligned}$$

上式推导中，第二行是由于 $-\log r(\mathbf z)$ 与 $A,B$ 无关，所以可以去掉。

不等式约束条件 $A_{ij} \ge 0$ 和 $B_{ik} \ge 0$ 可以忽略，这是因为目标函数中有 $\log A_{ij}$ 和 $\log B_{ik}$，这势必要求不等式约束条件必须成立，故只考虑等式约束条件（概率和为 1），使用拉格朗日乘子法，

$$\begin{aligned}
L(A,B,\delta,\epsilon)&=\sum_{\mathbf z} r(\mathbf z) \left(\sum_{j=1}^{|S|} \sum_{k=1}^{|V|}\sum_{t=1}^T \mathbb I[z_t=s_j \land x_t=v_k] \log B_{jk}\right) 

\\& \quad + \left(\sum_{i=1}^{|S|} \sum_{j=1}^{|S|}\sum_{t=1}^T \mathbb I[z_{t-1}=s_i \land z_t=s_j] \log A_{ij} \right)
\\& \quad + \sum_{j=1}^{|S|} \epsilon_j(1-\sum_{k=1}^{|K|}) + \sum_{i=1}^{|S|} \delta_i (1-\sum_{j=1}^{|S|} A_{ij})
\end{aligned}$$

求目标函数对各参数的梯度，

$$\frac {\partial L}{\partial A_{ij}} = \sum_{\mathbf z} r(\mathbf z) \frac 1 {A_{ij}} \sum_{t=1}^T \mathbb I[z_{t-1}=s_i \land z_t=s_j] - \delta_i \equiv 0$$

$$A_{ij} = \frac 1 {\delta_i} \sum_{\mathbf z} r(\mathbf z) \sum_{t=1}^T \mathbb I[z_{t-1}=s_i \land z_t=s_j] \tag{4}$$

类似地，

$$B_{jk}=\frac 1 {\epsilon_j} \sum_{\mathbf z} r(\mathbf z) \sum_{t=1}^T \mathbb I[z_t=s_j \land x_t=v_k] \tag{5}$$

对朗格朗日乘子参数求梯度，

$$\begin{aligned}\frac {\partial L}{\partial \delta_i} &= 1-\sum_{j=1}^{|S|}A_{ij}
\\&=1-\sum_{j=1}^{|S|}\frac 1 {\delta_i}\sum_{\mathbf z} r(\mathbf z) \sum_{t=1}^T \mathbb I[z_{t-1}=s_i \land z_t=s_j] \equiv 0
\end{aligned}$$

得

$$\begin{aligned}\delta_i &= \sum_{j=1}^{|S|}\sum_{\mathbf z} r(\mathbf z) \sum_{t=1}^T \mathbb I[z_{t-1}=s_i \land z_t=s_j]
\\&=\sum_{\mathbf z} r(\mathbf z) \sum_{t=1}^T \mathbb I[z_{t-1}=s_i]
\end{aligned}$$

类似地有

$$\epsilon_j = \sum_{\mathbf z} r(\mathbf z)\sum_{t=1}^T \mathbb I[z_t=s_j]$$

将 $\delta_i, \ \epsilon_j$ 代入 $A_{ij}, \ B_{jk}$ 的表达式 (4)、(5) 两式，得

$$\hat A_{ij}=\frac {\sum_{\mathbf z} r(\mathbf z) \sum_{t=1}^T \mathbb I[z_{t-1}=s_i \land z_t=s_j]} {\sum_{\mathbf z} r(\mathbf z) \sum_{t=1}^T \mathbb I[z_{t-1}=s_i]}$$

$$\hat B_{jk}=\frac {\sum_{\mathbf z} r(\mathbf z) \sum_{t=1}^T \mathbb I[z_t=s_j \land x_t=v_k]}{\sum_{\mathbf z} r(\mathbf z)\sum_{t=1}^T \mathbb I[z_t=s_j]}$$

然而，上两式仍然需要对所有 $|S|^T$ 个状态序列 $\mathbf z$ 计算求和。使用动态优化的前向和后向算法。考虑 $A_{ij}$ 中的分子计算，

$$\begin{aligned}&\sum_{\mathbf z} r(\mathbf z) \sum_{t=1}^T \mathbb I[z_{t-1}=s_i \land z_t=s_j]
\\=&\sum_{t=1}^T \sum_{\mathbf z}  \mathbb I[z_{t-1}=s_i \land z_t=s_j] r(\mathbf z)
\\=&\sum_{t=1}^T \sum_{\mathbf z}  \mathbb I[z_{t-1}=s_i \land z_t=s_j] p(\mathbf z|\mathbf x;A,B)
\\=& \frac 1 {p(\mathbf x;A,B)}\sum_{t=1}^T \sum_{\mathbf z}  \mathbb I[z_{t-1}=s_i \land z_t=s_j] p(\mathbf z,\mathbf x;A,B)
\\=& \frac 1 {p(\mathbf x;A,B)}\sum_{t=1}^T p(z_{t-1}=s_i,z_t=s_j,\mathbf x;A,B)
\\=& \frac 1 {p(\mathbf x;A,B)}\sum_{t=1}^T \alpha_i(t-1) A_{ij} B_{j,x_t} \beta_j(t)
\end{aligned}$$

类似地计算分母部分，

$$\begin{aligned}
&\sum_{\mathbf z} r(\mathbf z) \sum_{t=1}^T \mathbb I[z_{t-1}=s_i]
\\=& \sum_{j=1}^{|S|}\sum_{\mathbf z} r(\mathbf z) \sum_{t=1}^T \mathbb I[z_{t-1}=s_i \land z_t=s_j]
\\=& \frac 1 {p(\mathbf x;A,B)}\sum_{j=1}^{|S|}\sum_{t=1}^T \alpha_i(t-1) A_{ij} B_{j,x_t} \beta_j(t)
\end{aligned}$$

于是 $\hat A_{ij}$ 的表达式可写为

$$\hat A_{ij}=\frac {\sum_{t=1}^T \alpha_i(t-1) A_{ij} B_{j,x_t} \beta_j(t)}{\sum_{k=1}^{|S|}\sum_{t=1}^T \alpha_i(t-1) A_{ik} B_{k,x_t} \beta_k(t)} \tag{6}$$

这样，计算复杂度从 $|S|^T$ 降为 $T\cdot |S|^2$，例如 $|S|=4$，$T=10$，显然此时后者远小于前者，降低了计算复杂度。

同样地有 $\hat B_{jk}$ 的分子

$$\begin{aligned}
&\sum_{\mathbf z} r(\mathbf z) \sum_{t=1}^T \mathbb I[z_t=s_j \land x_t=v_k]
\\=& \frac 1 {p(\mathbf x;A,B)} \sum_{t=1}^T \sum_{\mathbf z} \mathbb I[z_t=s_j \land x_t = v_k] p(\mathbf z,\mathbf x;A,B)
\\=& \frac 1 {p(\mathbf x;A,B)} \sum_{i=1}^{|S|}\sum_{t=1}^T \sum_{\mathbf z}\mathbb I[z_{t-1}=s_i \land z_t=s_j \land x_t = v_k] p(\mathbf z,\mathbf x;A,B)
\\=& \frac 1 {p(\mathbf x;A,B)}\sum_{i=1}^{|S|}\sum_{t=1}^T \mathbb I[x_t=v_k] \alpha_i(t-1) A_{ij} B_{j,x_t} \beta_j(t)
\end{aligned}$$

$\hat B_{jk}$ 的分母为

$$\begin{aligned}
&\sum_{\mathbf z} r(\mathbf z)\sum_{t=1}^T \mathbb I[z_t=s_j]
\\=& \frac 1 {p(\mathbf x;A,B)}\sum_{i=1}^{|S|}\sum_{t=1}^T \sum_{\mathbf z}\mathbb I[z_{t-1}=s_i \land z_t=s_j] p(\mathbf z,\mathbf x;A,B)
\\=& \frac 1 {p(\mathbf x;A,B)}\sum_{i=1}^{|S|}\sum_{t=1}^T \alpha_i(t-1) A_{ij} B_{j,x_t} \beta_j(t)
\end{aligned}$$

于是

$$\hat B_{ij}=\frac {\sum_{i=1}^{|S|}\sum_{t=1}^T \mathbb I[x_t=v_k] \alpha_i(t-1) A_{ij} B_{j,x_t} \beta_j(t)}{\sum_{i=1}^{|S|}\sum_{t=1}^T \alpha_i(t-1) A_{ij} B_{j,x_t} \beta_j(t)}\tag{7}$$

---
<center> 用于 HMM 学习的前后向算法</center>

初始化：随机初始化 $A, \ B$ 为某有效概率矩阵，且使得 $A_{i0}=0$， $B_{0k}=0$。例如 $A_{:,1:}=1/S$，$B_{1:,:}=1/V$，这里 $A \in \mathbb R^{(S+1)\times(S+1)}$，$B \in \mathbb R^{(S+1)\times V}$。为了简单，令 $|S| \rightarrow S$，$|V| \rightarrow V$ 。

循环：

**for m=1,2,...**

1. E-step， 计算
    $$\alpha_i(t), \beta_i(t), \quad i=0,\ldots,S, \ t=1,\ldots, T$$
    $$\gamma_t(i,j)=\alpha_i(t-1) A_{ij}B_{j,x_t} \beta_j(t), \quad i=0,1,\ldots, S, \ j=1,2,\ldots, S$$

2. M-step， 计算
    $$A_{ij}=\frac {\sum_{t=1}^T \gamma_t(i,j)}{\sum_{j=1}^S \sum_{t=1}^T \gamma_t(i,j)}$$
    $$B_{jk}=\frac {\sum_{i=1}^S \sum_{t=1}^T \mathbf I[x_t=v_k]\gamma_t(i,j)}{\sum_{i=1}^S \sum_{t=1}^T \gamma_t(i,j)}$$
    
---


