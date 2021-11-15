---
title: 马尔可夫模型
date: 2021-10-29 11:32:45
tags: machine learning
mathjax: true
---
马尔可夫模型用于学习和处理（时间）序列数据，例如将音频转为文字，或者给一句话添加标注。
<!--more-->

# 马尔可夫模型
给定一个状态集 $S=\{s_1,s_2,\cdots, s_{|S|}\}$ 。观察到一个时间序列 $\mathbf z \in S^T$ 。 例如一个天气系统 $S=\{晴,多云,雨\}$，一个观察到的序列为 $(z_1=晴,z_2=多云,z_3=多云,z_4=雨,z_5=多云)$，此时 $T=5$ 。在这个例子中，观察到的状态可以看作是一个随机系统随着时间的输出。不做任何假设时，在 $t$ 时刻的观察状态 $s_j$ 可以是任意多变量的一个函数，这些变量包括从 $1$ 时刻到 $t-1$ 时刻的状态值，以及其他很多我们可能都无法知晓的变量，我们可以做马尔可夫假设以简化模型。马尔可夫假设为：

1. 有限视野假设（齐次假设）

    $t$ 时刻的状态仅依赖于 $t-1$ 时刻的状态，跟 $t-1$ 时刻之前的状态无关。

    $$P(z_t|z_{t-1},z_{t-2},\ldots,z_1)=P(z_t|z_{t-1})$$



按照惯例，我们假设有一个初始状态 $s_0$，以及一个初始观察 $z_0 \equiv s_0$ ，有了这个假设，我们就可以将第一个观察状态 $z_1$ 的分布 $p(z_1)$ 写成基于初始状态的后验概率 $p(z_1|z_0)$  ，有时候我们会用向量 $\pi \in \mathbb R^{|S|}$ 来表示初始状态的分布 $P(z_0)$ 。

定义状态转移矩阵 $A \in \mathbb R^{(|S|+1) \times (|S|+1)}$，算上初始状态，一共有 $|S|+1$ 个状态，$A_{ij}$ 表示从状态 $s_i$ 到状态 $s_j$ 的概率，由于 $s_i$ 一定会转移到下一个某个状态，故行和 $\sum_j A_{ij}=1$ 。例如上面那个天气的例子，

$$\begin{array}{c|cccc}
A_{ij} & s_0 & 晴 & 多云 & 雨\\
\hline\\
s_0 & 0 & .33 & .33 & .33\\
晴 & 0 & .8 & .1 & .1 \\
多云 & 0 & .2 & .6 & .2\\
雨 & 0 & .1 & .2 & .7
\end{array}$$

2. 平稳过程假设

    基于当前观测值的下一观测值的条件分布随着时间保持不变。

    $$P(z_t=s_j|z_{t-1}=s_i)=P(z_2=s_j|z_1=s_i), \quad t=2,3,\ldots,T$$

    这个假设其实是指，状态转移矩阵随着时间推移保持不变。

## 预测

预测是指已知转移矩阵，计算状态序列的概率

为了方便，以下用下标 $i$ 表示状态 $s_i$，观察状态序列 $\mathbf z$ 的概率为

$$\begin{aligned} P(\mathbf z)&=P(z_t,z_{t-1},\cdots,z_1;A)\\
&=P(z_t|z_{t-1},\cdots,z_1;A)P(z_{t-1}|z_{t-2},\cdots,z_1;A)\cdots P(z_1|z_0;A)\\
&=P(z_t|z_{t-1};A)P(z_{t-1}|z_{t-2};A)\cdots P(z_2|z_1;A) P(z_1|z_0;A)
\\&=\prod_{t=1}^T P(z_t|z_{t-1};A)\\
&=\prod_{t=1}^T A_{z_{t-1}z_t}
\end{aligned}$$

## 学习

根据观测到的序列，求转移矩阵 $A$ 。根据最大似然准则求解。对数似然函数为

$$\begin{aligned}l(A)&=\log P(\mathbf z;A)\\
&=\log \prod_{t=1}^T A_{z_{t-1}z_t}\\
&=\sum_{t=1}^T \log A_{z_{t-1}z_t}\\
&=\sum_{i=1}^{|S|}\sum_{j=1}^{|S|}\sum_{t=1}^T \mathbb I[z_{t-1}=s_i \land z_t=s_j] \log A_{ij}
\end{aligned}$$

于是问题转化为求优化问题的最优解：

$$\max_A \quad l(A)\\
s.t.  \quad \sum_{j=1}^{|S|}A_{ij}=1, \ i=1,2,\cdots, |S| \\
A_{ij} \ge 0, \ i,j=1,2,\cdots, |S|$$

使用拉格朗日乘子法求解，

$$L(A,\boldsymbol \alpha)=\sum_{i=1}^{|S|}\sum_{j=1}^{|S|}\sum_{t=1}^T \mathbb I[z_{t-1}=s_i \land z_t=s_j] \log A_{ij}+\sum_{i=1}^{|S|} \alpha_i \left(1-\sum_{j=1}^{|S|}A_{ij}\right)$$

求偏导数并令其为 0，

$$\begin{aligned}\frac {\partial L(A,\boldsymbol \alpha)} {\partial A_{ij}}&=\frac 1 {A_{ij}} \sum_{t=1}^T \mathbb I[z_{t-1}=s_i \land z_t=s_j]-\alpha_i=0\\
\Rightarrow \\
A_{ij}&=\frac 1 {\alpha_i} \sum_{t=1}^T \mathbb I[z_{t-1}=s_i \land z_t=s_j]
\end{aligned}$$

$$\frac {\partial L(A,\boldsymbol \alpha)} {\partial \alpha_i}=1-\sum_{j=1}^{|S|} A_{ij}=1-\sum_{j=1}^{|S|}\frac 1 {\alpha_i} \sum_{t=1}^T \mathbb I[z_{t-1}=s_i \land z_t=s_j]=0$$

于是，

$$\alpha_i=\sum_{j=1}^{|S|}\sum_{t=1}^T \mathbb I[z_{t-1}=s_i \land z_t=s_j]=\sum_{t=1}^T \mathbb I[z_{t-1}=s_i]$$

最优解为

$$\hat A_{ij}=\frac {\sum_{t=1}^T \mathbb I[z_{t-1}=s_i \land z_t=s_j]} {\sum_{t=1}^T \mathbb I[z_{t-1}=s_i]}$$

如果有多个序列，$\mathbf z_1, \cdots, \mathbf z_K$，那么似然函数为 $P(\mathbf z_1)\cdots P(\mathbf z_K)$，对数似然则变成求和，易知，最优解形式变为

$$\hat A_{ij}=\frac {\sum_{k=1}^K \sum_{t=1}^T \mathbb I[z_{t-1}=s_i \land z_t=s_j]} {\sum_{k=1}^K \sum_{t=1}^T \mathbb I[z_{t-1}=s_i]}$$

