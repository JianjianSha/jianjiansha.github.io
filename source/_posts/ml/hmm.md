---
title: 隐马尔可夫模型
date: 2021-10-29 15:27:35
tags: machine learning
mathjax: true
---

马尔可夫模型用于处理序列数据，但是实际上并不常用，这是因为马尔可夫模型中观测值就是我们所关心的值，然而实际上很多时候并非如此，例如 part-of-speech （POS） 标记任务中，我们观测到的是单词，而非 POS，或者在语音识别中，观测到的是语音而非单词自身。这个时候就需要用到 隐马尔可夫模型（HMM）了。
<!--more-->

# 隐马尔可夫模型

观测到的序列 $\mathbf x=\{x_1,\cdots, x_T\}$，其中每个观测值来自集合 $V=\{v_1,\cdots, v_{|V|}\}$，我们感兴趣的状态序列（不可观测）为 $\mathbf z=\{z_1,\cdots, z_T\}$，其中每个状态均来自状态集 $S=\{s_1,\cdots, s_{|S|}\}$ 。状态转移矩阵依然记作 $A$ 。

将状态看作隐变量，那么输出序列是状态的函数，并作如下假设：

1. 观测独立假设

    $$P(x_t=v_k|z_t=s_j)=P(x_t=v_k|x_1,\cdots, x_T, z_1,\cdots,z_T)=B_{jk}$$

    即，$t$ 时刻的的观测值仅跟 $t$ 时刻的状态有关。矩阵 $B$ 表示由隐式状态到观测值的概率。

例如 [马尔可夫模型]() 中的天气的那个例子，天气状态无法直接观测，但是可以观测当天冰淇淋消费的数量，以 4 天为一个观测序列，那么一个观测序列可以是 $\mathbf x=(x_1=3, x_2=2,x_3=1,x_4=2)$。

## 观测序列概率

我们这样构建数据生成过程：假设有一个状态序列 $\mathbf z$，由马尔可夫模型生成，对应的状态转移矩阵为 $A$，在每个时刻 $t$，观测输出 $x_t$ 是状态 $z_t$ 的函数，于是观测序列的概率为

$$P(\mathbf x;A,B)=\sum_{\mathbf z}P(\mathbf x, \mathbf z;A,B)=\sum_{\mathbf z} P(\mathbf x|\mathbf z;A,B)P(\mathbf z;A,B)$$

根据观测独立假设，可以进一步简化上面这个概率，

$$\begin{aligned}P(\mathbf x;A,B)&=\sum_{\mathbf z} P(\mathbf x|\mathbf z;A,B)P(\mathbf z;A,B)\\
&=\sum_{\mathbf z}\left(\prod_{t=1}^T P(x_t|z_t;B) \right)\left(\prod_{t=1}^T P(z_t|z_{t-1};A)\right)\\
&=\sum_{\mathbf z} \left(\prod_{t=1}^T B_{z_tx_t} \right)\left() \prod_{t=1}^T A_{z_{t-1}z_t}\right)
\end{aligned}$$

上式概率是对 $\mathbf z$ 的各种情况求和，而 $\mathbf z$ 序列长度为 $T$，每个状态有 $|S|$ 种可能，所以一共有 $|S|^T$ 个这样的状态序列，故计算复杂度为 $O(|S|^T)$ 。

**快速计算概率的方法——前向过程**

定义 $\alpha_i(t)=P(x_1,x_2,\cdots,x_t,z_t=s_i;A,B)$，表示截止到观测时刻 $t$，$t$ 时刻状态为 $s_i$ 的观测序列的概率，于是最终的观测序列的概率为

$$\begin{aligned}P(\mathbf x;A,B)&=P(x_1,x_2,\cdots,x_T;A,B)\\
&=\sum_{i=1}^{|S|} P(x_1,x_2,\cdots,x_T;z_T=s_i;A,B)\\
&=\sum_{i=1}^{|S|} \alpha_i(T)
\end{aligned}$$

根据 $\alpha_i(t)$ 的定义，不难知道有以下迭代关系，

$$\alpha_j(t)=\sum_{i=1}^{|S|} \alpha_i(t-1) A_{ij}B_{jx_t}, \quad j=1,\cdots,|S|, t=1,\cdots, T$$

上式表明，$t$ 长度的观测序列是在 $t-1$ 长度的观测序列的基础上，再观测一个$t$ 时刻的值得到，这个 $t$ 时刻观测值为 $x_t$，其状态 $z_t=s_j$ 可以是由 $t-1$ 时刻的状态转移得到，$z_{t-1}$ 可以是 $1,2,\cdots,|S|$ 中的任意一个，然后根据 $z_t=s_j$ 这个状态生成观测值 $x_t$，这个概率为 $B_{jx_t}$。迭代关系的初始条件为

$$\alpha_i(0)=A_{0i}, \quad i=1,\cdots, |S|$$

这表示，$0$ 时刻状态为 $s_i$，即初始状态为 $s_i$ 的概率，此时，观测序列长度为 0，故只要考虑初始状态的出现概率即可。

在每个时刻，我们均需要计算 $|S|$ 个值 $\alpha_1(t),\cdots,\alpha_{|S|}(t)$，故计算复杂度为 $O(|S| \cdot T)$ 。

**后向过程**

采用后向过程也可以快速计算概率，需要定义相关量为 $\beta_i(t)=P(x_T,x_{T-1},\cdots,x_{t+1},z_t=s_i;A,B)$，那么观测序列概率为

$$P(\mathbf x;A,B)=\sum_{i=1}^{|S|} A_{0i} B_{i,x_1}\beta_i(1)$$

迭代公式为

$$\begin{aligned}\beta_i(t)&=P(x_{t+1},\cdots,x_T,z_t=i)\\
&=\sum_{j=1}^{|S|}P(j|i)P(x_{t+1}|j)P(x_{t+2},\cdots,x_T,z_{t+1}=j)\\
&=\sum_{j=1}^{|S|}A_{ij}B_{j,x_{t+1}}\beta_j(t+1)
\end{aligned}$$

初始条件为 $\beta_i(T)=1, \quad i=1,\cdots,|S|$，这是因为根据定义 $\beta_i(T)=P(x_{T},x_{T+1},z_{T}=s_i)$，由于从 $T+1$ 时刻到 $T$ 时刻的观测（子）序列不存在，为了使上式迭代成立，令其为 1 。



<!-- **前后下关系**

前向算法和后向算法的相关量的关系，根据定义有

$$\begin{aligned}P(\mathbf x,\mathbf z;A,B)&=\sum_{i=1}^{|S|} P(x_1,\ldots,x_t,z_t=s_i;A,B) P(x_t,x_{t+1},\ldots, x_T, z_t=s_i)
\\&=\sum_{i=1}^{|S|} \alpha_i(t) \beta_i(t)
\\&=\sum_{i=1}^{|S|} \sum_{j}^{|S|} \alpha_i(t) A_{ij}\beta_j(t+1)
\end{aligned} \tag{1}$$ -->

## 最可能的状态序列（解码）

给定一个观测序列，求最有可能的状态序列，即求下式最大值问题的解

$$\arg \max_{\mathbf z} P(\mathbf z|\mathbf x;A,B)=\arg \max_{\mathbf z} \frac {P(\mathbf x,\mathbf z;A,B)}{P(\mathbf x;A,B)}=\arg \max_{\mathbf z} P(\mathbf x,\mathbf z;A,B)$$

如果我们对每个 $\mathbf z$ 计算上式右端概率，然后取最大值，那么计算复杂度为 $O(|S|^T)$ 。可以采用动态规划算法，我们先给出算法过程如下：

记 $t$ 时刻状态为 $z_t=s_i$ 时所有局部状态路径 $(z_1,\cdots,z_{t-1})$ 的最大概率为

$$\delta_i(t)=\max_{z_1,\cdots,z_{t-1}} \ P(x_1,\cdots,x_t,z_1,\cdots,z_{t-1},z_t=s_i;A,B), \quad i=1,\cdots,|S|$$

于是递归公示为，

$$\begin{aligned}\delta_j(t+1)&=\max_{z_1,\cdots,z_t} \ P(x_1,\cdots, x_{t+1},z_1,\cdots,z_t,z_{t+1}=j)
\\&=\max_{z_1,\cdots,z_t} \ P(x_1,\cdots,x_t,z_1,\cdots,z_t)A_{z_tj}B_{jx_{t+1}}
\\&\stackrel{z_t=i}=\max_i \delta_i(t)A_{ij}B_{jx_{t+1}}
\end{aligned}$$

上式中 $j=1,\cdots,|S|$ 。

初始条件 
    $$\delta_i(1)=\max_{z_0} P(x_1;z_1=s_i;A,B)=A_{0i} B_{i,x_1}, \ i=1,2,\ldots, |S|$$

这样，对 $t+1$ 时刻的每一种可能的状态 $j$，均可计算出一个确定的 $t$ 时刻的状态 $i$，使得 $\delta_j(t+1)$ 最大，这表示，在任意时刻 $t$，对 $t$ 时刻的任意一个状态 $i$，均可确定一条状态路径 $path(z_t=i)=(z_1,\cdots,z_{t-1})$ 使得局部概率 $\delta_i(t)$ 最大，根据 $\delta_i(t)$ 的定义，最终目标的最大概率为 $\max_i \delta_i(T)$，求出  $\hat i = \arg \max_i \delta_i(T)$ 后，就可以确定最优状态路径 $path(z_T=\hat i)=(z_1,\cdots,z_{T-1})$ 了。

使用一个列表 $L$，$L$ 里面有 $T-1$ 个 列表 $l^t, \ t=1,\cdots,T-1$，每个列表 $l^t$ 表示一个时刻，列表 $l_t$ 的长度均为 $|S|$，$l^t$ 中下标 $i$ 表示当前时刻 $t+1$ 在状态为 $i$ 时前一时刻状态。

另外使用一个列表 $L^1$，长度为 $|S|$，记录 $t$ 时刻为各个状态的局部路径最大值 $\delta_j(t)$，每个时刻均对 $L^1$ 进行更新，直到 $T$ 时刻计算后，就是全局路径（终结时刻为各个状态）的最大概率值，此时选择 $L^1$ 中最大的概率值，对应的下标就是 $z_T$ 的值，然后再根据 $L$ 列进行路径回溯，即 $l^{T-1}$ 列表中取下标为 $z_T$ 的元素值 $z_{T-1}=l_{z_T}^{T-1}$，这就是最优路径商 $T-1$ 时刻的状态，然后在 $l^{T-2}$ 表中取下标 $z_{T-1}$ 的元素值，此为最优路径上 $T-2$ 时刻的状态，依次如此回溯，直到取出 $t=1$ 时刻的状态。