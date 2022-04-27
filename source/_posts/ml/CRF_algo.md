---
title: CRF 算法
date: 2022-02-21 11:32:28
tags: machine learning
p: ml/CRF_algo
mathjax: true
categories: 机器学习
img: /images/ml/CRF2.png
---
本文讲述 CRF 算法实现部分。
<!--more-->

# 1 线性 CRF

为简单起见，从线性 CRF （即输出变量 $y_t$ 之间是一阶连接 $y_{t-1} \rightarrow y_t$）开始讨论。

本文约定 $\mathbf x_{1:t}$ 表示子序列 $(x_1,\ldots, x_t)$ ，其余同。

既然 HMM 可能转化为线性 CRF，就从 HMM 的前后向算法和 Viterbi 算法开始讨论。


HMM 的联合概率分布可写为

$$p(\mathbf x, \mathbf y)=\prod_t \Psi_t(y_t, y_{t-1}, x_t) \tag{1}$$

其中 $Z=1$，且 $\Psi_t(j,i,x):=p(y_t=j|y_{t-1}=i)p(x_t=x|y_t=j), \ t=1, 2, \ldots, T$。

以下讨论三个问题。

### 1.1 推断

推断是指计算 **观察序列的概率 $p(\mathbf x)$** 。

根据 (1) 式可得观察序列 $\mathbf x$ 的边缘概率为，

$$\begin{aligned} p(\mathbf x)&=\sum_{\mathbf y} \prod_{t=1}^T \Psi_t(y_t, y_{t-1},x_t)
\\ &=\sum_{y_T}\sum_{y_{T-1}} \Psi_T(y_T,y_{T-1},x_T) \sum_{y_{T-2}} \Psi_{T-1}(y_{T-1},y_{T-2},x_{T-1})\sum_{y_{T-3}}\cdots
\end{aligned}$$

**前向算法**

$$\alpha_t(j):=p(\mathbf x_{1:t}, y_t=j), \quad t=1,\ldots, T, j=1,\ldots, S \tag{2}$$

其中 $S$ 为隐变量 $y_t$ 的所有状态数量。

根据 (2) 式定义易知迭代公式为

$$\alpha_t(j)=\sum_{i=1}^S \Psi_t(j,i,x_t) \alpha_{t-1}(i) \tag{3}$$

初始条件为

$$\alpha_1(j)=\Psi_1(j,y_0,x_1)=p(j)\cdot p(x_t|j)$$

观测序列 $\mathbf x$ 的边缘概率为 

$$p(\mathbf x)=\sum_{j=1}^S \alpha_T(j) \tag{4}$$

**后向算法**

$$\beta_t(i):=p(\mathbf x_{t+1:T}, y_t=i) \tag{5}$$

迭代公式为

$$\beta_t(i)=\sum_{j=1}^S \Psi_{t+1}(j, i, x_{t+1}) \beta_{t+1}(j) \tag{6}$$

初始条件为 $\beta_T(i)=1$ 。

观测序列 $\mathbf x$ 的边缘概率为 

$$p(\mathbf x)=\beta_0(y_0):=\sum_{j=1}^S \Psi_1(j,y_0,x_1)\beta_1(j) \tag{7}$$


计算 $p(y_{t-1},y_t|\mathbf x)$，这在后面参数估计中会用到。

$$\begin{aligned}p(y_{t-1},y_t|\mathbf x)&=\frac {p(\mathbf x,y_{t-1},y_t)}{p(\mathbf x)}
\\&=\frac {p(\mathbf x_{1:t-1},y_{t-1})p(y_t|y_{t-1})p(x_t|y_t)p(\mathbf x_{t+1,T}, y_t)}{p(\mathbf x)}
\\&=\frac 1 {p(\mathbf x)} \alpha_{t-1}(y_{t-1}) \Psi_t(y_t, y_{t-1},x_t) \beta_t(y_t)
\end{aligned} \tag{7-1}$$

## 1.2 解码

给定一个新的观测序列 $\mathbf x$，求其最有可能的状态序列 $\mathbf y$ ，即

$$\hat {\mathbf y}=\arg \max_{\mathbf y} \ p(\mathbf y|\mathbf x) = \arg \max_{\mathbf y} p(\mathbf x, \mathbf y)$$

**Viterbi 算法**


$$\delta_t(j):=\max_{\mathbf y_{1:t-1}} \ p(\mathbf y_{1:t-1}, \mathbf x_{1:t}, y_t=j)$$

表示： 当前已经观察到序列 $\mathbf x_{1:t}$，且时刻 $t$ 的 label 为 $y_t=j$ 的概率，时刻 $t=1, \ldots, t-1$ 的 label 未知，求 $t=1,\ldots, t-1$ 的 label，使得这个部分观察序列的联合概率最大。

迭代公式为 

$$\begin{aligned}\color{fuchsia} {\delta_t(j)}&=\max_{\mathbf y_{1:t-1}} p(\mathbf y_{1:t-1}, \mathbf x_{1:t}, y_t=j)
\\&=\max_{i \in 1,\ldots, S} \Psi_t(j,i, x_t) \cdot \max_{\mathbf y_{1:t-2}} p(y_{1:t-2}, \mathbf x_{1:t-1}, y_{t-1}=i)
\\&= \color{fuchsia} {\max_{i \in 1, \ldots, S} \Psi(j,i,x_t) \delta_{t-1} (i)}
\end{aligned} \tag{8}$$

注意：这里使用的是类似于 HMM  的线性 CRF 结构。

同时还得到

$$y_{t-1}^{\star}=\arg \max_{y_{t-1}} \delta_t(j) \tag{9}$$

初始条件为 $\delta_1(j)=\Psi(j,y_0,x_1)=p(y_1=j)p(x_t|j)$ （一共 $S$ 个初始条件）。

根据 (9) 式，每个时刻 $t, t>1$，均可计算出上一时刻的最优状态 $y_{t-1}$。

最后迭代计算出 

$$y_T = \arg \max_j = \delta_T(j)$$

然后将 $y_T$ 的值代入 (9) 式进行状态回溯，得到最优的状态序列 $\mathbf y$。

**扩展到线性 CRF**

将 HMM 扩展到线性 CRF 的一般形式，前后向算法和 Viterbi 算法形式与上面各式相同，只是 $\Psi_t(y_t, y_{t-1},x_t)$ 不一定是 HMM 中条件概率，而是更一般的势函数，

$$\Psi_t(y_t, y_{t-1}, \mathbf x)=\exp \left(\sum_k \theta_k f_k(y_t, y_{t-1}, \mathbf x)\right)$$


联合概率为

$$p(\mathbf x, \mathbf y)=\frac 1 Z \prod_{t=1}^T \Psi_t(y_t, y_{t-1},\mathbf x)$$

后验概率为 

$$p(\mathbf y|\mathbf x)=\frac 1 {Z(\mathbf x)}\prod_{t=1}^T \Psi_t(y_t, y_{t-1},\mathbf x) \tag{10}$$

注意线性 CRF 中 $\alpha_t(j), \ \beta_t(i)$ 不再表示概率，且 $\sum_{j=1}^S \alpha_T(j)$ 也不表示 $p(\mathbf x)$ 而是 $Z(\mathbf x)$，即

$$Z(\mathbf x)=\sum_{\mathbf y} \prod_{t=1}^T \Psi_t(y_t, y_{t-1}, \mathbf x)= \sum_{j=1}^S \alpha_T(j)=\beta_0(y_0)$$

$$p(\mathbf x)=\sum_{\mathbf y} p(\mathbf x, \mathbf y)=\frac {Z(\mathbf x)}{Z}$$

特殊地，线性 CRF 为 HMM 时，有 $Z=1$，此时 $p(\mathbf x)=Z(\mathbf x)$ 。


## 1.3 参数估计

采用最大似然估计。

数据集 $\mathcal D=\{\mathbf x^{(i)}, \mathbf y^{(i)}\}_{i=1}^N$，其中 $\mathbf x^{(i)}=(\mathbf x_1^{(i)}, \ldots, \mathbf x_T^{(i)})$ 是输入序列，输出序列为 $\mathbf y^{(i)}=(y_1^{(i)}, \ldots, y_T^{(i)})$，序列长度 $T$ 不固定，即 $T_i$ 可变，但是下文为了表达简单，统一写成 $T$，不影响算法的理解和实现。

模型参数记为 $\theta=(\theta_1,\ldots, \theta_K)$，K 为特征函数数量。记对数条件似然为

$$l(\theta)=\sum_{i=1}^N \log p(\mathbf y^{(i)}|\mathbf x^{(i)}; \theta) \tag{11}$$

我们要求参数的最大似然估计 $\theta_{ML} = \arg \max_{\theta} l(\theta)$ 。现在已知线性 CRF 的条件概率为 (10) 式，代入 (11) 式得，

$$l(\theta)=\sum_{i=1}^N \sum_{t=1}^T \sum_{k=1}^K \theta_k f_k(y_t^{(i)}, y_{t-1}^{(i)}, \mathbf x_t^{(i)}) - \sum_{i=1}^N \log Z(\mathbf x^{(i)})$$

参数量 $K \ge S\times V+S^2$ （S 为状态数，V 为观测值的集合大小）非常大，可能达几十万，为了避免过拟合，使用 **正则项**，于是

$$l(\theta)=\sum_{i=1}^N \sum_{t=1}^T \sum_{k=1}^K \theta_k f_k(y_t^{(i)}, y_{t-1}^{(i)}, \mathbf x_t^{(i)}) - \sum_{i=1}^N \log Z(\mathbf x^{(i)}) -\sum_{k=1}^K \frac {\theta_k^2}{2\sigma^2}\tag{12}$$

其中 $\sigma^2$ 是平衡因子，引入正则项等效于参数的 MAP （最大后验 ）估计。

由于

$$\begin{aligned}Z(\mathbf x)&=\sum_{\mathbf y} \prod_t \Psi_t(y_t,y_{t-1}, \mathbf x_t)
\\&=\sum_{\mathbf y}\prod_t \exp\left(\sum_k \theta_k f_k(y_t,y_{t-1}, \mathbf x_t)\right)
\\&=\sum_{\mathbf y} \exp\left(\sum_t \sum_k \theta_k f_k(y_t,y_{t-1}, \mathbf x_t) \right)
\end{aligned}$$

$$\begin{aligned}\frac 1 {Z(\mathbf x)}\frac {\partial Z(\mathbf x)}{\partial \theta_k}&=\frac 1 {Z(\mathbf x)}\sum_{\mathbf y}  \exp(\cdots) \left(\sum_t f_k(y_t,y_{t-1}, \mathbf x_t)\right)
\\&=\sum_{\mathbf y} \frac 1 {Z(\mathbf x)} \prod_s \Psi_{s=1}^T (y_s,y_{s-1}, \mathbf x_s)  \left(\sum_t f_k(y_t,y_{t-1}, \mathbf x_t)\right)
\\&=\sum_{\mathbf y} p(\mathbf y|\mathbf x) \left(\sum_t f_k(y_t,y_{t-1}, \mathbf x_t)\right)
\\&=\sum_{y_t, y_{t-1}} p(y_t, y_{t-1}|\mathbf x)\left(\sum_t f_k(y_t,y_{t-1}, \mathbf x_t)\right)
\\&=\sum_{i,j}\sum_t p(j, i|\mathbf x)f_k(j,i, \mathbf x_t)
\end{aligned}$$

似然函数对参数求偏导，

$$\begin{aligned} \frac {\partial l}{\partial \theta_k}=&\sum_{i=1}^N \sum_{t=1}^T f_k(y_t^{(i)},y_{t-1}^{(i)}, \mathbf x_t^{(i)})
\\ &-\sum_{i=1}^N\sum_{t=1}^T \sum_{i,j} p(j, i|\mathbf x^{(i)})f_k(j,i, \mathbf x_t^{(i)})
-\frac {\theta_k}{\sigma^2}
\end{aligned} \tag{13}$$

(13) 式中第一项可看作 $f_k$ 在经验分布 $\tilde p(\mathbf x,\mathbf y)$ 下的期望，第二项为 $f_k$ 在分布 $p(\mathbf y|\mathbf x;\theta)\tilde p(\mathbf x)$ 下的期望。

(13) 式中含有 $p(j,i|\mathbf x)$，根据 (7) 和 (7-1) 式进行计算，令 (13) 式表示的梯度为 $\mathbf 0$，难以求得参数解析解，故考虑数值解即梯度上升法：

1. 初始化参数 $\theta^{(1)}$
2. 根据 (13) 式计算的似然函数对参数的梯度，$\nabla_{\theta} l$
3. 更新参数 $\theta^{(t+1)}=\theta^{(t)}+\alpha \nabla_{\theta}l$

# 2. 通用 CRF

$\mathbf y$ 内部连接不再是线性连接。最大团 index 集合记为 $F$，那么概率可写为

$$p(\mathbf y)=Z^{-1} \prod_{a \in F} \Psi_a(\mathbf y_a) \tag{10}$$

近似计算方法包括 MCMC (Markov Chain Monte Carlo) 和 variational algorithms (Belief Propagation) 等。

