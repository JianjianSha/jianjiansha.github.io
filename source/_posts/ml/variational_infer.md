---
title: 变分推断
date: 2022-05-25 15:34:04
tags: machine learning
---

# 1 简介

考虑一个带有隐变量的生成模型，隐变量 $\mathbf z = z_{1:m}$，观测变量 $\mathbf x=x_{1:n}$，联合概率密度为

$$p(\mathbf z, \mathbf x)=p(\mathbf z) p(\mathbf x|\mathbf z) \tag{1}$$

贝叶斯模型中，隐变量具有一个先验概率密度 $p(\mathbf z)$，隐变量和观测变量通过似然概率联系起来 $p(\mathbf x|\mathbf z)$，贝叶斯推断则是根据观测变量计算隐变量的后验概率 

$$p(\mathbf z|\mathbf x)=\frac {p(\mathbf z, \mathbf x)}{p(\mathbf x)}=\frac {p(\mathbf z, \mathbf x)}{\int_{\mathbf z} p(\mathbf z, \mathbf x) d \mathbf z} \tag{2}$$

上式由于分子中含有积分，通常无法准确的计算出来，故采样近似推断。MCMC 采样是一种可行的方法，但是缺点是速度太慢，尤其是在数据集较大或模型较复杂的情况下。

## 1.1 变分推断

我们使用一个概率分布 $q(\mathbf z)$ 逼近要求的后验概率 $p(\mathbf z|\mathbf x)$，这个概率分布从 $\mathscr L$ 中搜寻，

$$q^{\star}(\mathbf z)=\arg \min_{q(\mathbf z) \in \mathscr L} \ KL(q(\mathbf z)||p(\mathbf z|\mathbf x)) \tag{3}$$

这是一个优化问题。概率分布族 $\mathscr L$ 其参数称为 "变分参数"，优化就是寻找最优变分参数值。

这里我们将生成模型的所有参数均看作是隐变量（随机变量）而不是固定不变的未知参数，例如之前的 $p(\mathbf z, \mathbf x;\theta)=p(\mathbf z;\theta)p(\mathbf x|\mathbf z;\theta)$ 中，模型参数 $\theta$ 是固定不变的未知参数，这里则将 $\theta$ 和 $\mathbf z$ 均看作隐变量，前者是全局的，后者则是局部的针对单个数据点的随机变量。将隐变量统一到符号 $\mathbf z$ 中。

## 1.2 ELBO

根据 (3) 式，目标函数为

$$KL(q(\mathbf z)||p(\mathbf z|\mathbf x))=\mathbb E[\log q(\mathbf z)]-\mathbb E[\log p(\mathbf z|\mathbf x)]=\mathbb E[\log q(\mathbf z)]-\mathbb E[\log p(\mathbf z, \mathbf x)]+\log p(\mathbf x) \tag{4}$$

注意 (4) 式中的期望是 w.r.t. $q(\mathbf z)$。由于 $\log p(\mathbf x)$ 与 $q(\mathbf z)$ 无关，故最小化 KL 等价于最大化 ELBO，

$$ELBO(q) \stackrel{\Delta}=\mathbb E[\log p(\mathbf z, \mathbf x)] - \mathbb E[\log q(\mathbf z)] \tag{5}$$

ELBO 与 log evidence 的关系（$p(\mathbf x)$ 称为 evidence 证据），

$$\log p(\mathbf x) = KL(q(\mathbf z)||p(\mathbf z|\mathbf x)) + ELBO(q) \tag{6}$$

由于 $KL(\cdot) \ge 0$， 故有 

$$\log p(\mathbf x) \ge ELBO(q) \tag{7}$$

### 1.2.1 EM 算法
EM 算法用于求带有隐变量模型的最大似然估计 

$$\theta^{\star}= \arg \max_{\theta} \ \log p(\mathbf x)=\arg \max_{\theta} \ \log \sum_{\mathbf z} p(\mathbf x,\mathbf z;\theta) \tag{8}$$

根据 (7) 和 (5) 式，此优化目标函数 $\log p(\mathbf x)$ 满足 

$$\log p(\mathbf x) \ge \sum_{\mathbf z} q(\mathbf z) \log p(\mathbf x, \mathbf z;\theta)-\log q(\mathbf z)=ELBO(q)$$

当 $q(\mathbf z) = p(\mathbf z|\mathbf x)$ 时，ELBO 等于 $\log p(\mathbf x)$，(7) 式中等号成立，故再最大化 ELBO 中第一项即可，第二项 $\mathbb E[\log q(\mathbf z)]$ 与参数 $\theta$ 无关。迭代步骤：
1. 根据 $\theta^{(t)}$ 计算 $q(\mathbf z)$，即计算 $p(\mathbf z|\mathbf x; \theta^{(t)})$
2. 根据 $q(\mathbf z)$ 计算 $\theta^{(t+1)}=\arg\max_{\theta} \mathbb E[\log p(\mathbf x, \mathbf z;\theta)]$

以上两个步骤交替进行（迭代 N 次或 ELBO 改变较小时退出循环）。

EM 算法可行的条件：上述两个步骤中的表达式可求。

变分推断由于将模型参数 $\theta$ 也看作是隐变量，故无法计算 $p(\mathbf z|\mathbf x)$。下面使用一个例子说明。

### 1.2.2 VI 例子

__高斯分布的贝叶斯混合__

考虑一个具有单位标准差的单变量高斯分布的贝叶斯混合，共有 $K$ 个高斯混合成分，高斯成分的期望为 $\{\mu_1,\ldots, \mu_K\}$，期望并非是一个固定参数，而是随机变量，先验概率分布为 $\mu_k \sim \mathcal N(0, \sigma^2), \ k=1,\ldots, K$，其中 $\sigma^2$ 是超参数。生成数据 $x_i$ 时，首先根据多项分布得到一个分配向量 $c_i \in \mathbb N^K$，且是一个 one-hot vector，为 `1` 的元素指示使用对应的高斯成分进行采样，多项分布的先验是 $p(k)=1/K, \ k=1,\ldots, K$，得到 `i-th` 数据 $x_i$，如此共采样 $n$ 个数据，整个过程为

$$\begin{aligned} \mu_k & \sim \mathcal N(0, \sigma^2), & \quad &  k=1,\ldots, K
\\ c_i & \sim Cat(1/K,\ldots, 1/K), & \quad & i=1,\ldots,n
\\ x_i|c_i,\boldsymbol \mu & \sim \mathcal N(c_i^{\top}\boldsymbol \mu, 1), & \quad & i=1,\ldots, n
\end{aligned} \tag{9}$$


联合概率密度为 

$$p(\boldsymbol \mu, \mathbf c, \mathbf x)=p(\boldsymbol \mu)\prod_{i=1}^n p(c_i) p(x_i|c_i,\boldsymbol \mu) \tag{10}$$

其中隐变量为 $\mathbf z=\{\boldsymbol \mu, \mathbf c\}$。可以写出 evidence 为

$$p(\mathbf x)=\int p(\boldsymbol \mu) \prod_{i=1}^n \sum_{c_i}p(c_i)p(x_i|c_i,\boldsymbol \mu) d\boldsymbol \mu \tag{11}$$

上式中，有求和、连乘，还有积分，显然无法计算出 close form 的结果，也就无法根据 (2) 式计算 $p(\mathbf z|\mathbf x)$。


__高斯混合__
当然如果我们将 $c_i$ 看作隐变量 $z_i$，然后 $\mu_k$ 和多项分布的 $p(k)$ 看作模型参数 $\theta$，那么问题就是普通的高斯混合， evidence 为

$$p(\mathbf x)=\prod_{i=1}^n \sum_{c_i} p(c_i) p(x_i|c_i,\boldsymbol \mu)$$

仍可以根据 EM 算法求解模型参数 $\theta$。根据 (2) 式可知

$$p(\mathbf z|\mathbf x)=\frac {p(\mathbf z, \mathbf x)}{p(\mathbf x)}=\frac {\prod_{i=1}^n  p(z_i) p(x_i|z_i,\boldsymbol \mu)}{\prod_{i=1}^n \sum_{z_i} p(z_i) p(x_i|z_i,\boldsymbol \mu)}=\prod_{i=1}^n p(z_i|x_i)$$


## 1.3 mean-field 变分族

为了求解 (3) 式问题，我们首先考虑一族分布：mean-field variational family，指的是隐变量相互独立，如下

$$q(\mathbf z)=\prod_{j=1}^m q_j(z_j) \tag{12}$$

回顾前面的高斯分布贝叶斯混合的例子，使用 mean-field 变分族来近似后验分布则是 

$$q(\boldsymbol \mu, \mathbf c)=\prod_{k=1}^K q(\mu_k;m_k,s_k^2) \prod_{i=1}^n q(c_i;\phi_i) \tag{13}$$

$\mathbf m=(m_1,\ldots, m_K), \ \mathbf s^2=(s_1^2,\ldots, s_K^2)$ 是高斯型随机变量 $(\mu_1, \ldots, \mu_K)$ 的分布参数。$\phi_i \in \mathbb R_{\ge 0}^K$ 是一个 K 维向量，表示第 $i$ 个成分 $c_i$ （是 one-hot 向量）的多分类分布参数。

## 1.4 坐标上升mean-field变分推断

coordinate ascent variational inference(CAVI)：迭代进行优化某个变分因子 $q_j(z_j)$ 时，保持其他因子固定不变。

考虑 `j-th` 隐变量 $z_j$，其条件分布为 $p(z_j|\mathbf z_{-j}, \mathbf x)$，固定其他隐变量的变分因子 $q_l(z_l), \ l \neq j$，那么根据 CAVI 算法可计算出最优 $q_j(z_j)$ 为

$$q_j^{\star}(z_j) \propto \exp\{\mathbb E_{-j}[\log p(z_j|\mathbf z_{-j}, \mathbf x)]\} \tag{14}$$

**证明：**

由于 $\mathbb E_{-j}[\log p(z_j|\mathbf z_{-j}, \mathbf x)]=\mathbb E_{-j}[\log p(z_j, \mathbf z_{-j},\mathbf x)]-\mathbb E_{-j}[\log p( \mathbf z_{-j},\mathbf x)]$，其中 $\mathbb E_{-j}[\log p(\mathbf z_{-j},\mathbf x)]$ 对于 $z_j$ 而言由于已经固定不变，可以看作是常数，故根据 (14) 式可以得到

$$q_j^{\star}(z_j) \propto \exp\{\mathbb E_{-j}[\log p(z_j,\mathbf z_{-j}, \mathbf x)]\} \tag{15}$$

下面我们给出 (14)、(15) 式具体的推导过程。从 (5) 式最大化 ELBO 出发进行推导，由于迭代时只更新 $z_j$ 变分因子 $q_j(z_j)$，其他隐变量的变分因子 $q_l(z_l), \ l \neq j$ 则保持固定，那么我们将不包含 $z_j$ 的项剥离出来，视作常数，

$$\begin{aligned}ELBO(q)&=\int q(\mathbf z)[\log p(\mathbf x,\mathbf z) - \log q(\mathbf z)] d\mathbf z
\\&=\int \prod_i q_i(z_i) \log p(\mathbf x,\mathbf z)d\mathbf z-\int \left(\prod_i q_i(z_i) \right) \left(\log \prod_i q_i(z_i)\right) d\mathbf z
\\&=\int q_j(z_j) [\int \prod_{i \neq j} q_i(z_i) \log(\mathbf x,\mathbf z) d \mathbf z_{-j}] d z_j - \int \sum_l \log q_l(z_l)\left(\prod_i q_i(z_i) \right) d\mathbf z
\\&=\int q_j(z_j) \mathbb E_{-j} [\log p(\mathbf x,\mathbf z)] dz_j-\int\sum_l q_l(z_l)\log q_l(z_l) \left(\prod_{i\ne l} q_i(z_i) d\mathbf z_{-l}\right) dz_l
\\&=\int q_j(z_j) \mathbb E_{-j} [\log p(\mathbf x,\mathbf z)] dz_j-\sum_l \int q_l(z_l)\log q_l(z_l) \left(\int \prod_{i\ne l} q_i(z_i) d\mathbf z_{-l}\right) dz_l
\\&=\int q_j(z_j) \mathbb E_{-j} [\log p(\mathbf x,\mathbf z)] dz_j-\sum_l \int q_l(z_l) \log q_l(z_l) d z_l
\\&=\int q_j(z_j) \mathbb E_{-j} [\log p(\mathbf x,\mathbf z)] dz_j-\int q_j(z_j) \log q_j(z_j)dz_j-\sum_{l \ne j} \int q_l(z_l) \log q_l(z_l) dz_l
\\&= -KL(q_j(z_j)|| \exp \{\mathbb E_{-j} [\log p(\mathbf x,\mathbf z)]\}) + const
\end{aligned}$$

于是，在其他隐变量的变分因子 $q_l(z_l), \ l \neq j$ 则保持固定的条件下，当满足下式时， ELBO 最大。

$$q_j^{\star}(z_j)=\exp \{\mathbb E_{-j} [\log p(\mathbf x,\mathbf z)]\}=\exp \{\mathbb E_{-j} [\log p(z_j, \mathbf z_{-j}, \mathbf x)]\} \tag{16}$$ 

为了保证 $\int q_j(z_j) dz_j=1$，对上式做归一化，于是 (15) 式得证。

$$q_j^{\star}(z_j)= \frac {q_j(z_j)}{\int q_j(z_j) dz_j} \propto \exp \{\mathbb E_{-j} [\log p(\mathbf x,\mathbf z)]\}$$

**CAVI 算法：**
___
输入： 模型 $p(\mathbf x, \mathbf z)$，观测数据集 $\mathbf x$
输出： 变分概率密度 $q(\mathbf z)=\prod_{i=1}^m q_i(z_i)$
初始化：各个变分因子 $q_i(z_i), \ i=1,\ldots, m$
步骤：
**while** ELBO 未收敛 **do**
| &emsp; **for** $j=1,\ldots ,m$ **do**
| &emsp; | &emsp; $q_j(z_j) \propto \exp \{\mathbb E_{-j} [\log p(\mathbf x,\mathbf z)]\}$
| &emsp; **end**
| &emsp; $ELBO(q)=\mathbb E[\log p(\mathbf z,\mathbf x)]-\mathbb E[\log q(\mathbf z)]$
**end**
**return** $q(\mathbf z)$
___

## 1.5 CAVI 算法应用

__高斯分布的贝叶斯混合__

(9) 式描述了隐变量 $(\boldsymbol \mu, \mathbf c)$ 的先验分布和数据似然分布，

隐变量 $\mathbf z=(\boldsymbol \mu, \mathbf c)$，变分概率分布为 (13) 式，根据 (5) 式有

$$\begin{aligned}ELBO(q)&=\mathbb E[\log p(\mathbf x|\boldsymbol \mu, \mathbf c)] +\mathbb E[\log p(\boldsymbol \mu)] +\mathbb E[\log p(\mathbf c)] - \mathbb E[\log q(\boldsymbol \mu)] - \mathbb E[\log q(\mathbf c)]
\\&= \sum_{i=1}^n(\mathbb E[\log p(x_i|c_i,\boldsymbol \mu); \phi_i, \mathbf m, \mathbf s^2]+\mathbb E[\log p(c_i);\phi_i])+ \sum_{k=1}^K \mathbb E[\log p(\mu_k); m_k,s_k^2] 
\\& \quad - \sum_{k=1}^K \mathbb E[\log q(\mu_k; m_k,s_k^2)] - \sum_{i=1}^n \mathbb E[\log q(c_i;\phi_i)]
\end{aligned} \tag{17}$$



(17) 式中，分号 `;` 后面的是参数，不是随机变量。

求期望是求目标函数 w.r.t $q(\boldsymbol \mu, \mathbf c)$ 的期望，也就是 w.r.t. 目标函数中隐变量的变分分布的期望（因为其他无关的隐变量积分为 1），期望中的目标函数中的**隐变量的相关参数**位于 `;` 后面。

### 1.5.1 成分选择的变分概率

###
首先求解隐变量 $c_i$ 的变分概率，此时固定其他隐变量的变分因子 $q_l(z_l)$，那么

$$\begin{aligned}q^{\star}(c_i;\phi_i)&=\exp  \{\mathbb E_{-c_i} [\log p(\mathbf x,\mathbf z)]\}\\&=\exp  \{\mathbb E_{-c_i} [\log p(x_i|c_i,\boldsymbol \mu)]+\mathbb E_{-c_i}[\log p(c_i)]+const\}
\\&\propto \exp \{\log p(c_i)+\mathbb E[\log p(x_i|c_i,\boldsymbol \mu);\mathbf m, \mathbf s^2]\}
\end{aligned} \tag{18}$$


（18) 式推导中，$\mathbb E_{-c_i}[\log p(\boldsymbol \mu)]$ 和 $\mathbb E_{-c_i}[\log p(c_{j\ne i})]$ 均固定不变，可视作常数。$\mathbb E_{-c_i}[\log p(c_i)]=\log p(c_i)$ ，(18) 式最右端表达式为非归一化变分概率，其中第一项 $\log p(c_i)=-\log K$，这是 $c_i$ 的 log 先验概率，是一个常数 const，忽略。

第二项中，

$$p(x_i|c_i,\boldsymbol \mu)=\prod_{k=1}^K p(x_i|\mu_k)^{c_{ik}} \tag{19}$$

于是这个第二项可写为

$$\begin{aligned}\mathbb E[\log p(x_i|c_i,\boldsymbol \mu);\mathbf m, \mathbf s^2]&=\mathbb E[\sum_k c_{ik} \log p(x_i|\mu_k); \mathbf m, \mathbf s^2]
\\&=\sum_k c_{ik} \{\mathbb E[-\frac 1 2 (x_i-\mu_k)^2; m_k,s_k^2] -\frac 1 2 \log( 2\pi) \}
\\&=\sum_k c_{ik} \{\mathbb E[\mu_k; m_k,s_k^2] \cdot x_i - \mathbb E[\mu_k^2; m_k,s_k^2]\cdot \frac 1 2 - \frac 1 2 [x_i^2+\log (2\pi)]\}
\\&=\sum_k c_{ik} (\mathbb E[\mu_k; m_k,s_k^2] \cdot x_i - \mathbb E[\mu_k^2; m_k,s_k^2]\cdot \frac 1 2) - \frac 1 2 [x_i^2+\log (2\pi)]
\end{aligned} \tag{20}$$
其中 $\frac 1 2 [x_i^2+\log (2\pi)]$ 视作常数。

综上，

$$\phi_{ik}=q^{\star}(c_{ik}=1) \propto \exp \{\mathbb E[\mu_k; m_k,s_k^2] \cdot x_i - \mathbb E[\mu_k^2; m_k,s_k^2]\cdot \frac 1 2\} \tag{21}$$

(21) 式中，$\mathbb E[\mu_k; m_k,s_k^2]$ 是 $\mu_k$ 的期望，为 $m_k$；$\mathbb E[\mu_k^2; m_k,s_k^2]$ 为二阶原点矩，根据 $\mathbb E[X^2]=V+\mathbb E^2[X]$ 可知，其值为 $s_k^2+m_k^2$，故 (21) 式转换为

$$\phi_{ik} \propto \exp(x_i m_k -\frac 1 2 (s_k^2+m_k^2))$$

### 1.5.2 高斯分量的变分概率

仿照 $q(c_i)$ 求 $q(\mu_k)$，从 (16) 式的展开式中不包含 $\mu_k$ 的项视作 const，易得如下结果

$$q(\mu_k) \propto \exp\{\log p(\mu_k) + \sum_{i=1}^n \mathbb E[\log p(x_i|c_i,\boldsymbol \mu); \phi_i, \mathbf m_{-k}, \mathbf s_{-k}^2]\} \tag{22}$$

记住，期望是指 $\mathbb E_{-\mu_k} [\cdot]$，故 $\mathbb E_{-\mu_k} [\log p(\mu_k)]=\log p(\mu_k)$ ，保留含有 $\mu_k$ 的项，其他项均固定不变视作常数。

将 (20) 式进一步展开，注意到 $\phi_{ik}=\mathbb E[c_{ik}; \phi_i]$，于是

$$\begin{aligned} \log q(\mu_k) &= \log p(\mu_k) + \sum_i \mathbb E[\log p(x_i|c_i,\boldsymbol \mu); \phi_i, \mathbf m_{-k}, \mathbf s_{-k}^2] + const
\\&=\log p(\mu_k)+\sum_i \mathbb E[c_{ik} \log p(x_i|\mu_k); \phi_i] + const
\\&=-\mu_k^2/2\sigma^2 + \sum_i \mathbb E[c_{ik}; \phi_i] \cdot \log p(x_i|\mu_k) + const
\\&=-\mu_k^2/2\sigma^2+ \sum_i \phi_{ik} (-(x_i-\mu_k)^2/2) + const
\\&=-\mu_k^2/2\sigma^2+ \sum_i \phi_{ik} x_i \mu_k - \phi_{ik} \mu_k^2/2 + const
\\&=\left(\sum_i \phi_{ik} x_i\right)\mu_k-\frac 1 2 \left(1/\sigma^2 + \sum_i \phi_{ik}\right)\mu_k^2 + const
\end{aligned} \tag{23}$$

(23) 式第二个等式的推导中，将 $\mathbb E[\cdot]$ 中与 $\mu_k$ 无关的项提出去作为常数。后面的等式推导，同样是如此处理。注意：

1. 数据采用分布 $x_i \sim \mathcal N(c_i^{\top} \boldsymbol \mu, 1)$，先验概率 $\mu_k \sim \mathcal N(0, \sigma^2)$。

后验概率的假设形式为 $q(\mu_k) \sim \mathcal N(m_k,s_k^2)$，取对数，$\log q(\mu_k)=-(\mu_k - m_k)^2/2s_k^2$ ，与上式对比可得，


$$m_k = \left(\sum_i \phi_{ik} x_i\right)\left(1/\sigma^2 + \sum_i \phi_{ik}\right)^{-1}, \quad s_k^2 = \left(1/\sigma^2 + \sum_i \phi_{ik}\right)^{-1} \tag{24}$$


### 1.5.3 计算 ELBO

计算出 $\phi_{ik}, m_k ,s_k^2$ 之后， 根据 (17) 式计算 ELBO，下面分别计算各项，注意，求期望是求目标函数 w.r.t $q(\boldsymbol \mu, \mathbf c)$ 的期望，

$$\mathbb E[\log p(c_i);\phi_i])=\mathbb E[-\log K;\phi_i]=-\log K$$

$$\begin{aligned}\mathbb E[\log p(x_i|c_i,\boldsymbol \mu); \phi_i, \mathbf m, \mathbf s^2]&=\mathbb E[\sum_k c_{ik} \log p(x_i|\mu_k); \phi_i, \mathbf m, \mathbf s^2]
\\&=\sum_k \mathbb E[c_{ik}(-\frac 1 2 (x_i-\mu_k)^2-\frac 1 2 \log (2\pi)); \phi_i, m_k,s_k^2]
\\&=\sum_k \mathbb E[x_i \mu_k c_{ik}-\frac 1 2 c_{ik} \mu_k^2;\phi_i,m_k,s_k^2]-\frac 1 2 (x_i^2+\log (2\pi)) \mathbb E[c_{ik}; \phi_i]
\\&=\sum_k x_i \mathbb E[\mu_k; m_k,s_k^2] - \frac 1 2 \mathbb E [\mu_k^2;m_k,s_k^2] -\frac 1 2 (x_i^2+\log (2\pi)) \phi_{ik}
\\&=-\frac 1 2 (x_i^2+\log (2\pi)) \phi_{ik} + \sum_k  x_i m_k -\frac 1 2 (s_k^2+m_k^2)
\end{aligned}$$

$$\begin{aligned}\mathbb E[\log p(\mu_k); m_k,s_k^2]&=\mathbb E[-\frac {\mu_k^2}{2\sigma^2}-\frac 1 2 \log (2\pi \sigma^2); m_k,s_k^2]
\\&=-\frac 1 2 \log (2\pi \sigma^2)-\frac 1 {2\sigma^2} \mathbb E [\mu_k^2]
\\&=-\frac 1 2 \log (2\pi \sigma^2)-\frac 1 {2\sigma^2} (m_k^2+s_k^2)
\end{aligned}$$

$$\begin{aligned}\mathbb E[\log q(\mu_k; m_k,s_k^2)]&=\mathbb E[-\frac {(\mu_k -m_k)^2}{2 s_k^2} - \frac 1 2 \log(2\pi s_k^2);m_k,s_k^2]
\\&=-\frac 1 {2s_k^2}\mathbb E[2m_k \mu_k-\mu_k^2;m_k,s_k^2]-\frac 1 2 [\frac {m_k^2}{s_k^2}+\log (2\pi s_k^2)]
\\&=-\frac 1 {2s_k^2}(2m_k^2-(m_k^2+s_k^2))-\frac 1 2 [\frac {m_k^2}{s_k^2}+\log (2\pi s_k^2)]
\\&=-\frac 1 2(1+ \log (2\pi s_k^2))\end{aligned}$$

$$\begin{aligned}\mathbb E[\log q(c_i;\phi_i)]&=\mathbb E[\log \prod_k \phi_{ik}^{c_{ik}}; \phi_i]
\\&=\mathbb E[\sum_k c_{ik} \log \phi_{ik};\phi_i]
\\&=\sum_k \mathbb E[c_{ik} \log \phi_{ik}; \phi_{ik}]
\\&=\sum_k \phi_{ik} \log \phi_{ik}
\end{aligned}$$

**预测新数据的概率密度：**

$$p(x^{\star}|\mathbf x) \approx \frac 1 K \sum_{k=1}^K p(x^{\star}|m_k) \tag{22}$$

这里生成新数据依然服从 $x^{\star} \sim \mathcal N(c^{\star \top} \boldsymbol \mu, 1)$，使用 $\mathbf m$ 作为 $\boldsymbol \mu$ 的估计。注意因为我们的变分概率 $\phi_i=q(c_i)$ 是数据样本相关的，即具有下标 $i$，用哪一个 $\phi_i$ 显然都不合适，所以对于新数据，只能使用先验分布 $c^{\star} \sim Cat(1/K,\ldots, 1/K)$，而 $\boldsymbol \mu$ 则是所有样本共享，故可以使用 $\boldsymbol \mu$ 的后验分布的最大似然估计 $\mathbf m$ 。

---
算法 2： 高斯混合模型的 CAVI

输入：数据 $\mathbf x_{1:n}$，高斯成分数量 $K$，高斯成分的先验 $\mu_k \sim \mathcal N(0, \sigma^2)$

输出：变分分布 $q(\mu_k;m_k,s_k^2)$ （高斯型）和分类分布 $q(c_i;\phi_i)$ （K分类）

初始化：变分分布的参数 $\mathbf m_{1:K}, \ \mathbf s_{1:K}^2, \ \boldsymbol \phi_{1:n}$

**while** ELBO 未收敛 **do**
| &emsp; **for** $i=1,\ldots ,n$ **do**
| &emsp; | &emsp; set $\phi_{ik} \propto \exp(x_i m_k -\frac 1 2 (s_k^2+m_k^2))$
| &emsp; **end**
| &emsp; **for** $k =1,\ldots, K$ **do**
| &emsp; | &emsp; 按 (24) 式计算 $m_k, \ s_k^2$
| &emsp; **end**
| &emsp; 计算 ELBO
**end**

---