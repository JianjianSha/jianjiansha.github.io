---
title: 指数族分布
date: 2022-06-02 18:30:22
tags: math
mathjax: true
---
# 1. 定义

指数族分布具有如下形式，

$$p(x|\eta)=h(x) \exp \{\eta^ {\top} t(x)-a(\eta)\} \tag{1}$$

其中：
1. $\eta$ 为特性参数向量
2. $t(x)$ 为充分统计量，表示仅通过 $t(x)$ 就可获得参数 $\eta$ 的似然。
3. $h(x)$ 表示测度，例如计数测度或 Lebesgue 测度
4. $a(\eta)$ 为对数配分函数（对数归一化项）

    $$a(\eta)=\log \int h(x) \exp\{\eta^ {\top} t(x)\} dx$$

# 2. 例子

## 2.1 高斯分布

一元高斯分布
$$p(x|\mu, \sigma^ 2)=\frac 1 {\sqrt {2\pi} \sigma^ 2} \exp \{-\frac {(x-\mu)^ 2}{2\sigma^ 2}\} \tag{2}$$

改写上式为

$$p(x|\mu, \sigma^ 2)=\frac 1 {\sqrt {2\pi}} \exp \{\frac {\mu} {\sigma^ 2}x - \frac 1 {2\sigma^ 2} x^ 2-\frac 1 {2\sigma^ 2} \mu^ 2 - \log \sigma\} \tag{3}$$

对比 (1) 式可知，

$$\eta=[\frac {\mu} {\sigma^ 2}, -\frac 1 {2\sigma^ 2}]^ {\top}
\\\\ t(x)=[x, x^ 2]^ {\top}
\\\\ a(\eta)=\frac {\mu^ 2} {2\sigma^ 2}+\log \sigma=-\frac {\eta_1^ 2}{4\eta^ 2}-\frac 1 2 \log (-2\eta_2)
\\\\ h(x)=\frac 1 {\sqrt {2\pi}}$$

## 2.2 Bernoulli 分布

$$p(x|\pi)=\pi^ x(1-\pi)^ {1-x} \tag{4}$$

改写上式

$$\begin{aligned}p(x|\pi)&=\exp \{x\log \pi + (1-x)\log (1-\pi) \}
\\&=\exp \{x \log \frac {\pi}{1-\pi} + \log (1-\pi)\}
\end{aligned}$$

对比 (1) 式可知，

$$\eta = \log \frac {\pi} {1-\pi}
\\\\ t(x)=x
\\\\ a(\eta)=-\log (1-\pi)=\log (1+e^ {\eta})
\\\\ h(x)=1$$

## 2.3 多项分布

记多项分布中分类数量为 $M$，每个分类的概率为 $\pi_k, \ k=1,\ldots ,K$，满足 $\sum_{k=1}^ K \pi_k=1$，试验次数为 $M$，$\mathbf x=(x_1,\ldots, x_K)$ 表示各分类出现的次数，

$$p(\mathbf x|\pi)=\frac {M!}{x_1!\cdots x_K!} \prod_{k=1}^ K \pi_k^ {x_k} \tag{5}$$

改写上式为

$$\begin{aligned} p(x|\pi)&=\frac {M!}{x_1!\cdots x_K!} \exp \{\sum_{k=1}^ K x_k \log \pi_k\}
\\\\ &=\frac {M!}{x_1!\cdots x_K!} \exp \{\sum_{k=1}^ {K-1} x_k \log \pi_k + x_K \log \pi_K\}
\\\\ &=\frac {M!}{x_1!\cdots x_K!} \exp\{\sum_{k=1}^ {K-1} x_k \log \pi_k +(M-\sum_{k=1}^ {K-1} x_k)\log(1-\sum_{k=1}^ {K-1}\pi_k)\}
\\\\ &=\frac {M!}{x_1!\cdots x_K!} \exp\{\sum_{k=1}^ {K-1}\log (\frac {\pi_k}{1-\sum_{k=1}^ {K-1}\pi_k}) x_k + M\log (1-\sum_{k=1}^ {K-1} \pi_k)\}
\end{aligned}$$

于是

$$\eta_k=\log (\frac {\pi_k}{1-\sum_{k=1}^ {K-1}\pi_k})=\log \frac {\pi_k}{\pi_K}, \ k=1,\ldots, K-1
\eta_K=0
\\\\ t(x) _ k=x _ k, \ k=1,\ldots, K
\\\\ h(x)=\frac {M!}{x _ 1!\cdots x _ K!}
\\\\ a(\eta)=-M\log (1-\sum_{k=1}^ {K-1} \pi_k)=M\log(\sum_{k=1}^ K e^ {\eta_k})
$$

## 2.4 Poisson 分布

$$p(x|\lambda)=\frac {\lambda^ x e^ {-\lambda}} {x!}, \ x=0,1,\ldots \tag{6}$$

改写上式为

$$p(x|\lambda)=\frac 1 {x!} \exp\{x \log \lambda - \lambda \}$$

对比可知

$$\eta = \log \lambda
\\\\ t(x)=x
\\\\ a(\eta)=\lambda = e^ {\eta}
\\\\ h(x)=\frac 1 {x!}$$

# 3. 矩(Moments)

对数归一化项的导数就是充分统计量的一阶矩，

$$\begin{aligned}\frac d {d \eta} a(\eta)&=\frac d {d\eta} \left(\log \int \exp \{\eta^ {\top} t(x)\} h(x) dx \right)
\\\\ &=\frac {\int t(x) \exp\{\eta^ {\top} t(x)\}h(x)dx}{\int \exp\{\eta^ {\top} t(x)\}h(x)dx}
\\\\ &=\int t(x) \exp \{\eta^ {\top}t(x) - a(\eta)\} h(x) dx
\\\\ &= \mathbb E[t(x)]
\end{aligned}$$

**高阶导则对应高阶矩**，例如

$$\frac {d^ 2} {d \eta^ 2} a(\eta)=\mathbb V[t(x)]=\mathbb E[t(x)^ 2] - \mathbb E[t(x)]^ 2$$



# 4. 共轭

分布族 $\mathcal P$ 与分布族 $\mathcal M$，对来自 $\mathcal M$ 中的似然，和来自 $\mathcal P$ 中的先验，如果应用贝叶斯定理，得到的后验分布也来自 $\mathcal P$，那么称 $\mathcal P$ 共轭于 $\mathcal M$。（后验与先验具有相同的函数形式）

例如：

|似然|先验|
|--|--|
|Gaussian|Gaussian|
|multinomial|Dirichlet|
|Bernoulli|Beta|
|Poisson|Gamma|



考虑以下设定：

特性参数先验分布 $\eta \sim F(\cdot|\lambda)$

数据似然分布 $x_i \sim G(\cdot |\eta), \ i=1,\ldots, n$

根据贝叶斯定理，$\eta$ 的后验分布为

$$p(\eta|x_{1:n}, \lambda) \propto F(\eta|\lambda) \prod_{i=1}^ n G(x_i|\eta) \tag{7}$$


> 理论上每个指数分布均有对应的共轭先验。


**一个例子**

假设似然和先验分别具有如下形式：

$$p(x|\eta)=h_l(x) \exp \{\eta^ {\top} t(x) -a_l(\eta)\} \tag{8}$$

$$p(\eta|\lambda)=h_c(\eta) \exp \{\lambda_1^ {\top} \eta + \lambda_2^ {\top} (-a_l(\eta)) - a_c(\lambda)\} \tag{9}$$

对于先验分布有以下几点：
1. 特性参数 $\lambda = \begin{bmatrix}\lambda_1\\ \lambda_2\end{bmatrix}$ 的维度为 $\dim(\eta)+1$
2. 充分统计量为 $\begin{bmatrix} \eta \\ -a(\eta) \end{bmatrix}$

根据 (8) (9) 式计算后验，

$$\begin{aligned} p(\eta|x_{1:n}, \lambda) & \propto p(\eta|\lambda) \prod_{i=1}^ n p(x_i|\eta)
\\\\ &=h_c(\eta) \exp \{\lambda_1^ {\top} \eta + \lambda_2^ {\top} (-a_l(\eta)) - a_c(\lambda)\} \left(\prod_{i=1}^ n h_l(x_i) \exp \{\eta^ {\top} t(x_i) -a_l(\eta)\}\right)
\\\\ &=h_c(\eta) \exp \{\lambda_1^ {\top} \eta + \lambda_2^ {\top} (-a_l(\eta)) - a_c(\lambda)\} \left(\prod_{i=1}^ n h_l(x_i)\right) \exp \{\eta^ {\top}\sum_{i=1}^ n t(x_i) - na_l(\eta)\}
\\\\ & \propto h_c(\eta) \exp \{(\lambda_1 + \sum t(x_i))^ {\top} \eta + (\lambda_2+n)(-a(\eta))\}
\end{aligned}$$

上式推导为了简洁，省去了与 $\lambda, \eta$ 无关的因子，这个后验分布也是指数族，特性参数为

$$\hat \lambda_1 = \lambda_1 + \sum_{i=1}^ n t(x_1)
\\\\ \hat \lambda_2 = \lambda_2 + n$$

