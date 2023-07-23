---
title: 共轭先验
date: 2022-05-28 15:12:42
tags: math
mathjax: true
---

如果后验分布与先验分布类型相同，那么称先验共轭于似然函数，即 **如果 $p(z)$ 与 $p(x|z)$ 共轭** ，
那么 $p(z|x)$ 与 $p(z)$ 分布类型相同。这里 $x$ 是观测变量，$z$ 是隐变量。

# 1 二项分布和 Beta 分布


## 1.1 二项分布

伯努利(0-1)分布 $p(x|\mu)=\mu^x (1-\mu)^{1-x}$

二项分布，即独立重复 $n$ 次伯努利分布，那么 

$$p(n_1;n,\mu)=\begin{pmatrix} n \\\\ n _ 0 n _ 1\end{pmatrix} \mu^{n _ 1}(1-\mu) ^ {n_0}$$

## 1.2 Beta 分布

$$Beta(\mu;a,b)=\frac {\Gamma(a+b)}{\Gamma(a)\Gamma(b)} \mu^{a-1} (1-\mu)^{b-1}$$

$\mu$ 的后验分布为

$$\begin{aligned} p(\mu|X)&=\frac {p(X|\mu)p(\mu;a,b)}{\int p(X|\mu)p(\mu;a,b)d\mu}\\\\ &=\frac {\mu ^ {n_1+a-1}(1-\mu)^{n_0+b-1}}{\int \mu^{n_1+a-1}(1-\mu) ^ {n_0+b-1} d\mu}
\\\\ &=Beta(\mu;a+n_1,b+n_0)
\end{aligned}$$



# 2 多项分布和 Dirichlet 分布

## 2.1 多项分布
$\mathbf x$ 是一个 one-hot vector，服从类别分布，那么

$$p(\mathbf x|\boldsymbol \theta) = \prod_{k=1}^K \theta_k^{x_k}$$

其中 $\sum_k \theta_k =1, \ \theta_k \ge 0, \ k=1,\ldots, K$。

记随机变量 $X=(X_1,\ldots, X_K)$ 表示 n 次试验后每种结果出现的次数，那么概率

$$P(X_1=n_1,\ldots, X_K=n_K)=\begin{pmatrix}n \\\\ n_1\cdots n_K\end{pmatrix}\prod_{k=1}^K \theta_k^{n_k}$$

1. 类别分布 $\mathbf x \sim Cat(\boldsymbol \theta)$
2. 多项分布 $X \sim Mult(n, \boldsymbol \theta)$
3. Dirichlet 分布 $\boldsymbol \theta \sim Dir(\boldsymbol \alpha)$

## 2.2 Dirichlet 分布


$$p(\boldsymbol \theta;\boldsymbol \alpha)=\frac {\Gamma(\sum_{k=1}^K \alpha_k)}{\prod_{k=1}^K \Gamma(\alpha_k)} \prod_{k=1}^K \theta_k^{\alpha_k-1}, \quad \alpha_k > 0$$

其中 $\sum_k \theta_k=1, \ \theta_k \ge 0$。

为了方便，定义

$$B(\boldsymbol \alpha)\stackrel{\Delta}=\frac {\prod_{k=1}^K \Gamma(\alpha_k)}{\Gamma(\sum_{k=1}^K \alpha_k)}
$$

$B(\boldsymbol \alpha)$ 又称为多元贝塔函数或扩展贝塔函数，

$$\int p(\boldsymbol \theta;\boldsymbol \alpha) d \boldsymbol \theta=1 \Rightarrow \int \frac 1 {B(\boldsymbol \alpha)}\prod_{k=1}^K \theta_k^{\alpha_k-1} d \boldsymbol \theta=1 \Rightarrow B(\boldsymbol \alpha)=\int \prod_{k=1}^K \theta_k^{\alpha_k-1}d\boldsymbol \theta$$

$\boldsymbol \theta$ 的后验概率为

$$\begin{aligned}p(\boldsymbol \theta|X)&=\frac {p(X|\boldsymbol \theta) p(\boldsymbol \theta;\boldsymbol \alpha)}{\int p(X,\boldsymbol \theta) d\boldsymbol \theta}\\\\& =\frac {\begin{pmatrix}n \\\\ n_1\cdots n_K \end{pmatrix} \prod_{k=1}^K \theta_k^{n_k} \theta_k^{\alpha_k-1} / B(\boldsymbol \alpha)}{\int \begin{pmatrix}n \\\\ n_1\cdots n_K \end{pmatrix} \prod_{k=1}^K \theta_k^{n_k} \theta_k^{\alpha_k-1} / B(\boldsymbol \alpha) d\boldsymbol \theta}\\\\&=\frac {\prod \theta_k^{n_k+\alpha-1}}{\int \prod \theta_k^{n_k+\alpha_k-1}d \boldsymbol \theta}\\\\&=\frac 1 {B(\mathbf n+\boldsymbol \alpha)} \prod_{k=1}^K \theta_k^{n_k+\alpha_k-1}
\\\\&=Dir(\boldsymbol \theta|\mathbf n + \boldsymbol \alpha)
\end{aligned}$$

# 3 高斯分布

$$p(x|\mu,\sigma^2)=\frac 1 {\sqrt {2\pi }\sigma} \exp\{-\frac 1 {2 \sigma^2} (x-\mu)^2\}$$

多维高斯分布

$$p(\mathbf x|\mu, \Sigma)=\frac 1 {(2\pi)^{n/2} |\Sigma|^{1/2}} \exp \{-\frac 1 2 (\mathbf x-\mu)^{\top} \Sigma^{-1} (\mathbf x - \mu)\}$$

对于二维情况，记

$$\mu = (\mu_1, \mu_2)$$

$$\Sigma=\begin{bmatrix} \sigma_1^2 & \rho \sigma_1 \sigma_2 \\ \rho \sigma_1 \sigma_2 & \sigma_2^2 \end{bmatrix}$$

那么条件分布

$$p(x_1|x_2)=\mathcal N(\mu_1+\rho \sigma_1/\sigma_2 (x_2-\mu_2), (1-\rho^2 )\sigma_1^2 )
\\\\ p(x_2|x_1)=\mathcal N(\mu_2+\rho \sigma_2/\sigma_1(x_1-\mu_1), (1-\rho^2 )\sigma_2^2 )$$

## 3.1 期望的共轭先验

$\mu$ 共轭先验使用高斯分布，

$$p(\mu;\mu_0,\sigma_0)=\frac 1 {\sqrt{2\pi}\sigma_0} \exp \{-\frac 1 {2 \sigma_0^2 }(\mu-\mu_0)^2 \}$$

那么后验概率为

$$\begin{aligned}p(\mu|x)&=\frac {p(x|\mu)p(\mu;\mu_0,\sigma_ 0 ^ 2 )}{p(x)}
\\\\ &\propto \exp\{-\frac 1 {2\sigma ^ 2 }(x^ 2 -2x \mu+ \mu^ 2 )-\frac 1 {2 \sigma_ 0 ^ 2}(\mu^ 2 -2\mu_ 0 \mu - \mu_ 0 ^ 2 )\}
\\\\ &=\exp\{-\frac 1 {2 \sigma^ 2 \sigma_0 ^ 2 }[(\sigma ^ 2 +\sigma_0 ^2 )\mu ^ 2 - 2(x\sigma_ 0 ^ 2 +\mu _ 0 \sigma ^ 2 )\mu + const]\}\end{aligned}$$

易知 $p(\mu|x)$ 也服从高斯分布，且有

$$E(\mu|x)=\frac {x\sigma_0^2 +\mu_0\sigma^2 }{\sigma^2 +\sigma_0^2 }, \quad Var(\mu|x)=\frac {\sigma^2 \sigma_0 ^2 }{\sigma^2 +\sigma_0 ^2 }$$

## 3.2 协方差矩阵的共轭先验

**Wishart 分布**

一个随机矩阵 $X_{n \times p}$，每一行 $X_i$ 服从多元高斯分布，

$$X_i \sim N_p(0, \Sigma)$$

令 $A_{p \times p}=\sum_i X_i^{\top} X_i$，那么 $A$ 服从 wishart 分布，

$$A \sim W_p(n, \Sigma)$$

**Inverse Wishart 分布**
如果一个正定矩阵 $B$ 的逆矩阵 $B^{-1} \sim W_p(n,\Sigma)$，那么称

$$B \sim W_p^{-1}(n, \Sigma)$$

**协方差矩阵的共轭先验**
观测数据集 $X_{n \times p}$（$n$ 个数据，每个数据 $p$ 维），且有 

$$X_i \sim N_p(0, \Sigma)$$

即，每个数据服从均值为 0 的 $p$ 维高斯分布，协方差矩阵的先验分布为

$$\Sigma \sim W_p^{-1}(m, \Omega)$$

那么协方差矩阵的后验分布为

$$\Sigma|X \sim W_p^{-1}(m+n, A+\Omega)$$

其中 $A=\sum_i X_i^{\top} X_i$。

$\Sigma^{-1}$ 的共轭先验则服从 $\Sigma^{-1} \sim W_p(m, \Omega)$。

## 3.3 三种类型的共轭先验

### 3.3.1 期望未知，方差已知

独立样本 $x_1,\ldots,x_n$ 均服从正态分布，方差为 $V$ 已知，期望 $\mu$ 未知可将其看作一个随机变量，随机变量 $X|\mu \sim \mathcal N(\mu, V)$，先验分布为 $\mu \sim \mathcal N(\mu_0, V_0)$，那么后验分布 $\mu|\mathbf x \sim \mathcal N(\mu_1, V_1)$ 可以求得，为

$$\mu_1=\left(\frac {\mu_0}{V_0}+\frac {\sum_i x_i}{V}\right) V_1, \quad V_1=\frac {V V_0}{V+n V_0}$$

证明：

$$\begin{aligned}p(\mu|\mathbf x)&=\frac {p(\mathbf x|\mu)p(\mu)}{p(\mathbf x)}
\\\\&\propto \exp\{-\frac 1 {2V}\sum_i(\mu-x_i)^2-\frac 1 {2V_0}(\mu-\mu_0)^2\}
\\\\& \propto \exp \{-\frac 1 {2VV_0} \left ((nV_0+V)\mu^2 - 2 (V_0\sum_i x_i+\mu_0 V) \mu\right) \}
\\\\&=\exp \{- \frac {nV_0+V}{2VV_0}\left[\mu^2-2 \frac {VV_0}{nV_0+V}\left(\frac {\mu_0}{V_0}+\frac {\sum_i x_i}{V}\right) \mu\right]\}
\end{aligned}$$

得证。

### 3.3.2 期望已知，方差未知

与 #3.3.1 情况一样，只是这里是期望已知，方差未知，观测变量 $X|V \sim \mathcal N(\mu, V)$，隐变量先验 $V \sim Scaled-Inv-\mathcal X^2(v_0, s_0^2)$ (这里方差是标量，而非协方差矩阵，如是后者，则写成 inverse wishart 分布，$V \sim W^{-1}(v_0, s_0^2)$）。

于是后验分布为 $V|\mathbf x \sim Scaled-Inv-\mathcal X^2(v_1, s_1^2)$，满足

$$v_1=v_0+n, \quad s_ 1 ^ 2=v _ 0  s _ 0 ^ 2+ns ^ 2$$

其中 

$$s^2=\frac 1 n \sum_{i=1}^n (x_i - \mu)^2$$

证明：

$$\begin{aligned}p(V|\mathbf x)&=\frac {p(\mathbf x|V)p(V)}{p(\mathbf x)}
\\\\& \propto \frac 1 {V^{n/2}} \exp\{-\frac 1 {2V} \sum_i (x_i-\mu) ^ 2\} \frac {(v _ 0 s _ 0 ^ 2)^ {v_0/2} e^ {-v_0s_0 ^ 2/(2V)}}{\Gamma(v _ 0/2) V ^ {v_0/2+1}}
\\\\&= \frac {(v _ 0s _ 0 ^ 2)^{v _ 0/2}}{\Gamma(v_0/2) V ^ {(v_0+n)/2+1}} \exp \{-\frac 1 {2V}(v _ 0 s _ 0 ^ 2+ns ^ 2)\}
\end{aligned}$$

根据后验分布  $V|\mathbf x \sim Scaled-Inv-\mathcal X^2(v_1, s_1^2)$，其 pdf 形式为

$$p(V|\mathbf x; v_1, s_1^2) \propto \frac 1 {V^{v_1/2+1}} \exp \{-\frac 1 {2V} (v_1 s_1^2)\}$$

上们两式比对，可知

$$v_1=v_0+n, \quad s_1 ^ 2=\frac {v_0s_0 ^ 2 +ns ^ 2}{v _ 0+n}$$

注意这里的变量含义：

1. $n$ 表示观测样本的数量 $\mathbf x_{1:n}$
2. $v_0, v_1$ 分别表示方差先验和后验分布的自由度

### 3.3.3 期望方差均未知

假设期望 $\mu$ 和方差 $V$ 的联合先验概率密度函数具有以下关系

$$f(\mu, V)=f(V) f(\mu|V)$$

根据前面的分析，方差 $V$ 先验使用 $Scaled-Inv-\mathcal X^2(v_0, s_0^2)$ 分布，等价于

1. $V \sim Inv-Gamma(v_0/2, 2/(v_0s_0^2))$

2. $1/V \sim Gamma(v_0/2, v_0s_0^2/2)$

此外，令 $f(\mu|V) \sim \mathcal N(\mu_0, V^{\star})$，因为 $\mu$ 使用高斯分布是一种常用的方法， 这里 $V^{\star}$ 应该是与 $V$ 相关的，我们可以合理的假设 $V^{\star}=V/n_0$，即 $V^{\star}$ 与 $V$ 之间是线性关系，但是 $n_0$ 是未知的。于是 $\mu$ 的先验分布形式为 $\mu|V \sim \mathcal N(\mu_0, V/n_0)$，如果我们令 $\phi= 1/ V$ 表示精度，根据前面所列的等价关系可知，$\phi \sim Gamma(v_0, v_0s_0^2/2)$，于是 $\mu$ 的先验分布也可以写为 $\mu|\phi \sim \mathcal N(\mu_0, 1/ (n_0 \phi)$。

**定义1**

给定 

$$\phi \sim Gamma(\frac {v_0} 2, \frac {v_0s_0^2}2), \quad \mu|\phi \sim \mathcal N(\mu_0, \frac 1 {n_0 \phi})$$

那么 $\mu, \phi$ 的联合概率密度记为

$$\mu, \phi \sim NormGamma(\mu_0, n_0, v_0, s_0^2)$$


**定义2**

给定 

$$V \sim Scaled-Inv-\mathcal X^2(v_0, s_0^2), \quad \mu|V \sim \mathcal N(\mu_0, \frac V {n_0})$$

那么 $\mu, V$ 的联合概率密度记为

$$\mu, V \sim Norm\mathcal X^{-2}(\mu_0, n_0, v_0, s_0^2)$$

这两个定义是等价的。

假设相互独立的观测数据 $\mathbf x_{1:n}$ 来自一个高斯分布，期望和方差未知，即 $X|\mu, V \sim \mathcal N(\mu, V)$，并且期望和方差的联合概率先验为

$$\mu,\phi \sim NormGamma(\mu_0, n_0, v_0, s_0^2)$$

其中 $\phi = 1/V$，那么后验分布

$$\mu,\phi|\mathbf x \sim NormGamma(\mu_1, n_1, v_1, s_1^2)$$

满足关系

$$\mu_1=\frac {n_0\mu_0+n \overline x}{n_0+n}, \quad n_1=n_0+n, \quad v_1=v_0+n
\\\\ s_1^2=\frac 1 {v_1}\left[(n-1)s^2+v_0 s_0^2 +\frac {nn_0} {n_1}(\overline x-\mu_0)^2\right]$$

其中 $\overline x=\sum_i x_i/n$，$s^2=\sum_i($

证明：

1.

$$s^2=\frac 1 {n-1}\sum_i^n (x_i-\overline x)^2=\frac 1 {n-1} (\sum_i x_i^2 - 2 \overline x \sum_i x_i + n \overline x^2)=\frac 1 {n-1}(\sum_i x_i^2 - n \overline x^2)$$

$$\begin{aligned}\exp[-\frac {\phi} 2 \sum_i(x_i-\mu)^2]&=\exp[-\frac {\phi} 2 (\sum_i x_i^2 + n\mu^2 - 2n\overline x \mu+n\overline x^2-n \overline x^2)]
\\\\ &=\exp \{-\frac {\phi} 2 [\sum_i x_i^2-n\overline x^2+n(\mu-\overline x)^2]\}
\\\\ &=\exp \{-\frac {\phi} 2 [(n-1)s^2+n(\mu-\overline x)^2]\}
\end{aligned}$$

2. 
$$\begin{aligned}&p(\mu,\phi|\mathbf x)\propto p(\mathbf x|\mu,\phi) p(\mu,\phi)
\\\\ \propto & \phi^{n/2} \exp[-\frac {\phi} 2 \sum_i(x_i-\mu) ^ 2] \phi^{v_0/2-1} \exp \{-v _ 0 s _ 0 ^ 2 \phi/2\} (n_0\phi) ^ {1/2} \exp[-\frac {n_0\phi} 2 (\mu-\mu_0)^2]
\\\\=& \exp[-\frac {\phi} 2 \sum_i(x_i-\mu) ^ 2] \phi ^ {(v_0+n)/2-1} \exp \{-v_0s_0 ^ 2 \phi/2\} (n_0\phi) ^ {1/2} \exp[-\frac {n_0\phi} 2 (\mu-\mu_0) ^ 2]
\\\\= & \phi ^ {(v_0+n)/2-1} \exp \{-\frac {\phi} 2(v_0s_0 ^ 2+(n-1)s ^ 2)\}(n_0\phi) ^ {1/2} \exp[-\frac {n_0\phi} 2 (\mu-\mu_0) ^ 2-\frac {n\phi}2(\mu-\overline x) ^ 2]
\end{aligned}$$


$$-\frac {n_0\phi} 2 (\mu-\mu_0)^2-\frac {n\phi}2(\mu-\overline x)^2=-\frac {n_0 \phi}2(\mu^2-2\mu_0 \mu+\mu_0^2)-\frac {n \phi} 2 (\mu^2-2\overline x \mu + \overline x^2)
\\\\=-\frac {(n_0+n)\phi} 2( \mu -  \frac {n_0\mu_0+n\overline x}{n_0+n})^2  + \frac {(n_0\mu_0+n\overline x)^2}{2(n_0+n)}\phi-\frac {\phi} 2 (n_0\mu_0^2+n\overline x^2)
\\\\=-\frac {(n_0+n)\phi} 2( \mu -  \frac {n_0\mu_0+n\overline x}{n_0+n})^2  -\frac {n_0n\phi} {2(n_0+n)} (\overline x - \mu_0)^2$$

于是

$$p(\mu,\phi|\mathbf x)\propto p(\mathbf x|\mu,\phi) p(\mu,\phi)\propto 
\\\\ \phi^{(v_0+n)/2-1} \exp \{-\frac {\phi} 2(v_0 s _ 0 ^ 2+(n-1)s ^ 2+\frac {n_0n}{n_0+n}(\overline x-\mu_0) ^ 2)\}
\\\\ \cdot (n_0/(n_0+n)) ^ {1/2} [(n_0+n)\phi]^{1/2}
 \exp[-\frac {(n_0+n)\phi} 2( \mu -  \frac {n_0\mu_0+n\overline x}{n_0+n}) ^ 2]$$

后验分布可写成如下形式

$$p(\mu,\phi)=p(\phi) p(\mu|\phi)$$

由于 $\mu$ 以 $\phi$ 为条件，故首先看 $p(\phi)$，这是分布 $Gamma(v_1/2, v_1s_1^2/2)$，对比参数可知

$$v_1=v_0+n$$

$$s_1^2=[v_0s_0 ^ 2+(n-1)s ^ 2+\frac {n_0n}{n_0+n}(\overline x-\mu_0) ^ 2]/v_1$$

再看 $p(\mu|\phi)$ ，这是一个高斯分布 $\mathcal N(\mu_1, 1/n_1\phi)$，对比可知，
$$n_1=n_0+n, \quad \mu_1=\frac {n_0\mu_0+n\overline x}{n_1}$$

### 3.3.4 期望的边缘先验

相互独立的观测样本 $\mathbf x_{1:n}$ 来自于高斯分布 $X|\mu,V \sim \mathcal N(\mu, V)$，其期望和方差均未知，但是先验分布为

$$\mu, \phi \sim NormGamma(\mu_0, n_0, v_0, s_0^2)$$

其中 $\phi=1/V$ ，那么 $\mu$ 的边缘先验为 student's t 分布，

$$\frac {\mu-\mu_0} {\sqrt {s_0^2/n_0}} \sim T(v_0)$$

证明：对 $p(\mu, \phi)$ 积分 $\int_0^{\infty} f(\mu|\phi) f(\phi) d\phi$，略。

## 3.4 例子

对于城市的空气质量指数（AQI）我们的先验知识来自于历史数据：20 个样本并计算得到 AQI 估计值为 40，且方差估计为 100。现在得到 40 个 AQI 数据之后，发现这 40 个样本平均为 58，样本方差为 150。根据 #3.3.3 小节的内容求后验分布。

观测数据 $x$ 来自于高斯分布，期望和方差均未知，根据 #3.3.3 小节的知识，

$$\mu, \phi \sim NormGamma(\mu_1,n_1,v_1,s_1^2)$$



先验为 

$$\mu|\phi \sim \mathcal N(\mu_0, 1/(n_0 \phi), \quad \phi \sim Gamma(v_0/2, v_0s_0^2/2)$$

这里 $n_0=20, n=40$。期望的点估计为 $\mu_0=40$。观测到的样本方差 $s^2=150$。

根据历史数据推断的方差先验的自由度 $v_0=n_0-1=19$，因为我们方差先验估计为 $s_0^2=100$ 是已知且固定不变，故先验自由度为 $n_0-1$。

于是可计算

$$n_1=n_0+n=20+40=60
\\\\ \mu_1=\frac {n_0\mu_0+n \overline x}{n_0+n}=\frac {20\cdot 40+40\cdot 48}{60}=52
\\\\ v_1=v_0+n=19+40=59
\\\\ s_1 ^ 2=\frac 1 {v_1}[v_0s_0 ^ 2+(n-1)s ^ 2+\frac {n_0n}{n_0+n}(\overline x-\mu_0) ^ 2]=
\\\\ \frac 1 {59}[19\cdot 100+(40-1)\cdot 150+ \frac {20 \cdot 40}{60}(58-40) ^ 2]=204.6$$

# Ref

https://www.real-statistics.com/bayesian-statistics/bayesian-statistics-normal-data/conjugate-priors-normal-distribution/