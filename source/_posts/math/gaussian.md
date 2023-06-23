---
title: 高斯分布
date: 2022-06-29 15:30:53
tags: math
mathjax: true
---

# 1. 熵

## 1.1 单变量熵

$$\begin{aligned}H(x)&=-\int p(x) \log p(x) dx
\\\\&=-\mathbb E \left[\log [(2\pi \sigma ^ 2) ^ {-1/2} \exp(-\frac 1 {2\sigma^2}(x-\mu) ^ 2)]\right]
\\\\&=\frac 1 2 \log (2\pi \sigma ^ 2) + \frac 1 {2\sigma ^ 2} \mathbb E[(x-\mu) ^ 2]
\\\\&=\frac 1 2 \log (2\pi \sigma ^ 2) + \frac 1 2
\end{aligned} \tag{1}$$

(1) 式推导的最后一步用到了 **二阶中心矩是方差** 这一事实。

## 1.2 多变量熵

分布 $\mathbf x \sim \mathcal N_D(\mu, \Sigma)$

$$\begin{aligned}H(\mathbf x)&=-\int p(\mathbf x) \log p(\mathbf x) d \mathbf x
\\\\&=-\mathbb E \left[\log [(2\pi) ^ {-D/2} |\Sigma|^{-1/2} \exp(-\frac 1 2 (\mathbf x-\mu) ^ {\top}\Sigma^{-1}(\mathbf x-\mu))]\right]
\\\\&=\frac D 2 \log (2\pi) + \frac 1 2 \log |\Sigma|+\frac 1 2 \mathbb E[(\mathbf x-\mu) ^ {\top}\Sigma ^ {-1}(\mathbf x-\mu)]
\\\\&= \frac D 2(1+\log (2\pi)) + \frac 1 2 \log |\Sigma|
\end{aligned} \tag{2}$$

(2) 式推导的最后一步是因为

$$\begin{aligned}\mathbb E[(\mathbf x-\mu) ^ {\top}\Sigma^{-1}(\mathbf x-\mu)]&=\mathbb E[tr((\mathbf x-\mu) ^ {\top}\Sigma^{-1}(\mathbf x-\mu))]
\\\\&=\mathbb E[tr(\Sigma^{-1}(\mathbf x-\mu)(\mathbf x-\mu) ^ {\top})]
\\\\&=tr(\Sigma ^ {-1} \mathbb E[(\mathbf x - \mu)(\mathbf x - \mu) ^ {\top}])
\\\\&=tr(\Sigma ^ {-1} \Sigma)
\\\\&=tr(I_D)
\\\\&=D
\end{aligned} \tag{3}$$

(3) 式推导

1. 第一步：$(\mathbf x-\mu)^{\top}\Sigma^{-1}(\mathbf x-\mu)$ 是标量
2. 第二步：$tr$ 操作的矩阵顺序轮转不变性
3. 第三步：$tr$ 和 $\mathbb E$ 同为求和操作，求和顺序可变。另外 $\Sigma$ 为常量，可提取到 $\mathbb E$ 外面
4. 第四步：**二阶中心矩是协方差矩阵**

**条件熵**
$$\begin{aligned}H(X_2|X_1) &=-\mathbb E_{p(x_1)} \mathbb E_{p(x_2|x_1)} [\log p(x_2|x_1)]
\\ &=\mathbb E_{p(x_1)} [\frac D 2 \log(2\pi e)+ \frac 1 2 \log |\Sigma|]
\\ &=\frac D 2 \log(2\pi e)+ \frac 1 2 \log |\Sigma|
\end{aligned}$$

**交叉熵**

$q(\mathbf x)=\mathcal N(\mathbf x|\mu_q, \Sigma_q)$

$p(\mathbf x)=\mathcal N(\mathbf x|\mu, \Sigma)$

$$H(q,p)=-\mathbb E_q[\log p]=\frac 1 2 \log ((2\pi)^d |\Sigma|) + \frac 1 2 tr(\Sigma_q \Sigma^{-1}) + \frac 1 2 (\mu_q-\mu)^{\top} \Sigma^{-1}(\mu_q-\mu)$$

**KL散度**

$$D_{KL}(q||p)=\mathbb E_q [\log \frac q p]=H(q,p)-H(q)$$



# 2. 条件和边缘分布

$$p(\mathbf x)=\mathcal N(\mathbf x|\boldsymbol {\mu}, \Lambda^{-1})
\\ p(\mathbf y|\mathbf x)=\mathcal N(\mathbf y|A\mathbf x+\mathbf b, L^{-1})$$

那么有

$$p(\mathbf y)=\mathcal N(\mathbf y|A\boldsymbol \mu+\mathbf b, L^{-1}+A\Lambda^{-1}A^{\top})
\\ p(\mathbf x|\mathbf y)=\mathcal N(\mathbf x|\Sigma[A^{\top}L(\mathbf y-\mathbf b)+\Lambda \boldsymbol \mu], \Sigma)$$


其中 $\Sigma^{-1}=\Lambda + A^{\top} LA$。



