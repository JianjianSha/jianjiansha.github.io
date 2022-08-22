---
title: Analytic-DPM
date: 2022-07-22 14:01:56
tags: diffusion model
mathjax: true
---

论文：[Analytic-DPM: An analytic estimate of the optimal reverse variance in diffusion probabilistic models](https://arxiv.org/abs/2201.06503)

# 1. 背景

本文的 $x_n$ 表示向量。

DDPM 的前向过程是马尔科夫链，

$$q_M(x_{1:N}|x_0)=\prod_{n=1}^N q_M(x_n|x_{n-1}), \quad q_M(x_n|x_{n-1})=\mathcal N(x_n|\sqrt {\alpha_n} x_{n-1}, \beta_nI) \tag{1}$$

其中 $\beta_n \in (0, 1)$ 是 step $n$ 中噪声方差，且 $\alpha_n=1-\beta_n$。[DDIM](/2022/07/21/diffusion_model/ddim) 中介绍了一种更为一般化的非马尔可夫过程，$\lambda_{1:N}\in \mathbb R_{\ge 0}^N$ 表示反向过程的噪声方差，这里为了方便简要列出非马尔可夫过程，不用再翻看 DDIM 论文，

$$q_{\lambda}(x_{1:N}|x_0)=q_{\lambda} (x_N|x_0) \prod_{n=2}^N q_{\lambda}(x_{n-1}|x_n, x_0)
\\ q_{\lambda}(x_N|x_0)=\mathcal N(x_N|\sqrt {\overline \alpha_N} x_0, \overline \beta_N I)
\\ q_{\lambda}(x_{n-1}|x_n,x_0)=\mathcal N(x_{n-1}|\tilde \mu_n(x_n, x_0), \lambda_n^2 I)
\\ \tilde \mu_n(x_n,x_0)=\sqrt {\overline \alpha_{n-1}} x_0 + \sqrt {\overline \beta_{n-1}-\lambda_n^2} \cdot \frac {x_n - \sqrt {\overline \alpha_n}x_0}{\sqrt {\overline \beta_n}} \tag{2}$$

其中 $\overline \alpha_n = \prod_{i=1}^n \alpha_i, \ \overline \beta_n = 1-\overline \alpha_n$。

DDIM 中非马尔可夫过程：$\mathbf x_{n-1}$ 依赖于 $\mathbf x_0$ 和 $\mathbf x_n$，与 DDPM 中 $\mathbf x_n$ 依赖 $\mathbf x_{n-1}$ 不同，故给定训练样本数据 $\mathbf x_0$ 后，还需要事先确定 $\mathbf x_N|\mathbf x_0$ 的分布，如 (2) 中第二个等式所示。

[DDIM](/2022/07/21/diffusion_model/ddim) 中已经讨论过，当 

$$\lambda_n^2 = \frac {\overline \beta_{n-1}}{\overline \beta_n} \beta_n$$

时，非马尔可夫过程退化为 DDPM。

另一种特殊情况是 $\lambda_n=0$ 时，为确定性 DDIM。

(2) 式可推导出 $q_{\lambda}(x_n|x_0)=\mathcal N(x_n|\sqrt {\overline \alpha_n} x_0, \overline \beta_n I)$，与 $\lambda$ 无关，这表明模型对于不同的 $\lambda$ 不需要重新训练。下文为了表述简单，忽略下标 $\lambda$。

(2) 式的反向过程是一个马尔可夫过程，反向过程为

$$p(x_{0:N})=p(x_N) \prod_{n=1}^N p(x_{n-1}|x_n), \quad p(x_{n-1}|x_n)=\mathcal N(x_{n-1}|\mu_n(x_n), \sigma_n^2I) \tag{3}$$

根据 [DDPM](/2022/06/27/diffusion_model/ddpm) 分析，令 $p(x_{n-1}|x_n)$ 逼近 $q(x_{n-1}|x_n,x_0)$，后者

$$q( x_{n-1}| x_n,  x_0)=\mathcal N( x_{n-1}; \tilde {\mu}_n( x_n,  x_0), \tilde {\beta}_n I) \tag {4}$$

其中期望经过贝叶斯定理计算为

$$\tilde {\mu}_t(\mathbf x_t, \mathbf x_0)=\frac {\sqrt {\overline \alpha_{t-1}}\beta_t}{1-\overline \alpha_t}\mathbf x_0+\frac {\sqrt {\alpha_t}(1-\overline \alpha_{t-1})}{1-\overline \alpha_t} \mathbf x_t \tag{5}$$


根据 [Score-based SDE]() 中的 (9) 式可知，

$$\mathbf s_{\theta}( x_n,n)=\nabla_{ x} \log q_{\lambda}( x| x_0)|_{ x= x_n}=\frac {\sqrt {\overline \alpha_n}  x_0 - x_n}{\overline \beta_n} \tag{6}$$

变换得

$$x_0 = \frac 1 {\overline \alpha_n} (\overline \beta_n \mathbf s_{\theta}(x_n, n) + x_n) \tag{7}$$

代入 (5) 式，得 $p(x_{n-1}|x_n)$ 的期望为

$$\mu_n(x_n)=\tilde \mu_n \left(x_n, \frac 1 {\sqrt {\overline \alpha_n}}(x_n+ {\overline \beta_n} \mathbf s_{\theta}(x_n, n))\right) \tag{8}$$

训练目标为最小化负对数似然

$$\min \mathbb E_{q(x_0)}[-\log p_{\theta}(x_0)] \tag{9}$$

经过推导可知，等价于最小化变分下限 $L_{vb}$，如下（推导过程见 [DDPM](/2022/06/27/diffusion_model/ddpm)），

$$L_{vb}=\mathbb E_q \left[-\log p(x_0|x_1) + \sum_{n=2}^N D_{KL}(q(x_{n-1}|x_0,x_n)||p(x_{n-1}|x_n))+D_{KL}(q(x_N|x_0)||p(x_N)) \right] \tag{10}$$




# 2. 反向过程方差

(9) 式表示的训练目标其实等价于 

$$\min_{(\mu_n, \sigma_n^2)_{n=1}^N}  \ L_{vb} \Leftrightarrow  \min_{(\mu_n, \sigma_n^2)_{n=1}^N}  \ D_{KL} (q(x_{0:N})||p(x_{0:N})) \tag{11}$$

根据 [DDPM](/2022/06/27/diffusion_model/ddpm)

$$\begin{aligned}L_{vb}&=\mathbb E_q[-\log \frac {p(\mathbf x_{0:N})}{q(\mathbf x_{1:N}|\mathbf x_0)}]
\\&=\mathbb E_q[-\log \frac {p(\mathbf x_{0:N}) q(\mathbf x_0)}{q(\mathbf x_{0:N})}]\\&=D_{KL}(q_{0:N}||p(\mathbf x_{0:N}))+H(q(\mathbf x_0))
\end{aligned}$$

由于 $H(q(\mathbf x_0))$ 与 $\mu_n, \ \sigma_n$ 无关，所以 (11) 式成立。

(11) 式的最优解为

$$\mu_n^{\star}(x_n) = \tilde {\mu}_n \left(x_n,\frac 1 {\sqrt {\overline \alpha_n}}(x_n+ {\overline \beta_n} \nabla_{x_n} \log q_n(x_n))\right) \tag{12}$$

$$\sigma_n^{\star 2}=\lambda_n^2 + \left( \sqrt {\frac {\overline \beta_n}{\alpha_n}} - \sqrt {\overline \beta_{n-1}-\lambda_n^2} \right)^2 \left(1-\overline \beta_n \mathbb E_{q_n(x_n)} \frac {\|\nabla_{x_n} \log q_n(x_n)\|^2}{d} \right) \tag{13}$$

证明过程非常繁冗，见原论文，这里不贴出来了。

