---
title: Analytic-DPM
date: 2022-07-22 14:01:56
tags: diffusion model
mathjax: true
---

论文：[Analytic-DPM: An analytic estimate of the optimal reverse variance in diffusion probabilistic models](https://arxiv.org/abs/2201.06503)

源码：[baofff/Analytic-DPM](https://github.com/baofff/Analytic-DPM)

# 1. 简介

这篇论文作者得到 DPM 的最佳反向过程方差以及相应的最佳 KL 散度的解析解形式，基于此，提出 Analytic-DPM，提高一个预训练 DPM 的采样效率，而不用进行额外训练。

优点：提高 log-likelihood，生成高质量样本，采样速度提升 20x ~ 80x 。



# 2. 背景

本文的 $x_n$ 表示向量。

DDPM 的前向过程是马尔科夫链，

$$q_M(x_{1:N}|x_0)=\prod_{n=1}^N q_M(x_n|x_{n-1}), \quad q_M(x_n|x_{n-1})=\mathcal N(x_n|\sqrt {\alpha_n} x_{n-1}, \beta_n I) \tag{1}$$

其中 $\beta_n \in (0, 1)$ 是 step $n$ 中噪声方差，且 $\alpha_n=1-\beta_n$。[DDIM](/2022/07/21/diffusion_model/ddim) 中介绍了一种更为一般化的非马尔可夫过程，$\lambda_{1:N}\in \mathbb R_{\ge 0}^N$ 表示反向过程的噪声方差，这里为了方便简要列出非马尔可夫过程，不用再翻看 DDIM 论文，

$$q_{\lambda}(x_{1:N}|x_0)=q_{\lambda} (x_N|x_0) \prod_{n=2}^N q_{\lambda}(x_{n-1}|x_n, x_0)
\\\\ q_{\lambda}(x_N|x_0)=\mathcal N(x_N|\sqrt {\overline \alpha_N} x_0, \overline \beta_N I)
\\\\ q_{\lambda}(x_{n-1}|x_n,x_0)=\mathcal N(x_{n-1}|\tilde \mu_n(x_n, x_0), \lambda_n^2 I)
\\\\ \tilde \mu_n(x_n,x_0)=\sqrt {\overline \alpha_{n-1}} x_0 + \sqrt {\overline \beta_{n-1}-\lambda_n^2} \cdot \frac {x_n - \sqrt {\overline \alpha_n}x_0}{\sqrt {\overline \beta_n}} \tag{2}$$

其中 $\overline \alpha_n = \prod_{i=1}^n \alpha_i, \ \overline \beta_n = 1-\overline \alpha_n$。

DDIM 中非马尔可夫过程：$\mathbf x_{n-1}$ 依赖于 $\mathbf x_0$ 和 $\mathbf x_n$，与 DDPM 中 $\mathbf x_n$ 依赖 $\mathbf x_{n-1}$ 不同，故给定训练样本数据 $\mathbf x_0$ 后，还需要事先确定 $\mathbf x_N|\mathbf x_0$ 的分布，如 (2) 中第二个等式所示。

[DDIM](/2022/07/21/diffusion_model/ddim) 中已经讨论过，当 

$$\lambda_n^2 = \frac {\overline \beta_{n-1}}{\overline \beta_n} \beta_n \tag{2.1}$$

时，非马尔可夫过程退化为 DDPM。

另一种特殊情况是 $\lambda_n=0$ 时，为确定性 DDIM。

(2) 式可推导出 $q_{\lambda}(x_n|x_0)=\mathcal N(x_n|\sqrt {\overline \alpha_n} x_0, \overline \beta_n I)$，与 $\lambda$ 无关，这表明模型对于不同的 $\lambda$ 不需要重新训练，因为训练扩散模型，只是为了能得到 $\epsilon _ {\theta}$，这是模型对 $\epsilon$ 的预测，而根据前向过程 $x _ n=\sqrt {\overline \alpha _ n} x _ 0 + \sqrt {1-\overline \alpha _ n} \epsilon$，所以训练扩散模型时，训练输入为 $x _ n$，target 为 $\epsilon$，输出为 $\epsilon _ {\theta}$，均与 $\lambda$ 无关。

下文为了表述简单，忽略下标 $\lambda$。

(2) 式的反向过程定义为一个马尔可夫过程，反向过程为

$$p(x_{0:N})=p(x_N) \prod_{n=1}^N p(x_{n-1}|x_n), \quad p(x_{n-1}|x_n)=\mathcal N(x_{n-1}|\mu_n(x_n), \sigma_n^2I) \tag{3}$$

根据 [DDPM](/2022/06/27/diffusion_model/ddpm) 分析，令 $p(x_{n-1}|x_n)$ 逼近 $q(x_{n-1}|x_n,x_0)$，

$$q( x_{n-1}| x_n,  x_0)=\mathcal N( x_{n-1}; \tilde {\mu}_n( x_n,  x_0), \tilde {\beta}_n I) \tag {4}$$

其中期望经过贝叶斯定理计算为

$$\tilde {\mu} _ t(\mathbf x_t, \mathbf x_0)=\frac {\sqrt {\overline \alpha_{t-1}}\beta_t}{1-\overline \alpha_t}\mathbf x_0+\frac {\sqrt {\alpha_t}(1-\overline \alpha_{t-1})}{1-\overline \alpha_t} \mathbf x_t \tag{5}$$

事实上，当满足 (2.1) 式时，(2) 式具例化为 (5) 式。


根据 [Score-based SDE](diffusion_model/2022/07/26/score_based_SDE) 中的 (9) 式可知（注意 $q_{\lambda}(x_n|x_0)=\mathcal N(x_n|\sqrt {\overline \alpha_n} x_0, \overline \beta_n I)$ 就是前向过程），

$$\mathbf s_{\theta}( x_n,n)=\nabla_{ x} \log q_{\lambda}( x| x_0)|_{ x= x_n}=\frac {\sqrt {\overline \alpha_n}  x_0 - x_n}{\overline \beta_n} \tag{6}$$

变换得

$$x_0 = \frac 1 {\overline \alpha_n} (\overline \beta_n \mathbf s_{\theta}(x_n, n) + x_n) \tag{7}$$

代入 (5) 式或者 (2) 式，得 $p(x_{n-1}|x_n)$ 的期望为

$$\mu_n(x_n)=\tilde \mu_n \left(x_n, \frac 1 {\sqrt {\overline \alpha_n}}(x_n+ {\overline \beta_n} \mathbf s_{\theta}(x_n, n))\right) \tag{8}$$

训练目标为最小化负对数似然

$$\min \mathbb E_{q(x_0)}[-\log p_{\theta}(x_0)] \tag{9}$$

经过推导可知，等价于最小化变分下限 $L_{vb}$，如下（推导过程见 [DDPM](/2022/06/27/diffusion_model/ddpm)），

$$L_{vb}=\mathbb E_q \left[-\log p(x_0|x_1) + \sum_{n=2}^N D_{KL}(q(x_{n-1}|x_0,x_n)||p(x_{n-1}|x_n))+D_{KL}(q(x_N|x_0)||p(x_N)) \right] \tag{10}$$




# 3. 反向过程方差

(9) 式表示的训练目标其实等价于 

$$\min_{(\mu_n, \sigma _ n ^ 2) _ {n=1} ^ N}  \ L _ {vb} \Leftrightarrow  \min_{(\mu_n, \sigma_n ^ 2) _{n=1}^N}  \ D _ {KL} (q(x _ {0:N})||p(x _ {0:N})) \tag{11}$$

根据 [DDPM](/2022/06/27/diffusion_model/ddpm) 中的 (9) 式，

$$\begin{aligned}L_{vb}&=\mathbb E_q[-\log \frac {p(\mathbf x_{0:N})}{q(\mathbf x_{1:N}|\mathbf x_0)}]
\\&=\mathbb E_q[-\log \frac {p(\mathbf x_{0:N}) q(\mathbf x_0)}{q(\mathbf x_{0:N})}]\\&=D_{KL}(q_{0:N}||p(\mathbf x_{0:N}))+H(q(\mathbf x_0))
\end{aligned}$$

由于 $H(q(\mathbf x_0))$ 与 $\mu_n, \ \sigma_n$ 无关，所以 (11) 式成立。

实际应用中，为了提高生成样本质量，不直接优化 $L _ {vlb}$，而是采用下式得分函数的 MSE 进行优化，

$$\min _ {s _ n} \mathbb E _ n \overline \beta _ n  \mathbb E _ {q _ n (x _ n)} ||s _ n (x _ n) - \nabla _ {x _ n} \log q _ n (x _ n)|| ^ 2= \mathbb E _ {n, x _ 0, \epsilon} || \epsilon + \sqrt {\overline \beta _ n} s _ n(x _ n) || ^ 2 + c \tag{12}$$

上式推导中，因为 $q _ n (x _ n) = \mathcal N(\sqrt {\overline \alpha _ n} x _ 0, \overline \beta _ n I)$，所以 $\nabla _ {x _ n} \log q _ n (x _ n) = -\frac {x _ n - \sqrt {\overline \alpha _ n} x _ 0}{\overline \beta _ n}$，而 $x _ n$ 取值为 $x _ n = \sqrt {\overline \alpha _ n} x _ 0 + \sqrt {\overline \beta _ n} \epsilon$ 。由于消去了 $x _ n$，所以求期望时增加 $x _ n$ 和 $\epsilon$ 两个随机变量。

注意 (12) 式没有学习反向转移分布 (3) 式的方差 $\sigma _ n ^2$，在之前的论文中，$\sigma ^ 2$ 都是手动选取值，例如取前向转移的方差 $\beta _ n$，或者后验分布 $q(x _ {n-1}|x _ n, x _ 0)$ 的方差 $\tilde \beta _ n$，这分别对应方差的上下限，或者是通过模型学习一个向量 $v$，然后对上下限进行插值 $\exp (v \log \beta _ t + (1-v) \log \tilde \beta _ t)$ 。DDIM 中则取 $\sigma ^ 2 = \lambda _ n ^ 2$ 。

作者认为这些手动选择的值不是 (11) 式的最佳解，从而导致性能不是最优。

(11) 式的最优解即，最优期望 $\mu _ n ^ {\star} (x _ n)$ 和最优方差 $\sigma ^ {\star}$ 有解析形式，是关于得分函数的解析形式，如下所示，

$$\mu_n ^ {\star}(x_n) = \tilde {\mu} _ n \left(x _ n,\frac 1 {\sqrt {\overline \alpha_n}}(x_n+ {\overline \beta_n} \nabla _ {x_n} \log q _ n(x _ n))\right) \tag{13}$$

$$\sigma_n^{\star 2}=\lambda_n^2 + \left( \sqrt {\frac {\overline \beta_n}{\alpha_n}} - \sqrt {\overline \beta_{n-1}-\lambda_n^2} \right)^2 \left(1-\overline \beta_n \mathbb E_{q_n(x_n)} \frac {\|\nabla_{x_n} \log q_n(x_n)\|^2}{d} \right) \tag{14}$$

其中 $q _ n(x _ n)$ 是前向过程的边缘分布，$d$ 是数据维度，例如图像维度为 $3 \times H \times W$ 。



# A. 推导证明

## A.1 Lemmas

**# Lemma 1.** 与高斯分布的交叉熵

假设 $q(x)$ 的期望和方差分别为 $\mu _ q, \ \Sigma _ q$，且 $p(x) = \mathcal N(x|\mu, \Sigma)$ 是一高斯分布，那么 $q$ 与 $p$ 的交叉熵等于 $\mathcal N(x|\mu _ q, \Sigma _ q)$ 与 $p$ 的交叉熵，即

$$\begin{aligned}H(p,q)&=H(\mathcal N(x|\mu _ q, \Sigma _ q), p)
\\\\ &= \frac 1 2 \log ((2\pi) ^ d|\Sigma| + \frac 1 2 tr(\Sigma _ q, \Sigma ^ {-1}) + \frac 1 2 (\mu _ q - \mu) ^ {\top} \Sigma ^ {-1} (\mu _ q - \mu))
\end{aligned}$$

注：$q$ 是任意类型分布，不一定是高斯分布。

证明：

根据交叉熵定义，

$$\begin{aligned}H(q, p) &= - E _ {q(x)} \log p(x) = - E _ {q(x)}\left[ \log \frac 1 {\sqrt {(2\pi) ^ d |\Sigma|}} - \frac {(x-\mu) ^ {\top} \Sigma ^ {-1} (x-\mu)} 2 \right]
\\\\ &=\frac 1 2 \log[(2\pi) ^ d |\Sigma|] + \frac 1 2 E _ {q(x)} (x-\mu) ^ {\top} \Sigma ^ {-1} (x-\mu)
\\\\ & \stackrel{4}=\frac 1 2 \log[(2\pi) ^ d |\Sigma|] + \frac 1 2 E _ {q(x)} tr( (x-\mu)(x-\mu) ^ {\top} \Sigma ^ {-1})
\\\\ &=\frac 1 2 \log[(2\pi) ^ d |\Sigma|] + \frac 1 2 tr( E _ {q(x)} [(x-\mu)(x-\mu) ^ {\top} ]\Sigma ^ {-1})
\\\\ & \stackrel{6}=\frac 1 2 \log[(2\pi) ^ d |\Sigma|] + \frac 1 2 tr( E _ {q(x)} [(x-\mu _ q)(x-\mu _ q) ^ {\top} + (\mu _ q - \mu)(\mu _ q - \mu) ^ {\top}]\Sigma ^ {-1})
\\\\ &=\frac 1 2 \log[(2\pi) ^ d |\Sigma|] + \frac 1 2 tr( [\Sigma _ q + (\mu _ q - \mu)(\mu _ q - \mu) ^ {\top}]\Sigma ^ {-1})
\\\\ &=\frac 1 2 \log[(2\pi) ^ d |\Sigma|] + \frac 1 2 tr(\Sigma _ q \Sigma) + \frac 1 2 (\mu _ q - \mu) ^ {\top}\Sigma ^ {-1} (\mu _ q - \mu) \qquad \qquad \qquad \square
\end{aligned}$$

以上推导过程中，第四个等号关系推导用到了以下结论（第八个等号关系也用到这个结论）：

$$u ^ {\top} A v=\sum_j (u ^ {\top} A _ {:,j}) \cdot v_j=\sum _ j (\sum _ i u_i a_{ij}) v_j=\sum _ j \sum _ i (v _ j u _ i) a _ {ij}=\sum _ j (v u ^ {\top})_{j,:} A _ {:,j} = tr (v u^{\top} A)$$

其中 $u ^ {\top}, \ v$ 分别表示行列向量， $A$ 表示矩阵，$A _ {i,:}, \ A _ {:,j}$ 分别表示行列向量。

第六个等号推导用到 $x-\mu = (x - \mu _ q) + (\mu _ q - \mu)$ 。

第七个等号推导用到 二阶中心矩为协方差矩阵 这个结论。

根据这个推导，$q$ 的分布无论是何类型，只要期望和协方差矩阵为 $\mu _ q$  和 $\Sigma _ q$，均满足上式，那么自然对于高斯分布类型的 $q$，也是满足上式。

**# Lemma 2.** 与高斯分布的 KL 散度

假设概率密度 $q(x)$ 的期望和协方差矩阵分别为 $\mu _ q$ 和 $\Sigma _ q$，且 $p(x)=\mathcal N(x|\mu, \Sigma)$ 是高斯分布，那么

$$D _ {KL}(q||p) = D _ {KL} (\mathcal N(x|\mu _ q, \Sigma _ q)||p) + H(\mathcal N(x|\mu _ q, \Sigma _ q)) - H(q)$$

其中 $H(\cdot)$ 表示熵，$D_{KL}(q||p)$ 表示 KL 散度（相对熵）。

证明：

根据定义以及 lemma 1，

$$\begin{aligned}D_{KL}(q||p)&=H(q,p)-H(q)=H(\mathcal N(x|\mu _ q, \Sigma _ q), p) - H(q)
\\\\ &= H(\mathcal N(x|\mu _ q, \Sigma _ q), p) - H(\mathcal N(x|\mu _ q, \Sigma _ q)) + H(\mathcal N(x|\mu _ q, \Sigma _ q)) - H(q)
\\\\ &= D_{KL}(\mathcal N(x|\mu _ q, \Sigma _ q) || p) + H(\mathcal N(x|\mu _ q, \Sigma _ q)) - H(q) \qquad \qquad \square
\end{aligned}$$

**# Lemma 3.** 马尔可夫前向与后向之间的等价

假设 $q(x _ {0:N}) = q(x _ 0) \prod _ {n=1} ^ N q(x _ n | x _ {n-1})$ 是马尔可夫链，那么 q 的反向也是马尔可夫链，即 $q(x_{0:N})=q(x _ N) \prod _ {n=1} ^ N q(x _ {n-1} | x _ n)$ 。

证明：

$$\begin{aligned} q(x _ {n-1} | x_n, \cdots, x _ N) &= \frac {q(x _ {n-1}, x _ n, \cdots, x _ N)}{q(x _ n, \cdots, x _ N)}
\\\\ &=\frac {q(x _ {n-1}) \prod _ {i=n} q(x _ i | x _ {i-1})}{q(x _ n) \prod _ {i=n+1} q(x _ i | x _ {i-1})}
\\\\ &=\frac {q(x _ {n-1}) q(x _ n | x _ {n-1})}{q (x _ n)}
\\\\ &=q(x _ {n-1} | x _ n)
\end{aligned}$$

上式第二个等号推导用到马尔可夫转移的性质：$x _ i$ 的分布只与前一时刻即 $x _ {i-1}$ 有关，即 $q(x _ i|x _ {0:i-1})=q(x _ i | x _ {i-1})$，那么根据上式推导结论可知，

$$\begin{aligned} q(x _ {0:N}) &= q(N) \prod _ {n=1} ^ N q(x _ {N-n}|x _ {N-n+1}, \cdots, x _ N)
\\\\ &=q(N) \prod _ {n=1} ^ N q(x _ {N-n}|x _ {N-n+1})
\\\\ &= q(x _ N) \prod _ {n=1} ^ N q(x _ {n-1} | x _ n) \qquad \qquad \qquad \square
\end{aligned}$$

**# Lemma 4.** 马尔可夫链的熵

假设 $q(x _ {0:N})$ 是一马尔可夫链，那么

$$\begin{aligned} H(q(x _ {0:N})) &= H(q(x _ N)) + \sum _ {n=1} ^ N E _ q H(q(x _ {n-1}|x _ n)) 
\\\\ &= H(q(x _ 0)) + \sum _ {n=1} ^ N E _ q H(q (x _ n | x _ {n-1}))
\end{aligned}$$

根据熵的定义以及 lemma 3 可证。注意这里数学表达式的具体含义，

$$\begin{aligned}E _ q H(q(x _ n|x _ {n-1})) &=-E _ {q(x _ {n-1}, x _ n)} \log q(x _ n|x _ {n-1})
\\\\ &= -\int q(x _ {n-1}) q(x _ n|x _ {n-1}) \log q(x _ n|x _ {n-1}) d x _ n d x _ {n-1}
\\\\ &= \int q(x _ {n-1}) H(q(x _ n|x _ {n-1})) d x _ {n-1}
\\\\ &= E _ {q(x _ {n-1})} H(q(x _ n|x _ {n-1}))
\end{aligned}$$

**# Lemma 5.** DDPM 前向过程的熵

假设 $q(x _ {0:N})$ 是一马尔可夫链，且前向转移为 $q(x _ n|x _ {n-1})=\mathcal N(x _ n|\sqrt {\alpha _ n} x _ {n-1}, \beta _ nI)$，那么

$$H(q(x _ {0:N}))=H(q(x _ 0))+\frac d 2 \sum _ {n=1} ^ N \log (2 \pi e \beta _ n)$$

其中 $d$ 是数据 $x _ n$ 的维度。

证明：

$$\begin{aligned}H(q(x _ n|x _ {n-1})) &=- E _ {q(x _ n|x _ {n-1})} \log q(x _ n|x _ {n-1})
\\\\ &=\frac d 2 [1+\log (2\pi)] + \frac d 2 \log \beta _ n
\\\\ &= \frac d 2 \log (2\pi e \beta _ n)
\end{aligned}$$

注意 $q(x _ n | x _ {n-1})$ 的协方差矩阵是 $\beta _ nI$，所以 $|\beta _ n I|=d \cdot \beta _ n$。上述多维高斯分布熵的证明过程见 [高斯分布](/2022/06/29/math/gaussian) 一文的公式 (2) 和 (3)。

于是 $E _ {q(x _ {n-1},x_n)} H(q(x _ n|x _ {n-1})) = E _ {q(x _ {n-1},x_n)} [\frac d 2 \log (2\pi e \beta _ n)]= \frac d 2 \log (2\pi e \beta _ n)$，那么再根据 lemma 4 可知，

$$H(q(x _ {0:N}))=H(q(x _ 0)) + \frac d 2 \sum _ {n=1}^N \log (2 \pi e \beta _ n) \qquad \qquad \square$$


**# Lemma 6.** 条件马尔可夫链的熵

假设 $q(x _ {1:N}|x _ 0)$ 是一条件型马尔可夫链，即 $q(x _ {1:N}|x _ 0)=q(x _ 1|x _ 0)\prod _ {n=2} ^ N q(x _ n|x _ {n-1}, x _ 0)$，那么

$$H(q(x _ {0:N})) = H(q(x _ 0)) + E _ q H(q(x _ N|x _ 0)) + \sum _ {n=2} ^ N E _ q H(q(x _ {n-1}|x _ n, x _ 0))$$

证明：

已知 $q(x _ {0:N})= q(x _ 0) q(x _ {1:N}|x _ 0)$ （将其看作一个具有两个状态节点的马尔可夫链），那么根据 lemma 4 可知，

$$\begin{aligned}H(q(x _ {0:N})) &= H(q(x _ 0)) + E _ {q(x _ 0)} H(q (x _ {1:N} | x _ 0))
\\\\ &= H(q(x _ 0)) + E _ {q(x _ 0)} H(q (x _ N |x _ 0)) + \sum _ {n=2} ^ N E _ {q(x _ 0, x _ n)} H(q(x _ {n-1}|x _ n, x _ 0))  \qquad \square
\end{aligned}$$

上式第二个等号也是利用了 lemma 4，因为 $q(x _ {1:N}|x _ 0)$ 也是一马尔可夫链。上式推导中，将期望 $E _ q$ 的下标详细标注是关于哪些随机变量。

其实要搞懂关于哪些随机变量的分布求期望很简单，例如求 $E _ q H(q(x _ {n-1}|x _ n, x _ 0))$ ，其中 $H(q(x _ {n-1}|x _ n, x _ 0))$ 是 $x _ 0, x _ n$ 的函数（因为可以消去随机变量 $x _ {n-1}$），所以 $E _ q H(q(x _ {n-1}|x _ n, x _ 0))$ 是关于随机变量 $x _ 0, x _ n$ 的分布求期望。

注意，根据 lemma 3，马尔可夫链的等价关系，有

$$q(x _ {1:N}|x _ 0)=q(x _ 1|x _ 0)\prod _ {n=2} ^ N q(x _ n|x _ {n-1}, x _ 0)=q(x _ N|x _ 0)\prod _ {n=2} ^ N q(x _ {n-1}|x _ n, x _ 0)$$

**# Lemma 7.** 泛化 DDPM 前向过程的熵

假设 $q(x _ {1:N}|x _ 0)$ 是马尔可夫链，$q(x _ N|x _ 0)$ 是高斯分布，其协方差为 $\overline \beta _ N I$，且 $q(x _ {n-1}|x _ n,x _ 0)$ 是高斯分布，其协方差为 $\lambda _ n ^ 2 I$ ，那么

$$H(q(x _ {0:N})) = H(q(x _ 0)) + \frac d 2 \log (2 \pi e \overline \beta _ N) + \frac d 2 \sum _ {n=2} ^ N \log (2 \pi e \lambda _ n ^ 2)$$

其中 $\overline \beta _ n = 1 - \overline \alpha _ n$，所以 $q(x _ N|x _ 0)$ 的分布与前文 DDPM 的情况一致。这里仅对 $t= N$ 时刻的分布和反向转移过程的分布固定为高斯型，且给定协方差矩阵，期望则未指定，所以这是 DDPM 的一种泛化。

证明：

再次根据 [高斯分布](/2022/06/29/math/gaussian) 一文的公式 (2) 和 (3)，可以得到

$$E _ {q(x _ 0)} H(q(x _ N|x _ 0)) = E _ {q(x _ 0)} [\frac d 2 \log (2\pi e \overline \beta _ N)]= \frac d 2 \log (2\pi e \overline \beta _ N)$$

$$E _ {q(x _ 0, x _ n)} H(q(x _ {n-1}|x _ n, x _ 0))=E _ {q(x _ 0, x _ n)} [\frac d 2 \log (2 \pi \lambda _ n ^ 2)]= \frac d 2 \log (2 \pi \lambda _ n ^ 2)$$

利用 lemma 6 可以证明。

**# Lemma 8.** 对马尔可夫链的 KL 散度

假设 $q(x _ {0:N})$ 是一个概率分布，且 $p(x _ {0:N})=p(x _ N) \prod _ {n=1}^N p(x _ {n-1}|x _ n)$ 是一马尔可夫链，那么

$$E _ q D _ {KL} (q(x _ {0:N-1}|x _ N)||p(x _ {0:N-1}|x _ N))=\sum _ {n=1} ^ N E _ q D _ {KL} (q(x _ {n-1}|x _ n)|| p(x _ {n-1}|x _ n))+c$$