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

上式推导中，因为 $q _ n (x _ n) = \mathcal N(\sqrt {\overline \alpha _ n} x _ 0, \overline \beta _ n I)$，所以 $\nabla _ {x _ n} \log q _ n (x _ n) = -\frac {x _ n - \sqrt {\overline \alpha _ n} x _ 0}{\overline \beta _ n}$，而 $x _ n$ 取值为 $x _ n = \sqrt {\overline \alpha _ n} x _ 0 + \sqrt {\overline \beta _ n} \epsilon$ 。由于消去了 $x _ n$，所以 (12) 式等号右端求期望时增加 $x _ n$ 和 $\epsilon$ 两个随机变量。

注意 (12) 式没有学习反向转移分布 (3) 式的方差 $\sigma _ n ^2$，在之前的论文中，$\sigma ^ 2$ 都是手动选取值，例如取前向转移的方差 $\beta _ n$，或者后验分布 $q(x _ {n-1}|x _ n, x _ 0)$ 的方差 $\tilde \beta _ n$，这分别对应方差的上下限，或者是通过模型学习一个向量 $v$，然后对上下限进行插值 $\exp (v \log \beta _ t + (1-v) \log \tilde \beta _ t)$ 。DDIM 中则取 $\sigma _ n ^ 2 = \lambda _ n ^ 2$ 。

作者认为这些手动选择的值不是 (11) 式的最佳解，从而导致性能不是最优。

## 3.1 反向方差的最优解析解

**# 定理 1** 使用得分表示 (11) 式的最优解

(11) 式的最优解即，最优期望 $\mu _ n ^ {\star} (x _ n)$ 和最优方差 $\sigma ^ {\star}$ 有解析形式，是关于得分函数的解析形式，如下所示，

$$\mu_n ^ {\star}(x_n) = \tilde {\mu} _ n \left(x _ n,\frac 1 {\sqrt {\overline \alpha_n}}(x_n+ {\overline \beta_n} \nabla _ {x_n} \log q _ n(x _ n))\right) \tag{13}$$

$$\sigma_n^{\star 2}=\lambda_n^2 + \left( \sqrt {\frac {\overline \beta_n}{\alpha_n}} - \sqrt {\overline \beta_{n-1}-\lambda_n^2} \right) ^ 2 \left(1-\overline \beta_n \mathbb E_{q_n(x_n)} \frac {\|\nabla_{x_n} \log q_n(x_n)\|^2}{d} \right) \tag{14}$$

其中 $q _ n(x _ n)$ 是前向过程的边缘分布，$d$ 是数据维度，例如图像维度为 $3 \times H \times W$ 。

当 $\lambda _ n ^ 2 = \tilde \beta _ n$ 时，定理 1 可以进一步简化为 DDPM 前向过程，详情见附录 D。也可以将定理 1 扩展到连续型 timesteps，此时相应的最佳期望和最佳方差也是关于得分函数的解析形式，见附录 E.1 。

定理 1 表明，最佳反向过程的方差 $\sigma _ n ^ {\star 2}$ 在给定一个预先训练好的 score-based model $s _ n (x _ n)$ 之后不需要额外的训练，实际上，根据 (14) 式，要计算 $\sigma _ n ^ {\star 2}$，首先估计 $|| \nabla _ {x _ n} \log q _ n (x _ n)|| ^ 2 / d$ 的期望，使用模型得分代替得分函数 $\nabla _ {x _ n} \log q _ n (x _ n)$，期望计算使用蒙特卡洛估计，对 $x _ n$ 采样 $M$ 次，计算每个样本通过模型后的得分 $s _ n (x _ {n,m})$，那么蒙特卡洛估计为

$$\Gamma _ n = \frac 1 M \sum _ {m=1} ^ M \frac {|| s _ n (x _ {n, m})||^2} d, \quad x _ {n,m} \sim q _ n (x _ n) \tag{15}$$

对于 $N$ 个 timesteps，计算 $\Gamma = (\Gamma _ 1, \ldots, \Gamma _ N)$ 。对一个预训练好的模型，一次性计算好 $\Gamma$ 即可。附录 H.1 讨论了 $\Gamma$ 计算成本。 根据 (14) 式，可知 $\sigma _ n ^ {\star 2}$ 的估计为 

$$\hat \sigma _ n ^ 2 = \lambda _ n ^ 2 + \left (\sqrt {\frac {\overline \beta _ n} {\alpha _ n}} - \sqrt {\overline \beta _ {n-1} - \lambda _ n ^ 2} \right) ^ 2 (1-\overline \beta _ n \Gamma _ n) \tag{16}$$

图 1 (a) ，作者画出了 DDPM 基于 CIFAR10 数据集训练的模型，然后估计方差得到 $\hat \sigma _ n ^ 2$，并画出了 $\beta _ n$ 和 $\tilde \beta _ n$（这两者作为 baseline 方便对比），在 timestep 较小的时候，这三者相差较大。图 1 (b) 说明对 $L _ {vb}$ 的每一项（参见 (10) 式），$\hat \sigma _ n ^ 2$ 方案都比 baselines 更好，尤其在 timestep 较小时。在其他数据集熵，作者也得到相似的结果（见附录 G.1）。作者发现，估计 $\Gamma$ 时，只需要较小的蒙特卡洛样本数，例如 $M=10,100$ ，性能与大 $M$ 的估计值差不多（见附录 G.2）。在附录 H.2 中作者讨论了使用 $\hat \sigma _ n ^ 2$ 后 $L _ {vb}$ 的随机性。

![](/images/diffusion_model/analytic_dpm_1.png)

<center>图 1. </center>

## 3.2 讨论

我们这里讨论一下最佳反向期望和方差。

目标优化问题为 (11) 式，以下有两种方案解决 (11) 式优化问题。

### 3.2.1 方法一

将 $L _ {vb}$ 展开为 (10) 式，

对于最后一项 $D _ {KL}(q(x _ N|x _ 0)||p(x _ N))$，由于 $p(x _ N)=\mathcal N(x _ N|\mathbf 0,I)$，$q (x _ N|x _ 0)=\mathcal N(x _ N|\sqrt {\overline \alpha _ N} x _ 0, \overline \beta _ N I)$ 均为已知，可看作为常量。

对于中间项 $D_{KL}(q(x_{n-1}|x_0,x_n)||p(x_{n-1}|x_n))$，最优解处必然是，$p(x _ {n-1}|x _ n)$ 与 $q(x _ {n-1}|x _ 0, x _ n)$ 相同，而后者分布见 (2) 式，所以 $p(x _ {n-1}|x _ n)$ 的期望应该也是 $\tilde \mu _ n(x _ n, x _ 0)$，然而反向（采样）过程中，$x _ 0$ 未知，所以根据

$$\nabla _ {x _ n} q _ n (x _ n) = - \frac {x _ n - \sqrt {\overline \alpha _ n} x _ 0} {\overline \beta _ n}$$

变换得 

$$x _ 0 = \frac 1 {\sqrt {\overline \alpha _ n}}(x _ n + \overline \beta _ n \nabla _ {x _ n} q _ n (x _ n))$$

这样就得到 (13) 式 $\mu _ n ^ {\star}$ 。

$p(x _ {n-1}|x _ n)$ 的方差也应该与 $q(x _ {n-1}|x _ 0, x _ n)$ 相同，即方差为 $\lambda _ n ^ 2$。

对于第一项 $\mathbb E_q (-\log p(x_0|x_1))$，最小值需要满足 $p(x _ 0|x _ 1)=q(x _ 0|x _ 1)$，而 $q(x _ 0|x _ 1)$ 未知，在 DDPM 中，直接简单的使用 $q(x _ 0|x _ 1, \frac 1 {\sqrt {\overline \alpha _ 1}} (x _ 1 - \sqrt {1-\overline \alpha _ 1} \epsilon))$ 表示 $q(x _ 0|x _ 1)$。这么处理，并不能保证 $L _ {vb}$ 最小。

### 3.2.2 方法二

方法二就是本文的最佳期望和方差的推导。核心是 Lemma 9，求的是 $D _ {KL}(q (x _ {n-1}|x _ n)||p(x _ {n-1}|x _ n))$ 的最小值，而不是方法一中的 $D_{KL}(q(x_{n-1}|x_0,x_n)||p(x_{n-1}|x_n))$ 的最小值，这样就不再出现方法一中的 $\mathbb E_q (-\log p(x_0|x_1))$ 这项，而是全部统一为 $N$ 个 $D _ {KL}(q (x _ {n-1}|x _ n)||p(x _ {n-1}|x _ n))$ 项。 

## 3.2 最优反向方差的边界

根据 (14) 和 (16) 式，$\hat \sigma _ n ^ 2$ 的估计偏差为

$$|\sigma _ n ^ {\star 2} - \hat \sigma _ n ^ 2| = \left (\sqrt {\frac {\overline \beta _ n} {\alpha _ n}} - \sqrt {\overline \beta _ {n-1} - \lambda _ n ^ 2} \right) ^ 2 \overline \beta _ n |\Gamma _ n - E _ {q _ n (x _ n)} \frac {||\nabla _ {x _ n} \log q _ n(x _ n)|| ^ 2} d| \tag{17}$$

上式中，右边差的绝对值部分为近似误差，左边则是系数。由于使用模型得分近似得分函数，所以这部分的近似误差是不可约简的，同时如果 timesteps 步数较小，那么系数会较大，这里简单解释一下：$\beta_n$ 的起始截止范围为 $10 ^ {-4} \cdot \frac {1000} N$ 和 $0.01 \cdot \frac {1000} N$，$\alpha _ n = 1-\beta _ n$，$\overline \alpha _ n = \prod _ {i=1} ^ n \alpha _ i, \ \overline \beta _ n = 1- \overline \alpha _ n$，当 $N$ 减小，$\beta _ n$ 增大，$\alpha _ n$ 减小，$\overline \alpha _ n$ 减小，$\overline \beta _ n$ 增大，(17) 式系数部分中，$\overline \beta _ n/\alpha _ n=1/\alpha _ n - \overline \alpha _ {n-1}$，$\overline \beta _ {n-1}=1-\overline \alpha _ {n-1}$，前者比后者随 $N$ 减小而增加的幅度更大，因为多了一个 $1/\alpha _ n$，所以系数部分的差的平方 $(\cdot) ^ 2$ 也增大，所以系数增大，最终导致偏差 (17) 式增大，详见第 4 节内容。

为了降低偏差，首先推导出 $\sigma _ n ^ {\star 2}$ 的边界，然后将估计 $\hat \sigma _ n ^ 2$ clip 到边界内。注意，边界与数据分布 $q(x _ 0)$ 无关。

**# 定理 2** 最佳反向方差的边界

$$\lambda _ n ^ 2 \le \sigma _ n ^ {\star 2} \le \lambda _ n ^ 2 + \left(\sqrt {\frac {\overline \beta _ n} {\alpha _ n}} - \sqrt {\overline \beta _ {n-1} - \lambda _ n ^ 2} \right) ^ 2 \tag{18}$$

如果我们进一步假设 $q(x _ 0)$ 是一个在 $[a,b]^d$ 内的有界分布，这里 $d$ 是数据维度，那么方差上限变为

$$\sigma _ n ^ {\star 2} \le \lambda _ n ^ 2 + \left(\sqrt {\overline \alpha _ {n-1}} - \sqrt {\overline \beta _ {n-1} - \lambda _ n ^ 2} \right) ^ 2 \left(\frac {b-a} 2\right) ^ 2 \tag{19}$$

定理 2 表明 $\lambda _ n ^ 2$ 是下限，例如 DDPM 中 $\lambda _ n ^ 2 = \tilde \beta _ n$ （回顾 DDPM 中提到 $\tilde \beta _ n$ 和 $\beta _ n$ 分别是下限和上限）。图 1 (a) 与定理 2 一致。

# 4. 最佳轨迹的解析估计

full timesteps 步数 $N$ 太大，使得 inference 太慢，考虑构建一个较短的前向过程 $q(x _ {\tau _ 1}, \cdots, x _ {\tau _ K}|x _ 0)$，轨迹为 ${\color{red}1 = \tau _ 1} < \cdots < {\color{red}\tau _ K = N}$ 共 $K$ 个 timesteps，$K$ 可以明显小于 $N$，这样可以加快 inference。较短过程定义如下

$$\begin{aligned}q(x _ {\tau _ 1}, \ldots, x _ {\tau _ K}|x _ 0) &= q(x _ {\tau _ K}|x _ 0)\prod _ {k=2} ^ K q(x _ {\tau _ {k-1}}|x _ {\tau _ k}, x _ 0)
\\\\ q(x _ {\tau _ {k-1}}|x _ {\tau _ k}, x _ 0) &= \mathcal N(x _ {\tau _ {k-1}}|\tilde \mu _ {\tau _ {k-1}|\tau _ k} (x _ {\tau _ k}, x _ 0),  \lambda _ {\tau _ {k-1}|\tau _ k} ^ 2 I)
\\\\ \tilde \mu _ {\tau _ {k-1}|\tau _ k} (x _ {\tau _ k}, x _ 0) &= \sqrt {\overline \alpha _ {\tau _ {k-1}}} x _ 0 + \sqrt {\overline \beta _ {\tau _ {k-1}} - \lambda _ {\tau _ {k-1}|\tau _ k} ^ 2} \cdot \frac {x _ {\tau _ k} - \sqrt {\overline \alpha _ {\tau _ k}} x _ 0} {\sqrt {\overline \beta _ {\tau _ k}}}
\end{aligned} \tag{20}$$

反向过程为 $p(x _ 0, x _ {\tau _ 1}, \cdots, x _ {\tau _ K})=p(x _ {\tau _ K}) \prod _ {k=1} ^ K p(x _ {\tau _ {k-1}}|x _ {\tau _ k})$，其中反向转移过程定义为

$$p(x _ {\tau _ {k-1}}|x _ {\tau _ k})=\mathcal N(x _ {\tau _ {k-1}}|\mu _ {\tau _ {k-1}|\tau _ k} (x _ {\tau _ k}), \sigma _ {\tau _ {k-1}|\tau _ k} ^ 2 I) \tag{21}$$

以上均与使用 $N$ 个 timesteps 情况类似，仿照上文 (2) 和 (3) 式写出，注意下标 $\tau _ {k-1} | \tau _ k$ 表示时刻 $\tau _ {k-1}$ 与上一时刻 $\tau _ k$ 关联（反向过程）。根据定理 1，最佳（指具有最小 KL） $p ^ {\star}(x _ {\tau _ {k-1}}|x _ {\tau _ k})$ 的期望和方差为

$$\mu _ {\tau _ {k-1}|\tau _ k} ^ {\star} (x _ {\tau _ k})=\tilde \mu _ {\tau _ {k-1}|\tau _ k} \left(x _ {\tau _ k}, \frac 1 {\sqrt {\overline \alpha _ {\tau _ k}}} (x _ {\tau _ k} + \overline \beta _ {\tau _ k} \nabla _ {x _ {\tau _ k}} \log q(x _ {\tau _ k}))\right) \tag{22}$$

$$\sigma _ {\tau _ {k-1}|\tau _ k} ^ {\star 2}=\lambda _ {\tau _ {k-1}|\tau _ k} ^ 2 + \left(\sqrt {\frac {\overline \beta _ {\tau _ k}}{\alpha _ {\tau _ k|\tau _ {k-1}}}} - \sqrt {\overline \beta _ {\tau _ {k-1}} - \lambda _ {\tau _ {k-1}|\tau _ k} ^ 2} \right) ^ 2 \left(1 - \overline \beta _ {\tau _ k} E _ {q (x _ {\tau _ k}) } \frac {||\nabla \log q (x _ {\tau _ k})|| ^ 2} d \right)\tag{23}$$

其中 $\alpha _ {\tau _ k | \tau _ {k-1}}:= \overline \alpha _ {\tau _ k} / \overline \alpha _ {\tau _ {k-1}}$ ，这是因为根据 Lemma 13 的证明过程可知，

$$\alpha _ {\tau _ k | \tau _ {k-1}}=\sqrt {\overline \alpha _ {\tau _ {k-1}}} \cdot \sqrt {\frac {\overline \beta _ {\tau _ k}}{\overline \alpha _ {\tau _ k}}}= \sqrt {\frac {\overline \beta _ {\tau _ k}}{\overline \alpha _ {\tau _ k}/\overline \alpha _ {\tau _ {k-1}}}} \tag {24}$$

注意 $\alpha _ {\tau _ k | \tau _ {k-1}}$ 与 $\alpha _ n$ 是两种不同的概念。 这里使用 DDIM 的前向过程，更多的是关注并使用 $\overline \alpha _ {\tau _ k}$ 和 $\overline \beta _ {\tau _ k}$， $\alpha _ {\tau _ k | \tau _ {k-1}}$ 则是根据 Lemma 13 推导出来的概念，与原生的 $\alpha _ n$ 不同，而 $\overline \alpha _ {\tau _ k}$ 和 $\overline \beta _ {\tau _ k}$ 则分别与原生的 $\overline \alpha _ n$ 和 $\overline \beta _ n$ 相同。

$\overline \alpha _ {\tau _ k}=\prod _ {i=1} ^ {\tau _ k} \alpha _ i$，$\overline \beta _ {\tau _ k}=1-\overline \alpha _ {\tau _ k}$，且 (20) 式可以保证前向边缘分布为 $q(x _ {\tau _ k}|x _ 0) = \mathcal N(x _ {\tau _ k}| \sqrt {\overline \alpha _ {\tau _ k}} x _ 0, \overline \beta _ {\tau _ k} I)$，这与使用 $N$ 个 timesteps 时在相同的 $\tau _ k$ 时刻的前向边缘分布相同。

根据定理 2，$\sigma _ {\tau _ {k-1}|\tau _ k} ^ {\star 2}$ 有类似的边界，详情参见附录 C。根据 (16) 式，$\sigma _ {\tau _ {k-1}|\tau _ k} ^ {\star 2}$ 的估计为

$$\hat \sigma _ {\tau _ {k-1}|\tau _ k} ^ 2 = \lambda _ {\tau _ {k-1}|\tau _ k} ^ 2 + \left (\sqrt {\frac {\overline \beta _ {\tau _ k}} {\alpha _ {\tau _ k|\tau _ {k-1}}}} - \sqrt {\overline \beta _ {\tau _ {k-1}} - \lambda _ {\tau _ {k-1}|\tau _ k} ^ 2} \right ) ^ 2 (1 - \overline \beta _ {\tau _ k} \Gamma _ {\tau _ k}) \tag{25}$$

其中 $\Gamma$ 定义为 (15) 式，选用不同的轨迹，$\Gamma$ 都相同，所以只需要一次性事先计算好就行。

现在优化目标变为

$$\min _ {\tau _ 1, \cdots, \tau _ K} D _ {KL}(q(x _ 0, x _ {\tau _ 1}, \cdots, x _ {\tau _ K})||p ^ {\star}(x _ 0, x _ {\tau _ 1}, \cdots, x _ {\tau _ K})) = \frac d 2 \sum _ {k=2} ^ K J(\tau _ {k-1}, \tau _ k) + c \tag{26}$$

其中 $J(\tau _ {k-1}, \tau _ k)=\log (\sigma _ {\tau _ {k-1}|\tau _ k} ^ {\star 2}/ \lambda _ {\tau _ {k-1}|\tau _ k} ^ 2)$，$c$ 是与轨迹 $\tau$ 无关的常量。

(26) 式这个优化问题是一个经典的最小成本路径问题（动态规划），节点为 $\\{1, 2, \cdots, N \\}$，从 $s$ 到 $t$ 的边具有成本 $J(s, t)$ ，优化目标是找到一个包含 $K$ 个节点的最小成本路径，这 $K$ 个节点，固定从 $1$ 开始到节点 $N$ 节点。

这个动态规划问题的求解见附录 B。

我们也可以将 (26) 式扩展到连续 timesteps 情况，此时最佳 KL 散度也可以分解为各个 terms，每个 term 由得分函数确定。关于这个扩展的讨论见附录 E.2 。

# A. 推导证明

## A.1 Lemmas

**# Lemma 1.** 与高斯分布的交叉熵

假设 $q(x)$ 的期望和方差分别为 $\mu _ q, \ \Sigma _ q$，且 $p(x) = \mathcal N(x|\mu, \Sigma)$ 是一高斯分布，那么 $q$ 与 $p$ 的交叉熵等于 $\mathcal N(x|\mu _ q, \Sigma _ q)$ 与 $p$ 的交叉熵，即

$$\begin{aligned}H(q, p)&=H(\mathcal N(x|\mu _ q, \Sigma _ q), p)
\\\\ &= \frac 1 2 \log [(2\pi) ^ d|\Sigma|] + \frac 1 2 tr(\Sigma _ q \Sigma ^ {-1}) + \frac 1 2 (\mu _ q - \mu) ^ {\top} \Sigma ^ {-1} (\mu _ q - \mu)
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
\\\\ &=\frac 1 2 \log[(2\pi) ^ d |\Sigma|] + \frac 1 2 tr(\Sigma _ q \Sigma ^ {-1}) + \frac 1 2 (\mu _ q - \mu) ^ {\top}\Sigma ^ {-1} (\mu _ q - \mu) \qquad \qquad \qquad \square
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

其中 $c=\sum _ {n=1} ^ N E _ q H(q(x _ {n-1}|x _ n)) - E _ q H(q(x _ {0:N-1}|x _ N))$ 仅与 $q$ 相关。特别地，如果 $q(x _ {0:N})$ 也是马尔可夫链，那么 $c=0$ 。

证明：

$$\begin{aligned} & E _ q D _ {KL}  (q(x _ {0:N-1}|x _ N)||p(x _ {0:N-1}|x _ N)) = -E _ q \log p(x _ {0:N-1}|x _ N) - E _ q H(q(x_{0:N-1}|x _ N))
\\\\ = & -\sum _ {n=1} ^ N E _ q \log p(x _ {n-1}|x _ n) -  E _ q H(q(x_{0:N-1}|x _ N))
\\\\ =& \sum _ {n=1} ^ N E _ q D _ {KL} (q(x _ {n-1}|x _ n)||p(x _ {n-1}|x _ n)) + \sum _ {n=1} ^ N E _ q H(q(x _ {n-1}|x _ n)) - E _ q H(q(x _ {0:N-1}|x _ N))
\end{aligned}$$

令 $c = \sum _ {n=1} ^ N E _ q H(q(x _ {n-1}|x _ n)) - E _ q H(q(x _ {0:N-1}|x _ N))$，得证。

如果 $q(x _ {0:N})$ 也是马尔可夫链，那么根据 Lemma 4 有

$$H(q(x _ {0:N})) = H(q(x _ N)) + \sum _ {n=1} ^ N E _ q H(q(x _ {n-1}|x _ n)) $$

移项得

$$\begin{aligned}\sum _ {n=1} ^ N E _ q H(q(x _ {n-1}|x _ n)) &=H(q(x _ {0:N})) - H(q(x _ N))
\\\\ &= - E _ {q(x _ {0:N})} [\log q(x _ {0:N}) - \log q(x _ N)]
\\\\ &= - E _ {q(x _ {0:N})} \log q(x _ {0:N}|x _ N)
\\\\ &= E _ q(x _ N) H(q (x _ {0:N-1}|x _ N))
\end{aligned}$$

所以 $c = 0$， 证毕。

**# Lemma 9.** 最佳高斯转移的马尔可夫反向过程等价于 moment matching

假设 $q(x _ {0:N})$ 是一概率密度函数，且 $p(x _ {0:N})=\prod _ {n=1} ^ N p(x _ {n-1}|x _ n) p(x _ N)$ 是高斯型马尔可夫链，其中 $p(x _ {n-1}|x _ n)=\mathcal N(x _ {n-1}|\mu _ n(x _ n), \sigma _ n ^ 2 I)$，那么联合 KL 最优问题

$$\min _ { \{\mu _ n, \sigma _ n ^ 2 \} _ {n=1} ^ N } D _ {KL} (q (x _ {0:N})|| p(x _ {0:N}))$$

有最优解

$$\mu _ n ^ {\star}(x _ n) = E _ {q(x _ {n-1}|x _ n)}[x _ {n-1}], \quad \sigma _ n ^ {\star 2} = E _ {q _ n (x _ n)} \frac {tr(Cov _ {q(x _ {n-1}|x _ n)} [x _ {n-1}])} d$$

对应的最佳 KL 为

$$D _ {KL}(q(x _ {0:N})||p ^ {\star}(x _ {0:N}))=H(q(x _ N), p(x _ N)) + \frac d 2 \sum _ {n=1} ^ N \log (2\pi e \sigma _ n ^ {\star 2}) - H(q(x _ {0:N}))$$

## A.3 定理 2 证明

因为 $0 < \overline \beta _ n < 1$，且有

$$\mathbb E_{q_n(x_n)} \frac {\|\nabla_{x_n} \log q_n(x_n)\|^2}{d} > 0 \Rightarrow 1- \overline \beta _ n \mathbb E_{q_n(x_n)} \frac {\|\nabla_{x_n} \log q_n(x_n)\|^2}{d} < 1$$

根据定理 1 可知，

$$\lambda _ n ^ 2 \le \sigma _ n ^ {\star 2} \le \lambda _ n ^ 2 + \left( \sqrt {\frac {\overline \beta_n}{\alpha_n}} - \sqrt {\overline \beta_{n-1}-\lambda_n^2} \right) ^ 2$$

如果假设 $q(x _ 0)$ 是在 $[a,b]^d$ 范围内的有界分布，那么 $q(x _ 0|x _ n)$ 也是 $[a,b]^d$ 范围内的有界分布，根据 Lemma 12 有

$$E _ {q(x _ n)} \frac {tr (Cov _ {q(x _ 0|x _ n)}[x _ 0])} d \le (\frac {b - a} 2) ^ 2$$

再根据 Lemma 13 可得

$$\sigma _ n ^ {\star 2} \le \lambda _ n ^ 2 + \left(\sqrt {\overline \alpha _ {n-1}} - \sqrt {\overline \beta _ {n-1} - \lambda _ n ^ 2} \cdot \sqrt {\frac {\overline \alpha _ n} {\overline \beta _ n}} \right)^2 \left(\frac {b - a} 2 \right) ^ 2$$

## A.4 分解最佳 KL

**# 定理 3** shorter 前向过程和最佳反向过程之间的 KL 散度为

$$D _ {KL}(q(x _ 0, x _ {\tau _ 1}, \cdots, x _ {\tau _ K}) || p ^ {\star}(x _ 0, x _ {\tau _ 1}, \cdots, x _ {\tau _ K})) = \frac d 2 \sum _ {k=2} ^ K J(\tau _ {k-1}, \tau _ k) + c$$

其中 $J(\tau _ {k-1}, \tau _ k) = \log \frac {\sigma _ {\tau _ {k-1}|\tau _ k} ^ {\star 2}}{\lambda _ {\tau _ {k-1}|\tau _ k} ^ 2}$，$c$ 是与轨迹 $\tau$ 无关的常量。

证明：

$$\begin{aligned} & D _ {KL}(q(x _ 0, x _ {\tau _ 1}, \cdots, x _ {\tau _ K})|| p ^ {\star}(x _ 0, x _ {\tau _ 1}, \cdots, x _ {\tau _ K}))
\\\\ =& -E _ q [\log p ^ {\star}(x _ 0 | x _ {\tau _ 1}, \cdots, x _ {\tau _ K}) + \log p ^ {\star}(x _ {\tau _ 1}, \cdots, x _ {\tau _ K})-\log q(x _ 0 | x _ {\tau _ 1}, \cdots, x _ {\tau _ K}) - \log q(x _ {\tau _ 1}, \cdots, x _ {\tau _ K})]
\\\\ =& -E _ q [\log p ^ {\star}(x _ 0 | x _ 1) + \log p ^ {\star}(x _ {\tau _ 1}, \cdots, x _ {\tau _ K})-\log q(x _ 0 | x _ {\tau _ 1}, \cdots, x _ {\tau _ K}) - \log q(x _ {\tau _ 1}, \cdots, x _ {\tau _ K})]
\\\\ =& E _ q D _ {KL}(q(x _ 0|x _ {\tau _ 1}, \cdots, x _ {\tau _ K})|| p ^ {\star}(x _ 0|x _ 1)) + D _ {KL} (q(x _ {\tau _ 1}, \cdots, x _ {\tau _ K})||p ^ {\star}(x _ {\tau _ 1}, \cdots, x _ {\tau _ K}))
\\\\ \stackrel{Lemma-9}= & E _ q D _ {KL}(q(x _ 0|x _ {\tau _ 1}, \cdots, x _ {\tau _ K})|| p ^ {\star}(x _ 0|x _ 1)) + H(q(x _ N), p(x _ N)) \\\\ 
+ & \ \frac d 2 \sum _ {k=2} ^ K \log (2\pi e \sigma _ {\tau _ {k-1}|\tau _ k} ^ {\star 2}) - H(q(x _ {\tau _ 1}, \cdots, x _ {\tau _ K}))
\\\\ =& -E _ q \log p ^ {\star} (x _ 0|x _ 1) + H(q(x _ N), p(x _ N)) + \frac d 2 \sum _ {k=2} ^ K \log (2\pi e \sigma _ {\tau _ {k-1}|\tau _ k} ^ {\star 2}) - H(q(x _ 0, x _ {\tau _ 1}, \cdots, x _ {\tau _ K}))
\\\\ \stackrel{Lemma-7} = & -E _ q \log p ^ {\star} (x _ 0|x _ 1) + H(q(x _ N), p(x _ N)) + \frac d 2 \sum _ {k=2} ^ K \log (2\pi e \sigma _ {\tau _ {k-1}|\tau _ k} ^ {\star 2}) 
\\\\ - & \ H(q(x _ 0)) - \frac d 2 \log (2 \pi e \overline \beta _ N) - \frac d 2 \sum _ {k=2} ^ K \log (2 \pi e \lambda _ {\tau _ {k-1}|\tau _ k} ^ 2)
\\\\ = & -E _ q \log p ^ {\star} (x _ 0|x _ 1) +  H(q(x _ N), p(x _ N)) + \frac d 2 \sum _ {k=2} ^ K \log \frac {\sigma _ {\tau _ {k-1}|\tau _ k} ^ {\star 2}}{\lambda _ {\tau _ {k-1}|\tau _ k} ^ 2} - H(q(x _ 0)) - \frac d 2 \log (2 \pi e \overline \beta _ N)
\end{aligned}$$

令 $J(\tau _ {k-1}, \tau _ k) = \log \frac {\sigma _ {\tau _ {k-1}|\tau _ k} ^ {\star 2}}{\lambda _ {\tau _ {k-1}|\tau _ k} ^ 2}$，并且 $c = - E _ q \log p ^ {\star} (x _ 0|x _ 1) +  H(q(x _ N), p(x _ N))  - H(q(x _ 0)) - \frac d 2 \log (2 \pi e \overline \beta _ N)$，那么 $c$ 是与轨迹 $\tau$ 无关的常量，且有

$$D _ {KL}(q(x _ 0, x _ {\tau _ 1}, \cdots, x _ {\tau _ K})|| p ^ {\star}(x _ 0, x _ {\tau _ 1}, \cdots, x _ {\tau _ K})) = \frac d 2 \sum _ {k=2} ^ K J(\tau _ {k-1}, \tau _ k) + c$$

上面推导过程中，第二个等号成立是因为 $p^{\star}(x _ 0, x _ {\tau _ 1}, \cdots, x _ {\tau _ K})$ 是一个马尔可夫链，所以

$$p ^ {\star}(x _ 0 | x _ {\tau _ 1}, \cdots, x _ {\tau _ K})=p ^ {\star} (x _ 0 | x _ {\tau _ 1})$$

而 $\tau _ 1$ 固定为 $\tau _ 1 = 1$ 。

# B. DP 最小成本路径算法

给定一个成本函数 $J(s, t)$，其中 $1 \le s < t$。两个自然数 $k,n \ge 1$，我们需要找到一个路径 $1=\tau _ 1 < \cdots < \tau _ k = n$，路径中包含 $k$ 个节点，固定起始和终止节点分别为 $1$ 和 $n$，并且要使得总的成本为 $C[k,n]=\sum _ {i=2} ^ k J(\tau _ {i-1}, \tau _ i)$。

![](/images/diffusion_model/analytic_dpm_2.png)
图 2. 一个 mini 例子([来源](https://www.crcv.ucf.edu/wp-content/uploads/2018/11/paper_9_Analytic-DPM.pdf))

这是一个典型的动态规划（DP）问题，记 $C[k,n]$ 是最佳路径对应的最小成本，$D[k,n]$ 表示最佳路径中的 $\tau _ {k-1}$ 这个节点。为了简化问题，令 $J(s,t)=\infty$ 对 $s \ge t \ge 1$ 成立。那么，

1. $k=1$

    - 当 $n=1$ 时，路径 $1 = \tau _ k = n = 1$，起始节点就是截止节点，不需要经过任何路径，故总成本为 $0$
    - 当 $1\lt n \le N$ 时，此时没有满条条件的路径，故总成本为 $\infty$

    总结：

    $$C[1,n]=\begin{cases} 0 & n=1 \\\\ \infty & 1 \lt n \le N \end{cases}$$

    $D[1,n]=-1$，这是因为路径中就一个节点，故不存在最后第二个节点 $\tau _ {k-1}$

2. $2 \le k \le N$

    - $n < k$，此时不存在这样的路径，所以 $C[k,n]=\infty$ ，最后第二个节点 $D[k,n]=-1$。

    - $k \le n \le N$，将 $k$ 个节点路径，转换为两步：第一步 $k-1$ 个节点路径，路径末尾节点记作 $s$，第二步，从 $s$ 到 $n$，那么成本为 $C[k-1,s]+J(s,n)$。确定 $s$ 的取值范围，首先 $s < n$，即 $s\le n-1$，然后要求 $s \ge k-1$，否则第一步经过 $k-1$ 个节点到达 $s$ 无法实现，所以 $k-1 \le s \le n-1$，那么最终 
    
    $$C[k,n]=\min _ {k-1 \le s \le n-1} C[k-1,s]+J(s,n)=\min _ {1 \le s \le N} C[k-1,s] + J(s, n)$$

    第二个等式，当 $s \notin [k-1, n-1]$ 时，即 $s < k-1$ 时，$C[k-1,s]=\infty$（见上一种 case）；当 $s \ge n$ 时，$J(s,n)=\infty$（题目约定的，为了简化问题）。

    注意 $C[1,1]$ 与 $J(s,s)$ 不同，$C[1,1]$ 中不包含任何 $J$，而 $J(s, s)=\infty$ 这是为了禁止原地踏步。

    显然 $s$ 就是最佳路径的最后第二个节点，所以

    $$D[k,n]=\argmin _ {k-1 \le s \le n-1} C[k-1,s]+J(s,n)=\argmin _ {1 \le s \le N} C[k-1,s] + J(s,n)$$

计算最佳路径步骤：

按 $(k,s)$ pair 从小到大，依次计算 $C[k,s]$，以及对应的 $D[k,n]$，那么最小成本就是 $C[K,N]$，获取最优路径则需要倒序执行：令 $\tau _ K = N$，那么其前一个节点则是 $\tau _ {K-1}=D[K,\tau_K]$，从 $\tau _ 1=1$ 到 $\tau _ {K-1}$ 这个子路径的最小成本为 $C[K-1,\tau _ {K-1}]$，子路径的终点为 $\tau _ {K-1}$，其前一节点为 $\tau _ {K-2} = D[K-1,\tau _ {K-1}]$，依次进行下去，详见算法 1。

![](/images/diffusion_model/analytic_dpm_2.png)

<center>算法 1</center>