---
title: Score-Based Generative Modeling Through SDE
date: 2022-07-26 14:13:15
tags: diffusion model
mathjax: true
---

论文：[Score-based generative modeling through stochastic differential equations](https://arxiv.org/abs/2011.13456)

代码：[yang-song/score-sde](https://github.com/yang-song/score_sde)


# 1. 简介

两类概率生成模型。

## 1.1 SMLD

score matching + Langevin dynamics （SMLD）

score matching，模型 $s_{\theta}(\mathbf x)$ 模拟 $\nabla_{\mathbf x} \log p_{data}(\mathbf x)$，由于 $p_{data}(\mathbf x)$ 往往不可知，故采用 denoising score matching 方法，即，对数据做一个非常小的噪声扰动，

$$p_{\sigma}(\tilde {\mathbf x}|\mathbf x)=\mathcal N(\tilde {\mathbf x};\mathbf x,\sigma^2I) \tag{1}$$

根据上下文，可以知道 $I$ 为单位矩阵，下同。可以得到边缘分布

$$p_{\sigma}(\tilde {\mathbf x})=\int p_{data}(\mathbf x) p_{\sigma}(\tilde {\mathbf x}|\mathbf x) d\mathbf x \tag{2}$$

考虑一系列的扰动噪声水平 $\sigma_{min}=\sigma_1 < \ldots < \sigma_N = \sigma_{max}$ 。$\sigma_{min}$ 必须足够小使得 $p_{\sigma_{min}}(\mathbf x) \approx p_{data}(\mathbf x)$ ，且 $\sigma _ {max}$ 必须足够大，使得 $p _ {\sigma _ {max}} (\mathbf x) \approx \mathcal N(\mathbf 0, \sigma _ {max} ^ 2I)$。根据 [NCSN](/2022/07/22/diffusion_model/NCSN) 分析，由于数据位于低维度 manifolds，且多 mode 之间是数据低密度区域，会引起很多问题，所以**先选择最大的噪声扰动，然后逐步降低噪声幅度**。

（SMLD 论文中使用的网络框架作者称为 NCSN。）

训练的目标函数为

$$\theta^{\star}=\arg \min_{\theta} \sum_{i=1}^N \sigma_i^2 \mathbb E_{p_{data}(\mathbf x)}\mathbb E_{p_{\sigma_i}(\tilde {\mathbf x}|\mathbf x)}[\|\mathbf s_{\theta}(\tilde {\mathbf x}) - \nabla_{\tilde {\mathbf x}} \log p_{\sigma_i}(\tilde {\mathbf x}|\mathbf x)\|_2^2] \tag{3}$$

给定样本数据 $\mathbf x$，条件概率密度 $p_{\sigma_i}(\tilde {\mathbf x}|\mathbf x)$ 是我们预先设计好的 (设计扰动噪声方差 $\sigma_i$ 即可)，故条件概率密度已知，以此来监督训练模型参数 $\theta$ 。

假设数据足够多，且模型空间组大大，那么最优参数模型 $\mathbf s_{\theta{\star}}(\tilde {\mathbf x},\sigma)$ 几乎匹配所有的 $\nabla_{\mathbf x} \log p_{\sigma}(\tilde {\mathbf x}), \ i=1,\ldots, N$ （证明过程参见 [NCSN](/2022/07/22/diffusion_model/NCSN) 一文的 (3), (3.1), (3.2) 式以及相关结论）。

使用 Langevin dynamics 采样

$$\mathbf x_i^m = \mathbf x_i^{m-1}+ \epsilon_i \mathbf s_{\theta{\star}}(\mathbf x_i^{m-1}, \sigma_i) + \sqrt {2\epsilon_i} \mathbf z_i^m, \ m=1,\ldots, M \tag{4}$$

其中 $\epsilon_i$ 是 step size，$M$ 是 step number，$\mathbf z_i \sim \mathcal N(0,I)$。依据 (4) 式的采样过程为，重复 $i=N,N-1,\ldots, 1$（倒序是因为从大到小使用 $\sigma$ ），对每个 $i$ 值，设置 $\mathbf x_i^0=\mathbf x_{i+1}^M$，其中初始值 $\mathbf x_N^0 \sim \mathcal N(\mathbf x|\mathbf 0, \sigma_{max}^2 I)$，这是因为 $\sigma _ {max}$ 远大于 $\mathbf x _ 0$ 的 各像素值（范围为 [0, 1]，或者 scale 到 [-1,1]），故 $p _ {\sigma _ {max}} (\mathbf x) \approx \mathcal N(\mathbf 0, \sigma _ {max} ^ 2I)$。

## 1.2 DDPM

denoising diffusion probabilistic models，先前向过程对数据逐步加噪，然后反向过程中逐步降噪。即数据分布为 $\mathbf x_0 \sim p_{data}(\mathbf x)$，设置前向过程 

$$p(\mathbf x_i|\mathbf x_{i-1})=\mathcal N(\mathbf x_i; \sqrt {1-\beta_i} \mathbf x_{i-1},\beta_i I)\tag{5}$$

$\beta_i$ 是噪声方差，且 $0<\beta_1,\beta_2,\ldots, \beta_N < 1 $，由此可推导出

$$p_{\alpha_i}(\mathbf x_i|\mathbf x_0)=\mathcal N(\mathbf x_i; \sqrt {\alpha_i} \mathbf x_0, (1-\alpha_i)I) \tag{6}$$

其中 $\alpha_i = \prod_{j=1}^i (1-\beta_j)$，与原文中的 $\overline \alpha_i$ 相同，这里为了表达简单，省掉了顶部横线。

与 SMLD 相似，采用 denoising score matching，那么目标函数为

$$\theta^{\star}=\arg\min_{\theta} \sum_{i=1}^N (1-\alpha_i)\mathbb E_{p_{data}(\mathbf x)} \mathbb E_{p_{\alpha_i}(\tilde {\mathbf x}|\mathbf x)} [\|\mathbf s_{\theta}(\tilde {\mathbf x},i)-\nabla_{\tilde {\mathbf x}} \log p_{\alpha_i}(\tilde {\mathbf x}|\mathbf x_0)\|_2^2] \tag{7}$$

(7) 式中取 $\mathbf x=\mathbf x_0$。

根据 [NCSN](/2022/07/22/diffusion_model/NCSN) 分析，权重因子 $\lambda(\alpha_i) \propto (1-\alpha_i)$ 。

根据 [DDPM](/2022/06/27/diffusion_model/ddpm) 中 (12) 式可知 DDPM 的反向过程也是高斯分布，且高斯分布的期望为 $\tilde {\mu} _ i(\mathbf x_i, \mathbf x_0)$，这里再写出来如下，

$$\tilde {\mu} _ i(\mathbf x_i, \mathbf x_0)=\frac {\sqrt {\alpha_{i-1}}\beta_i}{1- \alpha_i}\mathbf x_0+\frac {\sqrt {\alpha_i}(1-\overline \alpha_{i-1})}{1- \alpha_i} \mathbf x_i \tag{8}$$

根据 (7) 式，训练完毕后应有

$$\mathbf s_{\theta ^ {\star}}(\mathbf x_i,i)=\nabla_{\mathbf x} \log p_{\alpha_i}(\mathbf x|\mathbf x_0)|_{\mathbf x=\mathbf x_i}=\frac {\sqrt {\alpha_i} \mathbf x_0 -\mathbf x_i}{1-\alpha_i} \tag{9}$$

(9) 式最后一个等号根据 (6) 式推导而得。训练阶段，$\mathbf x _ i = \sqrt {\alpha _ i} \mathbf x _ 0 + \sqrt {1-\alpha_i} \epsilon$，其中 $\epsilon \sim \mathcal N(\mathbf 0, I)$，代入 (9) 式有 $\mathbf s _ {\theta ^ {\star}}(\mathbf x _ i,i) = -\epsilon / \sqrt {1-\alpha_i}$ 。

测试阶段，$\mathbf x_0$ 是未知的，是我们通过反向过程生成的目标，那么 (8) 中反向过程的期望如何计算呢？将 (9) 式代入 (8) 式消去 $\mathbf x_0$ 得，

$$\mu_{\theta}(\mathbf x_i)=\frac 1 {\sqrt {1-\beta_i}} (\mathbf x_i+\beta_i \mathbf s_{\theta}(\mathbf x_i, i)) \tag{10}$$

于是反向过程的转换为

$$p_{\theta}(\mathbf x_{i-1}|\mathbf x_i)=\mathcal N(\mathbf x_{i-1}; \frac 1 {\sqrt {1-\beta_i}} (\mathbf x_i+\beta_i \mathbf s_{\theta}(\mathbf x_i, i)), \beta_iI) \tag{11}$$

其中方差直接使用 $\beta_i$ ，这与 DDPM 中一样（当然，也可以使用 $\tilde {\beta _ i}$，这是可以计算出来的，参见 [DDPM](/2022/06/27/diffusion_model/ddpm) 一文的 (12) 式，不过区别不大）。

根据 (7) 式训练结束后得到最优解 $\mathbf s_{\theta^{\star}}(\mathbf x,i)$，然后生成过程为从 $\mathbf x_N \sim \mathcal N(\mathbf 0, I)$ 开始，依据 (11) 式进行采样，

$$\mathbf x_{i-1}=\frac 1 {\sqrt {1-\beta_i}} (\mathbf x_i+\beta_i \mathbf s_{\theta^{\star}}(\mathbf x_i, i)) +\sqrt {\beta_i}\mathbf z_i, \quad i=N,N-1,\ldots, 1 \tag{12}$$

**# 与 DDPM 一文的方法联系**

DDPM 一文原本的训练目标是令模型输出 $\epsilon_ {\theta} (\mathbf x _ i, i)$ 逼近目标 $\epsilon$，结合 (9) 式，可知 score matching 的训练方法与 DDPM 原本的方法存在关系 $\mathbf s _ {\theta ^ {\star}} = - \epsilon _ {\theta} / \sqrt {1 - \alpha _ i} $ ，代入 (12) 式就得到  [DDPM](/2022/06/27/diffusion_model/ddpm) 一文得反向过程计算公式 (16) 式。

# 2. SDE 生成模型

思想：将上述按步加噪泛化到无限步加噪。为此，根据随机微分方程建立数据转换过程。

## 2.1 SDE 扰动

构造一个扩散过程 $\{\mathbf x(t)\}_{t=0}^T$，时刻 $t$ 是一个连续型变量，范围是 $t\in [0,T]$，数据分布记为 $\mathbf x(0) \sim p_0$，扩散最后的分布为 $\mathbf x(T) \sim p_T$。扩散/前向 过程使用 $It\hat o$ SDE 刻画，

$$d\mathbf x = \mathbf f(\mathbf x, t) dt + g(t) d\mathbf w \tag{13}$$

其中 $d\mathbf w =(W_{t+\Delta t}-W_t) \sim \mathcal N(0, \Delta t)$ 是标准布朗运动。$\mathbf f(\cdot, t): \mathbb R^d \rightarrow \mathbb R^d$ 是 $\mathbf x$ 的漂移系数，而 $g(\cdot): \mathbb R \rightarrow \mathbb R$ 是 $\mathbf x$ 的扩散系数。 关于 SDE 更多的知识可参考 [这里](/2022/07/25/math/SDE) 。

记 $\mathbf x(t)$ 的分布为 $p_t(\mathbf x)$，从 $\mathbf x(s)$ 到 $\mathbf x(t)$ 的转移记为 $p_{st}(\mathbf x(t)|\mathbf x(s))$，其中 $0 \le s < t \le T$ 。

## 2.2 反转 SDE

根据 $\mathbf x(T) \sim p_T$ 以及反向过程，可以采样得到来自 $p_0$ 的样本。根据 Brian D O Anderson. Reverse-time diffusion equation models. 1982. 的论文可知，反向 SDE 为

$$d\mathbf x=[\mathbf f(\mathbf x,t)-g(t)^2 \nabla_{\mathbf x} \log p_t(\mathbf x)] dt + g(t) d\overline {\mathbf w} \tag{14}$$

## 2.3 SDE 的得分估计

为了计算 (14) 式，我们回顾一下，

目标函数为

$$\theta^{\star}=\arg \min_{\theta} \mathbb E_t \{\lambda(t) \mathbb E_{p_0} \mathbb E_{p_{0t}}[\|\mathbf s_{\theta}(\mathbf x(t), t)-\nabla_{\mathbf x(t)} \log p_{0t}(\mathbf x(t)|\mathbf x(0))\|_2^2]\}\tag{15}$$

这里，$\lambda: [0,T] \rightarrow \mathbb R_{>0}$ 是一个权重函数，$t$ 是连续型在 $[0,T]$ 上均匀采样。当数据量足够以及模型容量足够大，那么可以求得 (15) 式的最优解记为 $\mathbf s_{\theta^{\star}}(\mathbf x,t) \approx \nabla_{\mathbf x} \log p_t$，对几乎所有的 $\mathbf x$ 和 $t$ 均成立。

根据 [NCSN](/2022/07/22/diffusion_model/NCSN) 中的 (7.1) 式，通常取

$$\lambda \propto 1 / \mathbb E[||\nabla_{\mathbf x(t)} \log p _ {0t} (\mathbf x(t) | \mathbf x (0))||^2]$$

这样每个时刻的误差期望值大小相当。

计算损失 (15) 式时，$\mathbf s _ {\theta}(\mathbf x (t),t)$ 就是模型输出，而得分函数这个 target 也很好计算，对于 DDPM，$p _ t(\mathbf x _ t|\mathbf x _ 0)=\mathcal N(\sqrt{\overline \alpha _ t} \mathbf x _ 0, (1-\overline \alpha _ t) I)$，于是得分函数为 $\nabla _ {\mathbf x _ t} \log p _ t (\mathbf x _ t|\mathbf x _ 0) = -(x _ t - \sqrt {\overline \alpha _ t} \mathbf x _ 0) / (1-\overline \alpha _ t)$，求这个梯度在模型输入数据 $\mathbf x _ t$ 处的值，模型输入为 $\mathbf x _ t = \sqrt {\overline \alpha _ t} \mathbf x _ 0+\sqrt {1-\overline \alpha _ t} \mathbf z$，代入梯度计算式得 $-\mathbf z/\sqrt {1-\overline \alpha _ t}$ ，所以 loss 计算代码为（下方 not likelihood_weighting 分支），

```python
# VESDE - SMLD
score = score_fn(perturbed_dat, t)  # 模型输出的得分估计：s(xt, t)

if not likelihood_weighting:    # 不使用似然权重，那么就是上述 lambda = \sigma ^ 2
    losses = torch.square(score * std[:,None,None,None] + z)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
else:
    losses = torch.square(score + z / std[:,None,None,None])
    # weighted
    g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2    # g(t)^2
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2
```

上述代码中，第一个分支，计算

$$\lambda(t) \cdot \left[\mathbf s _ {\theta} - \left(-\frac {\mathbf z} {\sqrt {1-\overline \alpha _ t}}\right)\right] ^ {\top} \left[\mathbf s _ {\theta} - \left(-\frac {\mathbf z} {\sqrt {1-\overline \alpha _ t}}\right)\right]=(\sigma \mathbf s _ {\theta} +\mathbf z) ^ {\top} (\sigma \mathbf s _ {\theta} +\mathbf z)$$

其中 $\lambda(t) = \sigma _ t ^ 2 = 1-\overline \alpha _ t$ 。

然后将 losses 从 `(B, 3, H, W)` reshape 为 `(B, 3 * H * W)`，然后 reduce （求和，因为计算 vector norm 平方就是求平方和）为 `(B,)`。

第二个计算分支，首先计算 unweighted loss，

$$\left[\mathbf s _ {\theta} - \left(-\frac {\mathbf z} {\sqrt {1-\overline \alpha _ t}}\right)\right] ^ {\top} \left[\mathbf s _ {\theta} - \left(-\frac {\mathbf z} {\sqrt {1-\overline \alpha _ t}}\right)\right]$$

然后乘以权重，这里权重使用 [Maximum Likelihood Training of Score-Based Diffusion Models](https://arxiv.org/abs/2101.09258) 中的方法，从代码上看，就是 SDE 中扩散系数的平方 $g(t) ^ 2$ 。

## 2.4 例子

前面分析的通过 (15) 式训练模型，但是其中 $p_{0t}$ 是什么形式，以及通过 (14) 式进行采样，(14) 式中的 $\mathbf f(\mathbf x,t)$ 和 $g(t)$ 又分别是什么。 这一节通过具体的例子进行讲解。

$p_{0t} = p _ {0t}(\mathbf x _ t|\mathbf x _ 0)$，描述了加噪后的数据分布，由 (13) 式决定。

### 2.4.1 VE
variance exploding （SMLD）

使用 N 个噪声 scale，每个扰动核为如下的马尔可夫链转移，

$$\mathbf x_i = \mathbf x_{i-1} + \sqrt {\sigma_i^2 - \sigma_{i-1}^2}\mathbf z_{i-1}, \quad i=1,\ldots,N \tag{16}$$

其中 $\mathbf z_{i-1} \sim \mathcal N(\mathbf 0, I)$，$\mathbf x_0 \sim p_{data}$。$\sigma_{min} = \sigma_1 < \sigma_2 < \cdots < \sigma_N=\sigma_{max}$ 。

**解释为什么是 (16) 的形式**：

在 SMLD 中，$\mathbf x_i$ 是在原数据 $\mathbf x_0$ 上加噪扰动而得，即 $\mathbf x _ i - \mathbf x _ 0 \sim \mathcal N(\mathbf 0, \sigma _i^2 I)$，类似地有 $\mathbf x _ {i-1} - \mathbf x _ 0 \sim \mathcal N(\mathbf 0, \sigma _ {i-1}^2 I)$，由于 $(\mathbf x _ i - \mathbf x _ 0)$ 与 $(\mathbf x _ {i-1} - \mathbf x _ 0)$ 独立（因为给定 $\mathbf x_0$ 时 $\mathbf x _ i$ 与 $\mathbf x _ {i-1}$ 条件独立），根据高斯分布的性质有

$$\mathbf x _ i - \mathbf x _ {i-1} \sim \mathcal N(\mathbf 0, (\sigma _ i ^ 2 - \sigma _ {i-1} ^ 2))$$

得到 (16) 式。注意 (16) 式与 DDPM 中的马尔科夫转移过程不同，DDPM 中 $\mathbf x _ i = \sqrt {1- \beta _ i} \mathbf x _ {i-1} + \sqrt {\beta _ i} \mathbf z _ i$ 。


为了统一表示，记 $\sigma _ 0 = 0$，于是

$$\mathbf x_i |\mathbf x_0 \sim \mathcal N(\mathbf x_0, (\sigma_i ^2 - \sigma_0^2)I) \tag{17}$$

当 $N \rightarrow \infty$ ，马尔科夫链 $\\{\mathbf x_i\\} _ {i=1}^N$ 变成连续型随机过程 $\\{\mathbf x(t)\\} _ {t=0}^1$，将原来的 $i$ 变为 $t=i/N$ 即可，$\sigma _i$ 和 $\mathbf z _ i$ 类似地变为 $\sigma(t)$ 和 $\mathbf z(t)$，那么 (16) 式马尔科夫转移过程变为

$$\mathbf x(t+\Delta t)=\mathbf x(t) + \sqrt {\sigma^2(t+\Delta t)-\sigma^2(t)}\mathbf z(t) = \mathbf x(t) + \sqrt {\frac {\Delta[\sigma^2(t)]}{\Delta t}\Delta t} \cdot \mathbf z(t) \tag{16.1}$$

由于 $\mathbf z(t) \sim \mathcal N(\mathbf 0,I)$，那么 $\sqrt {\Delta t} \mathbf z(t) \sim \mathcal N(0, \Delta t\cdot I)$，此即布朗运动的增量随机变量 $d\mathbf w$，结合上式可得

$$d\mathbf x = \sqrt {\frac {d[\sigma^2(t)]}{dt}}d\mathbf w \tag{18}$$

对比 (13) 式，可知

$$\mathbf f(\mathbf x,t)=\mathbf 0, \quad g(t)=\sqrt {\frac {d[\sigma^2(t)]}{dt}} \tag{19}$$

(18) 时微分方程的初始条件为 $t=0$ 时， $\mathbf x(0)$ 为真实数据。

于是我们知道了 $p_{0i}(\mathbf x_i|\mathbf x_0)$ 为 (17) 式，$\mathbf f(\mathbf x, t)$ 和 $g(t)$ 为 (19) 式，那么通过类比可知

$$p_{0t}(\mathbf x(t)|\mathbf x(0))=\mathcal N(\mathbf x(0), [\sigma ^ 2(t) - \sigma ^ 2(0)]I) \tag{17.1}$$

其中，$\sigma ^2 (t)$ 是预设的一个函数，例如某个线性递增函数，且 $t=0, \ t=1$ 两个端点值也预先给定。

### 2.4.2 VP

variance preserving（DDPM）

用在 DDPM 中，离散马尔科夫链为

$$\mathbf x _ {i+1} =\sqrt {1-\beta _ i} \mathbf x _ i + \sqrt {\beta _ i} \mathbf z _ i , \quad i=0,\ldots,N-1 \tag{20}$$

其中 $\mathbf z_i \sim \mathcal N(\mathbf 0, I)$ 。$0< \beta_i < 1, \ \forall i=0,\ldots,N-1$。 这里将变量下标做了调整，便于后续转为连续型变量时方便理解。

令 $\overline \beta_i = N \beta_i, \ N \rightarrow \infty$，经过变换得 $\beta_i = \overline {\beta _ i} \cdot \frac 1 N$，这里的 $\frac 1 N$ 将作为 $\Delta t$，将离散型变量 $\overline {\beta _ i}$ 换成连续型变量 $\overline {\beta} (t)$，那么 $\beta _ i$ 的连续型则变为 $\overline {\beta} (t) \cdot \Delta t$，另外 $\mathbf x _ i$ 的连续形式记为 $\mathbf x (t)$， 于是 (20) 式变为

$$\begin{aligned}\mathbf x(t +\Delta t)&=\sqrt {1-\overline {\beta}(t) \Delta t} \cdot \mathbf x(t)+\sqrt {\overline {\beta} (t)\Delta t} \cdot \mathbf z(t)
\\\\ & \stackrel{一阶展开} \approx \mathbf x(t) - \frac 1 2 \overline {\beta}(t)\Delta t \ \mathbf x(t) + \sqrt {\overline {\beta}(t)\Delta t} \ \mathbf z(t)
\end{aligned} \tag{21}$$


$t\in \\{0, \frac 1 N, \ldots, \frac {N-1} N \\}$，下标除以 N 使得 $t$ 的范围统一到 $[0,1]$ 。

根据上式推导结果可知

$$d\mathbf x = -\frac  1 2 \overline {\beta}(t) \mathbf x \ dt + \sqrt {\overline {\beta}(t)} \ d\mathbf w \tag{22}$$

对比 (13) 式可知

$$\mathbf f(\mathbf x, t)=-\frac 1 2 \overline {\beta}(t) \mathbf x, \quad g(t) =\sqrt {\overline {\beta}(t)} \tag{23}$$

其中 $\overline {\beta}(t)=N \beta _ i$ 。


**比较**：

1. SMLD 中，当 $t \rightarrow \infty$ 时，方差 $\sigma _ t ^2$ 将会爆炸，这是因为 $\sigma _ i ^ 2 - \sigma _ {i-1}^2 > 0$ 始终成立，求和时，如果级数不收敛，则 $\sigma _ t ^2 \rightarrow \infty$

2. DDPM 中，当 $t \rightarrow \infty$ 方差为 $1-\overline \alpha_t \rightarrow 1$

接下来计算 DDPM 的 $p(\mathbf x_i|\mathbf x_0)$，上面 (21) 式的种种变换和推导就是为了方便计算 $p(\mathbf x_i|\mathbf x_0)$。在原 DDPM 论文中，$q(\mathbf x_i |\mathbf x_0)$ 由 (6) 式给出。然而这里考虑连续情况，根据 SDE 进行推导，依据 (22) 式而非 (20) 式来描述马尔可夫链转移，故 $q(\mathbf x_i |\mathbf x_0)$ 不能使用 (6) 式，我们重新推导。

根据随机变量 $Z=X+Y$，其中 $X, \ Y$ 独立，那么有 $E[Z]=E[X]+E[Y]$ 的知识，对 (21) 式两边取期望，

$$\mathbf e(t+\Delta t)=\mathbf e(t)  - \frac 1 2 \overline {\beta}(t) \Delta t  \mathbf e(t)+\mathbf 0 \tag{24}$$

上式中，$\mathbf e(t) \stackrel{\Delta}= \mathbb E _ {\mathbf x(t)}[\mathbf x(t)]$, $\mathbb E _ {\mathbf z(t)}[\mathbf z(t)] = \mathbf 0$，$\mathbf e(t+\Delta t) \stackrel{\Delta}= \mathbb E _ {\mathbf x(t+\Delta t)} [\mathbf x(t+\Delta t)]$。



变换得

$$d \mathbf e(t) = -\frac 1 2 \overline {\beta}(t) \mathbf e(t) \ dt \tag{25}$$

对上式两边积分得，

$$\mathbf e(t)=C e^{-\frac 1 2 \int_0^t \overline {\beta}(s)ds} \tag{26}$$

根据初值条件 $\mathbf e(0)=\mathbf e_0$，可得 $C=\mathbf e_0$，于是

$$\mathbf e(t)=\mathbf e_0  e^{-\frac 1 2 \int_0^t \overline {\beta}(s)ds} \tag{27}$$

对 (21) 式两边取协方差，

$$\Sigma(t+\Delta t)=[1-\overline {\beta}(t) \Delta t] \cdot \Sigma(t)+\overline {\beta}(t)\Delta t I \tag{28}$$

故

$$d \Sigma(t)= \overline {\beta}(t) (I - \Sigma(t)) dt \tag{29}$$

积分得

$$\Sigma(t)=Ce ^ {\int_0 ^ t -\overline {\beta}(s) ds} + I \tag{30}$$

根据初值条件 $\Sigma(t)=\Sigma_0$，知 $C=\Sigma_0 -I$，于是

$$\Sigma(t)=(\Sigma_0 -I) e ^ {\int _ 0 ^ t -\overline {\beta}(s) ds} + I \tag{31}$$

于是条件分布为

$$\mathbf x_t| \mathbf x_0 \sim  \mathcal N\left(\mathbf e_0  e ^ {-\frac 1 2 \int _ 0 ^ t \overline {\beta}(s)ds}, (\Sigma_0 -I) e ^ {\int _ 0 ^ t -\overline {\beta}(s) ds} + I\right) \tag{32}$$

如果 $\mathbf x_0$ 已知，那么 $\mathbf e_0 =\mathbf x_0$，且 $\Sigma_0=\mathbf 0$，(32) 式变为

$$\mathbf x_t| \mathbf x_0 \sim  \mathcal N\left(\mathbf x_0  e ^ {-\frac 1 2 \int _ 0 ^ t \overline {\beta}(s)ds},  -I e ^ {\int _ 0 ^ t -\overline {\beta}(s) ds} + I\right) \tag{33}$$

**# post 1**:

上面 (33) 式给出了 VP SDE 的 $p(\mathbf x _ t | \mathbf x _ 0)$。VE SDE 中，也可以根据 (16.1) 式进行类似的计算得到 $p(\mathbf x(t)|\mathbf x(0))$ 的表达式，根据 (16.1) 式，期望 $\mathbf e(t+\Delta t)=\mathbf e(t)$，这是一个递归公式，可得 $\mathbf e(t)=\mathbf e(0)=\mathbf x(0)$，协方差 $\Sigma(t+\Delta t)=\Sigma (t)+ [\sigma ^ 2(t+\Delta t)-\sigma ^ 2 (t)] I$，根据这个递归公式并结合 $\Sigma(0)=0$ 可知 $\Sigma (t) = (\sigma ^ 2(t) - \sigma ^2 (0)] I$。

<details>
<summary>可以跳过的分析</summary>

**# post 2**:

VP SDE 中，如果从 (20) 式出发，那么

$$\mathbf e(t+\Delta t) = \sqrt {1- \beta(t)} \mathbf e(t) \Rightarrow d \mathbf e(t) = [\sqrt {1-\beta(t)}-1] \mathbf e(t)$$

这里的 $\beta(t)$ 是 $\beta _ i$ 的连续形式， 与上面的 $\overline {\beta}(t)= \beta(t) / \Delta t$ 不同。

上式变形，可得

$$d \log \mathbf e(t) = \sqrt {1-\beta(t)}-1$$

上式解为

$$\log \mathbf e(t) = \log \mathbf e(0) + \int _ 0 ^ t \sqrt {1-\beta(s)}-1$$

注意右侧积分项没有 $ds$ 这个因子，上式 变换之后得

$$\mathbf e(t) = \mathbf e(0) \cdot e^{\int _ 0 ^ t \sqrt {1-\beta(s)}-1}$$

DDPM 中 $\beta_i$ 一种选择方案是从 $10^{-4}$ 线性增大到 0.02，那么 $\beta(t)$ 形式为 $\beta(t)=kt + 10^{-4}$，但无论哪种方案，上式中的积分项都不好计算，所以想办法使用近似，使用泰勒一阶展开 $\sqrt {1-\beta(s)}=1-\frac 1 2 \beta(s)$，于是上式变为

$$\mathbf e(t) = \mathbf e(0) \cdot e ^ {-\frac 1 2 \int _ 0 ^ t \beta(s)}$$

上式指数部分的积分项中没有 $ds$，故依然不太好求积分，作变换 $\beta'(s)=\beta(s) N=\beta (s) / \Delta s$，这个 $\beta '(s)$ 实际上就是 $\overline {\beta} (s)$，于是 $\beta(s) =\overline {\beta}(s) ds$，于是上式变为

$$\mathbf e(t) = \mathbf e(0) \cdot e ^ {-\frac 1 2 \int _ 0 ^ t \overline {\beta}(s) ds}$$

上式与 (27) 式完全相同。嗯，这是可以预期的，毕竟 (20) 式的连续形式就是 (21) 式，且这里也与 (21) 式一样，做了泰勒一阶展开近似。

</details>

### 2.4.3 sub-VP

受 VP SDE 启发，作者提出了一个新的 SDE，称为 sub-VP SDE，如下

$$d\mathbf x = -\frac 1 2 \overline {\beta}(t) \mathbf x \ dt + \sqrt {\overline {\beta}(t)(1-e ^ {-2\int _ 0 ^ t \overline {\beta}(s)ds})} d\mathbf w \tag{34}$$

与 DDPM VP (22) 式类似，只是扩散系数 $g(t)$ 不同，$\mathbf x(t)$ 的期望满足 (25) 式，于是期望为 (27) 式。

协方差根据 (29) 式进行调整（多了一项指数积分），得

$$d\Sigma(t) =\overline {\beta}(t) (I - I e ^ {-2\int _ 0 ^ t \overline {\beta}(s)ds} - \Sigma(t)) dt$$

积分得 

$$\Sigma(t)=Ce ^ {\int _ 0 ^ t -\overline {\beta}(s) ds} I + I + I e ^ {-2\int _ 0 ^ t \overline {\beta}(s)ds}$$

根据初值条件 $\Sigma(t)=\Sigma_0$，于是 $\Sigma_0=CI+I+I$，于是

$$\Sigma(t)=(\Sigma_0-2I)e ^ {\int _0 ^ t -\overline {\beta}(s) ds} + I + I e ^ {-2\int_0 ^ t \overline {\beta}(s)ds}\tag{35}$$

**性质：**

1. 当 $\Sigma_{VP}(0)=\Sigma_{sub-VP}(0)$ 时，

    $$\Sigma_{VP}(t)-\Sigma_{sub-VP}(t)=I(e ^ {\int_0 ^ t -\overline {\beta}(s) ds}-e ^ {-2\int _ 0 ^ t \overline {\beta}(s)ds}) \succ 0 \cdot I$$

    由于 $-\int \overline {\beta}(s)ds - (-2 \int \overline {\beta}(s)ds )=\int \overline {\beta}(s)ds>0$，即 $-\int \overline {\beta}(s)ds > -2 \int \overline {\beta}(s)ds$，上式得证。

    这表明，使用 sub-VP SDE，$\mathbf x_t |\mathbf x_0$ 分布的方差更小。

2. 当 $\lim _ {t \rightarrow \infty} \int _ 0 ^ t \overline {\beta}(s) ds = \infty$ 时，

    $$\lim _ {t \rightarrow \infty} \Sigma _ {sub-VP} (t) = \lim _ {t \rightarrow \infty} \Sigma _ {VP} (t) = I$$

    将前向过程写成连续形式后，$t$ 可以不仅仅局限于 $[0, 1]$ 区间范围。

当 $\mathbf x_0$ 已知时，$\Sigma_0 = \mathbf 0$，此时 sub-VP 的扩散分布为

$$\mathbf x_t|\mathbf x_0 \sim \mathcal N \left(\mathbf x_0  e ^ {-\frac 1 2 \int _ 0 ^ t \overline {\beta}(t)ds},  (1-e ^ {\int_ 0 ^ t -\overline {\beta}(s) ds}) ^ 2 I\right) \tag{36}$$

# 3. 解反向 SDE

根据设计好的 $\mathbf f(\mathbf x,t)$ 和 $g(t)$ 可以计算出 $\mathbf x_t|\mathbf x_0$ 的分布，然后可以训练 score-based 模型 $\mathbf s_{\theta}$，训练 target 为 $\nabla_{\mathbf x _ t} \log p(\mathbf x _ t|\mathbf x)$ ，根据前向 SDE 可以得到 reverse-time SDE（根据论文 Anderson 1982），然后使用数值解法以生成来自 $p_0$ 分布的样本。
数值解法包括 Euler-Maruyama 和 随机 Runge-Kutta 方法等。

## 3.1 反向扩散采样

给定前向 SDE

$$d\mathbf x = \mathbf f(\mathbf x, t) dt + \mathbf G(t) d\mathbf w \tag{37}$$

其中 $\mathbf x \in \mathbb R^d, \ \mathbf G(t) \in \mathbb R^{d \times d}$ 。


离散化 (37) 式，将 $t, \ dt$ 从 (37) 式中剥离

$$\mathbf x_{i+1} = \mathbf x_i + \mathbf f_i(\mathbf x_i) + \mathbf G_i \mathbf z_i, \quad i=0,1,\ldots,N-1 \tag{38}$$

其中 $\mathbf z_i \sim \mathcal N(\mathbf 0, I)$ 。

根据论文 Anderson 1982，reverse-time SDE 为

$$d\mathbf x = [\mathbf f(\mathbf x, t)-\mathbf G(t)\mathbf G(t)^{\top} \nabla_{\mathbf x} \log p_t(\mathbf x)] dt + \mathbf G(t) d\overline {\mathbf w} \tag{39}$$

注意 (39) 式中 $\overline {\mathbf w}$ 表示从时间 $T$ 开始到时间 $0$ 结束，即 $\mathbf w_{s-t} - \mathbf w_s \sim \mathcal N(\mathbf 0, t I)$ 。(39) 式离散化为

$$\mathbf x_{i+1}-\mathbf x_i=\mathbf f_{i+1}(\mathbf x_{i+1})-\mathbf G_{i+1}\mathbf G_{i+1}^{\top} \mathbf s_{\theta}(\mathbf x_{i+1},i+1) - \mathbf G_{i+1}\mathbf z_{i+1}$$

变换得

$$\mathbf x_i=\mathbf x_{i+1}-\mathbf f_{i+1}(\mathbf x_{i+1})+\mathbf G_{i+1}\mathbf G_{i+1}^{\top} \mathbf s_{\theta}(\mathbf x_{i+1},i+1) + \mathbf G_{i+1}\mathbf z_{i+1}, \quad i=0,\ldots, N-1 \tag{40}$$

(40) 式就是离散化的采样规则。注意这里使用 $\mathbf f_{i+1}(\mathbf x_{i+1})$ 而不使用 $\mathbf f_{i}(\mathbf x_{i+1})$ 的写法是为了下标统一，故 (40) 式中的 $\mathbf f_1,\ldots, \mathbf f_N$ 就是 (38) 式中的 $\mathbf f_0, \ldots, \mathbf f_{N-1}$，对 $\mathbf G$ 的下标也是如此。

### 3.1.1 reverse-time VE SDE 采样

将 (40) 式应用于 (16) 式，得到 reverse-time VE (SMLD) SDE 采样器。将 (16) 和 (38) 式下标 $i$ 的取值范围统一为 $i=0,\ldots, N-1$，对比 (16) 和 (38) 式可知 (40) 式中的的相关函数为

$$\mathbf f_{i+1}(\mathbf x_{i+1})=\mathbf 0, \quad \mathbf G_{i+1}=I\sqrt {\sigma_{i+1}^2 - \sigma_i^2} \tag{40.1}$$

注意 (40.1) 式中表达式是离散形式，要与 (19) 式的连续形式区分开来。实际上，将 $d \mathbf w=\sqrt{dt} \mathbf z$ 中的 $\sqrt{dt}$ 放到 (19) 式 $g(t)$ 中，就得到 $\sqrt {d[\sigma ^ 2 (t)]}$，离散形式就是 $\sqrt {\sigma _ {i+1} ^ 2 - \sigma _ i ^ 2}$ ，所以离散形式与连续形式两者之间的转换其实不难理解。

将 (40.1) 式代入 (40) 式得到 VE 的采样过程，总结采样算法流程如下方算法 1 的蓝色部分。

---
**算法 1**： Predictor-Corrector(PC) 采样（VE SDE）

$\mathbf x_N \sim \mathcal N(\mathbf 0, \sigma_{max}^2I)$

**for** $i=N-1,\ldots, 0$ **do**

<font color='cyan'>&emsp; (Predictor)

&emsp; $\mathbf x_i' \leftarrow \mathbf x_{i+1} + (\sigma_{i+1}^2 - \sigma_i^2) \mathbf s_{\theta^{\star}}(\mathbf x_{i+1}, \sigma_{i+1})$

&emsp; $\mathbf z \sim \mathcal N(\mathbf 0, I)$

&emsp; $\mathbf x_i \leftarrow \mathbf x_i' + \sqrt {\sigma_{i+1}^2 - \sigma_i^2} \mathbf z$
</font>

<font color='orange'>&emsp; (Corrector)

&emsp; <font style='font-weight:bold'>for</font> $j=1,\ldots, M$ <font style='font-weight:bold'>do</font>

&emsp; &emsp; $\mathbf z \sim \mathcal N(\mathbf 0, I)$

&emsp; &emsp; $\mathbf x_i \leftarrow  \mathbf x_i + \epsilon_i \mathbf s_{\theta^{\star}}(\mathbf x_i, \sigma_i) + \sqrt {2 \epsilon_i} \mathbf z$
</font>

**return** $\mathbf x_0$

---


### 3.1.2 reverse-time VP SDE 采样

将 (40) 式应用于 (20) 式，得到 reverse-time VE (DDPM) SDE 采样器。对比 (20) 和 (38) 式可知 (40) 式中的的相关函数为

$$\mathbf f_{i+1}(\mathbf x_{i+1})=(\sqrt{1-\beta_{i+1}}-1)\mathbf x_{i+1}, \quad \mathbf G_{i+1}=I\sqrt {\beta_{i+1}} \tag{40.2}$$

于是采样算法流程如下方算法 2 的蓝色部分。

---
**算法 2**： Predictor-Corrector(PC) 采样（VP SDE）

$\mathbf x_N \sim \mathcal N(\mathbf 0, \sigma_{max}^2I)$

**for** $i=N-1,\ldots, 0$ **do**

<font color='cyan'>&emsp; (Predictor)

&emsp; $\mathbf x_i' \leftarrow (2-\sqrt{1-\beta_{i+1}})\mathbf x_{i+1} + \beta_{i+1} \mathbf s_{\theta^{\star}}(\mathbf x_{i+1}, i+1)$

&emsp; $\mathbf z \sim \mathcal N(\mathbf 0, I)$

&emsp; $\mathbf x_i \leftarrow \mathbf x_i' + \sqrt {\beta_{i+1}} \mathbf z$

</font>
<font color='orange'>&emsp; (Corrector)

&emsp; <font style='font-weight:bold'>for</font> $j=1,\ldots, M$ <font style='font-weight:bold'>do</font>

&emsp; &emsp; $\mathbf z \sim \mathcal N(\mathbf 0, I)$

&emsp; &emsp; $\mathbf x_i \leftarrow  \mathbf x_i + \epsilon_i \mathbf s_{\theta^{\star}}(\mathbf x_i, i) + \sqrt {2 \epsilon_i} \mathbf z$
</font>

**return** $\mathbf x_0$

---

作者称类似 算法 1 和 2 这样的基于 (40) 式的采样为 _reverse diffusion samplers_ 。

**# reverse-time SDE sampling 与 ancestral sampling 的联系**

以 VP (DDPM) 为例：

1. ancestral sampling

    DDPM 中 ancestral sampling 为 (12) 式，下面再次给出这个 (12) 式，省去前面翻看的麻烦。(12) 式可以通过贝叶斯定理计算出来，即根据前向马尔科夫转移分布 $p(\mathbf x _ i | \mathbf x _ {i-1})$ 计算后验分布 $p(\mathbf x _ {i-1} | \mathbf x _ i)$（这也是一个高斯分布），

    $$\mathbf x_{i-1}=\frac 1 {\sqrt {1-\beta_i}} (\mathbf x_i+\beta_i \mathbf s_{\theta}(\mathbf x_i, i)) +\sqrt {\beta_i}\mathbf z_i, \quad i=N,N-1,\ldots, 1 \tag{12}$$

    根据泰勒展开，

    $$(1-\beta_i)^{-\frac 1 2}=1+\frac 1 2 \beta_i + \frac 3 8 \beta_i^2 + \cdots$$

    当 $\beta_i \rightarrow 0$ 时，取一阶近似，代入 (12) 式

    $$\mathbf x_{i-1}=(1+\frac 1 2 \beta_i)\mathbf x_i + (\beta_i + \frac 1 2 \beta_i^2)\mathbf s_{\theta}(\mathbf x_i, i) + \sqrt {\beta_i} \mathbf z_i \tag{41.1}$$


2. reverse-time SDE sampling

    将 (40.2) 式代入 (40) 式得到 DDPM 的反向 SDE 采样，也就是算法 2 中蓝色部分，

    $$\mathbf x_{i-1}=(2-\sqrt {1-\beta _ i})\mathbf x_i + \beta_i\mathbf s_{\theta}(\mathbf x_i, i) + \sqrt{\beta_i} \mathbf z_i \tag{41.2}$$

    根据泰勒展开有 $(1-\beta _ i) ^ {\frac 1 2} = 1 - \frac 1 2 \beta _ i - \frac 1 8 \beta _ i ^ 2 + \cdots$ 取一阶近似并代入 (41.2) 式，得 
    
    $$\mathbf x_{i-1}=(1+ \frac 1 2 \beta _ i)\mathbf x _ i +  \beta_i\mathbf s_{\theta}(\mathbf x_i, i) + \sqrt{\beta_i} \mathbf z_i \tag{41.3}$$

忽略 (41.1) 式中的二阶项 $\beta_i^2$，可以发现 (41.1) 和 (41.3) 等价，即 **reverse-time SDE 出发推导出来的采样规则与 ancestral 采样结果等价**。


## 3.2 Predictor-Corrector 采样器

作者提出，可以使用 score-based MCMC 方法，例如 Langevin MCMC，从分布 $p _ i(\mathbf x _ i)$ 中直接采样，从而纠正 SDE 数值求解的结果。

具体而言，每个 step $i$，SDE 数值求解会给出一个样本估计，此过程充当 “predictor”，如算法 1 和 2 中蓝色部分的 $\mathbf x_i$，然后，score-based MCMC 方法纠正这个样本估计，如算法 1 和 2 中橙色部分（使用 Langevin dynamics 方法），此过程充当 “corrector”。

算法 1 和 2 中的 PC 采样就是指 Predictor-Corrector 采样。

## 3.3 概率流

对 scored-based 模型，求解 reverse-time SDE 有另一种数值法。对所有的扩散过程，均存在一种确定性过程，在其转移路径上边缘分布与 reverse-time SDE 的情况相同。这种确定性过程满足某个 ODE，

$$d\mathbf x=[\mathbf f(\mathbf x, t)-\frac 1 2 g(t)^2 \nabla_{\mathbf x} \log p_t(\mathbf x)]dt \tag{42}$$

详情参见附录 D，也就是将 (D3) 式中的 $\mathbf G(\mathbf x,t)$ 替换为 $g(t)$ 就得到 (42) 式。这样，一旦得分函数已知，(42) 式描述的过程就被确定，使用模型的得分函数 $\mathbf s_{\theta}(\mathbf x)$ 代替 (42) 式中的 $\nabla _ {\mathbf x} \log p _ t(\mathbf x)$。称 (42) 式这个 ODE 为 概率流 ODE （因为没有 $d\mathbf w$ 这一项，所以 (42) 式本质就是一个常微分方程）。当得分函数由时间相关的score-based model（这个 model 通常是神经网络 $\mathbf s_{\theta}(\mathbf x, t)$ ） 近似时，就是一个 neural ODE 。

附录 D 有关于概率流 ODE 更多的讨论和分析。


# 4. 总结

设计一个前向 SDE，包括 $\mathbf f(\mathbf x, t)$ 和 $\mathbf G(\mathbf x, t)$ ，从而可以计算出  $p(\mathbf x|\mathbf x_0)$ 以及 reverse-time SDE ，根据 reverse-time SDE 可以得到采样过程。根据 score matching，可训练出得分函数 $\mathbf s_{\theta}(\mathbf x _ t, t)$ （ $t$ 作为模型的 time embedding 输入）。然后使用 PC 采样生成来自 $p_0$ 的样本。

通过 SMLD 和 DDPM，验证了 SDE 求解问题的可行性，并与传统的离散化迭代过程（扩散过程）等价。所以我们可以直接从 SDE 出发，求解生成模型。


# APPENDIX

## A. general SDE 概述

考虑如下前向 SDE，

$$d\mathbf x = \mathbf f(\mathbf x, t) dt + \mathbf G(\mathbf x, t) d\mathbf w \tag{A1}$$

其中 $\mathbf f(\cdot, t): \mathbb R^d \rightarrow \mathbb R^d$，$\mathbf G(\cdot, t): \mathbb R^d \rightarrow \mathbb R^{d \times d}$ 。

注意 (37) 式是上式的一个特例：$\mathbf G$ 函数中没有 $\mathbf x$ 这个自变量。(A1) 式对应的 reverse-time SDE 为

$$d\mathbf x = \{\mathbf f(\mathbf x, t)-\nabla \cdot [\mathbf G(\mathbf x,t)\mathbf G(\mathbf x, t)^{\top}] - \mathbf G(\mathbf x,t)\mathbf G(\mathbf x, t)^{\top} \nabla_{\mathbf x} \log p_t(\mathbf x)\}dt + \mathbf G(\mathbf x, t)d\overline {\mathbf w} \tag{A2}$$

其中 $\mathbf G(\mathbf x,t)\mathbf G(\mathbf x, t)^{\top}$ 表示两个矩阵相乘。对于矩阵 $\mathbf F(\mathbf x)\stackrel{\Delta}=[\mathbf f^1(\mathbf x), \cdots, \mathbf f^d(\mathbf x)]^{\top}$，定义

$$\nabla \cdot F(\mathbf x)\stackrel{\Delta}=[\nabla \cdot \mathbf f^1(\mathbf x), \cdots, \nabla \cdot \mathbf f^d(\mathbf x)]^{\top}$$

**注意，这里的 $\cdot$ 不可省略**

上式是一个向量，向量各元素 $\nabla \cdot \mathbf f^i(\mathbf x)=\sum _ {i=1} ^ d \frac {\partial}{\partial x_i} f ^ i(\mathbf x)$。

这表示 $\nabla \cdot \mathbf F(\mathbf x)$ 是对矩阵 $\mathbf F(\mathbf x)$ 的第 $i$ 行求针对 $\mathbf x _ i$ 的偏导数，这是一个向量，所得的偏导矩阵再求每一行的和，最终就是一个列向量。

记 reverse-time SDE 为

$$d\mathbf x = \overline {\mathbf f}(\mathbf x, t)dt + \overline {\mathbf G}(\mathbf x, t) d \overline {\mathbf w} \tag{A3}$$

那么给定初值条件 $\mathbf x_T$ 时，可得 SDE 的解为

$$\mathbf x_T - \mathbf x_t=\int_t^{T} \overline {\mathbf f}(\mathbf x, s)ds + \int_t^T \overline {\mathbf G}(\mathbf x, s) d\overline {\mathbf w}(s) \tag{A5}$$

注意，实际上时间变量 $s$ 是从时刻 $T$ 到时刻 $t$，所以是 

$$\mathbf x _ t - \mathbf x _ T = \int _ T ^ t \overline {\mathbf f}(\mathbf x, s)ds + \int _ T ^ t \overline {\mathbf G}(\mathbf x, s) d\overline {\mathbf w}(s)$$

两边取反得到 (A5) 式。


## C. SDEs in the wild

本节讨论 VE 和 VP SDEs 的具体实例，这两者对应 SMLD 和 DDPM 模型，然后也分析了 sub-VP SDE。

### C.1 SMLD

SMLD 中，噪声 scales $\\{ \sigma_i \\} _ {i=1} ^ N$ 通常使用等比序列，其中固定 $\sigma_{min}=0.01$，$\sigma_{max}$ 则根据 《Improved Techniques for Training Score-Based Generative Models. Yang Song》一文中的 Technique 1 确定，根据等比数列性质有

$$\sigma(\frac i N)=\sigma_i = \sigma _ {min} \left(\frac {\sigma_{max}}{\sigma _ {min}}\right)^{\frac {i-1}{N-1}}$$

当 $N \rightarrow \infty$ 时有 

$$\sigma(t)=\sigma _ {min} (\frac {\sigma _ {max}} {\sigma_{min}})^t \tag{B1}$$

上式代入 VE SDE 方程 (18) 式，为

$$d \mathbf x = \sigma _ {min} \left(\frac {\sigma _ {max}} {\sigma _ {min}} \right)^t \sqrt {2 \log \frac {\sigma _ {max}} {\sigma _ {min}}} d \mathbf w \tag{B2}$$

根据 (17.1) 式，扰动核为 

$$p_{0t}(\mathbf x(t) | \mathbf x(0)) = \mathcal N \left(\mathbf x(t); \mathbf x(0), \sigma _ {min} ^ 2 \left ( \frac {\sigma _ {max} } {\sigma _ {min}}\right) ^ {2t} I  \right) \tag{B3}$$

注意这里为了统一，定义 $\sigma(0)=\sigma_0 = 0$，但是 $\sigma (0^+) := \lim _ {t \rightarrow 0^+} \sigma(t) = \sigma _ {min} \neq 0$。所以 $\sigma(t)$ 在 $t=0$ 处不可导，那么 VE SDE (18) 式在 $t=0$ 处未定义，实际应用中，我们解这个 SDE 以及对应得概率流 ODE 时，设置范围 $t \in [\epsilon, 1]$，例如作者使用 $\epsilon=10 ^ {-5}$ 。

### C.2 DDPM

DDPM 中 $\\{ \beta_i \\} _ {i=1} ^ N$ 通常是等差数列：$\beta_i = \beta _ 1 + \frac {i-1} {N-1} (\beta _ N - \beta _ 1), \ i=1,\ldots, N$。根据前面 (21) 式分析，令 $\overline {\beta} _ i = N \beta _ i$， 并将 $\overline \beta _ i$ 替换为连续型变量 $\overline \beta(t)$，那么

$$\overline \beta(t) = \overline \beta _ {min} + t (\overline \beta _ {max} - \overline \beta _ {min}), \ t \in [0, 1]$$

显然 $\overline \beta(t)$ 是 $t$ 的线性函数。

将上式代入 VP SDE (22) 式得到一个更具体的表达式如下，

$$d \mathbf x = -\frac 1 2 (\overline \beta _ {min} + t (\overline \beta _ {max} - \overline \beta _ {min})) \mathbf x dt + \sqrt {\overline \beta _ {min} + t (\overline \beta _ {max} - \overline \beta _ {min})} d \mathbf w \tag{B4}$$

作者实验中选择 $\overline \beta _ {min}=0.1, \ \overline \beta _ {max} = 20$。回顾一下 DDPM 中，选择 $\beta _ {min} = 10^{-4}, \ \beta _ {max}=0.02$（从 $10^{-4}$ 线性增大到 0.02），根据关系 $\overline \beta _ i = N \beta _ i$，可知 $N=1000$ 。根据 (B4) 计算出扰动核为，

$$p_{0t}(\mathbf x(t)|\mathbf x(0))= \\\\
\mathcal N \left(\mathbf x(t); e ^ {-\frac 1 4 t ^ 2(\overline \beta _ {max} - \overline \beta _ {min}) - \frac 1 2 t \overline \beta _ {min}} \mathbf x(0), I-I e ^ {-\frac 1 2 t ^ 2 (\overline \beta _ {max} - \overline \beta _ {min}) - t \overline \beta _ {min}} \right) \tag{B5}$$

参考 (33) 式，将 $\overline \beta (s)$ 替换为 $\overline \beta _ {min} + t (\overline \beta _ {max} - \overline \beta _ {min})$ 可得到 (B5) 式。

VP SDE (DDPM) 不存在 VE SDE 中的 $t=0$ 处不连续的问题，但是在 $t=0$ 处，训练和采样均存在数值不稳定的问题，因为 $t \rightarrow 0$ 时 $\mathbf x(t)$ 的方差太小，以至于无法表示（数值下溢），所以也限制范围为 $t \in [\epsilon, 1]$，例如作者使用 $\epsilon = 10 ^ {-3}$ 。

### C.3 sub-VP SDE

sub-VP SDEs 使用与 VP SDEs 相同的 $\beta(t)$，那么根据 (36) 式扰动核为

$$p_{0t}(\mathbf x(t)|\mathbf x(0))= \\\\
\mathcal N \left(\mathbf x(t); e ^ {-\frac 1 4 t ^ 2(\overline \beta _ {max} - \overline \beta _ {min}) - \frac 1 2 t \overline \beta _ {min}} \mathbf x(0), [1- e ^ {-\frac 1 2 t ^ 2 (\overline \beta _ {max} - \overline \beta _ {min}) - t \overline \beta _ {min}} ] ^ 2 I \right) \tag{B6}$$

## D 概率流

### D.1 概率流 ODE 推导

考虑 (A1) 式 SDE，推导出概率流 ODE 如下（推导过程略，见原论文附录），

$$d\mathbf x = \tilde {\mathbf f}(\mathbf x, t) dt + \tilde {\mathbf G}(\mathbf x, t) d\mathbf w \tag{D1}$$

其中 $\tilde {\mathbf G}(\mathbf x, t) := \mathbf 0$ 且

$$\tilde {\mathbf f}:=\mathbf f(\mathbf x, t) - \frac 1 2 \nabla \cdot [\mathbf G(\mathbf x,t)\mathbf G(\mathbf x, t)^{\top}] - \frac 1 2 \mathbf G(\mathbf x,t)\mathbf G(\mathbf x, t)^{\top}\nabla_{\mathbf x} \log p_t(\mathbf x) \tag{D2}$$

也就是说

$$d \mathbf x = \tilde {\mathbf f}(\mathbf x, t) dt \tag{D3}$$

推导过程使用了 Fokker-Planck 方程。

### D.2 似然计算

在 D.1 一节推导概率流 ODE 过程中有（详细过程见论文附录）

$$\frac {\partial p _ t (\mathbf x)}{\partial t}=-\sum _ {i=1} ^ d \frac {\partial} {\partial x _ i} [\tilde f_i(\mathbf x, t) p _ t(\mathbf x)]$$

变换得

$$\frac {\partial [\log p _ t(\mathbf x)]}{\partial t}=-\nabla \cdot \tilde {\mathbf f}(\mathbf x(t), t) \tag{D4}$$

其中 $\nabla \cdot \tilde {\mathbf f}(\mathbf x(t), t) \stackrel{\Delta}= \sum _ {i=1} ^ d \frac {\partial} {\partial x _ i} \tilde f _ i(\mathbf x, t)$ 。


注意我们是从 SDE 出发进行讨论，所以变量 $\mathbf x$ 是从时刻 $0$ 变化到时刻 $T, \ T \le 1$（即，前向过程），所以解 (D4) 式，得

$$\log p _ 0 (\mathbf x (0)) = \log p _ T (\mathbf x(T)) + \int _ 0 ^ T \nabla \cdot \tilde {\mathbf f}(\mathbf x(t), t) dt$$

由于 $\tilde {\mathbf f}$ 中包含了 $p _ t(\mathbf x(t))$ ，所以上式难以计算，使用模型输出 $\mathbf s _ {\theta} (\mathbf x, t)$ 代替 (D2) 式中的 $\nabla _ {\mathbf x} \log p _ t (\mathbf x)$，于是上式变为

$$\log p _ 0 (\mathbf x (0)) = \log p _ T (\mathbf x(T)) + \int _ 0 ^ T \nabla \cdot \tilde {\mathbf f}_{\theta}(\mathbf x(t), t) dt \tag{D5}$$

其中

$$\tilde {\mathbf f} _ {\theta}:=\mathbf f(\mathbf x, t) - \frac 1 2 \nabla \cdot [\mathbf G(\mathbf x,t)\mathbf G(\mathbf x, t)^{\top}] - \frac 1 2 \mathbf G(\mathbf x,t)\mathbf G(\mathbf x, t)^{\top} \mathbf s _ {\theta} (\mathbf x, t) \tag{D2.1}$$

不过很多情况下 $\nabla \cdot \tilde {\mathbf f}_{\theta}(\mathbf x, t)$ 计算量很大（计算步骤： $d$ 次循环，每次循环进行一次反向梯度传播，设置 loss=$\tilde f _ {\theta} ^ i$，求得一个梯度向量，循环结束后得到一个矩阵，也就是向量 $\tilde {\mathbf f} _ {\theta}$ 对向量 $\mathbf x$ 的梯度矩阵，最后求矩阵对角线之和），所以作者使用 Skilling-Hutchinson trace estimator 对其进行估算，具体地，有

$$\nabla \cdot \tilde {\mathbf f}_{\theta}(\mathbf x, t) = \mathbb E _ {p(\epsilon)} [\epsilon ^ {\top} \nabla \tilde {\mathbf f} _ {\theta} (\mathbf x, \epsilon) \epsilon] \tag{D7}$$

其中 $\nabla \tilde {\mathbf f} _ {\theta}$ 指 $\tilde {\mathbf f} _ {\theta} (\cdot, t)$ 的 Jacobian（对 $\mathbf x$ 求导，向量对向量求导），$\epsilon$ 为随机变量且满足 $\mathbb E _ {p(\epsilon)}[\epsilon] = \mathbf 0, \ \text{Cov} _ {p(\epsilon)}[\epsilon] = I$。

$\epsilon ^ {\top} \nabla \tilde {\mathbf f} _ {\theta} (\mathbf x, t)$ 可以使用自动微分软件计算（例如 PyTorch），这只需要一次反向梯度传播计算。因此，我们采样一个 $\epsilon \sim p(\epsilon)$，然后根据 (D6) 式计算 $\nabla \cdot \tilde {\mathbf f}_{\theta}(\mathbf x, t)$， 由于 (D6) 式是无偏估计，所以可以进行足够多次的计算，然后取平均值。

作者实验中，使用 RK45 ODE solver 解 (D5) 式这个积分方程。

### D.3 概率流采样

假设前向过程 SDE 为 $d \mathbf x = \mathbf f(\mathbf x, t) dt + \mathbf G(t) d\mathbf w$，其离散化形式为

$$\mathbf x _ {i+1} = \mathbf x _ i + \mathbf f _ i (\mathbf x _ i ) + \mathbf G _ i \mathbf z _ i, \quad i = 0, 1, \ldots, N-1 \tag{D7}$$

$\Delta t$ 已经放入 $\mathbf f_i$ 和 $\mathbf G_i$ 中。

注意上式我们假设了扩散系数的形式为 $\mathbf G(t)$，而 $\mathbf G(t)$ 与 $\mathbf x$ 无关，$\tilde {\mathbf f}$ 表达式中第二项为 0， 所以概率流 ODE (D3) 式变为

$$d\mathbf x = \\{\mathbf f(\mathbf x,t) - \frac 1 2 \mathbf G(t) \mathbf G(t) ^ {\top} \nabla _ {\mathbf x} \log p _ t (\mathbf x)\\} dt \tag{D8}$$

离散化形式为（注意是前向，所以 $d\mathbf x$ 离散化为 $\mathbf x _ {i+1} - \mathbf x _ i$），

$$\mathbf x _ i = \mathbf x _ {i+1} - \mathbf f _ {i+1} (\mathbf x _ {i+1}) + \frac 1 2 \mathbf G _ {i+1} \mathbf G _ {i+1} ^ {\top} \mathbf s _ {\theta}(\mathbf x _ {i+1}, i+1), \quad i = 0 , 1, \ldots, N-1 \tag{D9}$$

(D9) 是一个确定性迭代规则，这与反向扩散采样（本文 (40) 式）/原始采样（DDPM 中推导的 $\mathbf x_{i-1}$ 表达式，本文 (12) 式）不同，(D9) 中没有添加随机噪声。

**#1 SMLD** 概率流采样

参考 2.4.1 一节的 VE SDE 内容，结合 (16) 式和 (D7) 式，可知 $\mathbf f _ i = \mathbf 0, \mathbf G _ i=\sqrt {\sigma _ i ^ 2 - \sigma _ {i-1} ^ 2} I$，代入 (D9) 式，得

$$\mathbf x _ i = \mathbf x _ {i+1} + \frac 1 2 (\sigma _ {i+1} ^ 2 - \sigma _ i ^ 2) \mathbf s _ {\theta} ^ {\star} (\mathbf x _ {i+1}, \sigma _ {i+1}), \quad i = 0,1,\ldots, N-1 \tag{D10}$$

与 3.1.1 一节内容相呼应。

**#2 DDPM** 概率流采样

参考 2.4.2 一节得 VP SDE 内容，结合 (20) 式和 (D7) 式，可知 $\mathbf f _ i = (\sqrt {1-\beta _ i}-1) \mathbf x _ i, \ \mathbf G _ i = \sqrt {\beta _ i }I$，注意这里将 i 范围统一为 $i=0,1,\ldots, N-1$，代入 (D9) 式，得

$$\mathbf x _ i = (2-\sqrt {1- \beta _ {i+1}}) \mathbf x _ {i+1} + \frac 1 2 \beta _ {i+1} \mathbf s _ {\theta} ^ {\star} (\mathbf x _ {i+1}, i+1) \tag{D11}$$

与 3.1.2 一节内容相呼应。

## E. 反向扩散采样

见本文 3.1 一节内容，反向扩散采样过程为 (40) 式。DDPM 的原始采样与反向扩散采样在 $\beta_i \rightarrow 0$ 时是一致的，而 $\Delta t \rightarrow 0$ 时，$\beta _ i = \overline \beta _ i \Delta t \rightarrow 0$ 成立。见 (41.1) 式和 (41.2) 式。

## F. SMLD 原始采样

DDPM 的原始采样为 (12) 式，反向扩散采样为 算法 2 中的 Predictor 蓝色部分，概率流采样为 (D11) 式。

SMLD 反向扩散采样为 算法 1 中的 Predictor 蓝色部分，概率流采样为 (D10) 式，下面给出其原始采样。

（注：SMLD 原论文中使用 Langevin dynamics 采样，而本文将 Langevin dynamics 作为 SMLD 和 DDPM 的 Corrector 部分，见算法 1、2 橙色部分）

首先给出 SMLD 的前向马尔科夫链，

$$p(\mathbf x _ i | \mathbf x _ {i-1}) = \mathcal N (\mathbf x _ i; \mathbf x _ {i-1}, (\sigma _ i ^ 2 - \sigma _ {i-1} ^ 2) I), \quad i = 1, 2,\ldots ,N$$

边缘分布 $p(\mathbf x _ i|\mathbf x _ 0)=\mathcal N(\mathbf x _ 0; (\sigma _ i ^ 2 - \sigma _ 0 ^ 2)I)$，$p(\mathbf x _ {i-1}|\mathbf x _ 0)=\mathcal N(\mathbf x _ 0; (\sigma _ {i-1} ^ 2 - \sigma _ 0 ^ 2)I)$

根据高斯分布的条件分布关系有，

$$p(x _ 1) = \mathcal N(\mu _ 1, s _ 1 ^2), \quad p(x _ 2) = \mathcal N(\mu _ 2, s _ 2 ^2)
\\\\ p(x _ 1 | x _ 2)=\mathcal N(\mu _ 1 + \rho s _ 1 / s _ 2 (x _ 2 - \mu _ 2), (1-\rho ^ 2) s _ 1 ^ 2)
\\\\ p(x _ 2 | x _ 1)=\mathcal N(\mu _ 2 + \rho s _ 2 / s _ 1 (x _ 1 - \mu _ 1), (1-\rho ^ 2) s _ 2 ^ 2)$$

将 $\mathbf x _ {i-1}, \ \mathbf x _ i$ 分别看作 $x _ 1, x _ 2$，那么有以下等式关系，

$$\begin{aligned} \mu _ 1 = \mu _ 2 = \mathbf x _ 0
\\\\ s _ 1 ^ 2 = (\sigma _ {i-1} ^ 2 - \sigma _ 0 ^ 2), \ s _ 2 ^ 2 = (\sigma _ i ^ 2 - \sigma _ 0 ^ 2)
\\\\ \mathbf x _ 0 + \rho s _ 2 / s _ 1 (\mathbf x _ {i-1} - \mathbf x _ 0) = \mathbf x _ {i-1} 
\\\\ (1-\rho ^ 2) s _ 2 ^ 2 = (\sigma _ i ^ 2 - \sigma _ {i-1} ^ 2) = s _ 2 ^ 2 - s _ 1 ^ 2
\end{aligned}$$

根据以上关系可得 

$$\rho ^ 2 = \frac {\sigma _ {i-1} ^ 2 - \sigma _ 0 ^ 2}{\sigma _ i ^ 2 - \sigma _ 0 ^ 2} = \frac {s _ 1 ^ 2}{s _ 2 ^ 2} \\\\
\mu _ 1 + \rho s _ 1 / s _ 2 (x _ 2 - \mu _ 2)=\mathbf x _ 0 + \frac {\sigma _ {i-1} ^ 2 - \sigma _ 0 ^ 2}{\sigma _ i ^ 2 - \sigma _ 0 ^ 2}(\mathbf x _ i - \mathbf x _ 0) \\\\
(1-\rho ^ 2) s _ 1 ^ 2 = \frac { s _ 2 ^ 2 - s _ 1 ^ 2}{s _ 2  ^ 2} s _ 1 ^ 2= \frac {\sigma _ {i-1} ^ 2 - \sigma _ 0 ^ 2}{\sigma _ i ^ 2 - \sigma _ 0 ^ 2} (\sigma _ i ^ 2 - \sigma _ {i-1} ^ 2)$$

由于 $\sigma _ 0 = 0$，所以可以得到后验分布为

$$q(\mathbf x _ {i-1} | \mathbf x _ i, \mathbf x _ 0) = \mathcal N \left(\frac {\sigma _ {i-1} ^ 2}{\sigma _ i ^ 2} \mathbf x _ i + (1-\frac {\sigma _ {i-1} ^ 2}{\sigma _ i ^ 2}) \mathbf x _ 0, \ \frac {\sigma _ {i-1} ^ 2}{\sigma _ i ^ 2}(\sigma _ i ^ 2 - \sigma _ {i-1} ^ 2) I \right) \tag{F1}$$

记反向转移过程为 $p _ {\theta} (\mathbf x _ {i-1} | \mathbf x _ i)=\mathcal N(\mathbf x _ {i-1}; \mu _ {\theta} (\mathbf x _ i, i), \tau _ i ^2 I)$，显然需要 $p_{\theta} (\mathbf x _ {i-1} | \mathbf x _ i)$ 尽量逼近 $q(\mathbf x _ {i-1} | \mathbf x _ i, \mathbf x _ 0)$ ，根据前向过程 $\mathbf x _ i = \mathbf x _ 0 + \sigma _ i \mathbf z$，其中 $\mathbf z$ 为随机噪声，可得 $\mathbf x _ 0 = \mathbf x _ i - \sigma _ i \mathbf z$ 代入 (F1) 式，并用 $\mathbf s _ {\theta} (\mathbf x _ i, i)$ 估计 $\mathbf z/\sigma _ i$ 得，

$$\begin{aligned} \mu _ {\theta}(\mathbf x _ i, i) &= \frac {\sigma _ {i-1} ^ 2}{\sigma _ i ^ 2} \mathbf x _ i + (1-\frac {\sigma _ {i-1} ^ 2}{\sigma _ i ^ 2}) (\mathbf x _ i - \sigma _ i ^ 2 \mathbf s _ {\theta} (\mathbf x _ i, i))=\mathbf x _ i + (\sigma _ i ^ 2 - \sigma _ {i-1} ^ 2 ) \mathbf s _ {\theta} (\mathbf x _ i, i) \\\\
\tau_i ^ 2 &= \frac {\sigma _ {i-1} ^ 2}{\sigma _ i ^ 2}(\sigma _ i ^ 2 - \sigma _ {i-1} ^ 2)
\end{aligned} \tag{F2}$$

故 SMLD 的原始采样为

$$\mathbf x _ {i-1} = \mathbf x _ i + (\sigma _ i ^ 2 - \sigma _ {i-1} ^ 2)\mathbf s _ {\theta} ^ {\star} (\mathbf x _ i, i) + \sqrt {\frac {\sigma _ {i-1} ^ 2}{\sigma _ i ^ 2}(\sigma _ i ^ 2 - \sigma _ {i-1} ^ 2)} \mathbf z _ i, \quad i = 1,2,\ldots, N \tag{F3}$$

其中 $\mathbf x _N \sim \mathcal N (\mathbf 0, \sigma _ N ^ 2 I), \ \mathbf z _ i \sim \mathcal N (\mathbf 0, I)$ 。

$\mathbf z_i$ 是采样过程中使用的随机变量，而 $\mathbf z$ 是前向过程中使用的随机变量，模型输出 $\mathbf s _ {\theta} (\mathbf x _i, i)$ 用于估计 target $\mathbf z/\sigma _ i$ ，而 DDPM 的 target 为 $\mathbf z$ ，这里 SMLD 只是为了将 (F2) 表达形式与 SMLD 的 reverse-time SDE 采样类似（见算法 1 蓝色部分），理论上，将模型输出 $\mathbf s _ {\theta} (\mathbf x _i, i)$ 估计 target $\mathbf z$ 也是可以的，只要 (F2) 式对应的修改一下即可。

## G. PC Samplings

PC 采样过程如上面算法 1、 2 所示。

**# Denoising**

采样结束后，继续执行一个 denoising 步骤，使用 Tweedie 公式，如下所示

$$E[\mu | z] = z + \sigma ^ 2 l'(z) \tag{G1}$$

其中 $l'(z) = \frac d {dz} \log f(z)$ 。

实际上，这相当于又做了一次不加噪的 Predictor 操作，(G1) 式对比算法 1， $l'(z)$ 表示得分函数，使用 $\mathbf s _ {\theta}$ 代替，$z$ 相当于 $\mathbf x(0 ^ +)$，$\sigma ^ 2$ 相当于 $\sigma (0 ^ +) ^ 2$。

源码中，概率流采样的最后，就执行了这个 denoising 步骤，而 PC 采样的源码，将循环体中的 corrector 和 predictor 顺序调换，所以采样的最后没有额外使用 denoising 步骤，而是将最后一轮循环中的 predictor 不添加噪声。

## I. 可控生成

考虑前向 SDE： $d \mathbf x = \mathbf f (\mathbf x, t) dt + \mathbf G(\mathbf x, t) d \mathbf w$，并假设初始分布为 $p _ 0 (\mathbf x(0) | \mathbf y)$，在 t 时刻分布为 $p _ t (\mathbf x (t) | \mathbf y)$，即，基于 $\mathbf y$ 的条件分布。根据 Anderson (1982) 的论文研究，反向 SDE 如下，

$$d \mathbf x = \\{\mathbf f(\mathbf x,t) - \nabla \cdot [\mathbf G(\mathbf x, t) \mathbf G(\mathbf x, t) ^ {\top}] - \mathbf G(\mathbf x, t) \mathbf G(\mathbf x, t) ^ {\top} \nabla _ {\mathbf x} \log p_t (\mathbf x |\mathbf y)\\} dt + \mathbf G(\mathbf x, t) d \overline {\mathbf w} \tag{I1}$$

根据贝叶斯定理可知 $p _ t (\mathbf x|\mathbf y) \propto p _ t (\mathbf x) p(\mathbf y | \mathbf x)$，那么得分函数计算如下，

$$\nabla _ {\mathbf x} \log p _ t (\mathbf x|\mathbf y) = \nabla _ {\mathbf x} p _ t (\mathbf x) + \nabla _ {\mathbf x} \log p(\mathbf y | \mathbf x) \tag{I2}$$

前面所讨论的采样方法均可应用于条件型反向 SDE 的样本生成。

### I.1 分类条件的采样

我们不仅可以从 $p _ 0$ 分布中采样，还可以从 $p _ 0(\mathbf x (0)|\mathbf y)$ 中采样，条件是 $p _ t (\mathbf y|\mathbf x (t))$ 已知。给定前向 SDE 如上文 (13) 式所示，要从 $p _ t (\mathbf x (t)|y)$ 中采样，需要先从 $p _ T (\mathbf x (T)|\mathbf y)$ 开始，然后求解以下条件型反向 SDE，

$$d \mathbf x = \\{\mathbf f(\mathbf x, t) - g(t) ^ 2 [\nabla _ {\mathbf x} \log p _ t(\mathbf x) + \nabla _ {\mathbf x} \log p _ t (\mathbf y|\mathbf x)] \\} dt + g(t) d \overline {\mathbf w} \tag{I3}$$

以上 (I3) 式根据 (I1) 和 (I2) 式简化而得。(I3) 式中，$\nabla _ {\mathbf x} \log p _ t (\mathbf x)$ 可以使用模型的得分 $\mathbf s _ {\theta}(\mathbf x (t), t)$ 来估计，而 $\nabla _ {\mathbf x} \log p _ t (\mathbf y|\mathbf x)$ 则可以训练一个分类器，分类器输入则是添加了噪声的数据 $\mathbf x(t)$，计算分类器输出对输入的梯度即可。

将 (I3) 式与 (14) 式比较，发现 class-conditional 采样与之前的区别在于: $\nabla _ {\mathbf x} \log p _ t(\mathbf x) \rightarrow \nabla _ {\mathbf x} \log p _ t(\mathbf x) + \nabla _ {\mathbf x} \log p _ t (\mathbf y|\mathbf x)$。

$\mathbf y$ 表示分类 label。

训练一个分类器 $p _ t (\mathbf y | \mathbf x(t))$，用于条件型采样。

首先从数据集中采样得到 $(\mathbf x (0), \mathbf y)$，然后加噪生成 $(\mathbf x (t), \mathbf y)$ 作为训练分类器的训练数据，使用跨时刻的交叉熵损失，类似于 (15) 式，模型输出 $\overline {\mathbf y} _ {\theta} (\mathbf x, t)$，target 为 $\mathbf y$。


