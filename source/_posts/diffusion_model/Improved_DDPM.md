---
title: Improved Denoising Diffusion Probabilistic Models
date: 2022-07-08 11:58:42
tags: diffusion model
mathjax: true
---

论文：[Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)

# 1. DDPM


## 1.1 定义

给定一个数据分布 $x_0 \sim q(x_0)$，前向加噪过程生成 $x_1, \ldots, x_T$，噪声类型为高斯噪声，方差为 $\beta_t \in (0,1)$，

$$\begin{aligned}q(x_1,\ldots, x_T |x_0)=\prod_{t=1}^T q(x_t|x_{t-1})
\\ q(x_t|x_{t-1})=\mathcal N(x_t; \sqrt {1-\beta_t} x_{t-1}, \beta_t I) \end{aligned}\tag{1}$$


如果 $T$ 足够大，且 $\beta_t$ 值选择恰当，那么 $x_T$ 接近各项同性的高斯分布。如果能得到准确的反向分布 $q(x_{t-1}|x_t)$，那么采样 $x_T \sim \mathcal N(0, I)$ 根据反向过程可以得到来自 $q(x_0)$ 的样本。

反向过程的分布 $q(x_{t-1}|x_t)$ 根据贝叶斯定理，可以计算得到结果如下

$$q( x_{t-1}| x_t,  x_0)=\mathcal N( x_{t-1}; \tilde {\mu}_t( x_t,  x_0), \tilde {\beta}_t I) \tag {2}$$

其中 

$$\tilde {\mu} _t( x_t,  x_0) = \frac {\sqrt {\overline \alpha_{t-1}}\beta_t}{1-\overline \alpha_t} x_0 + \frac {\sqrt {\alpha_t}(1-\overline \alpha_{t-1})}{1-\overline \alpha_t}  x_t, \quad \tilde \beta_t = \frac {1-\overline \alpha_{t-1}}{1-\overline \alpha_t} \beta_t \tag{3}$$

显然 $\tilde u_t( x_t,  x_0)$ 的值依赖于 $ x_0$，这在反向过程中是未知的。使用神经网络来近似，

$$p_{\theta}( x_{t-1}| x_t)=\mathcal N(x_{t-1}; \mu_{\theta}(x_t,t),\Sigma_{\theta}(x_t, t))$$

神经网络的输入是 $(x_t, t)$，输出是高斯分布的期望和协方差 $\mu_{\theta}, \Sigma_{\theta}$ 。

训练目标是最小化 $\mathbb E_{q(x_0)}[-\log p(x_0)]$，变换后得到

$$\begin{aligned}L_{vlb}&=L_0+L_1+\cdots +L_T
\\\\ L_0&=-\log p_{\theta}(x_0|x_1)
\\\\ L_{t-1}&= D_{KL}(q(x_{t-1}|x_t, x_0)||p_{\theta}(x_{t-1}|x_t))
\\\\ L_T &= D_{KL}(q(x_T|x_0)||p(x_T))
\end{aligned} \tag{4}$$


$L_{t-1}, t=2,\ldots, T-1$ 和 $L_T$ 均为 KL 散度，可以计算出闭式解。$L_T$ 与模型参数 $\theta$ 无关，对 $L_0$ 的最小化处理则是将每个颜色通道分为 256 个 bin，具体参考 [DDPM](/diffusion_model/2022/06/27/ddpm) 。

## 1.2 训练

直接使用神经网络来预测 $\mu_{\theta}(x_t, t)$，神经网络的输出是 $\epsilon_{\theta}(x_t, t)$，那么反向过程 $p_{\theta}(x_{t-1}|x_t)$ 分布的参数为 

$$\mu_{\theta}(\mathbf x_t, t)=\frac 1 {\sqrt {\alpha_t}} (\mathbf x_t-\frac {\beta_t}{\sqrt {1-\overline \alpha_t}} \epsilon_{\theta}(\mathbf x_t, t)) \tag{5}$$

本节内容回顾了 Diffusion model 过程。


# 2. 改善对数似然

DDPM 并不能取得较好的 对数似然。本节探索部分算法上的修改，以便取得更好的对数似然。为了研究不同修改的效果，作者固定模型框架，模型参数也固定，数据集使用 ImageNet `64x64`。

## 2.1 学习反向过程的方差

DDPM 中作者设置 $\Sigma_{\theta}(x_t, t)=\sigma_t^2 I$，其中 $\sigma_t^2$ 使用 $\beta_t$ 或者 $\tilde \beta_t$（两者结果相近），而 $\beta_t$ 和 $\tilde \beta_t$ 分别表示方差的上限和下限，所以为何两个极端的 $\sigma_t^2$ 选择其结果相近？

如图 1，计算 $\beta_t$ 和 $\tilde \beta_t$ 的值，除了在 $t=0$ 附近，其他时候 $\beta_t$ 和 $\tilde \beta_t$ 值几乎相等。这表明模型处理极小的细节改变。如果增大 扩散 steps $T$，$\beta_t$ 和 $\tilde \beta_t$ 在多数时候依然保持相近，所以 $\sigma_t$ 不太会影响采样质量，而采样质量主要由 $\mu_{\theta}(x_t,t)$ 决定。

![](/images/diffusion_model/improved_ddpm_1.png)

图 1. $\tilde \beta_t / \beta_t$ 随 step 的关系图


虽然对于采样质量而言，固定 $\sigma_t$ 的值是一个不错的选择，但是对于 对数似然 则未必。如图 2，开始的几个扩散 step 贡献了大部分负对数似然（即损失 $L_{vlb}$）。由此可见，使用更好的 $\Sigma_{\theta}(x_t, t)$ 可以改善对数似然，而不会出现图 2 中那样的不稳定现象。

![](/images/diffusion_model/improved_ddpm_2.png)

<center>图 2. VLB vs step</center>


从 图 1 中可见，合适的 $\Sigma_{\theta}(x_t, t)$ 区间比较小（step 为 0 的附近），故直接使用神经网络预测 $\Sigma_{\theta}(x_t, t)$ 显得不太容易。作者选择在 log 空间对 $\beta_t$ 和 $\tilde \beta_t$ 进行插值作为方差。具体地，模型输出一个向量 $v$ （维度与 $x_t$ 相同），然后计算出协方差

$$\Sigma_{\theta}(x_t, t)=\exp (v \log \beta_t + (1-v) \log \tilde \beta_t) \tag{6}$$

由于是在 log 空间，所以对 $v$ 不设其他约束条件，这也导致模型预测的协方差可能会超出插值范围，但是作者实践中发现并没有出现这种现象。

对优化目标进行修改

$$L_{hybrid}=L_{simple}+ \lambda L_{vlb} \tag{7}$$

这是因为 $L_{simple}$ 与 $\Sigma_{\theta}(x_t, t)$ 无关。作者实验中设置 $\lambda=0.001$ 避免 $L_{vlb}$ 作用盖过 $L_{simple}$ 。

## 2.2 改善噪声机制

DDPM 中使用线性噪声 （$\beta_1=10^{-4}$ 线性增大到 $\beta_T=0.02$），这在高分辨率图像中效果很好，但是在诸如 $64 \times 64$ 或者 $32 \times 32$ 图像中效果次佳，因为线性加噪后的图像中噪声太强，如图 3，上一行是线性加噪过程，下一行是 cosine 加噪过程，显然最后的几个 step 中，上一行几乎是纯噪声，而下一行则是缓慢的加噪。

![](/images/diffusion_model/improved_ddpm_3.png)
<center>图 3. </center>

cosine 噪声机制如下，

$$\overline \alpha_t = \frac {f(t)} {f(0)}, \quad f(t)=\cos \left(\frac {t/T+s}{1+s} \cdot \frac {\pi} 2\right)^2$$

根据 DDPM 中 $\alpha_t$ 与 $\beta_t$ 的关系有

$$\beta_t = 1- \frac {\overline \alpha_t} {\overline \alpha_{t-1}}$$

如图 4，是线性和 cosine 两种噪声机制中 $\overline \alpha_t$ 的对比，cosine 机制中，在 diffusion step 的中间段，$\overline \alpha_t$ 值近似线性下降，在两端 $t=0, \ t=T$ 则缓慢变化，而线性机制中，$\overline \alpha_t$ 快速下降接近 0 值，使得图像信息过快地被损坏。

![](/images/diffusion_model/improved_ddpm_4.png)

图 4. $\overline \alpha_t$ 随 step  $t$ 的变化关系

作者发现需要一个小的 offset $s$ 值，以防 $\beta_t$ 在 $t=0$ 附近太小而导致开始时网络难以准确地预测 $\epsilon$。作者选择 $s=0.008$ 使得 $\sqrt {\beta_0}$ 稍微小于 pixel bin size $1/127.5$（将 $[0,255]$ 压缩至 $[-1,1]$，故 bin size 为 $2/255$）。

## 2.3 降低梯度噪声

我们期望直接通过优化 $L_{vlb}$ 以获得最佳对数似然，而不是优化 $L_{hybrid}$，但是 $L_{vlb}$ 实际上较难优化。作者寻找一种降低 $L_{vlb}$ 方差地方法来对 对数似然 进行优化。

$$L_{vlb}=E_{t \sim p_t} \left[ \frac {L_t}{p_t} \right]$$

$$p_t \approx \sqrt {E[L_t^2]}, \quad \sum p_t = 1$$

训练开始时，根据均匀分布对 $t$ 采样，知道每个 $t \in [0, T-1]$ 均有 10 个样本值，计算损失 $L_t$，然后开始计算 $p_t = \sqrt {E[L_t^2]}$ 并根据  $\sum p_t = 1$ 进行归一化，然后根据修正过地 $p_t$ 对 $t$ 进行采样，计算 $L_t$，注意从这里开始，一直需要动态地更新 $p_t$，每个 $t$ 值均维护最新的 10 个 $L_t$ 值用于更新 $p_t$ 。如图 5 中的 resampled 曲线。

![](/images/diffusion_model/improved_ddpm_5.png)
<center>图 5. ImageNet 64x64 上的学习曲线</center>

# 3. 改善采样速率

作者使用 $T=4000$ 个扩散 steps，导致生成一个样本都要耗费若干分钟。本节研究如果降低 steps，那么性能会如何，实验结果发现预训练的 $L_{hybrid}$ 模型，在降低 steps 后生成的样本图像质量依然很高。降低 steps 使得样本生成事件从几分钟降至几秒。

模型训练过程中通常使用 $t$ 的相同序列值 $(1,2,\ldots, T)$ 进行采样，当然也可以使用子序列 $S$ 进行采样。为了将 steps 从 $T$ 降到 $K$，使用 $1 \sim T$ 之间的 $K$ 个等间隔的数（四舍五入取整），这 $K$ 个数构成序列 $S=(S_1,\ldots, S_K)$，从原来的 $\overline \alpha_1, \ldots \overline \alpha_T$ 中取 $\overline \alpha_{S_t}, \ldots, \overline \alpha_{S_K}$，然后计算

$$\beta_{S_t}=1-\frac {\overline \alpha_{S_t}}{\overline \alpha_{S_{t-1}}}, \quad \tilde \beta_{S_t}=\frac {1-\overline \alpha_{S_{t-1}}}{1-\overline \alpha_{S_t}} \beta_{S_t}$$