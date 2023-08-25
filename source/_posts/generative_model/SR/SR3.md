---
title: Image Super-Resolution via Iterative Refinement
date: 2023-07-29 16:14:16
tags: super resolution
mathjax: true
---

论文：[Image Super-Resolution via Iterative Refinement](https://arxiv.org/abs/2104.07636)

# 1. 简介

作者提出 SR3（Super Resolution via Repeated Refinement），用于图像超分辨率任务。SR3 使用 DDPM（降噪扩散概率模型），这是一种有条件的图像生成方法，生成图像时，从一个高斯噪声开始，逐步微调带噪输出，微调过程使用了 U-Net。

SR3 在很多个放大倍数上均表现良好，SR3 也可以级联，例如从 `64x64` 通过 SR3 放大到 `256x256`，然后再通过 SR3 放大到 `1024x1024`，级联的生成效率更高，因为直接一步到位生成大 size 图像，需要更多的 steps 。

一些自动计算图像质量得分的指标例如 PSNR 和 SSIM 均不能反映人类对图像质量好坏的判断。作者采用了 2AFC：给人一个低分辨率图像，然后从模型输出和 gt image 两者中选择一个，即，让人判断哪个是 gt ，以此判断模型以假乱真的能力（模型的 fool rate）。作者实验表明，SR3 的 fool rate 接近 50% 。


# 2. 有条件的扩散模型

数据集为 `input-out` 图像对，记为 $\mathcal D = \lbrace x _ i, y _ i \rbrace _ {i=1} ^ N$ ，概率模型为 $p(y|x)$ ，即根据低分辨率图像生成高分辨率图像。借助 DDPM（降噪扩散模型）实现图像生成。

对于 DDPM，从一个高斯噪声开始 $y _ T \sim \mathcal N(0, I)$，逐步移除噪声得到 $(y _ {T-1}, \ldots, y _ 0)$，于是目标变为学习条件转移分布 $p _ {\theta}(y _ {t-1} | y _ t, x)$，使得 $y _ 0 \sim p (y|x)$ 。

## 2.1 高斯扩散模型

对一个高分辨率图像 $y _ 0$ 逐步添加噪声，即前向过程，

$$\begin{aligned} q(y _ {1:T} | y _ 0) &= \prod _ {t=1} ^ T q (y _ t | y _ {t-1})
\\\\ q(y _ t | y _ {t-1}) &= \mathcal N(y _ t | \sqrt {\alpha _ t} y _ {t-1}, (1-\alpha _ t) I)
\end{aligned} \tag{1}$$

这是一个马尔可夫过程。$0 < \alpha _ t < 1$ 是超参数。根据 (1) 式，方差满足

$$V(y _ t) = \alpha _ t V (y _ {t-1}) + (1-\alpha _ t) \tag{2}$$

当 $V ( y _ {t-1}) =1$ 时，$V(y _ t) =1$ 。

根据推导有

$$q(y _ t | y _ 0) = \mathcal N(y _ t|\sqrt {\gamma _ t} y _ 0, (1-\gamma _ t) I ) \tag{3}$$

其中 $\gamma _ t = \prod _ {i=1} ^ t \alpha _ i$，在其他论文中也写作 $\overline \alpha _ t$ 。

后验分布为

$$\begin{aligned}q ( y _ {t-1}| y _ 0, y _ t) &= \mathcal N(y _ {t-1} | \mu, \sigma ^ 2 I)
\\\\ \mu &= \frac {\sqrt {\gamma _ {t-1}}(1-\alpha _ t)}{1-\gamma _ t} y _ 0 + \frac {\sqrt {\alpha _ t} (1-\gamma _ {t-1})}{1-\gamma _ t} y _ t
\\\\ \sigma ^ 2 &= \frac {(1-\gamma _ {t-1})(1-\alpha _ t)} {1-\gamma _ t}
\end{aligned} \tag{4}$$

根据高斯型边缘分布 (3) 式和条件分布 (1) 式，可以计算后验分布，即 (4) 式 。

## 2.2 优化去噪模型

扩散模型的反向过程就是去噪并生成图像。对于图像超分辨率任务，那么去噪模型 $f _ {\theta}$ 有三个输入：

1. 低分辨率图像 $x$

2. 带噪图像 $\tilde y$

    训练阶段，$f _ {\theta}$ 的输入 $\tilde y$ 为添加了噪声后的图像，inference 阶段，$\tilde y$ 是反向过程中上一步的输出（初始时为纯高斯噪声）。

    $\tilde y$ 应该要与前向过程中加噪后的图像一致，即，在 timestep `t`，反向过程的输入应该接近前向过程的输出。前向过程也称扩散过程，过程如下

    $$\tilde y = \sqrt {\gamma} y _ 0 + \sqrt {1-\gamma} \epsilon, \quad \epsilon \sim \mathcal N(0,I) \tag{5}$$


3. $\gamma$，这是噪声的充分统计量。 $\gamma$ 作为输入，其实相当于需要 timestep $t$ 作为输入

去噪模型 $f _ {\theta} (x, \tilde y, \gamma)$ 的输出为对噪声 $\epsilon$ 的预测，记这个预测为 $\epsilon _ {\theta}$，那么根据 (5) 式，就可以得到 $y _ 0$ 的预测，记为 $\hat y _ 0$，然后代入 (4) 式，得到 $\tilde y _ {t-1}$ 的一个采样 。

训练模型的目标函数为

$$\mathbb E _ {x, y} \mathbb E _ {\epsilon, \gamma} ||f _ {\theta} (x, \sqrt {\gamma} y _ 0 + \sqrt {1-\gamma} \epsilon, \gamma ) - \epsilon|| _ p ^ p \tag{6}$$

## 2.3 反向逐步微调

inference 就是反向过程，从一个高斯噪声 $y _ T$ 开始，

$$\begin{aligned} p _ {\theta}(y _ {0:T}|x) &= p(y _ T) \prod _ {t=1} ^ T p _ {\theta} (y _ {t-1}|y _ t, x)
\\\\ p (y _ T) &= \mathcal N(y _ T | 0, I)
\\\\ p _ {\theta}(y _ {t-1}|y _ t, x) &= \mathcal N(y _ {t-1} | \mu _ {\theta}(x, y _ t, \gamma _ t), \sigma _ t ^ 2 I)
\end{aligned} \tag{7}$$

模型 $f _ {\theta} (x, y _ t, \gamma _ t)$ 的输出是对 $\epsilon$ 的预测，那么根据 (5) 式，可以预测 $y _ 0$ 为

$$\hat y _ 0 = \frac 1 {\sqrt \gamma _ t} (y _ t - \sqrt {1-\gamma _ t} f _ {\theta} (x, y _ t, \gamma _ t)) \tag{8}$$

然后根据 (4) 式，得到后验分布的期望，

$$\mu _ {\theta}(x, y _ t, \gamma _ t) = \frac 1 {\sqrt {\alpha _ t}} \left ( y _ t - \frac {1-\alpha _ t}{\sqrt {1 - \gamma _ t}} f _ {\theta} (x, y _ t, \gamma _ t) \right ) \tag{9}$$

$p _ {\theta}(y _ {t-1}|y _ t, x)$ 的方差使用 $1-\alpha _ t$ ，没有使用后验分布的方差 $\sigma ^ 2$，参见 (4) 式，这是因为两者相差不大，于是反向过程生成的图像为


$$y _ {t-1} \leftarrow  \frac 1 {\sqrt {\alpha _ t}} \left ( y _ t - \frac {1-\alpha _ t}{\sqrt {1 - \gamma _ t}} f _ {\theta} (x, y _ t, \gamma _ t) \right ) + \sqrt {1-\alpha _ t} z _ t \tag{10}$$

其中 $z _ t \sim \mathcal N(0, I)$ 。

## 2.4 SR3 模型架构

SR3 模型架构在 DDPM 中使用的 U-Net 基础上做了一些修改：

1. DDPM 的 U-Net 的 residual block 替换为 BigGAN 中的 residual block
2. 将 skip connection 调整比例为 $1/\sqrt 2$ 
3. 增加 residual block 数量

为了让模型使用条件 $x$，使用 bicubic 插值将 $x$ 上采样到与 $y$ 相同 size，上采样之后的结果再与 $y _ t$ 沿着 channel 连接起来（concatenate）。

作者研究了使用 FiLM 那样的方式利用条件 $x$，但是发现生成的图像质量与上面这种简单的 concatenate 方法的生成质量差不多。

### 2.4.1 模型结构

不同任务的模型使用不同的结构，如表 1 所示

|Task|Channel Dim| Depth Multipliers| ResNet Blocks|Parameters|
|--|--|--|--|--|
|$16 \times 16 \rightarrow 128 \times 128$ | 128 | {1,2,4,8,8}|3|550M|
|$64 \times 64 \rightarrow 256 \times 256$ |128|{1,2,4,4,8,8} | 3| 625M|
|$64 \times 64 \rightarrow 512 \times 512$|64|{1,2,4,8,8,16,16}|3|625M|
|$256 \times 256 \rightarrow 1024 \times 1024$|16|{1,2,4,8,32,32,32}|2|150M|

表 1. 不同任务的模型结构参数

图 1 是 $16 \times 16 \rightarrow 128 \times 128$ 任务的模型结构图，

![](/images/generative_model/SR3_1.png)
<center>图 1</center>

输入低分辨率图像 $x$ 先插值，得到绿色部分，然后与带噪图像 $y _ t$ 沿 channel 进行 concatenate 。