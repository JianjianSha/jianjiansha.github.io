---
title: Noise scheduling for diffusion models
date: 2023-06-13 17:18:34
tags: diffusion model
mathjax: true
---

论文：[On the Importance of Noise Scheduling for Diffusion Models](https://arxiv.org/abs/2301.10972)

本文作者研究了 diffusion models 中的 noise scheduling，发现：

1. noise scheduling 对性能至关重要，不同的 image size 其最佳 noise scheduling 也不同

2. 增大 image size， 最佳 noise scheduling 趋于更加 noiser

    这是因为大 size 的 image，其冗余像素较多，能提供足够的信息，为了匹配信噪比，那么噪声也应该更加 noiser

3. 简单地使用一个 factor $b$ 对输入数据 scaling，同时固定 noise scheduling 函数不变，那么相当于 shifting log SNR（信噪比对数），平移距离为 $\log b$ 。

# 1. noise scheduling 为何重要

扩散模型中 $x _ t = \sqrt {\gamma (t)} x _ 0 + \sqrt {1-\gamma (t)} \epsilon$（例如 DDPM 中 $x _ t \sim \mathcal N(\sqrt {\overline \alpha _ t} x _ 0, (1-\overline \alpha _ t) I)$，根据图1，当增大 image size，在相同的 noise level（图1 中 $\gamma=0.7$）下，denoising task （这个任务可以理解为训练模型，从而基于模型可以采样来自真实数据分布的样本）变得更加简单，这是因为高分辨率图像有更多的冗余像素信息（nearby），这说明，高分辨率图像的最佳 noise schedule 不一定是低分辨率图像的最佳 noise schedule。

![](/images/diffusion_model/noise_scheduling_1.png)

<center>图 1</center>

# 2. 调整 noise scheduling 策略

## 2.1 策略1：改变 noise schedule 函数


### 2.1.1 余弦 schedule

源自 [Improved-DDPM](/2022/07/08/diffusion_model/Improved_DDPM)

$$\overline \alpha _ t = \frac {f(t)} {f(0)}, \quad f(t) = \cos \left(\frac {t/T + s}{1+s} \cdot \frac {\pi} 2 \right)^2 \tag{1}$$

其中 $s=0.008$，$t=1,\ldots, T$ 。

本文中，作者直接令 $s=0$，timesteps $t \in [0,1]$，那么

$$\gamma(t)=\overline \alpha _ t = \cos \left(t \cdot \frac {\pi} 2 \right)^2 \tag{2}$$

$\gamma(t) \in [0,1]$ 在 $t\in [0,1]$ 范围内是递减函数。

可将 (2) 式作适当的泛化，即，将 $t$ 映射到范围

$$t':=t (t _ 2 - t _ 1) + t _ 1 \tag{3}$$

其中 $t_1 < t _ 2$ 。

将 $\gamma(t)$ 线性映射，

$$\gamma'(t'):=\frac {\gamma(t _ 2) - \gamma(t')} {\gamma(t _ 2) - \gamma(t _ 1)}\tag{4}$$

其中 $\gamma (t')$ 定义为 (2) 式。(2) 式 $\gamma(t)$ 可以看作是 (4) 式的 kernel。

### 2.1.2 sigmoid schedule

基于 (3) (4) 两式，选用 sigmoid 函数作为 kernel，

$$\gamma(t') = \sigma (t'/\tau) \tag{5}$$



### 2.1.3 linear schedule

基于 (3) (4) 两式，选用 linear 函数作为 kernel，

$$\gamma(t') = 1 - t' \tag{6}$$

下文为了表述方便，仍使用 $\gamma$ 代替 $\gamma'$。

## 2.2 相关代码

```python
def simple_linear_schedule(t, clip_min=1e-9):
    # A gamma function that simply is 1-t.
    return np.clip(1 - t, clip_min, 1.)
def sigmoid_schedule(t, start=-3, end=3, tau=1.0, clip_min=1e-9):
    # A gamma function based on sigmoid function.
    v_start = sigmoid(start / tau)
    v_end = sigmoid(end / tau)
    output = sigmoid((t * (end - start) + start) / tau)
    output = (v_end - output) / (v_end - v_start)
    return np.clip(output, clip_min, 1.)
def cosine_schedule(t, start=0, end=1, tau=1, clip_min=1e-9):
    # A gamma function based on cosine function.
    v_start = math.cos(start * math.pi / 2) ** (2 * tau)
    v_end = math.cos(end * math.pi / 2) ** (2 * tau)
    output = math.cos((t * (end - start) + start) * math.pi / 2) ** (2 * tau)
    output = (v_end - output) / (v_end - v_start)
    return np.clip(output, clip_min, 1.)
```

图 2 展示了不同超参数下各个 noise schedule 函数，以及对应的 logSNR，这里 SNR 定义为

$$SNR = \gamma/(1-\gamma) \tag{7}$$

![](/images/diffusion_model/noise_scheduling_2.png)

<center>图 2</center>

## 2.3 策略 2：调整输入 scaling 因子

scale 输入 $x _ 0$，记 scaling 因子 $b$，那么扩散过程如下

$$x _ t = \sqrt {\gamma (t)} b x _ 0 + \sqrt {1- \gamma(t)} \epsilon \tag{8}$$

信噪比则为 $SNR=\gamma b ^ 2 / (1-\gamma)$ 。

当降低 $b$ 时，等价于增加 noise level，如图 3（$\gamma=0.7$），受噪声影响越大的图像变得更暗，源于方差也降低。

![](/images/diffusion_model/noise_scheduling_3.png)

<center>图 3</center>

根据 (8) 式，方差计算如下，

$$V[x _ t] = \gamma b ^ 2 V[x _ 0] + (1-\gamma) V[\epsilon]=\gamma b ^ 2 + 1- \gamma$$

上式推导中，$x _ 0 , \ \epsilon$ 相互独立，且 $V[\epsilon]=1$，$V[x _ 0]=1$，前者是由于 $\epsilon \sim \mathcal N(0, I)$，后者是因为输入图像可以通过归一化预处理达到输入数据的方差为 $1$。那么，为了保持 $x _ t$ 的方差固定不变，可以对 $x _ t$ 进行 scale，

$$x _ t := x _ t \cdot \frac 1 {\sqrt {(b ^ 2 -1) \gamma + 1}} \tag{9}$$

不过，作者说实际应用中，只需要简单的对 $x _ t$ 进行归一化处理就能获得很好的效果，归一化如下

$$x _ t = \frac {x _ t} {\sqrt {S [x _ t]}}, \quad S[x _ t]=\frac 1 d \sum _ i (x _ {ti} - \sum _ j x _ {tj})^2 \tag{10}$$

## 2.3.1 相关代码

训练阶段的代码

```python
def train_loss(x, gamma=lambda t: 1-t, scale=1, normalize=True):
    """Returns the diffusion loss on a training example x."""
    bsz, h, w, c = x.shape
    # Add noise to data.
    t = np.random.uniform(0, 1, size=[bsz, 1, 1, 1])
    eps = np.random.normal(0, 1, size=[bsz, h, w, c])
    # diffusing at t timestep
    x_t = np.sqrt(gamma(t)) * scale * x + sqrt(1-gamma(t)) * eps
    # Denoise and compute loss.
    x_t = x_t / x_t.std(axis=(1,2,3), keepdims=True) if normalize else x_t
    eps_pred = neural_net(x_t, t)
    loss = (eps_pred - eps)**2
    return loss.mean()
```

采样阶段需要跟训练阶段保持一致，如果训练阶段使用了归一化，那么采样阶段也需要归一化，代码如下，

```python
def generate(steps, gamma=lambda t: 1-t, scale=1, normalize=True):
    x_t = normal(mean=0, std=1)
    for step in range(steps):   # step=0,...,T-1
    # Get time for current and next states.
    t_now = 1 - step / steps    # reverse-time, t is from 1 to 0
    t_next = max(1 - (step+1) / steps, 0)
    # Predict eps & jump to x at t_next.
    x_t = x_t / x_t.std(axis=(1,2,3), keepdims=True) if normalize else x_t
    eps_pred = neural_net(x_t, t_now)
    x_t = ddim_or_ddpm_step(x_t, eps_pred, t_now, t_next)
    return x_t
```

# 3. 实验

## 3.1 策略 1：noise schedule 的 效果

保持 $b=1$ 不变，评估三种 noise schedule： cosine, sigmoid, linear

结果如表 1 所示

![](/images/diffusion_model/noise_scheduling_4.png)

<center>表 1. 不同 noise schedule 的 FID 值。FID 越小越好。</center>

从表 1 可以看出：不同大小的图像需要不同的 noise schedule 函数，以便获得最佳效果，并且由于 noise schedule 函数有多个超参数，所以难以找到最佳 noise schedule。

## 3.2 策略 2：input scaling 的效果

使用不同的 $b$ 值，实验结果如表 2 所示，

![](/images/diffusion_model/noise_scheduling_5.png)

<center>表 2.</center>

从表 2 实验结果可以看出：

1. image size 增大时，最佳 input scaling factor（$b$ 值）变小
2. 与表 1 相比，调节 input scaling factor 效果更好，例如 $256 \times 256$ 的图像，最佳 FID 从 **4.28** 降至 **3.52** 。
3. 线性函数 $1-t$ 比余弦函数 $\cos(s=0.2,e=1,\tau=1)$ 好。



# 4. 结论

本文实验证明：对于每个的图像数据任务，训练 diffusion models 时，需要选择合适的 noise scheduling 。