---
title: Improved Denoising Diffusion Probabilistic Models
date: 2022-07-08 11:58:42
tags: diffusion model
mathjax: true
---

论文：[Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)

源码：[openai/improved-diffusion](https://github.com/openai/improved-diffusion)

# 1. DDPM


## 1.1 定义

给定一个数据分布 $x_0 \sim q(x_0)$，前向加噪过程生成 $x_1, \ldots, x_T$，噪声类型为高斯噪声，方差为 $\beta_t \in (0,1)$，

$$\begin{aligned}q(x_1,\ldots, x_T |x_0)=\prod_{t=1}^T q(x_t|x_{t-1})
\\ q(x_t|x_{t-1})=\mathcal N(x_t; \sqrt {1-\beta_t} x_{t-1}, \beta_t I) \end{aligned}\tag{1}$$


如果 $T$ 足够大，且 $\beta_t$ 值选择恰当，那么 $x_T$ 接近各项同性的高斯分布。如果能得到准确的反向分布 $q(x_{t-1}|x_t)$，那么采样 $x_T \sim \mathcal N(0, I)$ 根据反向过程可以得到来自 $q(x_0)$ 的样本。

反向过程的分布 $q(x_{t-1}|x_t)$ 根据贝叶斯定理，可以计算得到结果如下

$$q( x_{t-1}| x_t,  x_0)=\mathcal N( x_{t-1}; \tilde {\mu}_t( x_t,  x_0), \tilde {\beta}_t I) \tag {2}$$

其中 

$$\tilde {\mu} _ t ( x _ t,  x _ 0) = \frac {\sqrt {\overline \alpha _ {t-1}}\beta _ t}{1-\overline \alpha _ t} x _ 0 + \frac {\sqrt {\alpha _ t}(1-\overline \alpha_{t-1})}{1-\overline \alpha_t}  x _ t, \quad \tilde \beta _ t = \frac {1-\overline \alpha_{t-1}}{1-\overline \alpha _ t} \beta _ t \tag{3}$$

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

DDPM 中损失函数为 $L_{simple} = E _ {t, x _ 0, \epsilon} [||\epsilon - \epsilon _ {\theta} (x _ t, t)||^2]$。DDPM 中，作者发现优化 $L_{simple}$ 比优化 $L_{vlb}$ 更好，采样得到的样本质量更好。

DDPM 中固定方差为 $\Sigma_{\theta}(x_t, t)=\sigma _ t ^ 2 I$，所以即使 $L_{simple}$ 与 $\Sigma_{\theta}(x_t, t)$ 无关，无法对方差进行优化也没关系。

DDPM 中作者设置 $\Sigma_{\theta}(x_t, t)=\sigma_t^2 I$，其中 $\sigma_t^2$ 使用 $\beta_t$ 或者 $\tilde \beta_t$（两者结果相近），而 $\beta_t$ 和 $\tilde \beta_t$ 分别表示方差的上限和下限，所以为何两个极端的 $\sigma_t^2$ 选择其结果相近？

如图 1，计算 $\beta_t$ 和 $\tilde \beta_t$ 的值，除了在 $t=0$ 附近，其他时候 $\beta_t$ 和 $\tilde \beta_t$ 值几乎相等。这表明模型处理极小的细节改变。如果增大 扩散 steps $T$，$\beta_t$ 和 $\tilde \beta_t$ 在多数时候依然保持相近，所以 $\sigma_t$ 不太会影响采样质量，而采样质量主要由 $\mu_{\theta}(x_t,t)$ 决定。

![](/images/diffusion_model/improved_ddpm_1.png)

图 1. $\tilde \beta_t / \beta_t$ 随 step 的关系图


虽然对于采样质量而言，固定 $\sigma_t$ 的值是一个不错的选择，但是对于 对数似然 则未必。如图 2，开始的几个扩散 step 贡献了大部分负对数似然（即损失 $L_{vlb}$）。由此可见，使用更好的 $\Sigma_{\theta}(x_t, t)$ 可以改善对数似然，而不会出现图 2 中那样的不稳定现象。

![](/images/diffusion_model/improved_ddpm_2.png)

<center>图 2. VLB vs step</center>


从 图 1 中可见，合适的 $\Sigma_{\theta}(x_t, t)$ 区间比较小（step 为 0 的附近），故直接使用神经网络预测 $\Sigma_{\theta}(x_t, t)$ 显得不太容易。作者选择在 log 空间对 $\beta_t$ 和 $\tilde \beta_t$ 进行插值作为方差。具体地，模型输出一个向量 $v$ （维度与 $x_t$ 相同），然后计算出协方差

$$\Sigma_{\theta}(x_t, t)=\exp (v \log \beta_t + (1-v) \log \tilde \beta_t) \tag{6}$$

将 $\beta _ t$ 和 $\tilde \beta _ t$ （均为标量）expand 为与 $x _ t$ 相同大小的向量，然后 $v$ 与 $\log \beta _ t$ 按元素相乘，$1-v$ 与 $\log \tilde \beta _ t$ 也是按元素相差，最后得到的 $\Sigma _ {\theta}(x _ t, t)$ 也是一个与 $x _ t$ 相同大小的向量，然后反向方差 $\Sigma _ {\theta}(x _ t, t)$ 与噪声 $\epsilon$ 也是按元素相乘，得到相同大小的向量后再与 $\tilde \mu _ t$ 向量想加，得到 $t-1$ 时刻的样本 $x _ {t-1}$ 。

由于是在 log 空间，所以对 $v$ 不设其他约束条件，这也导致模型预测的协方差可能会超出插值范围，但是作者实践中发现并没有出现这种现象。

对优化目标进行修改

$$L_{hybrid}=L_{simple}+ \lambda L_{vlb} \tag{7}$$

作者实验中设置 $\lambda=0.001$ 避免 $L_{vlb}$ 作用盖过 $L_{simple}$ 。使用 (7) 式损失函数，就可以指导学习 $\Sigma _ {\theta}(x _ t, t)$ 。

### 2.1.1 源码分析

本来模型输出 $\epsilon _ {\theta}$，与 $x _ t$ 具有相同的 shape，现在再输出一个向量 $v$，与 $x _ t$ 也是相同 shape，所以模型输出 shape 为 `batch_size, C*2, H, W`，相关代码如下，

```python
# 训练阶段，计算 loss 的相关代码

model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)     # 模型输出：包含 eps 和 v
if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:   # 反向过程的 sigma 不固定，需要学习
    B, C = x_t.shape[:2]    # batch_size, channels
    assert model_output.shape == (B, C * 2, *x_t.shape[2:])
    model_output, model_var_values = torch.split(model_output, C, dim=1)    # 一半channels为 eps，一半channels为 v
    frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
    terms['vb'] = self._vb_terms_bpd(       # 计算 variance lower bound loss，bits per dim
        model=lambda *args, r=frozen_out: r,
        x_start=x_start,    # x0，输入图像数据
        x_t=x_t,
        t=t,
        clip_denoised=False
    )['output']     # L_{vlb}
    # MSE: Loss=L_{simple}+L_{vlb}
    # RESCALED_MSE: 对 L_{vlb} 先 rescale，然后与 L_{simple} 想加
    if self.loss_type == LossType.RESCALED_MSE:
        # 除以 1000. 就是 (7) 式中的 lambda=0.001
        # 乘以 num_timesteps：rescale vlb 损失 to estimate full vlb 损失，
        # L_{vlb} = L1+L2+...+LT = T * Lt
        terms['vb'] *= self.num_timesteps / 1000.0
    target = {
        ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )[0]
        ModelMeanType.START_X: x_start,
        ModelMeanType.EPSILON: noise,   # DDPM 中 target 为 eps
    }[self.model_mean_type]
    terms['mse'] = mean_flat((target - model_output) ** 2)  # L_{simple}
    if 'vb' in terms:
        terms['loss'] = terms['mse'] + terms['vb']
    else:
        terms['loss'] = terms['mse']
```

计算 variance bound （即 (7) 式中的 $L_{vlb}$）的函数如下，

```python
def _vb_terms_bpd(self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None):
    # q(x_{t-1}|x_t) 的 期望 \tilde \mu 和 log 方差
    true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)
    # 使用模型输出 eps 计算出 x0，然后得到 \tilde \mu_{\theta}
    out = self.p_mean_variance(
        model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
    )
    # q 和 p 的 KL 散度，注意这里计算的 kl 是一个 tensor，（对单个样本而言）
    kl = normal_kl(
        true_mean, true_log_variance_clipped, out['mean'], out['log_variance']
    )
    kl = mean_flat(kl) / np.log(2.0)    # kl 对每个样本求均值，得到一个 batch_size 的向量
                                        # 本来是要求和，但是要转为 bits per dim，所以除以 dims，就变成求均值了
    # t>0 时使用 KL散度作为损失，t=0时，使用负对数似然，如下所示
    decoder_nll = -discretized_gaussian_log_likelihood(
        x_start, means=out['mean'], log_scales=0.5*out['log_variance']
    )
    decoder_nll = mean_flat(decoder_nll) / np.log(2.0)  # 对每个样本求均值，并转为 bits/dim
    output = torch.where((t==0), decoder_nll, kl)   # t, decoder_nll, kl 均为 (batch_size,)
    return {"output":output, "pred_xstart": out['pred_xstart']} # pred_xstart: 根据模型输出，预测的 x0
```

计算 KL 散度损失的代码如下，

```python
def normal_kl(mean1, logvar1, mean2, logvar2):
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )
```

下面给出高斯分布的 KL 散度计算的数学推导，

$$D_{KL}(q(z)||p(z)) =\int q(z) \log \frac {q(z)}{p(z)}dz \tag{8}$$

其中 $q(z) \sim \mathcal N(\mu _ 1, V _ 1), \ p(z) \sim \mathcal N(\mu _ 2, V _ 2 )$ 。

那么上式为

$$\begin{aligned} E_{q} \left[ \log \left(\frac {\exp (-\frac 1 {2V _ 1} (z-\mu _ 1)^2)V _ 2 ^ {1/2}}{\exp (-\frac 1 {2V _ 2} (z-\mu _ 2) ^ 2) V _ 1 ^ {1/2}} \right)\right] &= E _ q \left[\frac 1 2 (\log V _ 2 - \log V _ 1) + \frac 1 {2 V _ 2} (z-\mu _ 2) ^2 - \frac 1 {2 V _ 1}(z-\mu _ 1)^2 \right]
\\\\ &= \frac 1 2 (\log V _ 2 - \log V _ 1) + \frac 1 {2 V _ 2} E _ q [z ^ 2 - 2 \mu _ 2 z + \mu _ 2 ^2]- \frac 1 {2 V _ 1} E _ q [(z - \mu _ 1)^2]
\\\\ &= \frac 1 2 (\log V _ 2 - \log V _ 1) + \frac 1 {2 V _ 2} (V_ 1  + \mu _ 1 ^ 2 - 2 \mu _ 2 \mu _ 1 + \mu _ 2 ^2)- \frac 1 2
\\\\ &= \frac 1 2 (-1 + \log V _ 2 - \log V _ 1 + \frac {V _ 1} {V _ 2} + (\mu _ 1 - \mu _ 2)^2 / V _ 2)
\end{aligned}$$

上式推导中，用到了 $E_q [C] = C$， $E_q [Cz]= C \mu _ 1$， $E_q [(z-\mu _ 1) ^ 2] = V _ 1$ 以及 $E _ q [z^2] = V _ 1 + \mu _ 1 ^ 2$

**# 采样阶段**

```python
def p_mean_variance(
    self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
):
    ...
    model_output = model(x, self._scale_timesteps(t), **model_kwargs)
    assert model_output.shape == (B, C * 2, *x.shape[2:])
    model_output, model_var_values = torch.split(model_output, C, dim=1)
    if self.model_var_type == ModelVarType.LEARNED_RANGE:
        min_log = _extract_into_tensor(                     # \tilde \beta_t
            self.posterior_log_variance_clipped, t, x.shape
        )
        max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)  # \beta_t
        # 假设输出的 v 范围在 [-1, 1]，这是合理的，因为layer 输出值在以 0 中心的邻域
        # 将 v rescale 到范围 [0, 1]
        frac = (model_var_values + 1) / 2
        model_log_variance = frac * max_log + (1 - frac) * min_log
        model_variance = torch.exp(model_log_variance)
```

## 2.2 改善噪声机制

DDPM 中使用线性噪声 （$\beta_1=10^{-4}$ 线性增大到 $\beta_T=0.02$），这在高分辨率图像中效果很好，但是在诸如 $64 \times 64$ 或者 $32 \times 32$ 图像中效果次佳，因为线性加噪后的图像中噪声太强，如图 3，上一行是线性加噪过程，下一行是 cosine 加噪过程，显然最后的几个 step 中，上一行几乎是纯噪声，而下一行则是缓慢的加噪。

![](/images/diffusion_model/improved_ddpm_3.png)
<center>图 3. </center>

cosine 噪声机制如下，

$$\overline \alpha_t = \frac {f(t)} {f(0)}, \quad f(t)=\cos \left(\frac {t/T+s}{1+s} \cdot \frac {\pi} 2\right)^2 \tag{9}$$

根据 DDPM 中 $\alpha_t$ 与 $\beta_t$ 的关系有

$$\beta_t = 1- \frac {\overline \alpha_t} {\overline \alpha_{t-1}}, \quad t=1,..., T \tag{10}$$

如图 4，是线性和 cosine 两种噪声机制中 $\overline \alpha_t$ 的对比，cosine 机制中，在 diffusion step 的中间段，$\overline \alpha_t$ 值近似线性下降，在两端 $t=0, \ t=T$ 则缓慢变化，而线性机制中，$\overline \alpha_t$ 快速下降接近 0 值，使得图像信息过快地被损坏。

![](/images/diffusion_model/improved_ddpm_4.png)

图 4. $\overline \alpha_t$ 随 step  $t$ 的变化关系

作者发现需要一个小的 offset $s$ 值，以防 $\beta_t$ 在 $t=0$ 附近太小而导致开始时网络难以准确地预测 $\epsilon$。作者选择 $s=0.008$ 使得 $\sqrt {\beta_0}$ 稍微小于 pixel bin size $1/127.5$（将 $[0,255]$ 压缩至 $[-1,1]$，故 bin size 为 $2/255$）。

### 2.2.1 相关源码

计算 cosine beta 的代码如下，

```python
def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))

# 调用
betas_for_alpha_bar(
    num_diffusion_timesteps, # T
    lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2  # f(t/T)
)
```

## 2.3 降低梯度噪声

我们期望直接通过优化 $L_{vlb}$ 以获得最佳对数似然，而不是优化 $L_{hybrid}$，但是 $L_{vlb}$ 实际上较难优化。作者寻找一种降低 $L_{vlb}$ 方差地方法来对 对数似然 进行优化。

观察图 2， 发现 VLB 损失在不同 timestep 的损失幅度也大不相同，作者猜测均匀的从 $[1,T]$ 采样得到时刻 $t$ 会给 VLB 损失带来不必要的噪声，如图 5，VLB 损失比 hybrid 损失更大。为了解决这个问题，使用如下采样方式，

$$L_{vlb}=E_{t \sim p_t} \left[ \frac {L_t}{p_t} \right] \tag{11}$$

$$p_t \propto \sqrt {E[L_t^2]}, \quad \sum p_t = 1 \tag{12}$$

即，根据 $t \sim p_t$ 而非均匀分布来对 $t$ 采样。然而 $E[L _ t ^ 2]$ 事先并不知道其值，且训练过程中一直变化，那么对每个损失项，保存 10 个历史值，并在训练过程中动态更新这 10 个值。训练开始时，根据均匀分布对 $t$ 采样，直到每个 $t \in [0, T-1]$ 均有 10 个值，计算损失 $L_t$ （有 10 个值），然后开始计算 $p_t = \sqrt {E[L_t^2]}=\sum _ {i=1}^{10} [L _ t ^{(i)}] ^ 2 / 10$ ，并根据  $\sum p_t = 1$ 进行归一化，然后根据修正过地 $p_t$ 对 $t$ 进行采样，计算 $L_t$，注意从这里开始，一直需要动态地更新 $p_t$，每个 $t$ 值均维护最新的 10 个 $L_t$ 值用于更新 $p_t$ 。如图 5 中的 resampled 曲线。

![](/images/diffusion_model/improved_ddpm_5.png)
<center>图 5. ImageNet 64x64 上的学习曲线</center>

### 2.3.1 相关代码

```python
class LossSecondMomentResampler(LossAwareSampler):
    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        # 每个 t 的损失 Lt
        self._loss_history = np.zeros(      # (T, 10)
            [diffusion.num_timesteps, history_per_term], dtype=np.float64
        )
        # 每个 t 出现次数统计
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int)
    
    def weights(self):
        if not self._warmed_up():   # 在每个 t 出现 10 次之前，使用均匀分布采样
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)

        weights = np.sort(np.mean(self._loss_history ** 2, axis=-1))    # (T,)
        weights /= np.sum(weights)  # 归一化，使得 weights 表示概率
        weights *= 1 - self.uniform_prob    # pt * (1- 1/T)
        weights += self.uniform_prob / len(weights)

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # shift out the oldest loss term
                self._loss_history[t,:-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_time).all()
```

上述代码中，根据 (12) 式得到采样概率 $p _ t$ 后还进行了如下微调

$$p _ t := p _ t \cdot (1 - \frac 1 T) + u _ t \cdot \frac 1 T \tag{13}$$

其中 $u _ t = 1/ T$ 就是均匀分布的采样概率 。

# 3. 改善采样速率

作者使用 $T=4000$ 个扩散 steps，导致生成一个样本都要耗费若干分钟。本节研究如果降低 steps，那么性能会如何，实验结果发现预训练的 $L_{hybrid}$ 模型，在降低 steps 后生成的样本图像质量依然很高。降低 steps 使得样本生成事件从几分钟降至几秒。

模型训练过程中通常使用 $t$ 的相同序列值 $(1,2,\ldots, T)$ 进行采样，当然也可以使用子序列 $S$ 进行采样。为了将 steps 从 $T$ 降到 $K$，使用 $1 \sim T$ 之间的 $K$ 个等间隔的数（四舍五入取整），这 $K$ 个数构成序列 $S=(S_1,\ldots, S_K)$，从原来的 $\overline \alpha_1, \ldots \overline \alpha_T$ 中取 $\overline \alpha_{S_t}, \ldots, \overline \alpha_{S_K}$，然后计算

$$\beta_{S_t}=1-\frac {\overline \alpha_{S_t}}{\overline \alpha_{S_{t-1}}}, \quad \tilde \beta_{S_t}=\frac {1-\overline \alpha_{S_{t-1}}}{1-\overline \alpha_{S_t}} \beta_{S_t} \tag{14}$$

特别地，当 $K=T$ 时，与前面情况相同。

2.1.1 一节的源码中显示， $\Sigma _ {\theta} (x _ {S _ t}, S _ t)$ 乘以了 `num_timesteps`，所以当 $T$ 下降为 $K$ 时， $\Sigma _ {\theta} (x _ {S _ t}, S _ t)$ 自动进行了 rescale 。

## 3.1 相关源码

```python
class SpaceDiffusion(GaussianDiffusion):
    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps) # 子集 S=[S1,...,SK]
        base_diffusion = GaussianDiffusion(**kwargs)
        last_alpha_cumprod = 1.0    # (14) 式中的 \overline \alpha_{S_t-1}
        new_betas = []      # K 次 timestep 情况， 存储对应的 betas
        for i, alpha_cumprod in enumerate(base_diffusion.alpha_cumprod):
            if i in self.use_timesteps: # 需要采样
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timesteps_map.append(i)
        kwargs['betas'] = np.array(new_betas)
        super().__init__(**kwargs)
```

上述代码中，构造函数参数 `use_timesteps` 获取过程如下，

```python
def space_timesteps(num_timesteps, section_counts):
    '''
    num_timesteps: 原始 timestep 次数，T
    section_counts: 如果是 list，那么将 [1,...,T] 先等间距划分为 len(section_counts) 份子区间，例如第一子区间 [1,...,t1]，这个子区间等间隔取
    section_counts[0] 个元素
    '''
    if isinstance(section_counts, str):
        ...

    # 将 [1,...,T] 等分为 len(section_counts) 份子区间，每份区间大小为 size_per
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts) # 等分之后剩余元素数量
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        # 第 i 个子区间，目标是从中等间隔取 section_count 个元素

        # 首先修改前 extra 个子区间的大小，+1，这样所有子区间大小总和为 T
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            rasie ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:  # 如果当前子区间要取的元素数量不超过1，那么修改为
            frac_stride = 1     # 取当前子区间中所有元素
        else:
            frac_stride = (size - 1) / (section_count - 1)  # 见下方说明
        
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps    # 累加当前子区间所取的元素列表
        start_idx += size           # 移动到下一个子区间的 start 位置
    return set(all_steps)
```

上述代码中，计算 `frac_stride` 使用数学表达如下，对于数组，长度为 $n$，要等间隔的取 $m$ 个元素，且第一个元素必须取得，那么就要从剩余 $n-1$ 个元素中再取 $m-1$ 个元素，间隔应该是 $(n-1)/(m-1)$。
    