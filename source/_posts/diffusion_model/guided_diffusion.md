---
title: guided-diffusion
date: 2022-09-30 14:02:23
tags: diffusion model
mathjax: true
img:
summury: 支持基于标签的图像生成
---

论文：[Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2015.05233)
源码：[guided-diffusion](https://github.com/openai/guided-diffusion)


# 1. 框架改进

## 1.1 AdaGN

adaptive group normalization（AdaGN）：将 timestep 和 class 嵌入向量合并到每个 residual block，合并操作位于 group normalization 之后，如下所示


$$ AdaGN(h,y)=y_s \cdot GN(h)+y_b \tag{1}$$

其中 $y=[y_s,y_b]$ 是 timestep 和 class label 的 embedding 经过一个线性变换而来，

$$y = [y _ s , \ y _ b ] = [e _ s, \ e _ b] \begin{bmatrix} W _ s \\\\ W _ b \end{bmatrix} + b$$

其中 $b$ 是 bias，$W _ s, W _ b$ 为 weights，$e _ s, e _ b$ 是 embedding vector。当 $W _ s = W _ b = W$ 时，上式变成 $y = (e _ s + e _ b) W + b$ 。

**# Group Norm**

Group Norm 是将 channels 分组，然后归一化，介于 Instance Norm（单个 channel，归一化用到数据量 $H \times W$）和 Layer Norm（全部 channels，归一化用到数据量 $C \times H \times W$）。

### 1.1.1 AdaGN 源码

```python
class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def normalization(channels):    # 将 channels 分成 32 个 group
    return GroupNorm32(32, channels)

class ResBlock(TimestepBlock):
    def _forward(self, x, emb):
        '''
        x: (batch_size, in_channels, H, W)  当前layer输入数据
        emb: embedding vector,  (batch_size, model_channels)
            参见下方 UNetModel.forward 方法中的 `emb` 变量
        '''
        h = self.in_layers(x)   # self.in_layer: GN32+SiLU+conv3x3
        # self.emb_layers: SiLU + FC
        emb_out = self.emb_layers(emb).type(h.dtype)    # 参见上方 y 的计算式
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            # self.out_layers: GN32+SiLU+FC
            # out_norm: GN32;  out_rest: SiLU+FC
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            # 参见上方 y=[ys, yb]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift   # 参见 (1) 式
            h = out_rest(h)
        return self.skip_connection(x) + h

class UNetModel(nn.Module):
    def forward(self, x, timesteps, y=None):
        '''
        x: (batch_size, 3, img_size, img_size)
        timesteps: (batch_size, )
        y: (batch_size, num_classes), one-hot vector
        '''
        hs = []
        # position encoding timesteps to embedding vectors
        # emb: (batch_size, model_channels)
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:    # conditioned on class labels
            # 1. embedding class indices into embedding vectors, (batch_size, model_channels)
            # 2. sum embedding vectors of timesteps and class labels
            emb = emb + self.label_emb(y)   # 作为 ResBlock 的一个输入
```

图 1 是 ResBlock 的结构示意图，

![](/images/diffusion_model/unet_2.png)
<center>图 1. ResBlock 结构图</center>

# 2. 分类器指导

通过对 GAN 的观察，扩散模型利用分类标签作为条件也有很多方法，例如前面 `1.1` 一节的 adaptive group normalization 用到了分类标签。这里作者使用了另一种方法：利用一个分类器 $p(y|x)$ 来改善扩散模型的样本生成过程。一个预训练好的扩散模型可以利用分类器的梯度为条件，具体地，训练一个分类器 $p_{\phi}(y|x_t,t)$ ，输入是添加噪声后的图像 $x _ t$，然后使用梯度 $\nabla_{x_t} \log p_{\phi}(y|x_t, t)$ 来指导采样过程，最终生成的图像趋于标签 $y$ 。

本节一步步推导如何使用分类器进行条件采样，以及实际应用中如何利用分类器提高采样样本的质量。

为了表述简洁，以下使用简化版符号 $p _ {\phi} (y|x _ t, t)=p _ {\phi} (y|x _ t), \ \epsilon _ {\theta} ( x _ t, t) = \epsilon _ {\theta} (x _ t)$，但是需要注意，对于不同 timestep $t$，他们均表示不同的函数，并且模型也以 $t$ 为条件，即，timestep 是模型的一个输入。

## 2.1 条件逆噪声过程

DDPM 是无条件的扩散模型，反向过程为 $p_{\theta}(x_t,|x_{t+1})$。现以分类标签 $y$ 作为条件，采样过程可表示为

$$p_{\theta, \phi}(x_t|x_{t+1},y) = Z p_{\theta}(x_t|x_{t+1}) p_{\phi}(y|x_t) \tag{2}$$ 

其中 $Z$ 是归一化常量。$\theta$ 是扩散模型参数，$\phi$ 是分类器模型参数。

证明过程参考论文附录 H。

反向过程就是通过 (2) 式逐步（timestep）采样，然而 (2) 式采样不好处理，通过一种经过扰动的高斯分布来近似 (2) 式的分布（见下文 (6) 式），下面给出推导过程。

回顾扩散模型的反向过程可知，

$$p_{\theta}(x_t|x_{t+1})=\mathcal N(\mu, \Sigma) \tag{3}$$

取对数

$$\log p_{\theta}(x_t|x_{t+1})=-\frac 1 2 (x_t -\mu)^{\top}\Sigma (x_t-\mu) + C \tag{4}$$

对于分类器预测概率对数，可以假设其曲率比 $\Sigma^{-1}$ 小（即 $\log p(y|x_t)$ 曲线变化更缓慢，或者说方差较大），这个假设是合理的，因为当扩散步数 $T \rightarrow \infty$ 时，有 $\|\Sigma\| \rightarrow 0$，原因说明如下：

<details>
<summary>方差趋于 0 的说明</summary>

$\beta_t$ 使用线性 schedule，当 $T=1000$ 时，$\beta _ t$ 从 $10^{-4}$ 线性增大到 $0.02$，那么对于任意的步数 $T$，反比缩放 $\beta_t$ 的范围，即 $\beta _ t$ 从 $10^{-4} \times \frac {1000}T$ 线性增大到 $0.02 \times \frac {1000} T$，那么当 $T \rightarrow \infty$ 时有 $\beta _ t \rightarrow 0$，于是反向过程的方差 $||\Sigma|| \rightarrow 0$ 。

**# 代码**

```python
if schedule_name == 'linear':
    scale = 1000 / num_diffusion_timesteps
    beta_start = scale * 0.0001
    beta_end   = scale * 0.02
    return np.linspance(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
```

为何 $\beta _ t$ 的范围要与 $T$ 成反比，能不能固定不变？这个问题目前我也不清楚。
</details>

<br/>

既然 $\log p(y|x_t)$ 曲线变化更缓慢（类似于直线）， 于是使用 Taylor 展开，高阶项全部忽略，

$$\begin{aligned}\log p _ {\phi}(y|x _ t) & \approx \log p _ {\phi}(y|x _ t)|  _ {x _ t=\mu} + (x _ t-\mu)\nabla_{x _ t} \log p _ {\phi}(y|x _ t)| _ {x _ t=\mu}
\\\\ &=(x _ t-\mu) g + C _ 1
\end{aligned} \tag{5}$$


结合上面两式，可知

$$\begin{aligned} \log (p_{\theta}(x_t|x_{t+1})p_{\phi}(y|x_t)) & \approx -\frac 1 2 (x_t-\mu)^{\top} \Sigma^{-1}(x_t-\mu) + (x_t-\mu) g + C_2 
\\\\ &= -\frac 1 2 (x_t -\mu -\Sigma g)^{\top} \Sigma^{-1} (x_t -\mu -\Sigma g) + C_3
\\\\ &= \log p(z) + C_4 
\end{aligned} \tag{6}$$

其中 $z \sim \mathcal N(\mu + \Sigma g , \Sigma)$ 。相较于 (2) 式可知 $C_4$ 对应归一化因子 $Z$ 。根据 (6) 式可知，条件转移过程也是高斯型，与 DDPM 中得非条件转移类似，只是在期望 $\mu$ 的基础上再平移 $\Sigma g$ 。采用步骤如下文算法 1 所示，这里作者引入了一个关于梯度的 scale 因子 $s$。

**# 关于 scale 因子 $s$ 的说明**

作者观察发现，当 $s=1$ 时，分类器对最终的样本在正确分类上的预测概率为 50% 左右（注：这句话是说，分类器对每个样本的分类预测概率是一个向量，向量长度为 `num_classes`，表示每个分类对应的预测概率，在 target 分类对应的预测概率则大约为 50%），但是这些样本从视觉观察角度看，与这些分类并不匹配。而提高分类器梯度则可以修复这个问题，可以大大提高 target 分类对应的预测概率（接近 100%）。如何理解？

注意到 $s \cdot \nabla _ x \log p(y|x) = \nabla _ x \log \frac 1 Z p(y|x) ^ s$ ，这表示使用 scale 因子 $s$ 后，分类器分布正比于 $p(y|x) ^ s$，当 $s>1$ 时，分类器分布变得更加尖锐（sharper），概率集中于分布的 mode，于是可以得到更高精确性的样本（但是多样性更低）。



根据 DDPM 中分析，反向转移分布的期望为

$$\mu = \frac 1 {\sqrt {\alpha_t}} \left(x_t - \frac {\beta_t}{\sqrt {1-\overline \alpha_t}} \epsilon_{\theta}(x_t)\right) \tag{7}$$

且 $\Sigma$ 我们选取固定的 $\beta_t$，并且 (6) 式 中对 $\Sigma g$ 项增加一个因子 $s=\frac 1 {\sqrt {\alpha_t}} > 1$，那么

$$\begin{aligned}\mu+ s\Sigma g &= \frac 1 {\sqrt {\alpha_t}} \left(x_t - \frac {\beta_t}{\sqrt {1-\overline \alpha_t}} \epsilon_{\theta}(x_t)\right) + \frac 1 {\sqrt {\alpha_t}}\beta_t \cdot g
\\\\ &=\frac 1 {\sqrt {\alpha_t}} \left(x_t -\frac {\beta_t}{\sqrt {1-\overline \alpha_t}}[\epsilon_{\theta}(x_t) - \sqrt {1-\overline \alpha_t} \cdot g]\right)
\end{aligned}$$

令

$$\hat \epsilon(x_t) := \epsilon_{\theta}(x_t) - \sqrt {1-\overline \alpha_t}  \nabla_{x_t} \log p_{\phi}(y|x_t) \tag{8}$$

于是，引入分类器后，扩散模型反向过程的改变为

$$\epsilon_{\theta}(x_t) \rightarrow \hat \epsilon(x_t) \tag{9}$$

当然分类器梯度因子 $s$ 也可以选择其他 $>1$ 的值，可以理解为 $\Sigma_{\theta}$ 选择不同于 $\beta_t$ 的某个值。

## 2.2 条件 DDIM

DDIM 与 DDPM 有所不同，DDIM 前向过程不是马尔可夫过程，但是 DDIM 前向过程依然满足

$$q(x_t|x_0)=\mathcal N(\sqrt {\overline \alpha_t} x_0,  {1-\overline \alpha_t}I) \tag{10}$$

存在关系

$$x_t = \sqrt{\overline \alpha_t} x_0 + \sqrt {1-\overline \alpha_t} \cdot \epsilon \tag{11}$$

模型预测 $\epsilon$ 为 $\epsilon_{\theta}(x_t)$ ，反向过程中 $p_{\theta}(x_t)$ 应尽量逼近 $q(x_t|x_0)$，于是

$$p_{\theta}(x_t) = \mathcal N(\sqrt {\overline \alpha_t} x_0',  {1-\overline \alpha_t} I) \tag{12}$$

其中 $x_0'$ 是模型在 $t$ timestep 对 $x_0$ 的一次预测，根据 (11) 式做变换，有 

$$x_0'(x_t) =\frac 1 {\sqrt {\overline \alpha_t}} [x_t - \sqrt {1-\overline \alpha_t} \cdot \epsilon_{\theta}(x_t)] \tag{13}$$

建立 score matching 与扩散模型之间的联系如下，

$$\nabla_{x_t}  \log p_{\theta}(x_t) = - \frac {x_t - \sqrt {\overline \alpha_t} x_0'}{1-\overline \alpha_t} = -\frac 1 {\sqrt {1-\overline \alpha_t}} \epsilon_{\theta}(x_t) \tag{14}$$

其中得分函数是 $\nabla _ {x _ t} \log p _ {\theta}(x _ t)$ ，扩散模型为 $\epsilon _ {\theta} (x _ t)$ ，使用 $p _ {\theta} ( x _ t)$ 代替 $q ( x _ t|x _ 0)$ 。

当考虑到分类 $y$ 时，使用 $p _ {\theta, \phi} (x _ t| y)$ 代替 $p _ {\theta} (x _ t)$，这里 $p _ {\theta, \phi} (x _ t| y) = Z p _ {\theta} (x _ t) p _ {\phi} (y|x _ t)$，$\theta, \ \phi$ 分别表示扩散模型和分类器参数，$Z$ 是与随机变量 $x _ t$ 无关的归一化常量，

<details>
<summary>细节</summary>

仿照附录 H 内容，基于条件 $y$ 的 $x _ t$ 分布为

$$\hat q (x _ t |  y) = \frac {\hat q (y|x _ t) \hat q(x _ t)}{\hat q(y)}$$

其中：

1. $\hat q (y)$ 是某个确定不变的值（即，某个确定的关于分类的分布，在分类 $y$ 处的概率）
2. $\hat q$ 的定义需要满足 $\hat q(x _ t) = q(x _ t)$。DDIM 中无论是什么样的转移过程（马尔可夫或非马尔可夫），均不改变 $x _ t$ 的分布，现在使用条件型转移过程，依然要保持 $x _ t$ 的分布不变这一原则。


</details>
<br/>


得分函数变为

$$\begin{aligned} \nabla _ {x _ t} \log p _ {\theta, \phi} (x _ t|y) &=\nabla_{x_t} \log (p_{\theta}(x_t) p_{\phi}(y|x_t)) 
\\\\ &= \nabla_{x_t} \log p_{\theta}(x_t) + \nabla_{x_t} \log p_{\phi}(y|x_t) 
\\\\ &=-\frac 1 {\sqrt {1-\overline \alpha_t}} \epsilon_{\theta}(x_t)+\nabla_{x_t} \log p_{\phi}(y|x_t)
\end{aligned}$$



作 (8) 式变换得到 $\hat \epsilon(x_t)$ ，于是 

$$\nabla_{x_t} \log (p_{\theta}(x_t) p_{\phi}(y|x_t))=-\frac 1 {\sqrt {1-\overline \alpha_t}} \hat \epsilon(x_t) \tag{15}$$

(14)、 (15) 两式分别为无条件扩散模型和有条件扩散模型的得分函数，两者形式一致，那么仿照无条件扩散模型的反向转移，根据得分函数与扩散模型之间的联系形式，容易写出有条件扩散模型的反向转移，步骤如下：

1. 根据 (13) 式可以得到 DDIM 反向过程中对 $x _ 0$ 的估计即 $x _ 0'$，注意要将 (13) 式中的 $\epsilon _ {\theta}(x _ t)$ 替换为 $\hat \epsilon(x _ t)$，得到 $x _ 0'$ 后代入下文 (16) 式中第三子式，可以得到 DDIM 反向过程的采样规则，具体如下方算法 2 所示。

---
**算法 1**：分类器指导的扩散模型（DDPM）采样过程，扩散模型描述为 $(\mu_{\theta}(x_t), \Sigma_{\theta}(x_t))$，分类器记为 $p_{\phi}(y|x_t)$

**输入**：分类标签 $y$，梯度因子 $s$

采样 $x_T \leftarrow \mathcal N(0,I)$

**for** $t=T,\ldots, 1$ **do**

&emsp; $\mu, \Sigma \leftarrow \mu_{\theta}(x_t), \Sigma_{\theta}(x_t)$

&emsp; 采样 $x_{t-1} \leftarrow \mathcal N(\mu+s\Sigma \nabla_{x_t} \log p_{\phi}(y|x_t)| _ {x _ t=\mu}, \Sigma)$

**end for**

**return** $x_0$

---
**算法 2**：分类器指导的 DDIM 采样过程，扩散模型为 $\epsilon_{\theta}(x_t)$，分类器为 $p_{\phi}(y|x_t)$

**输入** 分类标签 $y$，梯度因子 $s$

采样 $x_T \leftarrow \mathcal N(0, I)$

**for** $t=T,\ldots, 1$ **do**

&emsp; $\hat \epsilon \leftarrow \epsilon_{\theta}(x_t) - \sqrt {1-\overline \alpha_t} \nabla_{x_t} \log p_{\phi} (y|x_t)$

&emsp; $x_{t-1} \leftarrow \sqrt {\overline \alpha_{t-1}}\left(\frac {x_t - \sqrt {1-\overline \alpha_t} \hat \epsilon} {\sqrt {\overline \alpha_t}} \right) + \sqrt {1-\overline \alpha_{t-1}} \hat \epsilon$

**end for**

**return** $x_0$

---


DDIM 过程描述如下：

$$\begin{aligned} q _ {\sigma} (x _ {1:T}|x _ 0) &= q _ {\sigma} (x _ T |x _ 0) \prod _ {t=2} ^ T q _ {\sigma} (x _ {t-1}|x _ t, x _ 0)
\\\\ q _ {\sigma} (x _ T |x _ 0) &= \mathcal N(\sqrt {\overline \alpha _ T} x _ 0, (1-\overline \alpha _ T)I)
\\\\ q _ {\sigma} (x _ {t-1}|x _ t, x _ 0) &= \mathcal N \left(\sqrt {\overline \alpha _ {t-1}} x _ 0 + \sqrt {1 - \overline \alpha _ {t-1}-\sigma _ t ^ 2} \cdot \frac {x _ t - \sqrt {\overline \alpha _ t} x _ 0}{\sqrt {1-\overline \alpha _ t}}, \sigma _ t ^ 2 I \right)
\end{aligned} \tag{16}$$

# 3. 训练

## 3.1 模型说明

扩散模型框架采样 UNet，分类器使用 UNet 的下采样部分（即，Encoder 部分），使用 ImageNet 数据集，在输出 $8 \times 8$ spatial size 的 layer 之后使用 attention pool，从而将 spatial size 变成 $1 \times 1$ 。


分类器使用部分加噪的图片进行训练，这里加噪过程由对应的扩散模型进行加噪，加噪后的图片进行随机 crop 然后作为分类器的训练输入，以降低过拟合。

# 4. 源码

## 4.1 分类器训练

分类器模型

```python
class EncoderUNetModel(nn.Module):
    ...
    # timesteps 经过 position encoding，变成 embed_vector，size=model_channels=128
    # 然后 timesteps 经过 FC+SiLU+FC，vector size 变为 model_channels*4
    ch = int(channel_mult[0] * model_channels)  # 128
    # 另外一边，input data 经过 input_block LAYER（conv2d），输出channels=ch
    ds = 1  # down sampling rate
    # 以 img_size = 128 为例
    for level, mult in enumerate(channel_mult): # (1,1,2,3,4)
        for _ in range(num_res_blocks):     # 2
            layers = [
                ResBlock(
                    ch,
                    time_embed_dim, # model_channels * 4
                    dropout,
                    out_channels=int(mult * model_channels),
                    dims=dims,  # 2 -> 2d
                    ...
                )
            ]
            ch = int(mult * model_channels)
            if ds in attention_resolutions:     # 4, 8, 16
                layers.append(   # 对应 feature map size 32|16|8 时，添加attnblock
                    AttentionBlock(
                        ch,
                        use_checkpoint=use_checkpoint,
                        num_heads=num_heads,
                        num_head_channels=num_head_channels,
                        use_new_attention_order=use_new_attention_order
                    )
                )
            ... # 添加进 self.input_blocks
        if level != len(channels_mult) - 1:     # 除最后一组ResBlock 外，都要下采样
            out_ch = ch
            self.input_blocks.append(
                TimestepEmbedSequential(
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=out_ch,
                        dims=dims,
                        ...,
                        down=True
                    )
                )
            )
            ch = out_ch
            ds *= 2     # 下采样率 x2
    self.middle_block = TimestepEmbedSequential(
        ResBlock(...),
        AttentionBlock(...),
        ResBlock(...)
    )
    if self.pool == 'attention':
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            # out_channels: 分类数量，对于 ImageNet 数据集，为 1000
            # image_size // ds: baseline 输出 feature size
            # num_head_channels: multi-head attention，每个 head 的 channel
            AttentionPool2d(
                (image_size // ds), ch, num_head_channels, out_channels
            )
        )
```

通过代码可以看出分类器的网络结构，即，unet 的 encoder (down) ，其中部分 layer 后需要添加 AttentionBlock，downsampling 后的 middle block 由 ResBlock+AttentionBlock+ResBlock 组成。这里看下 AttentionBlock 的结构，

```python
class AttentionBlock(nn.Module):
    ...
        self.num_heads = self.channels // num_head_channels # channels // 64
        self.norm = normalization(self.channels)    # GroupNorm32
        # 将输入（四维）reshape 为 (B, C, -1)，然后经过 conv1d，输出 (B, 3C, -1)
        self.qkv = conv_nd(1, self.channels, self.channels * 3, 1)
        self.attention = QKVAttentionLegacy(self.num_heads)
        # zero_module：将 conv_nd 的参数初始化为 0
        self.proj_out = zero_module(conv_nd(1, self.channels, self.channels, 1))

    def _forward(self, x):
        # x: (B, C, H, W)
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)     # x: b, c, L
        qkv = self.qkv(self.norm(x))    # qkv: b, 3d, L
        h = self.attention(qkv)     # b,c,L
        h = self.proj_out(h)        # adjust via conv, (b,c,L)
        return (x + h).reshape(b, c, *spatial)  # (B, C, H, W)
```

通过以上代码不难知道，AttentionBlock 不改变输入 shape。由于 `channel_mult=(1,1,2,3,4)`，所以下采样率为 $2^{5-1}=16$，baseline 输出 `feat_size=(128/16, 128/16)=(8, 8)`，输出 `channels=model_channels * channel_mult[-1]=128 * 4=512`，故最终输出 feature shape 为 `(B, 512, 8, 8)`，然后使用 AttentionPool2d 池化，

```python
class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim, embed_dim, num_heads_channels, output_dim):
        '''
        spacial_dim: feature spatial size, H/W
        embed_dim: embed vector size。H*W 看作 sequence 长度，其中每个位置对应一个长度为 channels 的 vector，此即 embed vector
        num_heads_channels: multi-head attention，每个 head 的 channel
        output_dim: 输出（proj_out）channel
        '''
        super().__init__()
        # 这里 position encoding 没有采用正弦余弦，而是直接使用学习的 Embedding map
        self.positional_embedding = nn.Parameter(
            torch.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)
    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1) # convert to 1d data，(b, c, sequence_length)
        x = torch.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)    # (b, c, 1 + sequence_length)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)
        x = self.qkv_proj(x)    # (b, 3c, 1 + sequence_length)
        x = self.attention(x)   # (b, c, 1 + sequence_length)
        x = self.c_proj(x)      # (b, output_dim, 1 + sequence_length)
        return x[:, :, 0]       # (b, output_dim)
```

训练分类器调用代码，

```python
def forward_backward_log(data_loader, prefix='train'):
    batch, extra = next(data_loader)
    labels = extra['y'].to(dist_util.dev())     # 分类 label (B,)
    batch = batch.to(dist_util.dev())
    t, _ = schedule_sampler.sample(batch.shape[0], dist_util.dev()) # (B,)

    # xt=\sqrt{\overline alpha_t}x0+(1-\overline alpha_t)*eps
    batch = diffusion.q_sample(batch, t)    
    logits = model(batch, timesteps=t)  # log p
    loss = F.cross_entropy(logits, labels, reduction='none')
```

## 4.2 扩散模型训练

与 [Improved_DDPM](/2022/07/08/diffusion_model/Improved_DDPM) 的代码类似。

## 4.3 采样

**# Classifier guidance**

源文件 `classifier_sample.py` 。

```python
# unet, 扩散模型
model, diffusion = create_model_and_diffusion(...)
model.load_state_dict(...)
model.to(dist_util.dev())
model.eval()
# 分类器
classifier = create_classifier(...)
classifier.load_state_dict(...)
classifier.to(dist_util.dev())
classifier.eval()

while len(all_images) * args.batch_size < args.num_samples:
    # 随机获取一批分类 label
    classes = torch.randint(low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev())
    model_kwargs = {'y': classes}
    # 采样函数
    sample_fn = (
        # 使用 DDPM 或 DDIM
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )
    # 得到采样图像，（像素值范围 [-1, 1]）
    sample = sample_fn(model_fn, 
                       (args.batch_size, 3, args.image_size, args.image_size),
                       clip_denoised=args.clip_denoised,
                       model_kwargs=model_kwargs,
                       cond_fn=cond_fn,
                       device=dist_util.dev())
    # 恢复像素值到 [0, 255]
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.int8)
```

**# DDPM 采样函数**

```python
# 从 T-1,T-2,..., 0 逐步生成样本
def p_sample_loop_progressive(
    self,
    model,      # 执行 unet 前向传播的函数
    shape,      # 输入数据 shape
    noise=None, # 某个编码后的表示
    clip_denoised=True,
    denoised_fn=None,
    cond_fn=None,
    model_kwargs=None,
    device=None,
    progress=False
):
    if noise is not None:
        img = noise     # noise 表示某个编码后的表示，从这个编码开始生成样本
    else:
        img = torch.randn(*shape, device=device)    # 随机生成样本
    indices = list(range(self.num_timesteps))[::-1] # T-1,T-2,...,0
    for i in indices:
        t = torch.tensor([i] * shape[0], device=device) # (batch_size,)
        with torch.no_grad():
            out = self.p_sample(
                model,
                img,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs
            )
            yield out
            img = out['sample']

def p_sample(self, model, x, t, clip_denoised=True, denoised_fn=None,
             cond_fn=None, model_kwargs=None):
    '''根据 x_t, t，采样得到 x_{t-1}'''
    # 获取反向转移分布的期望和方差, p(x_{t-1}|x_t)=N(\mu,\Sigma)
    out = self.p_mean_variance(model, x, t, clip_denoised=clip_denoised,
                               denoised_fn=denoised_fn, model_kwargs=model_kwargs)
    noise = torch.randn_like(x)
    nonzero_mask = (            # 
        (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
    )
    if cond_fn is not None:     # classifier guidance
        out['mean'] = self.condition_mean(
            cond_fn, out, x, t, model_kwargs=model_kwargs
        )
    sample = out['mean']

def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
    # 参考算法 1，计算分类器概率的log 的梯度
    gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
    new_mean = (
        p_mean_var['mean'].float() + p_mean_var['variance'] * gradient.float()
    )
    return new_mean
```

计算 $\nabla _ {x _ t} \log p _ {\phi} (y|x _ t)$ 的代码如下，

```python
def cond_fn(x, t, y=None):
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)  # 要计算梯度
        logits = classifier(x_in, t)            # (batch_size, C)，非归一化得分
        log_probs = F.log_softmax(logits, dim=-1)   # (batch_size, C)，归一化，概率，再求 log
        # 取 log prob，结果为一 batch_size 长度的向量
        selected = log_probs[range(len(logits)), y.view(-1)]
        # classifier_scale: 上文的 scale 因子 s
        return torch.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale
```

**# DDIM 采样**

大体流程与 DDPM 采样类似，这里只看核心函数，

```python
def ddim_sample(
    self,
    model,
    x,
    t,
    clip_denoised=True,
    denoised_fn=None,
    cond_fn=None,
    model_kwargs=None,
    eta=0.0
):  # 某个 timestep 的采样
    # 扩散模型反向分布的期望和方差
    out = self.p_mean_variance(
        model, x, t, clip_denoised=clip_denoised,
        denoised_fn=denoised_fn,
        model_kwargs=model_kwargs
    )
    if cond_fn is not None:
        out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)
    # classifier guidance 后的新 eps，参考算法 2
    eps = self._predict_eps_from_xstart(x, t, out['pred_xstart'])
    alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
    alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
    # 参考 DDIM 一文的 sigma_t 的选值
    # 当 sigma_t 如下计算时，DDIM 转变为 DDPM，此时 eta=1
    # 选用不同的 eta 值，调节噪声 level
    sigma = (
        eta
        * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
        * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
    )
    noise = torch.randn_like(x)
    # 使用上文 (16) 式中第三子式计算 q(x_{t-1}|x_t) 转移分布的期望
    mean_pred = (
        out['pred_xstart'] * torch.sqrt(alpha_bar_prev)
        + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
    )
    nonzero_mask = (
        (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
    )
    sample = mean_pred + nonzero_mask * sigma * noise
    return {'sample': sample, 'pred_xstart': out['pred_xstart']}
    
def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
    # 根据 timesteps，提取对应的 \overline \alpha  (batch_size,)
    # 然后 expand to (batch_size, 1, ,1, 1)
    alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
    # xt=\sqrt{\overline alpha} x0 + \sqrt{(1-\overline alpha)} eps
    # => eps =
    eps = self._predict_eps_from_xstart(x, t, p_mean_var['pred_xstart'])
    # 参考算法 2
    eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
        x, self._scale_timesteps(t), **model_kwargs
    )
    out = p_mean_var.copy()
    # 根据新的 eps，重新计算 x0 的估计值
    out['pred_xstart'] = self._predict_xstart_from_eps(x, t, eps)
    # 重新计算反向转移分布的期望
    out['mean'], _, _ = self.q_posterior_mean_variance(
        x_start=out['pred_xstart'], x_t=x, t=t
    )
    return out
```


# 5. 附录

## H. 条件扩散过程

定义一个条件马尔可夫噪声过程 $\hat q$，与 $q$ 类似，并假定 $\hat q (y|x _ 0)$ 是已知确定的分布。对 $\hat q$ 定义如下，

$$\begin{aligned} \hat q(x _ 0) &:= q (x _ 0)
\\\\ \hat q(y|x _ 0) &:= 某个确定的分布
\\\\ \hat q (x _ {t+1}|x _ t, y) &:= q(x _ {t+1} | x _ t)
\\\\ \hat q (x _ {1:T}|x _ 0, y) &:= \prod _ {t=1} ^ T \hat q (x _ t | x _ {t-1}, y)
\end{aligned} \tag{H1}$$

从 (H1) 式中可见，马尔可夫转移过程多了一个条件：分类 label $y$，除此之外，全部与非条件型马尔可夫转移过程 $q$ 相同。

可以证明，当去掉 $\hat q$ 中的条件 $y$，$\hat q$ 所表示的转移过程也与 $q$ 相同，证明如下：

$$\begin{aligned} \hat q (x _ {t+1}|x _ t) &= \int _ y \hat q(x _ {t+1}, y|x _ t) dy
\\\\ &=\int _ y \hat q(x _ {t+1}|x _ t, y) \hat q (y|x _ t) dy
\\\\ &=\int _ y q(x _ {t+1}|x _ t) \hat q (y | x _ t) dy
\\\\ &= q(x _ {t+1}|x _ t) \int _ y \hat q (y | x _ t) dy
\\\\ &= q(x _ {t+1}|x _ t)
\\\\ &= \hat q(x _ {t+1}|x _ t, y)
\end{aligned} \tag{H2}$$

同样地，可以证明去掉条件 $y$ 后的联合分布 $\hat q(x _ {1:T}|x _ 0)$ 与 $q(x _ {1:T} | x _ 0)$ 一样，

$$\begin{aligned} \hat q (x _ {1:T}|x _ t) &=\int _ y \hat q(x _ {1:T}, y|x _ 0) dy
\\\\ &=\int _ y \hat q (y|x _ 0) \hat q (x _ {1:T}|x _ 0, y) dy
\\\\ &= \int _ y \hat q (y|x _ 0) \prod _ {t=1} ^ T \hat q (x _ t | x _ {t-1}, y) dy
\\\\ &= \int _ y \hat q (y|x _ 0) \prod _ {t=1} ^ T  q (x _ t | x _ {t-1}) dy
\\\\ &= \prod _ {t=1} ^ T  q (x _ t | x _ {t-1}) \int _ y \hat q (y|x _ 0) dy
\\\\ &= \prod _ {t=1} ^ T  q (x _ t | x _ {t-1})
\\\\ &= q(x _ {1:T}| x _ 0)
\end{aligned} \tag{H3}$$

现在推导边缘分布 $\hat q(x _ t)$ 如下，

$$\begin{aligned}\hat q(x _ t) &=\int _ {x _ {0:t-1}} \hat q(x _ {0:t}) d x _ {0:t-1}
\\\\ &= \int _ {x _ {0:t-1}} \hat q(x _ 0) \hat q(x _ {1:t}|x _ 0) d x _ {0:t-1}
\\\\ &= \int _ {x _ {0:t-1}} q(x _ 0)  q(x _ {1:t}|x _ 0) d x _ {0:t-1}
\\\\ &= \int _ {x _ {0:t-1}} q(x _ {0:t}) dx _ {0:t-1}
\\\\ &= q(x _ t) 
\end{aligned} \tag{H4}$$

以上结论 $\hat q (x _ t) = q (x _ t)$ 和 $\hat q (x _ {t+1}|x _ t) = q (x _ {t+1} | x _ t)$，根据贝叶斯定理可知，反向过程有 $\hat q(x _ t | x _ {t+1}) = q (x _ t | t _ {t+1})$。

考虑 $\hat q$ 的一个带噪分类函数 $\hat q(y|x _ t)$，其中 $x _ t$ 是 $x _ 0$ 添加噪声而得，以下证明 $\hat q (y|x _ t)$ 与 $x _ {t+1}$ 无关。

$$\begin{aligned} \hat q (y|x _ t, x _ {t+1}) &= \hat q(x _ {t+1}|x _ t, y) \frac {\hat q(y|x _ t)} {\hat q (x _ {t+1}|x _ t)}
\\\\ &=\hat q(x _ {t+1}|x _ t) \frac {\hat q(y|x _ t)} {\hat q (x _ {t+1}|x _ t)}
\\\\ &= \hat q(y|x _ t)
\end{aligned} \tag{H5}$$

考虑带条件 $y$ 的反向过程，

$$\begin{aligned} \hat q(x _ t | x _ {t+1}, y) &= \frac {\hat q(x _ t, x _ {t+1}, y)} {\hat q(x _ {t+1}, y)}
\\\\ &= \frac {\hat q(x _ {t+1}) \hat q(x _ t | x _ {t+1}) \hat q(y| x _ t, x _ {t+1})}{\hat q (x _ {t+1}) \hat q (y|x _ {t+1})}
\\\\ &= \frac {\hat q(x _ t | x _ {t+1}) \hat q(y| x _ t, x _ {t+1})}{\hat q (y|x _ {t+1})}
\\\\ &= \frac {q (x _ t|x _ {t+1}) \hat q (y|x _ t)} {\hat q (y|x _ {t+1})}
\end{aligned} \tag{H6}$$

这里 $x _ t$ 为随机变量， $\hat q(y|x _ {t+1})$ 不依赖于 $x _ t$，故可以看作是归一化常量，使用模型 $p _ {\theta}(x _ t | x _ {t+1})$ 来近似  $q(x _ t | x _ {t+1})$，于是剩下 $\hat q (y|x _ t)$，训练一个分类器 $p _ {\phi}(y|x _ t)$ 来近似，其输入可以通过对 $q(x _ t)$ 采样得到。



