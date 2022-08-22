---
title: Noise Conditional Score Network
date: 2022-07-22 18:12:18
tags: 
    - scored based model
    - generative model
mathjax: true
---

论文：[Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600)

代码：[ermongroup/ncsn](https://github.com/ermongroup/ncsn)

# 1. 基于 Score 的生成模型

记数据集 $\mathbf x_{1:N}$ 中每个样本 $\mathbf x_i \in \mathbb R^D$，来自某未知分布 $p_{data}(\mathbf x)$，定义概率密度 $p(\mathbf x)$ 的 score 为 $\nabla_{\mathbf x} \log p(\mathbf x)$，得分网络为 $\mathbf s_{\theta}: \mathbb R^D \rightarrow \mathbb R^D$，此神经网络被训练来模拟 $p_{data}(\mathbf x)$ 的 score。目的是利用此模型生成来自 $p_{data}(\mathbf x)$ 的样本。

## 1.1 score matching

score matching 原本是设计用来学习非归一化统计模型（非归一化概率密度）。这里我们训练得分网络 $\mathbf s_{\theta}(\mathbf x)$ 来估计 $\nabla_{\mathbf x} \log p_{data}(\mathbf x)$。训练目标为

$$\min_{\theta} \ \frac 1 2 \mathbb E_{p_{data}}[\|\mathbf s_{\theta}(\mathbf x)-\nabla_{\mathbf x} \log p_{data}(\mathbf x)\|_2^2] \tag{1}$$

(1) 式等价于 (2) 式

$$\min_{\theta} \ \mathbb E_{p_{data}} [tr(\nabla_{\mathbf x} \mathbf s_{\theta}(\mathbf x)) + \frac 1 2 \|\mathbf s_{\theta}(\mathbf x)\|_2^2] \tag{2}$$

证明过程可参考 [Score Matching](/2022/07/09/diffusion_model/score_match) 。

使用自动求导框架如 PyTorch 以及 SGD 可以轻松的求得最优解  $\theta^{\star}$  。最优解时，$\mathbf s_{\theta^{\star}}(\mathbf x) = \nabla_{\mathbf x} \log p_{data} (\mathbf x)$，也就是说，**模型近似为真实分布的得分函数**。


但是在网络深度很大以及数据维度 $D$ 很大时，$tr(\nabla_{\mathbf x} \mathbf s_{\theta}(x))$ 计算开销较大，所以此时 score matching 方法不太适合。下面讨论两种解决方法。

**去噪 score matching**

去噪 score matching 避开计算 $tr(\nabla_{\mathbf x} \mathbf s_{\theta}(x))$ 。首先对数据 $\mathbf x$ 进行噪声扰动，扰动过程 $q_{\sigma}(\tilde {\mathbf x}|\mathbf x)$ 事先设计好的已知分布，那么优化目标变为

$$\frac 1 2 \mathbb E_{q_{\sigma}(\tilde {\mathbf x}|\mathbf x), p_{data}(\mathbf x)}[\|\mathbf s_{\theta}(\tilde {\mathbf x})-\nabla_{\tilde {\mathbf x}} \log q_{\sigma} (\tilde {\mathbf x}|\mathbf x) \|_2^2] \tag{3}$$

根据以下推导

$$\begin{aligned}\mathbb E_{\tilde {\mathbf x}}[\nabla_{\tilde {\mathbf x}} \log q_{\sigma}(\tilde {\mathbf x})]&=\int_{\tilde {\mathbf x}} \int_{\mathbf x} \nabla_{\tilde {\mathbf x}} q_{\sigma}(\tilde {\mathbf x}|\mathbf x) p_{data}(\mathbf x) d\mathbf x d\tilde {\mathbf x}
\\&=\int_{\tilde {\mathbf x}} \int_{\mathbf x} \nabla_{\tilde {\mathbf x}} \log q_{\sigma}(\tilde {\mathbf x}|\mathbf x) \cdot q_{\sigma}(\tilde {\mathbf x}|\mathbf x) p_{data}(\mathbf x) d\mathbf x d\tilde {\mathbf x}
\\&= \int_{\tilde {\mathbf x},\mathbf x} q_{\sigma}(\tilde {\mathbf x}, \mathbf x) \nabla_{\tilde {\mathbf x}} \log q_{\sigma}(\tilde {\mathbf x}|\mathbf x)d\mathbf x d\tilde {\mathbf x}
\\&= \mathbb E_{\tilde {\mathbf x},\mathbf x} [\nabla_{\tilde {\mathbf x}} \log q_{\sigma}(\tilde {\mathbf x}|\mathbf x)]
\end{aligned}$$

（上式中 $\mathbb E_{\tilde {\mathbf x}}[\cdot]$ 表示是 over $q_{\sigma}(\tilde {\mathbf x})$ 上求 $[\cdot]$ 的期望，类似地，$\mathbb E_{\tilde {\mathbf x},\mathbf x}[\cdot]$ 是 over $q_{\sigma}(\tilde {\mathbf x},\mathbf x)=q_{\sigma}(\tilde {\mathbf x}|\mathbf x) p_{data}(\mathbf x)$ 上求 $[\cdot]$ 的期望）

可知 (3) 式的最优解满足： $\mathbf s_{\theta^{\star}}(\mathbf x)=\nabla_{\mathbf x} \log q_{\sigma}(\mathbf x)$ 几乎可以确定是成立的（依概率相等）。另一方面，我们需要扰动必须足够小（$\sigma$ 足够小），此时才有 $\mathbf s_{\theta^{\star}}(\mathbf x) = \nabla_{\mathbf x} \log q_{\sigma} (\mathbf x)\approx  \nabla_{\mathbf x} \log p_{data} (\mathbf x)$，因为 $q_{\sigma} (\tilde {\mathbf x}) := \int q_{\sigma}(\tilde {\mathbf x}|\mathbf x) p_{data}(\mathbf x) d\mathbf x$ ，在 $\sigma$ 足够小时有 $q_{\sigma} (\mathbf x) \approx p_{data}(\mathbf x)$。

特别地，当 $\sigma \rightarrow 0$ 时，$q_{\sigma}(\tilde {\mathbf x}|\mathbf x)$ 变成狄拉克分布，满足 $q_{\sigma}(\tilde {\mathbf x}=\mathbf x|\mathbf x)=1$，此时 $q_{\sigma} (\tilde {\mathbf x}) = \int q_{\sigma}(\tilde {\mathbf x}|\mathbf x) p_{data}(\mathbf x) d\mathbf x=p_{data}(\tilde {\mathbf x})$。

**sliced score matching** 

参考 [Sliced Score Matching](/2022/07/09/diffusion_model/score_estimation)

## 1.2 Langevin dynamics

上一小节讲述了根据目标函数来训练模型。模型训练好之后，还需要搞清楚如何采样生成图像样本。

Langevin dynamics 可以根据得分函数 $\nabla_{\mathbf x} \log p(\mathbf x)$ 生成来自概率密度 $p(\mathbf x)$ 的样本。给定一个固定 step size $\epsilon > 0$，以及一个初始值 $\tilde {\mathbf x}_0 \sim \pi(\mathbf x)$，这里 $\pi(\mathbf x)$ 是一个先验分布，那么 Langevin 方法递归的计算下式，

$$\tilde {\mathbf x}_t = \tilde {\mathbf x}_{t-1}+\frac {\epsilon} 2 \nabla_{\mathbf x} \log p(\tilde {\mathbf x}_{t-1}) + \sqrt {\epsilon} \mathbf z_t \tag{4}$$

其中 $\mathbf z_t \sim \mathcal N(0, I)$，$\nabla_{\mathbf x} \log p(\tilde {\mathbf x}_{t-1})$ 使用 $\mathbf s_{\theta^{\star}}(\tilde {\mathbf x}_{t-1})$。

当 $\epsilon \rightarrow 0, \ T \rightarrow \infty$ 时，$\tilde {\mathbf x}_T \sim p(\mathbf x)$ 。

当 $\epsilon > 0, \ T < \infty$ 时，使用 [Metropolis-Hasting](/2022/06/10/ml/sampling) 更新对 (4) 式进行修正，但是实际应用中，往往不需要这个修正，我们尽量使得 $\epsilon$ 小 且 $T$ 大即可。

以上就是根据 score matching 得到概率得分函数 $\mathbf s_{\theta^{\star}}(\mathbf x)=\nabla_{\mathbf x} \log q_{\sigma}(\mathbf x)$，然后根据 Langevin dynamics 进行采样。作者称基于这个思想而提出的框架为基于 score 的生成模型。

# 2. 基于 score 的生成模型挑战

基于 score 的生成模型面临以下两个问题。

## 2.1 manifold 假设

manifold 假设声称现实世界的数据趋于集中在高维空间中的低维 manifolds 上。根据这个假设，基于 score 的生成模型的问题包括：

1. $\nabla_{\mathbf x} \log p_{data}(\mathbf x)$ 求导是在高维空间中进行的，当数据 $\mathbf x$ 被限制在某个低维 manifold 上时，这个导数会出现未定义情况（不可求导）。

2. (2) 式仅在数据分布的支持集为全空间时可以得到一致的得分估计器（得分网络），若数据仅位于某低维 manifold 上，则非一致。

如图 1，实验使用 ResNet 见论文附录 B.1，数据集使用 CIFAR-10，图像像素归一化到 `[0,1]` 区间，估计得分结果，左侧是使用 Sliced score matching 的损失与迭代次数的关系图，右侧是对数据进行了一个小的高斯扰动 $\mathcal N(0,0.0001)$ 后的关系图，可见右侧图逐渐收敛，而左侧图则振荡。

![](/images/diffusion_model/NCSN_1.png)
<center>图 1.</center>

**总结：若数据位于低维 manifold 上，可使用小的高斯扰动解决。**

## 2.2 低数据密度区域

在低数据密度区域，数据样本将会非常少，这会导致 score 估计以及 Langevin dynamics 的采样。

### 2.2.1 不精确的 score 估计

低数据密度区域，数据样本很少不足以准确地估计位于此区域的得分。以一个高斯混合为例： $p_{data}=\frac 1 5 \mathcal N((-5,-5),I)+\frac 4 5 \mathcal N((5,5),I)$，具体实验细节见论文附录 B.1，使用 sliced score matching 进行得分估计，结果如图 2，左侧是使用真实地概率分布得分 $\nabla_{\mathbf x} \log p_{data}(\mathbf x)$，右侧是使用 $\mathbf s_{\theta}(\mathbf x)$ （根据 $p_{data}(\mathbf x)$ 采样得到训练集，然后训练得分网络得到得分估计），图 2 是橙色图，深色表示高概率密度，红色框区域有 $\nabla_{\mathbf x} \log p_{data}(\mathbf x) \approx \mathbf s_{\theta}(\mathbf x)$，中间部分由于是低密度区域，两个得分图在这个区域差别较大。

![](/images/diffusion_model/NCSN_2.png)
<center>图 2.</center>

### 2.2.2 Langevin dynamics 混合速度慢

如果数据分布的两个 mode 被低密度区域分隔，那么 Langevin dynamics 无法在合理的时间内正确地得到这两个 mode 的相关权重。

考虑一个混合概率分布 $p_{data}(\mathbf x)=\pi p_1(\mathbf x) + (1-\pi)p_2(\mathbf x)$ ，其中 $p_1(\mathbf x), \ p_2(\mathbf x)$ 均为归一化的概率分布，各自的支持集不同（没有交集），$\pi \in (0,1)$。 在 $p_1(\mathbf x)$ 的支持集中，$\nabla_{\mathbf x} p_{data}(\mathbf x)=\nabla_{\mathbf x}(\log \pi + \log p_1(\mathbf x)=\nabla_{\mathbf x} \log p_1(\mathbf x)$，同样地在 $p_2(\mathbf x)$ 的支持集中，$\nabla_{\mathbf x} p_{data}(\mathbf x)=\nabla_{\mathbf x}(\log (1-\pi) + \log p_2(\mathbf x)=\nabla_{\mathbf x} \log p_2(\mathbf x)$，这两种情况，得分 $\nabla_{\mathbf x} p_{data}(\mathbf x)$ 均不依赖 $\pi$，而 Langevin dynamics 根据 $\nabla_{\mathbf x} p_{data}(\mathbf x)$ 进行采样，所以采样也不依赖 $\pi$。实际应用中，如果两个支持集仅有低密度区域连接（交集为低密度区域），那么上述分析依然近似有效，在这种情况下 Langevin dynamics 理论虽然可以生成正确的样本，但是需要非常小的 step 值和非常大的 step 数量。

还以上面的高斯混合分布为例说明， $p_{data}(\mathbf x)=\pi p_1(\mathbf x) + (1-\pi)p_2(\mathbf x)$，如图 3，(a) 图是直接从 $p_{data}(\mathbf x)$ 进行采样。(b) 是使用 Langevin dynamics 生成样本。(c) 是使用退火 Langevin dynamics 进行采样。显然图 (b) 的两个 mode 的 weight 与实际 weight 不符合，采样不准确。

![](/images/diffusion_model/NCSN_3.png)
<center>图 3.</center>

**总结：低密度区域的缺点：1. score 估计不准确；2. Langevin dynamics 采样不准确。解决方法：引入大噪声**。

# 3. NCSN：学习和推断

noise conditional score networks

对数据进行高斯噪声扰动，使得： 1. 数据不再被限制在低维 manifold 上；2. 大的高斯噪声可以填充低密度区域。

使用多个噪声级别（$\sigma$ 大小不同，从大到小）可以获得一系列的经噪声扰动后的分布，这个序列收敛于真实的数据分布，而且获得的经噪声扰动后的分布序列可以对 Langevin dynamics 采样进行模拟退火。

## 3.1 NCSN

记噪声幅度序列为 $\{\sigma_i\}_{i=1}^L$，且满足 $\sigma_1>\cdots > \sigma_L > 0$。经过扰动后的数据分布为 $q_{\sigma}(\mathbf x)=\int p_{data}(\mathbf t) \mathcal N(\mathbf x|\mathbf t, \sigma^2 I) d\mathbf t$。

选择 $\sigma_i$ 使得 $\sigma_1$ 足够大，以解决低密度区域带来的问题，且 $\sigma_L$ 足够小，以降低对原生数据分布的影响。

模型的输入输出维度相同，这类似于图像分割任务，作者使用了带 膨胀/空洞卷积的 U-Net 网络，并使用了 instance normalization。

## 3.2 利用 score matching 学习 NCSN

sliced 和 denoising score matching 均可来训练 NCSN。作者实验了 denoising score matching，因为这个方法更快，并且恰好有高斯扰动，这也是我们所需要的。

高斯噪声扰动为 $q_{\sigma}(\tilde {\mathbf x}|\mathbf x)=\mathcal N(\tilde {\mathbf x}|\mathbf x, \sigma^2I)$，于是 $\nabla_{\tilde {\mathbf x}} \log q_{\sigma}(\tilde {\mathbf x}|\mathbf x) = (\mathbf x-\tilde {\mathbf x})/\sigma^2$ 。给定某个 $\sigma$ 值，那么优化目标 (3) 式变为

$$l(\theta;\sigma) \stackrel{\Delta}= \frac 1 2 \mathbb E_{p_{data}(\mathbf x)} \mathbb E_{\tilde {\mathbf x} \sim \mathcal N(\mathbf x, \sigma^2I)} \left[\begin{Vmatrix} \mathbf s_{\theta}(\mathbf x,\sigma)+\frac {\tilde {\mathbf x}-\mathbf x}{\sigma^2}\end{Vmatrix}_2^2\right] \tag{5}$$

其中 $\mathbf s_{\theta}(\mathbf x,\sigma)$ 其实就是指 $\mathbf s_{\theta}(\tilde {\mathbf x})$，因为 $\tilde {\mathbf x}$ 与 $\mathbf x$ 和 $\sigma$ 相关。对所有的噪声幅度序列 $\{\sigma_i\}_{i=1}^L$，训练目标为

$$\mathcal L(\theta;\{\sigma_i\}_{i=1}^L)=\frac 1 L \sum_{i=1}^L \lambda(\sigma_i) l(\theta; \sigma_i) \tag{6}$$

其中 $\lambda(\sigma_i)$ 是系数函数。如果模型 $\mathbf s_{\theta}(\mathbf x,\sigma)$ 的容量够大，那么 (6) 式最优解满足 $\mathbf s_{\theta^{\star}}(\mathbf x,\sigma)=\nabla_{\mathbf x} \log q_{\sigma_i}(\mathbf x), \forall i \in \{1,\ldots,L\}$ 。

理想情况下，我们希望所有的 $\lambda(\sigma_i) l(\theta; \sigma_i)$ 的量级相当。根据经验，模型训练到最优时，近似关系 $\|\mathbf s_{\theta}(\mathbf x,\sigma)\|_2 \propto 1/\sigma$，故 $l(\theta;\sigma) \propto 1/\sigma^2$，所以可以选择 

$$\lambda(\sigma)=\sigma^2 \tag{7}$$

此时 $\lambda(\sigma) l(\theta; \sigma)$ 不依赖于 $\sigma$ 。

## 3.3 NCSN 推断

当模型 $\mathbf s_{\theta}(\mathbf x, \sigma)$ 训练完成，作者提出一个样本生成方法——退火 Langevin dynamics，如算法 1 所示

---
**算法 1 退火 Langevin dynamics**
输入： $\{\sigma_i\}_{i=1}^L$ 是递减序列， Langevin dynamics 迭代的 step 值 $\epsilon$，以及 step 数量 $T$
初始化：$\tilde {\mathbf x}_0 \sim \pi(\tilde {\mathbf x})$  来自先验分布

**for** $i=1,\ldots, L$ **do**
&emsp;    $\alpha_i = \epsilon \cdot \sigma_i^2 /\sigma_L^2$
    &emsp;**for** $t=1,\ldots, T$ **do**
    &emsp; &emsp; 采样 $\mathbf z_t \sim \mathcal N(0,I)$
    &emsp; &emsp; $\tilde {\mathbf x}_t = \tilde {\mathbf x}_{t-1}+\frac {\alpha_i} 2 \mathbf s_{\theta}(\tilde {\mathbf x}_{t-1}, \sigma_i) + \sqrt {\alpha_i} \mathbf z_t$
&emsp; **end for**
&emsp; $\tilde {\mathbf x}_0 = \tilde {\mathbf x}_T$
**end for**
**return** $\tilde {\mathbf x}_T$

---

# 4. 实验

**setup**

1. 使用 MNIST, CelebA 以及 CIFAR-10 常见数据集，其中 CelebA 数据集需要先 center-cropped 为 `140x140` 大小然后 resized 为 `32x32`。

2. 所有图像数据归一化到 [0, 1] 区间。

3. 选取 $L=10$ 个不同的标准差 $\sigma$，从 $\sigma_1=1$ 逐渐降低到 $\sigma_{10}=0.01$ 。

4. 退火 Langevin dynamics 算法中，$T=100$，$\epsilon=2 \times 10^{-5}$ ，先验分布使用 均匀分布。

5. $\sigma$ 序列设置为 

    ```python
    sigmas = torch.tensor(
        np.exp(np.linspace(np.log(self.config.model.sigma_begin), 
                           np.log(self.config.model.sigma_end),
                           self.config.model.num_classes))
    ).float().to(self.config.device)
    ```



# 5. 附录

## 5.1 架构

NCSN 使用了 instance normalization，膨胀卷积以及 U-Net 网络。

### 5.1.1 instance normalization

instance normalization （IN）采用单样本单通道（$H \times W$）的归一化。

这里作者使用 conditional instance normalization，模型 $\mathbf s_{\theta}(\mathbf x,\sigma)$ 在预测得分的时候需要考虑 $\sigma$，即每个 $\sigma_i$ 都有各自的 scale 和 bias （不在 $\sigma_i$ 之间共享）。

假设 $\mathbf x$ 通道数（特征面数量）为 $C$，对第 $k$ 个 feature map，记 $\mu_k, \ s_k$ 分别表示相应的 mean 和 标准差，那么 conditional instance normalization 计算方法为

$$\mathbf z_k = \gamma[i,k] \frac {\mathbf x_k-\mu_k}{s_k} + \beta[i,k] \tag{8}$$

其中 $\gamma \in \mathbb R^{L \times C}$ 和 $\beta \in \mathbb R^{L \times C}$ 是可学习参数。$k$ 为 feature map 索引，$i$ 表示 $\sigma$ 序列中的 $\sigma_i$ 的下标，一共 $L$ 个不同的 $\sigma$ 值和 $C$ 个 channels。

然而，部分人认为 IN 不妥，原因是 IN 完全去掉了特征平面中的 $\mu_k$ 信息，导致生成的样本图像中颜色的偏移（shift）。为了解决这个问题，作者做了简单的修改，计算 $\mu_k$ 的 mean 和标准差，分别记为 $m, \ v$，增加一个可学习参数 $\alpha \in \mathbb R^{L \times C}$，那么修改后的 conditional instance normalization 为

$$\mathbf z_k = \gamma[i,k] \frac {\mathbf x_k-\mu_k}{s_k} + \beta[i,k]+\alpha[i,k] \frac {\mu_k - m}{v} \tag{9}$$

代码：

```python
class ConditionalInstanceNorm2dPlus(nn.Module):
    def __init__(self, num_features, num_classes, bias=True):
        '''
        num_features: C, number of x channels
        num_classes: L=10
        '''
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
        if bias:    # with beta, (8) 和 (9) 式中的 beta 项
            # 顺序为 gamma, alpha, beta，beta 为 bias 零值初始化
            self.embed = nn.Embedding(num_classes, num_features * 3)
            self.embed.weight.data[:, :2 * num_features].normal_(1, 0.02)
            self.embed.weight.data[:, 2 * num_features:].zero_() 
        else:       # without beta
            # 顺序为 gamma, alpha
            self.embed = nn.Embedding(num_classes, 2 * num_features)
            self.embed.weight.data.normal_(1, 0.02)

    def forward(self, x, y):
        '''
        x: 输入特征 (batch_size, C, H, W)
        y: denoising过程中的level，取值范围 [0,L-1]，shape 为 (batch_size)
        '''
        # 计算输入各 channel 的均值 u_k，shape 为 (batch_size, C)
        means = torch.mean(x, dim=(2, 3))
        # 计算 u_k 的均值 m，shape 为 (batch_size, 1)
        m = torch.mean(means, dim=-1, keepdim=True)
        # 计算 u_K 的标准差 v，shape 为 (batch_size, 1)
        v = torch.var(means, dim=-1, keepdim=True)
        means = (means - m) / (torch.sqrt(v + 1e-5))    # (u_k-m)/v
        # # 计算 instance_norm：h_k = (x_k-u_k)/s_k
        h = self.instance_norm(x)   

        # 代码中的 gamma_c 与(9)式中的 gamma_e 的关系 gamma_c:=gamma_e/alpha
        # beta 相同
        if self.bias:
            gamma, alpha, beta = self.embed(y).chunk(3, dim=-1)
            h = h + means[..., None, None] * alpha[..., None, None]
            out = gamma.view(-1, self.num_features, 1, 1) * h + beta.view(-1, self.num_features, 1, 1)
        else:
            gamma, alpha = self.embed(y).chunk(2, dim=-1)
            h = h + means[..., None, None] * alpha[..., None, None]
            out = gamma.view(-1, self.num_features, 1, 1) * h
        return out
```

对应的 layer 称为 CondInstanceNorm++，作者再每个 conv 和 pooling layer 之前均增加 CondInstanceNorm++ 。


**目标函数**

这里再次列出目标函数的数学表达式，根据 (5)~(7) 式，可知

$$\mathcal L(\theta;\{\sigma_i\}_{i=1}^L)=\frac 1 L \sum_{i=1}^L  \sigma_i^2 \frac 1 2 \mathbb E_{p_{data}(\mathbf x)} \mathbb E_{\tilde {\mathbf x} \sim \mathcal N(\mathbf x, \sigma^2I)} \left[\begin{Vmatrix} \mathbf s_{\theta}(\mathbf x,\sigma)+\frac {\tilde {\mathbf x}-\mathbf x}{\sigma^2}\end{Vmatrix}_2^2\right] \tag{10}$$



其中根据 $q_{\sigma}(\tilde {\mathbf x}|\mathbf x)=\mathcal N(\tilde {\mathbf x}|\mathbf x, \sigma^2 I)$，可知

$$\nabla_{\tilde {\mathbf x}} \log q_{\sigma} (\tilde {\mathbf x}|\mathbf x)=-\frac {\tilde {\mathbf x} - \mathbf x}{\sigma^2} \tag{11}$$

$\mathbf s_{\theta}(\tilde {\mathbf x}, \sigma)$ 是模型输出。

代码：

```python
# X: mini batch normalized image data, (batch_size, 3, H, W)
# sigmas: L=10 noise levels, 见上面 sigmas 赋值的代码片段
# 对这个 mini batch 中的每个数据 x，随机选取 noise level, (batch_size,)
labels = torch.randint(0, len(sigmas), (X.shape[0],), device=X.device)
# 抽取 noise level 对应的 sigmas，(batch_size, 1, 1, 1)
used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
# 噪声扰动后的数据，\tilde x = x + sigma * N(0, 1)
perturbed_samples = samples + torch.randn_like(samples) * used_sigmas
# target，见 (11) 式, (batch_size, 3, H, W)
target = - 1 / (used_sigmas ** 2) * (perturbed_samples - samples)
# predict，s_{\theta}(\tilde x, \sigma), UNet, out shape: (batch_size, 3, H, W)
scores = scorenet(perturbed_samples, labels)
target = target.view(target.shape[0], -1)   # (batch_size, 3HW)
scores = scores.view(scores.shape[0], -1)   # (batch_size, 3HW)
# (10) 式，loss shape (batch_size,)
loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power

return loss.mean(dim=0)
```

**UNet 结构**

![](/images/diffusion_model/unet_1.png)

<center>图 1. 每个蓝色矩形表示 feature maps。白色矩形表示 feature maps 的拷贝。箭头表示操作</center>

图 1 中，输入 channel 为 1 或 3（对应灰度图和彩色图），首先经一个 Conv2d 将 channel 转为 64，然后再经两个 conv 操作，故图 1 中左上角少了一个箭头和一个矩形框。

由于实验所用的数据集均为小尺寸图，故 feature size 与图 1 中不同。

没有使用 maxpooling 下采样，为了增大卷积视野，使用膨胀卷积，或者 `ConvMeanPool`，达到类似效果。

`conv`：(CondInstanceNorm++,ELU,Conv2d,CondInstanceNorm++,ELU,Conv2d)+shortcut

```python
self.norm1 = ConditionalInstanceNorm2dPlus(input_dim, num_classes)  # (C, L)
self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1)
self.norm2 = ConditionalInstanceNorm2dPlus(output_dim, num_classes)
self.conv2 = nn.Conv2d(output_dim, output_dim, 3, 1, 1)
self.act = nn.ELU()
if output_dim != input_dim:
    self.shortcut = partial(ConvMeanPool, kernel_size=1)

def forward(self, x, y):
    '''
    x: input data, (batch_size, C, H, W)
    y: noise level, (batch_size,)
    ''' 
    output = self.norm1(x, y)   # Conditional Instance Norm, (9) 式
    output = self.act(output)
    output = self.conv1(output)
    output = self.norm2(output, y)
    output = self.act(output)
    output = self.conv2(output)

    shortcut = x if self.output_dim == self.input_dim else self.shortcut(x)

    return shortcut + output

class ConvMeanPool(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True):
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, padding=kernel_size//2, bias=biases)
    def forward(self, inputs):
        output = self.conv(inputs)
        output = sum(
            [output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]
        ) / 4
        return output
```