---
title: CGANs with Projection Discriminator
date: 2022-08-18 16:20:05
tags: GAN
---

论文：[CGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637)

# 1. 简介

传统的 CGAN，是将分类信息 $\mathbf y$ (例如 one-hot vector) 和输入数据 $\mathbf x$ concatenate 起来，作为网络的输入，或者将 $\mathbf y$ 与网络中间某个 layer 的输出特征 concatenate 起来，作为下一 layer 的输入。作者则考虑到，对判别器所作的任何假设均可视作是对判别器进行正则化，于是提出了一种特殊的判别器，其条件分布 $p(\mathbf y|\mathbf x)$ 是离散型或单峰连续型，基于这个假设，我们需要在判别器结构中实现 class condition vector $\mathbf y$ 与特征向量的内积操作，如图 1 (d)，下文会解释为何要做这一操作。

![](/images/gan/cgan_proj_1.png)

图 1.

# 2. 框架

输入数据记为向量 $\mathbf x$，条件信息记为向量 $\mathbf y$，例如分类 $\mathbf y$ 为 one-hot 向量。判别器记为 

$$D(\mathbf x,\mathbf y;\theta):=\mathcal A(f(\mathbf x, \mathbf y;\theta)) \tag{1}$$

其中 $f$ 是变换函数，参数为 $\theta$，$\mathcal A$ 是激活函数。

分别使用 q 和 p 表示真实分布和模型输出分布。标准的对抗损失为

$$\min_D L(D)=-E_{q(\mathbf y)}[E_{q(\mathbf x|\mathbf y)}[\log D(\mathbf x,\mathbf y)]] - E_{p(\mathbf y)}[E_{p(\mathbf x|\mathbf y)}[\log (1-D(\mathbf x, \mathbf y))]] \tag{2}$$

(2) 式表示对判别器而言，对于来自真实分布的 $(\mathbf x, \mathbf y)\sim q(\mathbf x,\mathbf y)$，判别器输出的负对数概率越小越好，而对来自生成器 G 输出分布的数据 $(\mathbf x, \mathbf y)\sim p(\mathbf x,\mathbf y)$，判别器输出的负对数概率越大越好，即 $\log (1-D(\mathbf x,\mathbf y))$ 越小越好。

判别器中，最后的激活函数使用 sigmoid ，使得输出位于 $(0,1)$ 之间，从而表示概率。

传统上，将 $\mathbf y$ 与输入 $\mathbf x$ 或网络中间某个特征进行 concatenate。作者观察目标函数 (2) 式的最优解的形式，提出一种新型的 condition 信息融合方式。

考虑如下对数似然比率，

$$f^{\star}(\mathbf x, \mathbf y)=\log \frac {q(\mathbf x|\mathbf y)q(\mathbf y)}{p(\mathbf x|\mathbf y)p(\mathbf y)}=\log \frac {q(\mathbf y|\mathbf x)}{p(\mathbf y|\mathbf x)}+\log \frac {q(\mathbf x)}{p(\mathbf x)}:=r(\mathbf y|\mathbf x)+r(\mathbf x) \tag{3}$$

那么根据 sigmoid 函数有

$$\sigma(f(\mathbf x,\mathbf y))=\frac {e^{f(\mathbf x,\mathbf y)}} {e^{f(\mathbf x,\mathbf y)}+1}=\frac {\frac {q(\mathbf x,\mathbf y)}{p(\mathbf x,\mathbf y)}}{\frac {q(\mathbf x,\mathbf y)}{p(\mathbf x,\mathbf y)}+1}=\frac {q(\mathbf x,\mathbf y)}{q(\mathbf x,\mathbf y)+p(\mathbf x,\mathbf y)}$$

[GAN](gan/2019/07/23/gan) 中已经分析出 $\sigma(f(\mathbf x,\mathbf y))$ 就是 (2) 式目标函数的最优解。本文则直接使用 $f(\mathbf x,\mathbf y)$ 代替传统的 $D(\mathbf x,\mathbf y)$，显然 $D(\mathbf x,\mathbf y)$ 随着 $f(\mathbf x,\mathbf y)$ 单调递增，但是取值范围从 $(0,1)$ 扩展到整个实数域 $\mathbb R$ 。

作者假设 $p(\mathbf y|\mathbf x)$ 和 $q(\mathbf y|\mathbf x)$ 均为简单的分布如高斯分布。

对数似然比 $r(\mathbf y|\mathbf x)$ 和 $r(\mathbf x)$ 分别使用带参函数 $f_1$ 和 $f_2$ 描述，

$$f(\mathbf x,\mathbf y;\theta):=f_1(\mathbf x,\mathbf y;\theta)+f_2(\mathbf x,\mathbf y;\theta)=\mathbf y^{\top}V \phi(\mathbf x;\theta_{\phi})+\psi(\phi(\mathbf x;\theta_{\phi});\theta_{\psi}) \tag{4}$$

函数 $f_1, \ f_2$ 具有 (4) 式中的形式，对应图 1 (d)。$V$ 是 $\mathbf y$ 的嵌入矩阵，即从 one-hot 向量 $\mathbf y$ 到对应的 embedding vector，然后这个 embedding vector 再与中间层的输出作内积操作。

$\psi(\cdot; \theta_{\psi})$ 是标量函数。可学习参数为 $\theta=\{V, \theta_{\phi}, \theta_{\psi}\}$。

下面分析 (4) 式的作用。

# 3. Projection 判别器

在某些正则假设下，判别器的目标函数的最优解形式如 (4) 式。

## 3.1 离散型 y

$\mathbf y$ 为离散变量。

首先考虑 $y$ 是一个分类变量，分类数量为 $C$，那么 $p(y|\mathbf x)$ 最常见的是对数线性模型，我们就从这个 **对数线性模型** 所表示的条件分布 $p(y|\mathbf x)$ 出发，开始如下分析。

$$\log p(y=c|\mathbf x):=\mathbf v_c^{p\top} \phi(\mathbf x) - \log Z(\phi(\mathbf x)) \tag{5}$$

其中 

$$Z(\phi(\mathbf x)) := \left( \sum_{j=1}^C \exp (\mathbf v_j^{p\top}\phi(x))\right) \tag{6}$$

是配分函数，用作概率的归一化。

$\phi: \mathbf x \rightarrow \mathbb R^{d^L}$ 表示网络，最后一层输出 units 为 $d^L$，$L$ 表示网络中 layer 数量。$\mathbf v_c^p$ 表示分类 $c$ 对应的权重向量，上标 $p$ 表示是与模型输出相关（与真实分布相关的分类权重记为 $\mathbf v_c^q$ ），$\phi(\mathbf x)$ 表示特征向量。

于是有

$$\log \frac {q(y=c|\mathbf x)}{p(y=c|\mathbf x)}=(\mathbf v_c^q-\mathbf v_c^p)^{\top} \phi(\mathbf x) - (\log Z^q(\phi(\mathbf x))-\log Z^p(\phi(\mathbf x))) \tag{7}$$

我们使用一个明确向量 $\mathbf v_c:=\mathbf v_c^q - \mathbf v_c^p$ 来记录两个隐向量 $\mathbf v_c^q$ 和 $\mathbf v_c^p$ 之差，并记 $f_1(\mathbf x,y=c)=\mathbf v_c^{\top} \phi(\mathbf x)$，以及令 $\psi(\phi(\mathbf x)):= -(\log Z^q - \log Z^p) + r(\mathbf x)$，那么对比 (3) 式和 (7) 式 有，

$$f(\mathbf x; y=c):=\mathbf v_c^{\top} \phi(\mathbf x)+\psi(\phi(\mathbf x)) \tag{8}$$

令 $\mathbf y$ 表示 one-hot 分类向量，以及 $V$ 表示权重（真实分布与模型分布权重之差）矩阵，那么

$$f(\mathbf x, \mathbf y)=\mathbf y^{\top} V \phi(\mathbf x)+\psi(\phi(\mathbf x)) \tag{9}$$

$V$ 的每一行表示一个分类的两种分布的权重之差。

## 3.2 连续型 y

$\mathbf y \in \mathbb R^d$ 为连续型，$p(\mathbf y|\mathbf x)$ 为单峰连续型分布，我们假设 $q(\mathbf y|\mathbf x)$ 和 $p(\mathbf y|\mathbf x)$ 为高斯分布，因为高斯分布比较简单常见，且满足单峰连续型，于是记

$$q(\mathbf y|\mathbf x)=\mathcal N(\mathbf y|\mu_q(\mathbf x), \Lambda_q^{-1})
\\ p(\mathbf y|\mathbf x)=\mathcal N(\mathbf y|\mu_p(\mathbf x), \Lambda_p^{-1})$$

计算 (3) 式中的 $r(\mathbf y|\mathbf x)$ 为

$$\begin{aligned} r(\mathbf y|\mathbf x)&=\log \frac {q(\mathbf y|\mathbf x)}{p(\mathbf y|\mathbf x)}
\\&=\log \left(\sqrt {\frac {|\Lambda_q|}{|\Lambda_p|}} \frac {\exp(-\frac 1 2 (\mathbf y-\mu_q(\mathbf x))^{\top}\Lambda_q(\mathbf y-\mu_q(\mathbf x)))}{\exp(-\frac 1 2 (\mathbf y-\mu_p(\mathbf x))^{\top}\Lambda_p(\mathbf y-\mu_p(\mathbf x)))} \right)
\\&=-\frac 1 2 \mathbf y^{\top}(\Lambda_q-\Lambda_p)\mathbf y + \mathbf y^{\top}(\Lambda_q \mu_q(\mathbf x)-\Lambda_p \mu_p(\mathbf x))+\psi(\phi(\mathbf x))
\\&=-\frac 1 2 \mathbf y^{\top}(\Lambda_q-\Lambda_p)\mathbf y + \mathbf y^{\top}(\Lambda_q W^q-\Lambda_p W^p)\phi(\mathbf x)+\psi(\phi(\mathbf x))
\end{aligned} \tag{10}$$

上式推导，与 $\mathbf y$ 无关的项均合并入 $\psi(\phi(\mathbf x))$ 。

如果假设 $\Lambda_q = \Lambda_p := \Lambda$ ，那么 (10) 中的二次项可以忽略（即忽略第一项），并令 $V=\Lambda_q W^q-\Lambda_p W^p$ ，那么就得到 (4) 的形式。

# 4. 实验

判别器和生成器网络均基于 ResNet，且判别器网络中的权重参数均执行[光谱归一化](gan/2019/07/25/GAN_im) 。作者使用 hinge 版本的目标函数而非直接使用 (2) 式的目标函数，如下

$$\begin{aligned}L(\hat G, D)&= E_{q(y)}[E_{q(\mathbf x|y)}[\max(0,1-D(\mathbf x,y))]]+E_{q(y)}[E_{p(\mathbf z)}[\max (0, 1+D(\hat G(\mathbf z,y), y))]]
\\ L(G, \hat D) &=-E_{q(y)}[E_{p(\mathbf z)}[\hat D(G(\mathbf z,y),y)]]
\end{aligned} \tag{11}$$

当固定生成器 $\hat G$ 时：

对于真实数据 $(\mathbf x,y)$，$D(\mathbf x,y)$ 越大，损失 $L(\hat G, D)$ 越小，当 $D(\mathbf x, y) \ge 1$ 时，对判别器网络参数学习无指导作用。

对于生成样本数据 $(\hat G(\mathbf z, y), y)$ ，$D(\hat G(\mathbf z,y), y)$ 越小，损失 $L(\hat G, D)$ 越小，当 $D(\hat G(\mathbf z,y), y) \le -1$ 时，对判别器网络参数学习无指导作用。

上面损失函数为何要使得 $D(\mathbf x,y) \ge 1$ 以及 $D(\hat G(\mathbf z,y), y) \le -1$ 的时候对判别器网络参数学习无指导作用呢？这是 因为此时判别器的判别效果已经很好了，再继续优化判别器网络参数，收敛速度会非常慢，没有必要，而且太强的判别器会使得生成器性能变差。当然如果实在要加强判别器的判别能力，可以将 (11) 式 $\max$ 函数中的 1 改为更大的一个值。

## 4.1 类条件型图像生成
使用 ImageNet-1k，共 1000 个类，每个类大约 1300 个图片。每个图像压缩为 128x128 大小。网络框架如图 2 所示，

![](/images/gan/cgan_proj_2.png)

图 2.

判别器网络中使用了 projection 结构，将 one-hot vector $\mathbf y$ 的嵌入向量与中间层输出做内积。

为了对比，也使用了 concat 结构的判别器，与 projection 判别器网络结构相同，只是将 projection layer 去掉，然后将 $\mathbf y$ 的嵌入向量（经过 repeat）与第三个 ResBlock 的输出进行 concat。下面使用代码（来自官方代码）说明。

```python
class SNResNetProjectionDiscriminator(chainer.Chain):
    def __init__(self, ch=64, n_classes=0, activation=F.relu):
        super(SNResNetProjectionDiscriminator, self).__init__()
        self.activation = activation
        initializer = chainer.initializers.GlorotUniform()
        with self.init_scope():
            self.block1 = OptimizedBlock(3, ch)
            self.block2 = Block(ch, ch * 2, activation=activation, downsample=True)
            self.block3 = Block(ch * 2, ch * 4, activation=activation, downsample=True)
            self.block4 = Block(ch * 4, ch * 8, activation=activation, downsample=True)
            self.block5 = Block(ch * 8, ch * 16, activation=activation, downsample=True)
            self.block6 = Block(ch * 16, ch * 16, activation=activation, downsample=False)
            self.l7 = SNLinear(ch * 16, 1, initialW=initializer)
            if n_classes > 0:
                self.l_y = SNEmbedID(n_classes, ch * 16, initialW=initializer)

    def __call__(self, x, y=None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.activation(h)
        h = F.sum(h, axis=(2, 3))  # Global pooling
        output = self.l7(h)
        if y is not None:
            w_y = self.l_y(y)
            output += F.sum(w_y * h, axis=1, keepdims=True)
        return output

class SNResNetConcatDiscriminator(chainer.Chain):
    def __init__(self, ch=64, n_classes=0, activation=F.relu, dim_emb=128):
        super(SNResNetConcatDiscriminator, self).__init__()
        self.activation = activation
        initializer = chainer.initializers.GlorotUniform()
        with self.init_scope():
            self.block1 = OptimizedBlock(3, ch)
            self.block2 = Block(ch, ch * 2, activation=activation, downsample=True)
            self.block3 = Block(ch * 2, ch * 4, activation=activation, downsample=True)
            self.l_y = SNEmbedID(n_classes, dim_emb, initialW=initializer)
            self.block4 = Block(ch * 4 + dim_emb, ch * 8, activation=activation, downsample=True)
            self.block5 = Block(ch * 8, ch * 16, activation=activation, downsample=True)
            self.block6 = Block(ch * 16, ch * 16, activation=activation, downsample=False)
            self.l7 = SNLinear(ch * 16, 1, initialW=initializer)

    def __call__(self, x, y=None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)      # (batch_size, 256, H, W)
        if y is not None:
            emb = self.l_y(y)   # (batch_size, 128)
            H, W = h.shape[2], h.shape[3]
            # emb broadcast to (batch_size, 128, H, W)
            emb = F.broadcast_to(
                F.reshape(emb, (emb.shape[0], emb.shape[1], 1, 1)),
                (emb.shape[0], emb.shape[1], H, W))
            h = F.concat([h, emb], axis=1)  # (batch_size, 384, H, W)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.activation(h)
        h = F.sum(h, axis=(2, 3))  # Global pooling
        output = self.l7(h)
        return output
```

生成器网络中，$\mathbf z \sim \mathcal N(0, I)$， $\mathbf y \sim \mathcal U(0, C) \cap \mathbb Z$，这两个变量全部来自采样，shape 分别为 (batch_size, dim_z) 和 (batch_size, C)，其中 dim_z=128 可配置，C 为分类数量，注意 $\mathbf y$ 是 one-hot vector。

生成器网络模型代码为

```python
class ResNetGenerator(chainer.Chain):
    def __init__(self, ch=64, dim_z=128, bottom_width=4, activation=F.relu, n_classes=0, distribution="normal"):
        super(ResNetGenerator, self).__init__()
        initializer = chainer.initializers.GlorotUniform()
        self.bottom_width = bottom_width
        self.activation = activation
        self.distribution = distribution
        self.dim_z = dim_z
        self.n_classes = n_classes
        with self.init_scope():
            self.l1 = L.Linear(dim_z, (bottom_width ** 2) * ch * 16, initialW=initializer)
            self.block2 = Block(ch * 16, ch * 16, activation=activation, upsample=True, n_classes=n_classes)
            self.block3 = Block(ch * 16, ch * 8, activation=activation, upsample=True, n_classes=n_classes)
            self.block4 = Block(ch * 8, ch * 4, activation=activation, upsample=True, n_classes=n_classes)
            self.block5 = Block(ch * 4, ch * 2, activation=activation, upsample=True, n_classes=n_classes)
            self.block6 = Block(ch * 2, ch, activation=activation, upsample=True, n_classes=n_classes)
            self.b7 = L.BatchNormalization(ch)
            self.l7 = L.Convolution2D(ch, 3, ksize=3, stride=1, pad=1, initialW=initializer)

    def __call__(self, batchsize=64, z=None, y=None, **kwargs):
        if z is None:
            z = sample_continuous(self.dim_z, batchsize, distribution=self.distribution, xp=self.xp)
        if y is None:
            y = sample_categorical(self.n_classes, batchsize, distribution="uniform",
                                   xp=self.xp) if self.n_classes > 0 else None
        if (y is not None) and z.shape[0] != y.shape[0]:
            raise ValueError('z.shape[0] != y.shape[0], z.shape[0]={}, y.shape[0]={}'.format(z.shape[0], y.shape[0]))
        h = z
        h = self.l1(h)
        h = F.reshape(h, (h.shape[0], -1, self.bottom_width, self.bottom_width))
        h = self.block2(h, y, **kwargs)
        h = self.block3(h, y, **kwargs)
        h = self.block4(h, y, **kwargs)
        h = self.block5(h, y, **kwargs)
        h = self.block6(h, y, **kwargs)
        h = self.b7(h)
        h = self.activation(h)
        h = F.tanh(self.l7(h))
        return h
```

从上面代码中可见， block2~block6 这 5 个 ResBlock 均融合了 $\mathbf y$ 信息，这与图 2 一致。具体融合方式可查看生成器中 ResBlock 的代码，

```python
class Block(chainer.Chain):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=F.relu, upsample=False, n_classes=0):
        super(Block, self).__init__()
        initializer = chainer.initializers.GlorotUniform(math.sqrt(2))
        initializer_sc = chainer.initializers.GlorotUniform()
        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.n_classes = n_classes
        with self.init_scope():
            self.c1 = L.Convolution2D(in_channels, hidden_channels, ksize=ksize, pad=pad, initialW=initializer)
            self.c2 = L.Convolution2D(hidden_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
            if n_classes > 0:
                self.b1 = CategoricalConditionalBatchNormalization(in_channels, n_cat=n_classes)
                self.b2 = CategoricalConditionalBatchNormalization(hidden_channels, n_cat=n_classes)
            else:
                self.b1 = L.BatchNormalization(in_channels)
                self.b2 = L.BatchNormalization(hidden_channels)
            if self.learnable_sc:
                self.c_sc = L.Convolution2D(in_channels, out_channels, ksize=1, pad=0, initialW=initializer_sc)

    def residual(self, x, y=None, z=None, **kwargs):
        h = x
        h = self.b1(h, y, **kwargs) if y is not None else self.b1(h, **kwargs)
        h = self.activation(h)
        h = upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b2(h, y, **kwargs) if y is not None else self.b2(h, **kwargs)
        h = self.activation(h)
        h = self.c2(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = upsample_conv(x, self.c_sc) if self.upsample else self.c_sc(x)
            return x
        else:
            return x

    def __call__(self, x, y=None, z=None, **kwargs):
        return self.residual(x, y, z, **kwargs) + self.shortcut(x)
```

这里， `n_classes=1000`，估计 BN 使用  conditional batch normalization layer，整个 ResBlock 的结构如图 3 所示，

![](/images/gan/cgan_proj_3.png)

图 3. 

再来详细了解 conditional batch normalization 过程。对于普通的 BN，归一化过程为 

$$y=\frac {x-E[x]} {\sqrt{Var[x]+\epsilon}} \cdot \gamma + \beta$$

其中 $E[x], V[x]$ 表示对 batch features 的某个 channel 求均值和方差，故 $\gamma, \beta$ 都是是长度等于 channel 数量的向量。

条件BN，$\gamma, \beta$ 均为 $C \times N$ 的矩阵，其中 $C, \ N$ 分别表示分类数量和 features 的 channel 数量，所以这里的 条件BN 指，以分类为条件。 

从以上分析可知，生成器中，分类信息 $\mathbf y$ 融合进网络的 BN 层，即 class conditional BN。