---
title: BiCro 论文解读
date: 2023-07-11 17:52:14
tags: image-text retrieve
mathjax: true
---

论文：[BiCro: Noisy Correspondence Rectification for Multi-modality Data via
Bi-directional Cross-modal Similarity Consistency](https://arxiv.org/pdf/2303.12419.pdf)

源码：[xu5zhao/BiCro](https://github.com/xu5zhao/BiCro)

# 1. 简介

image-text pairs 数据集难以收集，且难以准确地对图像描述。从网络上收集的 image-text pairs 虽然很多，但是其中存在不少 mismatched pairs。本文 BiCro（Bidirectional Cross-modal similarity consistency）提出一种方法解决：如何利用 noisy datasets 学习 image-text retrieving。

对于一个数据 pair，可以分为三类：1. well-matched; 2. weakly-matched; 3. mismatched。第一种视作 clean 样本，而后两者则为 noisy 样本。拿到一个 noisy 数据集，如果将其中每个数据 pair 作为 clean 样本（$y=1$），显然不合适，所以想办法对 noisy 样本估计一个 soft label $ 0 \le y < 1$， $y$ 值越大表示 image 与 text 匹配越好。基于此，BiCro 的思想是：相似的图片，应该有相似的文本描述，反之亦然。如图 1，

![](/images/multi_modal/bicro_1.png)
<center>图 1. BiCro 思想说明</center>

图 1 中，将 image 和 text 映射到一个 shared 特征空间。红色表示 clean pairs（圆圈表示 image 特征，三角形表示 text 特征），绿色表示 noisy pairs。如果图 2 中红色是确定的两个 clean pairs，那么绿色一定是 noisy pairs，这是因为：

1. $\tilde I$ 靠近 $I_1$，表示两个 image 相似，但是 $\tilde T$ 却不靠近 $T_1$，这表示相应的 text 不相似。
2. 反过来，$\tilde T$ 靠近 $T_2$，表示两个 text 相似，但是 $\tilde I$ 不靠近 $I_2$，这表示相应的 image 不相似。

使用 BiCro 的基本步骤是：

1. 从给定的 noisy 数据集中选择一个 clean 数据子集作为 anchor points，$y=1$。

2. 不在 clean 数据集中的样本均视作 noisy 样本。借助 anchor points，计算 BiCro 得分作为 noisy 样本的 soft label 值。

3. 前两步得到的两个数据子集合并作为最终的数据集。以 co-teaching 方式训练 cross-modal matching 模型。

# 2. 方法

## 2.1 问题描述

给定一个数据集 $\mathcal D = \lbrace (I_i , T_i , y_i )\rbrace _ {i=1} ^ N$ ，$(I_i, T_i )$ 表示 image-text pair，$y _ i \in \lbrace 0, 1 \rbrace$ 是 label。这里 $y _ i$ 是 hard label，如果 $(I_i, T _ i)$ 是正相关，那么 $y _ i = 1$，否则 $y _ i = 0$ 。

cross-modal matching 模型将两种模态（图像和文本）映射到一个 shared 特征空间，记映射后 image 和 text 的特征分别为 $f(I)$ 和 $g(T)$，$f, g$ 是两种模态对应的特征提取器，计算两种特征之间的相似度 $S(f(I), g(T))$，简记为 $S(I, T)$ ，其中 $S$ 是选用的某种相似度测量方法。通过以下损失学习特征提取器 $f, \ g$，

$$L_{hard}(I_i, T _ i ) = [\alpha - S(I _ i, T _ i)+ S(I _ i, \hat T _ h)] _ + \ + \ [\alpha - S (I _ i, T _ i)+S(\hat I _ h, T _ i)] _ + \tag{1}$$

(1) 式中，$(I _ i, T _ i)$ 是正例，$\alpha > 0$ 是给定的一个边距（margin），$\hat I _ h$ 和 $\hat T _ h$ 均为 hard 负例（分别与 $I _ i$ 和 $T _ i$ 配对时，表示 mismatched）。$[x]_ + = \max(x, 0)$ ，边距 $\alpha$ 表示 **负例相似度比正例相似度小超过 $\alpha$ 才计算损失，小 $\alpha$ 以内都不计损失** 。

(1) 式中正例 $(I _ i, T _ i)$ 和负例 $(I _ i, \hat T _ h)$、$(\hat I _ h, T _ i)$ 如何确定？

1. 将数据集中所有 pairs 均看作正例，即 $y _ i = 1, \forall i = 1, \ldots, N$
2. 在 mini-batch 中，寻找最相似负例，即，相似度最高，但又不是正例，

    假设正例为 $(I _ i, T _ i)$，

    $$\hat I _ h = \argmax _ {I_j \neq I _ i } S (I _ j, T _ i), \quad \hat T _ h = \argmax _ {T _ j \neq T _ i} S(I _ i, T _ j)$$


实际应用中，不能简单的通过 (1) 式学习特征提取器 $f, \ g$，因为 **数据集 $\mathcal D$ 中不是所有 pairs 都是正例** ，数据集中有一部分是 mismatched（或 weekly matched），所以是一个 noisy 数据集，记作 $\tilde {\mathcal D} = \lbrace (I _ i, T _ i, \tilde y _ i) \rbrace _ {i=1} ^ N$。

## 2.2 健壮匹配损失

一个 noisy 数据集 $\tilde {\mathcal D}$，其中 $\tilde y _ i$ 值不知道是 0 还是 1，其次，对 mismatch 或 weekly matched pairs，0 和 1 两个值无法准确反应 $I _ i$ 和 $T _ i$ 之间的相关度，作者使用 soft label $y ^ {\star} \in [0, 1]$ 表示这个相关度。(1) 式损失调整为 (2) 式，

$$L _ {soft} (I _ i, T _ i) =[\hat \alpha _ i - S(I _ i, T _ i) + S(I _ i,  \hat T _ h)] _ + \ + \ [\hat \alpha _ i - S(I _ i, T _ i) + S(\hat I _ h, T _ i)] \tag{2}$$

其中 

$$\hat \alpha _ i = \frac {m ^ {y _ i ^ {\star}} - 1}{m - 1} \alpha$$

是 soft margin。$m$ 是一个超参数。$\hat I _ h, \ \hat T _ h$ 与 (1) 式一样，是 hard 负例。超参数选择 $\alpha = 0.2, \ m = 10$ 。

由于 $y _ i ^ {\star} \in [0, 1]$，所以 $\hat \alpha _ i \in [0, \alpha]$ 是 $y _ i ^ {\star}$ 的指数函数，这表示 $y ^ {\star}$ 越大，能容忍的误差越大，当 $y ^ {\star} \equiv 1$ 时，(2) 式退化为 (1) 式，对于 mismatched pair，其能容忍的误差较小（相比于原来 hard label），得到的损失更大，也就是说， mismatched pair 比原来更加参与到学习中来。

## 2.3 soft label 估计

(2) 式中用到 soft label，那么给定一个 noisy 数据集的情况下，如何得到 $y ^ {\star}$ ？

还记得上面说的 BiCro 的思想吗？相似的图像其应该有相似的文本描述，反之亦然。

假设通过某种方法收集到一些 clean pairs（对应的 $y ^ {\star} = 1$），那么以 clean pairs 作为锚点（anchor points），计算剩余 pairs 的 soft label。

### 2.3.1 anchor points 选择

深度神经网络（DNN）的记忆效应表明，DNN首先记住训练数据中的 clean labels，然后才会记住 noisy labels，这种现象说明 noisy 样本的损失大而 clean 样本的损失小。因此，给定一个 matching 模型 $(f,g,S)$，计算每个样本的损失如下，

$$l(f,g,S)=\lbrace l_i \rbrace _ {i=1} ^ N = \lbrace L_{hard} (I_i , T _ i) \rbrace _ {i=1} ^ N \tag{3}$$

然后利用损失值 loss 的分布来识别 clean pairs，也就是说对每个样本，判别其是 clean $\tilde y _ i = 1$，还是 mismatched（包括 weekly matched）$\tilde y _ i = 0$，这是一个二分类问题。loss 分布曲线通常是两个峰，很容易想到使用高斯混合模型 GMM 对其建模，

$$p(l) = \sum _ {k=1} ^ K \lambda _ k  p(l|k) \tag{4}$$

上式中 $K =2$，$\lambda _ k$ 是混合权重，$p(l|k)$ 是第 $k$ 个分量（高斯分布）的概率密度。于是，由前面 (3) 式计算的数据集中所有数据 pairs 的损失 $l(f,g,S)$，根据 EM 算法，可以求得所有分量 $p(l|k)$ 的参数（高斯分布的参数为 $\mu _ k , \sigma _ k ^ 2$），以及混合权重 $\lambda _ k$ 的最优解（最大似然意义下）。（GMM EM 算法求解可参考 [这篇文章](/2021/11/13/ml/GMM/)）

然而作者发现 clean pairs 的 loss 分布向 0 处歪斜（skew），使用高斯分布近似 clean 数据子集的 loss 分布，效果并不好，故使用更加灵活的 beta 分布，

$$p(l|\gamma, \beta) = \frac {\Gamma(\gamma + \beta)}{\Gamma(\gamma) + \Gamma(\beta)} l ^ {\gamma -1 } (1-\gamma)^ {\beta -1} \tag{5}$$

这里需要将损失归一化到范围 $l \in [0, 1]$ 。使用 (5) 式代替 $p(l|k)$ 代入 (4) 式，同样使用 EM 算法求 Beta 混合模型（BMM），BMM 包含 $K=2$ 个 beta 分布。

于是，给定第 $i$ 个样本 pair 的损失 $l _ i$，可以求其属于第 $k$ 个分量的概率为

$$p(k|l_i ) = \frac {p(k) p(l _ i | k)}{p (l _ i)} \tag{6}$$

这里 $p(k)=\lambda _ k$，$p(l _ i|k)$ 是第 $k$ 个分量（beta 分布）的概率密度函数，这两个分布都可以通过 EM 算法求解 BMM 得到。于是 (6) 式可以计算出来。

$k=0/1$ 分别表示 clean/noisy ，于是可以从 $\tilde {\mathcal D}$ 中选择一个 anchor points 集合 $\tilde {\mathcal D} _ c = \lbrace I _ c, T _ c, y _ c =1 \rbrace$，

$$\tilde {\mathcal D} _ c = \lbrace (I _ i, T _ i, y _ i=1) | p(k=0|l _ i ) > \delta, \forall (I _ i, T _ i) \in \tilde {\mathcal D} \rbrace \tag{7}$$

其中 $\delta$ 是阈值，剩余的 pairs 则认为是 noisy，记作

$$\tilde {\mathcal D} _ n = \lbrace (I _ i, T _ i) | p(k=0|l _ i ) \le \delta, \forall (I _ i, T _ i) \in \tilde {\mathcal D} \rbrace \tag{8}$$

至此，anchor points 已经选择完毕，但是细想会发现计算损失 (3) 式时用到模型 $(f,g, S)$，根据损失选择出来的 anchor points 被用来计算 soft label，然后再用于学习模型 $(f,g,S)$ ，这说明计算 (3) 式使用的模型肯定不是一个完美的或者最优的模型，否则就不需要学习模型了。那么计算 (3) 式使用的模型不是最优模型（实际上，初始时是一个初始化权重值的模型），得到的 anchor points 也不一定是真正的 clean pairs，用这样的 anchor points 再来训练模型，势必会带来问题。

作者在文中也指出，使用模型选择高置信度（低损失）的样本来训练模型自身会引起严重的误差累积问题。

>  training a model with high-confident (low loss) examples selected by the model itself would cause severe error accumulation problem

作者采用 co-teaching 方法减轻误差累积问题，具体而言：

1. 同时训练两个网络 $A=(f ^ A, g ^ A, S ^ A)$ 和 $B = (f ^ B, g ^ B, S ^ B)$ 。这两个网络结构相同，但是参数初始化不同，输入 batch 序列也不同。

2. 每个训练 epoch，A 计算自己的数据（记为 $\tilde {\mathcal D} ^ A$）上的损失分布，然后：

    - 根据 (7) 式选择 anchor points $\tilde {\mathcal D} _ c ^ A=\lbrace I _ c ^ A, T _ c ^ A, y _ c ^ A=1 \rbrace$
    - 计算 noisy data 的 soft label，得到 $\tilde {\mathcal D} _ n ^ A=\lbrace I _ n ^ A, T _ n ^ A, y _ n ^ A=y ^ {\star} \rbrace$
    - 数据 $\tilde {\mathcal D} _ c ^ A \cup \tilde {\mathcal D} _ n ^ A$ 用于训练 B 。

3. B 同时进行与 A 相同的操作，得到数据 $\tilde {\mathcal D} _ c ^ B \cup \tilde {\mathcal D} _ n ^ B$ 用于训练 A 。

在 co-teaching 之前，增加一个 warmup 操作，目的是让 A 和 B 到达一个初步的收敛，使用 (1) 式损失进行优化（因为这个阶段不涉及到锚点选择和 soft label 计算，自然就无法使用 (2) 式损失）。

warmup 过程其实就是前面说的，用模型计算损失分布并得到 anchor points 然后进一步训练模型自身，也就是说 warmup 过程存在误差累积问题，为了降低 mismatched 数据在 warmup 过程中的影响，使用一定比例的数据 pairs 进行 warmup，也就是说，本来每个 batch 计算 batch 内所有样本的损失的平均值，然后反向传播更新网络，现在选择 batch 内损失较小的那部分样本，计算这些样本的损失平均值，然后反向传播更新网络。

整个训练过程如图 2 所示，

![](/images/multi_modal/bicro_2.png)

<center>图 2. 训练过程</center>

因为是使用 co-teaching 方式训练，那么在 inference 阶段，需要将 A 和 B 的预测求均值。

### 2.3.2 soft label 估计

简单说，就是根据 anchor points $\tilde {\mathcal D} _ c$ 计算 $\tilde {\mathcal D} _ n$ 的 soft label $y ^ {\star} $。

对 $\tilde {\mathcal D} _ n$ 中某个 noisy 数据 $(I _ n ^ i, T _ n ^ i)$，搜索其在 $\tilde {\mathcal D} _ c$ （算法 1 中从 $\tilde {\mathcal D} _ c$ 采样的一个 batch，从这个 batch 中搜索，可以提高搜索速度）中最接近的图像 $I _ c ^ {\triangle}$，然后计算 image2text 相似度一致性，计算方法为比较图像特征距离 $D(I _ n ^ i, I _ c ^ {\triangle})$ 和文本特征距离 $D(T _ n ^ i, T _ c ^ {\triangle})$ ，这里 $D(\cdot, \cdot)$ 是特征空间中的距离函数，

$$\mathcal C _ {i2t} = \frac {D(I _ n ^ i, I _ c ^ {\triangle})}{D(T _ n ^ i, T _ c ^ {\triangle})}, \quad (I _ n ^ i, T _ n ^ i) \in \tilde {\mathcal D} _ n, \ (I _ c ^ {\triangle}, T _ c ^ {\triangle}) \in \tilde {\mathcal D} _ c \tag{9}$$


类似地，计算 text2image 相似度一致性，

$$\mathcal C _ {t2i} = \frac {D(T _ n ^ i, T _ c ^ {\diamond})}{D(I _ n ^ i, I _ c ^ {\diamond})}, \quad (I _ n ^ i, T _ n ^ i) \in \tilde {\mathcal D} _ n, \ (I _ c ^ {\diamond}, T _ c ^ {\diamond}) \in \tilde {\mathcal D} _ c \tag{10}$$

其中 $T _ c ^ {\diamond}$ 是 $\tilde {\mathcal D} _ c$ 中文本特征最接近 $T _ n ^ i$ 的。

soft label 就是双向跨模态相似度一致性，计算如下，

$$y ^ {\star} = \frac {\mathcal C _ {i2t} + \mathcal C _ {t2i}} 2 \tag{11}$$

计算出 soft label 后，得到 $\tilde {\mathcal D} _ n = \lbrace I _ n, T _ n, y _ n = y ^ {\star} \rbrace$，然后与 clean 数据集 $\tilde {\mathcal D} _ c = \lbrace I _ c, T _ c, y _ c = 1 \rbrace$ 合并后作为数据集训练模型，使用 (2) 式表示的损失。

完整的训练流程如算法 1 所示，

![](/images/multi_modal/bicro_3.png)

<center>算法 1. 完整的训练流程</center>

算法 1 中，$\mathcal P ^ A, \ \mathcal P ^ B$ 分别是使用网络 B 和网络 A 的 BMM 对数据集 $\tilde {\mathcal D}$ 中每个数据 pair 预测为 clean 的概率得分，

**# 合理性讨论**

我们需要进一步搞懂这两个相似度一致性计算式的合理性。

前面已经表明 BiCro 的思想：相似的图像其应该有相似的文本描述，反之亦然。

对 $\tilde {\mathcal D} _ n$ 中的某个数据 pair $(I _ n ^ i, T _ n ^ i)$，$T _ n ^ i$ 与 $T _ n ^ i$ 不匹配，我们在 clean 数据集 $\tilde {\mathcal D} _ c$ 搜索与 $I _ n ^ i$ 图像特征最接近的 clean pair $(I _ c ^ {\triangle}, T _ c ^ {\triangle})$ 作为锚点，即 $D(I _ n ^ i, I _ c ^ {\triangle})$ 很小，根据 BiCro 思想，那么对应的文本特征越接近越好，也就是说 $D(T _ n ^ i, T _ c ^ {\triangle})$ 越小，表示 $(I _ n ^ i, T _ n ^ i)$ 越接近 clean，那么 image2text 相似度一致性越高。

从图像特征空间搜索最近的锚点只是一个方向，还需要从文本特征空间搜索最近的锚点，分别计算 image2text 和 text2image 的相似度一致性，然后求平均值，作为 soft label。

# 3. 源码解读

在需要的时候也会进行实验说明。

以数据集 `Flickr30k` 为例。由于 Flickr30k 数据集匹配的很好，所以需要人为的对一部分数据 pair 破坏使其不匹配，这个比例为 `noise_ratio = 0.4` ，这部分代码为，

```python
# 以下代码见 PrecomDataset 类
self.length = len(self.captions)    # 训练数据集，所有图像的文本描述数量
if self.images.shape[0] != self.length:
    self.im_div = 5     # flickr30k，每个图像有 5 个文本描述句子。
else:
    self.im_div = 1

# 文本句子编号 -> 图像编号，
# [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,...]
self.t2i_index = np.arange(0, self.length) // self.im_div

self._t2i_index = copy.deepcopy(self.t2i_index)     # 深拷贝

idx = np.arange(self.length)    # 文本句子编号列表
np.random.shuffle(idx)          # 打乱编号，例如[5,10,4,7,1,3,...]
noise_length = len(noise_ratio * self.length)   # 人工加噪数量，用于下面设置部分数据 mismatched
shuffle_index = self.t2i_index[idx[:noise_length]]  # 获得随机选择的部分句子编号以及对应的图像编号（此时仍是 matched）
np.random.shuffle(shuffle_index)    # 把图像编号打乱
self.t2i_index[idx[:noise_length]] = shuffle_index  # 这部分随机选择句子，其对应的图像编号已经打乱，不再匹配
```

**# warmup**

```python
# 创建两个相同结构的模型，使用 SGRAF
model_A = SGRAF(opt)
model_B = SGRAF(opt)
for epoch in range(0, opt.warmup_epoch):    # warmup epoch 设置为 10
    warmup(opt, noisy_trainloader, model_A, epoch)
    warmup(opt, noisy_trainloader, model_B, epoch)
```

warmup 方法定义如下，

```python
def warmup(opt, train_loader, model, epoch):
    # images: batch 图像数据
    # captions: batch 图像对应的文本句子
    # lengths: captions 中每个句子的 token 数量
    # _: 文本句子的编号
    for i, (images, captions, lengths, _) in enumerate(train_loaders):
        loss = model.train(images, captions, lengths, mode=opt.warmup_type)
```

给出模型的关键组件，

```python
class SGRAF:
    def __init__(self, opt):
        # 图像特征提取器
        self.img_enc = EncoderImage(
            # no_imgnorm: 不要归一化图像特征向量，默认值为 false，表示需要归一化
            opt.img_dim, opt.embed_size, no_imgnorm=opt.no_imgnorm  
        )
        # 文本特征提取器
        self.txt_enc = EncoderText(
            opt.vocab_size,
            opt.word_dim,
            opt.embed_size,
            opt.num_layers,
            use_bi_gru=opt.bi_gru,
            no_txtnorm=opt.no_txtnorm
        )
        # 计算 batch 内， 图像与文本之间的相似度，输出矩阵 batch_size x batch_size
        # n_imgs x n_captions
        self.sim_enc = EncoderSimilarity(
            opt.embed_size, opt.sim_dim, opt.module_name, opt.sgr_step
        )
        # margin 就是上文的边距 alpha，warmup_rate 是最终用于 warmup 的样本占 batch 的比例
        self.criterion = ContrastiveLoss(margin=opt.margin, warmup_rate=opt.warmup_rate)

    def train(self, images, captions, lengths, hard_negative=True,
        labels=None, soft_margin=None, mode='train', sim_type='euc', ids='non'):
        # 提取图像和文本特征
        # cap_lens 表示 batch 内各句子的实际长度
        img_embs, cap_embs, cap_lens = self.forward_emb(images, captions, lengths)
        # 计算batch 内，图像与文本之间相似度矩阵，n_imgs x n_captions
        # 也就是这个 batch 内，每个图像与每个文本之间的相似度
        sims = self.forward_sim(img_embs, cap_embs, cap_lens)

        self.optimizer.zero_grad()
        loss = self.criterion(
            sims,
            hard_negative=hard_negative,
            labels=labels,
            soft_margin=soft_margin,
            mode=mode,
            noise_tem = self.noise_tem  # noise_soft 温度，设置为 0.5
        )
        loss.backward()
        if self.grad_clip > 0:  # 限制梯度，防止梯度过大引起不稳定
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()   # 更新网络参数
```

warmup 中使用 $L _ {hard}$ 作为损失，这在 `ContrastiveLoss` 中实现，代码如下，

```python
# 位于 ContrastiveLoss 类
def forward(self, scores, hard_negative=True, labels=None, 
    soft_margin='linear', mode='train', noise_tem=0.9):
    diagonal = scores.diag().view(scores.size(0), 1)    # 提取 S(Ii, Ti)
    # (n_imgs, n_imgs)
    d1 = diagonal.expand_as(scores) # 第 i 行表示 S(Ii, Ti)
    d2 = diagonal.t().expand_as(scores) # 第 i 列表示 S(Ii, Ti)

    if labels is None:  # hard label
        margin = self.margin    # 就是上文的边距 alpha
    elif soft_margin == 'linear': # 线性边距
        margin = self.margin * labels   # \hat alpha = alpha * (y ^ *)
    elif soft_margin == 'exponential':
        s = (torch.pow(10, labels) - 1) / 9
        margin = self.margin * s    # \hat alpha = (m ^ (y^*) - 1)/(m-1) * alpha
    elif soft_margin == 'sin':
        s = torch.sin(math.pi * labels - math.pi / 2) / 2 + 1 / 2
        margin = self.margin * s
    
    # (n_imgs, n_imgs)
    cost_im = (margin + scores - d2).clamp(min=0)   # 按 (1) 式计算 hard loss
    # cost_im 是一个矩阵 n_imgs x n_captions，(j,i) 位置值为
    # alpha + S(Ij,Ti) - S(Ii,Ti)  -> Lhard 第二项

    if labels is not None and soft_margin == 'exponential':
        margin = margin.t() # 这一 margin 是向量，转置后实际上不变

    # (n_imgs, n_imgs)
    cost_s = (margin + scores - d1).clamp(min=0)
    # cost_s，(i,j) 位置值为
    # alpha + S(Ii,Tj) - S(Ii,Ti)   -> Lhard 第一项

    mask = torch.eye(scores.size(0)) > 0.5  # 对角线为 True，(n_imgs, n_imgs)
    mask = mask.to(cost_s.device)
    # 将对角线上值置 0，因为求batch内最相似的特征时，要排除自己
    cost_s, cost_im = cost_s.masked_fill_(mask, 0), cost_im.masked_fill_(mask, 0)
    # Lhard 第一项，cost_s 矩阵，要求每一行的最大值：搜索列找到 Tj，使得 S(Ii,Tj) 最大
    # 这样的 Tj 就是上文中的 \hat T_h
    # Lhard 第二项，cost_im 矩阵，要求每一列的最大值：搜索列找到 Ij，使得 S(Ij,Ti) 最大
    # 这样的 Ij 就是上文中的 \hat I_h
    cost_s_max, cost_im_max = cost_s.max(1)[0], cost_im.max(0)[0]
    # 分别按行，按列求均值
    cost_s_mean, cost_im_mean = cost_s.mean(1), cost_im.mean(0)

    if mode == 'predict':
        ...
    elif mode == 'warmup_sele': # 这里关心 warmup
        all_loss = cost_s_mean + cost_im_mean
        # 取 loss 最小的 k 个 损失值
        y = all_loss.topk(k=int(scores.size(0)*self.warmup_rate), dim=0, largest=False, sorted=True)
        index = torch.zeros(scores.size(0)).cuda()
        index[y[1]] = 1 # batch 中选中用于 warmup 的样本位置 mask
        all_loss = all_loss * index
        return all_loss.sum()
```

以上代码，warmup 过程使用的模式是 `warmup_sele`，也就是选择部分样本用于 warmup。但是代码中使用的 $L_{hard}$ 与 (1) 式并不完全相同，区别在于 (1) 式是计算的负样本的最大相似度，

$$S(I_i , \hat T _ h) = \max _ {T _ j \neq T _ i} S(I _ i, T _ j)
\\\\ S(\hat I _ h, T _ i) = \max _ {I _ j \neq I _ i} S (I _ j, T _ i) \tag{3.1}$$

其中 $i \in [1, B]$ ，$B$ 表示 `batch_size` 。

而代码中 $L _ {hard}$ 是计算 batch 内负样本相似度均值，

$$\frac 1 B \sum _ {T _ j \neq T _ i} S(I _ i, T _ j), \quad \frac 1 B \sum _ {I _ j \neq I _ i} S(I _ j, T _ i) \tag{3.2}$$

网络 A 和 B 的 warmup 过程均使用整个训练数据集，这两个网络的 batch sequence 不同，也就是说，A 和 B 的 warmup 过程，会独立对训练集 shuffle ，然后按 batch 进行前向+后续传播。

注意，每个 batch 内，选择 `warmup_rate` 比例的最小 loss （之和）进行反向传播，更新网络参数。

**# 选择锚点**

网络 A 和 B 经过 warmup 之后，可以用来从数据集 $\tilde {\mathcal D}$ 中选择 clean pairs，即锚点，方法是先使用 $L _ {hard}$ 作为损失，计算每个数据 pair 的损失，根据损失分布，然后使用 GMM/BMM 拟合，根据 (6) 式计算属于哪个分量（clean/noisy）的后验概率。

```python
# all_loss = [[], []]
def eval_train(
    opt, model_A, model_B, data_loader, data_size, all_loss
):
    ...
    losses_A = torch.zeros(data_size)   # 保存所有数据 pair 的 Lhard
    losses_B = torch.zeros(data_size)
    # 参考上面代码中关于 images, captions, lengths, ids 的说明
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        with torch.no_grad():
        # cost_s_mean + cost_im_mean，shape 为 (batch_size,)
            loss_A = model_A.train(images, captions, lengths, mode='eval_loss')
            loss_B = model_B.train(images, captions, lengths, mode='eval_loss')
            for b in range(images.size(0)):
                losses_A[ids[b]] = loss_A[b]    # 设置训练集对应位置上的损失 Lhard
                losses_B[ids[b]] = loss_B[b]
    # 归一化 loss
    losses_A = (losses_A - losses_A.min()) / (losses_A.max() - losses_A.min())
    losses_B = (losses_A - losses_B.min()) / (losses_B.max() - losses_B.min())
    all_loss[0].append(losses_A)
    all_loss[1].append(losses_B)

    input_loss_A = losses_A.reshape(-1, 1)  # (N, 1)
    input_loss_B = losses_B.reshape(-1, 1)  # N 是 dataset size
    if opt.fit_type == 'gmm':
        ...
    else:   # bmm
        bmm_A = BetaMixture1D(max_iters=10)
        bmm_A.fit(input_loss_A.cpu().numpy())
        prob_A = bmm_A.posterior(input_loss_A.cpu().numpy(), 0)

        bmm_B = BetaMixture1D(max_iters=10)
        bmm_B.fit(input_loss_B.cpu().numpy())
        prob_B = bmm_B.posterior(input_loss_B.cpu().numpy(), 0)
    return prob_A, prob_B, all_loss
```

上述代码的模式为 `eval_loss`，这使得计算损失时，不像 warmup 那样选择部分最小 loss，而是选择全部损失 `cost_s_mean + cost_im_mean`，这个 tensor shape 为 `(batch_size,)`。计算出训练集所有数据 pairs 的 loss `prob_A, prob_B` 之后，根据 threshold 筛选出锚点，筛选函数为

```python
def split_prob(prob, threshld):
    if prob.min() > threshld:  # 阈值太小，所有样本均满足条件
        # 增大阈值，使得数据集中有 1% 的数据被过滤掉，即， 99% 的样本视作锚点
        threshld = np.sort(prob)[len(prob) // 100]
    pred = prob > threshld
    return pred
```

对于 Flickr30k，每个图像有 5 个文本句子，那么以句子数量为准，作为数据集的 size，每个图像 id 需要 repeat 5 次，然后在分别与 5 个句子组成数据 pair。所有数据 pairs 中，选择其中一部分数据 pairs ，对句子 id 重新 shuffle，那么这部分数据 pairs 相当于人工破环，与其他未破坏的数据 pairs 合并，得到 noisy 数据集。训练集做这种人工破坏，制造 noisy pairs，但是验证集则不做人工破坏，也就是不人为生成 noisy pairs（对 Flickr30k，没有 noisy pairs，对其他例如 Conceptual Captions 则自带 noisy pairs）。

**# 估计 soft label**

根据锚点，计算其他 noisy pairs 的 $y^{\star}$ 。以 B 计算出的训练集损失和选择的锚点来训练 A 为例说明。

```python
labeled_trainloader, unlabeled_trainloader = get_loader(
    captions_train,
    images_train,
    "train",
    opt.batch_size,
    opt.workers,
    opt.noise_ratio,
    opt.noise_file,
    pred=pred_B,
    prob=prob_B
)
```

上述代码中，`labeled_trainloader` 就是 B 计算出来的锚点组成的数据集，这些锚点 $y=1$。`unlabeled_trainloader` 就是训练集中（指 `captions_train` 和 `images_train`）不是锚点的其他数据 pair 组成的训练集，其 (soft) label 未知，需要计算。

`labeled_trainloader` 和 `unlabeled_trainloader` 均进行了部分数据 pairs 的人工破坏，模拟 noisy 数据集。

估计 soft label 这一步合并在训练网络过程中，相关代码为，

```python
# 用 B 计算的锚点和 soft label 训练 A 时，net 是 A，net2 是 B
def train(opt, net, net2, labeled_trainloader, unlabeled_trainloader=None, epoch=None):
    net.train_start()
    net2.val_start()
    unlabeled_train_iter = iter(unlabeled_trainloader)

    for i, batch_train_data in enumerate(labeled_trainloader):
        # 从 clean 数据集中随机取一个 batch
        (
            batch_images_l, # clean 数据集的一个 batch 的图像数据
            batch_text_l,   # 对应的文本数据
            batch_lengths_l,# batch 内每个文本的长度，token 数量
            _,              # batch 中每个数据 pair 在整个 clean 数据集中的 id
            batch_labels_l, # 既然来自 clean 数据集，那么 label = 1
            batch_prob_l,   # batch 内每个数据 pair 预测为 clean 的概率，由 B 计算得到
            batch_clean_labels_l, # 由于人工破坏，所以被破坏的数据 pair，其 label=0
        ) = batch_train_data
        # 从 非clean 数据集中随机取一个 batch
        (
            batch_images_u,
            batch_text_u,
            batch_lengths_u,
            _,
            batch_clean_labels_u,   # 如果数据 pair 被人工破坏，那么为 0，否则为 1
        ) = unlabeled_train_iter.next()
    if epoch < (opt.num_epochs // 2):   # 训练 epoch 还未完成一半
        loss_u = 0      # 训练的前半阶段，不使用 noisy batch，只使用 clean batch
        with torch.no_grad():
            # c_y: clean pairs' soft label
            # n_y: noisy paris' soft label，这个分支没有使用 noisy batch，故此值为空
            c_y, n_y = net2.predict(batch_images_l, batch_text_l, batch_lengths_l)
    else:
        with torch.no_grad():
            # 使用 B 计算 clean pairs' soft label
            # 使用 B 计算 noisy pairs' soft label
            c_y, n_y = net2.predict(batch_images_l, batch_text_l, batch_lengths_l, batch_images_u, batch_text_u, batch_lengths_u, epoch=epoch)
        # 使用 noisy batch 训练 A，注意提供了 soft label（由 B 计算）
        loss_u = net.train(
            batch_images_u,
            batch_text_u,
            batch_lengths_u,
            labels=n_y,     # 提供了 soft label，而前面 warmup 阶段是没有这个值
            hard_negative=True, # 计算损失时用到. True: 使用 max S; False: 使用 mean S。参见上面 (3.1), (3.2) 式
            soft_margin=opt.soft_margin,    # 边距类型：线性，指数，正弦
            mode=opt.noise_train    # 模式为 noise_soft
        )
    # 使用 clean batch 训练 A，提供了 soft label（由 B 计算）
    loss_l = net.train(
        batch_images_l,
        batch_text_l,
        batch_lengths_l,
        labels=c_y,
        hard_negative=True,
        soft_margin=opt.soft_margin
        mode=opt.noise_train
    )
```

以上代码可以看出：

1. 训练前半 epochs，不使用 noisy batch（可能是因为网络先记住低损失的，后面才记住高损失的，所以前面使用高损失的 noisy pairs 对学习帮助不大）。

2. noisy batch 需要计算 soft label，clean batch 也要计算 soft label，因为 clean batch 也可能包含人工破坏掉的 noisy pair。从下方的代码可以看出，对 clean batch，重新计算 Lhard，然后选取 `10% * B + 1` 个样本作为 clean pairs，即最终的锚点，其他样本则以这些锚点为基准计算 soft label。

3. 估计 soft label 值的相关代码在模型的 `predict` 方法内。

下面来看 `predict` 方法定义，

```python
def predict(self, images, captions, lengths, images_n='', captions_n='', lengths_n='', epoch=0):
    '''
    images: 来自 clean 数据集的一个 batch 的图像数据
    captions: 对应的文本数据
    lengths：文本的长度，token 数量
    带 `n` 后缀的则表示来自 noisy 数据集的 batch
    epoch: 当前训练轮次
    '''
    # (n_imgs, 36, emb_size)，每个图像分成 36 patches，每个 patch 的 local 特征
    # (n_caps, L, emb_size)，L 为最长句子长度
    img_embs, cap_embs, cap_lens = self.forward_emb(images, captions, lengths)
    # 计算 global 特征，实际上使用求均值的方法得到 global feat
    # (n_imgs, emb_size),  (n_caps, emb_size)
    img_g, text_g = self.sim_enc.glo_emb(img_embs, cap_embs, cap_lens)
    # 得到跨模态相似度矩阵，(n_imgs, n_caps)
    sims = self.forward_sim(img_embs, cap_embs, cap_lens)   # 计算跨模态相似度，S(Ii,Tj)
    # 再次根据 Lhard，选择 (10% * B + 1) 个样本作为 clean
    # 这里重新计算哪些是 clean 样本，并以这些样本为锚点
    clean_index, noise_index = self.criterion(sims, mode='predict_clean')
    image_c = img_g[clean_index]    # (n_clean, emb_size) ，锚点的图像特征
    text_c = text_g[clean_index]    # (n_clean, emb_size) ，锚点的文本特征
    # 计算 batch 内所有图像特征与锚点图像特征的相似度，这里使用余弦相似度
    sim_img = SIM_PAIR(image_c, img_g)      # (n_imgs, n_clean)
    # 计算 batch 内所有文本特征与锚点文本特征的相似度，这里使用余弦相似度
    sim_text = SIM_PAIR(text_c, text_g)     # (n_caps, n_clean)

    # 参考 (9) 和 (10) 式，其中距离函数 D（没有归一化） 换成余弦相似度（归一化了，范围 [-1, 1]）
    img2text = SIM_SELE(sim_img, sim_text)  # (n_imgs, 1)
    text2img = SIM_SELE(sim_text, sim_img)  # (n_caps, 1)
    # SIM_SELE 函数内部确保每个样本的相似度一致性值小于 1，但是下限好像不确定？

    dis_f_n = 0.5 + (img2text + text2img) / 4   # 上限为 1
    if self.noise_train == 'noise_soft':
        index = dis_f_n > self.noise_tem    # 筛选 > 0.5 的样本
        dis_f_n = dis_f_n * index           # 保留 > 0.5 的样本
    y_value = torch.zeros(images.size()[0]).cuda()      # (n_imgs, )
    y_value[clean_index] = 1                # 本次重新选择的锚点，其 soft label 直接设置为 1
    # 其他样本则使用计算出来的 soft label
    y_value.scatter_(0, noise_index, dis_f_n.clamp(0, 1).squeeze(1))
    c_y = y_value.clamp(0, 1)

    if epoch:   # 有值，表示进入后半阶段训练，已经用上了 noisy batch
        img_embs, cap_embs, cap_lens = self.forward_emb(images_n, captions_n, lengths_n)
        img_g_n, text_g_n = self.sim_enc.glo_emb(img_embs, cap_embs, cap_lens)
        sim_img = SIM_PAIR(image_c, img_g_n)    # 计算图像特征余弦相似度，使用前面 clean batch 中 10% *B + 1 个锚点
        sim_text = SIM_PAIR(text_c, text_g_n)   # 计算文本特征余弦相似度
        # 对于 noisy batch，也从前面的 10%*B + 1 个锚点中进行搜索：1. 图像特征最接近的锚点；2. 文本特征最接近的锚点
        img2text = SIM_SELE(sim_img, sim_text)
        text2img = SIM_SELE(sim_text, sim_img)
        dis_f_n = (img2text + text2img) / 2
        if self.noise_train == 'noise_soft':
            index = dis_f_n > self.noise_tem
            dis_f_n = dis_f_n * index
        n_y = dis_f_n.clamp(0, 1).squeeze(1)
    else:
        n_y = ''
    return c_y, n_y
```

最后是使用计算出来的 soft label 去训练网络 A，这时损失使用 $L _ {soft}$，其他则与 warmup 过程的训练基本相同。