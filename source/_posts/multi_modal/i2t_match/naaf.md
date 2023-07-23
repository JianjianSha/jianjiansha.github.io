---
title: NAAF：用于 Image-Text 匹配的负注意力机制
date: 2023-07-16 10:02:58
tags: image-text retrieve
mathjax: true
---
# 1. 简介

Image=Text 匹配问题，有两种方法：1. global-level matching，也就是将 image 和 text 映射到一个 shared 特征空间，得到 whole image 的特征以及 full text 的特征；2. local-level matching，将图像显著区域和文本单词之间进行匹配，这种匹配比起 global-level matching 更加精细。对于 local-level matching，现有的研究主要考虑到 image 与 text 匹配的部分，而不匹配的部分则直接忽略，例如 图 1 (b)，“boys”，“trees”，“road” 与 image 的某个 region 匹配的很好，而 “football” 则与 image region 不匹配（image 中为篮球），这个不匹配直接被忽视，导致最终的 image-text 匹配得分非常高，这显然不合适，因此，作者认为匹配的部分和不匹配的部分一起决定总的匹配得分，例如图 1 (c) 中，“football” 不匹配，所以需要扣分，导致最终总的匹配得分有所下降。

![](/images/multi_modal/naaf_1.png)

<center>图 1. </center>

# 2.方法

一个 image-text pair $(U, V)$，text 由单词的文本特征表示 $U=\lbrace u_i | i \in [1,m], u _ i \in \mathbb R ^ d \rbrace$，image 由 region 的视觉特征表示 $V=\lbrace v _ j | j \in [1, n], v _ j \in \mathbb R ^ d \rbrace$ ，$d$ 是特征维度，为了方便计算特新相似度，文本特征和视觉特征维度相同。

## 2.1 负注意力

给定一个 image-text pair，其中包含了很多 word-region fragment pair，也就是说，将 text 看作多个 words，image 划分为多个 region，那么就得到多个 word-region pairs，这些 fragment pairs 分为 matched 和 mismatched 两种，这两种全部利用可以得到更准确的匹配得分。为此，作者提出了 NAAF（negative-aware attention framework），包含了两个主要的模块：1. 有判别的 mismatch 挖掘；2. Neg-Pos 分支匹配。 下面分别介绍。

### 2.1.1 有判别的 mismatch 挖掘

存在 mismatched 的相似度分布和 matched 的相似度分布，给定一个 fragment pair 的相似度，需要判断其属于 mismatched 还是 matched。

mismatched 和 matched 两种 word-region fragment pair 的相似度集合分别记为，

$$S _ k ^ {-} = [s _ 1 ^ -, s _ 2 ^ -, \ldots, s _ i ^ -] \tag{1}$$

$$S _ k ^ + = [s _ 1 ^ +, s _ 2 ^ +, \ldots, s _ i ^ +] \tag{2}$$

在训练过程中，会动态更新这两个集合，每次更新使用不同的 $k$ 表示（即，更新轮次），其中 $s _ i ^ -$ 和 $s _ i ^ +$ 通过采样得到。采样策略后面 `### 2.1.3` 会介绍。

（下文很多变量的定义均是从文本的角度出发。）

根据采样得到的 $S _ k ^ -$ 和 $S _ k ^ +$，对 mismatched 和 matched 的 fragment pair 的相似度分布建模如下，

$$f _ k ^ - (s) = \frac 1 {\sigma _ k ^ - \sqrt {2 \pi}} \exp \left[- \frac {(s - \mu _ k ^ -) ^ 2} {2 (\sigma _ k ^ -) ^ 2}\right]
\\\\ f _ k ^ + (s) = \frac 1 {\sigma _ k ^ + \sqrt {2 \pi}} \exp \left[- \frac {(s - \mu _ k ^ +) ^ 2} {2 (\sigma _ k ^ +) ^ 2}\right]$$

假设存在一个边界 $t$，用于区分相似度属于 matched 还是 mismatched，如图 2 (b) 所示，

![](/images/multi_modal/naaf_2.png)

<center>图 2. NAAF 框架图</center>

判别错误（error）有两种：1. mathced 被判为 mismatched，如图 2 (b) 中红色部分 $E _ 2$；2. mismatched 被判为 matched，如图 2 (b) 中蓝色部分 $E_1$ 。

通过最小化如下加权错误（error）概率得到最佳边界 $t$，

$$\min _ t \quad \alpha \int _ t ^ {\infty} f _ k ^ - (s) ds + \int _ {-\infty} ^ t f _ k ^ + (s) ds , \quad s.t. \quad t \ge 0 \tag{3}$$

其中 $\alpha$ 是 $E _ 1$ 的惩罚权重。

求 (3) 式导数并令其为 0，可以得到 $t$ 的最优解。

$$\begin{aligned}\frac {\partial} {\partial t} (\alpha E _ 1 + E _ 2) &=f _ k ^ + (t) - \alpha f _ k ^ - (t)
\\\\ &= \frac 1 {\sigma _ k ^ + \sqrt {2 \pi}} \exp \left[- \frac {(t - \mu _ k ^ +) ^ 2} {2 (\sigma _ k ^ +) ^ 2}\right]-\alpha \frac 1 {\sigma _ k ^ - \sqrt {2 \pi}} \exp \left[- \frac {(t - \mu _ k ^ -) ^ 2} {2 (\sigma _ k ^ -) ^ 2}\right]
\\\\ &= 0
\end{aligned}$$

上式移项，两边取对数，

$$\frac {(t - \mu _ k ^ +) ^ 2} {2 (\sigma _ k ^ +) ^ 2}= \log \frac {\sigma _ k ^ -}{\alpha \sigma _ k ^ +} + \frac {(t - \mu _ k ^ -) ^ 2} {2 (\sigma _ k ^ -) ^ 2}$$

变换得，

$$(\sigma _ k ^ -) ^ 2 (t - \mu _ k ^ +) ^ 2 = 2 (\sigma _ k ^ + \sigma _ k ^ -) ^ 2 \log \frac {\sigma _ k ^ -}{\alpha \sigma _ k ^ +} + (\sigma _ k ^ +) ^ 2 (t - \mu _ k ^ -) ^ 2$$

这是 $t$ 的二次方程，根据韦达定理 $(-b \pm \sqrt{b ^ 2 - 4ac}) / 2a$，可以最优解为

$$t _ k = [(((\beta _ 2 ^ k) ^ 2 - 4 \beta _ 1 ^ k \beta _ 3 ^ k ) ^ {1/2} - \beta _ 2 ^ k) / (2 \beta _ 1 ^ k)] _ + \tag{4}$$

这里，

$$\begin{aligned}\beta _ 1 ^ k &= (\sigma _ k ^ +) ^ 2 - (\sigma _ k ^ -) ^ 2
\\\\ \beta _ 2 ^ k &= 2 [\mu _ k ^ + (\sigma _ k ^ -) ^ 2 -\mu _ k ^ - (\sigma _ k ^ +) ^ 2]
\\\\ \beta _ 3 ^ k &= (\sigma _ k ^ + \mu _ k ^ -) ^ 2 - (\sigma _ k ^ - \mu _ k ^ +) ^ 2 + 2 (\sigma _ + \sigma _ k ^ -) ^ 2 \log \frac {\sigma _ k ^ -}{\alpha \sigma _ k ^ +}
\end{aligned}$$

训练结束时，我们希望边界值 $t _ k$ 可以最大化挖掘 mismatched fragments，并同时避免 matched fragments 被误判，因为要做到最大化挖掘 mismatched fragments，根据图 2 (b) 不难知道 $t$ 应该较大，然而 $t$ 较大则会带来 matched fragments 被误判，为了兼顾这两个目的，对 mismatched fragments 的错误（error）使用一个惩罚参数 $\alpha$ 。

根据 (4) 式，$t_k$ 可以看作 $\alpha$ 的函数 $t _ k (\alpha)$ ，那么求 $\alpha$ 最佳值的问题可以转化为，

$$\begin{aligned} \alpha ^ {\star} = & \max _ {\alpha} \quad \int _ {-\infty} ^ {t _ k (\alpha)} f _ k ^ - (s)ds,
\\\\ & s.t. \quad \int _ {t _ k (\alpha)} ^ {\infty} f _ k ^ + (s)ds \approx 1, \ \alpha > 0
\end{aligned}$$

上式中 $\max (\cdot)$ 表示最大化挖掘 mismatched fragments，然后约束条件是 matched fragments 被判断正确的概率接近 1，即基本不会误判。

求解 $\alpha ^ {\star}$ 分两步：1. 求可行解集合（满足约束条件的解的集合）；2. 通过映射求最优解。

对于第一步，为了满足约束条件，根据概率极限理论，$t _ k (\alpha)$ 应该位于范围 $[0, t ^ {\star}]$，其中 $t ^ {\star}$ 根据经验可以取 $\mu - 3 \sigma$ （使用 $f _ k ^ +$ 分布的期望和标准差），根据 Chebyshev 不等式，有 $P(|X - EX| \ge \epsilon) \le V/ \epsilon ^ 2$，所以 $P(|X-\mu| \ge 3 \sigma) \le \sigma ^ 2 /(3\sigma) ^ 2=1/9$，于是 $P(X\le \mu - 3 \sigma) \le 1/ 18$ 。然后由于目标优化函数随 $t _ k (\alpha)$ 增大而增大，所以最优解取可行解 $[0, t ^ {\star}]$ 的最大值，即 

$$\lim _ {\alpha \rightarrow \alpha ^ {\star}} t _ k (\alpha) = t ^ {\star} = \mu _ k ^ + - 3 \sigma _ k ^ + \tag{4.1}$$

上式代入 (4) 式之前的一个等式，解出 $\alpha ^ {\star}$，不过与论文 (5) 式给出的 $\alpha ^ {\star}$ 并不相同，笔者没有搞清楚论文 (5) 式如何推导出来。

### 2.1.2 负-正 注意力分支

双分支框架同时考虑了 mismatched 和 matched 片段（fragments）。计算 word 和 region 之间的语义关联得分如下，

$$s _ {ij} = \frac {u _ i v _ j ^ {\top}}{||u _ i|| \ ||v _ j||}, \quad, i \in [1,m], j \in [1,n] \tag{6}$$


**# 负注意力**

**从文本的角度出发**，文本片段如果没有匹配的图像 region，那么被认为是 mismatched。一个模态（例如文本）的片段与另一个模态（例如图像）的所有片段之间的跨模态相似度最大值，反应了其是匹配的还是未匹配的，因此对文本片段 $u _ i, i \in [1,m]$ 与图像的所有 regions $\lbrace v _ j\rbrace _ {j=1} ^ n$ 之间的相似度做 max-pooling，

$$s _ i = \max _ j (\lbrace s _ {ij} - t _ k \rbrace _ {j=1} ^ n) \tag{7}$$



因此第 `i` 个 word 在 image-text pair $(U, V)$ 中的负效应（即，不相似）可以使用 (8) 式度量，

$$s _ i ^ {neg} = s _ i \odot \text {Mask} _ {neg}(s _ i) \tag{8}$$

其中掩码 $\text{Mask} _ {neg}(\cdot)$ 在输入为负时，输出 1，否则输出 0。

(7) 和 (8) 式表明，两种模态片段之间的相似度与区分边界 $t _ k$ 比较，如果大于等于 $t _ k$，那么被判别为 matched，负效应为 0，当小于 $t _ k$ 时，才被判为 mismatched，负效应为两者差值。

为了更准确地测量负效应，作者还考虑了文本片段地语义内关系，也就是模态内部 word 之间的匹配度，

$$\hat s _ i = \sum _ {l=1} ^ m w _ {il} ^ {intra} s _ l, \quad s.t. \ w _ {il} ^ {intra} = \text{softmax} _ {\lambda} (\lbrace \frac {u _ i u _ l ^ {\top}}{||u _ i|| \ ||u _ l||} \rbrace _ {l=1} ^ m) \tag{9}$$

模态内部的匹配度是 $s _ l$ 的加权求和，权重则使用余弦相关度的 softmax 值（softmax 用于归一化）。$\lambda$ 是 scaling 因子。

在 inference 阶段，将 (8) 式中 $s _ i$ 替换为 $\hat s _ i$ 。

**# 正注意力**

聚合与 query word 匹配的所有 image region，得到这个 word 在图像中的共享语义。聚合权重按如下计算，

$$w _ {ij} ^ {inter} = \text{softmax} _ {\lambda} (\lbrace \text{Mask} _ {pos} (s _ {ij} - t _ k)\rbrace _ {j=1} ^ n) \tag{10}$$

其中 $\text{Mask} _ {pos} (\cdot)$ 输入为正时，输出等于输入，否则输出为 $-\infty$，这样使用 softmax 归一化时，$-\infty$ 就变成 0。于是当 $s _ {ij} < t _ k$ 时，pair 被判为 mismatched，其正相关度的聚合权重应当为 0 。$w _ {ij} ^ {inter}$ 表示单词 $u _ i$ 与图像 region $v _ j$ 之间的语义关联。

于是，对于单词 $w _ i$，其在整个图像的共享语义为 $\hat v _ i = \sum _ {j=1} ^ n w _ {ij} ^ {inter} v _ j$ ，于是相似度计算为

$$s _ i ^ f = \frac {u _ i \hat v _ i ^ {\top}}{||u _ i|| \ ||\hat v _ i||} \tag{10.1}$$

word 与 region 之间的关联得分 $s _ {ij}$ 也反映了相似度，所以也计算了基于 $s _ {ij}$ 的加权相似度， $s _ i ^ r = \sum _ {j=1} ^ n w _ {ij} ^ {relev} s _ {ij}$，其中权重为 $w _ {ij} ^ {relev} = \text{softmax} _ {\lambda} (\lbrace \overline s _ {ij} \rbrace _ {j=1} ^ n)$ ，而 $\overline s _ {ij} = [s _ {ij}] _ + \ / \sqrt{\sum _ {i=1} ^ m [s _ {ij}] _ + ^ 2}$ 。

总的 matched fragment 的正效应为

$$s _ i ^ {pos} = s _ i ^ f + s _ i ^ r \tag{11}$$

综合考虑正负效应后，image-text pair $(U, V)$ 的相似度为

$$S(U,V)=\frac 1 m \sum _ {i=1} ^ m (s _ i ^ {neg} + s _ i ^ {pos}) \tag{12}$$

### 2.1.3 采样和更新策略

本节介绍如何采样得到 (1) 和 (2) 式中的 mismatched 和 matched word-region 片段。

对于一个匹配的文本，其中任意的单词一定可以在其匹配图像中找到至少一个匹配区域。将单词 $u _ i, i \in [1,m]$ 和正确的图像 region $\lbrace v _ j ^ +\rbrace _ {j=1} ^ n$ 最大相似度的作为 matched 样本，

$$s _ i ^ + = \max _ j (\lbrace v _ j ^ + u _ i ^ {\top} / (||v _ j ^ +|| \ ||u _ i||) \rbrace _ {j=1} ^ n ) \tag{13}$$

（这里的 $i$ 是也从文本角度出发。）

mismatched 的采样。对于单词 $u _ i, i \in [1,m]$，其与不正确的图像中的 region $\lbrace v _ j ^ - \rbrace _ {j=1} ^ n$ 的相似度最大值，作为 mismatched 样本，

$$s _ i ^ - = \max _ j (\lbrace v _ j ^ - u _ i ^ {\top} / (||v _ j ^ -|| \ ||u _ i ||)\rbrace _ {j=1} ^ n) \tag{14}$$

mini-batch 中每个文本句子均进行采样，然后用于更新 $S _ k ^ +, S _ k ^ -$ ，$k$ 为训练 epoch。

## 2.2 目标函数

给定一个 gt image-text pair $(U, V)$，以及其不匹配的 pair $(U, V')$ 和 $(U', V)$，后两者使用最难不匹配 pair，通过下式选择，

$$V'=\argmax _ {p \neq V} S(U, p), \quad U'=\argmax _ {q \neq U} S(q, V)$$

作者着重优化这些最难不匹配样本，这些样本有着最高损失。使用的目标函数为

$$L = \sum _ {(U,V)} [\gamma - S(U,V)+S(U, V')] _ + \ + [\gamma - S(U, V)+S(U',V)] _ + \tag{15}$$

其中 $\gamma$ 是超参数边距 。

## 2.3 特征提取

**# 视觉表征**

给定一个图像 $V$，其又一系列显著区域的特征 $[v _ 1, v _ 2, \ldots, v _ n]$ 表示，显著目标和区域使用 Faster-RCNN 检测得到（Faster-RCNN 在 Visual Genome 上预训练），然后选择 topK（$K=36$）个 proposals。在图像上检测得到的 regions 通过预训练的 ResNet-101 得到 mean-pooled 的卷据特征，卷积输出特征的 channel 为 $C$，然后通过 GAP（global average pooling）得到长度为 $C$ 的向量，最后使用一个全连接层将每个 region 映射为 1024 长度的特征向量。

**# 文本表征**

假设一个文本 $U$，包含 $m$ 个 words，将每个 word 编码为 1024 长度的特征向量，得到 $[u _ 1, u _ 2, \ldots, u _ m]$ ，每个 word 的编码步骤为：

1. 每个 word 首先编码为 one-hot 向量（根据词典 Vocab 得到）
2. 然后编码为 GloVe 词嵌入向量
3. GloVe 向量通过 BiGRU，整合前向和反向上下文信息，最终的词向量 $u _ i$ 就是双向隐层状态节点值的平均。

# 3. 实验与源码

## 3.1 数据集和实现细节

**# 数据集**

1. Flickr30k

    包含 31000 个图像和 155000 个文本句子，每个图像有 5 个文本句子。Flickr30k 划分为 `1000` 个测试图像，`1000` 个验证图像，以及 `29000` 训练图像。

2. MS-COCO

    包含 123287 个图像，616435 个文本句子，每个图像有 5 个文本句子。数据集划分为 `5000` 个测试图像，`5000` 个验证图像，以及 `113287` 个训练图像。

**# 数据集源码**

```python
class PrecompData(data.Dataset):
    ...
    def __getitem__(self, index):
        '''index: 句子 index'''
        img_id = index // self.im_div   # 句子 index // 5 ，得到图像 index
        # 这里，self.images 就是从文件中读取的图像区域的经过均值池化后的卷积特征
        # 即，图像先经过Faster RCNN 得到目标区域，然后在经过 ResNet-101，得到
        # 最后均值池化的卷积特征
        image = torch.Tensor(self.images[img_id]) 
        caption_non = self.caption_non[index]   # 句子
        # 将句子分词
        tokens = nltk.tokenize.word_tokenize(str(caption_non).lower())
        caption_non = []
        caption_non.append(vocab('<start>'))    # 添加 <start> 在词典内的 id
        caption_non.extend([vocab(token) for token in tokens])
        caption_non.append(vocab('<end>'))

        captions = torch.Tensor(caption_non)    # word id
        return image, captions, index, img_id
    
def collate_fn(data):
    ...
    # images: (B, 3, H, W)
    # targets: (B, max_L)
    # lengths: 长度为 B 的列表，元素值表示句子中 token 数量
    # ids: 长度为 B 的列表，句子 id
    return images, targets, lengths, ids
```

**# 训练代码**

模型为 `NAAF` 类，其中训练代码为，

```python
class NAAF:
    ...
    # 训练方法入口
    def train_emb(self, images, captions, lengths, *args):
        img_emb, cap_emb, cap_lens = self.forward_emb(
            images, captions, lengths
        )

        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb, cap_lens)
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()
    # 前向传播，得到图像和文本的特征
    def forward_emb(self, images, captions, lengths, volatile=False):
        ...
        img_emb = self.img_enc(images)
        cap_emb, cap_lens = self.txt_enc(captions, lengths)
        return img_emb, cap_emb, cap_lens
```

以上给出了方法的主要实现代码，由于比较简单，不多解释。下面看获取图像和文本特征向量的实现代码，

**# 图像特征提取代码**

```python
class EncoderImagePrecomp(nn.Module):
    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size    # 图像和文本的 shared 特征空间维度
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)
        self.init_weights()     # 初始化 self.fc 这个 layer 的参数
    
    def forward(self, images):
        '''images: 图像经过均值池化后的卷积输出特征
                    (batch_size, 36, 2048)
        '''
        features = self.fc(images)  # (batch_size, 36, embed_size)
        if not self.no_imgnorm:     # 需要归一化
            features = l2norm(features, dim=-1)
        return features
```

上面是提取图像区域特征的代码，实际上就是一个全连接层，其输入也是图像区域特征向量，经过这个全连接层，映射到图像与文本 shared 特征向量空间中（维度变换）。这里的输入 `图像区域特征向量` 如何得到，可以参考 [bottom-up Attention](https://openaccess.thecvf.com/content_cvpr_2018/papers/Anderson_Bottom-Up_and_Top-Down_CVPR_2018_paper.pdf) 这篇论文，源码为 [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention) 。

**# 文本特征提取代码**

文本特征有两种提取方法：1. 使用自学习嵌入向量（即 pytorch 中的 `nn.Embedding`）；2. Glove 嵌入向量。以下分别介绍

```python
class EncoderText(nn.Module):
    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        self.embed = nn.Embedding(vocab_size, word_dim)
        self.rnn = nn.GRU(word_dim, embed_size, num_layers,
                          batch_first=True, bidirectional=True)
        self.init_weights() # 初始化以上 layers 的参数
    
    def forward(self, x, lengths):
        '''
        x: one-hot 向量，实际上是 token 在 vocab 中的 id
            (batch_size, max_L)
        参考上文 # 文本表征 的编码步骤 1，2，3
        '''
        # 将 one-hot 转为嵌入向量
        x = self.embed(x)   # (batch_size, max_L, word_dim)
        # 将输入 x 打包成 RNN 要求的形式，batch_first=True 指出 x 第一维为 batch
        packed = pack_padded_sequence(x, lengths, batch_first=True)
        out, _ = self.rnn(packed)

        # 将 RNN 输出再打包成 Tensor 形式
        padded = pad_packedsequence(out, batch_first=True)
        # (batch_size, max_L, D * embed_size), D = 2 if bidirectional else 1
        # (batch_size, )
        cap_emb, cap_len = padded
        cap_emb = (cap_emb[:, :, :cap_emb.size(2)//2] +
                   cap_emb[:, :, cap_emb.size(2)//2:]) / 2
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)
        return cap_emb, cap_len
```

使用 Glove 嵌入向量的文本特征提取代码如下，

```python
class GloveEmb(nn.Module):
    def __init__(self, num_embeddings, glove_dim, glove_path,
                 add_rand_embed=False, rand_dim=None, **kwargs):
        super(GloveEmb, self).__init__()
        self.num_embeddings = num_embeddings    # 实际是 vocab_size, V
        self.add_rand_embed = add_rand_embed
        self.glove_dim = glove_dim              # 实际是 embed_dim, d
        self.final_word_emb = glove_dim

        # glove 模型的参数矩阵 W ，shape 为 (V, d)
        self.glove = nn.Embedding(num_embeddings, glove_dim)
        glove = nn.Parameter(torch.load(glove_path)) # 从实现训练好的文件中加载 glove 模型参数
        self.glove.weight = glove
        self.glove.requires_grad = False

        if add_rand_embed:
            self.embed = nn.Embedding(num_embeddings, rand_dim)
            self.final_word_emb = glove_dim + rand_dim
    
    def forward(self, x):
        '''
        x: (batch_size, max_L)
        return: glove vector of x, (batch_size, max_L, glove_dim)
        '''
        emb = self.glove(x)
        if self.add_rand_embed:
            emb2 = self.embed(x)
            emb = torch.cat([emb, emb2], dim=2)

class GloveRNNEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, latent_size, num_layers=1,
        use_bi_gru=True, no_txtnorm=False, glove_path=None, add_rand_embed=True):
        super(GloveRNNEncoder, self).__init__()
        self.latent_size = latent_size
        self.no_txtnorm = no_txtnorm

        self.embed = GloveEmb(
            vocab_size,
            glove_dim=embed_dim,
            glove_path = glove_path,
            add_rand_embed=add_rand_embed,
            rand_dim=embed_dim,
        )
        self.use_bi_gru = use_bi_gru
        self.rnn = nn.GRU(
            self.embed.final_word_embed,    # glove 输出的词向量维度
            latent_size, num_layers,
            batch_first=True,
            bidirectional=use_bi_gru
        )
    
    def forward(self, captions, lengths):
        '''
        captions: (batch_size, max_L)  文本，每个单词的 id
        lengths: (batch_size, )      每个文本句子的长度
        '''
        # (batch_size, max_L, glove_dim)
        emb = self.embed(captions)  # 得到 glove vectors
        packed = pack_padded_sequence(emb, lengths, batch_first=True)
        out, _ = self.rnn(packed)   # 得到输出单词向量，以及隐层输出
        padded = pad_packed_sequence(out, batch_first=True)
        # cap_emb: (batch_size, max_L, D * latent_size), D=2 if bidirectional else 1
        cap_emb, cap_len = padded

        cap_emb = (cap_emb[:, :, :cap_emb.size(2)//2] +
                   cap_emb[:, :, cap_emb.size(2)//2:]) / 2
        if not self.no_txtnorm:     # 需要归一化
            cap_emb = l2norm(cap_emb, dim=-1)
        return cap_emb, cap_len
```

**# 损失计算代码**

```python
class NAAF:
    ...
    def forward_loss(self, img_emb, cap_emb, cap_len, **kwargs):
        # 计算具有正负注意力的相似度得分，参见上文 (12) 式
        scores = xattn_score(img_emb, cap_emb, cap_len, self.opt)
        loss = self.criterion(scores, img_emb.size(0))
        return loss
```

上述代码中，`xattn_score` 方法计算具有正负注意力的相似度得分 (12) 式，且实现采样正负样本 (13) 和 (14) 式，并得到正负样本的分布，然后根据 (4) 式更新边界值 $t_k$，

```python
def xattn_score(images, captions, cap_lens, opt):
    '''
    images: 图像特征，(batch_size, n_regions, d)，d 是图像文本 shared 特征空间维度
    captions: 文本特征，(batch_size, max_L, d)
    cap_lens: 文本句子长度 (batch_size, )
    '''
    similarities = []
    max_pos = []        # (1) 式，S+
    max_neg = []        # (2) 式，S-
    max_pos_aggre = []
    max_neg_aggre = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    cap_len_i = torch.zeros(1, n_caption)
    n_region = images.size(1)       # 每个图像中的区域数量。K
    batch_size = n_image            # mini batch size
    N_POS_WORD = 0
    A = 0
    B = 0
    mean_pos = 0
    mean_neg = 0

    for i in range(n_caption):
        n_word = cap_lens[i]    # 当前文本句子长度
        cap_i = captions[i, :n_word, :].unsqueeze(0).continuous()   # 提取当前句子中的单词特征
        cap_len_i[0, i] = n_word
        cap_i_expand = cap_i.repeat(n_image, 1, 1)  # (n_image, n_word, d)

        # ======================= 求 (7) 式 ========================
        # text-to-image direction。目的是为了计算得到 (7) 式
        t2i_sim = torch.zeros(batch_size*n_word).double().cuda()
        contextT = torch.transpose(images, 1, 2)    # 图像上下文，(n_image, d, n_regions)
        # do batch matrix multiply operation, (b, n, m) * (b, m, p) -> (b, n, p)
        # attn: (6) 式，由于图像和文本特征均已归一化，所以 (6) 式分母省略了
        attn = torch.bmm(cap_i_expand, contextT)    # (n_image, n_word, n_regions)
        attn_i = torch.transpose(attn, 1, 2).contiguous()   # (n_image, n_regions, n_word)
        # opt.thres: (7) 式中的 tk，
        attn_thres = attn - torch.ones_like(attn) * opt.thres   # (n_image, n_word, n_regions)
        # n_image, n_word, n_regions
        batch_size, queryL, sourceL = images.size(0), cap_i_expand.size(1), images.size(1)
        attn_row = attn_thres.view(batch_size * queryL, sourceL)
        # 求 (7) 式，即 max_j
        Row_max = torch.max(attn_row, 1)[0].unsqueeze(-1)   # (n_image * n_word, 1)
        # ======================= 求 (7) 式 ========================

        # ======================= 求 (8) 式 ========================
        attn_neg = Row_max.lt(0).float()
        t2i_sim_neg = Row_max * attn_neg
        t2i_sim_neg = t2i_sim_neg.view(batch_size, queryL)  # (n_image, n_word)
        # 注意！！！这表示当前句子中的每个单词与 minibatch 内所有图像的负注意力相似度
        # 下同
        # ======================= 求 (8) 式 ========================

        # ======================= 求 (11) 式 ========================
        # w_ij^{inter}，参考 (10) 式
        attn_pos = get_mask_attention(attn_row, batch_size, sourceL, queryL, opt.lambda_softmax)
        # batch matrix multiply
        # 计算 单词 `i` 在整个图像的共享语义 \hat v_i
        weiContext_pos = torch.bmm(attn_pos, images)    # (n_image, n_word, d)
        # 计算 (10.1) 式，(n_image, n_word)
        t2i_sim_pos_f = cosine_similarity(cap_i_expand, weiContext_pos, dim=2)
        # 计算 w_ij^{relev}，(n_image, n_word, n_regions)
        attn_weight = inter_relations(attn_i, batch_size, n_region, n_word, opt.lambda_softmax)
        # 计算 s_i^r，参考 (11) 式
        # 两个tensor shape 均为 (n_image, n_word, n_regions)，然后沿着 region 维度求和
        t2i_sim_pos_r = attn.mul(attn_weight).sum(-1)   # (n_image, n_word)
        t2i_sim_pos = t2i_sim_pos_f + t2i_sim_pos_r     # (11) 式，(n_image, n_word)
        # ======================= 求 (11) 式 ========================

        t2i_sim = t2i_sim_neg + t2i_sim_pos
        sim = t2i_sim.mean(dim=1, keepdim=True)         # (12) 式，(n_image, )
        # sim 是当前句子与 minibatch 所有图像的综合相似度

        # Discriminative Mismatch Mining
        # 倒序排列，取最大值所在的 index，也就是相似度最大的图像在 batch 中的 index
        wrong_index = sim.sort(0, descending=True)[1][0].item()
        if (wrong_index == i):     # I, T 匹配
            # attn: s_ij (n_image, n_word, n_regions)
            # (13) 式，(n_image * n_word)
            attn_max_row = torch.max(attn.reshape(batch_size * n_word, n_region).squeeze(), 1)[0].cuda()
            # 获取第 `i` 句子的 n_word 个单词与第 `i` 个图像的某区域最大 s_ij，(n_word, )
            # 正样本
            attn_max_row_pos = attn_max_row[(i * n_word) : (i * n_word + n_word)].cuda()

            # 正序排列，取最小值所在的 index，即最小相似度所关联的图像 index
            neg_index = sim.sort(0)[1][0].item()
            # 负样本，参考 (14) 式，(n_word, )
            attn_max_row_neg = attn_max_row[(neg_index * n_word) : (nex_index * n_word + n_word)].cuda()

            max_pos.append(attn_max_row_pos)
            max_neg.append(attn_max_row_neg)
            N_POS_WORD = N_POS_WORD + n_word
            if N_POS_WORD > 200:    # 样本数超过 200
                # 开始更新 tk，根据 (4) 式
                max_pos_aggre = torch.cat(max_pos, 0)
                max_neg_aggre = torch.cat(max_neg, 0)
                mean_pos = max_pos_aggre.mean().cuda()
                mean_neg = max_neg_aggre.mean().cuda()
                stnd_pos = max_pos_aggre.std()
                stnd_neg = max_neg_aggre.std()

                A = stnd_pos.pow(2) - stnd_neg.pow(2)   # beta_1^k
                B = 2 * ((mean_pos * stnd_neg.pow(2)) - (mean_neg * stnd_pos.pow(2)))
                C = (mean_neg * stnd_pow).pow(2) - (mean_pos * stnd_neg).pow(2) + 2 * (stnd_pos * stnd_neg).pow(2) * torch.log(stnd_neg/(opt.alpha*stnd_pos)+1e-8)

                thres = opt.thres
                thres_safe = opt.thres_safe
                opt.stnd_pos = stnd_pos.item()
                opt.stnd_neg = stnd_neg.item()
                opt.mean_pos = mean_pos.item()
                opt.mean_neg = mean_neg.item()

                E = B.pow(2) - 4 * A * C
                if E > 0:
                    # (4) 式更新 tk
                    opt.thres = ((-B + torch.sqrt(E)) / (2*A + 1e-8)).item()
                    # (4.1) 式
                    opt.thres_safe = (mean_pos - 3*opt.stnd_pos).item()
                if opt.thres < 0 or opt.thres > 1:
                    opt.thres = 0
                if opt.thres_safe < 0 or opt.thres_safe > 1:
                    opt.thres_safe = 0
                
                opt.thres = 0.7 * opt.thres + 0.3 * thres
                opt.thres_safe = 0.7 * opt.thres_safe + 0.3 * thres_safe
        
        if N_POS_WORD < 200:
            opt.thres = 0
            opt.thres_safe = 0
        similarities.append(sim)
    similarities = torch.cat(similarities, 1)   # (n_image, n_caption)
    return similarities

# 计算 单词 `i` 在整个图像的共享语义的权重 w_ij^{inter}，参考 (10) 式
def get_mask_attention(attn, batch_size, sourceL, queryL, lamda=1):
    '''
    attn: s_ij - tk，(n_image * n_word, n_regions)
    '''
    mask_positive = attn.le(0)
    attn_pos = attn.masked_fill(mask_positive, torch.tensor(-1e9))
    attn_pos = torch.exp(attn_pos * lamda)  # (10) 式，
    attn_pos = l1norm(attn_pos, 1)  # 取绝对值
    attn_pos = attn_pos.view(batch_size, queryL, sourceL)
    return attn_pos     # (n_image, n_word, n_regions)

# word 与 region 之间的关联得分 s_ij 也反映了相似度，计算了基于 s_ij 的加权相似度的权重
# 计算 w_ij^{relev}
def inter_relations(attn, batch_size, sourceL, queryL, xlambda):
    '''
    attn: s_ij，(n_image, n_regions, n_word)
    '''
    attn = nn.LeakyReLU(0.1)(attn)   # [s_ij]+
    attn = l2norm(attn, 2)           # \overline s_ij，注意是沿着 word 维度归一化
    attn = torch.transpose(attn, 1, 2).contiguous() # (n_image, n_word, n_regions)
    attn = attn.view(batch_size * queryL, sourceL)  # (n_image * n_word, n_regions)
    attn = nn.Softmax(dim=1)(attn * xlambda)        # 沿着 region 维度求 softmax
    attn = attn.view(batch_size, queryL, sourceL)
    return attn # (n_image, n_word, n_regions)
```

最后计算损失，相关代码如下，

```python
class ContrativeLoss(nn.Module):
    ...
    def forward(self, scores, length):
        '''
        scores: xattn_score 方法的返回值，S(U, V) 参见 (12) 式、
                shape 为 (n_image, n_caption)
        length: n_image
        '''
        # 得到 matched (U, V) 的综合相似度 S(U, V)，参见 (12) 式
        diagonal = scores.diag().view(length, 1)    # (n_image, 1)
        d1 = diagonal.expand_as(scores) # (n_image, n_caption)
        d2 = diagonal.t().expand_as()   # (n_caption, n_image)

        # (15) 式第二项，S(U', V)，后缀 s 表示 sentence
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # (15) 式第一项，S(U, V')，后缀 im 表示 image
        cost_im = (self.margin + scores - d2).clamp(min=0)

        mask = torch.eye(scores.size(0)) > 0.5  # 对角线为 True
        I = Variable(mask).cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        if self.max_violation:
            # 其实是 (15) 第一项还是第二项无所谓，只要
            # 如果第一维 expand，那么这里沿着第一维 max
            # 如果第二维 expand，那么这里沿着第二维 max
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()
```