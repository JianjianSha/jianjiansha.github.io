---
title: 基于 transformer 的图像分类模型
date: 2023-08-03 22:37:05
tags: 
    - image classification
    - transformers
mathjax: true
---

# 1. ViT

论文名称：[AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE](https://arXiv.org/abs/2010.11929)

将 size 为 `(C, H, W)` 的图像切割分成 `N=H/patch_size * W/patch_size` 个块，那么一个 batch 的图像经过切割得到 `(B, N, patch_size*patch_size*C)` 的块序列，经过 learnable 嵌入矩阵，得到嵌入向量 `(B, N, D)`，然后每个块还需要一个 Position 嵌入向量（可以是正弦位置编码，或者是 learnable 嵌入矩阵），两种嵌入向量相加后送入网络，如下图，

![](/images/img_cls/vit_1.png)


网络使用 transformer encoder（注意没有 decoder），计算块的位置嵌入向量时，块的位置从第二个开始计算，第一个位置留着给 `<class>` 的嵌入向量，那么经过 transformer encoder 输出每个位置的特征向量，第一个位置也就是对应 `<class>` 这个位置的特征向量，用于对图像的分类，分类网络为一个线性变换层。

`<class>` 位置处嵌入向量也是一个 learnable 向量，shape 为 `(D,)`，直接将这个向量送入 transformer 即可。transformer encoder 输出特征的 shape 为 `(B, N, D)` 。

# 2. DeiT

论文：[Training data-efficient image transformers & distillation through attention](https://arXiv.org/abs/2012.12877)

ViT 训练需要超大的数据集（size 达到上亿），否则效果不如 CNN based 模型如 EfficentNet 等。为了能使得 transformer based 的模型可使用 ImageNet 训练，作者提出了 Data-effificient image Transformers (DeiT) ，引入了蒸馏学习。

**# 知识蒸馏**

一个大的复杂的模型作为 teacher 模型，一个简单的模型作为 student 模型。先在 teacher 模型上预训练，训练完成后，再来训练 student 模型，此时 student 可以利用图像的 soft label，这个 soft label 来自 teacher 模型输出。例如数据集分类共 $C$ 个，teacher 模型（经过 softmax 之后的）输出为 $\psi(Z _ t)$，这是一个 $C$ 维向量，$\psi$ 为 softmax，$Z _ t$ 为模型非归一化得分。那么一个图像的 true label 为 one-hot 向量，或者是向量中 `1` 的下标 $y$ ，而 soft label 则为 $\psi(Z _ t)$ ，硬指标为 $y _ t = \arg \max _ c \psi(Z _ t)$ 。

**# 模型**

DeiT 基于 ViT，增加了一个表示蒸馏的 token ，如下图所示，

![](/images/img_cls/deit_1.png)

## 2.1 蒸馏

 teacher 模型使用一个已经事先训练好的模型，例如 RegNetY 。

**# 软蒸馏**

记 $Z _ t$ 和 $Z _ s$ 分别是 teacher 和 student 模型的输出 logits（非归一化得分）。软蒸馏使用 KL 损失，

$$L _ {global} = (1-\lambda) L _ {CE}(\psi (Z _ t), y) + \lambda \tau ^ 2 KL( \psi(Z _ s / \tau), \psi ( Z _ t / \tau))$$

上式中，$\tau$ 为温度。

损失包含两部分，两者使用 $\lambda$ 平衡，

1. student 模型预测概率与 gt label 的交叉熵损失
2. student 与 teacher 预测概率的 KL 损失

上式第二项还有一个 $\tau ^ 2$ 因子，原因如下：

<details><summary>展开</summary>

为了简洁起见，记 $q = \psi(Z _ s /\tau), \ p = \psi(Z _ t/ \tau)$，student 模型输出 $z = Z _ s$，交叉熵损失记为 $L=\sum _ j - p _ j \log q _ j$，那么

$$\begin{aligned} \frac {\partial L} {\partial z _ i} &=\sum _ j -\frac {p _ j}{q _ j} \frac {\partial q _ j}{\partial z _ i}
\\\\ &=-\sum _ {j\neq i} \frac {p _ j} {q _ j} \frac { - \exp (z _ j /\tau)\exp (z _ i /\tau) /\tau}{(\sum \exp) ^ 2}-\frac {p _ i}{q _ i}\frac {(\sum \exp)\exp (z _ i /\tau)\tau  - \exp ^ 2(z _ i /\tau) /\tau}{(\sum \exp) ^ 2}
\\\\ &= \sum _ {j\neq i} \frac {p _ j} {q _ j} \frac {q _ j q _ i} {\tau} - \frac {p _ i}{q _ i}(q _ i - q _ i ^ 2)/ \tau
\\\\ &=\sum _ j \frac {p _ j} {q _ j} \frac {q _ j q _ i} {\tau}- p _ i / \tau
\\\\ &= \frac {q _ i} {\tau}\sum _ j  p _ j - p _ i / \tau
\\\\ &= \frac 1 {\tau}(q _ i - p _ i)
\end{aligned}$$


在温度足够高的情况下，

$$q _ i = \frac {\exp (z _ i / \tau)}{\sum _ j \exp (z _ j /\tau)}\approx \frac {1+z _ i/\tau}{C + \sum _ j z _ j /\tau}$$

假设 logits $z$ 是 zero-mean ，那么 $\sum _ j z _ j = 0$，上式转为

$$q _ i \approx \frac 1 C (1+\frac {z _ i} {\tau})$$

同理，

$$p _ i \approx \frac 1 C (1+\frac {v _ i}{\tau})$$

于是

$$ \frac {\partial L} {\partial z _ i} \approx \frac 1 {C \tau ^ 2}(z _ i - v _ i)$$

$q$ 的熵记为 $E$，即 $E=-\sum _ j q _ j \log q _ j$

$$\begin{aligned}\frac {\partial E}{\partial z _ i} &=-\sum _ j (\log q _ j+ 1)\frac {\partial q _ j}{\partial z _ i}
\\\\ &=\sum _ {j\neq i} (1+\log q _ j)\frac {q _ j q _ i} {\tau} - (1+\log q _ i)\frac {q _ i - q _ i ^ 2}{\tau}
\\\\ &=\sum _ j (1+\log q _ j)\frac {q _ j q _ i}{\tau} - (1+\log q _ i)\frac {q _ i}{\tau}
\\\\ &=\sum _ j \log q _ j \frac {q _ j q _ i}{\tau} - \sum _ j \log q _ i\frac {q _ j q _ i}{\tau}
\\\\ &=\sum _ j \log\left(\frac {\exp (z _ j /\tau)}{\exp (z _ i /\tau)}\right)\frac {q _ j q _ i}{\tau}
\\\\ &=\sum _ j \frac {1}{\tau ^ 2}(z _ j - z _ i) q _ i q _ j
\\\\ &=\frac {q _ i} {\tau ^ 2}\left [ \sum _ j z _ j q _ j - z _ i\right]
\\\\ & \approx \frac 1 {C \tau ^ 2}\left(1+ \frac {z _ i}{\tau}\right)\left(\frac 1 {C \tau}\sum _ j z _ j ^ 2 - z _ i\right)
\end{aligned}$$

根据 

$$\frac {\partial KL}{\partial z _ i}=\frac {\partial L} {\partial z _ i}+ \frac {\partial E} {\partial z _ i}$$

上式第二项，分母量级至少是 $\tau ^ 2$ ，综合考虑，KL 损失需要乘以因子 $\tau ^ 2$ 。

</details>

<br/>

**# 硬蒸馏**

使用 teacher 模型的输出预测分类 $y _ t = \arg \max \psi (Z _ t)$ 作为标签，使用交叉熵，总的损失如下，

$$L=\frac 1 2 L _ {CE}(\psi (Z _ s),y) + \frac 1 2 L _ {CE}(\psi (Z _ s), y _ t)$$

hard label 也可以通过平滑处理转为 soft label，此时真实 label 的概率为 $1-\epsilon$，其余 label 的概率之和为 $\epsilon$ 。


**# inference**

测试阶段，分类 token 和蒸馏 token 经过 transformer 生成的特征向量，分别经过一个线性 layer，得到两个分类预测得分（非归一化）向量，然后两个向量相加除以 2，得到最终的预测得分向量，

$$\begin{aligned}\mathbf z _ {cls} &= W _ {cls} ^ {\top} \mathbf h _ {cls} + \mathbf b _ {cls}
\\\\ \mathbf z _ {dis} &= W _ {dis} ^ {\top} \mathbf h _ {dis} + \mathbf b _ {dis}
\\\\ \mathbf z &= \frac {\mathbf z _ {cls} + \mathbf z _ {dis}} 2 \end{aligned}$$

# 3. Visual Transformer

论文：[Visual Transformers: Token-based Image Representation and Processing for
Computer Vision](https://arXiv.org/abs/2006.03677)

CV 中，将图像表示为一个像素值数组，然后使用卷积得到 local 特征，但是存在以下问题：

1. 像素并非均等地产生

    例如图像分类中，应该优先考虑前景目标；分割模型优先考虑行人而非天空，路面等。卷积则均匀的处理图像中所有的分块（patches），而不考虑其重要性。

2. 图像并非拥有全部概念

    图像的角落和边缘存在于所有图像中，故对所有图像使用低层卷积核是合适的。高层特征例如“耳朵”的形状，仅存在于部分图像中，所以对所有图像使用高层卷积核导致计算效率低。

3. 卷积难以获取空间距离大的概念

    因为卷积只针对局部区域。

为解决以上问题，作者提出了 Visual Transformer（VT），如下图，使用高层概念表示图像，

![](/images/img_cls/vt_1.png)

<center>Visual Transformer 结构图</center>

使用空间注意力机制将特征 map 转为一个小的语义 tokens 集合（16个 tokens），然后送入 transformer 。VT 可用于分类和分割任务。

如上图，整个过程为：

1. 输入图像经过若干个 conv blocks，得到输出 feature maps，学习得到密集 low-level 样式。


2. 将上一步的输出特征送入 VT 模块

## 3.1 tokenizer

一个图像可以使用若干个视觉 tokens 表示，这与卷积中使用几百上千个卷积 filters 成鲜明对比。使用一个 tokenizer 将 conv 输出特征转为视觉 tokens，记 tokenizer 输入特征为 $X \in \mathbb R ^ {HW \times C}$，视觉 tokens 为 $T \in \mathbb R ^ {L \times C}$，其中 $L \ll HW$ 。


以下介绍两种 tokenizer 。

### 3.1.1 filter-based tokenizer

对输入特征 map 应用一个 `1x1` 卷积，卷积核参数为 $W _ A \in \mathbb R ^ {C \times L}$，然后使用 softmax 得到一个 pixel 分配到 $L$ 个语义组中每个组的权重，即，这是一个权重矩阵，表示 $HM$ 个 pixels 与 $L$ 个语义组的关联权重，那么一个语义 token 为所有 pixels 的加权和，

$$T = [\underbrace {\text{softmax} (X W _ A)} _ {A \in \mathbb R ^ {HW \times L}} ] ^ {\top} \ X$$

其中 softmax 沿着 $HW$ 方向进行，也就是说其分母是 $HW$ 个项之和。这是因为 $L$ 个 tokens，每个 token 均使用 $HW$ 个特征（即，整个特征 map）加权和来表示一个语义概念。

过程如下图所示，

![](/images/img_cls/vt_2.png)

缺点：很多高级语义概率是稀疏的，可能只存在于少数图像中，故使用固定的权重 $W _ A$ 浪费计算。

### 3.1.2 recurrent tokenizer

为解决 filter-based tokenizer 的缺点，提出循环 tokenizer，其中权重与上一层视觉 tokens 相关。

$$\begin{aligned} W _ R &= T _ {in} W _ {T \rightarrow R}
\\\\ T &= [\text{softmax} (X W _ R)] ^ {\top} \ X
\end{aligned}$$

其中 $W _ {T \rightarrow R} \in \mathbb R ^ {C \times C}$，$T _ {in} \in \mathbb R ^ {L \times C}$ 。整个过程如下图所示

![](/images/img_cls/vt_3.png)

问题：$T _ {in}$ 值如何确定？

## 3.2 transformer

这里的 transformer 与标准的相比，做了一些小修改，直接给出计算式如下，

$$\begin{aligned} T' _ {out} &= T _ {in} + \text {softmax} ((T _ {in} K)(T _ {in} Q) ^ {\top}) T _ {in}
\\\\ T _ {out} &= T' _ {out} + \text{ReLU}(T ' _ {out} F _ 1) F _ 2
\end{aligned}$$

其中 $T _ {in}, T ' _ {out}, T _ {out} \in \mathbb R ^ {L \times C}$，$F _ 1 , F _ 2 \in \mathbb R ^ {C \times C}$ 。$T _ {in}, \ T _ {out}$ 为 transformer 的输入输出。

$(T _ {in} K)(T _ {in} Q) ^ {\top} \in \mathbb R ^ {L \times L}$

## 3.3 projecter

某些视觉任务需要 pixel 细节信息（例如分割任务），但是视觉 tokens 中没有保留 pixel 信息，因此将 transformer 输出与特征 map 进行融合，

$$X _ {out} = X _ {in} + \text{softmax} ((X _ {in} W _ Q)(T W _ K)^{\top}) T$$

其中 $X _ {in}, X _ {out} \in \mathbb R ^ {HL \times C}$ 分别是 VT 模块的输入特征和输出特征（例如 ResNet，最后一个 stage 使用 2/3 个 VT 替代）。$T \in \mathbb R ^ {L \times C}$ 是 transformer 的输出。

$W _ Q, W _ K \in \mathbb R ^ {C \times C}$

## 3.4 分类模型

backbone 来自 ResNet，使用了 ResNet-{18, 34, 50, 101} 四种，但是将 ResNet 中最后一个 conv stage 换成了 VT 模块。ResNet-{18, 34, 50, 101} 最后一个 conv stage 分别包含了 2 blocks，3 blocks，3 blocks，3 blocks，所以分别使用 (2, 3, ,3, 3) 个 VT 模块替换。

ResNet-{18, 34} backbone 输出 shape 为 $14 ^ 2 \times 256$，ResNet-{50, 101} backbone 输出 shape 为 $14 ^ 2 \times 1024$，所以设置 VT 的特征 channel size 为 (256, 256, 1024, 1024) （这是 $X _ {in}$ 中的 $C$ ）

采用 16 个视觉 tokens，即 $L =16$ ，设置 16 个视觉 tokens 的维度为 1024 （这是 $T _ {out}$ 中的 $C$ ）

**最后对 16 个视觉 tokens 使用均值池化，得到 $C$ 维向量，然后使用一个 FC layer，得到分类预测得分**。

整个网络层结构参数如下表所示

![](/images/img_cls/vt_4.png)

上表中，stage 5 为 VT 模块组。其中 `VT-C512-L16-CT1024 ×2` 表示 VT 模块输出特征 channel size 为 `512`，视觉 token 的 channel size 为 `1024`，视觉 tokens 数量为 `16`，stage 5 中使用 `2` 个 VT 模块。

疑问：所有的 $C$ 不应该相等吗，否则如何实现 element-wise 相加？应该要相等（或者使用线性映射使的各个 channel 相等）

## 3.5 语义分割

使用 全景 FPN 作为 baseline。

全景 FPN 使用 ResNet 作为 backbone 提取图像特征，不同的 stage 得到不同 size 的特征，这些特征使用 feature pyramid 网络进行融合，如下图左边，

![](/images/img_cls/vt_5.png)

FPN 中 high-level 特征的 channel 非常大，所以卷积计算量也大，使用 VT 代替卷积，修改过的网络称为 VT-FPN，如图右边，

1. 对每个分辨率的特征，VT 抽取 8 个视觉 tokens，每个 token 的 channel 为 `1024` 。

2. 所有 layers 的视觉 tokens 全部送入 transformer，transformer 输入输出 shape 相同

3. 将 transformer 输出映射回原来各层特征，这个映射操作就是上面的 projector 子模块

计算量减少，原来 FPN 中各层特征经过卷积（左图中灰色与棕色之间的箭头），现在改为经过 VT。

## 3.6 基于聚类的 tokenizer

这是为了解决 filter-based tokenizer 的局限性，filter-based tokenizer 对每个 image 均使用相同的 filters，这显然不是很合适，所以对一个具体的图像，考虑从图像自身通过对像素聚类以提取概念。

一个图像的特征 $X$ 视作像素集合 $\lbrace X _ p \rbrace _ {p=1} ^ {HW}$，使用 K-means 找出 $L$ 个中心，这 $L$ 个中心组成矩阵 $W _ K \in \mathbb R ^ {C \times L}$ ，每个中心均表示图像中的一个语义概念。

将 filter-based tokenizer 中的 $W _ A$ 替换为 $W _ K$，

$$\begin{aligned} W _ K &= \text{Kmeans}(X)
\\\\ T &= [\text{softmax} (X W _ K)] ^ {\top} X
\end{aligned}$$

**K-means 算法思想：**

1. 将所有 pixel 归一化为单位向量（L2 范数为 1）
2. 初始化中心的方法：

    将输入特征下采样，`(B, C, H, W)` -> `(B, C, H', W')` -> `(B, C, L)`，其中 `H' x W' = L`

3. 使用 Lloyd 算法，生成最终的中心

过程如下图所示，

![](/images/img_cls/vt_6.png)

伪代码如下，

```python
def kmeans(X_nchw, L, niter):
    # Input:
    # X_nchw - feature map
    # L - num token
    # niter - num iters of Lloyd’s
    N, C, H, W = X_nchw.size()
    # Initialization as down-sampled X， L 个中心
    U_ncl = downsample(X).view(N, C, L) # 这里 X 就是 X_nchw 吧？

    X_ncp = X_nchw.view(N, C, H*W) # p = h*w
    # Normalize to unit vectors
    U_ncl = U_ncl.normalize(dim=1)  # 每个 pixel 维度为 C
    X_ncp = X_ncp.normalize(dim=1)  # 每个中心维度为 C，全部归一化
    for _ in range(niter): # Lloyd’s algorithm
        dist_npl = (
            X_ncp[..., None] - U_ncl[:, :, None, :]
        ).norm(dim=1)       # (N, HW, L)
        # 找出 L 个中心中，与 pixel 距离最小的那个中心，此中心 mask=1，其他中心 mask=0
        mask_npl = (dist_npl == dist_npl.min(dim=2))    # (N, HW, L)
        U_ncl = X_ncp.MatMul(mask_npl)          # (N, C, L)
        U_ncl = U_ncl / mask_npl.sum(dim=1)     # (N, C, L)
        U_ncl = U_ncl.normalize(dim=1)          # 归一化
    return U_ncl # centroids
```

上述代码中，`dist_npl` 的 shape 为 `(N, HW, L)`，表示图像特征 pixel 与图像特征中心之间的距离矩阵。

`mask_npl` 表示 N 个 mask 矩阵，每个 mask 矩阵的行表示 pixel 属于哪个中心，列表示属于这个中心有哪些 pixel。

`X_ncp` 与 `mask_npl` 矩阵相乘，表示每个 centroid 所包含的那部分 pixels 向量相加

`U_ncl / mask_npl.sum(dim=1)` 表示再求均值，即，centroid 中所包含的 pixels 向量求平均，得到新的 centroid 向量


# 4. T2T-ViT

论文：[Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet](https://arXiv.org/abs/2101.11986)

源码：[yitu-opensource/T2T-ViT](https://github.com/yitu-opensource/T2T-ViT)

ViT 将 transformer 引入到视觉任务中，但是如果使用中等大小数据集如 ImageNet，那么性能不如 CNN。原因为：

1. 简单的将图像划分为 `16x16` 个 patches（tokenization）并不能很好地对局部结构如边缘和像素线建模
2. attention backbone 设计比较冗余，如果固定计算量和训练集样本数量，那么提取地特征丰富程度将不足

为解决以上问题，提出了 Tokens-To-Tokens Visual Transformer（T2T-ViT）。

## 4.1 Tokens-to-Token ViT

这是为了解决 ViT 中简单对图像 tokenize 地局限性。

T2T-ViT 包含两个组件：

1. Tokens-to-Token 模块，对图像地局部结构信息建模
2. T2T-ViT backbone 从 tokens 中提取全局注意力

如下图所示，

![](/images/img_cls/t2t_vit_1.png)

### 4.1.1 Tokens-to-Token

T2T 模块逐步对 image 进行 tokenize 并对局部结构信息建模，每一步都可以减少 tokens 数量。T2T 分两步：1. Re-structurization；2. Soft Split (SS)

**# Re-structurization**

过程如下图所示，

![](/images/img_cls/t2t_vit_2.png)

给定从上一个 transformer layer 得到的 tokens $T$，它首先通过 self-attention block （有时也称为一个 transformer layer）转换为

$$T' = \text{MLP}(\text{MSA}(T)) \tag{4.1}$$

其中 MSA 表示 self-attention layer，MLP 表示多次感知机。

(4.1) 式这个操作称为 T2T transformer，如上图蓝色部分所示。

然后将 $T'$ reshape 为一个图像（特征），

$$I = \text{Reshape}(T') \tag{4.2}$$

其中 $T' \in \mathbb R ^ {l \times c}, \ I \in \mathbb R ^ {h \times w \times c}$ ，$c$ 为 channel size，$l$ 为 tokens 数量，$l=h \times w$ 。

根据 (4.2) 式，可见完成了一次图像的 re-structurization 。

**# Soft Split**

对 re-structurized 图像应用 soft split，对图像的局部结构信息建模，并减少 tokens 数量。为了避免信息损失，将图像 split 为具有部分重叠的 patches 。 每个 patch 由周围 patches 校正以建立一个先验：此 patch 与周围 patches 有较强的联系。每次划分出的 tokens 进行 concatenate 操作得到一个大的 token ，如上图所示，称此操作为 soft split 。

soft split 中，每个 patch 的 size 记为 $k \times k$，overlapping size 为 $s$，图像（特征）使用 $p$ padding 。类比卷积 window 滑动，$k-s$ 类似于 stride，例如

1. $s=0$，stride 为 $k$，这样卷积 window 滑动过程中没有重叠
2. $s > 0$，那么 stride 就要减小 $s$，即 stride 为 $k-s$

对于 re-structurized 图像 $I \in \mathbb R ^ {h \times w \times c}$，soft split 之后的输出 tokens 数量为

$$l _ o = \lfloor \frac {h + 2p - k} {k-s} + 1 \rfloor \times \lfloor \frac {w + 2p - k} {k-s} + 1\rfloor \tag{4.3}$$

上式其实就是卷积输出平面 size 展开为一维。

每个 patch size 为 $k \times k \times c$，展开为一维向量，那么 $l _ o$ 个输出 patches 为 $T _ o \in \mathbb R ^ {l _ o \times ck ^ 2}$ ，这个 $T _ o$ 为输出 tokens，由于 stride $k -s > 1$，所有输出 tokens 数量比输入 tokens 数量减少。

**# T2T 模块**

结合 Re-structurization 和 Soft Split 两个过程，就是一次迭代过程。T2T 模块逐步减少 tokens 数量，并提取图像的局部信息。整个迭代过程为，

$$\begin{aligned} T' _ i &= \text{MLP}(\text{MSA}(T _ i))
\\\\ I _ i &= \text{Reshape}(T' _ i)
\\\\ T _ {i+1} &= \text{SS}(I _ i), \quad i =1,\ldots, (n-1)
\end{aligned} \tag{4.4}$$

对于输入图像 $I _ 0$，首先应用一个 soft split 生成 $T _ 1 = \text{SS} (I _ 0)$ 。

由于 T2T 模块操作过程中，tokens 数量（尤其是刚开始迭代过程中的 tokens 数量）比 ViT 中的 `16x16` 大，所以 MAC（内层访问量）和内存使用量均巨大，为了降低，设置 T2T 模块的 channel size 较小（32 或 64），并可选择地使用高效 transformer 如 Performer。

### 4.1.2 T2T-ViT backbone

ViT 的 backbone 中很多 channel 都是无效的（原论文中 Fig 2 对中间特征画图分析）。本文设计一个与 ViT 不同的结构，借鉴 CNN 中的一些设计，提高 backbone 的效率以及增加特征的语义丰富性。

由于 transformer layer 中有一个 skip 连续，与 ResNet 相同，类似地，借鉴 DenseNet 中的网络设计，使用密集连接，增强连接性以及特征语义丰富性，或者借鉴 Wide-ResNets 或 ResNeXt 结构，从而可修改 ViT backbone 中的 channel size 。本文考虑以下 5 中设计：

1. DenseNet 中的密集连接

2. Wide-ResNets 中的 deep-narrow 以及 shallow-wide

    deep 是说网络 layers 数量，wide 是说网络中每一 layer 的 channel size

3. SENet 中的 channel 注意力（使用一个小网络，提取每个 channel 的权重，作为 channel 注意力）

4. ResNeXt 中，多头 layer，增大 heads 数量

5. GhostNet 的 Ghost 操作

以上框架设计应用到 ViT 中的细节说明见论文附录。

经过作者系统全面的实验，发现：

1. 使用 deep-narrow 结构，降低了 channel 维度，增加了 layers 层数，可以降低模型大小以及 MAC，并且性能得到提升

2. SE block 中的 channel 注意力也可以提高 ViT 性能，但是不如 deep-narrow 结构那么明显。

于是，作者为 T2T-ViT 设计出 deep-narrow 框架，其中 channel size 和隐层维度 $d$ 较小，但是网络层数 $b$ 变大。T2T 模块最后一步输出的 tokens $T _ f$，与一个 class token 做 concat 操作，然后加上正弦位置编码 $E$，这些操作与 ViT 中相同，


$$\begin{aligned} T _ {f _ 0} &=[t _ {cls}; T _ f] + E, & E \in \mathbb R ^ {(l+1) \times d}
\\\\ T _ {f _ i} &= \text{MLP}(\text{MSA}(T _ {f _ {i-1}})), & i=1,\ldots,b
\\\\ y &= \text{fc}(\text{LN}(T _ {f _ b}))
\end{aligned} \tag{4.5}$$

其中 $E$ 为正弦位置编码，$l$ 为 T2T 模型最后一步的输出 tokens 数量，LN 表示 layer norm，fc 为线性变换，得到各分类预测得分。

# 5. MAE

论文：[Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)

本文提出了一个简单有效的 masked autoencoder（MAE），用于视觉表征学习。如图 1，输入图像被划分为很多分块（pathces），并且将大多数 patches mask 掉，然后将剩余 patches 连接起来送入 encoder，encoder 输出一个中间表示，然后再将 masked 的 patches 拼接进来得到一个完整 image size ，再送入 decoder 输出原始图像。注意 encoder 和 decoder 是非对称设计。

![](/images/img_cls/mae_1.png)
<center>图 1.</center>

**Masking**

与 ViT 相同，将 image 划分为一定数量的 patches，任意两个 patches 之间没有重叠。随机选择一部分 patches，其余 patches 则 maske 掉。

**MAE encoder**

encoder 是一个 ViT 结构，将输入 patches 通过线性映射得到一组嵌入向量。输入 patches 就是随机选择的 patches，称为可见 patches 。

**MAE decoder**

decoder 的输入包含：

1. encoder 输出的可见 patches 的特征
2. mask tokens

    mask token 是共享的，可学习的向量，用于表征 missing patches，这些缺失的 patches 就是需要被预测出来的。


decoder 也是由一系列 transformer blocks 构成。

decoder 用于重建 image，而 encoder 用于生成 image 特征表示，所以训练好模型之后，如果需要用于识别任务，那么只需要用到 encoder 。

**重建目标**

MAE 输出为 所有 patches 的像素预测值，即，decoder 为每个 patch 输出一个表示像素值的向量。decoder 的最后一个 layer 为线性映射 layer，其输出的 channel size 应该等于单个 patch 中的像素数 ，输出 units 为所有 patches 数量。

loss 函数为重建 image 与原始 image 之间的所有像素值的 MSE ，实际上是只计算 masked patches 的像素值的 MSE 。

作者也研究了一个变体：只输出 masked patches 的像素的归一化值。计算一个 patch 内所有像素值的均值和标准差，然后可以计算出归一化像素值。使用这种归一化像素值可以提高生成质量。

**简单实现**

将 image 切分为若干个 patches，为每个 patches 生成嵌入向量：使用一个线性映射，输出结果再加上位置嵌入向量

将所有嵌入向量 shuffle，然后取前面的一定比例的 embedding vectors，后面的则全部丢弃。但是需要记住每个 embedding 对应的 patch 在原 image 中的位置。

上述 embedding vectors 送入 encoder，输出每个 patch 对应的特征表示（也是一个向量）

上述中间表示与 masked token 向量合并，然后 unshuffle ，即，将中间表示按其在原 image 中的位置进行填充，其余位置则使用 masked token embedding 填充。

然后加上位置 embedding，然后送入 decoder ，输出每个 patch 的向量（向量表示这个 patch 内所有像素值）。