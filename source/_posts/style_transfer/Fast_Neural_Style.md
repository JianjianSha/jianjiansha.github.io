---
title: 快速风格迁移
date: 2022-02-07 15:06:38
tags: style transfer
mathjax: true
categories: 风格迁移
---
快速风格迁移(fast stype transfer)论文总结。
<!--more-->
论文：[Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)

补充材料：[fast-style-supp](https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf)

# 1. 网络

风格迁移中，有两类图片，一类是风格图片，例如艺术家的作品，另一类是内容图片，例如照片，利用风格迁移将内容图像转换为具有艺术家风格的图片。

如图 1，
![](/images/style_transfer/Fast_Neural_Style_1.png)

图 1

网络由两部分组成：
1. 一个图像转换网络

    这是一个 deep residual CNN，其网络参数记为 $W$，将输入图像 $x$ 转换为输出图像 $\hat y$，记 $\hat y = f_W(x)$

2. 一个用于计算损失的网络

    这个网络是一个已经用图像分类数据集预训练好的 CNN 网络，在风格迁移的训练中，固定这个网络（其参数保持不变）。将此网络记为 $\phi$。$\phi$ 计算图像转换网络的输出 $\hat y$ 与目标图像 $y_i$ 之间的损失 $l_i(\hat y, y_i)$，$y_i$ 包括内容图像和风格图像，分别记为 $y_c$ 和 $y_s$。利用损失最小化来训练图像转换网络 $f_W$，

    $$W^{\star}=\arg \min_W \mathbf E_x,\{y_i\} \left[\sum_{i=1} \lambda_i l_i(f_W(x), y_i)\right]$$

    风格迁移任务中，输入图像 $x$ 就是内容图像 $x=y_c$。

损失包含两部分：1. 特征重建损失（图 1 的蓝色箭头和黑色箭头）；2. 风格重建损失（红色箭头和黑色箭头）。通过这两个损失，使得 $\hat y=f _ W(x)$ 在内容特征上与原始输入图像 $x$ 相似，在风格上与风格图像 $y _ s$ 相似。

eval 阶段，输入内容图像经过图像转换网络的输出 $\hat y$ 就是风格迁移后的图像。

## 1.1 图像转换网络

![](/images/style_transfer/Fast_Neural_Style_2.png)

图 2. 风格迁移中的图像转换网络结构 ($f_W(x)$)

1. 输入图像大小为 $3 \times 256 \times 256$，size 默认形式为 `channels, height, width`。
2. 包含 `5` 个 residual blocks。
3. 对于 non-residual 中的 conv，除最后一个 output 层的 conv 之外，conv 的具体形式为 `Conv-InstNorm-ReLU`，最后一个 conv 的展开为 `Conv-InstNorm-ScaledTanh`，即，先激活到 `[-1,1]`之间，然后 scale 到 `[0,255]` 之间。

    （然而很多代码中直接使用 `Conv` 作为 output 层的 conv，这是因为后面的计算损失的网络 $\phi$ 会对输入进行归一化 $y:= (y-\mu) / \sigma$，具体请参见下文 `## 1.2` 一节。）
4. 第一个和最后一个 `conv` 使用 `9x9` 卷积核，其他 `conv` 使用 `3x3` 卷积核。
5. 不使用 pooling 层，作为替代，使用 strided **conv** 和 fractionally strided **conv** 作为下采样和上采样。实际实现中，不使用 ConvTransposed2d，这样会出现 Checkerboard Artifacts 现象，而是先 upsample（双线性插值），然后在使用 Conv。

    如图 2， 两个 `stride 2x` conv 用于下采样，中间经过几个 residual blocks，然后使用两个 `stride 0.5x` conv 进行上采样。

## 1.2 Perceptual Loss Functions

本节介绍网络 $\phi$ 以及两种损失计算方式。

1. $\phi$ 网络的初始输入图像需要经过处理：
    $$y:= (y-mean)/std$$
    其中 $y$ 为输出图像或 target 图像，mean 为三通道的像素均值（对彩色图像而言），std 为三通道的像素方差。有的 implementation 中，没有使用 std，即 $y:=y-mean$ 。

Perceptual Loss 用于测量两个图像的高层感知和语义上的差异（与此相对的是图像的 per-pixel 差异）。

本文采用 VGG-16 作为 $\phi$ 网络，在 ImageNet 上进行预训练，然后固定 $\phi$。

**特征重建损失**

图像转换网络的输出 $\hat y$ 与内容图像 $y_c$ 之间的损失。

如图 1，记 $\phi_j(x)$ 为网络 $\phi$ 的第 `j` 个 layer 的输出值（激活值），记其 size 为 $C_j \times H_j \times W_j$，那么特征重建损失为两个图像在这一层的输出特征的欧氏距离，

$$l_{feat}^{\phi,j} (\hat y, y)=\frac 1 {C_j H_j W_j} ||\phi_j(\hat y)-\phi_j(y)||_2^2$$

作者实验表明：
1. 最小化 $\phi$ 低层特征重建损失，利于生成图像 $\hat y$ ，使得与原内容图像 $y$ 视觉差异较小。
2. 最小化 $\phi$ 高层特征重建损失，图像内容以及整个空间结构得以保持，但是颜色纹理以及精确形状均有所改变。这正是风格迁移所需要的特性。

**风格重建损失**

这个损失着重对生成图像 $\hat y$ 与风格图像在风格：颜色，纹理以及一些通用模式上的差异进行惩罚。

显然风格考虑不是单个 pixel 的差异，所以不能像内容特征重建损失那样，计算 pixel 之间的差异。对于风格，应该从全局（整个 spatial size）进行考虑，为此，引入 Gram Matrix。

对于 $\phi$ 的第 `j` 层输出 $\phi_j(x)$，其 size 依然记为 $C_j \times H_j \times W_j$，定义 Gram matrix $G_j^{\phi}(x) \in \mathbb R^{C_j \times C_j}$，其中 $(c, c')$ 处的元素值为，

$$G _ j ^ {\phi}(x) _ {c, c'}=\frac 1 {C _ j H _ j W _ j}\sum _ {h=1} ^ {H_j}\sum _ {w=1} ^ {W_j} \phi _ j(x) _ {h,w,c} \phi _ j(x) _ {h,w,c'}$$

上式表示将输出特征 $\phi _ j(x)$ 的第 `c` 和 第 `c'` 两个 channel 的 feat map 均看作一维向量，然后求向量内积，最后除以 $\phi _ j(x)$ 的元素数量（element number），这是为了消除特征面 size 不同带来的影响。

如果将 $\phi_j(x)$ 看作是 $H_j \times W_j$ 大小的 grid 上所有点的 $C_j$ 维特征，那么 $G_j^{\phi}(x)$ 正比于 $C_j$ 维特征的非中心协方差（类比，两个 $C_j$ 维随机向量 $X, Y$，这里 grid 上所有点的取值构成随机向量的分布，非中心协方差为 $\mathbb E[XY]$） 。

计算 Gram matrix：将特征 $\phi_j(x)$ reshape 为 $\psi \in \mathbb R^{C_j \times H_jW_j}$，于是

$$G_j^{\phi}(x) = \psi \psi^{\top} / C_jH_jW_j$$

风格重建损失为两个图像特征的 Gram matrix 之差的 F 范数，
$$l _ {style} ^ {\phi,j}(\hat y, y)= ||G _ j ^ {\phi}(\hat y) - G _ j ^ {\phi}(y)||_F ^ 2$$


**即使 $\hat y$ 和 $y$ 的 spatial size 不同，风格重建损失也是可以计算的。因为两者的特征 $\phi_j(\hat y)$ 和 $\phi_j(y)$ 具有相同的 channel $C_j$，故两者的 Gram matrices 具有相同的 size。**

最小化风格重建损失有助于保留风格特征，但是不保留空间形状。

从一组 layers $J$ 而非单个 layer $j$ 中重建风格，定义 $l_{style}^{\phi, J}(\hat y, y)$ 为 $\lbrace l _ {style} ^ {\phi, j}(\hat y, y)\rbrace _ {j=1} ^ J$ 的风格重建损失之和。

## 1.3 Simple Loss

除了上述的特征重建损失和风格重建损失，作者还定义两个简单的损失，这两个简单损失使用 low-level pixel 信息。

**Pixel Loss**

$$l_{pixel}(\hat y, y)= ||\hat y - y||_2^2 / CHW$$

Pixel loss 为经过网络转换之后的图像与目标图像的归一化欧式距离（将图像展开为一维向量，计算两个向量的归一化欧式距离），其中 $C,H,W$ 表示 `channel, height, width`。这个损失在我们期望网络转换之后的图像与某个 target 图像接近的情况下使用。这里风格迁移任务没有使用它。

**Total Variation Regularization**

这个损失是为了图像空间更加平滑，

total variation（TV）function 离散形式如下：
$$l _ {TV}(\mathbf x)=\frac 1 {CHW}\sum _ {i,j} [(x _ {i,j+1}-x _ {i,j}) ^ 2+(x _ {i+1,j}-x _ {i,j}) ^ 2] ^ {\beta/2}$$

当 $\beta=1$ 时为 TV regularizer。


# 2. 风格迁移
风格迁移是使得图像转换网络的输出满足

$$\hat y = \arg \min_y \lambda_c l_{feat}^{\phi, j}(y, y_c)+\lambda_s l_{style}^{\phi,J}(y, y_s) + \lambda_{TV}l_{TV}(y)$$

图像转换网络训练细节：

1. 使用 COCO 数据集，将 80k 训练图片 resize 到 $256\times 256$ 大小
2. batch size 取 `4`
3. 优化方法为 `Adam`，学习率为 $1\times 10^{-3}$
4. $\phi$ 使用 `VGG-16`
5. 特征重建使用 `relu2_2` 的输出
6. 风格重建使用 `relu1_2`，`relu2_2`，`relu3_3` 以及 `relu4_3` 的输出。


# 3. 超分辨率

基于感知损失，本文还研究了超分辨率，与风格迁移一样，超分辨率也是 _ill-posed_ ，也就是说没有唯一的输出图像，但是由于感知损失是利用了 high-level 的特征，所以在超分辨率任务中，利用感知损失也能很好地学习。

网络结构如图 1 所示，

1. 输入是低分辨率图像，输出是高分辨率图像，spatial size 变大，所以图像转换网络需要调整

    整个网络的上采样率为 $f$，也就是说，将输入图像的 H，W 均提高 f 倍。

    网络构成：几个 residual blocks，后跟 $\log _ 2 f$ 个 `stride 0.5x` 的 conv 。
2. content 目标 $y _ c$ 是 gt 高分辨率图像
3. 不使用风格图像 $y _ s$，也不使用风格重建损失

作者在实验中，训练了 `x4` 和 `x8` 两种网络。使用 VGG-16 中的 `relu2_2` 的输出特征计算特征重建损失。数据集使用 MS-COCO 中的图像，从图像中抠出 `288x288` 的图像 patch，作为 target，然后输入图像则是将 target 图像进行下采样得到，过程是：先使用高斯模糊（$\sigma=1.0$），然后 bicubic 插值得到下采样后的图像。

**Baseline** ：使用 SRCNN
