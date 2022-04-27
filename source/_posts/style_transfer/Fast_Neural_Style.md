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

## 1.1 图像转换网络

![](/images/style_transfer/Fast_Neural_Style_2.png)

图 2. 风格迁移中的图像转换网络结构

1. 输入图像大小为 $3 \times 256 \times 256$，size 默认形式为 `channels, height, width`。
2. 包含 `5` 个 residual blocks。
3. 对于 non-residual 中的 conv，除最后一个 output 层的 conv 之外，conv 的具体形式为 `Conv-InstNorm-ReLU`，最后一个 conv 的展开为 `Conv-InstNorm-ScaledTanh`，即，先激活到 `[-1,1]`之间，然后 scale 到 `[0,255]` 之间。

    （然而很多代码中直接使用 `Conv` 作为 output 层的 conv，可能是认为，前面每个 layer 均有 normalization 操作，所以特征均在一个区间范围内，最后一个 `Conv` 通过训练其参数，能使得最终输出位于（或几乎位于） [0,255] 内。）
4. 第一个和最后一个 `conv` 使用 `9x9` 卷积核，其他 `conv` 使用 `3x3` 卷积核。
5. 使用 strided conv 和 fractionally strided conv 作为下采样和上采样。实际实现中，不使用 ConvTransposed2d，这样会出现 Checkerboard Artifacts 现象，而是先 upsample（双线性插值），然后在使用 Conv。

## 1.2 Perceptual Loss Functions

1. $\phi$ 网络的初始输入图像需要经过处理：
    $$y:= (y-mean)/std$$
    其中 $y$ 为输出图像或 target 图像，位于 [0,255]，mean 为三通道的像素均值（对彩色图像而言），std 为三通道的像素方差。有的 implementation 中，没有使用 std，即 $y:=y-mean$ 。

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

对于 $\phi$ 的第 `j` 层输出 $\phi_j(x)$，其 size 依然记为 $C_j \times H_j \times W_j$，定义 Gram matrix $G_j^{\phi}(x) \in \mathbb R^{C_j \times C_j}$，定义 Gram matrix 的元素，

$$G_j^{\phi}(x)_{c, c'}=\frac 1 {C_jH_jW_j}\sum_{h=1}^{H_j}\sum_{w=1}^{W_j} \phi_j(x)_{h,w,c} \phi_j(x)_{h,w,c'}$$

如果将 $\phi_j(x)$ 看作是 $H_j \times W_j$ 大小的 grid 上所有点的 $C_j$ 维特征，那么 $G_j^{\phi}(x)$ 正比于 $C_j$ 维特征的非中心协方差（类比，两个 $C_j$ 维随机向量 $X, Y$，这里 grid 上所有点的取值构成随机向量的分布，非中心协方差为 $\mathbb E[XY]$） 。


计算 Gram matrix：将特征 $\phi_j(x)$ reshape 为 $\psi \in \mathbb R^{C_j \times H_jW_j}$，于是

$$G_j^{\phi}(x) = \psi \psi^{\top} / C_jH_jW_j$$

风格重建损失为两个图像特征的 Gram matrix 之差的 F 范数，
$$l_{style}^{\phi,j}(\hat y, y)= ||G_j^{\phi}(\hat y) - G_j^{\phi}(y)||_F^2$$


**即使 $\hat y$ 和 $y$ 的 size 不同，风格重建损失也是可以计算的。因为两者的特征 $\phi_j(\hat y)$ 和 $\phi_j(y)$ 具有相同的 channel $C_j$，故两者的 Gram matrices 具有相同的 size。**

最小化风格重建损失有助于保留风格特征，但是不保留空间形状。

从一组 layers $J$ 而非单个 layer $j$ 中重建风格，定义 $l_{style}^{\phi, J}(\hat y, y)$ 为这组 layers 中单个 layer  $j \in J$ 的风格重建损失之和。

## 1.3 Simple Loss

**Pixel Loss**

$$l_{pixel}(\hat y, y)= ||\hat y - y||_2^2 / CHW$$

Pixel loss 为图像像素（展开为长向量）的归一化欧式距离，其中 $C,H,W$ 表示 `channel, height, width`。

**Total Variation Regularization**

这个损失是为了图像空间更加平滑，

total variation（TV）function 离散形式如下：
$$l_{TV}(\mathbf x)=\frac 1 {CHW}\sum_{i,j} [(x_{i,j+1}-x_{i,j})^2+(x_{i+1,j-x_{i,j}})^2]^{\beta/2}$$

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


