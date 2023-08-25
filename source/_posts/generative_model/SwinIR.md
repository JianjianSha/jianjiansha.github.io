---
title: 'SwinIR: Image Restoration Using Swin Transformer'
date: 2023-08-04 22:27:49
tags:
    - super resolution
    - denoising
---

论文：[SwinIR: Image Restoration Using Swin Transformer](https://arXiv.org/abs/2108.10257)

图像重建：从低质量图像（缩小的，带噪的或者压缩过的）恢复出高质量图像。

# 1. 简介

本文提出 SwinIR 用于图像重建。SwinIR 包含三个部分：

1. 浅层特征提取：使用 conv layer 提取浅层特征

2. 深层特征提取：由几个 residual Swin Transformer blocks（RSTB）组成，每个 block 包含若干 Swin Transformer layer，以获取局部注意力和区域间注意力。

3. 高质量图像重建：浅层特征和深层特征在图像重建模块中融合，以得到高质量图像。

# 2. 方法

## 2.1 网络架构

如图 1 所示，

![](/images/generative_model/swinir_1.png)

<center>图 1.</center>

**# 浅层和深层特征提取**

给定一个低质量（LQ）输入 $I _ {LQ} \in \mathbb R ^ {H \times W \times C _ {in}}$，使用 `3x3` 卷积 $H _ {SF}(\cdot)$ 提取得到浅层特征 $F _ 0 \in \mathbb R ^ {H \times W \times C}$，

$$F _ 0 = H _ {SF}(I _ {LQ}) \tag{1}$$

然后提取得到深层特征，

$$F _ {DF} = H _ {DF} (F _ 0) \tag{2}$$

深层特征提取模块 $H _ {DF}(\cdot)$ 包含 $K$ 个 RSTB，以及一个 `3x3` conv layer（这个 conv layer 用于加强特征）。$K$ 个 RSTB 以及 conv 输出如下，

$$\begin{aligned}F _ i &= H _ {RSTB _ i}(F _ {i-1}), \quad i = 1,2,\ldots, K
\\\\ F _ {DF} &= H _ {CONV}(F _ K)\end{aligned} \tag{3}$$

**# 图像重建**

以图像超分辨率为例，通过聚合浅层和深层特征重建高质量图像，过程如下，

$$I _ {RHQ} = H _ {REC}(F _ 0 + F _ {DF}) \tag{4}$$

图像重建模型，使用 sub-pixel 卷积层对特征进行上采样。

对于不需要上采样的任务如降噪或者降低 JPEG 压缩失真（JPEG 格式存储图像时会有压缩，导致图像存在失真），仅使用一个 conv layer 来重建图像。另外，作者使用了残差学习，即使用 SwinIR 学习 HQ 图像与 LQ 图像之间的残差，那么高质量图像重建过程为

$$I _ {RHQ} = H _ {SwinIR}(I _ {LQ}) + I _ {LQ} \tag{5}$$

**$F _ {DF}, \ F _ 0, \ I _ {LQ}$ 的 spatial size 均相等** 。

**# 损失函数**

图像超分辨率损失函数为

$$L = ||I _ {RHQ}-I _ {HQ}|| _ 1 \tag{6}$$

(6) 式适用于经典轻量化图像的超分辨率。对于现实中的图像（照片）超分辨率，结合 (6) 式这个 pixel loss 和 GAN loss（判别网络的损失） 以及 perceptual loss（两个图像经过预训练网络后输出特征的欧氏距离）。

图像降噪和减少 JPEG 压缩失真任务，使用 Charbonnier 损失，

$$L = \sqrt {||I _ {RHQ} - I _ {HQ}|| ^ 2 + \epsilon ^ 2} \tag{7}$$

其中常数 $\epsilon$ 通常设置为 $10 ^ {-3}$ 。