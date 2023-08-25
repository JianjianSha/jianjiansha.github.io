---
title: 超分辨率指标
date: 2023-08-01 18:19:41
tags: super resolution
mathjax: true
---

## 1.1 PSNR

peak signal-to-noise ratio，测量峰值信号与噪声的能量比，

$$PSNR = 10 \log _ {10}\left (\frac {M ^ 2 } {MSE}\right)$$

其中峰值信号为经过超分之后的图像 $I_y$ 的最大像素值，通常使用 8-bit 表示颜色空间，即像素值上限为 255，噪声为 $I _ y$ 和 reference 图像 $I _ r$ 之间的像素值之差的平方再求均值，

$$MSE = \frac 1 t \sum _ {i=1} ^ t [I _ r(i) - I _ y(i)] ^ 2$$

其中 $t$ 为单个图像中总像素数量。

## 1.2 SSIM

structural similarity index metric，通过比较亮度、结构化细节来测量两个图像之间的结构化相似度。

图像 $I _ r$，总像素数量记为 $M$，对比度 $C _ I$ 和亮度 $L _ I$ 分别为标准差和均值，

$$\begin{aligned} L _ r &= \frac 1 M \sum _ {i=1} ^ M I _ r(i) 
\\\\ C _ r &= \sqrt {\frac 1 {M-1}\sum _ {i=1} ^ M [ I _ r(i) - L _ r] ^ 2}
\end{aligned}$$

比较 reference 图像 $I _ r$ 和超分后图像 $I _ y$ 之间的亮度和对比度，

$$\begin{aligned} D _ l(I _ r, I _ y) = \frac {2 L _ r L _ y + \mu _ 1} {L _ r ^ 2 + L _ y ^ 2 + \mu _ 1}
\\\\ D _ c (I _ r, I _ y) = \frac {2 C _ r C _ y + \mu _ 2} {C _ r ^ 2 + C _ y ^ 2 + \mu _ 2}
\end{aligned}$$

根据 $\hat I _ r=(I _ r - L _ r) / C _ r$ 对 reference 图像像素值归一化，这个值表示了图像结构，那么两个归一化之后的图像（看作向量）内积就表示结构相似度，即 $\hat I _ r \cdot \hat I _ y / (M-1)$，这里除以 $M-1$ 是为了消除图像 size 对结果的影响。

将两个图像看作向量，将它们的协方差记为 

$$\sigma _ {r, y} = \frac 1 {M -1} \sum _ {i=1} ^ M (I _ r - L _ r) (I _ y - L _ y)$$

于是结构化细节相似度如下

$$D _ s (I _ r, I _ y) = \frac {\sigma _ {r, y} + \mu _ 3} {C _ r C _ y + \mu _ 3}$$

以上计算式中，$\mu _ 1, \mu _ 2, \mu _ 3$ 是为了防止除零。

SSIM 计算如下，

$$SSIM(I _ r, I _ y)=D _ l ^ {\alpha} + D _ c ^ {\beta} + D _ s ^ {\gamma}$$

其中 $\alpha, \beta, \gamma$ 是控制参数。

## 1.3 感知损失

感知损失（Perceptual Loss）是一种基于深度学习的图像风格迁移方法中常用的损失函数。与传统的均方误差损失函数（Mean Square Error，MSE）相比，感知损失更注重图像的感知质量，更符合人眼对图像质量的感受。

感知损失是通过预训练的神经网络来计算两张图片之间的差异。通常使用预训练的卷积神经网络（Convolutional Neural Network，CNN），这些网络已经在大规模的数据集上进行了训练，可以提取图像的高级特征。例如，VGG-19网络中的卷积层可以提取图像的纹理和结构信息，而网络的全连接层可以提取图像的语义信息。

感知损失的计算方式通常是将输入图像和目标图像分别通过预训练的神经网络，得到它们在网络中的特征表示。然后将这些特征表示作为损失函数的输入，计算它们之间的欧氏距离或曼哈顿距离。感知损失的目标是最小化输入图像和目标图像在特征空间的距离。
