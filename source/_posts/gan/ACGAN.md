---
title: 使用辅助分类器 GAN 带条件地合成图像
date: 2022-05-11 10:44:12
tags: GAN
mathjax: true
---
论文：[Conditional Image Synthesis with Auxiliary Classifier GANs](https://arxiv.org/abs/1610.09585)

# 1. 引言

使用带类别条件的 GAN 进行图像合成。本文首次实现了：

1. ImageNet 1000 类别的图像合成模型，空间分辨率为 `128x128`。
2. 测量了图像合成模型对输出分辨率的利用程度
3. 使用一个快速的容易计算的指标对 perceptual variability 和 GAN 中的 collapsing 行为进行测量
4. 强调大数量的类别是 GAN 图像合成的难点，本文提出了解决方法。

# 2. 背景

GAN 包含生成器 G 和判别器 D。G 的输入为一个随机向量，生成假图像，D 的输入为一图像，输出此图像的得分，用于判断是真实图像，还是 G 生成的假图像（二分类）。

D 的训练目标为

$$\max_D E[\log P(S=real|X_{real})]+E[\log P(S=fake|X_{fake})]$$

即，尽可能分类正确。

G 的训练目标为

$$\min_G E[\log P(S=fake|X_{fake})]$$

即，尽可能让 D 对假图像分类错误。

可以通过 __side information__ 增强 GAN。例如：

1. CGAN 中，对 G 和 D 提供图像类别。
2. 图像标题和 bounding box 作为 side information

除了将 side information 提供给 D，还可以设计 D 的结构用于重建 side information，例如 D 增加一个 decoder 分支，输出类别信息，或者其他 latent 变量。生成模型中，从这些 latent 变量生成图像数据 x。另外，辅助的 decoder 可以利用预训练的 D，从而进一步提高合成的图像质量。

鉴于此，作者介绍了一个模型，使用了 类别条件 的 GAN，同时有一个辅助 decoder，用于重建类别。


# 3. AC-GANs
auxiliary classifier GAN（AC-GAN）

每个生成的图像样本有一个对应的类别，$c \sim p_c$，于是

$$X_{fake} = G(c,z)$$

D 输出预测图像和其类别，$P(S|X), P(C|X)=D(X)$，目标函数为

$$L_S = E[\log P(S=real|X_{real})]+E[\log P(S=fake|X_{fake})]
\\L_C = E[\log P(C=c|X_{real})] + E[\log P(C=c|X_{fake})]$$

训练 D 使得 

$$\max_D L_S+L_C$$

训练 G 使得 

$$\max_G L_C-L_S$$

AC-GAN 相对于标准 GAN 的改动不算很大，但是可以得到很好的结果，且训练更加平稳。

除了 AC-GAN 模型，本文还提出了：

1. 测量模型对其输出分辨率的利用程度
2. 测量模型生成样本的 perceptual variability
3. 实验分析图像生成模型，此模型可生成 ImageNet 1000 类别的 `128x128` 的样本。

较早的实验证明了，训练中增加类别数量而保持模型固定，会降低模型生成的图像质量。

AC-GAN 模型的结构训练将大的数据集按类别划分为数据子集，然后在每个数据子集上训练 G 和 D。作者将 ImageNet 1000 个类别划分为 100 个组，每个组 10 个分类，即 10 个分类一个数据子集，100 个 AC-GANs 分别在 100 个数据子集上训练。

