---
title: U-Net
date: 2022-08-08 13:58:40
tags: CNN
---

论文：[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

本文主要介绍 UNet 框架结构，如图 1，

![](/images/diffusion_model/unet_1.png)

<center>图 1. 每个蓝色矩形表示 feature maps。白色矩形表示 feature maps 的拷贝。箭头表示操作</center>


根据 Improved_DDPM 的[源码](https://github.com/openai/improved-diffusion)，其使用的 UNet 结构经过魔改，如下图 2~5 所示，

![](/images/diffusion_model/unet_2.png)
![](/images/diffusion_model/unet_3.png)
![](/images/diffusion_model/unet_4.png)
![](/images/diffusion_model/unet_5.png)


