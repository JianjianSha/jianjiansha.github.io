---
title: DetNet
date: 2019-07-17 10:05:50
p: obj_det/DetNet
tags: object detection
mathjax: true
---
论文 [DetNet: A Backbone network for Object Detection](https://arxiv.org/abs/1804.06215)
<!-- more -->
本文创作动机是当前大多数的目标检测器都是在 ImageNet 上预训练后 finetune 到目标检测集，目标检测器的 backbone 原本是为了图像分类任务而设计的，这样的 backbone 显然不是最佳的，较大的下采样率带来较大的感受野 RF，这对图像分类是有益的，对目标检测尤其是小目标而言则是不利的，所以像 FPN 和 RetinaNet 就使用了额外的网络结构（extra stage）来处理目标的多尺度问题，但是这总归不是一个优雅的解决办法，所以本文提出了 DetNet，这是一个专为目标检测而设计的新型 backbone。

DetNet 保持了 FPN 中的额外网络结构（extra stage），毕竟是目标的多尺度问题的一个较为不错的解决方案。与 FPN 等基于 ImageNet 预训练的目标检测器不同的是，DetNet 的深层依然有较高的空间分辨率，不过考虑到高分辨率与计算资源的矛盾，我们采用了一种低复杂度的 dilated bottleneck 结构。

# DetNet
如图 1(A) 是 FPN 的部分网络结构，图像分类任务和目标分类任务本身就存在很大的不同，并且基于此结构的模型训练还存在以下问题：
![](/images/DetNet_fig1.png)<center>A. 具有传统的 backbone 的 FPN 结构；B. 图像分类中传统的 backbone；C. DetNet 的 backbone，比 FPN 的分辨率高</center>

1. 网络 stage 的数量不同。图像分类的网络包含 5 个 stages，每个 stage 下采样率为 2，故输出分辨率为 32 倍的下采样，而 FPN 拥有更多的 stages，比如增加 P6 以处理更大的目标，在 RetinaNet 中也同样增加了 P6 和 P7。
2. 大目标的可视性较差。具有 32 的步幅的 feature map 包含较强的语义信息，然而这对目标定位是不利的，FPN 中大目标是由较深 layer 进行预测，难以回归到准确的目标边界。
3. 小目标的不可见性。大的步幅显然会导致小目标的丢失，所以 FPN 在较浅 layer 上预测小目标，然而浅 layer 只有很弱的语义信息，可能不足以预测目标分类，故为了加强浅 layer 的目标分类能力，将深 layer 的特征上采样后合并进浅层特征，如图 1 A 所示，只不过，如果小目标在较深 layer 中已经丢失，那么深层特征上就没有小目标的 context 信息，这样的深层特征合并进浅层特征并不会增强对小目标的分类能力。

DetNet 经过如下设计可解决以上问题：
1. 直接为目标检测量身定制 stage 的数量
2. 即使 stage 的数量很多，如 6~7 个 stage，对于 deep layer，在保持较大感受野（有利于分类）的同时有较大的分辨率（有利于目标定位）。

## DetNet 设计
使用 ResNet-50 作为 baseline。在 ResNet-50 的基础之上构建 DetNet-59（类似地也可以在 ResNet-101 基础上构建 DetNet，在本文中这不是重点）。DetNet 的 stage 1,2,3,4 与 ResNet-50 的 stage 1,2,3,4 完全相同。这里给出 ResNet-50 前四个 stage 的结构描述，

|   ResNet        | output size | 50-layer             |
|:--------:       | :------:    |   :-------:          |
| conv1           | 112x112     | 7x7,64, stride 2     |
|   maxpool       | 56x56       | 3x3, stride 2        |
| conv2_x         | 56x56       | $\begin{bmatrix} 1 \times 1 & 64 \\\\ 3 \times 3 & 64 \\\\ 1 \times 1 & 256\end{bmatrix} \times 3$|
|conv3_x          | 28x28       | $\begin{bmatrix} 1 \times 1 & 128 \\\\ 3 \times 3 & 128 \\\\ 1 \times 1 & 512\end{bmatrix} \times 4$|
|conv4_x          | 14x14       | $\begin{bmatrix} 1 \times 1 & 256 \\\\ 3 \times 3 & 256 \\\\ 1 \times 1 & 1024\end{bmatrix} \times 6$|

从第五个 stage 开始介绍 DetNet，如图 2 D 所示，DetNet-59 的设计细节如下：
![](/images/DetNet_fig2.png)<center>fig 2. DetNet 的结构细节</center>

1. 从上图中可见，我们在 backbone 中引入了 extra stage，即 P6，与 FPN 中一样，也是用于目标检测，只不过，从 stage 4 开始，我们就固定了步幅 16，即每个 stage 的输出空间大小。
2. 从 stage 4 开始的空间大小就固定不变，本文引入一种 dilated bottleneck 和 1x1 卷积并列的结构，用于之后每个 stage 的最开始，如图 2 B。
3. bottleneck 中的 dilated conv 可以增大感受野。由于 dilated conv 较为耗时，所以 stage 5 和 6 的 channel 与 stage 4 保持相同（维持在256），这一点与传统 backbone 设计不一样，传统 backbone 的后一个 stage 的 channel 是前一个 stage 的两倍（如 ResNet-50 中的 64->128->256->512）。

DetNet 作为 backbone 可以很方便地移植到（具有/不具有 feature pyramid 的）目标检测器中。不失代表性地，我们采用 FPN 作为主检测器，除了 backbone 不同，其他结构与原先 FPN 中保持相同。由于 stage 4 之后的 stage 输出大小不变，所以将 stage 4,5,6 的输出相加，如图 2 E。

# 实验
实验和结果分析，略
