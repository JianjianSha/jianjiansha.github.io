---
title: lightweight
date: 2021-03-05 16:18:13
tags:
---

# ThunderNet
(two-stage detector)

移动设备上计算能力有限，而现在的很多 CV 实现方式都需要较强的计算力，这导致这些任务难以在移动设备上 real-time 的实现。本文研究了 two-stage 目标检测 real-time 的有效性，并提出了一个轻量级的 two-stage 检测器，名为 ThunderNet。

简介：

1. 研究了先前轻量级 backbone 的缺点，并提出了一个新的为目标检测而设计的轻量级 backbone  `SNet`
2. 沿用 Light-Head R-CNN 的 detection head 结构
3. 进一步压缩 RPN 和 R-CNN 两个 subnet，以加快计算速度。 
4. small backbone 会带来一定的性能降级，所以设计两个高效率模块 `Context Enhancement Module` 和 `Spatial Attention Module`。CEM 结合多个 scale 的 feature （backbone 中浅层到深层的 feature）以利用 local 和 global 信息。SAM 利用 RPN 学习到的信息来微调 RoI warping 中的特征。
5. input resolution：`320x320`，小 size 可以加快网络的 inference 速度。

## backbone

__Receptive Field:__

大感受野可以利用更多的上下文信息，同时能有效地 encode 像素间的 long-range 关系，这对目标尤其大目标的定位非常关键。

__early & late stage feature:__

高层特征具有更强的语义性，更具有辨别性，低层特征具有更丰富的空间细节信息，所以高低层特征都需要用到。


__SNet__

SNet 作为专为 real-time 目标检测而设计的轻量级的 backbone。SNet 以 ShuffleNetV2 为基础进行改造，将所有的 `3x3` depthwise conv 改为 `5x5` depthwise conv，以获取更大的感受野，同时保持差不多的计算速度。还有其他的一些改动这里不一一指出。

depthwise conv: 每个通道独立进行二维卷积，需要 $c_{in}$ 个 $k \times k$ 卷积，得到的 feature 的 shape 与输入 feature shape 相同，然后再执行 $1 \times 1 \times c_{in}\times c_{out}$ 的跨通道卷积，输出 feature 的channel 为 $c_{out}$。 

## Detection Part
压缩 RPN 和 Detection Head。Light-Head R-CNN 的 detection head 虽然是轻量级，但是配合小的 backbone 时，依然太过 heavy，导致 backbone 与 dection head 之间产生 imbalance。

压缩 RPN：将原来的 256-d `3x3` conv 替换为 `5x5` 的 depthwise 和 256-d `1x1` conv。anchor 的配置为 scale：`{32, 64, 128, 256, 512}`，aspect ratio：`{1:2, 3:4, 1:1, 4:3, 2:1}`。

detection head： Light-head R-CNN 中的 thin feature map $\alpha \times p \times p$，其中 $p=7, \ \alpha=10$，由于 thundernet 中 backbone 和 input image size 均较小，所以继续降低 $\alpha=5$。采用 PSRoI，由于 PSRoI 输出的 feature 仅 245-d，那么 R-CNN subnet 中的 fc 全连接层为 1024-d。

## CEM
context enhancement module。

Light-Head R-CNN 使用 global convolutional network（GCN） 生成 thin feature map，GCN 具有 large kernel，使得 Receptive Field 增大，从而可以 encode 更多的上下文信息，但是 GCN 会给 SNet 带来很多计算量，thundernet 不使用 GCN，而是使用 CEM 解决这个问题。

借鉴 FPN 的思想（FPN 本身结构比较复杂），聚合multi-scale 的 局部信息和全局信息，得到具有较强判别性的 feature。CEM merge 来自以下 layer 的 feature：$C_4, \ C_5, \ C_{glb}$，其中 $C_{glb}$ 表示 global feature，通过对 $C_5$ 执行 global average pooling 得到。对以上三个 scale 的 feature 使用 `1x1-245` conv，输出 channel 均为 245，且 $C_5$ 的输出特征还需要 `2x` upsample，使得与 $C_4$ 的输出 feature 具有相同的 size，而 $C_{glb}$ 的输出本质是是一个标量，所以经 broadcast 具有与 $C_4$ 输出 feature 具有相同的 size，然后这三组相同 spatial size 的 feature 再合并。

## SAM
spatial attention module。

在 RoI warping 的输入 feature （上面说讨论的 thin feature maps）上，我们希望 负例 region 内的 feature 值足够小，正例 region 内的 feature 足够大，但是 thundernet 比正常的检测网络小，所以会难以学习到正确的 feature 分布，本文使用 SAM 解决这个问题。

SAM 利用 RPN 得到的信息来微调 RoI warping 的输入 feature 分布。RPN 被训练用来区分正负例，那么 RPN 的输出 feature 可以利用起来，于是，SAM 的两个输入：1. RPN 的输出 feature；2. CEM 输出的 thin feature maps。SAM 的输出 feature 为，
$$\mathcal F^{SAM}=\mathcal F^{CEM} \cdot sigmoid[\theta(\mathcal F^{RPN})]$$
其中 $\theta$ 用于维度转换，使得 $\mathcal F^{RPN}$ 和 $\mathcal F^{CEM}$ 具有相同的维度，文中使用 `1x1` conv 来执行这个维度转换。

SAM 的输出将作为原先 RoI warping 的输入。

thunernet 整个网络结构如图 1，

![](/images/obj_det/lightweight_fig1.png)<center>图 1</center>

# Light-Head R-CNN
(two-stage detector)

设计了一个轻量级的 detection head，有如下两种设计：
1. L：配合 large backbone，文中采用 ResNet101
2. S：配合 small backbone，文中采用 Xception

backbone 最后一个 conv block 记为 $C_5$，$C_5$ 之后使用一个 separable conv（依次为 `kx1` 和 `1xk` 两个 conv），最终输出 channel 为 $10 \times p \times p$，而 R-FCN 中对应的 channel 为 $(C+1) \times p \times p$（$p \times p$ 表示 bin，因为是 positive-sensitive），所以相对 R-FCN，这里的设计更加 small。


__R-CNN subnet__

PSRoI pooling 之后，使用一个 2048-d 的全连接层，然后分两支，一支用于分类预测，一支用于box 回归预测，其中分类分支使用一个 C-d 的全连接，回归分支使用 4-d 的全连接层。

__RPN__

RPN 作用于 $C_4$ 之上，根据 anchor box 预测出一组 proposals，anchor 的 scale 为 `{32,64,128,256,512}`，aspect ratio 为 `{1:2,1:1,2:1}`。

整个网络的结构图如下，
![](/images/obj_det/lightweight_fig2.png)<center>图 2 </center>