---
title: FPN 回顾
date: 2022-01-12 14:28:06
tags: object detection
mathjax: true
---
论文：[Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)

本文是对 FPN 的回顾，适用于复习 FPN，或者快速查看 FPN 实现的关键细节。
<!--more-->

# 1. 网络

目标检测是使用 backbone 得到 feature maps，然后在 feature 上进行检测。根据 feature scale 可分如下四种情况，
![](/images/obj_det/fpn1.png)

图 1

**FPN 使用上图 (d) 的结构，这是因为，网络高层的输出 feature 的语义较强，但是感受野较大，scale 较小，导致小目标很难检测出来，而低层输出 feature 的 scale 够大，但是语义较弱，故自然而然想到将具有较强语义的高层 feature 融合进低层 feature 中。**

FPN 在 Faster R-CNN 的基础上进行修改，backbone（或称 baseline）使用 ResNet。

## 1.1 Bottom-up
Bottom-up 就是 backbone 的前向传播，使用 ResNet，使用 `stage2~5` 的输出特征，每个 stage 使用最后一个 layer 的激活输出（即 ReLU 后的数据），下采样率分别为 $[4, 8, 16, 32]$，每个 stage 的最后一个 block 记为 $[C_2, C_3, C_4, C_5]$。（不用 `conv1` 的输出特征是为了节约内存）

## 1.2 Top-down 和 横向连接

自顶向下，上层特征需要上采样（`2x`）然后下层特征经过 `1x1 Conv` 进行调整（使得 channels 相匹配），然后两者可进行融合。如图 2，

![](/images/obj_det/fpn2.png)
图 2. 

说明：
1. 上采样采用最近邻法（或者插值法）
2. 融合采用 element-wise 相加
3. 所有层横向连接中的 `1x1 Conv` 均降低 bottom-up 的特征 channels 为 `d`。例如 ResNet-n($n \ge 50$)，`C2,C3,C4,C5` 的 channels 最小值为 `256`，故可取 `d=256`，即，所有横向连接以及 top-down 结构中均为 `d=256`。
4. 每层融合后的特征分别经过一个 `3x3 Conv` 得到最终的 feature maps，这个 `3x3 Conv` 降低（调整）上采样的偏差影响。
5. 横线连接和 top-down 结构中的卷积**没有非线性激活**，论文中指出，去掉非线性激活对结果影响较小。
6. 横向连接的 `1x1 Conv` 不是所有 level 共享？

注：与前面保持一致地，`Conv` 只表示卷积这个单一操作，而 `conv` 表示 `Conv+BN+ReLU` 这个组合（早期的工作中，还未诞生 `BN`，彼时  `conv` 表示 `Conv+ReLU`）。

## 1.3 feature maps
最终得到 feature maps，记为 $\{P_2, P_3,P_4,P_5\}$。此外，为了覆盖更大的目标 size，额外增加一个 $P_6$，是对 $P_5$ 进行 `2x` 的下采样得到。

# 2. RPN 应用 FPN

由于是在 Faster R-CNN 进行修改，故先看 RPN 部分。首先 feature maps 由单一 sacle 变成 FPN 的多 scales，每个 scale 的 feature maps 上，均采用 `3x3 conv` 以及两个并列的 `1x1 conv` （分别用于生成 object/non-object 的二分类得分和基于 anchor 的坐标偏差回归）。

1. 每个 scale 的 feature maps 上使用 单个 scale 的 anchor，对于 ${P_2,P_3,P_4,P_5,P_6}$，anchor area 分别为 $\{32^2, 64^2, 128^2, 256^2, 512^2\}$，aspect ratios 取 $\{1/2, 1, 2\}$ 三种，故每个 spatial point 有 `k=3` 个 anchors。
2. 正例 anchors：与 gt boxes 的最大 IOU 大于 `0.7`
3. 负例 anchors：与 gt boxes 的最大 IOU 小于 `0.3`
4. network head（即，两个 `1x1 conv`，输出 channels 不同，分别为 `2k` 和 `4k`），在**所有 level 共享参数**

# 3. Fast RCNN 应用 FPN

通过 RPN 得到 ROIs 之后，由于 backbone 生成多 scale 的 feature maps，所以需要确定某个 ROI 应该对应哪个 scale。记 ROI 的 size 为 `(h, w)`，那么 feature pyramid $P_k$ 根据下式确定：
$$k=\lfloor k_0 + \log_2 (\sqrt {wh} / 224) \rfloor$$

其中 $k_0=4$ 。这是因为 ResNet based Faster R-CNN 中，输入 size 为 $224 \times 224$，使用 $C_4$ 作为单 scale 的 feature maps，作为类比，这里使用 $k_0=4$ 比较合适。

1. 在每个 level 上使用 predictor heads（一个分类器和一个 bbox 坐标偏差回归器），与 Faster R-CNN 一样，feature maps 首先经过 ROIPooling，得到 `7x7` 的特征 size，经过连续两个全连接层 `fc+relu+drop` （这两个 fc layer 输出 channels 均为 `124`），然后分别使用分类器（crossentropyloss）和回归器（MSE）。
2. 每个 level 上的 predictor heads 共享参数。
3. Fast R-CNN 不使用 $P_6$。