---
title: 'TOOD: Task-aligned One-stage Object Detection'
date: 2023-08-12 13:10:40
tags: object detection
mathjax: true
---

论文：[TOOD: Task-aligned One-stage Object Detection](https://arXiv.org/abs/2108.07755)

# 1. 简介

一阶段的目标检测通常使用两个并列的分支：分类 head 和坐标回归 head。本文提出这种两个 head 分开的设计具有局限性：

1. 分类和定位的独立，使得分类和定位两个分支之间缺乏互动，从而预测 box 时会出现不一致的情况，无法对齐。如图 1，

    ![](/images/obj_det/tood_1.png)
    <center>图 1.</center>

    图 1 上行是 ATSS 这个目标检测方法检测结果。黄色为 gt box，分类为“饭桌”。分类得分预测 heatmap 上最大的 pixel 为第二列白色箭头所指，IOU 最大的 pixel 为第三列白色箭头所指，对应到第一列图片上，就是红色和绿色 pixel，或者称为一个小的 patch，显然这两个小 patch 不重合。
    
    - 使用红色 patch 预测（注意这里使用 anchor free 方法），那么预测结果为：分类得分为 `0.13`，预测 box 为红色 box，IoU 为 `0.40`
    - 使用绿色 patch 预测，那么预测结果为：分类得分为 `0.04`，预测 box 为绿色 box（与黄色重叠了），IoU 为 `0.93`
    - 结论：分类得分最大的 patch 和 IoU 最大的 pixel 不是同一个 patch（不对齐）。而使用本文 TOOD 方法，则可以作用分类得分最大和 IoU 最大是同一个 patch（对齐）。

2. 任务未知的样本选择。大多数 anchor-free 检测器使用几何方法确定正负样本，pixel 位于 gt box 内则为正，否则为负。anchor-based 检测器则通过最大 IoU 确定正负样本。但是，用于分类的最佳 anchor 和用于回归的最佳 anchor 通常不一致，并且随着目标的形状以及特征而改变。而前述的确定 anchor 正负例的方法并没有考虑具体任务。例如图 1 第一行，最佳 location anchor（绿色 patch）并非目标中心，与最佳分类 anchor（红色 patch）也没有对齐，这就导致 NMS 之后，一个匹配比较准确的预测 box（分类得分较低）被一个匹配不太准确的 box（分类得分高）所抑制，因为 NMS 优先选择分类得分高的 box 。

为了解决这些问题，本位提出 Task-aligned One-stage Object Detection (TOOD)，其中包含了 Task-aligned head，用于对齐分类和回归预测。

**Task-aligned head** : 加强分类和回归两个任务之间的互动联系，使得两个任务更加协同配合，从而对齐两者预测，结果更加准确。工作原理：计算任务相互作用的特征，然后使用 Task-Aligned Predictor（TAP）进行预测。

**Task alignment learning** ：为了进一步克服不对齐的问题，本文提出 Task Alignment Learning（任务对齐学习），将两个任务的最佳 anchors 拉的更加靠近，甚至重合。

sample assignment scheme：样本选择机制，即 如何选择正负样本。对每个 anchor，计算它的 task-alignment degree（任务对齐度），然后通过 task-aligned loss（任务对齐损失）逐渐统一两个任务的最佳 anchor。如此，在推理阶段，具有最高分类得分的 box 同时也拥有最准确的定位预测。

# 2. 方法

TOOD 结构为 `backbone-FPN-head` 。如图 2，T-head 和 TAL 协同工作从而对齐两个任务。首先，T-head 根据 FPN 特征预测分类和定位，然后 TAL 根据两个任务的对齐度指标计算对齐 offset，最后 T-head 根据 TAL 反馈的对齐信息自动调整分类和定位预测。

![](/images/obj_det/tood_2.png)
<center>图 2.</center>

## 2.1 Task-aligned Head

![](/images/obj_det/tood_3.png)
<center>图 3. 传统 head 与 T-head 的对比 。 Cat 表示 Concat 操作</center>

图 3 (b) 是 T-head 结构图。为了加强分类和定位两个任务之间的相互作用，本文使用一个特征提取器用于学习一组 任务之间相互作用的 特征，此特征通过多个 conv layers 得到。

记 $X ^ {fpn} \in \mathbb R ^ {H \times W \times C}$ 表示 FPN 特征，特征提取器使用 `N` 个 conv layers 计算任务相互作用的特征，

$$X _ k ^ {inter} = \begin{cases} \delta( conv _ k (X ^ {fpn})) & k=1
\\\\ \delta(conv _ k (X _ {k-1} ^ {inter})) & k > 1
\end{cases} \tag{1}$$

其中 $\delta$ 表示激活函数 relu，$\forall k \in \lbrace 1, 2, \ldots, N \rbrace$ 。

然后，特征 $X _ {1 \sim N} ^ {inter}$ 送入两个 TAP 用于对齐分类和定位。

**# TAP**

TAP 中，根据上述一组特征 $X _ {1 \sim N} ^ {inter}$ 进行分类和定位。如图 `3 (c)` 所示，每个 level 的特征 $X _ k ^ {inter}$ 使用一个权重，此权重作为 layer attention，

$$X _ k ^ {task} = w _ k \cdot X _ k ^ {inter}, \quad k \in \lbrace 1, 2, \ldots, N \rbrace \tag{2}$$

其中 $\mathbf w \in \mathbb R ^ N$ 是权重向量，通过一个 2 FC layers 分支计算得到，

$$\mathbf w = \sigma ( fc _ 2 (\delta ( fc _ 1 (\mathbf x ^ {inter}) )) ) \tag{3}$$

其中 $fc _ 1, \ fc _ 2$ 是两个全连接层，$\sigma$ 表示 sigmoid 函数，$\delta$ 表示 relu，$\mathbf x ^ {inter} \in \mathbb R ^ {NC}$ 为 $X ^ {inter} \in \mathbb R ^ {H \times W \times NC}$ 的池化输出 。最后分类和定位的预测结果为，

$$Z ^ {task} = conv _ 2 (\delta (conv _ 1 (X ^ {task}))) \tag{4}$$

将所有 $X _ k ^ {task} \in \mathbb R ^ {H \times W \times C}$ concatenate 得到 $X ^ {task} \in \mathbb R ^ {H \times W \times NC}$，最后分类预测结果为 $Z ^ {task} \in \mathbb R ^ {H \times W \times 80}$（COCO 数据集），再使用 $\sigma$ 函数激活，得到归一化分类得分。

使用 `distance-to-bbox` 将 $Z ^ {task} \in \mathbb R ^ {H \times W \times 4}$ 转变为预测 box。

TOOD 是 anchor-free，所以定位的维度 `4` 表示 pixel 到 box 四个边的距离。

**Prediction alignment**

进一步对齐预测 。使用 $M \in \mathbb R ^ {H \times W \times 1}$ 调整分类预测，

$$P ^ {align} = \sqrt {P \times M} \tag{5}$$

其中 $P$ 是上一步分类预测结果 $\sigma(Z ^ {task}) \in \mathbb R ^ {H \times W \times 80}$ 。$M$ 是根据 $X ^ {inter}$ 计算得到。

对于定位任务，类似地得到 $O \in \mathbb R ^ {H \times W \times 8}$ 用于调整定位对齐。对齐方式为，

$$B ^ {align}(i,j,c) = B(i+O(i,j,2\times C), j+O(i,j,2 \times C+1),c) \tag{6}$$

从 (6) 式可见，$O$ 中相邻两个 channel 分别表示 $H$ 和 $W$ 两个方向地 offset 。

$$\begin{aligned} M &= \sigma(conv _ 2 (\delta (conv _ 1 (X ^ {inter}))))
\\\\ O &= conv _ 4(\delta (conv _ 3 (X ^ {inter})))
\end{aligned} \tag{7}$$

$M$ 和 $O$ 使用 TAL 学习（由 TAL 提供反馈机制）。

## 2.2 TAL

### 2.2.1 样本分配

样本分配原则：

1. 对齐的 anchor 应该预测出较高的分类得分和精确的定位

2. 没有对齐的 anchor 应该有较低的分类得分预测，从而被 NMS 抑制

基于这两个原则，本文设计了一个 anchor 对齐指标。

**anchor 对齐指标**

考虑分类得分以及，预测 box 和 gt box 的 IoU，设计如下指标用于测量 anchor 的对齐程度，

$$t = s ^ {\alpha} \times u ^ {\beta} \tag{8}$$

其中 $s, \ u$ 分别表示分类得分和 IoU 值，$\alpha, \ \beta$ 用于控制两个值对指标的影响程度。

**训练样本分配**

为了提高两个任务的对齐，我们着重于对齐了的 anchors，然后采取如下一种简单的样本分配策略：

1. 对于每个实例（图像），选择 top-m 个具有最大 $t$ 值的 anchors 作为正样本
2. 剩余的 anchors 作为负样本

### 2.2.2 损失

**分类目标函数**

为了提高对齐 anchors 的分类得分，同时降低未对齐 anchors 的分类得分，使用 $t$ 作为 gt label，不使用 二值 label，需要注意的是，这里每个分类单独预测，`C` 个分类就有 `C` 个独立的分类预测。但是如果提高 $\alpha, \beta$ 的值，$t$ 反而变小，导致学习很难收敛，于是使用 $t$ 的归一化版本 $\hat t$ 。每一个 instance（一个目标 gt box？）独立进行归一化：

$$\hat t _ i = \frac {t _ i} {\max (\mathbf t)} \tag{9}$$

$\mathbf t$ 为这个 instance 中所有 $t$ 值构成的列表。

以前，一个 anchor point 位于 gt box 内部，那么就为正例。同一个gt box 内部的所有正例 anchor points 的分类 label 均为 1 (对应分类的 label 为 1， 其他分类 label 为 0，各分类独立预测)

现在，由于两个任务的最佳 anchor point 不对齐，所以选择 top-m t 值的 anchor points 作为正例。同一个gt box 内部的各个正例 anchor points 的分类 label不同，为 $\hat t$，这样最对齐的那个 anchor pixel 才能既具有最大分类得分，又最有最大 IoU（即，最准确定位）。

正例分类损失为

$$L _ {cls \_ pos} = \sum _ {i=1} ^ {N _ {pos}} BCE(s _ i, \hat t _ i) \tag{10}$$

其中 BCE 表示二值交叉熵 $BCE = -y\log x - (1-y) \log(1-x)$ ，$N _ {pos}$ 表示正样本数量。

使用 focal loss: $FL=-y(1-x)^{\gamma} \log(x) -(1-y)x^{\gamma} \log(1-x)$，那么对比 (10) 式，可知分类（包括正负样本）损失为，

$$L _ {cls} = \sum _ {i=1} ^ {N _ {pos}} |\hat t _ i - s _ i| ^ {\gamma} BCE(s _ i, \hat t _ i) + \sum _ {j=1} ^ {N _ {neg}} s _ j ^ {\gamma} BCE(s _ j, 0) \tag{11}$$

**定位目标函数**

一个很好地对齐了的 anchor（即，具有较大 t 值）应该具有较大的分类得分以及精确定位，所以 t 值可以用于选择高质量的 box，方法是使用 t 值作为损失的权重，也就是说，box 质量越高，对模型学习帮助越大，所以应该分配更大的权重。

使用 GIoU 作为坐标损失，那么定位目标函数为

$$L _ {reg} = \sum _ {i=1} ^ {N _ {pos}} \hat t _ i  L _ {GIoU}(b _ i, \overline b _ i) \tag{12}$$

其中 $b$ 表示预测 bbox，$\overline b$ 表示 gt box 。