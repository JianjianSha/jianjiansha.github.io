---
title: DSOD
date: 2019-07-08 09:14:40
tags: object detection
mathjax: true
---
论文 [DSOD: Learning Deeply Supervised Object Detectors from Scratch](https://arxiv.org/abs/1708.01241)
# Introduction
近几年来提出了很多新型 CNN 网络结构，如 Inception、ResNet 以及 DenseNet 等，带动了包括目标检测在内的诸多 CV 任务的发展。通常来讲，目标检测都是在 backbone 后增加检测子网络，backbone 在分类 benchmark 如 ImageNet 进行预训练，然后使用目标检测数据集对整个网络进行 fine-tune，也就是所谓的迁移学习。但是这种设计范式具有三个不足之处：
1. 有限的结构设计空间。基于 ImageNet 预训练的 backbone 通常是较为庞大的网络，参数量巨大，所以用在目标检测时，不容易调整网络结构。
2. 学习偏向性。由于分类和目标检测任务两者的损失函数以及分类分布情况均不相同，导致不同的搜索/优化空间，对目标检测任务而言，模型学习可能偏向一个局部最优解。
3. 领域不匹配。fine-tuning 虽然可以缓和不同数据集的不同分类分布，但是当源域（ImageNet）与目标域（深度图像，医学图像等）有着严重不匹配时，这依然是个问题。

于是我们考虑两个问题：目标检测网络是否可以 train from scratch？如果可以，是否存在一些网络结构设计原则使得保持高检测准确率的同时让网络轻量？我们提出深度监督目标检测器 DSOD 以满足以上两个问题。

# DSOD
## 结构
DSOD 与 SSD 类似，是一个多尺度的无 proposal（one-stage）的目标检测网络。DSOD 结构分为两部分：用于抽取特征的 backbone 子网络，以及在多尺度响应图（response maps）上预测子网络（这里也称前端子网络）。backbone 是深度监督的 DenseNet 的变体（深度监督指的是对网络隐藏层和输出层直接使用目标检测数据集监督训练，而不是先使用 ImageNet 预训练，再使用目标检测数据集 fine-tune），这个 DenseNet 组成包括一个 stem block，四个 dense block，两个 transition layer 以及两个不带池化层的 transition layer。前端子网络使用一个dense结构融合了多尺度预测响应，如图 1 展示了 DSOD 前端子网络，以及 SSD 中使用的朴素多尺度预测 maps 结构。
![](/images/DSOD_fig1.png)<center>Fig 1: 预测子网络。左边是 SSD 中所用的朴素结构；右边是 dense 结构</center>

整个 DSOD 网络结构如表 1 所示。

|      Layers      | Output Size (Input 3x100x100) |       DSOD        |
|      :----:      |  :--------:                   |     :-----:       |
| Stem Convolution | 64x150x150                    | 3x3 conv, stride 2|
| Stem Convolution | 64x150x150                    | 3x3 conv, stride 1|
| Stem Convolution | 128x150x150                   | 3x3 conv, stride 1|
| Stem Convolution | 128x75x75                     | 2x2 max pool, stride 2|
| Dense Block (1)  | 416x75x75                     | $\begin{bmatrix} 1 \times 1 & conv \\\\ 3 \times 3 & conv\end{bmatrix} \times 6$|
| Transition Layer (1)| 416x75x75 <br> 416x38x38   | 1x1 conv <br> 2x2 max pool, stride 2|
| Dense Block (2)  | 800x38x38                     | $\begin{bmatrix} 1 \times 1 & conv \\\\ 3 \times 3 & conv\end{bmatrix} \times 8$|
| Transition Layer (2)| 800x38x38 <br> 800x19x19   | 1x1 conv <br> 2x2 max pool, stride 2|
| Dense Block (3)  | 1184x19x19                    | $\begin{bmatrix} 1 \times 1 & conv \\\\ 3 \times 3 & conv\end{bmatrix} \times 8$|
| Transition w/o Pooling Layer (1)| 1184x19x19     | 1x1 conv          |
| Dense Block (4)  | 1568x19x19                    | $\begin{bmatrix} 1 \times 1 & conv \\\\ 3 \times 3 & conv\end{bmatrix} \times 8$|
| Transition w/o Pooling Layer (2)| 1568x19x19     | 1x1 conv          |
| DSOD Prediction Layers | -                       | Plain/Dense       |

<center>Table 1: DSOD 结构 </center>

DSOD 设计原则如下：
### 无 Proposal
我们调查了如下三类 SOTA 的目标检测器：
1. R-CNN 和 Fast R-CNN，使用外部目标 proposal 生成器如 selective search。
2. Faster R-CNN 和 R-FCN 使用 RPN 生成 region proposals
3. YOLO 和 SSD，属于 single-shot 不生成 proposals（proposal-free），直接回归得到目标位置。

发现仅第三类（proposal-free）方法可以在没有预训练模型的情况下收敛成功。我们猜测这是由于前两类方法中的 RoI pooling 从每个 region proposal 中生成特征，这个 pooling 阻碍了梯度从 region 到 conv feature 的平滑反向传播。基于 proposal 的方法在有预训练的情况下工作良好是因为 RoI pooling 之前的 layers 的参数初始化足够好，而在 train from scratch 时由于没有预训练，所以那些 layers 参数初始化不够好，并在训练过程中梯度无法平法的反向传播过去，导致无法很好的更新这部分 layers 的参数。

于是，第一个设计原则为：training from scratch 需要 proposal-free 网络。

### 深度监督
中心思想是使用统一的目标函数对网络最初的隐藏层进行直接监督。这里我们使用密集层间连接如同 DenseNets 中那样来增强深度监督，即在一个 block 中当前 layer 与前面所有 layers 均有直接连接（也称 dense block），DenseNet 中初始的 layers 可通过 skip connections 得到来自目标函数的额外监督，所以只需要一个位于网络顶层的目标函数即可实现深度监督，并且能缓和梯度消失的问题。在 DenseNet 中，每个 transition layer 均包含池化层，所以要维持相同尺度的输出并增加网络深度，那么只能在 dense block 内部增加 layers，而我们所用的 Transition w/o pooling layer 由于不带有池化层，故消除了这种限制。

### Stem Block
Stem block 包含三个 3x3 卷积以及一个 2x2 最大值池化，其中第一个卷积步幅为 2。这个 stem block 明显提高了我们实验性能，相比较于 DenseNet 中的原始设计（7x7 卷积步幅为 2，后跟一个步幅为 2 的 3x3 最大值池化），stem block 可以降低输入 image 中的信息损失。

### 密集预测结构
图 1 展示了两种预测子结构：1. 朴素结构（源于 SSD）以及 2. 我们提出的密集结构。输入 image 大小为 300x300，6 种不同尺度的 feature maps 用于预测目标，其中 Scale-1 feature maps 来自 backbone 中间层，此 feature maps 尺度最大，为 38x38，用于小目标预测，其余五个尺度的 feature maps 来自于 backbone 之后的子结构。这个子结构构造方法为：如图 1 右边仅靠中心竖线的虚线框，相邻两个尺度 feature maps 之间使用 transition layer 连接起来，这个 transition layer 具有 bottleneck 结构：一个 1x1 卷积用于降低 previous scale 的 feature maps 的通道数，以及一个 3x3 卷积下采样得到 next scale 的 feature maps。

在图 1 中所示的 SSD 原始预测子结构中，每个尺度的特征均由上一个尺度的特征直接转变而来。我们提出的预测子结构是一个密集结构，融合了多尺度特征。为简单起见，限制每个尺度输出相等通道的 feature maps 用于预测。在 DSOD 中，除 scale-1 之外的每个尺度中，feature maps 有一半是通过一系列的 conv 从上一尺度中学习而来，这一系列的 conv 即图 1 右边仅靠中心竖线的虚线框所标注，剩余的一半 feature maps 则直接从相邻的高分辨率的 feature maps 中降采样得到，图 1 中最右边的虚线框标注，这个降采样包含 2x2 步幅为 2 的 max pooling，以及一个 1x1 步幅为 1 的 conv，其中 max pooling 是为了两边的 feature maps 的分辨率匹配从而能够 concatenate 起来，而 1x1 conv 则是为了将 feature maps 的通道数降为一半。max pooling 层位于 1x1 conv 之前可以降低计算损害。对每个 scale 而言，仅学习一半的新 feature maps，并重新利用一半的 previous feature maps。

# Experiments
实验部分略，可阅读原文以获取详细信息。

# Conclusion
提出 DSOD 用于 training from scratch，而这总训练方式适合 single-shot 的目标检测器，在 SSD 基础上，使用 DenseNet 作为 backbone，同时预测子网络也采用类似 DenseNet 的密集连接网络，实现了深度监督。