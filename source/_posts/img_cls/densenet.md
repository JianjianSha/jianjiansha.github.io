---
title: DenseNet
p: img_cls/densenet
date: 2019-12-31 15:27:49
tags: image classification
mathjax: true
---

论文 [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

随着现在数据集越来越大，网络容量也需要增大，否则容易出现过拟合现象，一种增大网络容量的方法是增加更多的 layer ，让网络更深（deep），但是这也会带来新的问题：随着输入（或梯度）经过越来越多的 layer，信息可能在到达最后一层（对于梯度则是反向传播在到达网络第一层）之前就消失了。ResNet 增加 shortcut，即 early layer 到 later layer 之间直接连接，以此来解决这个问题。本文提出 densenet，将这种连接风格贯彻到底，网络结构如图1，

![](/images/img_cls/densenet_1.png)


## DenseNet 数学描述
假设输入为 $\mathbf x_0$，网络共有 $L$ 层，每层非线性转换操作记为 $H_l(\cdot)$，输出记为 $\mathbf x_l$。
### ResNet
传统的网络，第 $l$ 层操作为 $\mathbf x_l = H_l(\mathbf x_{l-1})$ ResNet 增加一个 Identity 分支后为 $\mathbf x_l = H_l(\mathbf x_{l-1})+\mathbf x_{l-1}$，ResNet 中一个 layer 其实是一个 block，不是单层 layer，block 内部的 layer 没有 shortcut 分支，这点需要注意。
### Dense 连接
从某个 layer 到所有后继 layer 之间增加直接连接，对于第 $l$ 层 layer 来说，其输入是所有前序 layer 的输出以及原始网络输入，所以有
$$\mathbf x_l=H_l([\mathbf x_0, ... , \mathbf x_{l-1}])$$
其中 $[\mathbf x_0, ... , \mathbf x_{l-1}]$ 表示 `concatenate` 操作，这与 ResNet 的 `sum` 操作不同。

### 组合函数
DenseNet 中每个 layer 的操作 $H_l(\cdot)$ 是由：
1. 批规范化 BN；
2. ReLU；
3. `3x3` conv
   
组合而得。
### Pooling 层
网络肯定存在下采样，这就导致 early layer 与 later layer 可能有不同的 feature maps 大小，这就导致没法直接连接。一种解决的办法如图 2，将若干个 layer 分为一组作为一个 block，称为 Dense Block，block 内部的 feature maps 大小保持不变，从而 block 内部的 layer 之间可以进行密集连接，block 之间使用过渡层进行下采样，在作者实验中过渡层包含 BN 层、`1x1` 卷积层以及 `2x2` 均值池化层。
![](/images/img_cls/densenet_2.png)

### Growth rate
记 $k_0$ 为 Dense Block 初始输入的 channels，如果每个 $H_l$ 输出均为 $k$ 个 feature maps，由于 $l^{th}$ layer 的输入为初始网络输入 $\mathbf x_0$ 以及前 $l-1$ 个 layer 输出的 concatenation，所以共有 $k_0+k(l-1)$ 个 feature maps，所以 $k$ 值即使较小，随着 $l^{th}$ 增大，深层 layer 的 in_channels 也可以很大，因为可以使用较小 $k$ 值，这就使得 DenseNet 与传统网络相比，拥有更少的网络参数。记 $k$ 为 _growth rate_，表示 layer 输入 feature maps 增长量。作者实验表明，即使非常小的 _growth rate_ 也可以获取很好的测试结果。作者解释为，每个 layer 可以利用同 block 内的前面所有 layer 的输出 feature maps，也就是 “collective knowledge”，将 feature maps 看作网络的 global state，每个 layer 贡献自己的 k 个 feature maps 到这个 global state，同时 _growth rate_ 控制了每个 layer 对 global state 的贡献量，并且每次某个 layer 对 global state 贡献完毕，此时的 global state 可以被之后所有 layer 利用。传统网络为了在 layer 到 layer 之间传递这种 global state，不得不使用越来越多的输出通道数，达到复制前面各阶段的 global states 的效果。

### Bottleneck layers
later layer 的输入通道数较大，可以在这些 layer（通常是 `3x3` 卷积） 前面增加一个 `1x1` 卷积作为 bottleneck layer，降低 later layer 的输入通道数，提高计算效率。记这种带有 bottleneck layer 的 DenseNet 为 DenseNet-B，其中 layer 的操作 $H_l(\cdot)$ 变为 BN-ReLU-Conv(1x1)-BN-ReLU-Conv(3x3)（参考前面 __组合函数__ 小节）。

### Compression
为了进一步压缩模型，可以在过渡层降低 feature maps 的数量，记 dense block 的输出 feature maps 数量为 $m$，注意是 dense block 初始输入和各 layer 输出的叠加，参加图 1，过渡层输出 feature maps 数量为 $\lfloor \theta m \rfloor$，其中 $0 < \theta \le 1$，当 $\theta=1$，过渡层不改变 feature maps 的数量，当 $\theta <1$ 记这样的 DenseNet 为 DenseNet-C，而 DenseNet-BC 则表示既有 bottleneck layer 又有 $\theta <1$ 的 DenseNet。

### 实现细节
作者在 CIFAR-10,CIFAR-100,SVHN 以及 ImageNet 四个数据集上评估了 DenseNet。

在前三个数据集上（这三个数据集的图像 size 均为 `32x32`），DenseNet 使用了 3 个 dense block，每个 dense block 有相等数量的 layer。在进入第一个 dense block 之前，输入图像数据先经过了一个 `1x1` 16-out_channels 的卷积层（对于 DenseNet-BC 网络，输出通道数为 _growth rate_ 的两倍）。对于 `3x3` conv，进行 padding=1 的 zero 填充，以保持 feature map size 不变。在最后一个 dense block 之后，进行全局均值池化以及 softmax 操作。三个 dense block 的 feature maps 大小分别为 `32x32, 16x16, 8x8`。使用最普通的 DenseNet（不带 B\C 后缀）进行实验时，dense block 配置分别为 $\{L=40,k=12\}$, $\{L=100,k=12\}$ 和 $\{L=100,k=24\}$，使用 DenseNet-BC 进行实验时，配置分别为 $\{L=100,k=12\}$, $\{L=250,k=24\}$ 和 $\{L=190,k=40\}$。

对于 ImageNet，使用 4 个 dense block 的 DenseNet-BC，输入图像大小为 `224x224`，初始 layer 为一个 `7x7` 2k-out_channels，stride=2 的卷积，其中 k 表示 _growth-rate_，所有 dense block 均为 $k=32$，网络具体描述如表 1 所示，
![](/images/img_cls/densenet_3.png) <center>ImageNet 对应的四个 DenseNet-BC 网络配置。所有网络的 k=32，表中所有 conv 均表示 __BN-ReLU-Conv__</center>

## 实验
实验结果的对比可直接参考原论文。

### 数据集
__CIFAR：__  CIFAR-10 和 CIFAR-100 数据集包含 `32x32` 大小的彩色图像，其中 CIFAR-10 的图像分类数量为 10，CIFAR-100 图像分类数量为 100。训练集和测试集大小分别为 50000 和 10000，训练集中取 5000 作为验证集。数据扩增使用 镜像/平移 两种方法。预处理包括 RGB 三通道分别使用 `mean` 和 `std` 归一化。最后一轮训练时使用全部 50000 个图像，然后计算测试错误。

__SVHN：__  SVHN 数据集包含 `32x32` 大小的彩色数字图像，训练集和测试集大小分别为 73257 和 26032，另有 531131 个图像作为额外的训练数据。作者实验中，不采用任何数据扩增手段，从训练集中取 6000 个图像作为验证集。选择具有最低验证错误的训练结果，然后计算测试错误。图像像素值除以 255 以落于 `[0,1]` 范围作为归一化处理方法。

__ImageNet：__  使用 ILSVRC 2012 分类数据集，包含 120 万训练集以及 5 万的验证集，分类总数为 1000。使用与 ResNet 中相同的数据扩增方法，采用 single-crop 或者 10-crop 方法得到 `224x224` 的输入大小，最后计算验证集上的错误率。

### 训练
CIFAR 和 SVHN 训练 batch 大小为 64，epoch 分别为 300 和 40。初始学习率 lr = 0.1，在 50% 和 75% 的 epoch 时分别再降为 10%。

ImageNet 训练 batch 大小为 256， epoch 为 90。初始学习率 lr=0.1，在 epoch = 30 和 epoch = 60 时分别降为 10%。

权重衰减因子为 $10^{-4}$，Nesterov momentum 为 0.9。

由于本文着重记录 DenseNet 的网络结构，所以实验结果数据的分析以及与其他网络的比较此处省略，可参考原文进行反复阅读理解。

## DenseNet 特点：

1. dense block 内每两个 layer 之间均存在直接连接。
2. 直接连接与普通前向传播的合并采用 `concatenate` 方式，而 ResNet 中则是 `sum` 方式。
3. 比传统网络有更少的参数。layer 之间的连接更加密集，所以 layer 只需要很小的 filter 数量。
4. 易于训练。这得益于 DenseNet 更优的信息流（梯度流）传递。early layer 可以直接利用到损失函数对原始输入的梯度，这种深度监督有益于训练。
5. layer 之间的密集直接连接有正则化效果，在小训练集上有助于降低过拟合。