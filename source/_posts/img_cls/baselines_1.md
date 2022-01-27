---
title: 图像分类网络（一）
date: 2021-12-23 16:44:03
tags: image classification
mathjax: true
---

**总结** 图像分类网络的结构，包括 `VGG`, `ResNet`, `DenseNet`, `MobileNet`， `MobileNetV2` 以及 `Inception` 。

<!--more-->

# 1. VGG
[论文地址](https://arxiv.org/abs/1409.1556)

VGG 网络结构如图 1 所示，
![](/images/img_cls/baselines_1.png)

图 1. VGG 网络结构。卷积层为 "conv<kernel size>-<output channels>" 的表示形式。表格中粗体表示新增的 layer。ReLU 省略（实际上每个 Conv 后面都跟一个 ReLU 激活层，Conv-ReLU 表示一个 layer。）

说明：

1. **LRN** 表示 Local Response Normalisation。表示对某层的激活值做归一化，用到同一 output feature 空间位置 (x,y)，不同 channel 的激活值加权平均，具体可参考论文 [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) 。VGG 论文中指出 LRN 在 ILSVRC 数据集上并不能带来提升，故对于较深的 VGG13 ~ VGG19 均没有采用 LRN。

2. 整个网络的 down sampling rate 为 $2^5=32$。如果网络 input size 为 `(224,224)`，那么输出特征 size 为 `(7,7)`。

# 2. ResNet
[论文地址](https://arxiv.org/abs/1512.03385)

ResNet 网络结构如图 2，
![](/images/img_cls/resnet_1.png)

图 2. ResNet 结构。

说明：

1. `conv2_x` 中 `x` 表示第二组卷积中的第 `x` 个卷积块，conv-block。每个 conv block 由 方括号中所列的 conv layer 堆叠而成。每个 conv layer 均为 `Conv-bn-ReLU` 的形式。

2. 当网络较深时（50-layers 以上），改变 conv block，如图 3，

    ![](/images/img_cls/resnet_2.png)
    图 3. 左：普通 conv block；右：bottle-neck conv block

3. 每个 conv block 有一个并列的 shortcut 分支。例如 `conv2_1` 这个 conv block，其输入 记为 `x`，在 `conv2_1` 最后一个 conv layer 的 ReLU 之前，输出记为 `out`，那么 `out+x` 再由 `conv2_1` 最后一个 ReLU 激活，输出为 `ReLU(out+x)`。

4. `conv3_1`，`conv4_1` 和 `conv5_1` 这三个 conv block 中的第一个 `3x3` 的 conv layer 具有 **stride=2**，故整个网络的 down sampling rate 为 $2^5=32$。

5. `conv3_1`，`conv4_1` 和 `conv5_1` 这三个 conv block 存在下采样，故融合 `out+x` 存在一个问题：`x` 与 `out` shape 不同。以图 2 为例说明，对于 34-layer，`conv3_1` 的输入 `x` shape （h,w,c）为 `56*56*64`，而 `conv3_1` 的 `out` shape 为 `28*28*128`。有两种解决方案：
    - 将 `x` 通过 zero-pad 升维，升维后，x 通过 stride=2 采样，使得 feature size 与 `out` 的 feature size 相同。
    - 对 `x` 使用 `1x1` conv 进行升维，同时 conv stride=2，降低 feature size，然后在接一个 `BN` 。（torchvision 实现中统一使用这种方法， projection shortcut）

    如果 `x` 与 `out` 的 shape 相同，那么不对 `x` 进行处理，即 identity shortcut。


# 3. DenseNet

[论文地址](https://arxiv.org/abs/1608.06993)

引入了 DenseBlock 结构，如图 4 ，
![](/images/img_cls/densenet_1.png)

图 4. 一个具有 5 层的 dense block（dense block 后面的那个 transition layer 也计算在内？否则，只有 4 层）。

简单而言，记 dense block 层数为 `L`，那么第 `l` 层的输出作为后续 `L-l` 个 layer 的输入。

dense block 说明：

1. 第 `l` 层输出记为 $x_l$，它有 `l` 个输入 $x_0, x_1, \cdots, x_{l-1}$，其中 $x_0$ 为本 dense block 的输入。这 `l` 个输入的融合 **不是相加**，而是 **沿着 channel 方向 concatenate**，这一点与 ResNet 不同。
2. dense block 中各层的输出 feature size 相同，这使得 concatenate 操作简单。
3. dense block 中各层的输出 channel 也相同，记为 `k`。（使用了 bottle neck 结构除外）


![](/images/img_cls/densenet_3.png)

图 5. DenseNet 网络结构（应用于 ImageNet 数据集上）。**图中每个 conv 均表示 `BN-ReLU-Conv`。

DenseNet 说明：

1. 每个 conv layer 由 `BN-ReLU-Conv` 构成（注意顺序，参考 [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) ）
2. dense block 中每层输出 channel 数 `k` ，称为 **growth rate**。`k` 可以很小，例如 `12, 24, 32` 等。
3. dense block 中可以采用 bottleneck 结构，即，dense block 由 `L` 个 bottleneck 构成，bottleneck 机构 为 `BN-ReLU-Conv(1x1)-BN-ReLU-Conv(3x3)`，采用这种结构的网络称为 `DenseNet-B` 。论文中，bottleneck 中的 `Conv(1x1)` 的输出 channel 为 `4k` 。bottleneck layer 的输出仍为 `k` ，这与普通 dense block 中各层输出 channel 相同。
4. 整个网络的 down sampling rate 为 $2^5=32$。
5. transition layer 中的 `1x1 conv` 的输入 channel 为 $m=k_0+k \times (L-1)$，输出 channel 为 $\lfloor \theta m\rfloor$，其中 $0 < \theta \le 1$，$\theta$ 用于压缩模型。当 $\theta < 1$ 时，称为 `DenseNet-C`，那么 `DenseNet-BC` 表示既采用了 bottleneck layer，又压缩了 transition layer 的输出 channel。

对于 cifar 数据集，由于图片尺寸较小（32*32），故没有 图 5 中的 `7x7` 卷积和 `maxpool`。采用了两种网络：
1. basic DenseNet
    
    图片输入首先通过一个 `3x3 conv` 的 layer，输出 channel 为 `16`。 然后通过三个 dense block，（每两个 dense block 依然有 transition layer，共 2 个 transition layer）。dense block 配置为 `{L=40, k=12}, {L=100, k=12}, {L=100, k=24}`。down sampling rate 为 $2^2=4$ 。

2. DenseNet-BC

    图片输入首先通过一个 `3x3 conv` 的 layer，输出 channel 为 `32`，后面的结构与 basic DenseNet 中总体相同，三个 dense block 的配置为 `{L=100, k =12}, {L=250, k=24}, {L=190, k=40}`。

# 4. MobileNet

## 4.1 Depthwise Separable Convolution

记为 `Conv dw`。记输入 channel 为 `M`，卷积核为 `k*k*1`，共 `M` 个这与的 kernel，输入特征的每个 channel 分别使用一个 kernel 进行卷积，故，输出 channel 依然是 `M` 。

## 4.2 Pointwise Convolution
kernel shape 为 `N*M*1*1`，用于 `Conv dw` 之后，使得输出 channel 变成 `N` 。

![](/images/img_cls/baselines_2.png)

图 6. MobileNet 网络结构。其中 Filter shape 为 `(h, w, in_channel, out_channel)` 的顺序。图中每个 Conv 都是 `Conv-BN-ReLU` 结构。（图中最后一个 s2 应该改为 s1）。

## 4.3 Width Multiplier
引入参数 $\alpha \in (0,1)$， 原来的 conv layer 的输出 channel `M, N` 分别变为 $\alpha M$，$\alpha N$。常用的 $\alpha$ 可取 `1, 0.75, 0.5` 等。

## 4.4 Resolution Multiplier
引入参数 $\rho$，使得每层的输入 feature 的空间 size $(D_F, D_F)$ 变为 $(\rho D_F, \rho D_F)$ 。

实际应用中，是将图 6 中每层的输出 channel 分别变为原来的 $\alpha$ 倍。将 input size 变为原来的 $\rho$ 倍即可实现 “每层的输入 feature size 变为 $\rho$ 倍” 这一目的。

# 5. MobileNetV2

MobileNetV2 是在 MobileNet 基础上进行改进：
1. 高 channel 的 layer 后保留使用 ReLU，而低 channel 的 layer 后不使用 ReLU，这是因为如果 channel 较大，即使使用 ReLU，大部分信息仍然保持，而如果 channel 较小，那么 ReLU 会损失较多信息。
2. bottleneck 结构会压缩 channel ，所以根据第 `1` 点，bottleneck 的 `1x1 conv` （正是这个 layer 用于压缩 channel）后不应该使用 ReLU。
3. MobileNet 没有使用 ResNet 中的 shortcut，MobileNetV2 则使用了 shortcut ，但是与 ResNet 中的 residual block 不同，而是 Inverted residual block。如图 7，

    ![](/images/img_cls/baselines_3.png)
    图 7. 普通 residual block 和 inverted residual block。（都是 bottleneck block）

    图 7 的右边的这个 inverted sidual block，block 的输入 channel 经过 `1x1 conv` 使得 channel 膨胀到 `k` 倍，然后使用 `3x3 dw conv` ，由于是 depthwise 可分离卷积，参数量和计算量都降下来了，而 channel 经过膨胀，故可以先使用 ReLU，然后再做 `3x3 dw conv`，得到的 feature channel 保持不变，故继续使用 ReLU 进行非线性激活，然后再使用 `1x1 conv`，得到的输出将作为下一个 block 的输入。具体的 block 各层结构如图 8

    ![](/images/img_cls/baselines_4.png)
    图 8

4. shortcut 融合采用按位相加，即 bottleneck 的输入与输出按位相加。

5. ReLU6 就是限制 ReLU 的最大输出值为6（对输出值做clip）。

整个网络结构

![](/images/img_cls/baselines_5.png)
图 9. MobileNetV2：`t` 表示 channel 的膨胀率，`c` 表示 bottleneck block 的输出 channel。`n` 表示相同的 bottleneck 重复堆叠几次。`s` 表示这一组重复的 block 总的 stride。

对图 9 中的结构作补充说明：

1. 除 bottleneck block 中最后一个 `1x1 conv` 之外，所有的 `conv` 具体展开均为 `Conv-BN-ReLU6`。如果是 `dw conv`，那么对应的则为 `dw Conv-BN-ReLU6`。bottleneck block 中最后一个 `1x1 conv` 则展开为 `Conv-BN`，这是因为两个 block 之间的 feature channel 较小，不应该使用 ReLu，而 block 内部由于第一个 `1x1 conv` 使得 channel 经过膨胀，所以可以使用 ReLU。

2. 一组相同的 block 中，如果 `s>1`，那么由这组中的第一个 block 执行 `stride=s`，具体而言是由这个 block 中的 `3x3 dw conv` layer 执行。参见 图 8。
3. 在 block 的 `stride=1` 且 block 的输入和输出 channel 相等时，才有 shortcut 连接。如图 8 所示，当 `stride=2` 时，没有 shortcut 。另外，从图 9 中可见，两个相邻的 block 如果跨组了，那么后一个 block 的输入输出 channel 肯定不同，即，每组中的第一个 block 不使用 shortcut，这样就保证了输入输出 feature 的 shape 完成相同，方便实现按位相加。

4. torchvision 中的 MobileNetV2 实现源码，如果 block 的膨胀系数 `t=1`，那么就省去了第一个 `1x1 conv`，直接是 `3x3 dw conv` ，可能是因为既然 `1x1 conv` 没有提高 channel，就没必要用了，在图 9 中，第一个 `conv` 的输出 channel 32，可能是因为觉得在网络的浅层阶段，这个 channel 已经足够大了，没必要再用 `1x1 conv` 进行膨胀，即，第一个 bottleneck 中省去了第一个 `1x1 conv`。

# 6. Inception
论文：[Going deeper with convolutions](https://arxiv.org/abs/1409.4842)

思路：提高性能的一个方法是增加网络的 `depth` 以及 `width`，但是随之而来的缺点是：
1. 大尺寸网络意味着大量的参数，在数据集不是足够大时，容易过拟合
2. 大尺寸网络增加计算量。

设计细节：
1. 低层网络负责一个小的 local region，到下一 layer 时，采用 `1x1 Conv`，其感受野不变，但是我们也希望有更大的感受野，可采用 `3x3 Conv` 和 `5x5 Conv`，如图 10 左边部分所示，但是这种设计存在问题：`5x5 Conv` 计算量太大，故采用图 10 右侧部分，使用 `1x1 Conv` 先降维，然后再卷积。
![](/images/img_cls/baselines_6.png)
图 10. 左：朴素的 Inception 模块；右：具有降维的 Inception 模块。

说明：
1. Inception 中的 max pooling `stride=1`，feature 空间 size 不变。

## 6.1 GoogLeNet 

![](/images/img_cls/baselines_7.png)
图 11. GoogLeNet 网络。图中 `#3x3 reduce` 实际上是 `1x1 Conv` 对应的是图 10 中右图的 `3x3 Conv` 前的降维卷积。`depth` 表示这个 module 中堆叠了几个 `Conv`。

说明：
1. 表中第三行，参数量为 $64\times 1^2 \times 64+64\times 3^2 \times 192=112\times 1024=112K$

2. 表中第五行，根据图 10 中右图，可知参数量为 $192\times 1^2 \times 64+192\times 1^2 \times 96 + 96\times 3^2 \times 128+ 192 \times 1^2 \times 16 +16 \times 5^2 \times 32 + 192\times 1^2 \times 32=159.5K$。这一 layer 的输出 channel 为 $64+128+32+32=256$。

3. 所有的 `Conv` 后接 ReLU。

4. 由于网络相对较深，梯度反向传播可能较为困难。网络的中间层已经具有辨别区分能力了，故在网络的中间层增加两个辅助分类分支，参见论文 [Going deeper with convolutions](https://arxiv.org/abs/1409.4842) 中的 `Figure 3`，（图太大，不贴出来）这两个辅助分支说明：
    - 在 `(4a)` 和 `(4d)` 后使用一个 `5x5 avg pooling`，`stride=3` ，得到输出为 `4x4x512` 和 `4x4x528`
    - 一个 `1x1 Conv, 128` 的 filters 用于降维（后接 ReLU）
    - 一个全连接层，输出 units 为 `1024`，后接 ReLU
    - drop ratio 为 `70%` 的 dropout layer
    - 一个全连接层，输出 units 为 `1000`（ImageNet 数据集），这个 layer 后面没有 ReLU，而是 softmax loss 层。
    - 这两个辅助分类分支在 inference 阶段被移除

