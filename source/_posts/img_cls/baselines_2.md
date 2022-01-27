---
title: 图像分类网络（二）
date: 2021-12-27 10:44:36
tags: image classification
mathjax: true
---
**总结** 图像分类网络的结构，包括 `ResNeXt`, `ShuffleNet`, `ShuffleNetV2`, `SENet`， 以及 `Res2Net` 。

<!--more-->

# 1. ResNeXt
论文地址：[Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)

主要思路：**分组卷积**，改造 ResNet 中的 Bottleneck block，如图 1，
![](/images/img_cls/baselines2_1.png)
图 1. 左：ResNet 中的 block；右：ResNeXt 中的 block，称 path 的数量为 `cardinality`，图中 `cardinality=32`。（每一 layer 的标注顺序：`in_channels, filter size, out_channels`）

> 图 1 中多路分支的 `3x3` 卷积实际上就是 `128, 3x3, 128` 的分组卷积。

除了图 1 中的这种划分方法，还有其他两种方法，如图 2，
![](/images/img_cls/baselines2_2.png)
图 2. ResNeXt 可用的三种 block 结构。

整个 ResNeXt 结构如图 3，从图中可见，大体框架相同，只是 block 不同。
![](/images/img_cls/baselines2_3.png)
图 3. ResNet 与 ResNeXt 结构对比。`C` 表示 `cardinality` 。

说明：
1. ResNeXt 就是在 ResNet 基础之上，对 block 加以改进，即 `3x3 conv` 变为分组卷积，其他则几乎保持不变，故：
2. `stage 3/4/5` 的第一个 block（内的 `3x3 conv group`）有 `stride=2`。
3. 整个网络的 down sampling rate 为 $2^5=32$。
4. 在 `stride=2` 的 block 上的 shortcut 分支需要一个 `downsample` 映射函数，将 `x` 的 channel 增加一倍，feature size 缩小原来的到 $1/2 \times 1/ 2$。

# 2. ShuffleNet
论文地址：[ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)

## 2.1 Channel Shuffle
MobileNet 中采用了对 `3x3 conv` 采用了 depthwise seprable conv，导致在一个 block 中， `1x1 conv` 计算量占比非常高，ShuffleNet 对 `1x1 conv` 也采用分组卷积，但是连续的分组卷积，导致某个输出 channel 只能从一部分输入通道中获取信息，如图 4，于是 ShuffleNet 在 `1x1 conv` 分组卷积之前，将 channel 顺序重组，降低连续分组卷积导致的 side effect，
![](/images/img_cls/baselines2_4.png)
图 4. 分组数量为 `3`。左：连续两次分组卷积；中：第一次分组卷积之后，将 channel 重组，使得重组后每个分组具有所有原来分组中的部分 channel；右：使用 channel shuffle 实现 channel 重组，与（中）等效。


> channel shuffle 操作：记 channel 数量为 `g x n`，`g` 为分组数量，`n` 为每个分组内的 channel 数量；reshape channel 为 `(g,n)`，然后转置为 `(n, g)`，然后再 flatten 为 `n x g`，即可。

## 2.2 ShuffleNet 网络结构

ShuffleNet Unit 如图 5，
![](/images/img_cls/baselines2_5.png)
图 5. 左：ResNet 中的 bottleneck block。中：第一个 `1x1 conv` 后使用了 channel shuffle，第二个 `1x1 conv` 后没有 channel shuffle（实验验证说明使用了也没有什么改善）；右：stride=2 的 block。

图 5 的说明：
1. `GConv` 表示分组卷积。`DWConv` 表示深度可分离卷积。
2. `DWConv` 后没有 `ReLU`。
3. `stride=2` 时，shortcut 采用 `(3x3 avg pool stride=2)` 进行降采样，然后使用 `concat` 进行 **特征融合**。


ShuffleNet 总体结构如图 6 所示，
![](/images/img_cls/baselines2_6.png)
图 6. ShuffleNet 网络。`Stage 2` 的第一个 `1x1 conv` 没有应用 `group conv`，这是因为其输入 channel 太小（为 `24`）。

图 6 的补充说明：
1. 三个 `stage`，每个 `stage` 的第一个 `block` 具有 `stride=2`，具有 `stride=2` 的 block 结构参见图 5 右。
2. ShuffleNet Unit 的各 layer 参数为 `c, 1x1, m, g` -> `channel shuffle` -> `m, 3x3, m, dwConv` -> `m, 1x1, c,g`，其中 `c` 为 block 的输入 channel，`m` 为 bottleneck 的 channel。
3. `block` 的内部 bottleneck 的 channel （参见图 5 中间 block 的第一个 `1x1 GConv` 的输出 channel）为 block 输出 channel 的 `1/4`，例如，`stage 2` 的每个 block 的输出 channel 为 `200`（对应 `g=2` 这个 case），那么 bottleneck channel 为 `50`。
4. 整个网络的 down sampling rate 为 $2^5=32$。
5. 为进一步控制网络的复杂度，对各 layer 的输出 channel，乘以一个因子， `s`，图 6 中则对应 `s=1` 的情况，记为 `ShuffleNet 1x`，那么 `ShuffleNet sx` 则大约是 `ShuffleNet 1x` 复杂度的 $s^2$ 倍。

# 3. ShuffleNetV2
论文：[ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164)

本文所考虑的点：
1. FLOP 是间接指标，network 运行速度才是直接指标，而影响速度的除了 FLOP，还有 memory access cost（**MAC**）等。此外，CUDNN 库中，`3x3` 卷积并不比 `1x1` 卷积慢 `9` 倍。
2. network 运行在不同的设备上，其速度也不同，例如 GPU 和 ARM，过去声称为了 移动设备 而设计的网络如 MobileNet 和 ShuffleNet 并没有真正在移动设备上测试。


分析（以下结论的前提是 `FLOP` 保持不变）：
1. 输入输出 channel 相等时 MAC 最小。
2. 分组卷积中，分组数量越大, MAC 也越大。
3. 增加网络分支数量会降低计算的并行度。
4. Element-wise 计算不可忽略。

## 3.1 ShuffleNetV2 网络设计
在 ShuffleNetV1 基础上进行修改，如图 7 所示，
![](/images/img_cls/baselines2_7.png)

图 7. 左一： basic ShuffleNet block；左二：具有 down sampling(2x) 的 ShuffleNet block；左三：basic ShuffleNetV2 block；右一：具有 down sampling(2x) 的 ShuffleNetV2 block。

ShuffleNetV2 的说明：
1. block 的输入沿 channel 方向 split 成两部分（通常是 `二等分`）：一部分作为 identity shortcut；另一部分经过**三个卷积层**，输入输出 channel 均相同。
2. block 中的两个 `1x1 conv` 不是 group Conv，而是普通 Conv（分组数量大会增加 MAC）。另外 channel split 可视作一种分组。
3. 最后两个分支 concatenate，使得 block 的输入输出 channel 保持相等。
4. block 的最后使用 `channel shuffle`，这样是为了使得两个分组之间实现信息交流。
5. down sampling rate 为 `2` 时，去掉 `channel split` 这个操作，故 block 的输出 channel 变为 `2` 倍。

整个网络的结构如图 8，

![](/images/img_cls/baselines2_8.png)
图 8. ShuffleNetV2 的网络结构。分别取四种不同的 channel。

说明：
1. Global Pool 之前增加了一个 `1x1 conv`，用于 mix up features。

# 4. SENet
论文：[Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)

思路：上一 layer 的输出经过 channel 加权后再输入到下一 layer。如图 9 所示，
![](/images/img_cls/baselines2_9.png)

## 4.1 squeeze

$$z_c=\mathbf F_{sq}(\mathbf u_c)=\frac 1 {H \times W} \sum_{i=1}^H \sum_{j=1}^W u_c(i,j)$$

## 4.2 Excitation

$$\mathbf s = \mathbf F_{ex}(\mathbf z, \mathbf W)=\sigma(g(\mathbf z, \mathbf W)) = \sigma(\mathbf W_2 \cdot \delta(\mathbf W_1 \cdot \mathbf z))$$

其中 $\mathbf W_1 \in \mathbb R^{\frac C r \times C}$，$\mathbf W_2 \in \mathbb R^{C \times \frac C r}$。$\delta$ 是 ReLU 函数， $\sigma$ 是 sigmoid 函数。

最终的输出则为

$$\hat {\mathbf x_c} = \mathbf F_{scale} (\mathbf u_c, s_c) = s_c \mathbf u_c$$

$c$ 为 channel 编号。


整个过程如图 10 所示，

![](/images/img_cls/baselines2_10.png)

图 10. 左：在 Inception 上使用 SE block；右：在 Residual 上使用 SE block

![](/images/img_cls/baselines2_11.png)
图 11. 左：ResNet-50；中：SE-ResNet-50；右：SE-ResNeXt-50

# 5. Res2Net
论文：Res2Net: [A New Multi-scale Backbone Architecture](https://arxiv.org/abs/1904.01169)

设计一个具有 multi-scale 特征的网络，以前的网络为了达到这一目的，均依赖于多个 layer，每个 layer 的输出 feature 的空间 size 不同，这是从 layer 层面实现，本文在 layer 内部实现 multi-scale 特征，具有更细粒度的级别，如图 12，

![](/images/img_cls/baselines2_12.png)

图 12. 左：普通的 bottleneck block；右：Res2Net 模块，图中 scale dimension `s=4`。

说明：
1. 将 `3x3 conv` 输出 channel 为 `n` 的 filters 替换为若干组小的 filters，每组 filters 的输出 channel 为 `w` （即，每组 `w` 个 filter）。
2. 将输入 feature maps 分成若干组。取第一组 filters，对一组 input feature maps 抽取特征，然后与另一组 input feature maps 一起（elementwise add?）送入第二组 filters，直到用完所有 filters。
3. 每次引入一个 `3x3 conv`，感受野就增大，从而得到 multi-scale 的特征。


## 5.1 Res2Net 模块

Res2Net 模块中，`1x1 conv` 之后，feature maps 被等分成 `s` 份，记 $\mathbf x_i, \ i \in \{1,2,\ldots , s\}$，每个 $\mathbf x_i$ 空间 size 相等，但是 channel 为原来的  $1/s$，除 $\mathbf x_1$ 之外，每个 $\mathbf x_i$ 对应一组 `3x3 conv` filters，记为 $\mathbf K_i$，显然有 $i > 1$，这组 filters 的输出为 $\mathbf y_i$，$\mathbf x_i$ 与 $\mathbf K_{i-1}$ 的输出相加，然后作为 $\mathbf K_i$ 的输入，于是

$$\mathbf y_i = \begin{cases}\mathbf x_i & i=1 \\ \mathbf K_i(\mathbf x_i) & i=2 \\ \mathbf K_i(\mathbf x_i + \mathbf y_{i-1}) & 2 < i \le s \end{cases}$$

所有的 $\mathbf y_i$ concatenate，然后通过 `1x1 conv` 进行特征融合。

如图 13，还可以结合 SE block（先 squeeze 得到 `1x1xC` 的特征，然后 excite 得到 `1x1xC` 的 channel 权重） 和 cardinality dimension（分组卷积），

![](/images/img_cls/baselines2_13.png)