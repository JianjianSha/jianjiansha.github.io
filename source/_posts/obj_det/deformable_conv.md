---
title: 可变形卷积网络
date: 2022-04-26 09:13:40
tags: object detection
mathjax: true
summary: 可变形卷积，可变形 RoIpooling
---
论文：[Deformable Convlutional Networks](https://arxiv.org/abs/1703.06211)
源码：[Deformable-ConvNets](https://github.com/msracver/Deformable-ConvNets)

# 1. Introduction
传统 CNN 的固定几何结构：固定的卷积单元（window），固定的池化，以及固定的 RoI pooling。某些时候这并不是我们想要的，例如 CNN 的高层 layer 对语义特征进行编码，不同的目标具有不同的 scale 和形状，此时，需要自动调节感受野 size。

本文提出两个新模块：可变形卷积和可变性 RoI pooling，增强了 CNN 的几何变换的建模能力。

# 2. 可变形卷积网络
在标准卷积的 grid 采样位置上增加一个 2D offsets（位移），使得采样 grid 可以自由变形，如图 1，

![](/images/obj_det/deformable_conv1.png)

<center>图 1. (a) 规则采样 grid；(2) 变形采样点（深蓝色点），offsets 为浅蓝色箭头; (c) (d) 为 (b) 的特例。</center>

在 3D 特征的每个 channel 上均做相同的可变形卷积，以下仅考虑 2D 情况。

## 2.1 可变形卷积
普通的 2D 卷积过程：在输入特征上使用规则 grid $\mathcal R$ 采样，计算采样点的加权和，例如 `3x3` 卷积，

$$\mathcal R = \{(-1, -1),(-1,0),\ldots, (0,1),(1,1)\}$$

输出特征平面 $\mathbf y$ 上一位置点 $\mathbf p_0$，有

$$\mathbf y(\mathbf p_0)=\sum_{\mathbf p_n \in \mathcal R} \mathbf w(\mathbf p_n) \cdot \mathbf x(\mathbf p_0+\mathbf p_n)\tag{1}$$

可变形卷积中，使用 offset $\{\Delta \mathbf p_n |n=1,\ldots, N\}$ 对规则 grid $\mathcal R$ 进行增强，

$$\mathbf y(\mathbf p_0)=\sum_{\mathbf p_n \in \mathcal R} \mathbf w(\mathbf p_n) \cdot \mathbf x(\mathbf p_0+\mathbf p_n+\Delta \mathbf p_n
) \tag{2}$$

$\Delta \mathbf p_n$ 是分数，故 (2) 式计算还需要通常线性插值来完成，记 $\mathbf p = \mathbf p_0+\mathbf p_n+\Delta \mathbf p_n$ 为目标位置处（分数），$\mathbf q$ 是特征平面 $\mathbf x$ 上所有的整数位置，$G(\cdot, \cdot)$ 是双线性插值 kernel，那么插值结果为

$$\mathbf x(\mathbf p)=\sum_{\mathbf q}G(\mathbf p, \mathbf q) \cdot \mathbf x(\mathbf q) \tag{3}$$

由于坐标位置是 2D 的，可以将 $G$ 写成

$$G(\mathbf p, \mathbf q)=g(q^x, p^x)\cdot g(q^y, p^y) \tag{4}$$

其中 $g(a,b)=\max (0, 1- |a-b|)$，即只需要 $\mathbf p$ 四周最近的 4 个点即可计算 $\mathbf x(\mathbf p)$ 的值。

### 2.1.1 获得 offsets

如图 2，__通过一个卷积层对输入特征进行卷积，得到这个输入特征的 offset__，

![](/images/obj_det/deformable_conv2.png)

<center>图 2. 3x3 的可变形卷积</center>

输出的 offsets 与输入特征平面具有相同的 spatial size，通道维度 $2N$ 表示卷积核中采样点数量 $N$ 个 2D offset（x,y 方向），参见 (2) 式中的 $\Delta \mathbf p_n$，注意，输出特征平面上的每一个点 $\mathbf p_0$，均有对应的独立的 $N$ 个 $\Delta \mathbf p_n$。

## 2.2 可变形 RoI Pooling

RoI pooling 是将 backbone 的输出特征上的 RoI 部分 pooling 为一个固定 size 的特征。

输入特征 $\mathbf x$，一个 RoI 其 size 为 $w \times h$，左上角坐标 $\mathbf p_0$，RoI pooling 将其分成 $k \times k$ 的 bins，输出 $k \times k$ 的特征 $\mathbf y$，对输出特征上一位置 $(i,j)$ 有

$$\mathbf y(i,j)=\sum_{\mathbf p_n \in bin(i,j)} \mathbf x(\mathbf p_0+\mathbf p_n) /n_{ij} \tag{5}$$

可见这是一个 均值 pooling，$n_{ij}$ 表示这个 bin 中像素的数量，bin 索引满足 $0 \le i,j < k$，`bin(i,j)` 内像素的相对坐标 $\mathbf p_n$ 满足

$$\lfloor i \frac w k \rfloor \le p_n^x < \lceil (i+1) \frac w k \rceil, \quad \lfloor j \frac h k \rfloor \le p_n^y < \lceil (j+1) \frac h k \rceil$$

__可变形 RoI pooling__ 是给普通 RoI pooling 增加一个 offsets $\{\Delta \mathbf p_{ij}|0 \le i,j < k\}$，即 RoI 中每个 bin 有一个独立的 2D offset，那么 (5) 式变为

$$\mathbf y(i,j)=\sum_{\mathbf p_n \in bin(i,j)} \mathbf x(\mathbf p_0+ \mathbf p_n + \Delta \mathbf p_{ij})/n_{ij} \tag{6}$$

同样地，由于 $\Delta \mathbf p_{ij}$ 是分数，(6) 式计算需要使用 (3),(4) 式的双线性插值计算。

图 3 显示了可变形 RoI pooling 过程：首先 RoI pooling 生成池化后特征，即 (5) 式；然后池化特征经过一个 `fc` layer 生成归一化的 offsets $\Delta \hat {\mathbf p}_{ij}$，然后再变换到 $\Delta \mathbf p_{ij}$，

$$\Delta \mathbf p_{ij}=\gamma \cdot \Delta \hat {\mathbf p}_{ij} \circ (w, h) \tag{7}$$

其中 $\gamma$ 是一个预定义标量用于调节 offset 的幅值，根据经验设置为 $\gamma=0.1$。这个 offset 归一化的设置使得 offset 与 RoI size 大小无关。


![](/images/obj_det/deformable_conv3.png)

代码见源码中的函数 `get_deformable_roipooling`。


### 2.2.1 PS RoI pooling
position-sensitive RoI pooling，输入特征平面上每一个 location，其对应的 offset 与这个位置所属分类以及 RoI pooling 后所属 bin 均相关。

输入特征经过一个 conv layer，得到 $k^2$ 个分类相关的 score maps，其中 $k^2$ 为 RoI pooling 后的 bin 数量，即 score maps 一共有 $k^2 \cdot (C+1)$ 个 channel，这里 $C$ 表示分类总数，$+1$ 表示背景分类。

使用 $(i,j)$ 表示 RoI pooling 后的 bin，RoI pooling 之后所有 bin 对应的 score maps 记为 $\{\mathbf x_{i,j}| 0\le i,j < k\}$。

在这些 score maps 上执行 RoI pooling，`(i,j)-th` bin 的值使用 $\mathbf x_{ij}$ 上对应 bin 的像素之和得到（笔者注：这里应该是求平均）。

简单而言，将 (5) 式中的 $\mathbf x$ 改为 $\mathbf x_{ij}$。

__可变形 PS RoI pooling__

1. 将 (6) 式中的 $\mathbf x$ 替换为 $\mathbf x_{ij}$。

2. 如何确定 (6) 式中的 $\Delta \mathbf p_{ij}$？

    输入特征经另一个 conv layer 得到 channel 为 $2k^2 \cdot (C+1)$ 的输出 offset fields，如图 4 中上面那个分支，对每个 RoI，使用 PS RoI pooling 作用于 RoI，过程为：
    - $k \times k$ 个 bins，每个 bin 使用独立的 offset fields，求 bin 内的像素均值，得到归一化的 offsets $\Delta \hat {\mathbf p}_{ij}$。单个 RoI 的 offsets 的 shape 为 $2(C+1) \times k \times k$
    - 使用 (7) 式转变为最终的 offsets。
3. 使用 (6) 式（$\mathbf x$ 替换为 $\mathbf x_{ij}$）得到最终的池化特征，其 shape 为 $(C+1)\times k \times k$

![](/images/obj_det/deformable_conv4.png)

<center>图 4. 3x3 可变形 PS RoI pooling</center>

### 2.2.2 RoI pooling 小结

四种 RoI pooling：
1. (普通) RoI pooling
2. 可变形 RoI pooling
    - 输入特征经 RoI pooling 后的特征再经 fc 得到 offsets
    - offsets 作用于输入特征，实现可变形 RoI pooling
3. PS RoI pooling
    - 输入特征经 conv 得到 $k^2(C+1) \times h \times w$ 的 score maps
    - 每个 bin 使用对应的 score map 进行均值池化，得到 $(C+1) \times k \times k$ 的池化特征 
4. 可变形 PS RoI pooling
    - 输入特征经 conv 得到 $k^2(C+1) \times h \times w$ 的 score maps
    - 输入特征经另一 conv 得到 $2k^2(C+1) \times h \times w$ 的 offset fields
    - 对 offset fields 执行 PS RoI pooling，得到 $2(C+1) \times k \times k$ 的 offsets
    - 每个 bin 有独立的 score maps，shape 为 $(C+1)\times h \times w$，以及独立的 offset，shape 为 $C+1$，求均值得到这个 bin 的池化结果 shape 为 $C+1$
    - 最终的特征 shape 为 $(C+1)\times k \times k$

__注意：__
以上 PS RoI pooling 中的 $(C+1)$ 表示是在分类分支中执行的 PS RoI pooling，如果是在坐标回归分支中，那么将 $(C+1)$ 替换为 $4 \cdot N_{reg}$，其中 $N_{reg}$ 表示回归分支上的分类数量，如果配置是类别不可知（），那么 $N_{reg}=2$ 表示 fg/bg 两个类别，如果是类别可知的，那么 $N_{reg}=C+1$。 $4$ 表示 4 个坐标。

可变形 PS RoI pooling 有上下两个分支如图 4，在坐标回归分支中，下分支中的 $(C+1)$ 替换为 $4\cdot N_{reg}$；上分支中的 $(C+1)$ 替换为 $1$，表示同一个 bin 中，不分类别不分坐标（x1 y1 x2 y2），共享同一个 offset $\Delta \mathbf p_{ij}$。


## 2.3 可变形卷积网络

本文提出的可变形卷积和可变形 RoI pooling 与普通版本均具有相同的输入输出，故容易用在现有的 CNN 中。训练时，这些新增的用于学习 offsets 的 conv 和 fc 的权重初始化为 0，这些权重的学习率为其他 layer 的 $\beta$ 倍，默认情况下 $\beta=1$，作者特别指出在 Faster R-CNN 中，新增的 fc 的权重学习率倍数 $\beta=0.01$。

SOTA CNN 框架通常包含两部分：深度全卷积网络用于从图像抽取特征；任务相关的浅网络，根据特征生成结果。

### 2.3.1 用于特征抽取的可变形 conv
作者采用两个 SOTA 框架进行特征抽取：ResNet-101 和 Inception-ResNet，在 ImageNet 上预训练。

Inception-ResNet 原本用于图像识别，使用了多个 valid conv/pooling，导致特征不对齐，在密集预测任务中存在特征不对齐的问题，使用对齐版本的网络 (参见论文 《Aligned-inception-resnet model》)，记为 “Aligned-Inception-ResNet”。

对于这两个网络的改造：

1. 将网络最后的 全局均值池化和 1000-way fc 层移除，然后附加 `1x1` conv 将 channel 降为 `1024`。

2. 将最后一个 block（conv5）中第一个 layer 的 stride 从 `2` 降为 `1`，从而使得整个网络的 stride 由 `32` 变为 `16`，以便增加特征平面分辨率。为了补偿这种变化，将 `conv5` 中的所有的 kernel size > 1 的卷积层的 dilation 从 `1` 变为 `2`。

可变形卷积用在最后的 $n$ 个 kernel size > 1 的 conv layer 上，作者实验了几个不同的 $n$ 值，发现 $n=3$ 在性能和复杂之间取得较好的平衡。如图 5，

![](/images/obj_det/deformable_conv5.png)

<center>图 5. 不同可变形 conv 数量的结果。数据集为 VOC2007 test</center>

__分割和检测网络__

在特征抽取网络之上是一个任务相关的网络。记 $C$ 为目标类别数量。

DeepLab 是一个语义分割的 SOTA 方法，在特征平面上使用 `1x1` conv 生成 `(C+1)` 个 maps，表示每个像素的分类得分，然后使用 softmax 输出每个像素的分类概率。

Category-Aware RPN 与 Faster R-CNN 的 RPN 几乎相同，只是将 二分类（bg/fg）改为 `(C+1)` 分类，即，由 RPN 预测最终的 anchor 的坐标和分类，属于 one-stage 目标检测，类似于一个简化版的 SSD（单 level 的 feature maps 进行预测）

Faster R-CNN，使用 RPN 得到 RoI，然后使用 RoI pooling + 2 fc layers，得到 `1024` 维度特征，然后是分类和坐标回归两个分支。这里的 RoI pooling 可被替换为 deformable RoI pooling。

R-FCN，由于 per-RoI 的计算量较小，可以将 RoI pooling 改为 可变形 PS RoI pooling。


# 附录 

## 反向传播

梯度反向传播到输出特征平面 $\mathbf y$， 得到 $\frac {\partial L}{\partial \mathbf y}$，其中 $L$ 是目标函数，

__可变形卷积__

根据 (2) 式，

$$\begin{aligned}\frac {\partial \mathbf y(\mathbf p_0)}{\partial \Delta \mathbf p_n}&=\sum_{\mathbf p_n \in \mathcal R} \mathbf w(\mathbf p_n) \cdot \frac {\partial \mathbf x(\mathbf p_0+\mathbf p_n+\Delta \mathbf p_n)}{\partial \Delta \mathbf p_n}
\\ &=\sum_{\mathbf p_n \in \mathcal R} \left[\mathbf w(\mathbf p_n) \cdot \sum_{\mathbf q} \frac {\partial G(\mathbf q, \mathbf p_0+\mathbf p_n+\Delta \mathbf p_n)}{\partial \Delta \mathbf p_n} \mathbf x(\mathbf q)\right]
\end{aligned} \tag{8}$$

根据 (4) 式计算 $\frac {\partial G(\mathbf q, \mathbf p_0+\mathbf p_n+\Delta \mathbf p_n)}{\partial \Delta \mathbf p_n}$，注意 $\partial \Delta \mathbf p_n$ 是 2D vector，

$$\frac {\partial G(\mathbf q, \mathbf p_0+\mathbf p_n+\Delta \mathbf p_n)}{\partial \Delta p_n^x}=\begin{cases} g(q^y,p_0^y+p_n^y+\Delta p_n^y) & p^x \le q^x <1+p^x \\ -g(q^y,p_0^y+p_n^y+\Delta p_n^y) & q^x < p^x <1+q^x \\ 0 & \text{otherwise} \end{cases} \tag{9}$$

其中 $p^x=p_0^x+p_n^x+\Delta p_n^x$。注意 $p^x=q^x$ 时使用的是次梯度，不过这一情况实际几乎不会出现。

__可变形 RoI pooling__

根据 (6) 式，

$$\begin{aligned}\frac {\partial \mathbf y(i,j)}{\partial \Delta \mathbf p_{ij}}&=\frac 1 {n_{ij}}\sum_{\mathbf p_n \in bin(i,j)} \frac {\partial \mathbf x(\mathbf p_0+ \mathbf p_n + \Delta \mathbf p_{ij})} {\partial \Delta \mathbf p_{ij}}
\\&= \frac 1 {n_{ij}} \sum_{\mathbf p_n \in bin(i,j)} \left[\sum_{\mathbf q} \frac {\partial G(\mathbf q, \mathbf p_0+\mathbf p_n+\Delta \mathbf p_n)}{\partial \Delta \mathbf p_{ij}} \mathbf x(\mathbf q)\right]
\end{aligned} \tag{10}$$

其中 $\frac {\partial G(\mathbf q, \mathbf p_0+\mathbf p_n+\Delta \mathbf p_n)}{\partial \Delta \mathbf p_{ij}}$ 的计算与 (8) 式类似。

然后计算对归一化 offset 的梯度，

$$\frac {\partial \Delta \mathbf p_{ij}}{\partial \Delta \hat {\mathbf p}_{ij}}=\begin{bmatrix}\gamma w & 0 \\ 0 & \gamma h\end{bmatrix}$$

