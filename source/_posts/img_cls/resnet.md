---
title: resnet
p: img_cls/resnet
date: 2021-01-19 17:25:25
tags: image classification
mathjax: true
---
本篇罗列一些 Resnet 网络的细节。
<!-- more -->

# ImageNet
## 数据增强
1. scale 增强：输入图像的短边（w,h 较小者）被 resize 到 `[256,418]` 中的一个随机数。
2. 在 resized 的图像或其水平翻转图像上随机位置（随机 center） crop 出一个 `224x224` 大小的部分。
3. 10-crop testing：测试阶段，对每个测试图像，crop 出上下左右四个角以及center处共 5 个 patch，以及水平翻转后同样的 5 个 patch，这 10 个 patch 经过网络之后的预测结果（softmax 层的输出）再进行平均，即得到这个图像的最终预测，预测为长度等于类别数量的向量。

## 网络结构
1. baseline 使用 VGG，记为 plain，每个 conv 层与 activation 层之间均插入一个 BatchNorm 层。
2. 网络结构如下图，其中 conv3_1,  conv4_1, conv5_1 进行了下采样（stride=2）
![](/images/img_cls/resnet_1.png)<center>图 1. 网络结构</center>
3. 每个 conv block 均额外增加一个 shortcut，两个 output 在通道维度上进行 concatenate。由于 conv3_1,  conv4_1, conv5_1 进行了下采样，output 的 (H,W) 分别减小到一半，而通道 (C) 增大一倍，故需要将对应的 shortcut 作调整，有两种方法：a. identity mapping（注意 stride=2，以降低 H 和 W），额外维度上 0 填充；b. `1x1` conv，stride=2 的下采样，且输出 channel 增大一倍。
4. 在 ResNet-18 和 ResNet-34 上，使用 identity mapping（以及 zero-padding 维度填充）。
5. conv3_1,  conv4_1, conv5_1 进行了下采样，具体而言，对 ResNet-18 和 ResNet-34，conv block 是由两个 `3x3` conv layer 组成，所以是 conv3_1,  conv4_1, conv5_1 这三个 conv block 中的第一个 conv layer 执行了下采样；对于 ResNet-50，ResNet-101 以及 ResNet-152，conv block 是由三个 conv layer 组成的（也叫 bottleneck），这三个conv 按顺序分别是 `1x1`, `3x3`, `1x1`， 所以其实是 conv3_1,  conv4_1, conv5_1 这三个 conv block 中的 `3x3` 执行了下采样操作。
6. 整个网络的下采样率为 `2x2x2x2x2=32`，输入为 `224x224`，那么最后的 conv 输出 feature 大小为 `7x7`。对每个 feature map 进行全局均值池化 GAP，那么每个 channel 得到一个值，然后经 full connection 层，得到 1000 个分类得分，最后由 softmax 将其归一化。
7. ResNet-x，后缀 `x` 表示有参数的 layer 数量。
8. shortcut 与 residual function 按 channel 维度连接之后再经过 relu 层。
9. 如果 shortcut 需要下采样，那么令 stride=2 即可。
## 实验说明
### plain vs. resnet
在 18 layers 和 34 layers 两个网络上，对比 plain 和 ResNet，top-1 错误率如下表

| | plain | ResNet|
|:--------:       | :------:    |   :-------:          |
|18 layers| 27.94 | 27.88|
| 34 layers| 28.54| <b> 25.03 </b>|
 <center>表 1. plain 与 Resnet 结果比较。ResNet 采用 identity shortcut。ImageNet 验证集上的错误率（10-crop testing）</center>
说明：

- plain，随着网络加深，准确率下降。
- ResNet 采用 identity mapping （zero padding 增加channel）作为 shortcut。ResNet-34 比 ResNet-18 好，这表明加深网络导致恶化的问题可通过 ResNet 得到解决，且能获得更高的准确率。
- ResNet-18 比 plain-18 收敛更快。

### Projection Shortcut
使用 `1x1` conv 代替 identity mapping，输出如下，其中第一项为 conv block 的输出，第二项为 shortcut 的输出，

$$\mathbf y = \mathcal F(\mathbf x, \{W_i\}) + W_s \mathbf x $$
以 34 layers 网络为例，对比如下三种情况：

* A：所有 shortcut 均没有参数，即 identity mapping，以及 zero-padding 以增加维度。与上述 ResNet-34 一样。
* B：除了需要增加维度的时候（conv3_1, conv4_1, conv5_1）使用 projection shortcut，其他 shortcut 采用 identity mapping。此时无需 zero-padding。
* C：所有的 shortcuts 均采用 projection shortcut。此时无需 zero-padding。

实验结果如下：
|model|top-1 err | top-5 err|
|:--|:--|:--|
|plain-34| 28.54| 10.02|
|ResNet-34 A|25.03|7.76|
|ResNet-34 B|24.52|7.46|
|ResNet-34 C|24.19|7.4|
|ResNet-50|22.85|6.71|
|ResNet-101|21.75|6.05|
|ResNet-50|<b>21.43</b>|<b>5.71</b>|
<center>表 2. projection shortcut 方案比较，以及 ResNet 不同深度的比较。<font color='clan'>ResNet-50/101/152 使用方案 B</font>。ImageNet 验证集上的错误率（10-crop testing）</center>

结果分析：
- 错误率：三种情况均比 plain 有所降低。B 比 A 有所降低，C 比 B 轻微降低。由于 A 中没有 projection shortcut，这表明对于解决准确率恶化的问题，projection shortcut 不是本质性地重要；另一方面，C 中由于引入过多的参数，内存占用和计算时间均增大，故综合起来，使用 B 方案。

### Bottleneck 结构
图 1 中，每个 conv block 是由两层 conv 组成，把它改成 bottleneck 结构，即三层 conv：`1x1`, `3x3`, `1x1`，第一个 `1x1` 用于降低 channel（维度）（降为 1/4），(H,W) 不变，`3x3` 执行了下采样，(H,W) 均降为一半，channel 不变，最后一个 `1x1` 用于增加 channel，增大到 4 倍，这三个 conv layer 形成一个 bottleneck 结构，如下图右侧，
![](/images/img_cls/resnet_2.png)<center>图 2. ImageNet 上使用的深度残差函数。左：与图 1 中相同的conv 块结构；右：bottleneck 结构，用在 ResNet-50/101/152 中</center>

使用 Bottleneck 时，identity shortcut 显得尤其重要，如果使用 projection，model 大小和时间复杂度都会大大增加，因为此时 shortcut 连接的两端都是高维（即，输入输出的 channel 都增大到原来 4 倍），这使得 shortcut 上的参数量大大增加。

### ResNet-50/101/152
将 ResNet-34 中的所有 conv block 替换为 bottleneck 结构，就得到 ResNet-50，如图 1，ResNet-34 有 `3+4+6+3=16` 个 conv block，增加的 layer 数量也就是 16。ResNet-50 使用方案 B。

在 ResNet-50 基础上增加更多的 bottleneck 块，得到 ResNet-101 和 ResNet152。

ResNet-50/101/152 比 ResNet-34 改善较多。实验结果如表 2，可见，使用 ResNet 结构可享受深度带来的益处。

除了单模型，作者还对比了各种模型的集成结果，这个集成结果我也不知道是怎么计算的，也许只是几种不同深度的 ResNet 网络的结果平均，这个<font color="red">有待考证</font>。

再在其他数据集上研究 ResNet 的表现。

## 训练

1. 使用 SGD 方法学习
2. mini-batch=256
3. 学习率初始为 `0.1`，当错误率进入平稳期时，以  `1/10` 的倍率降低学习率。
4. 权重衰减因子为 `0.0001`，momentum=0.9
5. 由于使用了 BatchNorm，所以不使用 Dropout。

# CIFAR-10
作者这里的目的是为了研究非常深的网络，而非为了 SOTA 的性能，所以仅使用了简单的框架，以 ResNet-34 和对应的 plain 网络为主体框架进行改造，这是因为输入 size 为 `32x32`，这比 ImageNet 数据集的输入 size `224x224` 小很多。
## 网络结构

第一个 layer 的 kernel 需要调小（ImageNet 数据集上使用的是 `7x7`，太大），使用的是 `3x3` 的 conv，stride=1，无下采样（输出 channel 增大到 16），接着，使用 6n（6 的整数倍）个 conv layer，每 2n 个 conv layer 划分为一组，共 3 组，每组的输出 feature size 分别为 `(32, 16, 8)`，channel （也就是卷积 filter 的数量）分别为 `(16, 32, 64)`。最后使用全局均值池化 GAP + full connection + softmax。有参数的 layer 一共 `6n+2` 个。
> 第二组和第三组各做了一次 stride=2 的下采样。论文里面没有明说，但我认为可以使用第二、三组中各自第一个 conv layer 来做下采样，与图 1 中一致。

网络结构说明如下：

|输出 map size|32x32|16x16|8x8|
|:---|:---|:---|:---|
|#layers|2n+1|2n|2n|
|#filters|16|32|64|


从这个表格可见，如果采用 identity shortcut，如果不进行下采样，那么这个 identity shortcut layer 的输出与输入完全一样，如果进行 rate=2 的下采样，那么 (H,W) 各变为一半大小，而 channel 增加一倍，变为原来 channel 的两倍，residual function 分支的 channel 也是原来输入 channel 的两倍，两个分支的输出 channel 和 (H,W) 均分别保持相同，才能进行 element-wise 的 add 操作。

6n 个 conv layer，每两个 conv layer 使用一个 shortcut，共 3n 个 shortcut，作者使用 identity shortcut。

## 训练说明

- 与 ImageNet 上的训练类似，区别是 mini-batch=128，学习率初始为 `0.1`，在第 32k 次和第 48k 次迭代时，学习率降为 `1/10`，训练共迭代 64k 次。
- CIFAR-10 训练集大小为 50k，测试集大小为 10k，类别数量为 10。将训练集按 45k/5k 分割为 train/val 集。
- 原图像采用 4 pixel 的填充，然后以 0.5 的概率水平翻转，再随机 crop 一个 32x32 的 patch，作为网络输入。

## 实验结果
当 `n={3,5,7,9}` 时，分别得到 `20, 32, 44, 56` 个有参数的 layer 的网络。实验结果如图 3，
![](/images/img_cls/resnet_3.png)<center>图 3. CIFAR-10 训练结果。细线表示训练错误率，粗线表示测试错误率。左图中 plain-110 的错误率大于 60%，没有在图中显示。</center>

从图 3 中可见，ResNet 可以解决网络深度增加带来的准确率恶化问题。n=18 得到 ResNet-110，此时初始学习率 0.1 过大，导致开始训练时一直难以收敛，故初始化学习率为 0.01，当训练错误率降低到 80% 以下时（大约 400 次迭代），将学习率重置为 0.1 以加快收敛速度继续训练，之后就与训练 ImageNet 一样（阶梯降低学习率）。

训练结果：

ResNet-110 表现最好，比其他 deep & thin 的网络有更少的参数和更好的性能，对比如下表，
||#layers|#params(M)|error(%)|
|:---|:---|:---|:---|
|FitNet|19|2.5|8.39|
|Highway|19|2.3|7.54(7.72 ± 0.16)|
|Highway|32|1.25|8.8|
|ResNet|20|0.27|8.75|
|ResNet|32|0.46|7.51|
|ResNet|44|0.66|7.17|
|ResNet|56|0.85|6.97|
|ResNet|110|1.7|<b>6.43</b>(6.61±0.16)|
|ResNet|1202|19.4|7.93|
<center>CIFAR-10 测试集上的分类错误率。其中，与 Highway 中类似，对 ResNet-110 试验了 5 次，得到最佳结果 6.43% 的错误率，均值为 6.61%，标准差为0.16</center>

## Layer 响应分析
图 4 是 CIFAR-10 上网络（训练好之后）中各个 `3x3` layer 响应的标准差，响应指 BN 之后，activate 之前的 layer 响应。
![](/images/img_cls/resnet_4.png)<center>图 4. 网络（3x3）各层的响应标准差。上：网络原始各层先后顺序；下：按标准差倒序排列。</center>

从图 4中可见，ResNet 有着比 plain 更小的响应标准差，所以 ResNet 中各层的输出更加集中在 0 附近，避免了梯度消失（现在都使用 relu 而非 sigmoid 激活，避免梯度消失这一说还有意义吗？）。同时注意到，更深的 ResNet 的响应幅度更小，这表明当 layer 数量越多时，均摊到单个 layer 上，其对信号的改变越小。

## 超深网络 layers>1000
n=200，此时网络层数 6n+2=1202。训练结果（错误率）如上表 和图 3 中所示，训练集错误率 `<0.1%`，测试集错误率 `7.93%`，差强人意，但是比起 ResNet-110，虽然训练错误率差不多，但是测试错误率已经上升，作者认为这是过拟合导致，此时网络太大，而数据集太小，可以使用 `maxout`，`dropout` 等手段来改善过拟合问题，但是作者自己并没有这么做，而只是在设计网络架构时，遵循 deep & thin 的原则，简单地将网络正则化，这是因为本论文的重点在于解决（deep 网络的）优化困难，而非网络正则
> deep: 网络层数多；thin: 每层的操作少，例如常见的一层为 conv-bn-relu；wide: 每层的 feature size 较大。


# 目标检测
检测采用 Faster R-CNN，将 baseline 从 VGG-16 替换为 ResNet-101。

## 训练方法
1. 在 ImageNet 上训练，然后再在目标检测数据上对网络进行 fine-tune。