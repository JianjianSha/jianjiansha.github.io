---
title: One-stage Object Detection
date: 2021-02-23 10:36:44
tags: object detection
p: obj_det/one_stage
mathjax: true
---

# YOLOv1

1. one-stage detector

2. unified detection：image 经过网络得到 `SxS` 大小的 feature map，相当于把 image 划分为 `SxS` 的 grid，如果其中目标中心落于某个 grid cell，那么这个 grid cell 负责预测这个目标，网络最终的输出为 `S*S*(C+B*5)`，其中 C 表示分类数量，最多 B 个目标中心落于同一个 grid cell，每个 box 有 4 个坐标和 1 个 conf，这个 conf 表示预测 box 包含目标的置信度，也可以认为是预测 box 与 gt box 的 IOU。C 个预测值表示在此处有目标时的条件概率值，`Pr(Classi|Object)`，这里分类条件概率与 box 数量 `B` 无关。测试阶段，分类类型相关的 conf 值则为
    $$Pr(Class_i|Object) * P(conf)$$
3. 没有 SSD 中的 default box，也没有 Faster R-CNN 中的 anchor/proposal，YOLO 直接在 feature map 上每个点预测 box 坐标和分类概率，所以还需要一个 conf，表示预测 box 包含目标的置信度

4. 文中input image size 为 `448x448`，经过6次下采样，得到`7x7`的feature，通过一个 fully connection（输出unit数量 `1470=7*7*30`），得到feature 上所有 box 的预测坐标、conf 以及分类得分

5. 在 feature map 上使用 fully connection生成每个 grid cell 的预测数据，其中 (x,y) 表示预测目标中心坐标，这是归一化的，且表示<b>距所在 cell 的左端和上端的距离</b>。

# YOLOv2

YOLOv1 虽然是 fast 的，但是比起 SOTA 检测系统，缺点在于定位错误较明显，相较于 region proposal-based 的检测方法，YOLOv1 的 recall 低。YOLOv2 中对其进行改善，使用了：
1. Batch Normalization。

2. High Resolution
    \
    YOLOv1 中baseline 分类预训练时 image size 为 `224x224`，然后迁移到检测数据集上训练时，image size 为 `448x448`，这种输入大小的突变对目标检测不友好，所以训练过程中一直调整输入大小，使得网络适应以增加稳定性，具体策略为：每隔 10 个训练 batch，调整输入大小为
    ```
    int dim = (rand() % 10 + 10) * 32
    ```
    保证输入大小是 32 的整数倍，YOLOv2 中有 5 次下采样，这是匹配的。YOLOv1 有 6 次下采样，这里去掉一次下采样，为了使feature 具有 higher resolution，而这又是为了配合下面的 anchor box，在feature 上每个 position 使用一组 anchor box 来预测，可以降低定位误差。
3. Convolution with Anchor Boxes
    \
    参考 Faster R-CNN 中的 RPN，使用 anchor box。feature 上每个 position 使用大小形状不同的 k 个 anchor 进行预测，k 的取值以及各 anchor 的大小形状，根据数据集中gt box 聚类（k-means）计算得到，聚类使用的距离采用 IOU，
    ```
    d(anchor, cluster-center)=1-IOU(anchor, cluster-center)
    ```
    
4. Direct location prediction
    \
    基于 region proposal 的位置预测过程为：记位置预测值 $(t_x, t_y)$ 为相对offset，根据anchor box坐标，计算最终预测box 中心坐标为
    $$x=t_x \cdot w_a+x_a, \quad y=t_y \cdot h_a + y_a$$
    在训练开始阶段，由于是随机初始化模型参数，上式会导致预测 box 位置与 anchor 偏差很大，这会使得训练要花很长一段时间才能使得 box 的预测位置稳定下来（<b>注意：不使用上式</b>）。YOLOv2 沿用 YOLOv1 中预测中心坐标与所在 cell 的左边线和上边线的距离，这样偏差就不会很大，设cell `(i,j)` 处的某个 anchor 对应的坐标预测值记为 $t_x, t_y, t_w, t_h$，feature 大小为 $(w_f, h_f)$，anchor 基于 feature 的宽高为 $(w_a,h_a)$，那么计算预测 box 的实际归一化坐标为
    $$x=(i+t_x)/w_f$$
    $$y=(j+t_y)/h_f$$
    $$w=\exp(t_w) \cdot w_a/w_f$$
    $$h=\exp(t_h) \cdot h_a/h_f$$

每个 box 均有 5 个坐标预测值（包括 4 个坐标偏差和 1 个是否包含目标的 conf）和 C 个分类得分预测值，最终输出大小则为 $k \cdot s \cdot s \cdot (5+C)$，其中 k 为 anchor 数量。

# YOLOv3
主要是借鉴别的好的 idea 整合到 YOLO 里面来。

1. 沿用 YOLOv2 中 anchor box，使用聚类得到 k 个 anchor，每个 anchor 预测 4 个坐标 offset，1 个 objectness conf，以及 C 个分类概率。坐标 offset 的计算与 YOLOv2 中相同

2. 与 gt box 有最大 IOU 的 anchor 的 conf target 值为 1，而其他非最佳 IOU 但是 IOU 大于某个阈值（0.5）的 anchor 则被忽略。IOU 低于 0.5 的则为负例 anchor，负例 anchor 只需要计算 conf 损失，不需要计算坐标 offset 损失和分类损失。

3. YOLOv3 在三个 scale 的 feature 上进行预测，YOLOv1 和 YOLOv2 均只有单个 scale 的 feature。这是为了借鉴 FPN 的思想。由于有了 multi-scale 的 features，每个 feature 上的每个 position 处只预测 3 个 anchor boxes，假设某个 feature size 为 `NxN`，那么预测 tensor 为 `NxNx[3*(4+1+C)]`，其中 C 为 foreground 分类数量。

4. Baseline 结构如 darknet-53 所示，用于抽取 feature，在 ImageNet 上预训练。目标检测网络结构如 yolov3.cfg 配置文件中所示。借鉴了 ResNet 中 shortcut 技巧（主要也是因为网络更 deep 了）。聚类得到 9 个 anchor size，然后按大小排序，每 3 个一组作为对应 scale feature 上所用的 anchor。

# SSD
1. baseline: VGG 等
2. one-stage detector。与 Faster R-CNN 相比，省去了 proposals 生成过程，而是在 feature map 上每个 position 有一组 prior box（k 个），然后 feature maps 上使用具有 `(c+4)k` filters 的 conv，而非 fully connection，进行预测输出，每个position 输出 `(c+4)k` 个值，表示预 c 个分类得分，和此处 box 的坐标 offsets。同时，使用 multi scale 的 feature maps，以覆盖多个不同大小级别的目标预测。
3. 论文中针对 300x300 的输入图像，一共使用了 6 个不同 scale 的 feature，每个 feature 上的各点生成 prior box 的数量为 `4,6,6,6,4,4`，因为认为数据集中，中间 scale 的目标数量要多一些。各 feature 的边长为 `38, 19, 10, 5, 3, 1`，单个 image 上所有 prior box 数量为 `(38*38+3*3+1*1)*4+(19*19+10*10+5*5)+6=8732`
4. 由于使用 multi scale feature maps，所以不同 level 的 feature 负责不同大小的目标检测，假设共 m 个不同 scale 的 feature（文中 m=6），那么每个 level 的 feature 上的 default box 的基础边长为
    $$s_k=s_{min}+\frac {s_{max}-s_{min}}{m-1}(k-1), \ k \in [1,m]$$
    其中最小最大边长为 `[min, max]=[0.2, 0.9]`，所有不同 scale 的边长 s 均匀散落在这个区间上
5. 一个image上的 default box 数量非常多（第3点中指出高达 8732个），其中匹配的 default box是指与 gt 有最大 IOU 或者 IOU > 0.5 的那些，称为正例，其余的为负例，显然负例会特别多，导致数据 unbalanced，所以将负例按 conf 预测损失倒序排列，选择 top N 的负例，这里 N 取正例数量的 3 倍，每个 level 的 feature 独立进行这种 hard negative mining

# DSSD

在 SSD 的基础上增加 deconvolution layer，具体是对 SSD 中用于预测所有 level 的 feautre，自顶向下，最顶 level 的 feature 上使用一个 prediction module 进行预测，然后这个 feature 经过 deconvolution，再与 SSD 中 resolution 更大一级的 feature 进行融合，然后使用 prediction module 进行预测，上一个融合后的 feature 再经 deconvolution，与 SSD 中 resolution 更大一级的 feature 进行融合，递归进行这个过程，直到原 SSD 中所有 level 的 feature 均进行了融合和预测，整个网络形成一个 hour-glass 结构，也就是 “encoder-decoder”。这与 FPN 其实很类似，只是这个 top-down 模块中的 upsample 换成了 deconvolution。


# RetinaNet(Focal Loss)

## Loss
one-stage 速度更快，结构更简单，但是比起 two-stage，准确率还差的不少，其中一个原因是 one-stage 使用了密集 location 采样，这就导致 fg-bg 分类不均衡，本文使用 Focal Loss，通过附加低权重，降低已经分类好的样本的对 loss 的贡献，从而 focus on hard examples。

> 他方法如 OHEM 等也可以解决 one-stage 中的分类不平衡问题

### Balanced Cross Entropy
$$CE(p,y)=\begin{cases} - \alpha \log p & y=1 \\ -(1-\alpha) \log(1-p) & y=0\end{cases}$$
其中 $\alpha \in [0, 1]$，其值可取类频数的倒数，例如数据集大小 N，fg 数量为 $N_1$，bg 数量为 $N_0$（$N=N_1+N_0$），那么
$\alpha=\frac {N_0} N$，表示增大正例的损失贡献。

### Focal Loss
$$FL(p,y)=\begin{cases} - （1-p)^{\gamma} \log p & y=1 \\ -p^{\gamma} \log(1-p) & y=0\end{cases}$$
其中 $\gamma>0$。

记
$$p_t=\begin{cases} p & y=1 \\ 1-p & y=0\end{cases}$$
$$\alpha_t=\begin{cases} \alpha & y=1 \\ 1-\alpha & y=0\end{cases}$$

于是 $\alpha$ balanced CE 损失为
$$CE(p_t)=-\alpha_t \log(p_t)$$

Base Focal Loss 为
$$FL(p_t)=-(1-p_t)^{\gamma} \log (p_t)$$

$\alpha$ balanced Focal Loss 为
$$FL(p_t)=-\alpha_t (1-p_t)^{\gamma} \log (p_t)$$

## RetinaNet
为了验证 Focal Loss 的有效性，设计了这个 RetinaNet。Focal Loss 用在 Classification Subnet 中。

__backbone:__ FPN on ResNet。使用 $P_3 \sim P_7$ level 的 feature，其中 $P_3 \sim P_5$ 由 ResNet 的 $C_3 \sim C_5$ 获得，然后再使用一个 $3 \times 3$-s2 的 conv（无 ReLU） 得到 $P_6$，最后使用 ReLU + $3 \times 3$-s2 conv 得到 $P_7$。$P_l$ feature 的 stride 是 $2^l$，每个 feature 均为 C=256 channels。feature 上 anchor 的 base size 为 $2^{l+2}$，每个 position 有 9 个 anchors，aspect ratio 由配置给出，每个 anchor 均有 K 个分类得分（包含了背景），4 个位置坐标。

$IOU \ge 0.5$ 的为 正 anchor，$IOU < 0.4$ 的为负 anchor，$0.4 \le IOU < 0.5$ 的 anchor 忽略，不参加训练。正 anchor 与 对应的 gt box 之间计算 offset，作为 box regression target，classification target 则为 one-hot vector，向量中 anchor 所对应的目标分类的 entry 为 1， 其余 entry 为 0。

backbone 后接两个 subnetworks：用于分类和 box 回归（每个 level 的 feature 上均如此）。

__Classification Subnet:__ 这是一个 FCN 子网络，参数在所有 pyramid level 之间共享。在 pyramid feature 上，使用 4 个 `3x3` conv，每个 conv 均有 C=256 个 filters，且每个 conv 后跟一个 ReLU，然后是一个 `3x3` 的 conv，有 `KA` 个 filters，其中 K 为分类数量，A 为 anchor 数量。这个子网络比 RPN 有更 deep 的结构，文中发现，这种设计比某些超参数的选择还要重要。

__Box Regression Subnet:__ 与 Classification Subnet 结构类似，只是最后一个 conv 的 filters 数量为 `4A`。

这两个 subnet 的结构就像天线一样位于 FPN 之上，故称 RetinaNet。

以前使用 heuristic sampling（RPN）或 hard example mining(OHEM, SSD) 来选择 mini-batch（数量为 256）的 anchors，但是这里使用 Focal Loss，单个 image 上的 anchor 数量达到 ~100k（正例 anchor 与 负例 anchor 之和），总的 focal loss 则是这所有 anchor 上 Focal Loss 之和，并除以正例 anchor 数量。

# STDN
Scale-Transferrable Detection Network，为了解决目标 scale 多样性的问题。

主流的目标检测方法中， Faster RCNN 中只有单一 scale 的 feature，其 receptive field 是固定的，而目标的 scale 和 aspect ratio 则是各不相同的，所以存在不一致问题。 SSD 在不同 depth 的 layer 的 feature 上预测，anchor 的 scale 与 feature 的 scale 有关，这一定程度上解决了目标 scale 多样性的问题，但是在小目标上表现并不好，因为 low feature 用于预测小目标，而 low feature 的语义性较弱，于是使用 FPN，通过径向连接和 top-down 模块，将高层 feature 与低层 feature 融合，使得低层特征在保持更多细节信息的同时，兼具语义特征，FPN 缺点在于需要谨慎地构建 feature pyramids，并且 FPN 网络结构带来了一定的计算负担（FPN 是在 Faster RCNN 基础上将 baseline 增加 FPN 结构，所以是一个 two-stage 检测器）。

STDN 以 DenseNet 为 baseline，利用了 DenseNet 中高低层 feature concatenation 的特性，使得 feature 具有更强的表征能力。在 DenseNet 最后一个 DenseBlock 的最后一个 Layer 之上， 使用Scale-Transfer Module（STM），获得 multi scale features，用于预测，STM 没有参数，不会引入很多计算负担。

1. 使用 DenseNet-169 为 baseline (growth rate=32)
2. 将 stem block 改为 3 个 `3x3` 的 conv 和一个 `2x2` 的 mean-pooling，其中第一个 `3x3` conv 的 stride=2。原来 DenseNet 中采用 `7x7-s2` 和 `3x3-s2` 的 conv，我们认为大卷积核和连续的下采样对检测小目标的准确性不利。
3. 当 input size 为 `300x300`，DenseNet 的 输出 feature size 为 `9x9`。
4. 网络结构为 stem --> DB1 --> T1 --> DB2 --> T2 --> DB3 --> T3 --> DB4 => STM，其中 DB 表示 DenseBlock，T 表示 Transition Layer。T3 输出为 `640x9x9`，STM 包含 6 个 scale 的 features，如下表所示
    | output size | layer |
    | -- | -- |
    |800x1x1 | 9x9 mean-pool, stride 9 (Input DB4_concat5)|
    |960x3x3 | 3x3 mean-pool, stride 3 (Input DB4_concat10)|
    |1120x5x5| 2x2 mean-pool, stride 2 (Input DB4_concat15)|
    |1280x9x9| Identity layer (Input DB4_concat20) |
    |360x18x18| 2x scale-transfer layer (Input DB4_concat25)|
    |104x36x36| 4x scale-transfer layer (Input DB4_concat32)|

    已知 DenseBlock 中第 $l$ 个 layer 的 output channen 为 $k_0+l*32$，那么上表中第一个 layer 为 `9x9` 的均值池化层，输出为最小 scale 的 feature，输出 size 为 `800x1x1`，这个 layer 的输入为 DB4 中第 5 个 layer 的 output，根据公式其输出 channel 为 $640+5\times 32=800$。其他 layer 的输入也是 DB4 中某个 layer 的输出。

    - Identity layer 表示输出就是输入本身
    - scale-transfer layer 表示将输入的 channel 压缩 $r^2$ 倍（$r \times$ scale-transfer layer），而 $W, H$ 则均增大 $r$ 倍，rearrange 公式为，
        $$I_{x,y,c}^{SR}=I_{\lfloor x/r \rfloor,\lfloor y/r \rfloor, r\cdot mod(y,r)+mod(x,r)+c\cdot r^2}^{LR}$$

5. 每个 scale 的 feature 分别根据 dense anchor 进行预测，anchor 与 gt box 匹配标准为：有最大 IOU 或者 IOU > 0.5，其余 anchor 为负例，根据 hard negative mining 使得正负例数量比为 `1:3`。

6. 抽取的 feature 分两路，分别到分类分支和 box 回归分支。分类分支由一个 `1x1` conv 和两个 `3x3` conv 组成，每个 conv 后接 BN+ReLU，最后一个 conv 的 channel 为 `KA`，其中 K 为分类数量（fg 数量 + 一个 bg），A 为每个 position 预测的 anchor 数量。回归分支的结构与分类分支相同，只是最后一个 conv 的 channel 为 `4A`。