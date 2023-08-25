---
title: mask-rcnn
date: 2019-07-08 17:39:57
tags: object detection
mathjax: true
---
论文 [Mask R-CNN](https://arxiv.org/abs/1703.06870)
<!-- more -->
# Introduction
这是一篇实例分割的文章。实例分割结合了目标检测和语义分割，这看似是需要一个复杂的模型才能完成的任务，实际上本文提出的 Mask R-CNN 出奇的简单灵活且高效。  

Mask R-CNN 是对 Faster R-CNN 的扩展，增加了一个分支用于预测每个 RoI 的分割掩模（segmentation masks），如图 1，这个分支与原先的分类和回归分支并列。mask 分支在 RoI 上以 pixel-to-pixel 方式预测得到一个 segmentation mask，有语义分割背景的话，不难想象 mask 分支应是一个全卷积网络 FCN。如何构建这个 mask 分支则至关重要。  
![](/images/mask-rcnn_fig1.png)

Faster R-CNN 的网络输入和输出之间不是点与点对齐，这是由于 RoIPool 层在抽取特征时使用了离散化空间坐标（坐标值必须为整数，RoI 坐标从输入 image 平面映射到 feature map 时，坐标变为原来的 1/16，并四舍五入取整），而 mask 分支是 pixel-wise 的，所以必须要解决这个不对齐问题，为此我们提出了 RoIAlign 层以保持准确的空间位置，这个改动虽小，但效果却十分明显：提高了大约 10%~50% 的 mask 准确度。  

另外，有必要将分类预测和 binary mask 预测解耦，每个分类独立进行 binary mask 预测，并根据 RoI 分类分支来确定目标分类。相反在语义分割 FCN 方法中，每个像素位置均进行多分类，这就耦合了分割和分类，如果用在实例分割任务中则表现较差。

# Mask R-CNN
Faster R-CNN 中每个候选区域均对应两个输出：分类标签和坐标偏差。Mask R-CNN 则在此基础上增加第三个输出：目标 binary mask，与前两个输出不同的是，此输出需要非常精确的目标空间位置，这是 Mask R-CNN 的关键点之一。

__Faster R-CNN:__ 简单的回顾一下 Faster R-CNN，这是一个 two-stage 目标检测器，其中第一个 stage 为 RPN，用于生成 proposals，第二个 stage 其本质就是 Fast R-CNN，使用 RoIPooling 从每个 proposal 中提取固定长度的特征并进行分类和 bbox 回归。

__Mask R-CNN:__ 在 Faster R-CNN 基础上增加第二个 stage 的输出，即为每个 RoI 生成 binary mask。

训练时，每个 RoI 的损失为 $L=L_{cls}+L_{box}+L_{mask}$，其中分类损失 $L_{cls}$ 和回归损失 $L_{box}$ 均与 Fast/Faster R-CNN 中相同，
$$L_{cls}=L_{cls}(p,u)=-\log p_u \tag{1}$$
上式为 log loss，proposal 对应的 gt 分类为 u，$p_u$ 为 proposal 分类为 u 对应的置信度（分类得分）。
$$L_{loc}=L_{loc}(t ^ u,v)=\sum_{i \in \{x,y,w,h\}} smooth_{L_1}(t_i ^ u,v_i)\tag{2}$$
上式为 smooth L1 loss，$t_u$ 为在分类 u 下的 bbox 的四个偏移值，v 表示 gt box 相对 proposal 的偏移 target。  
__mask 分支会为每个 RoI 生成 $Km ^ 2$ 维输出向量__，然后对这个输出向量应用 pixel-wise sigmoid，表示 K 个 binary mask，每个 mask 分辨率为 $m \times m$（m 值参见下文图 4），这里 K 表示所有分类数量，定义 $L_{mark}$ 为平均二值交叉熵损失，记 RoI 的 gt 分类为 k，$L_{mark}$ 仅由第 k 个 binary mask 计算得到，其他 K-1 个 binary mask 均不参与 $L_{mark}$ 的计算，
$$L_{mark}=-\frac 1{m ^ 2} \sum _ {i=1} ^ {m ^ 2} \sum _ {j=0} ^ 1 [t _ i=j] \cdot \log f(s _ i ^ j)=-\frac 1{m ^ 2} \sum _ {i=1} ^ {m ^ 2} [t _ i \cdot \log f(s _ i) + (1-t _ i) \cdot \log (1-f(s _ i))]\tag{3}$$
其中 $f(\cdot)$ 表示 sigmoid。

__Mask Representation:__ 对单个 RoI 而言，无论其大小，对应的分类和 bbox 偏移这两个输出都是固定长度，可由 fc 层输出得到，而 mask 则以 pixel-to-pixel 方式表征 RoI 中目标的空间布局，所以适合使用卷积。事实上，我们正是使用了全卷积网络 FCN 来为每个 RoI 生成 $m \times m$ 空间尺寸的 mask。然而需要注意的是，pixel-to-pixel 的方式要求 RoI 特征能如实地保留每个像素的空间对应关系，于是我们提出 RoIAlign 来解决这个问题。

__RoIAlign:__ RoIPool 是从 RoI 中抽取固定长度特征（例如 $7\times 7$）的标准方法，首先将浮点数 RoI 量化成整数粒度的 feature map，然后将量化后的 RoI 切分得到一系列空间 bins ，每个空间 bin 的大小也是浮点数，所以每个空间 bin 的位置也需要量化，然后将其中的像素值聚合得到这个空间 bin 的值，一般使用最大值池化进行聚合。

可见前后有两次量化过程，第一次量化是在将 RoI 的坐标 x 从输入 image 平面上映射到特征平面上，在 Faster R-CNN 中，这个特征的 stride 为 16，所以 RoI 在特征平面上的坐标为 $[x/16]$，其中 $[\cdot]$ 表示四舍五入成整数；第二次量化是在计算空间 bin 位置时，假设 RoI 为 $(x_1,y_1,x_2,y_2)$（通常均为浮点数，因为 RPN 中对 anchor 位置进行偏移得到 RoI），一共将 RoI 划分为 7x7 个空间 bins，经过第一次量化后特征平面上 RoI 表示为，
$$x_1'=[x_1/16] \quad y_1'=[y_1/16]
\quad x_2'=[x_2/16]
\quad y_2'=[y_2/16]\tag{4}$$
RoI 的大小 和 空间 bin 的大小分别为
$$w'=x_2'-x_1'+1
\quad h'=y_2'-y_1'+1
\\\\ w ^ b=w'/7 \quad h ^ b=h'/7$$
对于第 (i,j) 个 bin，其位置为
$$x_1 ^ b=\lfloor j \cdot w ^ b\rfloor \quad y_1 ^ b=\lfloor i \cdot h ^ b\rfloor \quad x_2 ^ b=\lceil (j+1) \cdot w ^ b\rceil \quad y_2 ^ b=\lceil (i+1) \cdot h ^ b\rceil\tag{5}$$

其中 $0 \le i<7, \ 0\le j<7$。（当然还需要对 bin 的位置是否越界进行检查，这里略）

坐标 $x _ 1 ^ b, y _ 1 ^ b, x _ 2 ^ b, y _ 2 ^ b$ 是基于 feature map 的，这个 feature 的 size 为输入图像 size 的 $1/16$ 。

两次量化使得 RoI 与抽取到的特征不对齐，这对 pixel-to-pixel 的 mask 而言是非常不利的，所幸 RoIAlign 可以解决这个问题。使用 RoIAlign 代替 RoIPool，避免量化操作，如图 3，
![](/images/mask-rcnn_fig3.png)

<center>图 3. RoIAlign 示意图，图中黑矩形框表示 backbone 输出 feature maps 上的一个 RoI，具有 2x2 个 bin，实际是 7x7 个 bin，这里仅作示例</center>

特征平面上的 RoI 的位置为 $x/16$，其中每个 bin 采样 4 个位置点，采样位置处的值通过双线性插值计算得到，然后每个 bin 的值使用这四个采样位置的值进行聚合得到（max 或者 average 聚合）。整个过程 __没有任何量化操作__。实验的最终结果对采样位置不敏感，对采样位置的数量也不敏感。

__Network Architecture:__ Mask R-CNN 网络组成包括 1. 用于抽取特征的 backbone，2. network head，用于 bbox 分类和回归，以及 mask 预测。

Backbone 网络的命名法：我们使用了 ResNet 和 ResNeXt（深度为 50 或101）。Faster R-CNN 中使用 ResNet 的 4-th stage 的最后一个 conv 的输出作为特征，这里记为 C4。于是，当 ResNet 为 ResNet-50 时，我们称 backbone 为 ResNet-50-C4。

我们也研究了其他的 backbone 例如 FPN，FPN 使用 top-down 结构以及横向连接生成 feature pyramid。使用 ResNet-FPN 作为 backbone 时，Mask R-CNN 的准确率以及响应速度均有提升。

对于 Network head，如图 4，
![](/images/mask-rcnn_fig4.png)

<center>图 4. 检测 heads </center>

ResNet-C4 作为 backbone 时，后面的 head 结构包含 ResNet 的 5-th stage（即，具有 9 个 conv 的 res5）。ResNet-FPN 作为 backbone 时，由于 backbone 已经包含了 res5，故后面的 head 结构较为简单高效。  
图 4 左边部分，res5 表示 ResNet 的 5-th stage，为简单起见，作用到 $7x7$ 的 RoI feature maps 上的第一个 conv 的 stride 为 1，而原始 ResNet 中对应的这个 conv 由于作用在（conv4_x 输出的）$14x14$ feature maps 上，这个 conv 的 stride 为 2，这一点有所不同。
# Experiments
实验部分略，请阅读原文。

# Appendix
有关 mask 分支，这里详细说明一下处理过程。如图 4，mask 分支输出大小为 $(R,K,m,m)$，根据 bbox 回归得到预测 box 的坐标数据，数据块大小为 $(R,4)$，其中 R 为检测到的所有预测 box 的数量，K 为目标分类数量，$mxm$ 为 mask 的空间大小。对于第 i 个 目标，$0 \le i < R$，记预测 box 位置为 $(x_1,y_1,x_2,y_2)$，对于第 k 个分类，记对应的 mask map 为 $M_i ^ k$，
1. 计算第 i 个 box 的宽高  
    $w=x_2-x_1, \ h=y_2-y_1$
2. 将 mask map resize 到 box 宽高的大小  
    ```python
    mask=cv2.resize(M_i_k, (w,h))
    ```
3. 将 mask map 二值化，因为 mask 是 pixel-wise sigmoid 之后的值，介于 (0,1) 之间，所以需要二值化处理  
    ```python
    mask=np.array(mask>0.5)
    ```
4. 将 binary mask 映射到原始输入 image 平面。记原始输入 image 的宽高为 (W,H)，于是得到分割 mask  
    ```python
    im_mask=np.zero((H,W), dtype=np.uint8)
    im_mask[y1:y2,x1:x2]=mask
    ```

# 实例分割

参考以下源码。

[code](https://github.com/facebookresearch/detectron2)

实例分割训练命令，

```sh
./train_net.py --num-gpus 8 \
  --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml
```

## backbone

backbone 使用 `resnet_fpn`。

以 resnet50 为例，结构如下，

```sh
stem -> res2 -> res3 -> res4 -> res5

# stem: conv7x7+maxpool
# resx: stage x
# stem/resx: stride 均为 2
# stem 和 res2 冻结网络参数。参数使用 ImageNet 上预训练模型参数
# resx 的输出 channel: 256, 512, 1024, 2048
```

resnet 输出特征为 res2, res3, res4, res5 的输出特征（4 个 scale 的特征）

然后构造 FPN 网络，其中 bottom-up 结构就是上面构造的 resnet 结构，FPN 结构示意图为，

```sh
                           +-> mp -> P6
                           |
res5 -> lat5 ---+--> out5 -+-------> P5
 ^              | (nearest，下同)
 |              v
res4 -> lat4 -->O--> out4 ---------> P4
 ^              |
 |              v
res3 -> lat3 -->O--> out3 ---------> P3
 ^              |
 |              v
res2 -> lat2 -->O--> out2 ---------> P2
 ^
 |
stem
 ^
 |                                        (FPN)
```

FPN 输出 multi scale 特征平面，记为 `P2, P3, P4, P5, P6`，top-down 采样 nearest 插值，P6 是经过 `maxpool` 的输出。`O` 表示两个 tensor 执行 elementwise 相加，`+` 表示作为两个 layer 的输入，例如 `lat5` 的输出，作为 `out5` 和 `nearest` 的输入。

经过横向连接 `latx` layer，输出 channel size 全部都是 `256` ，避免了 top-down merge 时的 padding。

## proposal_generator

proposal_generator 使用 `RPN`。

`P2, P3, P4, P5, P6` 5 个 scale 的特征平面，分别对应的 anchor size 基准是 `32, 64, 128, 256, 512`，每个 anchor 的 aspect ratio 为 `0.5, 1, 2`，所以每个 scale 的特征平面上每个 point 处有 `3` 个 anchors。

## rpn_head

rpn_head 使用 `StandardRPNHead` 。

对每个 scale 的特征平面，先使用 `conv1x1` 进行调整，输出 channel 不变，仍是 `256`，然后再分别经过：

1. `conv1x1` 输出 objectness 预测，输出 channel 为 `3`，分别对应 3 个 anchors，预测每个 anchor 处是否有目标
2. `conv1x1` 输出每个 anchor 预测的目标坐标 offset，输出 channel 为 `3*4=12`

**注意** rpn_head 中的 3 个 conv layer 对所有 scales 共享，代码如下，

```python
pred_objectness_logits = []
   pred_anchor_deltas = []
   for x in features:   # 遍历 P2,3,4,5,6
      t = self.conv(x)
      pred_objectness_logits.append(self.objectness_logits(t))
      pred_anchor_deltas.append(self.anchor_deltas(t))
   return pred_objectness_logits, pred_anchor_deltas
```

## RPN 损失

对于某个 image，假设其中包含 `M` 个 gt boxes，由于有 5 个 scales 特征平面，每个特征平面的 size 均不同，所以每个特征平面的 anchors 也不同，总共的 anchors 数量为 $\sum _ {i=1} ^ 5 h _ i \times w _ i \times 3$，为了方便表示记 anchors 总数为 $N$。

计算 gt boxes 与 anchors 的 IoU 矩阵，shape 为 $M \times N$，对矩阵每列求最大值，得到每个 anchor 的最大 IoU 值，考察这个 IoU 值范围：

1. $(-\infty, 0.3]$，anchor label 为 `0`  （负样本）
2. $(0.3, 0.5]$，anchor label 为 `-1`     （难例，忽略）
3. $(0.5, +\infty)$，anchor label 为 `1`  （正样本）

此外，还考虑了其他匹配，即，对 IoU 矩阵按行求最大值，最大值如果 `>0`，此位置对应的 anchor label 也是 `1` 。

每个 image 取 256 个样本，其中一半是正样本，即，随机选择 128 个 label 为 `1` 的 anchors 为正样本，随机选择 128 个 label 为 `0` 的 anchors 为负样本。这 `256` 个样本用于计算 RPN loss 。

**计算 loss**

现在，整理预测值和 target 。

objectness 预测值是一个具有 5 个 tensor 的 list，对应 5 个 scale，每个 tensor 的 shape 为 $ (B, 3, h _ i, w _ i), \ i=1,2,3,4,5$ </font>，其中 $B$ 是 batch size， $3$ 是指特征平面上每个 pixel 处使用 3 个 anchors 预测。坐标 offset 即 $(t _ x, t _ y, t _ w, t _ h)$ 的预测值也是一个具有 5 个 tensor 的 list，每个 tensor 的 shape 为 $(B, 3*4, h _ i, w _ i)$ ，经过 flatten 和 concatenate，这两个 list 分别转为两个 tensor，shape 分别为 $\color{cyan}(B, N)$ 和 $\color{cyan}(B, N, 4)$ 。

anchors 也是具有 5 个 tensor 的 list，每个 tensor shape 为 $(h _ i \times w _ i \times 3, 4)$ ，concatenate 之后 shape 为 $\color{cyan} (N, 4)$。

对于单个 image 而言，分类 target 的 shape 为 $(N, )$，其中 $N$ 是 5 个 scale 特征平面的 anchor 总数，坐标 x1y1x2y2 的 target shape 为 $(N, 4)$。每个 anchor 对应一个 objectness label 值（`0` 负例，`1` 正例，`-1` 忽略），每个 anchor 使用与其具有最大 IoU 的那个 gt box 的坐标作为 target 。target 通过这种表示方法，即使每个 image 的 gt boxes 数量不等，但是 target shape 是相同的。

所以一个 batch 的 分类 target shape 为 $\color{cyan} (B, N)$，坐标 target 为 $\color{cyan} (B, N, 4)$ 。

anchor boxes 和 gt boxes 坐标均需要 scale 到基于输入图像 size 。根据 anchor boxes 和 gt boxes 计算坐标 offset 即 $(t _ x, t _ y, t _ w, t _ h)$ 的 target，其 shape 为 $\color{cyan} (B, N, 4)$ ，当然了这里面计算了所有样本的 target （包括正例、负例和被忽略的 anchor）。

**计算正例的坐标回归损失**

使用 smoothL1 损失。根据分类 target `=1` 可以得到正例 mask，从而筛选出正例坐标 offset 的预测值，和正例对应的坐标 offset 的 target。

**计算分类损失**

使用 binary cross entropy 损失。计算正例和负例（label 为 `0` 和 `1`，不包括 `-1`）。

两种损失均分别求和，然后再除以 `B * 256`，因为每个 image 中正负例数量之和为 `256` 。

由于 RPN 用于生成 proposals，所以这两个损失记作 `proposal_losses` 。

### 获取 proposals

根据坐标 offset 的预测值和 anchor 计算出预测 boxes 的坐标。坐标 offset 预测值 shape 为 $(B, N, 4)$，anchors shape 为 $(N, 4)$，所以先 expand 为 $(B, N, 4)$，然后再计算，预测 boxes shape 也是 $(B, N, 4)$ 。

**筛选 proposals**

每个 scale 独立进行以下操作，

1. 当前处理第 `i` 个 scale 。根据分类预测值，选择 top $K _ 1=2000$ 个 proposals

2. 对这 top $K _ 1$ 个 proposals ，记录对应的分类预测值，以及坐标值

将所有 scales 的 top $K _ 1$ 的 proposals 值 concatenate，那么分类得分 shape 为 $(B, 5 K _ 1)$，坐标 shape 为 $(B, 5 K _ 1, 4)$ 。

每个 image 独立进行以下操作，

1. 对第 `b` 个 image，有 $5 K _ 1$ 个 proposals，筛选分类得分和坐标值正常（非 inf）的 proposals，然后将坐标值 `clip` 到这个 image size 范围内，然后筛选 size `> 0` 的 proposals

2. 执行 NMS

3. 根据预测得分取 NMS 之后 top $K _ 2=1000$ 的 proposals

那么每个 image 均得到 top $K _ 2$ 的 proposals。于是，得到一个 list，其中每个 element 表示一个 image 的 $K _ 2$ 个 proposals。

## ROI head

对于目标检测任务，ROI head 有分类和坐标回归两个分支。对于实例分割任务，还需要一个 mask 分支。

**样本分配**

对于 RPN 输出的每个图像的 $K _ 2$ 个 proposals，首先将每个 image 的 gt boxes 也作为 proposals，由于 proposal 有 objectness 预测值，所以 gt box 也设置一个 objectness 值，即 logit 值（非归一化得分），令 gt box 的 objectness 概率为 $1-\epsilon$，那么其 logit 值为

$$\log \frac {1-\epsilon}{\epsilon}$$

根据 sigmoid 函数可恢复其 objectness 概率

$$\sigma \left(\log \frac {1-\epsilon}{\epsilon}\right)=\frac 1 {1+\exp \left(-\log \frac {1-\epsilon}{\epsilon}\right)}=\frac 1 {1 + \frac {\epsilon}{1-\epsilon}}=1-\epsilon \tag{6}$$

然后根据一个策略进行样本分配（确定正负样本）。分配策略为：

1. 计算 gt boxes 与 proposals 的 IoU 矩阵
2. 按列求最大值，即，为每个 proposal 求最大 IoU
3. 最大 IoU `>0.5`，那么 proposal 为正样本；否则为负样本
4. 为每个 proposal 分配分类 label：最大 IoU 对应的 gt box 的分类，范围为 $[0, C-1]$
5. 将负样本的分类 label 设置为 $C$，表示是背景。
6. 每个样本的坐标 target 为最大 IoU 对应的 gt box。

每个 image 取 `512` 个样本，其中 `1/4` 为正样本，所以从上述正 proposals 中随机选择 `128` 个作为正样本，从负 proposals 中随机选择 `384` 个作为负样本。这 `512` 个样本用于后面计算 loss 。


ROI_head 子网络的输入来自 FPN 的输出的 `P2,P3,P4,P5`，注意这里没有使用 `P6`，P6 仅用于 RPN 子网络。

### ROIPool

由于每个 proposal 大小不等，而我们最终要为每个 proposal 输出 `C+1` 个分类预测以及 `4` 个坐标预测，所以需要使用 ROIPool 将 size 不等的 proposal 下采样为 size 相等的特征，这里采样 `7x7` 的 size 。

**分配 proposals 到 scale level**

每个 image 有 `512` 个 propopsals 样本，每个 proposal 应该对应到哪个 scale level 呢？记 scale level 为 `2,3,4,5`（注 ROI_head 中不使用 P6 特征），那么分配策略为：

1. 计算 proposal 的面积 $S=w \times h$，然后计算其平方根 $s=\sqrt S$

2. 一个预先设定的基准边长 $s _ 0 = 224$，对应的 scale level 为 $4$，那么为 proposal 分配的 scale level 为

    $$l=\lfloor 4 + \log _ 2 \frac {s}{s _ 0}\rfloor \tag{7}$$

3. 为了保证 scale level 位于 $[2,5]$ 之间，取 $l=\max(2, \min(l, 5))$

**ROIAlign**

输入特征是 `P2,P3,P4,P5`，RPN 输出的 proposals 也已经按 scale level 进行了分配，那么就每一个 scale level 单独处理 ROI 对齐。

以 `P2` 为例进行说明。

记 `P2` 特征平面 size 为 $(h, w)$，下采样率为 $d=4$，将 proposal 的坐标（当前是基于输入图像 size）转换到基于特征平面 size，即

$$x _ 1 = x _ 1 / 4, \ y _ 1 = y _ 1 / 4, \ x _ 2 = x _ 2 / 4, \ y _ 2=y _ 2 / 4$$

调整之后就可以得到 proposal 的 size 为 $w = x _ 2 - x _ 1, \ h = y _ 2 - y _ 1$ 。

现在要将 proposal 划分为 `7x7` 的 bins，那么每个 bin 的 size 则为 $w _ b = w/7, \ h _ b = h / 7$ 。

以前的 ROIPooling 方法参考上面的 (4) 式和 (5) 式，经过了两次量化，显然精确度下降，而实力分割对坐标精确度要求较高，所以不能按上面的方法进行 ROIPooling ，而是按图 3 所示，对每个 bin 采样 `n` 个点，每个点位于特征平面某个 grid cell 内，记某个点坐标为 $(x, y)$，$x, y$ 均为浮点数且没有经过任何量化，那么取周围四个点 $(\lfloor x \rfloor, \lfloor y \rfloor), \ (\lfloor x \rfloor+1, \lfloor y \rfloor), \ (\lfloor x \rfloor, \lfloor y \rfloor+1), \ (\lfloor x \rfloor+1, \lfloor y \rfloor+1)$，然后使用双线性插值得到 $(x, y)$ 处的值。

代码中没有采样 `n` 个点，而是对 bin 内所有点进行了双线性插值。

<details><summary>bin 内采样点双线性插值源码解读</summary>
代码位于 `torchvision` 项目的 `ops/roi_align.py` 文件中的 `roi_align` 函数。

```python
def roi_align(input, boxes, output_size, spatial_scale, sampling_ratio, aligned) -> Tensor
```

参数说明：

1. `input`，当前某个 scale level 的输入特征，由 FPN 输出提供。shape 为 $(B, 256, h, w)$，其中 $B$ 是训练的 batch size 。
2. `boxes`，分配到当前 scale level 的 proposals，shape 为 $(K, 5)$，其中 $K$ 是整个 batch 中分配到此 level 的 proposals 数量（整个 batch 中所有 level 的 proposals 数量为 $512\times B$），维度 $5$ 依次表示 proposal 所在 image 的 batch index，以及 $x _ 1, y _ 1, x _ 2, y _ 2$
3. `output_size`，ROIPooling 之后的 spatial size，这里是 $7 \times 7$
4. `spatial_scale`，下采样率的倒数，例如对于 `P2`，值为 $1/4$
5. `sampling_ratio`，指定每个 bin 内采样点数量，如果此值 `<=0`，那么对 bin 内所有点采样
6. `aligned`，是否对齐。此参数后面会解释。

来看具体实现代码。

```python
_, _, height, width = input.size()  # 输入特征的 size

ph = torch.arange(pooled_height, device=input.device)  # [PH] 	bin 的 y 轴 index：0~6
pw = torch.arange(pooled_width, device=input.device)  # [PW]	bin 的 x 轴 index：0~6

roi_batch_ind = rois[:, 0].int()  # [K]  K 个 proposals 所在 image 的 batch index
offset = 0.5 if aligned else 0.0

# proposal box 左上右下坐标映射到特征平面，然后位移 -0.5，这个位移保证 bin 内采样点从 0 处开始采样
roi_start_w = rois[:, 1] * spatial_scale - offset  # [K]	
roi_start_h = rois[:, 2] * spatial_scale - offset  # [K]
roi_end_w = rois[:, 3] * spatial_scale - offset  # [K]
roi_end_h = rois[:, 4] * spatial_scale - offset  # [K]
roi_width = roi_end_w - roi_start_w  # [K]
roi_height = roi_end_h - roi_start_h  # [K]

bin_size_h = roi_height / pooled_height  # [K]
bin_size_w = roi_width / pooled_width  # [K]
exact_sampling = sampling_ratio > 0		# False，使用全部采样

# y 方向采样点数量为 bin 的 height，x 方向采样点数量为 bin 的 width
# 由于 bin 的 width 和 height 均非整数，所以向上取整，得到采样点数量
# 例如 bin size 为 2.2 x 2.2 ，那么采样点数量为 3x3
roi_bin_grid_h = sampling_ratio if exact_sampling else torch.ceil(bin_size_h)  # [K]
roi_bin_grid_w = sampling_ratio if exact_sampling else torch.ceil(bin_size_w)  # [K]

count = torch.clamp(roi_bin_grid_h * roi_bin_grid_w, min=1)  # [K]	每个 proposal 对应的 bin 内采样点数量
iy = torch.arange(height, device=input.device)  # [IY]		# 特征平面 y 坐标: 0 ~ h-1
ix = torch.arange(width, device=input.device)  # [IX]		# 特征平面 x 坐标: 0 ~ w-1
ymask = iy[None, :] < roi_bin_grid_h[:, None]  # [K, IY]	# 掩码，过滤掉不属于 bin 的点
xmask = ix[None, :] < roi_bin_grid_w[:, None]  # [K, IX]	# 后面会详细解释这个掩码

def from_K(t):				# 维度适配调整
	return t[:, None, None]

# 计算采样点坐标，对应下文 (11) 式
y = (
	from_K(roi_start_h)
	+ ph[None, :, None] * from_K(bin_size_h)
	+ (iy[None, None, :] + 0.5) * from_K(bin_size_h / roi_bin_grid_h)
)  # [K, PH, IY]
x = (
	from_K(roi_start_w)
	+ pw[None, :, None] * from_K(bin_size_w)
	+ (ix[None, None, :] + 0.5) * from_K(bin_size_w / roi_bin_grid_w)
)  # [K, PW, IX]

# 采样点坐标在特征平面上不是整型数值，所以采用双线性插值
val = _bilinear_interpolate(input, roi_batch_ind, y, x, ymask, xmask)  # [K, C, PH, PW, IY, IX]
output = val.sum((-1, -2))  # remove IY, IX ~> [K, C, PH, PW]
output /= count[:, None, None, None]	# 求均值
# output 就是 ROIpooling 的输出，shape 为 (K, C, PH, PW)
```

**确定采样点坐标**

以 y 坐标为例说明。已知 当前 scale level 的下采样率为 $d$，proposal 坐标为 $x _ 1, y _ 1, x _ 2, y _ 2$，那么转换到基于特征平面，roi 坐标为

$$x _ 1 ^ r = x _ 1/d, \quad y _ 1 ^ r = y _ 1/ d, \quad x _ 2 ^ r = x _ 2 /d, \quad y _ 2 ^ r = y _ 2 / d \tag{8}$$

roi 的 top size 即 y 轴起始坐标为 $y _ 1 ^ r$ 。每个 bin 的 size 为 $h ^ b = (y _ 2 ^ r - y _ 1 ^ r) / 7$，那么这个 roi 内各个 bin 的 y 轴起始坐标为 $y _ 1 ^ b = y _ 1 ^ r + i\cdot h ^ b$，其中 $i=0,1,\ldots, 6$ 是 bin 的 y 方向 index。

正如代码注释所说的那样，在这里的策略中，采样点数量为 $\lceil w ^ b \rceil \times \lceil h ^ b \rceil$，且所有采样点均匀分布于 bin 内，于是 bin 内各个采样点 y 坐标（基于 bin 而非基于特征平面）为 

$$y ^ p = 0.5+k, \quad k = 0,1,\ldots, \lceil h ^ b \rceil - 1 \tag{9}$$

(9) 式这个采样点坐标是基于向上取整的 bin size，所以为了精确性，将坐标映射回原来的 bin size，

$$y ^ p := y ^ p \cdot \frac {h ^ b}{\lceil h ^ b \rceil} \tag{10}$$

其中 $y ^ p / \lceil h ^ b \rceil$ 是归一化的 y 坐标，然后乘以 $h ^ b$ 就恢复到基于 bin height 的坐标。

于是基于特征平面坐标系，采样点的 y 坐标为

$$y _ g ^ p = y _ 1 ^ r + y _ 1 ^ b + y ^ p \tag{11}$$

如图 5 所示，红色点是第一个采样点即，最左上角区域的采样点，那么将来使用双线性插值的时候则是使用红色点所在 cell 的四个角点，而这个 bin 的左上角实际上是位于蓝色点所在 cell，故如果想要对齐（参数 `aligned=True`），应该使用蓝色点所在 cell 的四个角点进行插值，这种对齐可以通过将 roi 坐标位移 $(-0.5, -0.5)$ 得到，这样所有的坐标包括 roi，bin，采样点坐标，整体全部都位移了 $(-0.5, -0.5)$ ，于是 (8) 式 roi 坐标改为

$$x _ 1 ^ r = x _ 1/d-0.5, \quad y _ 1 ^ r = y _ 1/ d-0.5, \quad x _ 2 ^ r = x _ 2 /d-0.5, \quad y _ 2 ^ r = y _ 2 / d-0.5 \tag{8}$$


![](/images/mask-rcnn_fig5.png)

<center>图 5.</center>


计算采样点坐标的代码中，`y` 有 3 个维度，分别表示 proposal(roi)，bin 和采样点。`iy` 的范围是 $[0, h-1]$（$h$ 是特征平面 height），这里 `iy` 应该对应 (9) 式中的 $k$，而 $k$ 范围是 $[0, \lceil h \rceil -1)$，所以对于超出当前 bin 范围的采样点进行过滤，方法见代码中的 `ymask = iy[None, :] < roi_bin_grid_h[:, None]`，正好就是将范围缩小到 $[0, \lceil h \rceil -1)$ 。

**双线性插值**

采样点 $(x, y)$ 的值使用所在 cell 的四个角点进行双线性插值，四个角点坐标为 

$$(\lfloor x \rfloor, \lfloor y \rfloor), \ (\lfloor x \rfloor+1, \lfloor y \rfloor), \ (\lfloor x \rfloor, \lfloor y \rfloor+1), \ (\lfloor x \rfloor+1, \lfloor y \rfloor+1)$$

4 个权重为

$$\begin{aligned}w _ 1 &= (1+\lfloor x \rfloor - x) (1+\lfloor y \rfloor - y)
\\\\ w _ 2 &= (x-\lfloor x \rfloor) (1+\lfloor y \rfloor - y)
\\\\ w _ 3 &= (1+\lfloor x \rfloor - x) (y-\lfloor y \rfloor)
\\\\ w _ 2 &= (x-\lfloor x \rfloor) (y-\lfloor y \rfloor)\end{aligned} \tag{12}$$

代码中变量 `val` 的 shape 为 `[K, C, PH, PW, IY, IX]`，分别表示 proposal(roi)，channel(=256)，bin y 坐标，bin x 坐标，采样点 y 坐标，采样点 x 坐标。

将每个 bin 内的所有采样点求均值就得到 ROIPooling 的输出，shape 为 `(K, 256, 7, 7)` 。

</details>

每个 scale level 的 ROIPooling 输出 shape 为 `(K, 256, 7, 7)`，其中 `K` 是某个 level 的 proposals 数量，组合起来就是 `(512B, 256, 7, 7)`，其中 `B` 是训练的 batch size 。

## box head

ROIPooling 之后，得到特征，其 shape 为 `(512B, 256, 7, 7)`，其中 `B` 是训练 batch size，每个 image 中选择 `512` 个 proposals 样本（正负样本比例 `1:3`，根据 RPN 输出 objectness 得分排序，选择 top $K _ 1$，然后 NMS，然后选择 top $K _ 2$，然后将 gt boxes 也加入 proposals，然后根据 IoU 阈值分配正负样本，最后选择计算适量正负样本）。

现在这个特征分别经过 box head ，进行坐标回归预测，和分类预测。

box head 由 若干 conv + 若干 FC 构成，代码中 FC 输出 channel 为 `1024`，所以 box head 输出的特征 shape 为 `(512B, 1024)` 。

box head 的输出特征分别经过一个 FC 将 channel 调整为 `C+1` 和另一个 FC 将 channel 调整为 `C*4`，前者用于预测分类，后者预测 box 坐标，每个分类单独预测 box 坐标。

**box head 损失**

坐标损失使用 smooth L1，分类损失使用交叉熵损失。

proposals 数量为 `512*B`，分类预测值 shape 为 `(512B, C+1)`，分类 target 的 shape 为 `(512B, )`，正例 proposal 为最大 IoU 对应的 gt box 分类，负例 proposal 为 `C` 。

计算正例 proposals 的坐标回归损失，由于坐标预测维度是 `C*4`，所以先根据正例 proposals 的分类 target 提取对应的 box 预测，得到预测数据的 shape 为 `(N, 4)`，其中 `N` 是正例数量。

正例的坐标 target 需要根据正例 proposals 的坐标和对应的 gt boxes 坐标计算得到，如下所示，

$$\begin{aligned} t _ x &= \frac {x _ g - x _ p}{w _ p}, \quad t _ y = \frac {y _ g - y _ p}{h _ p}
\\\\ t _ w &= \log \frac {w _ g}{w _ p}, \quad t _ h = \log \frac {h _ g}{h _ p}
\end{aligned} \tag{13}$$

其中下标 $g$ 表示 gt box，$p$ 表示 proposal。（RPN 中其实也是类似处理得到坐标回归 target，把 $p$ 改成 $a$ 表示 anchor box）

**注意**

所有 **正例** proposals 的坐标回归损失求和，然后 **除以总的 proposals 数量** 求平均，为什么不是除以正例数量，源码注释解释了原因：为了给每个正例相等的训练影响，这里正例是指全数据集中的正例。考虑以下两种 mini batch 的情况：

1. 仅有一个正例
2. 有 100 个正例

如果除以正例数量，那么情况 `1` 的正例梯度是情况 `2` 中正例梯度的 100 倍，所以导致学习影响不均衡。

## mask head

实例分割比目标检测多了一个 mask head。与 box head 类似，mask head 也是将 FPN 的输出特征（仅使用 `P2,P3,P4,P5`，P6 特征不用）使用 proposals 进行 ROIPooling ，这里 proposals 与 box head 中的完全相同，只是 ROIPooling 输出 spatial size 为 `14x14`，因为 `7x7` 太小，损失了太多信息。

mask head 完成处理过程如下：

1. 使用与 box head 相同的 proposals，即每个 image 取 512 个 proposals，其中 `128` 个正例
2. 提取正例 proposals，数量为 `128*B`，这里 `B` 是训练的 batch size。使用 ROIPooling 将 `128*B` 个正例 proposals 的特征池化为 `14x14` 的 size，即特征 shape 为 $(128B, 256, 14, 14)$

	这里使用 proposal boxes 而非 proposals 对应的 gt boxes 对特征进行框定，然后再进行池化。

3. 经过 `4` 个 conv3x3 ，输出 shape 不变，为 $(128B, 256, 14, 14)$
4. 经过 `1` 个转置卷积，用于增大输出特征的 spatial size，输出的 shape 为 $(128B, 256, 28, 28)$

	代码为，

	```python
	self.deconv = ConvTranspose2d(
		# feat_in: 256, feat_out: 256
		cur_channels, conv_dims[-1], kernel_size=2, stride=2, padding=0
	)
	```

	计算输出 spatial size 时，将输入输出 size 置换即可，例如求 h，

	$$h _ i = \frac {h _ o + 2p - k} {s} + 1$$

	代入 $h _ i = 14, \ s = 1, \ p = 0, \ k = 2$，解得 $h _ o = 28$

5. 经过 `1` 个 conv1x1，输出 channel 为 $C$，预测特征平面上每个 pixel 的分类得分，数据 shape 为 $(128B, C, 28, 28)$

	代码为，

	```python
	self.predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)
	```

**计算 mask head 的损失**

每个 proposal 均有自己的 box（由 RPN 预测得到），以及匹配的 gt box，而每个 gt box 均有一个 gt mask，其 shape 为 $(H, W)$ ，即 network 输入 image 的 size。相关代码片段为，

```python
# sampled_idxs: 当前 image 中选择的 512 个 proposals 的 idxs
# matched_idxs: 当前 image 中所有的 proposals 对应的 gt boxes 的 idxs
sampled_targets = matched_idxs[sampled_idxs]	# 得到 512 个 proposals 对应的 gt boxes 的 idxs
for (trg_name, trg_value) in targets_per_image.get_fields().items():
	if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):# 例如 gt_masks
		proposals_per_image.set(trg_name, trg_value[sampled_targets])	# 得到 512 个 proposals 对应的 gt_masks
```

现在每个 image 有 `128` 个正样本 proposals，所以其对应的 gt masks 的 shape 为 $(128, H, W)$。

1. 根据 proposal 的 boxes 坐标，从 gt masks 中抠出，然后 resize 到 `28x28` 的大小，得到单个 image 的 mask target，其 shape 为 $(128, 28, 28)$，每个 pixel 的 target 值为 `0` 或 `1` 。
2. 每个 proposals 的预测得分为 $(128, C, 28, 28)$，根据每个 proposal 对应 gt box 的分类，取分类所对应的 channel map，那么所得数据的 shape 为 $(128, 28, 28)$
3. 以上两者进行交叉熵计算，spatial map 上每个 pixel 处独立计算交叉熵损失。由于预测得分表示 logits，非归一化，所以先对预测得分使用 $\sigma$ 函数，然后计算 binary 交叉熵。

**如何将 mask map 进行 crop and resize**

这里与前面 ROIAlign 原理完全相同，在 gt mask map 上将 proposal box 划分为 $28\times 28$ 个 bin，每个 bin 内完全采样，然后每个采样点所在 cell 的 4 个角点使用双线性插值得到这个采样点的值，bin 内所有采样点计算均值，然后判断均值是否 `> 0.5`，如是，那么这个 bin 的 mask target 为 `1`，否则为 `0` 。

这里 gt mask map 的 size 为 $(H,W)$ 即网络的输入 image size，而 proposal 的坐标刚好也是基于输入 image size 的，所以无需将 proposal 坐标 rescale，如下方代码中的 ROIAlign 初始化参数 `1.0`（回顾前面对 scale level 的特征进行 ROIAlign 则需要非 1 的 `spatial_scale` 值），

```python
output = (
	ROIAlign((mask_size, mask_size), 1.0, 0, aligned=True)	# 这里参数 1.0 表示 spatial_scale
	.forward(bit_masks[:, None, :, :], rois)
	.squeeze(1)
)
output = output >= 0.5
```

## inference

实力分割任务的推理阶段的流程：

1. FPN 生成 multi scale 的输出特征
2. 上一步的特征送入 RPN，得到基于 anchor 的 objectness 预测得分以及坐标回归预测。
3. 每个 level 取 top $K _ 1 = 1000$ 的 proposals，然后每个 image 的 $5 K _ 2$ 个 proposals 进行 NMS，然后再取 top $K _ 2 = 1000$

4. 上述 $BK _ 2$（其中 B 是 batch size）个 proposals 与 FPN 的输出特征一起送入 roi_head，roi_head 取 `P2,P3,P4,P5` 特征，然后使用 proposals 框定出区域特征

5. 将 proposals 的区域特征使用 ROIPooling 得到 `7x7` 的 spatial size 特征，其 shape 为 $(BK _ 2, 256, 7, 7)$，flatten 之后送入两个 FC（即 box head）将输出 channel 调整为 `1024`

6. 上一步 box head 输出的 $(BK _ 2, 1024)$ 分别经过：一个 conv1x1 输出 $(BK _ 2, C+1)$ 的分类预测得分；一个 conv1x1 输出 $(BK _ 2, 4C)$ 的坐标回归预测

7. 根据上一步的坐标回归预测，将这个偏差预测，应用到 proposals 上，得到最终的 boxes 坐标预测。将上一步的分类预测得分通过 softmax 进行归一化。于是，每个 image 的 boxes 坐标值 shape 为 $(K _ 2, 4C)$，分类概率预测的 shape 为 $(K _ 2, C+1)$

8. 每个 image 中，上述坐标值 reshape 为 $(K _ 2, C, 4)$，分类得分预测去掉最后一列，因为最后一列是 bg 的概率，故所有前景的分类概率预测 shape 为 $(K _ 2, C)$，使用一个阈值 `0.05`，筛选出 `> 0.05` 的分类概率预测，根据这个 mask，筛选出相应的预测坐标值，以及相应的分类概率预测值，记分类预测概率矩阵 $K _ 2 \times C$ 中有 $R$ 个元素值 `> 0.05`，那么筛选出来的分类概率 shape 为 $(R, )$，相应的坐标 shape 为 $(R, 4)$

9. 将上一步中的 boxes 进行 NMS，注意是每个分类独立进行 NMS，NMS 阈值为 `0.5` ，记 NMS 之后保留的 box 数量为 $r$。

10. 上一步 NMS 之后保留下来的 boxes 中，再按分类预测得分，选择最多 top `100` 即 $r:=\min(r, 100)$ 个预测 boxes

（以上步骤，就得到了每个 image 中的预测 box 坐标和所属分类。接下来看 mask 预测。）

11. 根据上一步得到的 $r$ 个预测 boxes，将 FPN 的输出特征 `P2,P3,P4,P5` 使用 mask_head 的 ROIPooling，池化得到 $(r, 256, 14, 14)$ 的特征，对 `B` 个 images，则得到所有 proposals 的池化特征 shape 为 $(\sum _ {i=1} ^ B r _ i, 256, 14, 14)$ ，经过 4 个 conv3x3，以及 1 个 deconv，输出特征为 $(\sum _ {i=1} ^ B r _ i, 256, 28, 28)$，最后使用 1 个 conv1x1，调整输出 channel，得到 mask 的预测 logits，其 shape $(\sum _ {i=1} ^ B r _ i, C, 28, 28)$ 。

12. 根据所有 images 的 $\sum _ {i=1} ^ B r _ i$ 个预测 boxes 的分类预测，从 mask 的预测 logits 中提取各个分类对应的 channel map，得到 logits 其 shape 为 $(\sum _ {i=1} ^ B r _ i, 1, 28, 28)$，然后应用 $\sigma$ 函数，就得到 mask 预测概率。

现在，得到所有 $\sum _ {i=1} ^ B r _ i$ 预测 boxes 的分类，坐标，以及 mask 概率，但是需要注意每个 box 的 mask map 是 $28\times 28$ 大小的，而非输入 image $H \times W$ 大小，所以还需要将 mask size rescale 到 image size 。backbone 是全卷积网络，所以一个 batch 中各个 image size 可能也不同，并且同一 image 的宽度和高度也可能不等。

每个预测 box 的 mask map 为 $28\times 28$，注意其对应的应该是 box 自身，而非整个 image ，记 box 的预测坐标为 $(x _ 1, y _ 1, x _ 2, y _ 2)$， 所以常见的方法是将 mask map 通过双线性插值 resize 到 $(y _ 2 - y _ 1, x _ 2 - x _ 1)$ ，然后 paste 到 image 中，左上角顶点位置为 $(x _ 1, y _ 1)$ 。代码中使用 `torch.nn.functional.grid_sample` 这一函数同时实现 resize and paste 功能。

**grid_sample 函数简介**

```python
torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None)
```

`input` 为数据源，shape 为 $(N, C, H_{in}, W _ {in})$， 我们这里就是 $28 \times 28$ 大小的 mask map

`grid` shape 为 $(N, H _ {out}, W _ {out}, 2)$，其值指定 `input` 中的数据位置。

输出：shape 为 $(N, C, H _ {out}, W _ {out})$

计算输出数据每个位置 $[n,:, h, w]$ 的值时，根据 grid 中相同位置 $[n, h, w]$ 的值，这个值是一个 length=2 的向量，记为 $z$ ，这个 $z$ 就指定了 `input` 中的 spatial location，数学表达式如下，

$$\begin{aligned} & z = \text{grid}[n,h,w]
\\\\ & x = 0.5\times (z[0] + 1) \times W _ {in}, \quad y = 0.5\times (z[1]+1) \times H _ {in}
\\\\ & \text{output}[n,:,h, w] =\text{input}[n,:, y, x]\end{aligned}$$

注意上式中，

1. 如果 $z [i] \in [-1, 1], \ i =1,2$，那么 $x \in [0, W _ {in}]$，$y \in [0, H _ {in}]$，$x, y$ 不是整型数值，所以使用双线性插值
2. 如果 $z[i] \notin [-1,1], \ i=1,2$，那么输出值使用指定的填充模式进行填充

如此，就得到 image size 的 mask，最后根据 mask 的阈值例如 `0.5` 进行二值化。