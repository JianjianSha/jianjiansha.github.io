---
title: yolov8 解读
date: 2023-08-11 17:51:10
tags: object detection
mathjax: true
---

源码：[ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

# 1. 网络结构

yolov8 与 yolov5 一样，不仅用于检测，还可以用于分类，分割等任务。本文主要讨论目标检测任务。

yolov8 有 5 个 size 不同的模型，结构相似，在模型 depth，width，max_channels 三个维度上不同，所有模型配置位于文件 `ROOT/cfg/models/v8/yolov8.yaml` 。

目标检测模型类位于文件 `nn/tasks.py`，为 `DetectionModel`，其他任务如分割，分类，姿态估计等模型类也位于文件 `nn/tasks.py` 中。

以 yolov8l 为例说明，因为 `l` 这个模型的 depth 和 width 的缩放比例均为 `1` ，网络配置为，

```sh
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]  # 9

# YOLOv8.0n head
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]]  # cat backbone P4
  - [-1, 3, C2f, [512]]  # 12

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 4], 1, Concat, [1]]  # cat backbone P3
  - [-1, 3, C2f, [256]]  # 15 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]  # cat head P4
  - [-1, 3, C2f, [512]]  # 18 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]  # cat head P5
  - [-1, 3, C2f, [1024]]  # 21 (P5/32-large)

  - [[15, 18, 21], 1, Detect, [nc]]  # Detect(P3, P4, P5)
```

网络的整体结构与 yolov5 类似，其中 `C3` 换成了 `C2f` 模块，并且去掉了 head 中 upsample 前面的 conv 。

**几个特殊的 模块**

**C2f**

```sh
    +-------+                               +-------+
--->| Conv =+==+--------------------------->|       |
    +-------+  |                            |       |
               +--------------------------->|       |
               |        +-------------+     |       |
               |   +--->| Bottleneck -+---->|       |
               |   |    +-------------+     |       |
               |   |        .        ------>| Conv -+--->
               +---+        .        ------>|       |
                   |        .        ------>|       |
                   |    +-------------+     |       |
                   +--->| Bottleneck -+---->|       |
                        +-------------+     +-------+       (C2f)
```

```python
class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
```

**SPPF**

```sh
-> conv -+-> maxpool -+-> maxpool -+-> maxpool -> c
         |            |            +------------> o ->
         |            +-------------------------> n
         +--------------------------------------> v         (SPPF)
```

```python
class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
```

整个网络结构的关键部分如图 2 所示，

![](/images/obj_det/yolov8_2.png)
<center>图 2. </center>

图 2 中浅蓝色的特征为网络的输出特征，与 FPN 的主要区别是：每个 scale 的输出特征既有高层语义又有低层语义，而 FPN 通过自上而下的融合，使得小 scale 的输出特征仅有高层语义。

预测包含了 3 个 scale 的特征

```sh
# name  下采样率
P3      8
P4      16
P5      32
```

网络结构中 Detect 将分类和坐标检测头分开（解耦），且使用 anchor free ，如图 1 所示，

![](/images/obj_det/yolov8_1.png)

<center>图 1. 图源：参考文章</center>

> 本文参考了文章 https://mmyolo.readthedocs.io/zh_CN/latest/recommended_topics/algorithm_descriptions/yolov8_description.html，记为参考文章

训练阶段，`Detect` 对 3 个 scale 的预测特征平面分别处理，每个 scale 特征均经过图 1 下方所示的检测头，检测头有一个分类分支和一个回归分支，分别经过 3 个 conv，得到分类输出 shape 为 `(b, nc, h, w)`，其中 `nc` 表示前景分类数量。回归输出 shape 为 `(b, 4 * reg_max, h, w)`，其中 `reg_max=16` ，两者 concat 为 shape  `(b, 4 * reg_max + nc, h, w)` ，这就是某个 scale 对应的预测输出。 

相关代码如下，

```python
def forward(self, x):
    """x: A list of 3 tensors, and each tensor's shape is (b, c, h, w)"""
    shape = x[0].shape  # BCHW
    for i in range(self.nl):
        # cv2 output is for box regression: (b, 4*reg_max, h, w)
        # cv3 output is for classification: (b, nc, h, w)
        x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
    if self.training:
        return x # x 是一个list，包含 3 个 tensor，每个 tensor 形如 (b,4*reg_max+nc,h,w)
```

我们需要知道：yolov8 是 anchor free 模型，所以每个 anchor point 处均需要预测 4 个坐标值以及分类得分。yolov8 预测每个坐标的分布而非一个确定的值，将坐标分布离散化，使用 `reg_max=16` 个离散值，每个离散值预测对应的概率，计算分布的期望作为坐标的预测值，参见下文 (2) 式。

# 2. Loss 计算

loss 计算包括 如何分配正负样本和如何计算损失两部分。

yolov8 使用动态分配策略：TOOD 的 TaskAlignedAssigner，TaskAlignedAssigner 策略为：根据分类与回归的分数加权和选择正样本。

$$t = s ^ {\alpha} \times u ^ {\beta}\tag{1}$$

其中 $s$ 表示标注类别对应的预测得分，即，每个 anchor point 取所在 gt box 的分类得分，如果 anchor point 不在任何 gt box 内部，那么 $s=0$。

如果 anchor point 在 n 个 gt box 内部，那么保留这 n 个 gt box 对应的分类得分，因为实现过程中，使用一个得分矩阵，矩阵行数为 gt box 数量，矩阵列数为 anchor point 数量，故这个 anchor point 所在列将有 n 个值非零。

$u$ 是预测 box 与 gt box 的 IoU。$\alpha, \beta$ 为两个控制参数，分别控制 $s$ 和 $u$ 对 t 值指标的影响。

这里 $t$ 是一个指标，用于衡量分类任务的 anchor point 和定位任务的 anchor point 对齐程度，一个 anchor，如果既能有大的分类得分预测，又能有精确的定位，那么这个 anchor 就是很好对齐的。

基于 $t$ 指标，为每个 gt box 选择 top-K 大的 t 值对应的 anchor points 作为正样本，其他 anchor points 作为负样本。

代码中相关超参数选择为，

```python
self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
```

**损失** 包含分类损失和回归损失，没有置信度（objectness）损失。

1. 分类损失使用 BCE 损失（二值交叉熵损失），计算对象：所有 anchor points
2. 回归损失使用  DFL 损失（distribution focal loss），还使用了 CIoU 损失。

**问题**

1. 为什么取消了置信度预测之和，分类数量没有包含背景，即，为什么不使用 `nc+1`?
2. 为什么 `reg_max` 取值 16，即，如何选择合适的 `reg_max` 值？

这两个问题等下文介绍了损失如何计算之后再来解答。

## 2.1 动态分配样本

根据 (1) 式动态分配样本。

有 3 个 scale 的 feat maps，但是可以通过 concatenate 操作，看作是 $h _ 1 w _ 1 + h _ 2 w _ 2 + h _ 3 w _ 3$ 个 anchor points，本节为了表示简洁，记总共 anchor points 数量为 $3hw=h _ 1 w _ 1 + h _ 2 w _ 2 + h _ 3 w _ 3$ 。（anchor free 模型，每个特征平面的 point 可看作是一个 anchor）。

batch 数据中，目标 gt boxes 的 x1y1x2y2 坐标，数据 shape 为 `(b, max_obj_num,  4)`，其中 `b` 是 batch size，`max_obj_num` 是这批数据中单个图像的目标数量的最大值，显然这是经过填充的，使用一个 `mask_gt` 作为真实目标 box 坐标数据的掩码，易知掩码 shape 为 `(b, max_obj_num)` 。

gt boxes 的分类 labels 的 shape 为 `(b, max_obj_num, 1)` ，前景分类 index 为 `0 ~ nc-1`。

<font color="magenta">模型预测输出点距离左上右下的 distance 数据 shape 为 `(b, 3hw, reg_max * 4)`</font>。这里使用 [DFL](/2023/08/13/obj_det/GFL)，box 坐标预测值是一个分布，离散化处理，则是 `reg_max` 个离散值，每个值对应一个概率，所以回归分支实际上预测的是每个离散值的概率，例如 anchor point 距离 box 左边的距离是一个离散型随机变量，取值范围为 `0, 1, ..., reg_max-1`，模型预测输出距离左边 distance 为一个 `reg_max` 长度的向量，表示各个离散值的概率，那么距离左边 distance 的预测结果为 

$$\hat y = \sum _ {x=0} ^ {reg\_max} p(x) x \tag{2}$$

距离其余三个边的 distance 类似处理，处理后得到预测 boxes 的 x1y1x2y2 坐标，shape 为 `(b, 3hw, 4)` ，即，每个 anchor point 处有一个预测 box，需要注意，这里得到的预测 box 的坐标是基于各个 scale 特征平面的，也就是说，相对于原输入 image size，box 的坐标缩小到 `1/stride` 。

<font color="magenta">模型预测分类得分 `pd_scores` 的 shape 为 `(b, 3hw, nc)`</font> 。

**# 动态分配**

有了以上数据说明，接下来看如何动态分配样本。调用语句为

```python
# pred_scores: 所有锚点处的预测分类得分，(b, 3*hw, nc)
# pred_bboxes: 所有锚点处的预测 box 坐标，基于相应 scale 特征平面，(b, 3*hw, 4)
# anchor_points: 所有锚点坐标，基于相应 scale 特征平面，(3*hw, 2) y-x 坐标
# gt_labels: gt box 的分类 id，(b, max_obj_num, 1)
# gt_bboxes: gt box 的坐标，基于 image size，(b, max_obj_num, 4)
# mask_gt: gt box 的 mask，因为部分 gt box 是填充的，(b, max_obj_num, 1)
_, target_bboxes, target_scores, fg_mask, _ = self.assigner(
    pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
    anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
```

1. 根据 gt boxes 的 x1y1x2y2 和 `3hw` 个 anchor points，筛选出位于 gt boxes 内部的 anchor points，得到筛选掩码 `mask_in_gt`，其 shape 为 `(b, max_obj_num, 3hw)`，再与 `mask_gt` 相乘，得到 <font color="magenta">最终掩码</font>（即，位于有效非填充 gt boxes 内部的 anchor points），shape 依然为 `(b, max_obj_num, 3hw)`

2. 从预测得分中根据 gt 分类 label 取值，然后使用最终掩码提取有效 anchor points 的预测分类得分。

    对 batch 中某个 image 其预测得分矩阵 shape 为 `(3hw, nc)`，这个 image 有 `max_obj_num` 个 gt boxes，每个 gt box 有一个分类 index，可以取得 `(3hw,)` 的预测分类，那么取得这个 image 的所有 gt boxes 一共取得预测得分 的 shape 为 `(max_obj_num, 3hw)`，`b` 个 images 一共取得的预测得分为 `(b, max_obj_num, 3hw)` 。当然了，只有最终掩码为 True 的位置上的预测得分有效，其他位置上无效值使用 `0` 表示。

    预测分类得分提取结果记为 `bbox_scores`，这个 `bbox_scores` 就是 (1) 式中的 $s$ 。

3. 类似第 `2` 步，提取预测 boxes 坐标，然后再根据最终掩码提取有效位置处的预测坐标，其 shape 为 `(N, 4)`，N 为最终掩码中 true 元素数量。

    batch 数据中总共 `(b, max_obj_num)` 个 gt boxes，每个 gt boxes 提取 `(3hw, 4)` 的预测 boxes 坐标，那么一共 `(b, max_obj_num, 3hw, 4)` 的提取数据，然后取最终掩码 `(b, max_obj_num, 3hw)` 中 true 位置处的值。

4. gt boxes 坐标数据 shape 为 `(b, max_obj_num, 4)`，通过扩充（repeat）操作得到 `(b, max_obj_num, 3hw, 4)`，然后同样的取最终掩码为 true 位置处的值，得到 `(N, 4)` 。

5. 第 `3` 和第 `4` 步提取的预测 boxes 与 gt boxes，计算 CIoU，由于两个 tensor shape 均为 `(N, 4)`，计算出来的 CIoU 为 `(N, )` 向量。创建一个 `overlaps`，其 shape 为 `(b, max_obj_num, 3hw)`，根据最终掩码为 true 的位置，将 CIoU 设置到 `overlaps` 中。这个 `overlaps` 就是 (1) 式中的 $u$ 。

6. 根据 (1) 式计算 $t$ 值，使用超参数 $\alpha=0.5, \ \beta=6.0$ ，代码如下：

    ```python
    # (b, max_obj_num, 3hw)
    align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
    ```

    易知， `align_metric` 的 shape 为 `(b, max_obj_num, 3hw)`，每个 gt box 有 `3hw` 个 $t$ 值，最终掩码 true 值位置处值有效，其他值均为 0 。

7. 将 `align_metric` 沿着 `dim=-1` 取 topK 大的值，以及对应的位置，即，每个 gt box 取 topK 大 $t$ 值对应的 anchor points 作为正样本。

    取出来的 topK 大 anchor points 实际上还需要使用 `mask_gt` 做掩码过滤，因为 `align_metric` 有 `(b, max_obj_num)` 个 gt boxes，显然要过滤掉填充的假 gt boxes 。最后所选择的 topK 的 anchor points 也使用掩码表示，记为 `mask_topk`，其 shape 为 `(b, max_obj_num, 3hw)`，一共 `b * max_obj_num` 个 gt boxes，每个 gt box 的 `3hw` 长度向量 mask 中，有 K 个 元素为 true 或者全部为 false（这种对应填充的假 gt box）。

8. `mask_topk` 与上述 `mask_in_gt` 和 `mask_gt` 三者按 elementwise 相乘，得到用于提取 **正样本** 的掩码，记为 `mask_pos`，其 shape 为 `(b, max_obj_num, 3hw)` 。

    **某个 anchor point 被多个 gt boxes 包含，那么与这个 anchor point 的预测 box 有最大 CIoU 的 gt box 被选中**。以某一个 image 为例，那么正样本掩码 shape 为 `(max_obj_num, 3hw)`，这是一个矩阵，按行求和，那么得到每个 gt boxes 所对应的正样本数量。按列求和，那么能判断每个 anchor point 被用作了几次正样本，或者是，有几个 gt boxes 将这个 anchor point 作为正样本，如果某个 anchor point 被不止 1 个 gt boxes 看作正样本，那么这个 anchor point 只选择具有最大 CIoU 的那个 gt box，也就是说其余 gt box 不再将这个 anchor point 看作正样本，经过这样的调整后的正样本掩码仍用变量 `mask_pos` 存储 。


## 2.2 计算损失

有了以上说明，我们来看如何计算损失：

1. 分类损失为所有样本（**包括正负样本，所有 anchor points**）的 BCE 损失。因为取消了置信度损失，所以这里分类损失除了考虑正样本，还需要考虑负样本。
    
    `target_scores`: shape 为 `(b, 3*hw, nc)`，为每个 anchor point 设置 gt label。只有正样本 anchor points 处有 one-hot 向量，此向量长度为 `nc`，向量中 正样本对应的 gt box 分类 id 处的元素值为 1，但是这里实际上不使用 `1` 作为 target 值，而是 

    $$\hat t _ {ij} \cdot \max(CIoU _ i) = \frac {t _ {ij}}{\max (\mathbf t _ i)} \cdot \max (CIoU _ i)\tag{3}$$

    以单个 image 为例理解上式更容易些，$t$ 值矩阵 shape 为 `(max_obj_num, 3*hw)`，对于每一个 gt box 编号为 `i` 分布进行归一化得到 $\hat t$ 。然后 CIoU 矩阵 shape 也是 `(max_obj_num, 3*hw)`，一个 gt box 对应多个正样本 anchor points，自然就有多个 CIoU 值，求出最大的 CIoU ，记为 $CIoU _ i$ ，以此值为这个 gt box 的分类得分基准，那么与这个 gt box 匹配的正样本 anchor points 的分类得分 target 就是这个基准乘上 $\hat t$ 。

    负样本 anchor point 则是全 0 向量。
    
    分类损失使用 BCE，正负样本均参与计算，相关代码为，

    ```python
    target_scores_sum = max(target_scores.sum(), 1) 
    loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
    ```

2. 坐标损失，只考虑正样本的坐标损失。损失包含两种：

    - CIoU 损失：$1-CIoU$ 。CIoU 损失还考虑了权重，使用分类得分 `target_scores` 作为权重，分类得分 target 实际上考虑了定位质量，值越大，越是需要注重学习优化，所以使用 `target_scores` 作为权重是合理的 。计算加权平均。

    - DFL 损失：

    $$DFL(\mathcal S _ i, \mathcal S _ {i+1}) =-((y _ {i+1}-y)\log \mathcal S _ i + (y - y _ i)\log \mathcal S _ {i+1}) \tag{4}$$
    
    上式就是 [GFL](/2023/08/13/obj_det/GFL) 一文中的 (6) 式。下面给出计算 DFL 的代码，然后根据代码进行说明。

    ```python
    def _df_loss(pred_dist, target):
        """Return sum of left and right DFL losses."""
        # pred_dist: (n * 4, reg_max)
        # target: (n, 4)， left top right bottom distance
        # 返回值：(n, 4) -> (n, )

        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)
    ```

    这个函数的参数为正例的 box 预测值，实际上是正例 anchor point 距离预测 box 左上右下 4 个 distance 的离散概率分布的预测，即一个 anchor point 预测 4 个向量，每个向量表示一个 distance 的概率分布，记 batch 中一共 `n` 个正样本。4 个 distance 的 target 值为变量 `target`，其 shape 为 `(n, 4)`，也就是 (4) 式中的 $y$ 。`tl` 表示 (4) 式中的 $y _ i$，`tr` 为 $y _ {i+1}$ ，`wl` 为 $y _ {i+1}-y$， `wr` 为 $y - y _ i = y - (y _ {i+1}-1)=1-(y _ {i+1}-y)$

    DFL 与 CIoU 一样，使用加权平均，权重也是使用正例的 `target_scores` 分数。

## 2.3 总结

对于一个 image，其中有 `obj_num` 个 gt boxes，`h*w` 个 anchor points（只考虑单个 scale 预测平面），那么可以计算 CIoU 矩阵，shape 为 `obj_num x hw`，每行表示一个 gt box，每列表示一个 anchor point。

需要注意，这里回归分支预测输出不是 anchor points 距离预测 box 的左上右下 4 个 distance 值，而是 4 个离散分布，每个 distance 取值范围为 `0 ~ reg_max-1` 是固定的预先设置好的，预测的是 distance 的概率分布，分布的期望，作为 distance 的预测值。

正样本不仅仅是指某个 anchor point ，还包含与之相关联的 gt box，即，这个 anchor point 位于这个 gt box 内部，也可以称作正样本对 `(anchor point, gt box)` 。那么如果 anchor point 位于多个 gt boxes 内部呢？根据 CIoU 矩阵这个 anchor point 所在列求最大值，最大值所在行，决定了 anchor point 关联到哪个 gt box 上，其他 gt boxes 不再与这个 anchor point 关联。另外，还通过计算 (1) 式选择 top-k 的 anchor points 为正样本，其余正样本置为负样本。


### 2.3.1 分类

正样本 anchor point 的分类 target 是一个 one-hot 向量，向量长度为分类总数 `nc` ，向量中 `1` 值所在位置就是 anchor point 关联的 gt box 的分类 id （范围为 `0 ~ nc-1`）。负样本分类 target 则是全 `0` 向量。

由于分类预测最佳 anchor point 与回归预测最佳 anchor point 不对齐，那么 NMS 时会导致差的预测 anchor point 抑制好的预测 anchor piont，所以为了对齐，正样本分类 target 的 one hot 向量中的 `1` 值替换为 (3) 式计算结果。(3) 式基于三点考虑：

1. 同一 gt box 内部不同正样本 anchor points 的分类 target 应该与各自的 CIoU 成正比，这样 CIoU 最大的 anchor point 既是分类预测最佳 anchor point，又同时是回归预测最佳 anchor point。作者实现中，使用的是 $t$ 值，即，分类 target 与 $t$ 值成正比。

2. 不同 gt boxes 对应的正样本 anchor points，其分类 target 也应该不同。为每个 gt box 设置一个分类 target 基准，对 CIoU 矩阵按行取最大值，那么每行最大值就是这个 gt box 的基准。

所有样本的分类 target 应该是 shape 为 `(h*w, nc)` 的矩阵，每行是一个类似 one-hot 向量（其中 `1` 由 (3) 式替代）

### 2.3.2 回归

与分类损失不同，**回归损失仅考虑正样本**

回归损失包含两部分：CIoU 损失和 DFL 损失。

CIoU 损失为 $1-CIoU$，根据前述的 CIoU 矩阵，取正样本的 CIoU 代入计算即可。

DFL 损失根据 (4) 式，也很容易计算。

CIoU 损失和 DFL 损失均使用加权平均，权重为正样本的分类 target，这在上一小节 `### 2.3.1` 已讲。

# 3. 分割任务