---
title: Anchor-free Object Detection
date: 2021-02-25 12:31:28
tags: object detection
---
# FCOS
每个 scale 的 detection head 输出的 feature maps 上的每一 location 处预测 `C` 个分类和 `4` 个坐标，以及一个 center 得分，对每种分类采用二值分类。某个 location 位于 gt box 内，则为正例样本，否则为负例样本，box 回归 target 记为 $\mathbf t^{*}=(l^{*}, t^{*}, r^{*}, b^{*})$，某一 location `(x,y)` 位于 gt box $B_i=(x_0^{(i)}, y_0^{(i)}, x_1^{(i)}, y_1^{(i)})$ 内部，那么有
$$l^{*}=x-x_0^{(i)}, \quad t^{*}=y-y_0^{(i)}$$
$$r^{*}=x_1^{(i)} - x, \quad b^{*}=y_1^{(i)} - y$$

损失为
$$\begin{aligned} L({\mathbf p_{x,y}}, {\mathbf t_{x,y}})&=\frac 1 {N_{pos}} \sum_{x,y} L_{cls}(\mathbf p_{x,y}, c_{x,y}^{*}) \\ &+ \frac {\lambda} {N_{pos}} \sum_{x,y} \mathbb I(c_{x,y}^{*}>0) \cdot L_{reg}(\mathbf t_{x,y}, t_{x,y}^{*})\end{aligned}$$

其中，$c_{x,y}^{*}$ 表示 location `(x,y)` 处所属分类，如果是负例，$c_{x,y}^{*}=0$，如果是正例，那么 $c_{x,y}^{*}$ 为fg 分类id。分类损失 $L_{cls}$ 使用 Focal Loss。坐标损失 $L_{reg}$ 使用 IOU Loss。

如果某个 location 位于多个 gt box 交叠的区域，那么这个 location 该回归哪个 gt box 呢？ 答案是使用 multi-level 预测，见如下。
## FPN for FCOS
引入 FPN，使用 multi-level 的 feature maps，分别记为 ${P_3, P_4, P_5, P_6, P_7}$，其中 $P_3, P_4, P_5$ 由 backbone 中的 $C_3, C_4, C_5$ 经 `1x1` conv 输出得到。$P_6, p_7$ 则分别在 $P_5, P_6$ 上使用 stride=2 的 conv 得到，各 level feature 上的 stride 分别为 `8,16,32,64,128`。

在不同 scale 的 feature 上限制所预测的 box 的回归值范围，具体而言，对于 $P_i$，如果某 location 满足 $\max(l^{*},t^{*},r^{*},b^{*})>m_i$ 或者 $\max(l^{*},t^{*},r^{*},b^{*}) < m_{i-1}$，那么这个 location 为负例，不计入坐标回归损失 $L_{reg}$。本文中 $m_2=0, \ m_3=64, \ m_4=128, \ m_5=256, \ m_6=512, \ m_7=\infty$（这是借鉴了 anchor based 中的 anchor 的 base scale 为 8）。

这样，不同的 level 的 feature 负责预测不同 size 范围的目标。同一 location 如果有多个目标，通常，这些交叠的目标大小不在同一个范围，于是降低了 ambiguous location，如果仍然存在 location 属于多个大小差不多目标，那么人为的将此 location 用来预测 size/area 最小的那个目标。

与其他基于 FPN 的目标检测器一样，不同 level 的 检测 head 分支上参数共享。观察到不同 level
的 feature 回归不同范围的目标（例如 $P_3$ 回归范围为 $[0,64]$），那么不同 level 的检测 head 如果完全相同，会显得不合理，所以，比起 single scale FCOS 中使用的 $exp(x)$（因为回归 target 为正，所以需要使用 $exp(x)$ 处理一下），在 multi scale FCOS 中，改为 $exp(s_i x)$，这里的 $s_i$ 是需要训练学习的参数，以自动调整其大小。
## Center-ness for FCOS
实验发现，有大量低质量的预测 box，这是有距目标中心较远的 location 预测出来的结果。如何抑制它们？

增加一个 single-layer 分支，与分类分支并列，如图 1，

![](/images/obj_det/anchor_free_fig1.png)<center>图 1</center>

新增分支用于预测 location 的中心度 “center-ness”：是目标中心的置信度，用 location 与所属目标中心的归一化距离来表示，记回归目标为 $\mathbf t^{*}=(l^{*},t^{*},r^{*},b^{*})$，center-ness 目标为
$$\text{centerness}^{*}=\sqrt{\frac {\min(l^*, r^*)} {\max(l^*,r^*)} \times \frac {\min(t^*, b^*)} {\max(t^*,b^*)}}$$
使用平方根是为了降低 center-ness 值的衰减速度，center-ness 值范围位于 $[0,1]$，使用 BCE loss 进行训练。

测试阶段，__最终的得分计算为 center-ness 与分类得分 相乘__，以最终得分进行预测 box 的 ranking，然后经 NMS，过滤掉那些低质量的预测结果。

另一种 center-ness 的替换方案是，仅仅使用 gt box 中心区域的 location 作为正例，代价是需要引入一个额外参数来指定中心区域的大小。当然，也可以两者结合使用。

# CornerNet
anchor based 目标检测的缺点：1. 需要大量的 anchor，导致正负例不均衡，训练速度慢。2. 需要很多超参数（例如，anchor 数量、大小以及 aspect ratio 等），在 multi-scale feature 情况下，超参数更多。

本文介绍了一个 anchor free 的目标检测方法，检测目标的左上和右下两个角点。为什么检测角点呢？

1. box 中心难以定位，因为中心依赖于目标的 4 个边坐标，而定位一个角点仅需要两个边坐标，加上本文提出的另一个设计 `corner pooling`，利用角点的定义，例如左上角，`corner pooling` encoding 当前某 location 右侧的 feature vector，以及下方的 feature vector，这种 encoding 使得角点预测更加准确。
2. 密集离散化 box 时，记 feature 的 size 为 `(w,h)`，那么所有可能的角点的数量为 O(wh)，而 anchor box 的可能性则非常大，根据排列组合原理， anchor 的宽有 w 种可能，高有 h 种可能，故 anchor 的 size 有 wh 种可能，然后分布在 feature 上时，每个 anchor 有 wh 种可能，故一共 $O(w^2h^2)$。当然如果考虑到 anchor 不要超出 feature map 之外，那么 anchor 的宽为 1 时，有 w 种可能，宽为 2 时，有 w-1 种可能，... 宽为 w 时，有 1 种可能，一共有 $w+(w-1)+\cdots +1$，同样地，高也一样，所以分布在 feature 上各种 anchor 的可能性一共有 $(1+2+\cdots+w)(1+2+\cdots+h)=\frac {w(w+1)h(h+1)} 4$。当然，anchor based 检测方法为了缩减 anchor 数量，在 feature 上每个 location 使用不同 scale 和 aspect ratio 的 anchor 共 K 个，故此时一共有 $Kwh$ 个。

## 网络框架
base ConvNet 采用 hourglass network，后接两个模块，分别用于左上角点和右下角点。每个模块有独立的 corner pooling 模块（因为左上和右下角点的 corner pooling 的方向不同），hourglass network 的 feature 经过 pooling 之后，得到 heatmaps，embeddings 和 offsets。

__heatmap:__

两组 heatmaps，每组 heatmap 有 C channels，其中 C 为分类数量，heatmap 的大小为 $H \times W$，heatmap 表示某个 location 是角点且属于某个分类的得分，可见 heatmap 是 binary mask。

对某个 corner 而言，只有一个 location 对应到它，这个 location 作为正例，其他 location 相对这个 corner 而言均为负例，这里不对所有负例等值惩罚，而是对这个 corner 一定半径范围的负例 location 降低惩罚，这是考虑到，即使是负例 location，如果很靠近对应的 corner，这些负例构成的预测 box 依然可以很好的覆盖 gt box。半径大小由对应的 gt box 确定，根据 IOU 的阈值下限 t（文中 t 设为 0.3）。如何降低惩罚？

使用一个二维高斯函数，随着距离增大平滑的递减，
$$f_{cij}=\begin{cases} e^{-\frac {x^2+y^2}{2 \sigma^2}} & x^2+y^2 \le r^2 \\ 0 & \text{otherwise} \end{cases}$$
其中，下标 `(i,j)` 表示 location （样本）的坐标，$x, y$ 表示 location 据对应 corner 的横纵坐标差，$c$ 表示 gt 的分类。$\sigma=r/3$ 用于控制衰减速度。$r$ 为半径。

heatmap 表示 location 是 corner 且属于某一分类的得分，采用 focal loss，考虑以上二维高斯函数作为惩罚因子，
$$L_{det}=-\frac 1 N \sum_{c=1}^C\sum_{i=1}^H \sum_{j=1}^W \begin{cases} (1-p_{cij})^{\alpha} \log p_{cij} & y_{cij}=1 \\ (1-f_{cij})^{\beta}p_{cij}^{\alpha} \log (1-p_{cij}) & o.w. \end{cases}$$
其中，$p_{cij}$ 表示 heatmap 上某 location 预测值。$y_{cij}$ 表示 heatmap 上某处的 gt target 值。N 表示一个 image 上 object 的数量。$\alpha=2, \ \beta=4$ 为超参数，控制各项损失贡献。从上式可见，仅对 corner 一定半径范围内的 negative location 降低了惩罚。

__offset:__

由于下采样，hourglass 的 feature size 较原 image size 小，记下采样率为 n，那么 gt corner $(x,y)$ 到 heatmap 上的位置为 $(\lfloor x/n, y/n \rfloor)$，再映射回 image 上是，坐标精度有所损失，损失范围为 $[0,n)$，当目标size 较小时，影响 IOU，所以增加一个坐标偏差的预测，gt offset 为
$$\mathbf o_k=\left(\frac {x_k} n - \lfloor \frac {x_k} n \rfloor, \frac {y_k} n - \lfloor \frac {y_k} n \rfloor \right)$$
其中 $k$ 表示第 k 个 gt corner，预测的 offset 记为 $\hat \mathbf o_k$，使用 smooth L1 损失。

__embedding:__

左上 corner 和右下 corner 是分开预测的，如何确定哪个左上和哪个右下是一对呢（来自同一个 gt box）？

为每个 corner 生成 embedding vector，如果某一对 corner 来自同一个 gt box，那么它们的 embedding 应该相似，例如 vector 的 L1 范数足够小。

由于 embedding 没有 target 值，作者使用 `pull` 和 `push` 两种损失来学习，记 $e_{tk}$ 为第 k 个目标左上 corner 的 embedding，$e_{bk}$ 为第 k 个目标右下 corner 的 embedding，那么损失为 
$$L_{pull}=\frac 1 N \sum_{k=1}^N [(e_{tk}-e_k)^2+(e_{bk}-e_k)^2]$$
$$L_{push}=\frac 1 {N(N-1)} \sum_{k=1}^N \sum_{j=1, j\ne k}^N \max (0, \Delta-|e_k-e_j|)$$
$$e_k=\frac {e_{tk}+e_{bk}} 2$$
其中，N 为 image 中目标数量，$\Delta=1$ 表示两个来自不同目标的 corner 的 embeding 的最大区分度。

测试阶段，分别得到 K 个 top-left 角点和 K 个 bottom-right 角点，一共 $K^2$ 个 pair 的可能组合，计算每个 pair 的 embedding 的 L1 范数，如果超过一个阈值（设为 0.5），那么这个 pair 不成立，当然还有其他筛选条件，例如来自 heatmap 的得分筛选等，然后使用 soft nms，去除冗余检测结果。

## Corner Pooling
这个模块比较关键，从 hourglass 的输出特征到上面说的三个 feature `heatmap, offset, embedding`，中间经过了 Corner Pooling 模块。

hourglass 输出为 $f_t, f_l, f_b, f_r$ 分别表示 上左下右 四个 side，其中 $f_t, f_l$ 经过左上角点预测分支，$f_b, f_r$ 经过右下角点预测分支。以左上角点预测分支为例，记 corner pooling 的输入 feature size 为 $H \times W$，某一 location `(i,j)` 在 $f_t$ 上，经 corner pooling 后，从 `(i,j)` 到 `(H,j)` 之间的特征向量（从这个 location 开始垂直向下看）执行 max 操作得到输出值 $t_{ij}$，同理，从这个 location 向右看的特征向量经过 max 操作得到输出值 $l_{ij}$，
$$t_{ij}=\begin{cases} \max[f_t(i,j), t_{(i+1)j}] & i < H \\ f_t(H,j) & i=H \end{cases}$$
$$l_{ij}=\begin{cases} \max[f_l(i,j), l_{i(j+1)}] & i < H \\ f_l(i,W) & i=H \end{cases}$$
对于 $f_t$，从下到上递归执行，对于 $f_l$，从右到左递归执行。

预测模块的网络结构如图 2，
![](/images/obj_det/anchor_free_fig2.png)<center>图 2. backbone 后接一个修改过的 residule module，下方是 shortcut 分支，上方是 2x conv，分别表示 left-most 和 top most，然后分别经 corner pooling 后，再 element-wise add，然后经 conv ，再与 shortcut 分支进行 merge（element-wise add)，然后经过右边的网络</center>

## Hourglass Network
之前的 backbone 都是 VGG，ResNet 比较常见，也有用 densenet 的，但是 hourglass 网络还没见过，所以这里简单介绍一下。

Hourglass network 是由一个或多个 hourglass 模块组成的全卷积网络。Hourglass 模块先将 input feature 下采样，然后在上采样到原来的 resolution，由于下采样中的 maxpooling 损失了一些细节信息，所以使用 skip layer 将信息带到上采样的 feature 中。堆叠多个 hourglass 模块可捕获更高 level 的信息（这些信息的语义性更强）。

本文中，下采样使用 stride=2 的 conv 代替 maxpooling。

总损失为 
$$L=L_{det}+\alpha L_{pull} + \beta L_{push} + \gamma L_{off}$$