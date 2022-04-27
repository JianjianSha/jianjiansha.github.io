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
$$L_{cls}=L_{cls}(p,u)=-\log p_u$$
上式为 log loss，proposal 对应的 gt 分类为 u，$p_u$ 为 proposal 分类为 u 对应的置信度（分类得分）。
$$L_{loc}=L_{loc}(t^u,v)=\sum_{i \in \{x,y,w,h\}} smooth_{L_1}(t_i^u,v_i)$$
上式为 smooth L1 loss，$t_u$ 为在分类 u 下的 bbox 的四个偏移值，v 表示 gt box 相对 proposal 的偏移 target。  
__mask 分支会为每个 RoI 生成 $Km^2$ 维输出向量__，然后对这个输出向量应用 pixel-wise sigmoid，表示 K 个 binary mask，每个 mask 分辨率为 $m \times m$（m 值参见下文图 4），这里 K 表示所有分类数量，定义 $L_{mark}$ 为平均二值交叉熵损失，记 RoI 的 gt 分类为 k，$L_{mark}$ 仅由第 k 个 binary mask 计算得到，其他 K-1 个 binary mask 均不参与 $L_{mark}$ 的计算，
$$L_{mark}=-\frac 1{m^2} \sum_{i=1}^{m^2} \sum_{j=0}^1 [t_i=j] \cdot \log f(s_i^j)=-\frac 1{m^2} \sum_{i=1}^{m^2} [t_i \cdot \log f(s_i) + (1-t_i) \cdot \log (1-f(s_i))]$$
其中 $f(\cdot)$ 表示 sigmoid。

__Mask Representation:__ 对单个 RoI 而言，无论其大小，对应的分类和 bbox 偏移这两个输出都是固定长度，可由 fc 层输出得到，而 mask 则以 pixel-to-pixel 方式表征 RoI 中目标的空间布局，所以适合使用卷积。事实上，我们正是使用了全卷积网络 FCN 来为每个 RoI 生成 $m \times m$ 空间尺寸的 mask。然而需要注意的是，pixel-to-pixel 的方式要求 RoI 特征能如实地保留每个像素的空间对应关系，于是我们提出 RoIAlign 来解决这个问题。

__RoIAlign:__ RoIPool 是从 RoI 中抽取固定长度特征（例如 $7\times 7$）的标准方法，首先将浮点数 RoI 量化成整数粒度的 feature map，然后将量化后的 RoI 切分得到一系列空间 bins ，每个空间 bin 的大小也是浮点数，所以每个空间 bin 的位置也需要量化，然后将其中的像素值聚合得到这个空间 bin 的值，一般使用最大值池化进行聚合。

可见前后有两次量化过程，第一次量化是在将 RoI 的坐标 x 从输入 image 平面上映射到特征平面上，在 Faster R-CNN 中，这个特征的 stride 为 16，所以 RoI 在特征平面上的坐标为 $[x/16]$，其中 $[\cdot]$ 表示四舍五入成整数；第二次量化是在计算空间 bin 位置时，假设 RoI 为 $(x_1,y_1,x_2,y_2)$（通常均为浮点数，因为 RPN 中对 anchor 位置进行偏移得到 RoI），一共将 RoI 划分为 7x7 个空间 bins，经过第一次量化后特征平面上 RoI 表示为，
$$x_1'=[x_1/16] \quad y_1'=[y_1/16]
\quad x_2'=[x_2/16]
\quad y_2'=[y_2/16]$$
RoI 的大小 和 空间 bin 的大小分别为
$$w'=x_2'-x_1'+1
\quad h'=y_2'-y_1'+1
\\\\ w^b=w'/7 \quad h^b=h'/7$$
对于第 (i,j) 个 bin，其位置为
$$x_1^b=\lfloor j \cdot w^b\rfloor \quad y_1^b=\lfloor i \cdot h^b\rfloor \quad x_2^b=\lceil (j+1) \cdot w^b\rceil \quad y_1^b=\lceil (i+1) \cdot h^b\rceil$$

其中 $0 \le i<7, \ 0\le j<7$。（当然还需要对 bin 的位置十分越界进行检查，这里略）

两次量化使得 RoI 与抽取到的特征不对齐，这对 pixel-to-pixel 的 mask 而言是非常不利的，所幸 RoIAlign 可以解决这个问题。使用 RoIAlign 代替 RoIPool，避免量化操作，如图 3，
![](/images/mask-rcnn_fig3.png)

<center>图 3 RoIAlign 示意图，图中黑矩形框表示 backbone 输出 feature maps 上的一个 RoI，具有 2x2 个 bin，实际是 7x7 个 bin，这里仅作示例</center>

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
有关 mask 分支，这里详细说明一下处理过程。如图 4，mask 分支输出大小为 $(R,K,m,m)$，根据 bbox 回归得到预测 box 的坐标数据，数据块大小为 $(R,4)$，其中 R 为检测到的所有预测 box 的数量，K 为目标分类数量，$mxm$ 为 mask 的空间大小。对于第 i 个 目标，$0 \le i < R$，记预测 box 位置为 $(x_1,y_1,x_2,y_2)$，对于第 k 个分类，记对应的 mask map 为 $M_i^k$，
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
