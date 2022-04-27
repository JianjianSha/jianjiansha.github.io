---
title: YOLOv4 论文解读
date: 2022-04-11 11:18:21
tags: object detection
mathjax: true
summary: 
---

论文：[YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934)

源码：[AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)

__摘要：__

有很多特性可以用来提高 CNN 准确率，在大型数据集上综合这些特征需要实际测试并对结果进行理论说明。有的特性仅仅针对某种模型或问题而设计的，而有些特性如 residual connection、BN 是普适的，我们认为这类普适的特性包括：

1. Weighted-Residual-Connections (WRC) 带权残差连接
2. Cross-Stage-Partial-connections (CSP) 跨阶段部分连接
3. Cross mini-Batch Normalization (CmBN) 跨批次归一化
4. Self-adversarial-training (SAT) 自对抗训练
5. Mish-activation
6. Mosaic 数据增强
7. DropBlock 正则
8. CIoU 损失

作者综合使用了这些特性，并获得了 SOTA 结果：MS COCO 数据集上 43.5% AP，在 Tesla V100 上实时速度为 ~65 FPS。


文章贡献点：

1. 设计了一个高效且强大的目标检测模型。
2. 验证了目标检测训练中 Bag-of-Freebies 和 Bag-of-Specials 方法的影响。
3. 修改之前的 SOTA 模型使得更加高效且适用于单 GPU 训练。

总结目标检测的组成部分：

__Input:__ Image, Patches, Image Pyramid

__Backbones:__ VGG16, ResNet-50, SpineNet, EfficientNet-B0/B7, CSPResNeXt50, CSPDarknet53

__Neck:__

1. 新增块： SPP, ASPP, RFB, SAM
2. 路径聚合块：FPN, PAN, NAS-FPN, Fully-connected FPN, BiFPN, ASFF, SFAM

__Heads:__

1. 密集预测（一阶段）

    - RPN, SSD, YOLO, RetinaNet (anchor based)
    - CornerNet, CenterNet, MatrixNet, FCOS (anchor free)

2. 稀疏预测（二阶段）

    - Faster R-CNN, R-FCN, Mask R-CNN (anchor based)
    - RepPoints (anchor free)

# 1. Bag of freebies

仅仅是改变训练策略或者说仅增加训练成本而不增加推断成本的方法，称为 Bag of freebies。目标检测中常用的 Bag of freebies 为数据增强。两种常用的数据增加方式：

## 1.1 光照变化 

亮度，对比度，hue，饱和度，噪声

## 1.2 几何畸变

随机伸缩，裁剪，翻转，旋转

## 1.3 目标遮挡
此外，还有其他研究者使用目标遮挡进行数据增强，具体方法包括：

### 1.3.1 random erase

arxiv：https://arxiv.org/pdf/1708.04896v2.pdf

源码：https://github.com/zhunzhong07/Random-Erasing

在图像内随机选择一个矩形区域，将区域内像素值改为随机值或者数据集像素均值（RGB三通道均值）

### 1.3.2 Cutout

arxiv：https://arxiv.org/pdf/1708.04552v2.pdf

源码：https://github.com/uoguelph-mlrg/Cutout

固定 size 的正方形区域，随机选择位置，然后将区域内像素 0 填充。为了降低 cutout 的影响，在 cutout 之前先将数据集归一化处理。

$$x:= \frac {x-\mu} {v}$$

其中 $x$ 是归一化后的像素值，$x:=x/255$，$\mu$ 是 ImageNet RGB 通道平均，

$$\mu=[109.9/255,109.7/255,113.8/255]$$

$v$ 是标准差，

$$v=[50.1/255, 50.6/255, 50.8/255]$$

代码片段：
```python
train_transform = transforms.Compose([])
if args.data_augmentation:
    train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(normalize)
if args.cutout:
    train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))
```

### 1.3.3 Hide-and-Seek

arxiv：https://arxiv.org/pdf/1811.02545.pdf

源码：https://github.com/kkanshul/Hide-and-Seek

将图像划分成 $S \times S$ 网格，每个网格有一定概率被遮挡，遮挡部分使用整个数据集的均值。

### 1.3.4 Grid Mask

arxiv: https://arxiv.org/abs/2001.04086

源码：https://github.com/akuxcw/GridMask

如图 1，将原输入图像乘上 mask，得到 Grid Mask 的图，
![](/images/obj_det/yolov4_1.png)

<center>图 1. Grid Mask</center>

数学表达为 

$$\tilde {\mathbf x}=\mathbf x \times M$$

其中 $\mathbf x \in \mathbb R^{H \times W \times C}$ 表示一图像，$M \in \{0,1\}^{H \times W}$。如何生成这个 binary mask $M$？如图 2，

![](/images/obj_det/yolov4_2.png)

<center>图 2. 黄色虚线表示 mask 的一个 unit</center>

使用 $(r,d,\delta_x, \delta_y)$ 4 个参数表征 $M$，$r$ 是一个 unit 内灰度边的比例（灰度边长相对于 $d$ 的比例），灰度表示 mask 像素值为 `1`，$d$ 是 unit 的长度。$\delta_x, \ \delta_y$ 是完整 unit 到图像边界的距离。

对于一个给定的 mask $M$，定义保留率 $k$ 如下：

$$k=\frac {sum(M)} {H \times W}$$

表示图像中像素值被保留的比例。$k$ 太大，CNN 可能仍会过拟合，$k$ 太小则会丢失较多信息从而导致欠拟合。忽略图像左侧和上侧的不完整 units，那么 $r, \ k$ 的关系如下，

$$k=1-(1-r)^2=2r-r^2$$

其中 $1-r$ 表示 unit 中黑块的 size。

### 1.3.5 正则

对网络中间层的 features 进行 mask，例如 Dropout，DropConnect， DropBlock

### 1.3.6 多图像

利用多个图像进行数据增强，例如 MixUp 使用两个图像分别乘以不同的系数，然后进行叠加，同时 gt labels 也根据系数进行调整。

## 1.4 GAN

style transfer GAN 用于数据增强。

## 1.5 OHEM

对于具有语义分布偏向的问题，例如数据分类不均衡，使用难负例挖掘，或者在线难负例挖掘。

Focal loss 是另一种解决分类不均衡的方法，对于少数分类的样本，其损失的系数比多数分类样本的系数大。

## 1.6 label 表征
one-hot 表征不能体现不同分类之间的关系度。[Rethinking the inception architecture for computer vision](https://arxiv.org/abs/1512.00567) 在训练过程中将 hard label 转换为 soft label，这是模型更加 robust。[Label refinement network for coarse-to-fine semantic segmentation](https://arxiv.org/abs/1703.00551) 利用知识蒸馏来设计网络以优化 label 表征。

## 1.7 目标函数
BBox 回归的目标函数通常采用 MSE 对 BBox 的中心点坐标和宽高或者左上右下坐标进行回归。这种方法将 4 个坐标独立对待分别进行回归，没有考虑到目标整体，后来有了 IoU 损失，IoU 损失具有尺度不变性这个优势。

为了进一步改善 IoU 损失，又提出了 GIoU 以及 DIoU 损失。GIoU 解决了 IoU 损失与 BBox 参数的距离损失不等价问题，且对无重叠的 BBox 也可进行优化（详情可参考这篇博文 [GIoU](https://jianjiansha.github.io/2019/06/13/obj_det/GIoU/)）。

DIoU 则额外考虑了目标中心点的距离。CIoU 则同时考虑了重叠区域面积，中心点距离以及 aspect ratio。

如图 3，

![](/images/obj_det/yolov4_3.png)

<center>图 3. IoU, GIoU, DIoU 对比。绿色为 gt box，红色为预测 box</center>

$$L_{DIoU}=1-IoU+\frac {\rho^2(\mathbf b, \mathbf b^{gt})} {c^2}$$

其中 $\mathbf b, \mathbf b^{gt}$ 是预测 box 和 gt box 的中心点，$\rho$ 是欧氏距离，$c$ 是包含预测box 和 gt box 的最小框的对角线长度。

$$L_{CIoU}=1-IoU+\frac {\rho^2(\mathbf b, \mathbf b^{gt})} {c^2}+\alpha v$$

其中 $\alpha$ 是一个正 trade-off 参数，$v$ 是预测 box 与 gt box aspect ratio 一致性的测度，
$$\alpha=\frac v {(1-IoU)+v}$$

$$v=\frac 4 {\pi^2}(\arctan \frac {w^{gt}}{h^{gt}} - \arctan \frac w h)^2$$

# 2. Bag of specials

部分模块和 post-processing 方法仅增加推断成本，但是可以显著提高准确性，称其为 “Bag of specials”。通常增加的这些模块是为了增强某个模型的某些属性，例如：增大感受野，增强特征整合能力等。post-processing 方法则是用于筛查模型的预测结果。

## 2.1 感受野
常用的用于增大感受野的模块有 SPP, ASPP, RFB 等。

SPP 在 backbone 之后增加一个 Spatial Pyramid Matching，将 backbone 的输出特征 features 分别 pooling 为 `1x1`, `2x2`, `4x4` 三种空间大小的特征，然后再 flatten 并 concatenate 送入全连接层。

## 2.2 attention

注意力模块也常用在目标检测种，分为 channel-wise 注意力和 point-wise 注意力，例如 Squeeze-and-Excitation, Spatial Attention Module。

## 2.3 特征整合
关于特征整合，早期的实现是 skip connection, 或者 hyper-column，融合高低层特征。另外 FPN 系列的多尺度特征上采用轻量检测 heads，例如 SFAM, ASFF, BiFPN 等。

## 2.4 激活函数
部分研究人员注意力集中在激活函数上，例如 ReLU，LReLU，PReLU，ReLU6， SELU, Swish, hard-Swish 以及 Mish 等。

## 2.5 post-processing
非极大抑制 NMS

soft NMS（参考这篇博文中的 [Soft-NMS](https://jianjiansha.github.io/2019/06/24/cv/cv-mtds/)一节 ）核心思想是对于两个预测 box，其 IoU 超过阈值时较小得分的那个预测 box，其得分并不直接置 0，而是改为一个更小的值。

DIoU NMS，在 soft NMS 的基础上考虑中心点的距离信息。



# 3. 方法论
## 3.1 框架选择
在 network 输入大小，卷积层数量，参数量（`filter_size * filter_size * filters * channel / groups`）和 layer 输出数量（`filters`） 之间寻找一个最佳平衡。

在 ImageNet 数据集上（图像分类任务） CSPResNext50 比 CSPDarknet53 效果好，但是在 MSCOCO 数据集上（目标检测任务）后者要好些。

分类任务的最佳模型在检测任务上不一定是最佳。与分类不同，检测需要：
1. 更大的网络输入 size，用于检测小尺寸目标
2. 更多 layers。由于网络输入 size 的增大，这就要求更大的感受野以足够覆盖网络输入 size。
3. 更多的参数，使得拥有检测单个 image 中的多个不同 size 目标的能力。

不同 size 的感受野 RF：
1. RF 达到目标 size - 可以看到整个目标
2. RF 达到网络输入 size - 可以看到目标周围的上下文

YOLOv4 整个网络概述：
1. 在 CSPDarknet53 上增加 SPP 模块，以增大感受野
2. PANet 路径聚合
3. YOLOv3（基于 anchor）的检测 head

## 3.2 BoF 和 BoS

### 3.2.1 激活函数

ReLU，Leaky-ReLU，parametric-ReLU，Swish，Mish

PReLU 和 SELU 难以训练，故不使用。ReLU6 用于量化网络也不使用。

### 3.2.2 回归损失

MSE，IoU，GIoU，CIoU，DIoU

### 3.2.3 数据增强
CutOut，MixUp，CutMix

### 3.2.4 正则

DropOut，DropPath，Spatial DropOut，DropBlock

由于先前论文研究认为 DropBlock 效果最好，故作者也使用了 DropBlock。

### 3.2.5 归一化

BN，Filter Response Norm，Cross-Iteration BN

不使用 syncBN ，因为为了简化训练，仅使用单 GPU 训练。

### 3.2.6 Skip 连接

Residual 连接，加权 residual 连接，多输入加权 residual 连接，Cross stage partial（CSP）连接

## 3.3 其他改进

1. 介绍了新数据增强方法 Mosaic，以及 self-adversarial training（SAT）
2. 使用遗传算法时，使用最优超参数
3. 对现有方法的改进，使得检测更加高效，修改了 SAM，PAN，CmBN（Cross mini-batch BN）

    修改 SAM 从 spatial-wise 注意力到 point-wise 注意力，如图 4，

    ![](/images/obj_det/yolov4_4.png)

    将 PAN shortcut 连接替换为 concatenation，如图 5，

    ![](/images/obj_det/yolov4_5.png)

## 3.4 YOLOv4

- Backbone: CSPDarknet53
- Neck: SPP, PAN
- Head: YOLOv3

总结 YOLOv4 的 BoF 和 BoS

- backbone BoF：CutMix , Mosaic 数据增强，DropBlock 正则，分类标签平滑
- backbone BoS：Mish 激活，CSP 连接，多输入加权 residual 连接
- detector BoF：CIoU 损失，CmBN 归一化，DropBlock 正则，Mosaic 数据增强，SAT，grid 敏感度消除，单 GT 多 anchor，余弦退火调度，最优超参数，随机训练 shapes。（好多不懂是啥）
- detector BoS：Mish 激活，SPP 模块，空间注意力模块（SAM），PAN 路径聚合模块，DIoU-NMS

# 4. 实验

