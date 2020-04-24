---
title: TridentNet
date: 2019-06-21 16:24:19
tags: object detection
mathjax: true
---
论文：[Scale-Aware Trident Networks for Object Detection](https://arxiv.org/abs/1901.01892)
<!-- more -->
代码：[TuSimple/simpledet](https://github.com/TuSimple/simpledet)
# 简介
目标检测通常分为：
1. one stage，如 YOLO, SSD
2. two stage，如 Faster R-CNN, R-FCN

这些方法在目标尺度变化范围较大时，均存在问题，尤其在目标尺度很小或很大时，性能较差。为了解决目标尺寸多变性的问题，有如下方法：
1. 生成多尺度 image pyramids 作为网络输入，如图 1(a)，缺点是计算量大，耗时
2. 利用网络中的不同分辨率的 feature maps，不同分辨率的 feature maps 负责检测不同尺度的目标，如 SSD
3. 2 方法中 low level 的 feature 注重于局部细节，而 high level 的 feature 因为感受野 RF 更大，则注重于整体（语义）为了补偿 low level 的 feature 所缺失的语义，FPN 在原有 bottom-up 的基础上增加 top-down pathway 和 径向连接，如图 1(b)。但是由于不同分辨率 features 来自网络不同的 layers，所以对不同尺度的目标的表征能力差异较大，所以 feature pyramids 不能认为是 image pyramids 的替代。
   
![](/images/TridentNet_fig1(a).png) <center> fig1(a)</center>
![](/images/TridentNet_fig1(b).png) <center> fig1(b)</center>
![fig1(c)](/images/TridentNet_fig1(c).png) <center> fig1(c)</center>

本文提出的新网络结构能适应不同的目标尺度，如图 1(c)，使用 trident 块生成多个尺度相关的 feature maps。trident 块的各个分支结构相同，且共享权重参数，但是由于使用了空洞卷积（膨胀系数不同），所以具有不同的 RF，每个分支负责处理一定尺度范围的目标。由于参数共享，所以 inference 阶段，可以使用一个主分支来近似 TridentNet 。

# 感受野
backbone 中的影响最终目标检测的几个设计因素为：下采样率、网络深度和感受野。更深的网络和更低的下采样率会增加网络的复杂度，但往往也有益于检测。为了研究 RF 在检测中的作用，可以将 backbone 的一些卷积层的卷积改为空洞卷积。

假设膨胀率为 $d_s$，那么一个膨胀后的 3x3 卷积的 RF 与 kernel size 为 $3+2(d_s-1)$ 卷积核的 RF 相当。记当前 feature map 相对于输入 image 的下采样率为 s，那么此时膨胀率为 $d_s$ 的卷积相较于普通卷积，其 RF 将增加 $2(d_s-1)s$，因此，如果将 n 个卷积改为空洞卷积，那么 RF 将增加 $2(d_s-1)sn$，其中，这 n 个卷积所作用的 feature map 相对于输入 image 的下采样率均为 s。

实验基于 COCO benchmark 使用 Faster R-CNN，backbone 分别使用 ResNet-50 和 ResNet-101，在 _conv4_ stage 的 residual block 上 3x3 卷积层使用空洞卷积，膨胀率在 1-3 之间。测试结果指标 AP 分别基于： a. 所有目标；b. 小目标；c. 中等目标；d. 大目标，结果如表 1，

| Backbone  |   Dilation | AP    | AP<sub>s</sub> | AP<sub>m</sub> | AP<sub>l</sub> |
| ----------|----------- | :---: | :------------: | :------------: | :------------: |
| ResNet-50 |  1         | 0.332 | __0.174__      | 0.384          | 0.464          |
| ResNet-50 |  2         | 0.342 | 0.168          | __0.386__      | 0.486          |
| ResNet-50 |  3         | 0.341 | 0.162          | 0.383          | __0.492__      |
| ResNet-101|  1         | 0.372 | __0.200__      | __0.430__      | 0.528          |
| ResNet-101|  2         | 0.380 | 0.191          | 0.427          | __0.538__      |
| ResNet-101|  3         | 0.371 | 0.181          | 0.410          | __0.538__      |

<font size=2> Table 1 COCO 数据集上具有不同 RF 的 Faster R-CNN 的检测结果</font>

从表中可见，当 RF 增加时，ResNet-50 和 ResNet-101 上的小目标的检测性能持续下降，而大目标的检测性能则越来越好。不难发现：
1. 网络的 RF 能影响不同尺度的目标上的检测性能。一个合适的 RF 是与目标尺度强相关的
2. 尽管 ResNet-101 拥有足够大的理论 RF 以覆盖大尺度（大于 96x96）的目标，但是当增大膨胀率，仍能提高大目标上的性能。这说明实际上有效 RF 比理论 RF 要小

# Trident 网络
TridentNet 包括共享权重参数的 trident 块，以及一个精心设计的与 scale-aware 训练机制。
## 网络结构
如图 2，
![](/images/TridentNet_fig2.png)

网络输入为一个单尺度的 image，然后通过并行的分支生成不同尺度的 feature maps，这些并行分支共享权重参数，但是其卷积层的空洞卷积具有不同的膨胀率。

__多分支块__ 在目标检测器的 backbone 中，使用 trident 块代替普通卷积。trident 块包含多个并行的分支，这些分支结构与原先普通卷积相同，只是膨胀率不同（普通卷积可以看作是膨胀率为 1 的空洞卷积）。

以 ResNet 为例作为 backbone，bottleneck 风格（ResNet-50, ResNet=101 等）的 residual 块包含三个卷积：1x1，3x3，1x1。trident 块则基于 residual 块构建，即，将单个 residual 块改为并行的多个 residual 块，其中每个块中 3x3 的空洞卷积的膨胀率不同。通过堆叠多个 trident 块我们可以有效的调整不同分支上的感受野 RF。通常将 backbone 中最后一个 stage 中的 residual 块替换为 trident 块，这是因为靠后的 stage 其 stride 较大，所以并行分支中的 RF 差距较大。

__分支间共享权重__ 多分支的一个显著问题是参数数量成倍增加，可能会导致过拟合，故分支间除了空洞卷积的膨胀不同，结构和参数均相同，包括每个分支的 RPN 和 Fast R-CNN head（分类预测和回归预测）。
参数共享优点有三：
1. 降低参数数量。相比于常规目标检测器，TridentNet 不需要额外的参数
2. 对不同尺度的目标，输入均通过统一的转换得到 feature maps，具有相同的表征能力。（这是与 feature pyramid 的区别）
3. 因为是多分支，相当于增加了训练参数的样本。换句话说，在不同的 RF 下，训练同样的参数以应对不同的尺度范围。

## scale-aware 训练机制
根据预先定义好的膨胀率，trident 框架将生成尺度相关的 feature maps。但是尺度不匹配可能会导致性能降级，例如表 1 中具有大膨胀率的分支检测小目标。因此，很自然地做法就是不同分支负责检测不同尺度的目标。我们提出了 scale-aware 训练机制，加强各分支对尺度认识，从而避免在不匹配的分支上训练具有极端尺度的目标（极大 or 极小）。

每个分支定义一个有效范围 $[l_i,u_i]$。训练时，某个分支上训练所使用的 proposal 和 gt box 其尺度应该落入此分支的有效范围。具体而言，某个 ROI 大小为 `(w,h)`，如果 $l_i \le \sqrt{wh} \le u_i$，那么这个 ROI 适合在分支 i 上训练。

scale-aware 训练机制可以应用于 RPN 和 Fast R-CNN 上。原先 RPN 用于判断 anchors 目标/非目标 的二值分类，以及 box 回归。在 scale-aware 训练机制下，根据 gt box 尺度决定其用在哪个分支上，然后判断这个分支上的 anchor 是否是目标或非目标。训练 Fast R-CNN head 时，每个分支根据其有效范围筛选出有效的 proposal。

## Inference 和近似
Inference 阶段，所有分支均生成检测结果，然后根据分支的有效范围筛选出有效的检测结果。然后使用 NMS 或 soft-NMS 合并多个分支的检测结果。

__快速推断近似__ 为了进一步提高速度，在 inference 阶段我们可以仅使用一个主分支来近似 TridentNet。具体来说，设置主分支的有效范围为 [0,&infin;] 以预测所有尺度的目标。例如图 2 中的三分支网络，我们使用中间分支作为主分支，因为中间分支的有效范围覆盖了大目标和小目标。使用主分支近似 TridentNet 时，没有额外的计算和参数，故与原先的 Faster R-CNN 检测时间相当，与 TridentNet 相比，性能下降较小。

# 实验
实验采用 COCO 数据集，模型训练使用 80k 训练图片和 35k 的验证图片子集（_trainval35k_），模型评估使用 5k 验证图片子集（_minival_）。

## 实现细节
使用 Faster R-CNN 的 MXNet 版本作为 baseline。网络 backbone 使用 ImageNet 进行预训练，然后迁移网络到检测数据集上微调。resize 输入 image，使得短边为 800 像素。Baseline 和 TridentNet 均进行 end-to-end 训练。我们在 8 块 GPU 上训练，batch size 为16。总共训练了 12 epochs，学习率初始值为 0.02，在第 8 个 和 第 10 个 epoch 之后分别下降 10%。使用 ResNet 的 conv4 stage 的输出作为 backbone 的 feature maps，而 conv5 stage 作为 baseline 和 TridentNet 的 rcnn head。对 TridentNet 的每个分支， 从每个 image 中采样 128 个 ROIs。若无特别说明，我们使用三分支结构作为默认 TridentNet 结构，膨胀率分别为 1，2，3.采用 scale-aware 训练机制时，设置三个分支的有效范围为 [0,90]，[30,160]，[90,&infin;]。

性能评估时采用 COCO 标准评估指标 AP，和 $AP_{50}/AP_{75}$，以及 $AP_s, AP_m, AP_l$，目标尺度范围分别为 小于 32x32, 32x32 ~ 96x96, 大于 96x96。

## 消融学习

__TridentNet 组件__ Baseline (Table 2(a)) 的评估结果分别使用 ResNet-101 和 ResNet-101-Deformable 作为 backbone。然后我们逐步在 Baseline 上应用 多分支、权重共享和 scale-aware 训练机制。
![](/images/TridentNet_fig3.png)

1. __Multi-branch__
   如 Table 2(b)，多分支版本比 baseline 的性能有所提升，尤其在大目标检测上，这种提升更加明显。这说明即使只应用最简单的多分支结构，也能受益于不同的 RF。
2. __Scale-aware__
   Table 2(d) 显示了在 Table 2(b) 多分支版本上增加 scale-aware 训练机制后的结果。在小目标检测上性能有所提升，但是在大目标检测上 $AP_s$ 值掉了。我们猜测，scale-sware 训练机制虽然能阻止分支去训练极端尺寸的目标，但也可能引入过拟合问题，因为每个分支上训练的有效样本数量减少。
3. __Weight-sharing__
   Table 2(c) 为在 多分支版本 Table 2(b) 基础上增加权重共享这一设计，Table 2(e) TridentNet 为在 Baseline 上应用以上三个设计。这两个网络的性能均得到提升，这证实权重共享是有效的。由于分支共享权重参数，所以参数的训练利用了所有尺度的目标，从而降低了 scale-aware 训练中的过拟合问题。

__分支数量__ Table 3 显示了使用 1-4 个分支时的评估结果。这里没有增加 scale-aware 训练，这是为了避免精心地调整不同分支的有效范围。Table 3 说明 TridentNet 比单分支结构（baseline）方法的评估指标高。可以注意到，四分支结构比三分支结构没有带来提升效果，所以我们选择三分支结构作为默认 TridentNet。

| Branches | AP    | AP<sub>50</sub> | AP<sub>s</sub> | AP<sub>m</sub> | AP<sub>l</sub> |
| :------: |:-----:| :-------------: | :------------: | :------------: | :------------: |
| 1        | 33.2  | 53.8            | 17.4           |  38.4          | 46.4           |
| 2        | 35.9  | 56.7            | __19.0__       |  40.6          | 51.2           |
| 3        | __36.6__  | __57.3__    | 18.3           |  __41.4__      | __52.3__       |
| 4        | 36.5  | __57.3__        | 18.8           |  __41.4__      | 51.9           |

<font size=2> Table 3 COCO _minival_ 目标检测结果。ResNet-50，使用不同分支数量</font>

其他的消融学习，如在哪个 conv stage 上使用 trident 块，和 trident 块的数量等等，以及 TridentNet 与其他 SOTA 目标检测器的结果对比，可参考原文的实验结果及说明。

# 结论
提出了 TridentNet 网络，可以生成具有相同表征能力的 scale 相关的 feature maps。提出 scale-aware 训练机制，使得不同的分支善于处理不同尺度范围的目标。快速 inference 方法使用一个主分支来近似 TridentNet，提高了检测效果（相比于 baseline），并且不引入额外的参数和计算量。