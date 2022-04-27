---
title: GA-RPN
date: 2019-06-25 17:01:57
tags: object detection
mathjax: true
---
论文：[Region Proposal by Guided Anchoring](https://arxiv.org/abs/1901.03278)
<!-- more -->
目标检测中，通常使用 anchor 来生成 proposal（two-stage）或者直接对 anchor 进行分类和回归（one-stage）。以 two-stage 的 Faster R-CNN 为例，先在 feature map 上生成密集均匀分布的 anchors，然后对其进行二分类预测及坐标回归得到 proposals，最后再对 proposals 进行分类和坐标回归微调。

合理的 anchor 设计遵循两个通用规则：alignment 和 consistency：
1. anchor 中心应与 feature map 的像素点对准
2. 感受野 RF 和语义范围应与 anchor 的尺度和形状保持一致

滑窗就是一种简单且被广泛采用的 anchor 生成机制，大多数目标检测方法均采用滑窗来生成均匀密集的 anchors，即，在 feature map 上每个像素点位置按预先设置的 scale 和 aspect ratio 生成 k 个 anchors。然而，这种 anchor 生成机制的困难在于：
1. 对不同的检测问题，需要预先精心设计适合的 scale 和 aspect ratio，否则会影响检测性能指标。
2. 为了得到高的 recall，需要生成大量的 anchors，这导致大部分 anchors 为负例（non-object），同时，大量的 anchors 增加计算量

本文提出一个有效的 anchor 生成方法，此方法受以下观点启发：
1. 根据观察，image 中的目标位置不是均匀分布的
2. 目标的 scale 与目标位置和 image 内容相关
   
于是我们的稀疏非均匀 anchor 生成方法步骤为（guided anchoring）：
1. 确定可能包含目标的子区域
2. 根据子区域位置确定其 shape

可学习的 anchor shape 虽然合理，但是打破了前述的 consistency 规则，即，学习到（动态生成）的 anchor shape 可能与 RF 和 语义 scope 不一致。由于现在的 scale 和 aspect ratio 是可变的而非固定的，所以 feature map 上不同的像素点的 anchor shape 也不尽相同，需要学习适配 anchor shape 的表征以维持 consistency 原则。为了解决此问题，我们介绍了一个有效的模块：基于 anchor shape 来修改 features 使其适配，此即 feature adaptation 机制。

使用前述的 guided anchoring 和 feature adaptation 机制，我们制定了 Guided Anchoring Region Proposal Network （GP-RPN）。由于动态预测 anchors，recall 值比常规 RPN（baseline，使用滑窗生成密集均匀分布的 anchor）高了 9.1%，而 anchors 数量下降了 90%。通过预测得到 scale 和 aspect ratio 而非固定的预设值，我们的检测方法能更加有效地处理 aspect ratio 小于 1/2 或 大于 2 的宽/高目标。除了用于生成 region proposals， guided anchoring 方法可集成到任何依赖 anchor 的检测器中（比如 SSD，直接拿 anchor 分类和回归得到最终的预测 box）。guided anchoring 机制能使得各种目标检测器获得一致的性能提升，比如在 COCO 数据集上， GA-Fast-RCNN，GA-Faster-RCNN 和 GA-RetinaNet 的 mAP 比相应的使用滑窗的 baseline 分别提升了 2.2%, 2.7% 和 1.2%。

# Guided Anchoring
目标的 location 和 shape 可以使用 (x,y,w,h) 来刻画，其中 (x,y) 是中心点的坐标，(w,h) 表示宽高。给定一个 image $I$，那么目标 location 和 shape 遵循如下分布：
$$p(x,y,w,h|I)=p(x,y|I)p(w,h|x,y,I)$$
这种因式分解基于如下两点：
1. 给定 image，目标仅位于某些特定区域
2. shape 即 scale 和 aspect ratio 与 anchor 的位置有关
   
根据以上公式，我们的 anchor 生成模块如图 1，

![](/images/GA-RPN_fig1.png) <center>Fig 1 框架结构。每个feature map 均使用 anchor 生成模块，模块中有两个分支，分别预测 anchor 位置和 shape。应用 feature 适配模块到 feature map 上得到新的 feature map，使其注意到 anchor</center>

给定 image $I$，首先得到 feature map $F_I$，在 $F_I$ 上 位置预测分支生成一个概率 map，表示每个位置处存在目标的概率，shape 预测分支生成位置相关的 shape，即预测每个位置的 w,h。使用一个概率阈值，选择大于阈值的位置，以及这些位置上最有可能的 shape，从而生成 anchor。考虑到 anchor 的 shape 可变，不同位置处的 feature 应该捕获不同范围内的视觉内容，所以我们进一步引入了特征适配模块，根据 anchor 的shape 使 feature 适配。

由于最近的研究表面，使用不同 leve 的 feature maps 有助于目标检测，如 FPN 和 RetinaNet，所以如图 1，我们也使用了多 level 的 anchor 生成机制，需要注意的是，不同 level 的 anchor 生成分支所用的参数是相同的。

## Anchor 位置预测
anchor 位置预测分支生成的概率 map $p(\cdot|F_I)$ 与 feature map $F_I$ 大小相同，其上每点位置的概率值 $p(i,j|F_I)$ 对应原输入 image $I$ 上位置 $((i+\frac 1 2)s,(j+\frac 1 2)s)$，其中 s 是  $F_I$ 相对于 $I$ 的步幅，即两个相邻 anchor 中心点的距离，$p(i,j|F_I)$ 表示在 $F_I$ 位置 (i,j) 处是某个目标中心的概率。

使用一个子网络 $\mathcal N_L$ 来预测得到 $p(i,j|F_I)$，$\mathcal N_L$ 组成包括一个 1x1 的卷积和一个 element-wise 的 sigmoid 函数。当然更复杂的 $\mathcal N_L$ 可以使得预测更加准确，但是为了平衡计算效率和准确性，我们仍然采用当前 $\mathcal N_L$ 的组成。

预定义一个概率阈值 $\epsilon_L$，概率 map 上小于阈值的位置均被过滤掉，也就是说，过滤掉以这些位置为中心的 region（由于当前只考虑了 anchor 中心，尚未考虑 shape，所以此时称 region），这可以过滤掉 90% 的 region 且能同时维持相同的 recall（与普通的 RPN 相比）。如图 4(b)，像天空和大海所在的 region 均被排除，而集中于人和冲浪板。由于不需要考虑那些排除掉的 region，我们将卷积替换为 masked convolution 使推断过程更加高效。

## Anchor shape 预测
确定了 anchor 位置之后，下一步就是确定 anchor 的 shape。如图 1，此预测分支与传统的 bbox 回归预测不同，因为此分支不改变 anchor 的位置，所以不会打破前述 alignment 原则。给定 $F_I$，此分支预测每个位置上的最佳 shape (w,h)，这个最佳 shape 是指 anchor 与最近的 gt box 有最高覆盖度。

由于 w,h 的范围较大，直接预测这两个值不稳定，故做如下转换将输出值域控制在 [-1,1] 这样一个较小的范围内，
$$w=\sigma \cdot s \cdot e^{dw}, \quad h = \sigma \cdot s \cdot e^{dh}$$
于是 shape 分支预测输出为 (dw,dh)，其中 s 为 $F_I$ 相对于 $I$ 的步幅，$\sigma$ 为经验尺度因子，我们实验中 $\sigma=8$。使用子网络 $\mathcal N_S$ 得到 shape 预测，$\mathcal N_S$ 组成包含一个 1x1 卷积核输出通道为 2 的卷积层，以及一个 element-wise 转换层，前者生成的 2 通道分别对应 dw map 和 dh map，后者实现上式转换得到 w map 和 h map。

以上 anchor 生成模块的设计与传统的 anchor 生成机制（滑窗）有本质不同，使用我们这里的 anchor 生成机制，每个位置仅一个 anchor，其 shape 动态预测得到，而传统的滑窗根据不同 scale 和 aspect ratio 每个位置生成 k 个 anchor。实验证明，由于我们的 anchor 生成机制中，shape 与 location 有密切关联，所以可以获取更高的 recall，且由于不是预设固定的 aspect ratio 而是动态预测 shape，我们的 anchor 机制可以捕获那些极高或者极宽的目标。

## Anchor-Guided 特征适配
传统的 RPN 或者 one-stage 检测器采用滑窗机制生成 anchors 均匀分布在 feature map 上，每个位置处 anchor 的 shape/scale 均相同，所以 feature map 能学习到一致的表征。但是在我们的 anchor 机制中，由于 shape 任意可变的，所以不适合像传统方法那样在 feature map 上使用全卷积分类器对 anchor 进行分类（例如 RPN 的二分类或者 one-stage 的前景分类）。最好的做法是，大 anchor 的 feature 应该使用大 region 的内容，小 anchor 的 feature 则使用小范围内容。于是我们进一步提出 anchor-guided 特征适配模块，根据 anchor shape 对 feature 进行转换以使其适配，如下，
$$\mathbf f_i'= \mathcal N_T(\mathbf f_i, w_i,h_i)$$
其中，$\mathbf f_i$ 是 位置 i 处的 feature，$(w_i,h_i)$ 是此处的 anchor shape。由于此特征转换是位置相关的，所以我们采用 3x3 的可变形卷积来实现 $\mathcal N_T$，如图 1，首先根据 anchor shape 分支的输出预测 offset，然后应用可变形卷积到原始 feature map 上，在转换后的 feature map 上，我们可以按传统方法进行分类和 bbox 回归。

## 训练
### 联合目标函数
本文提出的网络框架使用多任务损失进行端到端优化，除了传统的分类损失和回归损失，还包括 anchor 的位置损失和 shape 损失，目标函数的优化使用如下联合损失：
$$\mathcal L=\lambda_1 \mathcal L_{loc}+ \lambda_2 \mathcal L_{shape} + \mathcal L_{cls} + \mathcal L_{reg}$$

### Anchor location targets
为了训练 anchor 位置分支，对每个 image 我们需要一个 二值 label map，每个像素点值 1 表示有效位置，0 表示无效位置。这个二值 label map 根据 gt box 得到。我们希望在目标中心的周围附近放置较多的 anchor，而远离目标中心的位置则放置较少的 anchor。首先，将 gt box $(x_g,y_g,w_g,h_g)$ 映射到 feature map $F_I$上得到 $(x_g',y_g',w_g',h_g')$，然后使用 $\mathcal R(x,y,w,h)$ 表示一个矩形区域。Anchors 将被放置到 gt box 中心的附近以得到较大的 IOU，对每个 gt box 定义如下三种类型的矩形区域：
1. 中心区域 
   
   $CR=\mathcal R(x_g',y_g',\sigma_1 w_g', \sigma_1 h_g')$，此区域中心与 gt box 中心重合，宽高分别是 gt box 宽高的 $\sigma_1$ 倍。CR 内像素点值为 1（positive）

2. 忽略区域
   
   $IR=\mathcal R(x_g',y_g',\sigma_2 w_g', \sigma_2 h_g') \setminus CR$，其中$\sigma_2 > \sigma_1$，IR 内的像素点被标记为 `ignore`，不参与训练，这一点类似于 Faster R-CNN 中训练 RPN，anchor 与 gt box 的 IOU 大于 0.7 时为标记为 1 positive，小于 0.3 时标记为 0 negative，而位于 `[0.3,0.7]` 范围则标记为 -1，标记为 -1 的 anchor 不参与训练

3. 外围区域
   
   $OR=F_I \setminus IR$，OR 内的像素点值为 0（negative）

由于我们使用多 level features，每个 level 的 feature map 应该仅瞄准特定 scale 范围的目标，故对某个 feature map 匹配的 scale 范围内的目标，我们设置相应（这些目标）的 CR，对 IR 也是同样处理，如图 2。如果多个目标重叠，那么 CR 抑制 IR， IR 抑制 OR，显然这是合理的，因为 CR, IR, OR 优先级应该逐步下降，才能保证 recall。由于 CR 只占 feature map 中的一小部分，所以我们采用 Focal Loss 平衡正负例来训练 anchor 位置分支。
![](/images/GA-RPN_fig2.png)

### Anchor shape targets

分两步来决定最佳 shape target：
1. 将 anchor 与某个 gt box 匹配起来
2. 预测 anchor 的宽高，使其最佳覆盖所匹配的 gt box

Faster R-CNN 为 anchor 选定一个具有最大 IOU 的 gt box，然后根据 anchor 和 gt box 计算 $(t_x,t_y,t_w,t_h)$ 作为回归 target，这里的 anchor 其 $(x,y,w,h)$ 是已知的预设值。

但是这种方法不适合我们的 anchor 生成机制，因为 anchor 的 w,h 不再是固定的预设值，而是变化的，也就是说，我们 shape 分支预测得到某位置的 $(w_p,h_p)$ 值，但是我们怎么确定该位置处的回归 target $(t_w,t_h)$ 呢？为了解决此问题，我们定义一个 anchor 变量 $a_{\mathbf {wh}}=\{(x_0,y_0,w,h)|w>0,h>0\}$ 与一个 gt box $gt=(x_g,y_g,w_g,h_g)$ 之间的 IoU （记作 vIoU）为，
$$\text{vIoU}(a_{\mathbf {wh}},gt)=\max_{w>0,h>0} IoU_{normal}(a_{wh},gt)$$
对于任意给定的 anchor 位置 $(x_0,y_0)$ 和 gt box $gt$，上式的解析解是非常复杂的，在一个端到端的网络中很难去实现这个计算，因此使用一个替代方法得到近似解。给定位置 $(x_0,y_0)$，我们取一些 w 和 h 的常见值来模拟所有 w 和 h 的枚举，然后计算所取（这些常见 w 和 h）的 anchor 与某个 gt box 的 IoU，其中最大 IoU 作为 $\text{vIoU}(a_{\mathbf {wh}}, gt)$ 的近似。在我们的实验中，我们选取了 9 组 (w,h) 来估算 vIoU。这 9 组 (w,h) 使用 RetinaNet 中的 scales 和 aspect ratios 生成。理论而言，使用越多的 (w,h) 那么 vIoU 的近似越准确，当然计算量也跟着增加。我们采用 bounded iou loss 的变体来优化 shape 预测分支，这个损失如下：
$$\mathcal L_{shape}=\mathcal L_1(1-\min(\frac w {w_g}, \frac {w_g} w)) + \mathcal L_1 (1-\min(\frac h {h_g}, \frac {h_g} h))$$
其中 (w,h) 是预测 anchor shape，(w<sub>g</sub>,h<sub>g</sub>) 是与 anchor 有着最大 vIoU 的那个 gt box 的 shape。从上式损失函数中可见，我们希望 $\min(\frac w {w_g}, \frac {w_g} w)$ 和 $\min(\frac h {h_g}, \frac {h_g} h)$ 越大越好，也就是说，w 越接近 w<sub>g</sub>，h 越接近 h<sub>g</sub>，就越好。

总结一下以上过程：
1. 选择 9 组 (w,h)
2. 给定位置 (x<sub>0</sub>,y<sub>0</sub>)，计算 anchor 与所有 gt box 的 vIoU，每个 vIoU 的计算均使用 9 组 (w,h)
3. 最大 vIoU 的那个 gt box 与此位置 anchor 相匹配
4. shape 预测分支在此位置预测的 (w,h) 与此位置 anchor 匹配的 gt box 的 (w<sub>g</sub>,h<sub>g</sub>) 一起计算得到此处的 shape 损失

那么，为何不直接用 shape 分支预测的 (w,h) 与所有 gt box 计算 IoU，然后选择最大 IoU 的那个 gt box 作为该位置 anchor 所匹配的 gt box 呢？

当然不行，由于 shape 分支预测的 (w,h) 在每次训练迭代过程中均会变化，如果使用上述方法求匹配的 gt box，那么该位置 anchor 所匹配的 gt box 在每次迭代时都有可能不一样，如果 anchor 训练回归的 target 都一直会变化，那就没法训练了。

## 使用高质量 proposals
得到 guided anchoring 加强的 RPN（GA-RPN）可以生成更高质量的 proposals。通过使用这些高质量的 proposals，我们探索了如何提高传统的 two-stage 目标检测器的性能。首先，研究了 RPN 和 GA-RPN 生成的 proposals 的 IoU 分布，如图 3，
![](/images/GA-RPN_fig3.png) <center>Fig 3 不同 IoU 下的 proposals 数量</center>

比起 RPN，GA-RPN 有如下两个明显优点：
1. 正例 proposals 数量更多
2. 高 IoU 处两者的 proposals 数量比例更明显

在现有模型下将 RPN 直接替换为 GA-RPN 然后端到端训练（从头开始训练），然而，如果采用相同的训练设置，性能指标提升会非常有限（不到 1 一个点）。通过我们的观察发现使用高质量 proposals 的前提条件是训练样本的分布需要与 proposal 分布一致。因此，设置一个更高的正负例阈值，从而使用更少的样本去训练。

除了端到端训练，GA-RPN 还可以通过微调提升一个训练好的 two-stage 检测器的性能。具体而言，给定一个训练好的模型，我们舍弃其中的 proposal 生成模块，例如舍弃 RPN，然后使用预先计算好的 GA-RPN proposals 来微调这个模型，仅需要几个 epochs（默认是 3 个 epochs）即可。GA-RPN proposals 还可以用于 inference。这种简单的微调机制因为只需要少数 epochs，所以可以大大提高性能。

![](/images/GA-RPN_fig4.png)

# 实验
实验参数、实现细节以及结果分析这里不展开讨论，直接阅读原文。

# 结论
提出了 Guided Anchoring 机制，利用语义特征生成位置非均匀且 shape 任意的 anchor。