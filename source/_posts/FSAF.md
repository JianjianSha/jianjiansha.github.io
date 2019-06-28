---
title: FSAF
date: 2019-06-27 09:14:42
tags: object detection
mathjax: true
---
论文：[Feature Selective Anchor-Free Module for Single-Shot Object Detection](https://arxiv.org/pdf/1903.00621)

目标检测中一个具有挑战性的问题是目标尺度的变化，即，在检测极小目标或极大目标时，往往检测性能不够好。为了达到尺度不变性，SOTA 检测器使用 feature pyramid 或 image pyramid。比如使用 feature pyramid 时，高 level 的 feature 对应大 anchor，低 level 的 feature 对应小 anchor，如图 2，高 level 的 feature 拥有更多的语义信息，适合检测大目标，而低 level 的 feature 由于保持了细粒度的信息，所以适合检测小目标。但是这种网络设计有两个局限：
1. 启发式导向的特征选择
2. 基于 overlap 选取 anchor 

第 1 点是指对某个目标的检测选择哪个 level 的 feature 是启发式的，或者说是通过不断的实验、试错从而找到这个问题的解，这就导致为某个目标所选的 feature level 可能不是最优的。第 2 点则指出每个目标总是需要根据 IoU 去匹配到最近的 anchor 上去。

![](/images/FSAF_fig2.png)

本文则提出一个简单而有效的方法同时解决以上两个局限，此方法名为 feature selective anchor-free (FSAF)，目的是为了让每个目标实例选择最佳 feature level。如图 3，

![](/images/FSAF_fig3.png) <center>Fig 3 FSAF 插入到传统基于 anchor 的检测模块中。训练阶段，根据特征选择将每个目标实例分配到一个pyramid level上</center>

对 feature pyramid 的每个 level 均使用一个 anchor-free 分支，此分支与 anchor-based 分支类似，包含一个分类子网络和回归子网络（图 3 中没有展示出来）。目标实例可被分配到任意 level 的 anchor-free 分支。训练阶段，基于实例内容而不仅仅是实例 box 为每个目标实例动态选择最佳 feature level。这个 feature level 负责学习并检测这个被分配过来的目标实例。Inference 阶段，可以单独使用 FSAF 模块或者将其与 anchor-based 分支结合使用。此外，anchor-free 分支和在线特征选择可以使用复杂的结构，但是在我们的实验中，我们选择简单的 FSAF 模块结构，所以 FSAF 模块的计算量与整个网络相比是很小的。

# FSAF 模块
我们来看下如何实现 FSAF 模块以及如何将其整合到具有 feature pyramid 的 single-shot 检测器（如 SSD, DSSD, RetinaNet）中。不失一般性，我们将 FSAF 应用到 SOTA 的 RetinaNet。从以下几个方面来说明我们的设计：
1. 如何创建 anchor-free 分支
2. 如何生成 anchor-free 分支的监督信号（GT target）
3. 如何为每个实例动态选择 feature level
4. 如何联合训练/测试 anchor-free 分支和 anchor-based 分支

## 网络框架
图 4 是将 FSAF 应用到 RetinaNet 的网络结构。简单而言，Retina 由一个 backbone 网络以及两个特定任务的子网络组成。从 backbone 网络中构建 feature pyramid，其 level 为 $\{P_l|l\in [3,7]\}$，$P_l$ 分辨率为输入 image 的 $1/2^l$ 倍。图 4 中为了简单起见仅显示了三个 level 的 feature pyramid，每个 level 负责检测一定 scale 范围的目标，每个 feature level 后接分类子网络和回归子网络，这俩子网络均为小型全卷积网络。

基于 RetinaNet，FSAF 仅在每个 feature level 增加两个卷积层，如图 4，

![](/images/FSAF_fig4.png)<center>Fig 4 具有 FSAF 的 RetinaNet 框架</center>

这两个卷积层分别负责 anchor-free 分支的分类预测和回归预测。具体地，具有 3x3 大小的 K 个卷积核的卷积层附在分类子网络上，这个卷积层后跟一个 sigmoid 函数用于将分类得分归一化，与 anchor-based 的分类卷积层并列，用于预测空间每个位置点的 K 个分类的得分（置信度）。回归子网络则类似的使用 3x3 大小的 4 个卷积核的卷积层后跟一个 ReLu 函数，用于预测 anchor-free 方式的 box 偏差。anchor-free 分支和 anchor-based 分支以多任务方式联合运作并共享所属 level 的 feature。

## Ground-truth and Loss
给定一个目标，我们知道其分类 k 和 bbox 坐标 b=[x,y,w,h]。此目标可被分配到任意 feature level，定义此目标映射到 $P_l$ 的 box 为 $b_p^l=[x_p^l,y_p^l,w_p^l,h_p^l]$，由于 $P_l$ 分辨率是输入 image 的 $1/2^l$，故 $b_p^l=b/2^l$。定义一个有效 box $b_e^l=[x_e^l,y_e^l,w_e^l,h_e^l]$ 和一个 ignore box $b_i^l=[x_i^l,y_i^l,w_i^l,h_i^l]$，均为 $b_p^l$ 的线性缩放，比例分别为 $\epsilon_e, \ \epsilon_i$，于是有
$$x_e^l=x_p^l, \ y_e^l=y_p^l, \ w_e^l=\epsilon_e w_p^l, \ h_e^l=\epsilon_e h_p^l
\\\\x_i^l=x_p^l, \ y_i^l=y_p^l, \ w_i^l=\epsilon_i w_p^l, \ h_i^l=\epsilon_i h_p^l$$
（到这里就发现与 [GA-RPN](/2019/06/25/GA-RPN) 中完全一样有木有，所以 anchor-free 到底是什么，是不是也突然明白了什么，如果与 GA-RPN 中一样的话，那么 anchor-free 就是指没有预设 scale 和 aspect ratio 生成的均匀密集分布的 anchor，也就是说 anchor-free 还是有 anchor 的，只不过其 shape 是任意的、动态生成的，而不是 anchor-based 那样固定的 scale 和 aspect ratio。好的，先不管是不是这样，我们继续往下讨论。）

图 5 是一个 car 实例的 GT 生成（GT target）的例子

![](/images/FSAF_fig5.png)

__分类输出：__ 分类的 GT output 是 K-channel maps，每个 map 对应一个分类。假设目标分类为 k，那么对第 k 个 GT map 有：
- 位于 $b_e^l$ 内的为正例，值为 1，如图 5 中白色区域
- 位于 $b_i^l - b_e^l$ 内的点忽略，此区域的梯度不进行反向传播，如图 5 中灰色区域
- 如果存在邻近 feature level，那么其上的 $b_i^{l-1}, b_i^{l+1}$ 区域也被忽略

这里需要注意的是，由于在线特征选择模块，单个实例最终只用在最佳的某个 feature level 上。

如果两个实例的有效 box 重叠了，较小尺度的实例优先权更高。gt map 的剩余区域则是负例，值为 0，如图 5 中黑色区域。分类损失使用 Focal loss，
$$FL(p_t)=-\alpha_t (1-p_t)^{\gamma} \log p_t
\\\\p_t=\begin{cases} p & y=1 \\\\1-p & y=0 \end{cases}
\\\\\alpha_t=\begin{cases} \alpha & y=1 \\\\1-\alpha & y=0 \end{cases}$$
anchor-free 的总分类损失为除 ignore box 之外的区域内所有点的 focal loss 之和，并除以有效 box 内点的数量进行归一化。

__Box 回归输出：__ 回归输出的 gt 为 4-channal maps，表示 4 个偏差值（与分类无关，否则就是 4K-channel 了）。实例仅影响 gt maps 上 $b_e^l$ 区域的值，对 $b_e^l$ 内某一像素点位置 (i,j)，我们使用一个 4-d 向量来表示 $b_p^l$：
$$\mathbf d_{i,j}^l=[d_{t_{i,j}}^l,d_{l_{i,j}}^l,d_{b_{i,j}}^l,d_{r_{i,j}}^l]$$
其中 $d_t^l,d_l^l,d_b^l,d_r^l$ 分别为当前位置点 (i,j) 到 $b_p^l$ 的 top,left,bottom,right 四条边的距离。这个与 [FCOS](FCOS) 是差不多的，毕竟都是 anchor-free 的。然后点 (i,j) 处的 4-d 向量归一化为 $\mathbf d_{i,j}^l/S$，根据经验 S=4（可能是训练过程中发现这样归一化后不容易出现梯度饱和的现象，或者是训练更加稳定）。有效 box 之外的区域的梯度全部忽略。采用 IoU 损失来优化此分支参数。anchor-free 的总回归损失为所有有效 box 区域的 IoU 损失的平均（损失之和对有效 box 内点的数量取平均），其中单点 IoU 损失为
$$L_{IoU}=-\log IoU
\\\\ IoU = \frac {I(b_p,b_{gt})} {U(b_p,b_{gt})}$$
具体可参考 UnitBox。

Inference 阶段，从分类输出和回归输出中解码出预测 box。在位置 (i,j)，假设预测偏差输出为 $[\hat o_{t_{i,j}},\hat o_{l_{i,j}},\hat o_{b_{i,j}},\hat o_{r_{i,j}}]$，那么预测距离为 $[S\hat o_{t_{i,j}},S\hat o_{l_{i,j}},S\hat o_{b_{i,j}},S\hat o_{r_{i,j}}]$，于是左上角和右下角坐标分别为 $(i-S\hat o_{t_{i,j}},j-S\hat o_{l_{i,j}}), \ (i+\hat o_{b_{i,j}},j+\hat o_{r_{i,j}})$，最后再乘以 $2^l$ 就恢复到输入 image 上的预测框，其置信度得分和分类则可以根据分类输出 maps 上 (i,j) 处的 K-d 向量决定。

## 在线特征选择
FSAF 模块为每个实例选择最佳 level 的 feature $P_l$，这种选择是基于实例的内容，而 anchor-based 中则是基于实例 box 大小，显然基于实例内容更加合理。

给力实例 $I$，定义其在 $P_l$ 上的分类损失和回归损失分别为 $L_{FL}^I(l), \ L_{IoU}^I(l)$，计算式如下
$$L_{FL}^I(l)=\frac 1 {N(b_e^l)} \sum_{i,j \in b_e^l} FL(l,i,j)
\\\\L_{IoU}^I(l)=\frac 1 {N(b_e^l)} \sum_{i,j \in b_e^l} IoU(l,i,j)$$
其中，$N(b_e^l)$ 为有效 box 内点的数量。注意因为只考虑实例 $I$ 的损失，故分类损失只考虑了正例损失的那部分。

图 6 显示了在线特征选择的过程。

![](/images/FSAF_fig6.png) <center>Fig 6 在线特征选择机制。每个实例通过所有level的anchor-free分支以相应的计算平均分类损失和平均回归损失，然后具有最小两种损失之和的分支为最佳分支，在此分支上设立此实例的监督信号（gt target）</center>

首先实例 $I$ 前向传播到 feature pyramid，然后计算每个 anchor-free 分支的 $L_{FL}^I(l) + L_{IoU}^I(l)$ 的和，最后根据最小损失之和选择最佳 feature pyramid leve $P_l$，
$$l^*=\arg \min_l L_{FL}^I(l) + L_{IoU}^I(l)$$
对于一个训练批次，更新某 level 的特征仅使用分配到此 level 上的实例。直觉上，根据这种方法选择的特征最适合对实例进行建模，因为此时的损失在特征空间构成损失下限，而经过训练又进一步地拉低损失下限。Inference 阶段，我们不需要手动选择使用哪个特征，在线选择的最适合的特征将会输出高置信度得分。

我们比较了启发式特征选择和在线特征选择。启发式特征选择仅依赖于 box size，例如在 FPN 检测器中，实例 $I$ 将被分配到 $P_{l'}$，其中
$$l' = \lfloor l_0+\log_2(\sqrt{wh}/224) \rfloor$$
其中，(w,h) 是实例 size，224 是典型的 ImageNet 预训练尺寸，224x224 应该映射到 $l_0$ 这个 target level 上。如何理解上式？首先 $P_l$ 的分辨率是原始输入 image 的 $1/2^l$，然后将上式变形如下就能理解了，
$$ \sqrt{wh}/2^{l'} \approx 224/2^{l_0}$$
可见是将一个 scale 范围按 $1/2^l$ 的比例分配。我们这里选择 $l_0=5$，因为 ResNet 使用 conv5_x 卷积组的 feature map 进行分类预测。

## Joint Inference and Training
当 FSAF 模块插入到 RetinaNet 中时，如图 4，我们保持原来的 anchor-based 分支不变，所有的超参也不变。

__Inference:__ FSAF 仅增加少量的卷积层。对于 anchor-free 分支，我们对每个 pyramid level 的分类输出使用置信度阈值 0.05 进行过滤，然后分别选取 top 1k 得分的位置点，从这些位置解码出预测 box，所有 level 的预测 box 与 anchor-based 分支的预测 box 合并起来，并使用非极大抑制 NMS，NMS 阈值为 0.5，得到最后的检测结果。

__初始化：__ backbone 网络使用 ImageNet1k 预训练。RetinaNet 中的 layers 与原始 RetinaNet 中 layers 的初始化相同。FSAF 中的 分类分支的 layers 初始化所用的高斯分布权值 $\sigma=0.01$，偏置 bias 为 $-\log((1-\pi)/\pi)$，其中 $\pi$ 指明训练初始时各像素点输出是否存在目标的得分值在 $\pi$ 上下。我们遵循原始 RetineNet 中的设置 $\pi=0.01$。所有回归分支的 layers 初始化使用偏置 b=0.1，高斯权值 $\sigma=0.01$。以上初始化过程由于避免生成较大的损失，从而有助于训练初期过程的稳定。

__优化：__ 整个网络的损失来自于 anchor-free 分支和 anchor-based 分支。记原始 RetinaNet 的总损失为 $L^{ab}$，而 $L_{cls}^{af}, \ L_{reg}^{af}$ 分布为 anchor-free 分支的分类损失和回归损失。那么，整个网络的总损失为 $L=L^{ab}+\lambda (L_{cls}^{af} + L_{reg}^{af})$，其中 $\lambda$ 用于平衡两者，实验中设置 $\lambda=0.5$。

# 实验
实验介绍以及结果分析略，请阅读原文以获取详细信息。

#总结
本文指出了具有 feature pyramid 的 anchor-based single-shot 目标检测器中启发式选择特征的不足之处，并提出 FSAF 模块以解决这个问题，FSAF 使用了 anchor-free 分支以及在线特征选择，显著提高了检测性能，inference 的耗费增加较少，但是性能超过最近的 SOTA single-shot 检测器。