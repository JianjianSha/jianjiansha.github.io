---
title: libra-rcnn
date: 2019-07-03 20:07:44
tags: object detection
mathjax: true
---
论文 [Libra R-CNN: Towards Balanced Learning for Object Detection]()

# Introduction
当前大多数目标检测器无论是 one-stage 还是 two-stage，其训练范式都是 image 上区域选择（以下使用原英文单词 Region 表示），从 region 抽取特征，以及联合目标分类和定位的多任务目标函数优化。基于此训练范式，有如下三点关于训练是否成功：
1. 所选 region 是否具有代表性
2. 抽取的特征是否被充分利用
3. 设计的目标函数是否最优

如图 1，很多训练过程均存在以上三点问题，
![](/images/libra-rcnn_fig1.png) <center> 不平衡包括 a. 样本级别；b. 特征级别；c. 目标函数级别 </center>

由于以上三个不平衡问题的存在，即使一个设计良好的模型，也可能最终训练出来的性能不佳。以下我们具体讨论这三个不平衡问题：

## 样本不平衡
为了避免模型倾向于预测为负，很多训练过程设置了一个训练批次中正负样本的比例（如 1:3）。训练目标检测器时，困难样例更具有价值，可以加快训练收敛速度，有效提高检测性能，然而事实上随机选择的 region 主要都是简单负样例，这些简单负样例贡献不了什么有用的特征，于是有了在线难负例发掘方法 OHEM [1]，这个 OHEM 对于噪声标签非常敏感，因为噪声标签会使得误分类为负也就是难负例筛选不准确，此外 OHEM 显然提高了内存占用并增加了计算量。通过进一步降低分类正确的那部分损失，Focal loss 也可以缓和样本不平衡这个问题，但是 Focal loss 通常用在 one-stage 模型中，在 two-stage 模型中则作用不明显，因为大部分的简单负例在 first stage 已经被过滤掉了（正如前面所说的正负样本比例为 1:3），此时若再使用 Focal loss 则会使得正负例样本所产生的梯度不平衡，较小的梯度淹没在较大的梯度里，难以起到梯度优化指导作用。

## 特征不平衡
深度高层的特征具有更丰富的语义信息，而浅层特征则保留了更多的视觉内容描述（局部细节信息）。近年来，FPN 和 PANet 则通过 top-down 结构和 横向连接来进行特征整合，提高了目标检测性能，这说明高底层的特征对于目标检测的作用确实是互补的。但是，如何最佳地整合特征？前面提高的特征整合方法，feature pyramid 中每一层的特征整合更多的是关注邻近的特征（直接相连），而很少关注其他非邻近特征（非直接相连），非邻近特征需要经过一个或多个中间层才能到达本层特征，显然非近邻特征的语义信息被稀释的非常淡。如下图所示，
![](/images/libra-rcnn_figa.png)<center>FPN</center>

从上图可知，融合后的 a' 特征，其来自于 b,c 层的特征信息不是均衡的。

## 优化目标不平衡
目标检测器是多任务的：分类和定位。两者的目标函数加起来作为最终的优化目标，如果这两者之间不均衡，会导致次优的结果，较小的梯度会淹没在较大的梯度里，起不到优化指导作用。这个情况与训练中样本导致的梯度不平衡的情况是相同的，均会限制模型进一步的性能调优。

我们提出 Libra R-CNN（天秤 R-CNN），以平衡以上三个问题，Libra R-CNN 框架包含三个创新组件：
1. IoU 均衡采样，根据与 gt box 的 IoU 来挖掘难例
2. 均衡的 feature pyramid，使用 __相同深度__ 合并到一起的均衡语义特征进行强化
3. 均衡的 L1 loss，提升关键的梯度从而平衡 1)分类 2) 大致定位 3) 准确定位 这三者

# Methodology
Libra R-CNN 结构如图 2，
![](/images/libra-rcnn_fig2.png)

其中所有组件的详细介绍如下。

## IoU-balanced Sampling
首先一个基本问题是：训练样本 region 和其对应的 gt box 之间的重合与此样本的难易程度是否有关联？图 3 显示了三种 region 采样的 IoU 分布，
![](/images/libra-rcnn_fig3.png)

我们仅考虑难负例，因为难负例是上文我们分析的关键三点之一。从图 3 中可见超过 60% 的难负例有大于 0.05 的 IoU（因为图 3 中橙色部分 IoU 低于 0.05 的占比大约为 37%），而随机采样时仅仅有大约 30% 的训练样本其 IoU 大于 0.05，这意味着如果随机采样会得到很多 IoU 位于 [0,0.05) 区间的样本，而分布在这个区间的难负例样本较少，所以随机采样会得到很多简单样本。

受以上结论启发，我们提出 IoU-balanced 采样：既然难负例分布在各个 IoU 区间，那么我们就对各个 IoU 区间分别采样。假设从 M 个候选中选出 N 个负样本，随机采样下每个样本被选择的概率为
$$p=\frac N M$$

为了提高选择难负例的概率，根据 IoU 将采样区间等分成 K 个桶，从每个桶中选择 N/K 个负样本，于是在 IoU-balanced sampling 下，第 k 个桶中每个样本被选择的概率为
$$p_k=\frac N K \cdot \frac 1 {M_k}, \ k \in [0,K)$$
其中，$M_k$ 是第 k 个桶中的样例候选数量。实验中 K=3。

IoU-balanced sampling 结果如图 3，可以看到使用这种采样方式得到的训练样本分布与难负例的分布非常接近。采样候选数量不足，这种采样方法难以扩展到正例采样，为了得到均衡采样过程，使用一种替换方案：对每个 gt box 我们进行数量相等的采样。

### SOURCE CODE
经过阅读源码，本人总结 IoU balanced sampling 负例采样过程为：
1. 获取所有 proposals 的最大 IoU，记为 `max_overlaps`
2. 获取所有 proposals 的最大 IoU 的最大值，`max_iou=max_overlaps.max()`
3. 设置阈值下限 `floor_thr`，对 `(floor_thr, max_iou)` 范围内的 proposals 进行 IoU balanced sampling
4. 设置桶（bin）数量 K，假设所需要的负例数量为 N，对每个桶采样数量为 N/K，每个桶的 IoU 范围跨度为 `(max_iou-floor_thr)/K`
5. 对于第 k 个桶，计算对应的 IoU 范围，记为 `[sk,ek)`
6. 获取第 k 个桶内的 proposals 的 index
   ```python
   tmp_set = np.where(np.logical_and(max_overlaps>=sk, max_overlaps<ek))[0]
   ```
7. 获取第 k 个桶内的负例的 index
   ```python
   tmp_inds = list(tmp_set & full_set) # full_set 为 floor_thr<iou<0.5  proposals 的 index
   ```
8. 从第 7 步中得到的第 k 个桶中所有负例的 index，在随机抽取 N/K 个负例
   ```python
   random_choice(tmp_inds, N/K)
   ```
IoU balanced sampling 正例采样过程：
1. 获取正例 proposals 所对应的 gt 的 index，记为 `gt_inds`
2. 将第 1 步的结果去重，`unique_gt_inds=gt_inds.unique()`，得到所有 gt 的 index
   
   为什么说是所有 gt 呢？因为所有 gt 均作为正例被添加到正例 proposals 中
3. 所有 gt 的数量为 `num_gts=len(unique_gt_indx)`，假设总共要采样 N 个正例，以每个 gt 为中心，均需要采样 `num_per_gt=N/num_gts` 个正例
4. 对于第 i 个 gt，获取与其匹配的所有正例，并从中随机选择 `num_per_gt` 个正例
   ```python
   inds = torch.nonzero(assign_result.gt_inds == i.item())
   inds = random_choice(inds, num_per_gt)
   ```
以上采样过程均经过简化，如需彻底理解细节问题则直接阅读源码

## Balanced Feature Pyramid
使用 __相同深度__ 合并到一起的均衡语义特征来加强 multi-level features，如图 4，特征经过四个步骤：尺度缩放、整合、精修和加强。
![](/images/libra-rcnn_fig4.png)

### 获取均衡语义特征
记 l 层级的特征为 $C_l$，层级数量为 L。最低层和最高层分别记为 $l_{min}, l_{max}$。如图 4，将所有层级的特征 $\{C_2,C_3,C_4,C_5\}$ resize 到一个中间大小即 $C_4$ 的大小，resize 操作使用插值或者最大值池化实现，所有特征经过尺度缩放后，均衡语义特征可由下式获得，
$$C=\frac 1 L \sum_{l_{min}}^{l_{max}} C_l$$

这个均衡语义特征可经过相反的 rescale 操作来加强各自原始特征（图 4 中的 Identity）。

### 精修均衡语义特征
进一步精修均衡语义特征使其更具判别力。使用嵌入的高斯 non-local 注意力机制[2]精修均衡语义特征。

Balanced feature pyramid $\{P_2,P_3,P_4,P_5\}$ 可用于目标检测，检测网络结构与 FPN 的一致。

## Balanced L1 Loss
遵循 Fast R-CNN 中的分类和目标定位的损失，定义如下，
$$L_{p,u,t^u,v}=L_{cls}(p,u) + \lambda [u\ge 1] L_{loc}(t^u,v)$$
上式为单个样本的损失，其中，预测和 target 分别记为 p 和 u。t<sup>u</sup> 表示回归预测，v 表示回归 target。$\lambda$ 为平衡系数。我们称损失大于等于 1.0 的样本为外点 outliers，损失小于 1.0 的样本为内点 inliers。

由于回归 target 值是无界的，如果直接增大 $\lambda$ 会使得模型对 outliers 更为敏感，outliers 可以视作困难样本（困难样本可以认为是误差较大的样本），由于较大的损失使得梯度也较大，这对训练过程是不利的。Inliers 可以看作简单样本，与 outliers 相比，其梯度贡献较小，具体而言，inliers 平均每个样本仅贡献了 30% 的梯度，基于这些考虑，我们提出均衡 L1 损失，从传统的 smooth L1 损失演变而来，记为 $L_b$。设置一个拐点分离 inliers 和 outliers，并使用最大值 1.0 来剃平 outliers 的较大梯度，如图 5(a) 所示,
![](/images/libra-rcnn_fig5.png)<center>横坐标 regression error 为 |x|，参见下文中的说明</center>

均衡 L1 损失的核心思想是提升关键的回归梯度，也就是来自 inliers 的梯度，使得所有样本的所有任务的梯度达到平衡。使用均衡 L1 损失的定位损失为，
$$L_{loc}=\sum_{i \in \{x,y,w,h\}} L_b (t_i^u-v_i)$$
相关的梯度满足，
$$\frac {\partial L_{loc}} {\partial w} \propto \frac {\partial L_b} {\partial t_i^u} \propto \frac {L_b} x$$
上式中，w 表示网络权重参数（我是这么认为的），x 表示 $t_i^u - v_i$，因为 smooth L1 损失就是这么表示的，回顾一下 smooth L1 损失，其定义如下，
$$L_{loc}(t^u, v) = \sum_{x,y,w,h} smooth_{L_1} (t_i^u-v_i)$$
其中，
$$smooth_{L_1}(x)=\begin{cases} 0.5 x^2 & |x|<1
\\\\ |x|-0.5 & otherwise \end{cases}$$
于是 smooth L1 损失对应的梯度为，
$$\frac {\partial L_1} {\partial |x|} = \begin{cases} |x| & |x|<1
\\\\ 1 & |x| \ge 1 \end{cases}$$
我们将 |x| 看作是回归误差（regression error），显然误差总是非负的。现在，我们要想提升 inliers 的梯度，也就是 |x|<1  的梯度（因为 |x|<1 表示样本损失较小），首先对于 smooth L1 损失在 |x|<1 范围内的梯度为 $\nabla_{|x|} L = |x|$ 也就是一条经过 (0,0) 和 (1,1) 的线段，要提高这个范围内的梯度，很自然的想法是位于直线 y=x 上方的曲线，当然曲线必须要经过原点(0,0)，表示预测与 target 相等即误差为零时损失也为零，为了与 $|x| \ge 1$ 的梯度保持连续，梯度曲线仍然经过 (1,1) 点，同时还要保持单调递增，这说明曲线是 __上凹__ 的，满足这些特性的一组曲线其函数为，
$$\frac {\partial L_b} {\partial x} = \begin{cases} \alpha \ln (b|x|+1) & |x|<1
\\\\ \gamma & otherwise \end{cases}$$
其中 $\alpha$ 越小，对 inliers 的梯度提升越大，$\gamma$ 控制 outliers 的梯度，或者说整个梯度的上限，$\gamma$ 参数用于平衡回归损失和分类损失，平衡后的梯度曲线如图 5(a)所示。参数 b 则用于确保损失在 |x|=1 处连续，对梯度积分得到损失函数为，
$$L_b(x)=\begin{cases} \frac \alpha b (b|x|+1) \ln (b|x|+1) - \alpha |x| & |x| < 1
\\\\ \gamma |x| + C & otherwise \end{cases}$$
根据损失在 |x|=1 处连续，得到
$$\frac \alpha b (b+1) \ln (b+1) - \alpha=(\alpha + \frac \alpha b) \ln(b+1) -\alpha = \gamma + C$$
由于 C 可以是任意常数，所以可令 $C=\frac \alpha b \ln(b+1) -\alpha$，于是有
$$\alpha \ln (b+1)=\gamma$$
解得，
$$b=e^{\gamma / \alpha} -1
\\\\ C=\gamma/b-\alpha$$

损失函数曲线如图 5(b) 所示。

# Experiments
实验部分略

# Conclusion
提出了 Libra R-CNN，包含三点：
1. IoU balanced sampling
2. balanced feature pyramid
3. balanced L1 loss

# Reference
1. Training Region-based Object Detectors with Online Hard Example Mining. Abhinav Shrivastava
2. Non-local neural networks. Xiaolong Wang