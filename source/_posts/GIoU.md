---
title: GIoU
date: 2019-06-13 15:00:48
tags: object detection
mathjax: true
---
论文 [Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression](https://arxiv.org/abs/1902.09630)
<!-- more -->
# 摘要
IoU 是目标检测benchmarks中使用最广的评估指标，然而，优化回归bbox参数的距离损失并不等价于最大化IoU指标。对于轴对齐的2D bbox，IoU 可直接用作回归损失，但是 IoU 无法优化无重叠的bbox，所以本文提出一种泛化版的Iou，名为 GIou。结合 GIoU 和 sota 目标检测框架，在流行的目标检测benchmarks例如 PASCAL VOC 和 MS COCO中，分别使用标准 IoU 和 GIoU 损失，我们发现性能一致得到提升。
# 简介
在很多2D/3D 计算机视觉任务中，bbox 回归是最基本的组成之一。例如目标定位，多目标检测，目标跟踪以及实例分割，均依赖准确的bbox回归。提升性能的主流趋势是使用深度神经网络，然而，还有一种提升方法被广泛忽略，那就是使用基于 IoU 的指标损失来代替传统的回归损失如 $l_1, l_2$ 等。

IoU 将要比较的目标的形状属性例如 bbox 的宽、高和位置等信息编码成区域属性，并基于面积（体积）计算出一个归一化的值。语义分割、目标检测和跟踪等任务中的性能测量均采用IoU作为测量指标。

然而，最小化损失如$l_n$范数与提高IoU并不是强相关。考虑一个2D场景，如图1(a)，
![](/images/GIoU_fig1.png)

<!-- {% asset_img fig1.png This figure 1 %} -->

预测box（黑矩形）和gt box（绿矩形）均由左上角和右下角坐标表示$(x_1,y_1,x_2,y_2)$，为简单起见，我们令两个box的其中一个corner的距离（例如$l_2$范数）固定，于是，以 gt box 另一个 corner 为圆心，某半径长的圆，无论预测 box 的另一个 corner 的坐标，只要其位于这个圆上，其 $l_2$ 距离均保持不变；然而IoU却不同。这个问题可以延伸到其他损失和bbox表示上，例如图1(b)。

直觉而言，这些类型的损失的一个较好的局部最优解可能并非 IoU 的局部最优解。而且，与 IoU 不同的是，$l_n$ 不具有尺度不变性，相同重叠程度的几对 bbox，其损失值各不相同。另外，一些 bbox 的表示方法，由于没有对不同类型的表示参数进行正则处理，使得这个问题更加严重。例如在 center+size 表示法中，$(x_c,y_c)$ 是中心坐标，$(w,h)$ 是 box size。当表示参数变多时，如旋转度，复杂度继续上升。

为了解决这些问题，sota 检测器引入了 anchor 的概念，并使用了非线性表示方法简单地处理尺度问题（例如，faster-rcnn中坐标偏差的计算）。但即使使用了这些手工设计，优化回归损失和IoU值依然存在偏差。

本文探索了轴对齐矩形之间的IoU计算，以及轴对齐超矩形（$ndim \ge 2$）之间的IoU计算，此时IoU 有解析解，并且可反向传播，也就是说，IoU 可以直接用作目标函数进行优化，而优化Iou目标函数与优化某个损失函数之间，显然选择优化IoU目标函数，能与提高Iou指标强相关，但是这也导致一个问题：如果两个目标没有重叠，IoU则为0，无法知道两个目标距离有多远，IoU为0，其梯度也将为0，导致无法优化。

我们将IoU这一概念延伸到无重叠情况下来解决上述问题。这种泛化：(a) 沿袭 IoU 能将被比较的目标的形状属性编码进区域属性；(b) 维持 IoU 的尺度不变性；(c) 在目标有重叠情况下与IoU强相关。这个泛化版的IoU，我们称为GIoU，将GIoU引入到 sota 目标检测框架，在流行的目标检测的benchmarks上，比较标准的IoU和GIoU，发现性能一致均得到提升。

主要贡献如下：

- 介绍了GIoU，作为比较两个任意形状差距的指标
- 以GIoU作为轴对齐矩形或超矩形的损失时，使用解析解
- 将GIoU引入sota 检测器如Faster R-CNN，Mask R-CNN和YOLO v3，并在标准目标检测benchmark上验证性能得到提升

# 相关工作
__目标检测准确性测量：__ IoU作为评估指标在目标检测任务中广为使用，常用于确定预测box是真阳性还是假阳性。使用IoU时需要选择一个阈值。在PASCAL VOC上，计算mAP时选择 IoU阈值=0.5，但是随意选择的IoU阈值不能完全反映定位性能，所有定位准确性大于这个阈值的检测结果认为是真阳性，从而参与mAP的计算，IoU阈值的选择将直接影响mAP值。为了是性能测量对IoU阈值不敏感，MS COCO benchmark 则选择不同的IoU阈值计算多个mAP然后取平均。

__bbox表示和损失：__ 2D目标检测中，bbox参数非常重要。最近的文献提出多种不同bbox表示和损失：
1. YOLO v1
   
   YOLO v1 直接回归 bbox 参数$(x_c,y_c,w,h)$，坐标损失使用平方差。计算损失时，为了降低目标scale对(w,h)损失项的影响，将这一损失项由$(w-\hat w)^2+(h-\hat h)^2$ 改为 $(\sqrt w - \sqrt {\hat w})^2+(\sqrt h - \sqrt {\hat h})^2$。
2. R-CNN
   
   R-CNN使用selective search先获得候选boxes，然后回归bbox中心点偏差（求差）和size的偏差（求商），为了降低scale敏感度，将size 偏差转换到对数空间（求log），然会对偏差使用$l_2$范数（最小均方差MSE）作为目标函数进行优化。
3. Fast R-CNN
   
   Fast R-CNN对坐标偏差使用 $l_1$-smooth 损失，使得模型即使在异常值情况下也具有较好的鲁棒性（异常值情况一般指偏差非常大的情况，此时若使用 $l_2$ 范数的目标函数，其梯度比较大，使得模型训练初期非常不稳定）。
4. Faster R-CNN
   
   Faster R-CNN使用密集anchor boxes，然后对其中心坐标和size的偏差进行回归，根据anchor boxes的得分（分类置信度）按正负例的一定比例（1:3）得到一个batch （数量为128）的proposals，然后再使用Fast R-CNN的分类和回归两个分支进行最终的预测。为了进一步解决正负例不平衡问题，RetinaNet 使用 focal loss。

大部分目标检测器都是结合以上某种bbox表示和某种损失。这些努力推动目标检测有了明显的发展。我们的工作表明，使用GIoU 损失可以进一步提高目标定位，因为如前面所分析的那样 bbox 回归损失并不能够直接反映检测评估指标IoU。 

__使用近似或替代函数优化IoU：__ 在语义分割任务种，曾使用近似函数或替代损失优化IoU。类似地，目标检测任务中，最近的一些研究工作也尝试直接或间接利用IoU以更好地进行bbox回归，然而却在非重叠情况下优化IoU时遇到近似或梯度平坦问题。本文我们通过引入GIoU解决IoU在非重叠情况下的问题。

# 泛化IoU
用于比较两个任意形状 $A,B \subseteq S \in \mathbb{R}^n$ 的 IoU 计算方法为：
$$IoU = \frac {|A \cap B|} {|A \cup B|} $$
两个显著特性使得这种相似性测量方法流行于评估2D/3D计算机视觉任务中：
- IoU作为距离同时也作为评估指标。
  
  IoU距离即 $\mathcal L_{IoU}=1-IoU$，这意味着 $\mathcal L_{IoU}$ 满足 IoU 指标的所有性质，例如非负性，不可分的同一性，对称性和三角不等式
- IoU具有尺度不变性。这意味着，两个任意形状A B的相似度与它们在 S 空间的尺度无关
   
但是，IoU的主要问题是：
- $|A \cap B|=0 \Rightarrow IoU(A,B)=0$，此时，IoU无法分辨两个形状A B是靠的非常近还是非常远

为了解决这个问题，我们提出了泛化版IoU，即 GIoU。

两个任意的凸形 $A, B \subseteq S \in \mathbb S^n$，首先在 S 空间中寻找包含 A 和 B 的最小凸形 C。如果比较两个具体类型的几何图形，C 可以也是这个具体类型，例如比较两个椭圆形，C 则是包含这两个椭圆形的最小椭圆形。然后我们计算 C 中扣掉 A 和 B 剩余部分的面积（体积）与 C 自身的面积（体积）的比例，这个比例代表了一种归一化的且注重 A 和 B 之间的空白部分面积（体积）的测量方法，然后，从 IoU 中减去这个比例就得到 GIoU。（面积/体积对应 2D/3D）

整个计算过程总结如下算法1：
___
算法1：GIoU
___
输入： 两个任意凸形 $A,B \subseteq S \in \mathbb S^n$

输出： GIoU
- 在 S 空间中寻找包含 A B 的最小凸形 C
- 计算 IoU
  
  $IoU=\frac {|A \cap B|} {|A \cup B|}$
- 计算 GIoU
  
  $GIoU = IoU - \frac {|C \setminus (A \cup B)|} {|C|}$
___

作为新的指标，GIoU 具有性质：

- 与 IoU 类似，GIoU 作为距离具有指标的所有性质：非负性，不可分的同一性，对称性和三角不等式
  
  IoU 距离即 $\mathcal L_{GIoU} = 1-GIoU$。
- 与 IoU 类似，GIoU 具有尺度不变性。
- GIoU 上限为 IoU
  
  $\forall A,B \subseteq \mathbb S, GIoU(A,B) \le IoU(A,B)$，当 A B越靠近且形状越相似，则 GIoU 越接近 IoU，即 $\lim_{A \rightarrow B} GIoU(A,B)=IoU(A,B)$。
- IoU 和 GIoU 的值域
  
  $\forall A,B \subseteq \mathbb S, 0 \le IoU(A,B) \le 1$，但是 GIoU 的值域则关于零点对称，$-1 \le GIoU(A,B) \le 1$。

  我们看下如何获得边界值：
   * 与 IoU 相同，只有当 A B 完成重合的时候，即$|A \cup B|=|A \cap B|$，此时$GIoU =IoU=1$
   * 当 A B 两个形所占面积（体积）与 C 所在面积（体积）之比趋于 0，GIoU 趋于 -1，即$\lim_{\frac{|A \cup B|}{|C|}\rightarrow 0} GIoU(A,B)=-1$

综上，GIoU 保持了 IoU 的主要性质并避免了 IoU 的缺点，所以在2D/3D计算机视觉任务的性能测试中可以使用 GIoU 代替 IoU。本文我们侧重于 2D 目标检测，推导 GIoU 的解析解，GIoU 同时担当性能指标和损失。在非轴对齐 3D 场景下的 GIoU 则待以后的工作去研究。

## GIoU用作BBox回归损失
我们已经介绍了 GIoU 可以作为任意两个形状的距离测量指标，但是与 IoU 一样，没有解析解计算两个任意形状的交，也没有解析解可以计算包含这俩形状的最小凸形。

好在2D 目标检测任务中，bbox 是轴对齐的，此时 GIoU 有解析解。两个形状 A B 的交，以及包含 A B 的最小凸形均具为矩形，对 A B 的顶点坐标使用 min 或 max 操作可以求得它们的顶点坐标。为了确定 A B 是否重叠，还需要进行条件检查，比如 A B 的交，作为矩形，其左上顶点的 x 坐标必然比右下顶点的 x 坐标小即 $x^{tl} < x^{br}$，而 $x^{tl}=\max (x_A^{tl}, x_B^{tl}), \ x^{br}=\min (x_A^{br},x_B^{br})$，所以有 $x_B^{tl} \le x_A^{tl}<x_B^{br}$ 或 $x_A^{tl} \le x_B^{tl}<x_A^{br}$。

反向传播中，min、max和按位计算的线性函数如 ReLU 的梯度计算均是可行的，算法2 中每个部分均可以求导，故 IoU 和 GIoU 均可以直接用作损失即 $\mathcal L_{IoU}, \mathcal L_{GIoU}$ 来优化基于深度神经网络的目标检测器。
_________
算法2：IoU和GIoU用作BBox回归损失
_________
输入：预测框 $B^p$ 和 GT 框 $B^g$ 的坐标，$B^p=(x_1^p,y_1^p,x_2^p,y_2^p), \quad B^g=(x_1^g,y_1^g,x_2^g,y_2^g)$

输出：$\mathcal L_{IoU}, \ \mathcal L_{GIoU}$

1. 因为预测框各个坐标是独立预测出来的，所以需要确保 预测 box 坐标有效即，
   
   $x_2^p>x_1^p, \ y_2^p>y_1^p$，故进行如下转换：
   - $\hat x_1^p=\min(x_1^p,x_2^p), \ \hat x_2^p=\max(x_1^p,x_2^p)$
   - $\hat y_1^p=\min(y_1^p,y_2^p), \ \hat y_2^p=\max(y_1^p,y_2^p)$
2. 计算 GT box 面积：
   
   $A^g=(x_2^g-x_1^g)\times (y_2^g-y_1^g)$
3. 计算预测 box 面积：
   
   $A^p=(x_2^p-x_1^p)\times (y_2^p-y_1^p)$
4. 计算交：
   
   - $x_1^{\mathcal I}=\max(\hat x_1^p, x_1^g), \ x_2^{\mathcal I}=\min(\hat x_2^p,x_2^p)$
   - $y_1^{\mathcal I}=\max(\hat y_1^p, y_1^g), \ y_2^{\mathcal I}=\min(\hat y_2^p,y_2^p)$
   - $\mathcal I=\begin{cases} (x_2^{\mathcal I}-x_1^{\mathcal I})\times (y_2^{\mathcal I}-y_1^{\mathcal I}) & x_2^{\mathcal I} > x_1^{\mathcal I}, y_2^{\mathcal I} > y_1^{\mathcal I} \\ 0 & \text{otherwise} \end{cases}$
5. 计算最小包含凸形 c：
   
   - $x_1^c=\min(\hat x_1^p, x_1^g), \ \max(\hat x_2^p, x_2^g)$
   - $y_1^c=\min(\hat y_1^p, y_1^g), \ \max(\hat y_2^p, y_2^g)$
6. 计算 c 的面积：
   
   $A^c=(x_2^c-x_1^c)\times (y_2^c-y_1^c)$
7. 计算 IoU：
   
   $IoU = \frac {\mathcal I}{\mathcal U}$，其中 $\mathcal U = A^p+A^g-\mathcal I$
8. 计算 GIoU：
   
   $GIoU = IoU - \frac {A^c-\mathcal U} {A^c}$
9. 计算 GIoU 损失：
    
   $\mathcal L_{IoU}=1-IoU, \ \mathcal L_{GIoU}=1-GIoU$
_________
根据指标检测性能时，以指标本身作为损失来优化显然是最佳选择，但是在bbox非重叠场景下，IoU=0，其梯度也为0，影响训练质量和收敛速度，相反，GIoU 则一直有有效梯度指导如何优化模型。另外，根据性质3，GIoU 与 IoU 强相关，在 IoU 较大时，这种强相关更加显著。图2 定性的分析了这种相关性，

![](/images/GIoU_fig2.png)

图2中，随机选择了1万组 2D 矩形pair，计算其 IoU 和 GIoU，观察发现，在重叠较小时，例如 $IoU \le 0.2, \ GIoU \le 0.2$，GIoU 可以比 IoU 变化更显著，而且在任何情况下，GIoU 的梯度都可以很陡，所以将 GIoU 作为损失$\mathcal L_{GIoU}$，比使用 IoU 作为损失$\mathcal L_{IoU}$，更有利于优化，并且最终的性能测量指标只要是基于IoU，无论使用哪种指标均可。

### 损失稳定性
我们也考察了预测值为任意的情况下，损失是否会不稳定或者出现未定义情况（比如除数为0）。

假设 GT box 是矩形，且面积大于0即，$A^g > 0$，算法2中第1点和第4点分别确保了预测框和两个bbox的交均非负即，$A^p \ge 0, \ \mathcal I \ge 0, \forall B^p \in \mathbb R^4$，又根据$\mathcal U \ge A^g$，故 $\mathcal U > 0$，所以 IoU 的分母为正非零。又 $\mathcal U \ge \mathcal I$，故 $0 \le IoU \le 1$，于是 IoU 损失范围为 $0 \le \mathcal L_{IoU} \le 1$

检查 GIoU 的稳定性，需要考察项 $\frac {A^c-\mathcal U} {A^c}$，显然包含 A B 的最小凸形不小于 A B 的并，即 $A^c \ge \mathcal U > 0$，所以 $\frac {A^c-\mathcal U} {A^c} \ge 0$。理论上来讲，$\frac {A^c-\mathcal U} {A^c} <1$，且当 A B 中心点的几何距离比 A B 的 size 大很多时，即 A B 离得很远，此时 $\frac {A^c-\mathcal U} {A^c} \rightarrow 1$，故 $-1 < GIoU \le 1$，为了对称，改写为 $-1 \le GIoU \le 1$。

### IoU=0时$\mathcal L_{GIoU}$的行为
GIoU 损失 $\mathcal L_{GIoU}=1-GIoU=1+\frac {A^c-\mathcal U} {A^c} - IoU$，当 $B^p$ 和 $B^g$ 不相交，即 $\mathcal I=0, IoU=0$，此时 GIoU 损失简化为 $\mathcal L_{GIoU}=1+\frac{A^c-\mathcal U}{A^c}=2-\frac {\mathcal U}{A^c}$，最小化 GIoU 损失则需要最大化 $\frac {\mathcal U}{A^c}$，这一项已经是归一化的，即 $0\le \frac {\mathcal U}{A^c} \le 1$，并且最大化这一项则需要最小化 $A^c$，同时最大化 $\mathcal U$，因为 $\mathcal I=0$，故此时 $\mathcal U=A^p+A^g$，由于 $A^g$ 已知且固定，所以需要最大化 $A^p$，也就是说，最小化 $A^c$ 且同时最大化 $A^p$，显然，这就使得 $B^p$ 趋于与 $B^g$ 重合。

# 实验结果
引入 bbox 回归损失 $\mathcal L_{GIoU}$ 到2D目标检测器中如 Faster R-CNN、Mask R-CNN 和 YOLO v3，即，将原来 Faster R-CNN/Mask R-CNN 中的 $l_1$-smooth 损失和 YOLO v3 中的 MSE 损失替换为 $\mathcal L_{GIoU}$，并且我们还对比了 baseline 和使用 $\mathcal L_{IoU}$ 损失时的结果。记使用目标检测器原先的损失为 baseline。（具体实验数据和结果分析请参考原论文，这里省略）

__数据集__ 使用PASCAL VOC 和 MS COCO 数据集，训练方案的细节和对应的评估见下文。

__评估方案__ 本文采取 MS COCO 的性能测试方法，在所有分类上计算 mAP，计算在不同 IoU 阈值下的 mAP 值，IoU 阈值用于判断正负例（因为计算mAP需要知道正例数量），取阈值 $IoU=\{.5,.55,...,.95\}$，计算这些 IoU 阈值下 mAP 值的平均，记为 __AP__，然后使用 GIoU 代替 IoU 来判断正负例，同样取阈值 $GIoU=\{.5,.55,...,.95\}$，计算这些阈值下 mAP 的平均，__AP__，特别地，文中还报导了当 IoU 和 GIoU 阈值为 0.75 时的 mAP，记为 __AP75__。

## YOLO v3
__训练方案__ 使用 YOLO v3 的 Darknet 使用版本。为了得到 Baseline 结果（使用 MSE 损失），我们使用 DarkNet-608 作为 backbone，训练所使用的参数与 YOLO v3 中一致。使用 IoU 和 GIoU 损失训练 YOLO v3 时，我们仅仅将原先的 MSE 损失替换为 $\mathcal L_{IoU}, \mathcal L_{GIoU}$，考虑到 MSE 损失无边界，我们的新损失则是有边界的，而 YOLO v3 损失还包含了分类损失，所以需要针对分类损失将 bbox 回归损失进行正则处理。当然，我们做了一个极小的努力来进行正则处理。

## Faster R-CNN 和 Mask R-CNN
__训练方案__ 使用最新的 Faster R-CNN/Mask R-CNN 的 PyTorch 实现。为了得到baseline结果（使用$l_1$-smooth损失），我们使用 ResNet-50 作为 backbone，其他训练所使用的参数与原先保持一致。当使用 IoU 和 GIoU 损失时，在最后的坐标改进阶段（而不是RPN阶段）使用 $\mathcal L_{IoU}, \mathcal L_{GIoU}$，与 YOLO v3 的情况一样，我们进行了极小的努力来进行正则处理。所有的实验中，简单的将 $\mathcal L_{IoU}, \mathcal L_{GIoU}$ 均乘以 10。

# 结论
介绍了 GIoU 作为新的指标来测量两个任意形状的距离。GIoU 继承了 IoU 的优秀特性且避免了 IoU 的缺点（非重叠情况），所以在基于 IoU 作为指标的 2D/3D 的计算机视觉任务中，GIoU 是一个很好的选择。

我们也提供了轴对齐的两个矩形之间 GIoU 的解析解。 GIoU 作为距离其导数/梯度可计算，故可以使用 GIoU 作为 bbox 回归损失。将 GIoU 损失结合进 sota 目标检测器，其检测性能在多个数据集上均一致得到提升。我们认为，指标自身就是针对指标的最优损失，GIoU 可以作为最佳 bbox 回归损失用于需要 2D bbox 回归的所有计算机视觉任务中。

# 后记
这篇文章主要是提出了一个新的损失来优化模型，文章通俗易懂，实在没什么可分析的，于是就写成了翻译，也算是一种阅读记录吧。