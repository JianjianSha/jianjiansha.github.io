---
title: Generalized Focal Loss
date: 2023-08-13 14:38:03
tags: object detection
mathjax: true
---

论文：[Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection](https://arXiv.org/abs/2006.04388)

# 1. 简介

当前目标检测包含两个部分：分类（通常使用 Focal Loss），和坐标回归（使用 Dirac delta 分布，就是坐标在中心的概率密度极大，趋于无穷大，这里的中心指的是 target 值）。当然也有使用一个独立的分支预测定位质量，例如 FCOS 中预测 centerness score，然后将 centerness score 与分类 score 相乘作为最后的得分，最后根据这个最后得分进行 NMS 。然而，这会带来以下问题：

**问题1：定位质量得分与分类得分在训练和推理两个阶段的不一致**

1. 训练阶段，定位质量（例如 centerness score，或者 IoU 得分）与分类得分是分开训练的，但是在推理阶段却组合（相乘）使用，这会导致前后不一致问题，也就是说，乘积最大的 anchor point，不一定是 centerness score 最大的点，或者不是分类得分最大的点。如图 1，

    ![](/images/obj_det/gfl_1.png)
    <center>图 1. </center>

2. 定位质量估计仅分配了正样本，然而负样本也可能有较大的质量预测结果。如图 2 (a) ，

    ![](/images/obj_det/gfl_2.png)
    <center>图 2. </center>

    图 2 (a) 中，左图左下角预测 box 的 cls 得分为 `0.095`，IoU 得分为 `0.927`，右图的右上角 cls 得分为 `0.101`，IoU 得分为 `0.913`，显然这两个预测应该均为负样本，然而其 IoU 得分非常大，所以推理阶段可能会因为两种得分的乘积较大而被判为正，从而 NMS 之后抑制了其他正样本。

**问题2：BBox 表征不灵活**

bbox 表征使用 dirac delta 分布，中心为 ground truth target ，但是数据集中有些目标边界是模糊不清的，如图 3，

![](/images/obj_det/gfl_3.png)
<center>图 3. 由于遮挡，阴影，模糊等原因，目标边界不够清晰，导致 gt labels（白色 box）有时不可信，故 Dirac delta 分布不足以解决这些问题。</center>

最近部分研究工作使用高斯分布对 bbox 建模，但是高斯分布过于简单，不足以捕获 bbox 的实际分布。实际上目标 bbox 的分布比较任意，且非常灵活，而不像高斯分布那样对称。

为了解决以上问题，本文提出 bbox 以及定位质量的一种新表征。

**定位质量表征**

将分类得分与定位质量得分合并到一起：一个分类向量，在 gt 分类 index 位置处的向量元素表示定位质量（本文使用 IoU，即预测 box 与 gt box ），称之为 “分类-IoU 联合表征” ，如图 1 (b) 所示，于是，可以消除 训练-推理 两个阶段的不一致。图 2 (b) 显示了随机采样的点的分类得分和定位得分（IoU），其中绿色点表示使用本文提出的 “分类-IoU 联合表征” 方法，由于这种表征使得分类得分与 IoU 相等，所以是一条直线；而以前的方法中，分类得分和 IoU 是弱关联的，图 2 (a) 左边两个图的两个区域 A、B 对应 (b) 图中红色圈内的 A、B 两点，显然 IoU 值较高，而分类得分较低。

**bbox 表征**

学习 bbox 的任意型分布，而不使用较强的先验知识。如图 3 所示。


以前分类损失使用 Focal Loss，可以解决分类不平衡问题，然而对于 “分类-IoU 联合表征”，由于 IoU label 是连续型 $[0,1]$，而 FL 当前仅处理二分类 label $\lbrace 0, 1\rbrace$，所以本文提出 Generalized Focal Loss（GFL）。

GFL 可以具体化为 Quality Focal Loss（QFL）和 Distribution Focal Loss（DFL）。

# 2. 方法

回顾一下 **Focal Loss**

$$FL(p) = -(1-p _ t) ^ {\gamma} \log p _ t \tag{1}$$

其中，

$$p _ t = \begin{cases} p & y=1 \\\\ 1 - p & y=0
\end{cases}$$

**Quality Focal Loss (QFL)**

![](/images/obj_det/gfl_4.png)
<center>图 4.</center>

图 4 左边是以前的分类和回归分支。右边是 GFL ，将分类得分和定位质量（IoU）结合起来。$y=0$ 表示负样本，$0 < y \le 1$ 表示正样本，其 target IoU 得分为 $y$ 。

对每个分类独立进行二分类预测，所以最后激活函数使用 sigmoid 函数 $\sigma$ 。

仿照 FL 的形式，有

$$QFL(\sigma) = -|y - \sigma| ^ {\beta} ((1-y)\log (1-\sigma) + y \log \sigma) \tag{2}$$

其中 $\sigma$ 就类似于 FL 中的 $p$，$y$ 是连续型 label 而非二值类型。

**Distribution Focal Loss (DFL)**

传统的 bbox 回归模型将回归 label $y$ 看作是一个 dirac delta 分布 $\delta(x-y)$，满足 $\int _ {-\infty} ^ {+\infty} \delta(x -y)=1$，预测结果由 FC layer 输出。于是，按如下方法恢复 $y$ ，

$$y = \int _ {-\infty} ^ {+\infty} \delta(x-y) x dx \tag{3}$$

现在，我们不使用 Dirac delta 分布或者 高斯分布作为对回归 label 的建模，而是直接学习一般类型的分布 $P(x)$ 。

假设 $y$ 范围为 $y _ 0 \le y \le y _ n$，那么模型的估计值为

$$\hat y = \int _ {-\infty} ^ {+\infty} P(x) x dx = \int _ {y _ 0} ^ {y _ n} P(x) x dx \tag{4}$$

为了方便神经网络处理，将连续域积分转为离散和。将范围 $[y _ 0, y _ n]$ 离散化为一个集合 $\lbrace y _ 0, y _ 1, \ldots, y _ {n-1}, y _ n \rbrace$，连续两个值间隔相等记为 $\Delta$（本文为简单起见，使用 $\Delta =1$），于是根据概率分布和为 `1` 可知 $\sum _ {i=1} ^ n P(y _ i)=1$，回归估值值则为

$$\hat y = \sum _ {i=0} ^ n P (y _ i) y _ i \tag{5}$$

因此，分布 $P(x)$ 可以使用 $n+1$ 个输出单元实现，即最后的 layer 输出单元数量为 $n+1$ ，layer 输出还需要使用 softmax 使得输出值归一化。记 $P(y _ i) = \mathcal S _ i$ 。

但是 $P(x)$ 的离散分布有无数种可使得 (5) 式计算结果 $\hat y$ 逼近 $y$ ，如图 5 左边所示，分布 (3) 比 分布 (1) 和 (2) 更窄，对 bbxo 预测的置信度更高，这促使我们通过鼓励在 $y$ 附近有较大的概率来优化 $P(x)$ 的形状。本文提出 DFL ，目的是使得网络快速聚焦到 $y$ 附近，$y$ 附近的两个值 $y _ i \le y \le y _ {i+1}$，即，增大 $y _ i, y _ {i+1}$ 的概率。

由于学习 BBOX 仅针对正样本，所以没有类别不平衡问题，所以去掉 QFL 中不平衡调谐模块，从而得到 DFL 定义，

$$DFL(\mathcal S _ i, \mathcal S _ {i+1}) =-((y _ {i+1}-y)\log \mathcal S _ i + (y - y _ i)\log \mathcal S _ {i+1}) \tag{6}$$

(6) 式中，$y _ {i+1} - y > 0$，$y - y _ i > 0$，且 $y _ i , y _ {i+1}$ 是最靠近 $y$ 的两个值，故 $y _ {i+1} - y, \ y - y _ i$ 是最小的两个值，为了使得 DFL 尽量小，那么 $\mathcal S _ i$ 和 $\mathcal S _ {i+1}$ 

$$\mathcal S _ i + \mathcal S _ {i+1} = 1$$

另外一方面，为了使得 $\hat y = y$，那么 

$$\hat y = \sum _ {j=0} ^ n P(y _ j)y _ j = \mathcal S _ i y _ i + \mathcal S _ {i+1} y _ {i+1} = y$$

结合以上两式可知，

$$\mathcal S _ i = \frac {y _ {i+1}-y}{y _ {i+1}-y _ i}, \quad \mathcal S _ {i+1} = \frac {y - y _ i}{y _ {i+1}-y _ i}$$

当然，根据 (6) 式可知 DFL 不可能最小为 0，因为右端 $\log(\cdot)$ 项为 负，$\log(\cdot)$ 项的系数 $y _ {i+1}-y > 0, \ y - y _ i > 0$，所以右端应该为正，不可能为 0 。 

