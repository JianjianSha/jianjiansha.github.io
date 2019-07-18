---
title: RepPoints
date: 2019-07-17 16:05:41
tags: object detection
mathjax: true
---
论文 [RepPoints: Point Set Representation for Object Detection](https://arxiv.org/abs/1904.11490)

大多数目标检测器使用 bbox 表示目标的位置，但是本文认为 bbox 对目标的位置描述较为粗糙，所以提出了一种新型的更加精确的目标表示方法 RepPoints（representative points），RepPoints 使用一组采样点进行目标定位和识别。通过训练给定目标的 gt 位置和分类，RepPoints 可以学习自动排布这些点使得这些点能勾画出目标边缘并在语义上确定目标所在区域。这是一种 anchor-free 的目标检测器，避开了 anchor 导致的搜索空间大和 anchor 难于设计等缺点。

如图 1，
![](/images/RepPoints_fig1.png)

RepPoints 使用一个点集，这个点集中各点位于目标边缘，并且语义上表明了目标所在的局部区域。~~RepPoints 与其他非矩阵表示的目标检测器区别在于其他目标检测器采用 bottom-up 的方式确定一些独立的点（比如角点或极点），然后依靠手工设计的聚合方法聚合这些点以获得预测 box，而 RepPoints 则是 top-down 的方式从输入 image/目标特征中学习并且能够端到端的训练。~~

# RepPoints
## BBox 表示
目标 bbox 可表示为 $\mathcal B=(x,y,w,h)$。我们回归一下 multi-stage 目标检测器，每经过一个 stage，目标定位都会得到调整，过程如下，

bbox anchors -(bbox reg)->  bbox proposals (S1)
             -(bbox reg)->  bbox proposals (S2)
             ...
             -(bbox reg)->  bbox object targets

开始时使用多个具有不同 scale 和 aspect ratio 的 anchors。anchor 中心处的特征（向量）用于预测目标分类的得分（二值分类，前景/背景），以及预测坐标偏差。调整后的 bbox 称为 proposal (S1)。在第二 stage，从 S1 中继续抽取特征，通常是 RoIpooling/RoIAlign，对于 two-stage 目标检测器，S1 中抽取的特征将用于最终的 box 的分类预测和坐标偏差预测。对于 multi-stage 目标检测器，S1 中抽取的特征用于预测生成 S2，逐次进行此过程，直到最后一个 stage 用于预测最终的 bbox target。

坐标回归预测为一个 4-d 向量 $(\Delta x_p, \Delta y_p, \Delta w_p, \Delta h_p)$，再结合 bbox proposal 的坐标 $\mathcal B_p=(x_p,y_p,w_p,h_p)$ 可解码出调整后的预测 bbox，
$$\mathcal B_r=(x_p+w_p \Delta x_p, \ y_p+h_p\Delta y_p, \ w_p e^{\Delta w_p}, \ h_p e^{\Delta h_p})$$

记目标的 gt 位置为 $\mathcal B_t=(x_t,y_t,w_t,h_t)$，gt 坐标偏差（gt target）为，
$$\hat {\mathcal F}(\mathcal B_p, \mathcal B_t)=(\frac {x_t-x_p} {w_p},\ \frac {y_t-y_p} {h_p},\ \log \frac {w_t} {w_p}, \ \log \frac {h_t} {h_p})$$

回归损失使用 smooth L1 损失。

## RepPoints
前文提到，4-d bbox 是一种比较粗糙的目标位置表示，不能反映目标的性质、姿势以及语义上的区域。RepPoints 能够解决这些问题，使用一组自适应采样点，
$$\mathcal R = \{(x_k,y_k)\}_{k=1}^n$$
其中 n 为采样点数量，本文设置为 9。

RepPoints 的坐标调整可表示为，
$$\mathcal R_r = \{(x_k+\Delta x_k,\ y_k+\Delta y_k)\}_{k=1}^n \qquad (5)$$
其中 $\{(\Delta x_k,\ \Delta y_k)\}_{k=1}^n$ 表示新采样点与旧采样点之间的偏差，refine 旧采样点之后得到新采样点。

__RepPoints 变换到 gt box:__ $\mathcal {T: R}_P \rightarrow \mathcal B_P$，其中 $\mathcal R_P$ 表示目标 P 的 RepPoints，$\mathcal T(\mathcal R_p)$ 表示伪 box，变换函数考虑以下三种，
1. Min-max function  
   $\mathcal {T=T_1}$：根据所有 RepPoints 确定横轴和纵轴上的最小最大值，以得到 $\mathcal B_p$

2. Partial min-max function  
   $\mathcal {T=T_2}$：根据部分 RepPoints 确定横轴和纵轴上的最小最大值，以得到 $\mathcal B_p$

3. Moment-based function  
   $\mathcal {T=T_3}$：使用 RepPoints 的期望值和二阶矩（方差）来计算 $\mathcal B_p$ 的中心和 scale，其中 scale 需要乘上全局共享的可学习系数 $\lambda_x, \ \lambda_y$。使用坐标的均值作为 box 的中心，这一点不难理解，RepPoints 为目标边缘的点，故目标越大，RepPoints 坐标的方差越大，两者应该成正比，所以将方差乘以系数可得到目标 size。考虑到任意目标的 RepPoint 都不是固定的，所以系数可以全局共享，并且可通过 point loss 学习得到。

__RepPoints 的学习__ 学习过程由目标定位损失和目标分类损失驱动。对于定位损失，首先使用上述某个转换函数将 RepPoints 转换为 pseudo box，然后计算与 gt box 之间的距离，这里使用左上角和右下角的 smooth L1 距离作为定位损失。

# RPDet
类似地，multi-stage 中使用 RepPoints 的目标表示演进过程为，

object centers -(RP refine)-> RepPoints proposals(S1) -(RP refine)-> RepPoints proposals(S2) ... -(RP refine)-> RepPoints object targets

RPDet (RepPoints Detector) 结构如图 2，
![](/images/RepPoints_fig2.png)

其中 N 表示 RepPoints 的数量。

采用 FPN backbone，得到的 feature maps，一支经过 3x3 卷积得到 offset field，这是一个与输入 feature maps 相同 spatial size 的 2N-channel 的特征，每个空间位置的 feature vector 是 2N-d 的，表示这个位置处的 offsets，offsets 是用于 deformable conv，同时也是 RepPoints，相当于给 deformable conv 的 offsets 赋予了物理含义，变换这个 offsets 得到 pseudo box，与 gt box 之间的左上和右下两点的 point loss（smooth L1 损失）用于优化这一分支。

得到的 offsets 用于 deformable conv，然后再经过卷积得到第二组的 offsets，根据 (5) 式得到 refinement 之后的 RepPoints（也就是说第二组的 offsets 其实是 $\Delta x, \Delta y$？），表示最终的目标定位，另一分支经过卷积得到目标分类 score maps。

## 与可变形 RoI pooling 的关系
可变形卷积和可变形 RoI pooling 用于改善特征抽取，而 RepPoints 是一种灵活的目标几何表示方式，可以抽取语义特征，从而准确地定位目标。作者认为，可变形 RoI pooling 无法学习到表示目标精确位置的采样点，原因为：假设可以学习到目标位置的几何表示，那么对于同一目标的两个靠的很近的 proposals，可变形 RoI pooling 将生成相同的特征，这表示目标检测器会失败；然而，可变形 RoI pooling 已经通过实验证明可以区分两个靠的很近的 proposals，这说明，可变形 RoI pooling 无法学习到目标的准确位置表示。

# 实验
略

# 结论
本文比较晦涩难懂，需要多读几遍，并且期待作者放出源码。