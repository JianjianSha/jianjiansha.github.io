---
title: loss
date: 2019-07-16 17:32:26
tags: CV
mathjax: true
---
总结一些常见的损失（虽然我把本文归类到 CV，但实际上这些损失函数并不仅仅用于 CV 中，只是目前我只关注 CV 而已）
<!-- more -->
# 1. Cross-Entropy Loss
交叉熵损失常用于分类任务中，比如共有 C 中可能的分类，（softmax 之后的）预测向量为 $P=(p_1,...,p_C)$，其中 $p_i$ 表示分类为 i 的概率，且有 $\sum_i^C p_i=1$，目标真实分类为 c，那么 gt target 为 $T=(t_1,...,t_C)$，其中
$$t_i=\begin{cases} 1 & i=c \\\\ 0 & i\ne c \end{cases}$$
于是交叉熵损失为
$$CE=-\sum_{i=1}^C t_i \log p_i$$

## 1.1 Binary Cross-Entropy Loss
特别地，当分类数量 C=2 时，目标为正的预测概率为 p，真实分类为 t，$t \in \{0,1\}$，
$$CE=-t \log p - (1-t) \log (1-p)$$
为方便起见，记
$$p_t=\begin{cases} p & t=1 \\\\ 1-p & t=0 \end{cases}$$
于是，
$$ CE=-\log p_t $$

## 1.2 Balanced Cross-Entropy Loss
如果样本分类分布不均（long-tail distribution），即少数分类的占据了绝大多数样本，而其他分类的样本数量则非常少，比如二分类中，分类为 1 的样本很少而分类为 0 的样本很多，那么从分类为 1 的样本中学习到的信息就有限，或者说分类为 1 的样本对损失贡献较小从而对优化过程作用较弱，故引入权重因子，t=1 具有权重 $\alpha$，t=0 具有权重 $1-\alpha$，$\alpha \in [0,1]$。实际操作中，设置 $\alpha$ 反比例于分类样本频次，或将 $\alpha$ 作为超参数通过交叉验证设置其值（RetinaNet 中设置为 0.25）。于是平衡交叉熵损失为，
$$CE=-\alpha_t \log p_t$$

## 1.3 Focal Loss
虽然 balanced cross-entropy loss 中 $\alpha$ 平衡了正负样本，但是并没有区分简单样本和困难样本，我们知道 $p_t \gg 0.5$ 属于简单样本，当简单样本数量很多时，其贡献的总损失不容忽视，显然，我们更应该重视困难样本，因为从困难样本中更能学习到有用（对模型至关重要的）信息，所以，降低简单样本的损失权重，比如这里的 Focal loss，
$$FL=-(1-p_t)^{\gamma} \log p_t \ , \ \gamma \ge 0$$
其中 $(1-p_t)^{\gamma}$ 称为调节因子。

Focal loss 的性质：
1. $p_t$ 较小，表示误分类，困难样本，此时 $(1-p_t)^{\gamma}$ 相对较大
2. $p_t$ 较大，表示分类正确，简单样本，此时 $(1-p_t)^{\gamma}$ 相对较小

# 2. MSE
均方误差为
$$MSE = \frac 1 n \sum_{i=1}^n (Y_i-\hat Y_i)^2$$
表示 n 个样本的 L2 范数误差的平均，其中 $Y_i, \hat Y_i$ 分别表示第 i 个样本的真实值和预测值。

## 2.1 L2 Loss
$$L_2=(Y_i-\hat Y_i)^2$$
缺点：当 $|Y_i-\hat Y_i|>1$ 时，误差会被放大很多，导致模型训练不稳定。
## 2.2 L1 Loss
$$L_1=|Y_i-\hat Y_i|$$
缺点：当 $|Y_i-\hat Y_i|<1$ 时，梯度（的绝对值）不变，导致优化过程出现震荡。
## 2.3 Smooth L1 Loss
结合以上两点，得到 Smooth L1 损失，
$$L=smooth_{L_1}(Y_i-\hat Y_i)
\\\\ smooth_{L_1}(x)=\begin{cases} 0.5 x^2 & |x|<1
\\\\ |x|-0.5 & otherwise \end{cases}$$

## 2.4 Regularized Loss
机器学习中，为防止过拟合加入正则项损失，通常是参数的 L1 范数或 L2 范数，略。


# 3. IOU 

## 3.1 IOU loss

最后用于预测的特征平面记为 $(C+1+4, H, W)$，其中 $C+1$ 为前景分类和一个背景。$4$ 表示当前 pixel 距离预测框的上下左右的距离。IOU 损失用于坐标回归任务，与坐标 L2 损失不同在于：

1. 大 box 的L2 损失比小 box 的 L2 损失惩罚更大，导致网络更加专注于优化大目标而忽略小目标

用于预测的特征平面上每个 pixel 作为一个样本，当 pixel 在 gt box 内部时，那么这个 pixel 为正样本，只考虑正样本的 IOU 损失。

假设某个 pixel 的坐标 $(i,j)$ ，位于 gt box 内，这个 pixel 处的距离偏差预测 $\mathbf p=(x _ l, x _ t, x _ r, x _ b)$，这个 pixel 距离 gt box 的上下左右距离为 $\mathbf y = (y _ l, y _ t, y _ r, y _ b)$，那么计算 IOU 如下

$$\begin{aligned} X &= (x _ t + x _ b) *(x _ l + x _ r)
\\\\ Y &=(y _ t + y _ b) * (y _ l + y _ r)
\\\\ I _ h &= \min (x _ t, y _ t) + \min(x _ b, y _ b)
\\\\ I _ w &= \min (x _ l, y _ l) + \min(x _ r, y _ r)
\\\\ I &= I _ h * I _ w
\\\\ U &= X + Y - I
\\\\ IoU &= I/U
\end{aligned}$$

于是 IOU 损失为

$$L = -\log IoU$$

或者使用 IoU 距离作为损失

$$L = 1- IoU$$

相较于L2损失，IoU损失将bbox作为一个整体去优化，并且 IoU损失范围归一化为[0,1]，所以没有L2损失中由于box尺寸不同损失惩罚力度也不同的问题。

IOU 损失用在 UnitBox 模型中，这个模型用于检测人脸图片，通常一个图片中只有一个人脸，也就是说只有一个 gt box，故不存在某个 pixel 同时位于两个 gt boxes 内部这个问题。

## 3.2 GIOU loss

泛化 IOU 损失。IOU 为

$$IoU = \frac {|A \cap B|}{|A \cup B|} \tag{3.1}$$

其中 $|\cdot|$ 表示求面积（二维）或者体积（三维）。

当 $|A \cap B| = 0$ ，即 region A 和 region B 没有重叠部分时，A 和 B 靠的近还是远，无法通过 IOU 判断，为了解决此问题，提出泛化 IOU ，即 GIOU，算法描述如下，

输入：两个任意凸形 $A, B \subseteq S \in \mathbb S ^ n$

输出： GIOU

1. 在 S 空间寻找包含 A 和 B 的最小凸形 C
2. 计算 IOU，按 (3.1) 式计算
3. 计算 GIOU

    $$GIoU = IoU - \frac {|C \backslash (A \cup B)|}{|C|} \tag{3.2}$$

使用 GIOU 距离作为损失，

$$L = 1- GIoU$$

**比较**

IOU 的范围 $[0, 1]$，GIOU 的范围 $[-1, 1]$ 。

**应用**

将 GIOU 应用到 YOLO，RCNN 中，将原来的 Smooth-L1 或者 MSE 替换为 GIOU 损失，正负例选择策略与模型各自原来的策略相同。使用 GIOU 损失，主要是 GIOU 损失比 L1 和 MSE 更能反应预测 box 与 gt box 匹配程度。


## 3.3 DIOU loss

DIOU 考虑了目标中心点距离，记预测 box 中心点坐标为 $\mathbf b$，gt box 中心点坐标为 $\mathbf b ^ {gt}$，那么 DIOU 距离（损失）为

$$L = 1 - IoU + \frac { \rho ^ 2 (\mathbf b, \mathbf b ^ {gt})}{c ^ 2} \tag{3.3}$$

其中 $c$ 是包含预测 box 和 gt box 的最小框的对角线长度，$\rho (\cdot, \cdot)$ 是求欧氏距离。

## 3.4 CIOU loss

CIOU 考虑了目标中心点距离和 aspect ratio，

$$L = 1 - IoU + \frac { \rho ^ 2 (\mathbf b, \mathbf b ^ {gt})}{c ^ 2} + \alpha v \tag{3.4}$$

其中 $\alpha$ 是 trade-off 参数，$v$ 是预测 box 与 gt box 的 aspect ratio 一致性测度，

$$\begin{aligned} \alpha &= \frac v {1- IoU + v} 
\\\\ v &=\frac 4 {\pi ^ 2} \left(\arctan \frac {w ^ {gt}}{h ^ {gt}} - \arctan \frac w h\right) ^ 2
\end{aligned}$$