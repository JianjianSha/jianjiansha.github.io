---
title: Loss 2
p: pytorch/loss_2
date: 2021-01-13 16:54:38
tags: PyTorch
mathjax: true
---
继上一篇 [loss 1](2021/1/12/pytorch/loss_1)，本篇介绍 PyTorch 的其他损失。
<!-- more -->


# MarginRankingLoss
给定两个输入 $x_1, \ x_2$，以及一个 label 值 $y \in \{1,-1\}$。当 $y=1$，认为 $x_1$ 应该比 $x_2$ 大；当 $y=-1$，认为 $x_1$ 应该比 $x_2$ 小，所以损失为
$$l=\max(0, -y(x_1-x_2) + \text{margin})$$
上式中增加了一个 `margin` 项，根据
$$-y(x_1-x_2)+\text{margin} \le 0$$

当 $y=1$ 时，需要满足 $x_1\ge x_2+\text{margin}$ 损失才降为 0。

当 $y=-1$ 时，需要满足 $x_1+\text{margin} \le x_2$ 损失才降为 0。

适用于（二）<b>分类</b>问题。
# MultiLabelMarginLoss
适用于多标签多分类问题。每个类别独立进行二分类（为正 or 为负）预测，预测值 x 是一个 2D tensor，shape 为 $(N,C)$，其中 $N$ 表示批大小，$C$ 表示类别数。target 与 x 同 shape。暂且考虑单个样本，此时 x 和 target 均为长度 `C` 的向量，x 表示各分类的预测概率，target （用 y 表示）表示样本所属分类索引，例如 $y=(3,0,-1,1)$，表示样本属于 `0` 分类和 `3` 分类，从第一个负值开始，之后的全部忽略。借鉴 `MarginRankingLoss` 思想，对于预测值 x，认为其中<b>样本所属分类的元素值比样本不属分类的元素值大</b>，这个例子中，样本所属分类为 $\{0,3\}$，所以认为应该是 $x_0,\ x_3 > x_1,\ x_2$，据此不难理解单个样本的损失为
$$l=\sum_{i,j} \frac {\max[0, 1-(x_{y_j} - x_i)]} C$$
其中，$j \in \mathcal J=\{0,1,...,k-1\}$，且 $y_k<0$，$i \in \{0,1,...,C-1\}-\{y_j|j \in \mathcal J\}$，即， $j$ 为 target 向量中开始的连续非负元素索引，$y_j$ 表示样本所属分类索引，i 为样本不属分类索引。

当分类正确时，损失为0，此时需要满足条件 $1-(x_{y_j}-x_i)\le 0 \Rightarrow x_{y_j}\ge 1+x_i$，这说明降低损失会使得样本所属分类的预测概率 $x_{y_j} \rightarrow 0$，样本不属分类的预测概率 $x_i \rightarrow 0$。在 test 阶段，对预测值 x 设置一个低阈值即可。

# SoftMarginLoss
适用于二分类问题。上面两种 MarginLoss 均采用了 `max(0,x)` 函数，这个函数在 `x=0` 处不可导。`SoftMarginLoss` 借助 logistic 函数解决了这个问题。Logistic 函数
$$\sigma(x)=\frac 1 {1+\exp (-x)}$$
预测值 x，分类 $y\in \{1,-1\}$，似然函数为
$$\mathcal L =\mathbb I(y=1)f(x)+\mathbb I(y=-1)(1-f(x))=[1+\exp(-yx)]^{-1}$$
 负对数似然函数（损失）为
$$l= \log(1+\exp(-yx))$$
所以 `SoftMarginLoss` <b>就是 logistic 回归的负对数似然损失</b>。预测输入 input tensor 的 shape 为 $(*)$，其中 $*$ 表示任意维度，target 与 input 的 shape 相同。损失按像素计算，输出与 input 同 shape，如果按求和或平均归约，那么输出为一标量。

# MultiLabelSoftMarginLoss
适用于多标签多分类问题。每个类别各自独立做二分类（为正或负）。input 和 target 有相同的 shape：$(N,C)$，target 值为 0 或 1（这与 SoftMarginLoss 的 1 或 -1 竟然不统一）。于是，单个样本的损失为，
$$l=-\frac 1 C \sum_{i=1}^C y_i \log \left(\frac 1 {1+\exp(-x_i)}\right )+(1-y_i)\log \left(\frac {\exp(-x_i)} {1+\exp(-x_i)}\right)$$
由于这里考虑单个样本，所以上式 $x, \ y$ 均为长度 $C$ 的向量，由于 y 值取值范围不同，所以上式与 `SoftMarginLoss` 的损失表达式略有不同，但是本质上都是 logistic 负对数似然损失。

输出 tensor 的 shape 为 $(N,)$，如果按求和或平均归约，那么输出 tensor 为一标量。

类签名：
```python
MultiLabelSoftMarginLoss(weight: Optional[torch.Tensor] = None, size_average=None, reduce=None, reduction: str='mean')
```
`weight` 如果给定，那么是一个 shape 为 $(C,)$  的 tensor，用于给每个 channel/类别 一个权重。

# MultiMarginLoss
适用于多分类（单标签）问题。input 为一个 2D tensor $(N,C)$，target shape 为 $(N,)$，表示样本的分类索引，故 $y_i \in \{0,1,...,C-1\}$，对于单个样本而言，此时输入为一个长度 C 的向量 x，target 为标量，也记为 y，表示样本分类索引，显然我们要 $x_y > x_i$，其中 $i \neq y$，margin-based 损失为
$$l=\frac {\sum_{i \neq y} \max(0, d-(x_y-x_i))^p} C$$
其中 $d$ 为 margin，也就是说需要 $x_y \ge x_i+d$，样本所属分类的预测概率比其他分类的预测概率大 $d$，损失才为 0。

p 值可为 1 或 2，用于控制损失变化速度。还可以给每个类型增加一个权重，此时损失为，
$$l=\frac {\sum_{i \neq y} \max[0, w_y(d-(x_y-x_i))^p]} C$$
注意，权重 $w_y$ 不参与幂运算，且只有样本所属分类对于的权重因子起作用。

类签名：
```python
MultiMarginLoss(p: int=1, margin: float=1.0, weight: Optional[torch.Tensor]=None, size_average=None, reduce=None, reduction: str='mean')
```
input shape 为 $(N,C)$，target shape 为 $(N,)$，output 的 shape 为 $(N,)$，可以按求和或平均归约，此时 output 为一标量。

# TripletMarginLoss
三个tensor：$a, \ p, \ n$，分别表示 anchor，正例和负例，shape 均为 $(N,D)$，其中 $N$ 为批大小，$D$ 为特征数。p 表示与 a 同类的另一个样本的特征，n 表示与 a 不同类的样本特征，显然，需要 p 与 a 的特征尽量相近，n 与 a 的特征尽量远离。
传统上是以 pair 的形式来度量损失，即 $(p,a)$ 为正例对，$(n,a)$ 为负例对，一般表示为 $(x_1, x_2， l)$，当 $l=1$ 表示是正例对，$l=-1$ 表示是负例对，此时损失定义为
$$l=\begin{cases} \Vert \mathbf x_1-\mathbf x_2 \Vert_2 & l=1 \\ \max(0, d-\Vert \mathbf x_1- \mathbf x_2\Vert_2) & l=-1 \end{cases}$$
$l=1$ 是正例对，所以 $\mathbf x_1$ 应该要尽量接近 $\mathbf x_2$；$l=-1$ 是负例对，$\mathbf x_1$ 尽量要远离 $\mathbf x_2$，且要相距 $d$ 以上。

这里 `TripletMarginLoss` 将 `(a,p,n)` 三者当成一个整体，margin ranking-based 损失定义如下，
$$l=\max[d(a,p) - d(a,n)+d_0, 0]$$
$$d(\mathbf x_1, \mathbf x_2)=\Vert \mathbf x_1 - \mathbf x_2 \Vert_p$$
其中，$d_0$ 为 margin，计算特征空间中的距离时，使用的是 p 范数，这个 p 与前面正例 p 不一样，根据上下文不难区分。

类签名：
```python
TripletMarginLoss(margin: float=1.0, p: float=2.0, eps: float=1e-06, swap: bool=False, size_average=None, reduce=None, reduction: str='mean')
```
`swap` 指示是否交换 anchor 和 positive，这用于 hard negative mining。若 `swap=True`，那么 $d(a,n) = d(p,n)$，也就是说，使用 `(p,n)` 的距离作为 negative 与 anchor 的距离。

forward 方法的参数为 anchor, positive 和 negative 三个特征 tensor，shape 均为 $(N,D)$，输出 tensor 的 shape 为 $(N,)$，如果按求和或平均归约，那么输出为一标量。

更多细节可以参考 [`Learning local feature descriptors with triplets and shallow convolutional neural networks`](www.bmva.org/bmvc/2016/papers/paper119/paper119.pdf)

# TripletMarginWithDistanceLoss
`TripletMarginLoss` 中距离使用的是 p 范数，这里是通过参数提供自定义的距离参数。anchor，positive 和 negative 三个 tensor 的 shape 为 $(N,*)$ ，其中 $*$ 为任意维度，输出 tensor 的未归约 shape 为 $(N,)$，否则为一标量。

# HingeEmbeddingLoss
$x$ 表示距离（例如 L1 范数），$y \in \{1,-1\}$ 标识是相似还是相反，损失为，
$$l = \begin{cases} x & y=1 \\ \max(0, d-x) & y=-1 \end{cases}$$
其中 $d$ 为 margin。

输入 x 和 y 的 shape 均为任意维度 $(*)$，输出未归约的 shape 也是 $(*)$，否则为一标量。

# CosineEmbeddingLoss
`y=1` 表示两个（归一化）向量应该相近，`y=-1` 表示应该相差很远。
损失如下，
$$l=\begin{cases} 1- \cos(x_1,x_2) & y=1 \\ \max[0, \cos(x_1,x_2) - d] & y=-1 \end{cases}$$
其中 $d$ 表示 margin，默认为 0。

# CTCLoss
参考文献 [Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
RNN 相关的应用领域暂未涉及。略，以后填坑。