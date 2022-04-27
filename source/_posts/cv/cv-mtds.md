---
title: CV 中的常用方法总结
date: 2019-06-24 17:33:22
tags: CV
mathjax: true
---
总结 CV 中的一些概念和操作（并不局限于 CV）。
<!-- more -->
# RF
第 k layer 上的感受野大小为
$$l_k=l_{k-1}+[(f_k-1)\prod_{i=0}^{k-1}s_i]$$
其中，$s_i$ 为第 i layer 上卷积的步幅，$f_k$ 为第 k layer 上卷积核大小，$l_0=1, \ s_0=1$。
# NMS
以 Faster R-CNN 为例，Test 阶段时，ProposalLayer 生成密集均匀分布的 anchors，RPN 得到所有 anchors 的得分（置信度）以及偏差回归值，根据偏差值对 anchors 进行坐标转换得到 proposals，proposals 的得分就是对应 anchors 的得分，然后经过如下处理：
1. proposal 的坐标不能超过输入 image 的范围 [0,w-1], [0,h-1]，故需要对超过范围的 proposal 进行 clip 以使得 proposal 坐标位于范围内
2. 过滤极小尺度的 proposal，proposal 对应在原始 image 上的 box 尺度必须大于 16（配置值）
3. 按 proposals 得分倒排，保留 top N1 的 proposals（N1 为配置值）
4. 非极大抑制 NMS
5. 按 proposals 得分倒排，保留 top N2 的 proposals（N2 为配置值）

NMS 过程如下：
1. 对于所有的 proposals 列表 P，计算其面积列表 A，根据 proposals 的得分倒排得到其列表下标 I。最终要保留的 proposals 的列表下标将被保存到 K 中
2. 找到当前得分最高的 proposal，其列表下标为 I[0]，将其添加到最终需要保留的 K 中，`K.append(I[0])`。计算当前 I 中与此最高得分的 proposal 的 IOUs，从 I 列表中移除 IOU 大于阈值（配置值）的那些 proposals 的下标值（注意，包括 I[0] 处的 proposal 也被移除，因为 I[0] 已经添加到 K 中）
3. 重复过程 2，直到当前 I 为空 
4. K 中保存了 NMS 之后的 proposals 的列表下标值

# Soft-NMS
NMS会过滤到两个靠的很近的 boxes 中得分较低的那个 box，但是有时候确实是存在两个靠的很近的 gt boxes，强行过滤到得分较低的 box 会导致 recall 较低，所以此时可改用 soft-NMS，源于论文 [Improving Object Detection With One Line of Code](https://arxiv.org/pdf/1704.04503.pdf)

Soft-NMS 与 NMS 的最主要区别是 NMS 将近邻低得分 box 重置其得分为 0，而 Soft-NMS 则是根据一个函数降低其得分，使得近邻 box 的置信度更低，但仍然在检测 rank list 中。算法如下：


__Input__
   * $\mathcal B=\{b_1,...,b_N\}, \mathcal S = \{s_1,...,S_N\}, N_t, m$
   * 分别表示初始检测 boxes，相应的 scores，NMS 阈值（0.7）， $m=1 \rightarrow \text{NMS}; \ m=2\rightarrow \text{Soft-NMS}$

$\mathcal D \leftarrow \{\}$

__while__ $\mathcal B \ne \varnothing$ __do__

&emsp; &emsp; $m \leftarrow \arg \max \mathcal S$

&emsp; &emsp; $\mathcal M \leftarrow b_m$

&emsp; &emsp; $\mathcal D \leftarrow \mathcal {D \cup M}; \mathcal B \leftarrow \mathcal{B-M}$

&emsp; &emsp; __for__ $b_i \in \mathcal B$ __do__

&emsp; &emsp; &emsp; &emsp; __if__ $iou(\mathcal M, b_i) > N_t$ __then__

&emsp; &emsp; &emsp; &emsp; &emsp; &emsp; __if__ $m=1$ __then__

&emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; $\mathcal {B \leftarrow B} - b_i; \mathcal {S \leftarrow S} - s_i$

&emsp; &emsp; &emsp; &emsp; &emsp; &emsp; __else if__ $m=2$

&emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; $s_i \leftarrow s_i f[iou(\mathcal M, b_i)]$

&emsp; &emsp; &emsp; &emsp; &emsp; &emsp; __end__

&emsp; &emsp; &emsp; &emsp; __end__

&emsp; &emsp; __end__

__end__

__return__ $\mathcal {D,S}$

现在来看 Soft-NMS 中的得分衰减因子 f 函数，首先无论 NMS 还是 Soft-NMS，我们都可以统一可将 box 的得分修改为 $s_i=s_i f(i)$，其中 f 函数为，
1. NMS
   $$f(i) = \begin{cases} 1 & iou(\mathcal M, b_i) < N_t \\ 0 & iou(\mathcal M, b_i) \ge N_t\end{cases}$$

2. Soft-NMS
   
   $$f(i) = \begin{cases} 1 & iou(\mathcal M, b_i) < N_t \\ 1-iou(\mathcal M, b_i) & iou(\mathcal M, b_i) \ge N_t \end{cases}$$
   可见，靠的越近 iou 越大，得分衰减的越厉害。但是这个式子中函数值在 iou=N<sub>t</sub> 处附近不连续，可使用高斯惩罚函数，
   $$f(i)=e^{-\frac {iou(\mathcal M, b_i)^2} \sigma}, \ \forall b_i \notin \mathcal D$$
   在 iou=0 时取得最大值，参数 $\sigma$ 控制衰减速度，$\sigma$ 越小表示得分随 iou 增大而衰减越快。

# Deconvolution
在很多 CV 任务例如 semantic segmentation 中，需要上采样，简单的上采样可以使用 bilinear interpolation，但有时为了得到更好的上采样结果会使用反卷积。
(to be continued...)