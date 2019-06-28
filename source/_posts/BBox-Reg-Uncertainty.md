---
title: BBox-Reg-Uncertainty
date: 2019-06-28 09:23:16
tags: object detection
mathjax: true
---
论文：[Bounding Box Regression with Uncertainty for Accurate Object Detection](https://arxiv.org/abs/1809.08545)
# 简介
大型目标检测集如 ImageNet，MS-COCO 和 CrowdHuman 等都致力于定义足够明确的 ground truth bounding box。但是有时候 gt bbox 的边界是不明确的，使得难以去打标签，也难以学习 bbox 回归函数（的参数），如图 1，
![](/images/BBox-reg_fig1.png) <center>Fig 1 MS-COCO 数据集中 gt bbox 不明确的情况。(a)(c) 标签不准确导致歧义；(b) 遮挡导致歧义；(d) 目标边界本身就不明确</center>

当前 SOTA 目标检测器如 Faster R-CNN，Cascade R-CNN 和 Mask R-CNN 等均依赖于 bbox 回归来定位目标。传统的 bbox 回归损失如 smooth-L1 没有考虑到 gt box 的不明确性，所以损失较大，并且认为分类得分越高时 bbox 回归越准确（应该说的是 Inference 阶段），但事实不总是如此，如图 2，分类得分高的 bbox 但是回归不够准确，回归不准确还是说明 __回归 loss 较大__。
![](/images/BBox-reg_fig2.png) <center>Fig 2 MS-COCO 上使用 VGG-16 Faster R-CNN 的失败案例。(a) 两个预测框均不准确；(b) 高分类得分 bbox 的左边界不准确</center>

为了解决以上问题，我们介绍一种新型 bbox 回归损失 KL loss，同时学习 bbox 回归和定位不确定性，从而使得 __回归 loss 较小__。学习 gt box 的不确定性肯定是针对整个数据集的，首先将预测 box 和 gt box 分别建模为 Gaussian 分布和 Dirac delta 函数。KL loss 定义为预测分布和 gt 分布之间的 KL 散度，我们知道 KL 散度用于衡量两个分布之间的距离（其实不满足距离的对称性，即不满足交换律）或者说差异，差异越大，KL 散度越大。假设目标分布为 P(x)，使用 Q(x) 去匹配目标分布，那么 KL 散度为
$$D_{KL}(P||Q)=\sum_{i=1}^N P(x_i) \log \frac {P(x_i)} {Q(x_i)}$$
这是离散分布的情况，对于连续分布则为，
$$D_{KL}(P||Q)=E_P \left[\log \frac {p(x)} {q(x)} \right]=\int p(x) \log \frac {p(x)} {q(x)} dx$$
注意，此时 p(x) 和 q(x) 表示概率密度而非概率。
显然如果 P,Q 完全匹配，那么 KL 散度达到最小值 0。

使用 KL loss 学习 bbox 回归有以下三个优点：
1. 可以成功捕获数据集中的不明确性，对于有歧义的 bbox，回归损失更小
2. 学习到的方差在后续处理中非常有用。我们提出 var voting (variance voting)，通过使用附近 box 的位置和位置方差来票选（加权平均）出当前候选 box 的位置。这么做是为了解决 Fig 2 中的问题
3. 学习到的概率分布是可解释的。由于分布反应的是预测 box 的不确定性，故在汽车自动驾驶或机器人等下游应用中非常有用



我们提出了 KL loss 和 var voting，为了验证这两者的通用性，我们使用了 PASCAL VOC 2007 和 MS-COCO 两个 benchmark，多个目标检测器包括 VGG-CNN-M-1024, VGG-16, ResNet-5-FPN 以及 Mask R-CNN（前两者属于 Faster R-CNN），实验证明使用我们提出的方法均提升了目标定位的准确率。

# 方法
## BBox 参数化
基于 Faster R-CNN 或 Mask R-CNN 如图 3，我们分别回归 bbx 的四条边坐标，即 Box 分支输出 shape 为 (N, 84)，其中 N 表示使用 proposals 的 batch size，84 是 21 个分类下 4 个坐标预测（这里以 PASCAL VOC 为例，共 21 个分类），Box std 分支输出 shape 也是 (N, 84)，表示 21 分类下 4 条边坐标分布的标准差 $\sigma$，坐标是分类相关的（not class-agnostic），前面简介部分所讲的将 box 建模为高斯分布，就是指四条边的坐标均为高斯分布，具体请往下看，
![](/images/BBox-reg_fig3.png)

令 $(x_1,y_1,x_2,y_2) \in \mathcal R^4$ 表示预测 bbox，那么偏差 $\{t_i| i=x_1,y_1,x_2,y_2\}$ 为：

$$t_{x_1}=\frac {x_1-x_{1a}} {w_a}, \quad t_{x_2}=\frac {x_2-x_{2a}} {w_a}
\\\\ t_{y_1}=\frac {y_1-y_{1a}} {h_a}, \quad t_{y_2}=\frac {y_2-y_{2a}} {h_a}
\\\\ t_{x_1}^{\ast}=\frac {x_1^{\ast}-x_{1a}} {w_a}, \quad t_{x_2}^{\ast}=\frac {x_2^{\ast}-x_{2a}} {w_a}
\\\\ t_{y_1}^{\ast}=\frac {y_1^{\ast}-y_{1a}} {h_a}, \quad t_{y_2}^{\ast}=\frac {y_2^{\ast}-y_{2a}} {h_a}$$

其中带 * 的为 gt offset，不带 * 的为预测 offset，$(x_{1a},y_{1a},x_{2a},y_{2a})$ 为 anchor box。后面的讨论中，由于各坐标独立进行优化，故我们统一使用 x 表示这四个坐标，x 取值为$\{x_1,y_1,x_2,y_2\}$。

我们的网络不仅仅预测 bbox 的定位，还预测其概率分布。这种分布可以是复杂的如多变量高斯分布或混合高斯分布，但是本文为了简单起见，我们假定各坐标互相独立，故使用单变量高斯分布，
$$P_{\Theta}(x)=\frac 1 {\sqrt {2 \pi \sigma^2}}e^{- \frac {(x-x_e)^2} {2 \sigma^2}}$$
其中 $\Theta$ 是可学习的参数，$x_e$ 是 bbox 位置估计，标准差 $\sigma$ 衡量位置估计的不确定性，越大越不确定。当 $\sigma \rightarrow 0$，表示网络对 bbox 位置估计非常十分自信。

~~（以 Faster R-CNN 为例说明，bbox 回归分支其实是两组输出 blob，分别使用两个全连接层得到，分别表示 4 个 坐标估计以及 4 个坐标分布的标准差，所以可以说，$\Theta$ 就是这两个全连接层的权重参数。这段话不一定准确，需要看源码待定）~~

gt box 也可以使用高斯分布，只是其中标准差无限趋于 0： $\sigma \rightarrow 0$，此时退化为 Dirac delta 函数，
$$P_D(x)=\delta(x-x_g)$$
其中 $x_g$ 是 gt box 位置 x 坐标。

## 使用 KL Loss 的 BBox 回归
最小化 $P_{\Theta}(x)$ 和 $P_D(x)$ 之间的 KL 散度来估计参数 $\hat \Theta$，即，使用 KL 损失优化网络参数，
$$\hat \Theta = \arg \min_{\Theta} \frac 1 N \sum D_{KL}(P_D(x)||P_{\Theta}(x))$$
其中 N 表示样本数量，x 表示 4 个坐标中的一个。KL 散度作为回归损失，而分类损失维持原来不变。
$$\begin{aligned} L_{reg} &=D_{KL}(P_D(x)||P_{\Theta}(x)) 
\\\\ &=\int P_D(x) \log P_D(x) dx - \int P_D(x) \log P_{\Theta}(x) dx
\\\\ &=-H(P_D(x))-\int P_D(x) \log \frac 1 {\sqrt {2 \pi \sigma^2}}e^{- \frac {(x-x_e)^2} {2 \sigma^2}} dx
\\\\ &=-H(P_D(x))+ \log \sqrt{2\pi \sigma^2}\int P_D(x) dx+\int P_D(x) \frac {(x-x_e)^2} {2 \sigma^2} dx
\\\\ &=\frac {(x_g-x_e)^2}{2\sigma^2}+\frac {\log \sigma^2} 2 + \frac {\log 2\pi} 2 - H(P_D(x))
\end{aligned}$$

其中，$H(P_D(x))$ 是 Dirac delta 分布的信息熵。

如图 4，
![](/images/BBox-reg_fig4.png)

当 box 位置 $x_e$ 估计不正确时，我们希望方差 $\sigma^2$ 更大，从而降低回归损失 $L_{reg}$。由于 $H(P_D(x)), \log (2\pi)/2$ 均与估计参数 $\Theta$ 无关，故有，
$$L_{reg} \propto \frac {(x_g-x_e)^2}{2\sigma^2}+\frac {\log \sigma^2} 2$$
当 $\sigma=1$，KL Loss 退化为标准的欧氏距离，
$$L_{reg} \propto \frac {(x_g-x_e)^2} 2$$
损失关于估计位置 $x_e$ 和定位标准差 $\sigma$ 可导，
$$\frac d {dx_e}L_{reg}=\frac {x_e-x_g} {\sigma^2}
\\\\ \frac d {dx_e}L_{reg}=-\frac {(x_e-x_g)^2} {\sigma^3} + \frac 1 \sigma$$

由于 $\sigma$ 位于分母上，所以训练初期可能会出现梯度爆炸，为了避免这种现象，在训练阶段，使用 $\alpha=\log \sigma^2$ 代替 $\sigma$，即，图 3 中 Box std 输出为 $\alpha$，此时
$$L_{reg} \propto \frac {e^{-\alpha}} 2 (x_g-x_e)^2+\frac \alpha 2$$
反向传播时使用 $L_{reg}$ 关于 $\alpha$ 的梯度。测试阶段，则将 $\alpha$ 转变为 $\sigma$，即测试阶段中，需要将 Box std 的输出经过 $\sigma=\sqrt{e^{\alpha}}$ 转换才能得到标准差。

当 $|x_g - x_e| > 1$ 时，我们参考 smooth-L1 改写回归损失，这是为了避免 $x_g,x_e$ 相差太多时，损失过大造成训练不稳定，于是最终有，
$$L_{reg} \begin {cases} \propto \frac {e^{-\alpha}} 2 (x_g-x_e)^2+\frac \alpha 2 & |x_g - x_e| \le 1
\\\\ = e^{-\alpha} (|x_g-x_e|-\frac 1 2 )+\frac \alpha 2 & |x_g - x_e| > 1 \end{cases}$$

根据以上分析可见，网络 bbox 回归分支输出两组数据，分别是预测位置 offset 以及位置分布标准差 $\sigma$。训练阶段，将预测 $\sigma$ 改为预测 $\alpha$，$\alpha$ 预测的那个全连接层参数使用随机 Gaussian 初始化，这个 Gaussian 使用标准差 0.0001，期望 0。

## Variance Voting
得到预测位置坐标的方差 $\sigma^2$ 后，根据附近 bbox 的位置方差票选出当前候选框的位置，这里附近是指与当前 box 有重叠（IoU>0）的 box。使用 Variance Voting 是为了解决 Fig 2 中的问题。算法如下，

__Algorithm 1__ var voting
*****
$\mathcal B$ 是 Nx4 的矩阵，表示初始检测 boxes

$\mathcal S$ 为相应的检测得分，是长度为 N 的一维向量

$\mathcal C$ 是相应的方差，也是一个 Nx4 的矩阵

$\mathcal D$ 为最终的检测结果集，$\sigma_t$ 是 var voting 的一个参数，其值可调整

$\mathcal B=\{b_1,...,b_N\}, \ \mathcal S=\{s_1,...,s_N\}, \ \mathcal C=\{\sigma_1^2,...,\sigma_N^2\}$

$\mathcal D \leftarrow \{\}, \ \mathcal T \leftarrow \mathcal B$

__while__ $\mathcal T \ne \varnothing$ __do__

- $m \leftarrow \arg\max \mathcal T$ （论文中为 $\arg \max \mathcal S$，但是我觉得不对）
- $\mathcal T \leftarrow \mathcal T - b_m$
- <font color='cyan'>$\mathcal S \leftarrow \mathcal S f(IoU(b_m, \mathcal T)) \qquad \qquad \qquad \qquad \ \ \triangleright$ soft-NMS </font>
- <font color='gree'>$idx \leftarrow IoU(b_m, B) > 0 \qquad \qquad \qquad \qquad \triangleright$    var voting </font>
- <font color='gree'> $p \leftarrow exp(-(1-IoU(b_m, \mathcal B[idx]))^2/\sigma_t)$ </font>
- <font color='gree'> $b_m \leftarrow p(\mathcal B[idx]/\mathcal C[idx])/p(1 / \mathcal C[idx])$</font>
- $\mathcal D \leftarrow \mathcal D \cup b_m$
 
__end while__

__return__ $\mathcal {D, S}$
***

我们已经知道，当前检测 box 的近邻 box 指与当前 box 的 IoU 超过一定阈值的 box。NMS 是移除得分较低的近邻预测 box ，soft-NMS 是 NMS 的修改版，将得分较低的近邻预测 box 重新修改为一个更低的得分，简单来讲就是得分低，则进一步抑制其得分，衰减因子为函数 $f(IoU(b_m,b_i))$ 的值，关于这两者的具体解释可参考 [CV 中的常用方法总结](/2019/06/24/cv-mtds)。

算法 1 中，对于当前得分最高的检测 box，记为 b， $\{x_1,y_1,x_2,y_2,s,\sigma_{x1},\sigma_{y1},\sigma_{x2},\sigma_{y2}\}$，先使用 soft-NMS 衰减其近邻 boxes 的得分，然后获取其附近（IoU>0） boxes，根据附近 boxes $\sigma$ 的加权来计算当前 box 的新位置，这里加权是基于这样一个认识：某个附近 box 如果越靠近当前 box，那么用它的值来计算当前 box 就越有把握，不确定性越低。用 x 表示坐标（例如 x<sub>1</sub> 坐标），x<sub>i</sub> 表示第 i 个 box 的坐标，坐标新值按如下计算：
$$p_i = e^{-(1-IoU(b_i,b))^2/\sigma_t}
\\\\ x=\frac {\sum_i p_i x_i/\sigma_{x,i}^2} {\sum_i p_i / \sigma_{x,i}^2}
\\\\ \text{s.t.  IoU}(b_i, b) >0$$
上面两式非常明显了，我们不直接使用检测 box 的初始预测位置值，而是通过附近 boxes 的位置和位置方差加权平均值作为当前 box 的位置坐标值。当附近 box 与当前 box 靠的越近，IoU 越大，然后 p<sub>i</sub> 越大，然后 voting 当前 box 的坐标时，权值越大，即贡献越大。另外，上两式也表明附近 box 的方差也影响权值， 当 $\sigma^2$ 越小，权值越大，贡献也越大。以上 voting 过程没有考虑分类得分值，因为低得分的 box 其定位置信度可能还更高，所以让分类得分影响权值，也许会降低准确性。

# 实验
实验介绍及结果分析略，请阅读原文以获得更详细的信息。

# 结论
大型数据集中 gt box 的不确定性会阻碍 SOTA 检测器性能的提升。分类置信度与定位置信度不是强相关的。本文提出新型 bbox 回归损失用于学习目标的准确定位。使用 KL Loss 训练网络学习预测每个坐标的分布方差。预测的方差用在 var voting 中，从而改良 box 的坐标。

从网络结构上来看，在 Faster R-CNN/Mask R-CNN 基础上修改回归预测分支，使用 KL Loss 替换 smooth L1 Loss，并使用 var voting 得到坐标新值，其中坐标初始预测值（也就是算法 1 中的输入 $\mathcal B$）与 Faster R-CNN 中相同。