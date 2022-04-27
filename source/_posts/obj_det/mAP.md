---
title: mAP
date: 2019-06-16 11:43:57
tags: object detection
mathjax: true
---
# mAP
目标检测中，不同比赛的评估指标通常也不相同，我们先以 PASCAL VOC 为例进行说明。
-目标检测中常用的评价标准是 mAP（mean Average Precision），入坑目标检测的应该都知道 mAP 是 AP 的平均，即每个分类单独计算出一个 AP 值，然后对所有分类的 AP 值求平均就得到 mAP。
<!-- more -->
## 相关概念
0. Positive 表示检测结果
1. True Positive (TP): IoU 大于等于阈值的检测 box
2. False Positive (FP): IoU 小于阈值的检测 box
3. Precision = TP/(TP+FP) = TP/(所有检测)
4. Recall = TP/(TP+FN) = TP/(所有gt)

由于现在我们专注于目标检测这个场景，所以首先需要弄清楚目标检测中 TP,FP,TN,FN 这四个基本概念。（以下4点均基于个人理解，如有错误，请及时通知本人修改，若博客不支持评论，可在[项目](https://github.io/shajian/shajian.github.io)提 issue）：
1. TP
   
   检测结果为P (Positive)，其中与 gt box 最大 IoU 超过阈值（$Threshold_{VOC}=0.5$）的检测为 TP
2. FP
   
   检测结果为P (Positive)，其中与 gt box 最大 IoU 低于阈值的检测为 FP。如果某个检测与某 gt box 有最大 IoU 且超过阈值，但是这个 gt box 已被另一个检测匹配（match），且另一个检测的 confidence 更高，则当前检测也被认为是 FP。用数学语言描述为：

   $$\left. \begin{array}{} GT_1=\underset{GT_i} {\text{argmax}} \quad \text{IoU}(Det_a, GT_i) \\\\
   GT_1=\underset{GT_i} {\text{argmax}} \quad \text{IoU}(Det_b, GT_i) \\\\
   \text{Conf}_a > \text{Conf}_b \end{array} \right] \Rightarrow Det_b \in FP$$
3. FN
   
   如果某个 gt box 未被检测到，即没有检测结果与这个 gt box 的 IoU 大于0，则认为这个 gt box 为 FN
4. TN
   
   目标检测中没有阴性预测，TN = 0。以二分类问题为例，则分类判断不是 Positive 就是 Negative，TN 表示判断为 Negative，而实际是 Positive。

VOC 使用阈值 `0.5`。
## 指标
### PR 曲线
每个预测 box 均有一个 score 表示 confidence，对这个 confidence 设置阈值，仅考虑大于等于这个阈值的预测 box，小于这个阈值的检测结果则忽略，于是每个不同的 confidence 阈值均对应一对 PR（Precision x Recall）值。实际计算中，按 confidence 降序排列，将预测数量从 1 增加到全部预测数量（从 rank=1 到全部预测数量），每次计算一对 PR 值，于是得到原始的 PR 曲线，对于召回率 R' >= R 选取最大的 P 值则得到插值 PR 曲线。我们使用一个例子予以说明（搬运自[stackexchange](https://datascience.stackexchange.com/questions/25119/how-to-calculate-map-for-detection-task-for-the-pascal-voc-challenge)）。

给定目标分类 "Aeroplane"，假设检测结果如下,
```
BB  | confidence | GT
----------------------
BB1 |  0.9       | 1
----------------------
BB2 |  0.9       | 1
----------------------
BB3 |  0.7       | 0
----------------------
BB4 |  0.7       | 0
----------------------
BB5 |  0.7       | 1
----------------------
BB6 |  0.7       | 0
----------------------
BB7 |  0.7       | 0
----------------------
BB8 |  0.7       | 1
----------------------
BB9 |  0.7       | 1
----------------------
```
（BB 表示检测结果所匹配 "match" 的 GT box）

以上表格中已经按 confidence 降序排列，GT=1 表示 TP，GT=0 表示 FP。 BB 那一列的 `BBi` 表示 GT box 编号。预测 box 总共有 10 个，上表中列出了 9 个，还有一个 预测 box，gt boxes 中与这个预测 box 有最大 IOU 的是 `BB1`，不过其相应的 confidence 比上表中第一行的预测 box （与 `BB1` ) 的 confidence （值为 0.9）小（注意，这里是 confidence 比较，不是 IOU 比较，所以不知道这两个预测 box 哪个与 `BB1` 的 IOU 更大），所以这个预测 box 被第一行的预测 box 抑制，故认为此检测是 FP。除了此外，还有两个未检测到的 BBox 未在上表中列出， FN=2。

TP=5 (对应的匹配 gt box 为 BB1,BB2,BB5,BB8,BB9)，FP=5 ，其中 4 个在上表中列出，还有一个是上面所说的被第一行的 TP 预测 box 所抑制的预测 box，这个被抑制的 box 对应如下的 rank=3 这个 case，舍弃这个检测。这一点在 PASCAL VOC 主页的 Detection Task 的 [Evaluation](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html#SECTION00054000000000000000) 一节也进行了说明。

GT box 数量为 TP+FN=5+2=7。计算所有点的 PR 值如下，
```
rank=1  precision=1.00 and recall=0.14
----------
rank=2  precision=1.00 and recall=0.29
----------
(被第一行预测 box 所抑制的预测box，其 confidence <0.9，例如可取 0.8，它们两个都匹配 BB1)
rank=3  precision=0.66 and recall=0.29
----------
rank=4  precision=0.50 and recall=0.29
----------
rank=5  precision=0.40 and recall=0.29
----------
rank=6  precision=0.50 and recall=0.43
----------
rank=7  precision=0.43 and recall=0.43
----------
rank=8  precision=0.38 and recall=0.43
----------
rank=9  precision=0.44 and recall=0.57
----------
rank=10 precision=0.50 and recall=0.71
----------
```
稍作解释：（使用所匹配的 GT box 编号进行说明）

1. rank=1，检测数量为 1（此时其他检测结果均被舍弃），TP 仅 BB1 一个，没有 FP，故 P=1，R=1/7=0.14
2. rank=2，检测数量为 2，TP 包括 BB1,BB2，没有 FP，故 P=1，R=2/7=0.29
3. rank=3，检测数量为 3，TP 包括 BB1,BB2，FP 为 BB1，故 P=2/3=0.66，R=2/7=0.29
4. ...

### AP
VOC 在 2010 之前，选择固定的 11 个 R 值 等分点，即 R={0,0.1,...,1}，然后对 R' >= R 选择最大 P 值得到插值 PR 曲线。 AP 则是每个 R 阈值处的平均正确率（average precision）。VOC 2010 之后，仍然是对 R' >= R 选择最大 P 值，但是 R 是 [0,1] 之间的所有值（参考上一节内容 __PR 曲线__ 中的计算过程），此时 AP 为 PR 曲线下方的面积 AUC （area under the curve）。两种计算方法如下：

#### 11-点插值
取11个 R 值的 [0,1] 区间等分点计算平均正确率：
$$AP=\frac 1 {11} \sum_{r \in {0,0.1,...,1}} \rho_{interp(r)} \tag(1)$$
$$\rho_{interp(r)}=\max_{\tilde r:\tilde r \ge r} \rho(\tilde r) \tag(2) $$

其中，$\rho(\tilde r)$ 为计算得到的正确率。
举个例子如图（完整例子请参考[这里](https://github.com/rafaelpadilla/Object-Detection-Metrics)），
![](/images/mAP_fig1.png)

（上图中的 PR 数据与上面表格的数据无关，它们来自不同的例子）

蓝色折线的 **上顶点** 为根据预测结果计算得到的 PR 值，即，每一个竖直线段的上端点为 PR 值。红色点则是根据11个固定的 R 值进行插值得到的 PR 值，比如计算阈值 R=0.2 处的插值，根据式 (2)，大于等于 0.2 的 $\tilde r$ 值可取 {0.2,0.2666,0.3333,0.4,0.4666}，当 $\tilde r=0.4$ 时，显然 P 有最大值为 0.4285。根据 11-点插值，计算 AP：

$AP=\frac 1 {11} \sum_{r \in {0,0.1,...,1}} \rho_{interp(r)}$

$AP=\frac 1 {11}(1+0.6666+0.4285+0.4285+0.4285+0+0+0+0+0+0)$

$AP=26.84\%$

#### 所有点插值
AP 计算式为，
$$AP=\sum_{r=0}^1(r_{n+1}-r_n) \rho_{interp}(r_{n+1}) \qquad(3) \\\\
\rho_{interp}(r_{n+1})=\max_{\tilde r: \tilde r \ge r_{n+1}} \rho(\tilde r) \qquad(4)$$
其中，$\rho (\tilde r)$ 为 Recall $\tilde r$ 处的正确率。这种 AP 计算方法首先插值得到每个召回率值的正确率，然后计算插值后 PR 曲线下的面积 AUC。
如下图，
![](/images/mAP_fig2.png)

蓝色折线顶点表示根据检测结果计算出来的 PR 值，红色虚线表示插值后的 RP 值，可将 AUC 划为 4 个区域，如下图，
![](/images/mAP_fig3.png)

于是计算 AP 为，

$AP=A_1+A_2+A_3+A_4=(0.0666-0) \times 1+(0.1333-0.0666) \times 0.6666 \\\\ +(0.4-0.1333) \times 0.4285+(0.4666-0.4) \times 0.3043=24.56\%$

# ROC 曲线
## 相关概念
1. TPR (true positive rate)，又称灵敏度 (sensitivity)、召回率 (recall)：TPR = TP/(TP+FN)
2. TNR (true negative rate)，又称特异度 (specificity): TNR = TN/(FP+TN)
3. FNR (false negative rate)，又称漏诊率: FNR = 1 - TPR = FN/(TP+FN)
4. FPR (false positive rate)，又称误诊率: FPR = 1 - TNR = FP/(FP+TN)
5. LR+ (positive likelihood ratio):
   
   $LR^+=\frac {TPR} {FPR} = \frac {Sensitivily} {1-Specificity}$
6. LR- (negative likelihood ratio):
   
   $LR^-=\frac {FNR} {TNR} = \frac {1-Sensitivity} {Specificity}$
7. Youden index: Youden index = Sensitivity + Specificity - 1 = TPR - FPR

## ROC 曲线

ROC 是常见的评价分类器的指标。

ROC 全称 [receiver operating characteristic](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)（以下很多内容均来自于这个维基百科词条）。

根据不同的判别阈值（大于等于阈值为正，否则为负），得到一组 TPR-FPR 值，所画曲线就是 ROC 曲线。
如下图所示，
![](/images/mAP_fig4.png)

图中 (0,0) 和 (1,1) 两点分别对应：
1. 当阈值为 1 时，全部判断为 Negative，故 TP=FP=0，所以 TPR=FPR=0
2. 当阈值为 0 时，全部判断为 Positive，故 TN=FN=0，所以 TPR=FPR=1

实际上，阈值可以位于范围 $(-\infty,0) \cup (1,+\infty)$，位于 $(-\infty,0)$ 是与第 2 点相同，位于 $(1,+\infty)$ 是与第 1 点相同。

一个好的分类器其 ROC 曲线应该位于直线 y=x 的上方，直线 y=x 对应随机猜测的分类器，也就是说，不管选择什么阈值，都应该让真阳性率大于误诊率。理想情况下，TPR 接近 1，FPR 接近 0，故 ROC 曲线越接近 (0,1)，越偏离直线 y=x，就越好。

## ROC 空间
二分类中，每个实例的分类预测均基于一个连续随机变量 X，即实例对应的得分 score，例如逻辑回归中的概率。给定阈值 T，如果 X>T，为正例，否则为负例。如果实例属于正例，那么 X 的概率密度为 $f_1(x)$，如果实例属于负例，那么 X 的概率密度为 $f_0(x)$。因此,
$$TPR=\int_T^{\infty} f_1(x)dx \\
FPR = \int_T^{\infty} f_0(x)dx$$
两者均为阈值 T 的函数。

1. TPR(T) 表示在该阈值下随机选择一个正例，判断该正例为正例的概率
2. FPR(T) 表示在该阈值下随机选择一个负例，判断该负例为正例的概率。

下图表示某分类器的分类情况，
![图 5](/images/mAP_fig5.png)

横轴为随机变量 X 的取值（表示计算得分 score），与纵轴的交点处为判断阈值，纵轴表示概率密度，越大则表示此 score 对应的实例越多。两个曲线相聚越远，则表示越容易区分正负例。
## AUC
通常使用 ROC 曲线下方的面积 AUC 来评价一个分类器的好坏。

AUC 等于一个概率值：当随机选择一个正例和随机选择一个负例时，分类器计算正例的 Score 大于计算负例的 Score 的概率。根据ROC 曲线，可以将 TPR 看作是 FPR 的函数，而实际上这两者均是判断阈值 T 的函数，所以有
$$TPR(T): T \rightarrow y(x) \\\\
FPR(T): T \rightarrow x$$
于是，
$$
A =\int_0^1 y(x) \ dx  =\int_0^1 TPR[FPR^{-1}(x)] \ dx \\\\ \stackrel{x=FPR(T)} =\int_{-\infty}^{+\infty} TPR(T) \ d[FPR(T)] =\int_{-\infty}^{+\infty} TPR(T) \cdot FPR \ '(T) \ dT \\\\ = \int_{-\infty}^{+\infty} \left( \int_T^{+\infty}  f_1(T') \ dT' \right) f_0(T) \ dT \\\\ =\int_{-\infty}^{+\infty}\int_T^{+\infty}  f_1(T')f_0(T) \ dT' dT \\\\ = P(X_1>X_0)
$$
其中，$X_1$ 表示正例的得分，$X_0$表示负例的得分。

最后一个等号可能不容易理解，我们将 $X_1$ 和 $X_0$ 均看作随机变量，其分布函数为:
$$F_1(x)=\int_{-\infty}^{x} f_1(x) dx \\\\
F_0(x)=\int_{-\infty}^{x} f_1(x) dx$$
概率密度分别为 $f_1,f_0$。

由于$X_1, X_0$ 互相独立，二维随机变量 $(X_1,X_0)$ 的联合概率密度为 $f(x_1,x_0)=f_1(x_1) f_0(x_0)$，于是 $X_0 < X_1$ 的概率为：
$$P(X_1>X_0)=\iint_{G} f(x_1,x_0) dx_1 dx_0=\int_{-\infty}^{+\infty}\int_{x_0}^{+\infty}f_1(x_1) f_0(x_0) \ dx_1 dx_0$$
与上面的计算式形式完全一样，证毕。