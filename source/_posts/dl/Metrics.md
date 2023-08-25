---
p: dl/metrics
title: Metrics
date: 2021-02-20 10:09:36
tags: deep learning
---

总结机器学习/深度学习中常用的一些指标
<!-- more -->

- Precision
$$\frac {\text{true positives}}{\text{true positives + false positives}}$$

- Recall
$$TPR=\frac {\text{true positives}}{\text{true positives + false negatives}}$$
我们既想要高准确率，又想要高召回率，也就是 <b>假阳和假阴</b> 都要少，这需要一个很好的模型。判断 positive 或 negative 需要一个阈值，调节这个阈值时，往往会提高一个指标，同时降低另一个指标。

- F1 score
$$\frac {2 \times \text {precision} \times \text{recall}} {\text{precision + recall}}$$
F1 score 随着 precision 或 recall 的增大而增大，当固定 precision 和 recall 其中一个时，F1 score 是另一个的增函数，已知 precision, recall  $\in [0,1]$，易知 F1 score $\in [0,1]$，F1 score 值越大越好。


- ROC curve
\
受试者工作特征曲线（receiver operating characteristic curve），描绘了不同分类阈值下，真阳率（TPR）与假阳率（FPR）的关系。横坐标为 FPR， 纵坐标为 TPR。假阳率为
$$FPR=\frac {\text{false positives}}{\text{false positives + true negatives}}$$
每个样本都有一个预测（为 positive 的）得分 score，当 score 大于阈值时判为 positive，否则为 negative。\
ROC 曲线图上的四个重要的点位：
  1. `(0,1)`，左上角，FPR=0，TPR=1，<b>假阳和假阴均为0</b>，表示是一个完美的分类器
  2. `(1,0)`，右下角，FPR=1, TPR=0，<b>真阳和真阴均为0</b>，避开所有正确答案，是最差的分类器
  3. `(0,0)`，左下角，FPR=TRP=0，<b>真阳和假阳均为0</b>，该分类器将所有样本全部判断为 negative
  4. `(1,1)`，右上角，FPR=TRP=1，<b>真阴和假阴均为0</b>，该分类器将所有样本全部判断为 positive

  当阈值分别为 1 和 0 时，分别对应上面 `3.` 和 `4.` 两小点，故 ROC 曲线一定经过 `(0,0)` 和 `(1,1)` 这两点。显然，越靠近左上角的分类器性能越好。一个好的模型，它的 ROC 曲线从 0 上升到 1。

  如何从 ROC 曲线上找到这个最优点（对应的阈值）呢？借助 ISO 精度线，表示为 $y=ax+b$，其斜率为数据集中负样本数目和正样本数目之比，
  $$a=\frac N P$$
  ISO 精度线寻找最优点步骤：
    1. 初始化截距 `b=0`，逐渐增大 b 的值，直到直线与 ROC 只有一个交点，这个交点就是最优点。


- AUC
\
ROC 曲线下方区域面积（Area under the ROC curve），计算这个面积得到一个标量值指标，便于量化表示模型好坏。 AUC=1 表示分类器最好，AUC=0.5 表示最差（随机猜测），AUC < 0.5 表示分类器连随机猜都不如，这种分类器的预测，我们在反过来预测，就能优于随机猜测。
\
AUC 计算方法：
  1. 选择一组阈值，计算一组 (FPR, TPR)
  2. 按 FPR 从小到大排序，计算第 `i` 和 第 `i+1` 个 FPR 之间的差，记为 `dx`
  3. 获取 第 `i` 或第 `i+1` 个 TPR 的值，记为 `y`
  4. 计算柱形面积 `ds=y * dx`
  5. 所有柱形面积相加求和，得到 AUC 值

### 指标名称：
1. 召回率 Recall / 命中率 hit rate / 灵敏度 Sensitivity /TPR
2. FPR / fall-out
2. 特异度 Specificity / TNR  = 1-FPR