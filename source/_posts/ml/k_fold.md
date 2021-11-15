---
title: k fold 交叉验证
date: 2021-09-19 16:50:18
tags: machine learning
p: ml/k_fold
mathjax: true
---

交叉验证通常用于训练集很小的情况，这时候如果再从中取一部分数据作为验证集，导致训练集进一步减小，更加难以反映真实数据分布，从而使得经验误差与真实误差的差距更大。这时，K 折交叉验证则是一个不错的方法。

<!--more-->
# 交叉验证
操作步骤：

1. 将训练集（大小为 m）分为 k 份，每一个样本数量为 $m/k$。
2. 循环 k 次，在第 i 次循环时，除第 i 份之外其他所有样本作为本次训练集进行训练，训练完成后计算模型在第 i 份样本上的验证误差。
3. k 个验证误差的平均，则为对真实误差的估计值。

# 模型选择
有时候模型有一些超参数，例如多项式中的阶数 $d$，步骤如下：

---
<center>用于模型选择的 k fold 交叉验证</center>

**input:**
1. 训练集 $S=(\mathbf x_1, y_1), \cdots, (\mathbf x_m, y_m)$
2. 超参数集合 $\Theta$
3. 模型学习算法 $A$
4. k-fold 中的 $k$

**切分**
1. 将 $S$ 切分为 $S_1, \cdots, S_k$

**foreach** $\theta \in \Theta$

&emsp; **for** $i=1,\cdots,k$

&emsp; &emsp; $h_{i,\theta}=A(S - S_i; \theta)$

 &emsp; error$(\theta)=\frac 1 k \sum_{i=1}^k L_{S_i}(h_{i, \theta})$

 **output**
 1. $\theta^*=\argmin_{\theta} \[\text{error}(\theta)]$
 2. $h_{\theta^*}=A(S;\theta^*)$

---

总结：
1. 遍历超参数集合，对于每个超参数，分别计算 k-fold 验证误差的平均
2. 选择最小平均误差所对应的超参数，然后利用这个超参数和所有数据集作为训练集，进行训练