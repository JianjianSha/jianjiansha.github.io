---
title: 机器学习中的核方法
date: 2021-09-26 18:07:50
tags: machine learning
p: ml/kernel
mathjax: true
---

# 前言
为了更好的使用线性分类，有时候需要将线性不可分的数据映射到一个更高维空间，使其线性可分。虽然高维空间的 Halfspace 假设更加具有表达能力（假设空间更加丰富），但是同时也会引入样本复杂度以及计算复杂度的上升。通过引入 margin （训练集中某样本与超平面具有最小距离），可以解决样本复杂度这个问题，而计算复杂度则可以通过另一种方法解决，即本篇所介绍的 “核方法”。
<!--more-->

# 特征空间

以一个简单的例子开始，考虑域中样本点 $\{-10,-9,-8,\cdots, 0, 1, \cdots 9, 10\}$，其中满足 $|x|>2$ 的样本分类为 $+1$，其余样本分类为 $-1$，显然这是一个非线性可分的问题，但是我们可以通过以下映射 $\psi: \mathbb R \rightarrow \mathbb R^2$

$$\psi(x)=(x, x^2)$$

使得数据在 $\mathbb R^2$ 空间中线性可分，令 $\mathbf w=(0, 1), \ b=5$，那么 $h(x)=\text{sign}(\mathbf w^{\top} \psi(x) - b)$ 线性可分。

**这一过程总结如下：**

1. 给定域集合 $\mathcal X$ 和一特定的学习任务，根据先验知识，选择一个映射函数 $\psi: \mathcal X \rightarrow \mathcal F$。

2. 给定样本集 $S=\{(\mathbf x_1, y_1), \cdots, (\mathbf x_m, y_m)\}$，计算得到映射后的集合 $\hat S=\{(\psi(\mathbf x_1), y_1), \cdots ,(\psi(\mathbf x_m), y_m)\}$

3. 在 $\hat S$ 上训练一个线性分类器

4. 预测一个新的样本 $\mathbf x$ 为 $h(\psi(\mathbf x))$

**多项式映射是一个常用的较好的 $\psi$**


考虑一维情况，为 $p=\sum_{i=0}^k w_i x^i$，那么 $\psi(x)=(x^0, x^1, \cdots, x^k)$。考虑一般情况下的多变量 $\mathbf x$，多项式 $p(\mathbf x): \mathbb R^n \rightarrow \mathbb R$ 可写为

$$p(\mathbf x)=\sum_{J \in [n]^r: \ r\le k} w_J \prod_{i=1}^r x_{J_i}$$


未知向量 $\mathbf x$ 是 $n$ 维，$J$ 是具有 $r$ 个元素的列表，每个元素表示 $\mathbf x$ 的下标。当固定一个 $r$ 时，然后每次从 $\mathbf x=[x_1, \cdots, x_n]$ 中可重复（或称 可放回）地选择 $r$ 个变量，这 $r$ 个变量相乘为 $\prod_{i=1}^r x_{J_i}$，最后每个乘积项加权求和。具体过程如下：

1. 阶数为 0， $r=0$，表示不选择任何变量，此时乘积项 为 1，共 $1$ 个乘积项。

2. 阶数为 1，$r=1$，从 $[x_1, x_2, \cdots, x_n]$ 中选择一个变量即 $x_{J_1}$，共 $n$ 个乘积项。

3. 阶数为 2，$r=2$，从 $[x_1, x_2, \cdots, x_n]$ 中可放回地选择两个变量即 $x_{J_1}, \ x_{J_2}$，乘积项 $x_{J_1} x_{J_2}$ 共有 $n^2$ 个。

4. 阶数为 k，$r=k$，从 $[x_1, x_2, \cdots, x_n]$ 中可放回地选择 k 个变量即 $x_{J_1}, \cdots, x_{J_k}$，乘积项 $\prod_{i=1}^k x_{J_i}$ 共 $n^k$ 个。

各阶数的乘积项数量形成一个等比数量，根据求和公式，所有乘积项数量为

$$S_{k+1}=\frac {a_1(1-n^{k+1})}{1-n}=\frac {n^{k+1}-1} {n-1}$$

特别的，当公比 $n=1$ 时，求和为 $S_{k+1}=a_1 \times (k+1)=k+1$。

于是，多变量的多项式映射 $\psi(\mathbf x): \mathbb R^n \rightarrow \mathbb R^d$，当考虑多项式为 k 阶时，特征空间的维数 $d=S_{k+1}$。$\psi(\mathbf x)$ 中每一项就是上们的乘积项 $\prod_{i=1}^r x_{J_i}$。

从上面分析来看，特征空间的维度 $d=S_{k+1}$ 很大，导致计算复杂度增加，所以不使用常规的线性分类学习方法（如 SVM），下文介绍的 核方法 就是为了解决计算复杂度增加这个问题。

# 核方法

给定映射函数 $\psi$ 将域空间 $\mathcal X$ 映射到某个 Hilbert 空间，定义核函数为映射后向量的内积

$$K(\mathbf x, \mathbf x')=\psi^{\top}(\mathbf x) \psi(\mathbf x')$$

可以认为核函数 $K$ 用于指定两个映射向量之间的相似度（未归一化）。

我们将线性分类问题转化为如下的更一般的优化问题

$$\min_{\mathbf w} f(\mathbf w^{\top} \psi(\mathbf x_1), \cdots, \mathbf w^{\top} \psi(\mathbf x_m)) + R(\|\mathbf w\|) \quad(1)$$

其中 $f:\mathbb R^m \rightarrow \mathbb R$ 是任意函数，$R: \mathbb R_+ \rightarrow \mathbb R$ 是单调递增函数，作为正则惩罚项。

例如 Soft-SVM 中（非齐次型可转换为齐次型），$R(a)=\lambda a^2$，$f(a_1, \cdots, a_m)=\frac 1 m \sum_{i} \max \{0, 1- y_ia_i\}, \ a_i=\mathbf w^{\top}\mathbf x_i$。

Hard-SVM 中，$R(a)=a^2$，且如果 $\forall i \in [m]$ 均有 $y_i(a_i+b)\ge 1$，那么 $f(a_1, \cdots, a_m)=0$，否则 $f(a_1, \cdots, a_m)=\infty$，这是因为 Hard-SVM 保证是线性可分的，所以只要有一个样本分类错误，那么总的分类错误损失就是 $\infty$。

**存在一个向量 $\boldsymbol \alpha \in \mathbb R^m$，使得 $\mathbf w=\boldsymbol \alpha^{\top} \boldsymbol \psi$ 是 (1) 式的最优解。**

其中 $\boldsymbol \psi = [\psi(\mathbf x_1), \cdots, \psi(\mathbf x_m)]$。事实上，在上一篇文章 [SVM](/2021/09/22/ml/svm) 中，我们已经讨论到，$\mathbf w$ 是训练集中若干支持向量的线性组合，自然也是训练集中所有向量的线性组合。

证：

令 $\mathbf w^{\ast}$ 是 （1）式的最优解。由于 $\mathbf w^{\ast}$ 位于 Hilbert 空间，必然可以分解为两个向量，其中一个来自 $[\psi(\mathbf x_1), \cdots, \psi(\mathbf x_m)]$ 所张（span）空间的一个向量 $\mathbf w$，另一个向量 $\mathbf u$ 位于与这个所张空间垂直的空间，故可写为

$$\mathbf w^{\ast}=\mathbf w + \mathbf u=\sum_{i=1}^m \alpha_i \psi(\mathbf x_i)+\mathbf u$$

其中 $\mathbf u^{\top} \psi(\mathbf x_i)=0, \ \forall i \in [m]$。

于是有 $\|\mathbf w^{\ast}\|^2=\|\mathbf w\|^2+\|\mathbf u\|^2+2\mathbf w^{\top} \mathbf u$，由于 $\mathbf w$ 所在空间与 $\mathbf u$ 所在空间垂直，故 $\mathbf w^{\top}\mathbf u=0$，故 $\|\mathbf w^{\ast}\|^2=\|\mathbf w\|^2+\|\mathbf u\|^2 \ge \|\mathbf w\|^2$，由于 $R$ 是单调增函数，所以 $R(\|\mathbf w^{\ast}\|) \ge R(\|\mathbf w\|)$。

另外注意到

$$\mathbf w^{\top} \psi(\mathbf x_i)=(\mathbf w^{\ast}-\mathbf u)^{\top} \psi(\mathbf x_i)=\mathbf w^{\ast}\psi(\mathbf x_i)$$

所以

$$f(\mathbf w^{\top} \psi(\mathbf x_1), \cdots, \mathbf w^{\top} \psi(\mathbf x_m))=f(\mathbf w^{\ast \top} \psi(\mathbf x_1), \cdots, \mathbf w^{\ast \top} \psi(\mathbf x_m))$$

记 (1) 式中的优化目标为 $L(\cdot)$，所以 $L(\mathbf w^{\ast}) \ge L(\mathbf w)$ ，由于 $\mathbf w^{\ast}$ 是最优解，那么 $\mathbf w$ 也是最优解。证毕。

于是 （1）式的最优解具有形式 $\mathbf w =\sum_{i=1}^m \alpha_i \psi(\mathbf x_i)$。由于

$$\mathbf w^{\top}\psi(\mathbf x_i)=\left(\sum_{j=1}^m \alpha_j \psi(\mathbf x_j)\right)^{\top} \psi(\mathbf x_i)=\sum_{j=1}^m \alpha_j \psi^{\top}(\mathbf x_j)\psi(\mathbf x_i) \quad (2)$$


$$\|\mathbf w\|^2=\Vert \sum_{i=1}^m \alpha_i \psi(\mathbf x_i) \Vert^2=\sum_{i=1}^m \sum_{j=1}^m\alpha_i \alpha_j \psi^{\top}(\mathbf x_i)\psi(\mathbf x_j) \quad(3)$$

所以 （1）式可改写为

$$\min_{\boldsymbol \alpha \in \mathbb R^m} f\left(\sum_{j=1}^m \alpha_j K(\mathbf x_j, \mathbf x_1), \cdots, \sum_{j=1}^m \alpha_j K(\mathbf x_j, \mathbf x_m)\right) + R \left(\sqrt{\sum_{i=1}^m \sum_{j=1}^m\alpha_i \alpha_j K(\mathbf x_i,\mathbf x_j)}\right) \quad(4)$$

将变量从 $\mathbf w \in \mathbb R^d$ 转换为 $\boldsymbol \alpha \in \mathbb R^m$，当 $m << d$ 时，可以降低计算复杂度。

记 $G_{m \times m}$，其中 $G_{ij}=K(\mathbf x_i,\mathbf x_j)$，表示样本之间的内积矩阵，通常称为 Gram 矩阵。那么，(2,3) 式变为

$$\mathbf w^{\top}\psi(\mathbf x_i)=\sum_{j=1}^m \alpha_j K(\mathbf x_i, \mathbf x_j)=G_{i,:} \cdot \boldsymbol \alpha=(G \boldsymbol \alpha)_i$$

$$\|\mathbf w\|^2=\sum_{i=1}^m \sum_{j=1}^m\alpha_i \alpha_j K(\mathbf x_i, \mathbf x_j)=\boldsymbol \alpha^{\top} G \boldsymbol \alpha$$


以 Soft-SVM 为例，根据 $\min_{\mathbf w} (\lambda \|\mathbf w\|^2+\frac 1 m \sum_{i=1}^m \max\{0, 1-y_i(\mathbf w^{\top} \mathbf x_i)\})$， 写成（4）式的形式为

$$\min_{\boldsymbol \alpha \in \mathbb R^m} \ \frac 1 m \sum_{i=1}^m \max\{0, 1-y_i(G \boldsymbol \alpha)_i\} + \lambda \boldsymbol \alpha^{\top} G \boldsymbol \alpha$$

学习到了 $\boldsymbol \alpha$ 之后，那么对新样本的预测为

$$\mathbf w^{\top}\psi(\mathbf x)=\sum_{j=1}^m \alpha_j K(\mathbf x, \mathbf x_j)$$

注意，$K(\mathbf x, \mathbf x_j)$ 不是 G 矩阵元素。

# 带核 Soft-SVM 的实现

上一篇 [SVM](/2021/09/22/ml/svm) 中介绍了 Soft-SVM 的 SGD 学习过程。现在考虑带核 Soft-SVM 的学习。首先给出优化目标

$$\min_{\mathbf w} \left(\frac {\lambda} 2 \|\mathbf w\|^2+\frac 1 m \sum_{i=1}^m \max \{0, 1-y_i \mathbf w^{\top} \psi(\mathbf x_i)\}\right) \quad(5)$$

仿照  [SVM](/2021/09/22/ml/svm) 中 Soft-SVM 的 SGD 实现中的分析，彼时令 

$$\boldsymbol {\theta}^{(t+1)}=\boldsymbol {\theta}^{(t)}-\mathbf v_t=\boldsymbol {\theta}^{(t)}+y_i \mathbf x_i$$

那么带核后，变为

$$\boldsymbol \theta^{(t+1)}=\boldsymbol \theta^{(t)} +y_i\psi(\mathbf x_i)$$

且有

$$\mathbf w^{(t)}=\frac 1 {\lambda t} \boldsymbol \theta^{(t)}$$

由于 $\mathbf w, \boldsymbol \theta \in \mathbb R^d$，维数 $d$ 较大，所以我们不直接更新 $\boldsymbol \theta$。由于最优解可写成 $\mathbf w = \sum_{i=1}^m \alpha_i \psi(\mathbf x_i)$ 的形式，所以我们考虑直接求解 $\boldsymbol \alpha$，或者说，是以 $\psi(\mathbf x_1), \cdots, \psi(\mathbf x_m)$ 作为基向量时的坐标，对应地，$\mathbf w$ 是对应标准基向量的坐标，维度 $dim(\boldsymbol \alpha)=m > dim(\mathbf w)=d$。我们还需要给出 $\boldsymbol \theta$ 在以$\psi(\mathbf x_1), \cdots, \psi(\mathbf x_m)$ 作为基向量时的坐标，记为 $\boldsymbol \beta$，有

$$\boldsymbol \theta^{(t)}=\sum_{j=1}^m \beta_j^{(t)} \psi(\mathbf x_j)$$


显然有以下关系

$$\boldsymbol \alpha = \frac 1 {\lambda t} \boldsymbol \beta$$

于是，[SVM](/2021/09/22/ml/svm) 中 Soft-SVM 的 SGD 实现中，迭代更新（基于标准基向量的） $\boldsymbol \theta$，等价的，这里带核情况的实现中，则是迭代更新（基于 $\psi(\mathbf x_1), \cdots, \psi(\mathbf x_m)$ 的）$\boldsymbol \beta$。

根据前面 $\boldsymbol {\theta}$ 的更新公式，

$$\boldsymbol \theta^{(t+1)}=\boldsymbol \theta^{(t)} +y_i\psi(\mathbf x_i)=\sum_{j=1}^m \beta_j^{(t)} \psi(\mathbf x_j)+y_i \psi(\mathbf x_i)=\sum_{j=1}^m \beta_j^{(t+1)} \psi(\mathbf x_j)$$

可知 $\boldsymbol \beta$ 的更新为（在随机选择一个样本下标 $i$ 后）

$$\boldsymbol \beta_{j}^{(t+1)}= \begin{cases} \boldsymbol \beta_{j}^{(t)} & j \neq i \\\\ \boldsymbol \beta_{i}^{(t)}+y_i & j=i \end{cases}$$

最后，随机选择了样本下标 $i$ 后，判断条件变为

$$y_i \mathbf w^{\top} \psi(\mathbf x_i)=y_i \left(\sum_{j=1}^m \alpha_j^{(t)} \psi(\mathbf x_j)\right)^{\top} \psi(\mathbf x_i)=y_i\sum_{j=1}^m \alpha_j^{(t)} K(\mathbf x_j, \mathbf x_i)$$

最后，整个过程与 [SVM](/2021/09/22/ml/svm) 中 Soft-SVM 的 SGD 实现基本类似，现总结如下：

<center>带核 Soft-SVM 的 SGD 实现</center>

**目标：** 求解（5）式

**参数：** 总迭代次数 $T$

**初始化：** $\boldsymbol \beta^{(1)}=\mathbf 0$

**for** $\ t=1,\cdots, T$

&emsp; $\boldsymbol \alpha^{(t)}=\frac 1 {\lambda t} \boldsymbol \beta^{(t)}$

&emsp; 随机选择一个下标 $i \in [m]$

&emsp; 对 $\forall j \in [m]$ 且 $j\neq i$， $\beta_j^{(t+1)}=\beta_j^{(t)}$

&emsp; 如果 $\ y_i\sum_{j=1}^m \alpha_j^{(t)} K(\mathbf x_j, \mathbf x_i) < 1$

&emsp; &emsp; $\beta_i^{(t+1)}=\beta_i^{(t)}+y_i$

&emsp; 否则

&emsp; &emsp; $\beta_i^{(t+1)}=\beta_i^{(t)}$

**输出** $\overline {\boldsymbol \alpha}=\frac 1 T \sum_{t=1}^T \boldsymbol \alpha^{(t)}$

最后从 $\boldsymbol \alpha$ 变回 $\mathbf w$，

$$\overline {\mathbf w}=\sum_{i=1}^m \overline {\boldsymbol \alpha}\psi(\mathbf x_i)$$

将 $\overline {\boldsymbol \alpha}$ 的表达式代入上式，验证如下

$$\overline {\mathbf w}=\sum_{i=1}^m \left(\frac 1 T \sum_{t=1}^T \boldsymbol \alpha^{(t)}\right)\psi(\mathbf x_i)=\frac 1 T \sum_{t=1}^T\left(\sum_{i=1}^m \boldsymbol \alpha^{(t)}\psi(\mathbf x_i)\right)=\frac 1 T \sum_{t=1}^T \mathbf w^{(t)}$$

可见于普通的 Soft-SVM 完全一致。