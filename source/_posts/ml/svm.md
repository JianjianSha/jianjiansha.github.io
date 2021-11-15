---
title: 支持向量机
date: 2021-09-22 16:04:37
tags: machine learning
p: ml/svm
mathjax: true
---

支持向量机搜索具有最大“边距”的分类器，最大“边距”使得分类器更加健壮。

<!--more-->

# 边距和 Hard-SVM

训练集记为 $S=(\mathbf x_1,y_1), \cdots, (\mathbf x_m, y_m)$，其中 $\mathbf x \in \mathbb R^d, \ y \in \{1,-1\}$。假设这个训练集线性可分，那么有

$$\forall i \in [m], \quad y_i (\mathbf w^{\top} \mathbf x_i+b )>0$$

满足上式的所有 halfspace $(\mathbf w , b)$，有多解， 均属于 ERM 假设（$L_S(h)=0$），我们根据“边距”最大这一条件来选择最好的 $(\mathbf w, b)$。

> 判别超平面与数据集之间的“边距”定义为：超平面与数据集中某样本点之间的最小距离。

如果这个“最小距离”较大，那么即使对样本点有进行小的扰动，也不影响判别效果。

## 距离

**样本点 $\mathbf x$ 与超平面 $(\mathbf w, b)$ （其中 $\|\mathbf w\|=1$）之间的距离为 $|\mathbf w^{\top}\mathbf x+b|$。**

这里给了 $\Vert \mathbf w \Vert=1$ 这个约束，如果没有这个约束，那么同一个超平面会有无穷多个表达，即 $a\mathbf w \mathbf x+ab=1, \ \forall a \neq 0$。给定 $\mathbf w$ 约束条件后，$b$ 可以唯一确定，故不需要对其作约束。

**证：**

这个距离定义为

$$\min\{\Vert\mathbf {x-v}\Vert: \mathbf w^{\top}\mathbf v+b=0\}$$

取 $\mathbf v=\mathbf x-(\mathbf w^{\top} \mathbf x+b)\mathbf w$，验证这个值在超平面上，因为

$$\mathbf w^{\top} \mathbf v+b=\mathbf w^{\top}\mathbf x-(\mathbf w^{\top} \mathbf x+b)\|\mathbf w\|^2+b=0$$

验证完毕。

此时 $\mathbf x, \ \mathbf v$ 之间的距离为

$$\|\mathbf x-\mathbf v\|=|\mathbf w^{\top} \mathbf x+b|\|\mathbf w\|=|\mathbf w^{\top} \mathbf x+b|$$

另一方面，任取超平面上一点 $\mathbf u$，有

$$\begin{aligned} \Vert \mathbf {x-u}\Vert^2 &=\|\mathbf {x-v+v-u}\|^2
\\ &=\|\mathbf {x-v}\|^2+\|\mathbf {v-u}\|^2+2(\mathbf {x-v})^{\top}(\mathbf {v-u})
\\ &\ge\|\mathbf {x-v}\|^2+2(\mathbf {x-v})^{\top}(\mathbf {v-u})
\\ &=\|\mathbf {x-v}\|^2+2(\mathbf w^{\top} \mathbf x+b)\mathbf w^{\top}(\mathbf {v-u})
\\ &=\|\mathbf {x-v}\|^2\end{aligned}$$

上式倒数第二个等式用了 $\mathbf v$ 的定义，倒数第一个等式是因为 $\mathbf w^{\top}\mathbf v=\mathbf w^{\top}\mathbf u=-b$。

从上式可见，$\mathbf x$ 与超平面的距离应该为 $\|\mathbf {x-v}\|^2$，也就是 $|\mathbf w^{\top} \mathbf x+b|$。

## Hard-SVM

Hard-SVM 的学习准则是令超平面与数据集的“边距”最大，即，求解以下最优问题的解，

$$\arg \max_{(\mathbf w, b): \|\mathbf w\|=1} \ \min_{i \in [m]} |\mathbf w^{\top} \mathbf x_i+b| \quad s.t. \ \forall i \in [m], \ y_i(\mathbf w^{\top} \mathbf x_i+b)>0$$

在训练集线性可分这一前提条件下，上式等价于

$$\arg \max_{(\mathbf w, b): \|\mathbf w\|=1} \ \min_{i \in [m]} y_i(\mathbf w^{\top} \mathbf x_i+b) \tag{1} \label{1}$$

### 变换

对满足 $\|\mathbf w\|=1$ 的任意 $(\mathbf w, b)$, 记 $\gamma=\min_{i \in [m]} \ y_i(\mathbf w^{\top}\mathbf x_i+b)$，于是 $\forall i \in [m]$ 有

$$y_i (\mathbf w^{\top}\mathbf x_i+b) \ge \gamma$$

根据数据集线性可分这一假设，有 $\gamma>0$，于是变换上式为

$$y_i (\frac {\mathbf w^{\top}} {\gamma}\mathbf x_i+\frac {b}{\gamma}) \ge 1$$



而上面 $\eqref{1}$ 式要求 $\gamma$ 最大，
而 $\|\mathbf w\|=1$，
那么意味着 $\frac {\|\mathbf w^{\top}\|}{\gamma}$ 最小（$b$ 由 $\mathbf w$ 唯一确定，不对其作约束），也就是说，(1) 式可以变换为如下问题：

$$\arg \min_{(\mathbf w, b)}\|\mathbf w\|^2 \quad s.t. \quad \forall i \in [m], \ y_i(\mathbf w^{\top} \mathbf x_i+b) \ge 1 \tag{2}$$


# Soft-SVM

Hard-SVM 假设训练集线性可分，但是这是一强假设，对这个假设适当放宽就得到 Soft-SVM，即训练集可能线性不可分。我们在 (2) 式的基础上引入松弛变量 $\{\xi_i:\xi_i \ge 0, \forall i \in [m]\}$，使得约束条件变为 $y_i(\mathbf w^{\top} \mathbf x_i+b) \ge 1-\xi_i$，我们的目的除了使 $\Vert \mathbf w\Vert$ 最小之外，还需要使松弛变量尽量小，即尽量减小这种放宽量，或者说尽量满足 Hard-SVM 中的约束条件，这就是 Soft-SVM 优化问题：

---

<center>Soft-SVM 求解思路</center>

**input:** $(\mathbf x_1, y_1), \cdots (\mathbf x_m, y_m)$

**parameter:**  $\lambda >0$ （平衡因子）

**solve:**

$$\min_{\mathbf w, b, \boldsymbol \xi} \left(\lambda \|\mathbf w\|^2 + \frac 1 m \sum_{i=1}^m \xi_i\right)$$
$$s.t. \ \forall i, \ y_i(\mathbf w^{\top} \mathbf x_i+b) \ge 1- \xi_i, \ \xi_i \ge 0$$

**output:** $\mathbf w, b$

---

我们可以使用正则化的损失最小化来改写上式。使用 hinge 损失，

$$l(\mathbf w, b, \mathbf x, y)=\max \{0, 1-y(\mathbf w^{\top}\mathbf x+b)\}$$

分类器在训练集 $S$ 上的损失记为 $L_S(\mathbf w, b)$，那么正则化损失最小问题为

$$\min_{\mathbf w, b} (\lambda \|\mathbf w\|^2+L_S(\mathbf w, b))$$

固定 $(\mathbf w, b)$ 的值，对于某个样本 $(\mathbf x_i, y_i)$，由于 $\xi_i \ge 0$，所以如果 $y_i(\mathbf w^{\top} \mathbf x_i+b) \ge 1$，那么 $\xi_i=0$，如果 $y_i(\mathbf w^{\top} \mathbf x_i+b) < 1$，那么损失为 $1-y_i(\mathbf w^{\top} \mathbf x_i+b) =\xi_i$，所以 $L_S(\mathbf w, b)=\frac 1 m \sum_{i=1}^m \xi_i$。

另外，如果 $0<\xi<1$，表示样本虽然分类正确，但是太过靠近判别超平面；如果 $\xi_i \ge 1$，这表示第 $i$ 个样本分类错误。故，**Soft-SVM 允许一定程度的错误，这是我们所期望的，因为有时候训练集的数据由于噪声干扰，出现了一些错误数据，如果完全按照训练集误差为零进行训练，得到的分类器在真实数据上性能反而会下降。**


## 支持向量
{% raw %}
SVM 中的“支持向量”这一词语来自 Hard-SVM 中的 $\mathbf w_0=\frac {\mathbf w^{\ast}} {\gamma^{\ast}}$，其中 $\mathbf w^{\ast}$ 是 (1) 式的解，$\gamma^{\ast}$ 则是 (1) 式的值，即 $\forall i \in [m], \ y_i(\mathbf w^{{\ast}\top} \mathbf x_i+b)\ge\gamma^{\ast}$，那么，向量 $\mathbf w_0$ 由训练集中的样本子集$\{(\mathbf x_i, y_i): y_i(\mathbf w^{{\ast}\top} \mathbf x_i+b)=\gamma^{\ast}\}$ 支持（支撑），这个样本子集中的样本到判别超平面距离最小，均为 $\gamma^{\ast}$，称这个样本子集中的数据向量为“支持向量”。
{% endraw %}

考虑齐次型，即偏置 $b=0$（事实上，可以将 $b$ 作为 $w_1$，且 $\mathbf x$ 前面增加一个元素 $\mathbf x_1=1$，使得非齐次型转换为齐次型）。那么对于 Hard-SVM 有

$$\min_{\mathbf w} \ \|\mathbf w\|^2 \quad s.t. \quad \forall i \in [m], \ y_i \mathbf w^{\top} \mathbf x \ge 1 \tag{3}$$ 

上面所说的 $\mathbf w_0$ 则是上式的解，支持向量则为 $I=\{\mathbf x_i: |\mathbf w_0^{\top}\mathbf x_i|=1\}$。**存在系数 $\alpha_1, \cdots$ 使得**

$$\mathbf w_0=\sum_{i \in I} \alpha_i \mathbf x_i$$

可以使用拉格朗日乘子法证明上述结论，略。

## 对偶

考虑以下函数

$$g(\mathbf w)=\max_{\boldsymbol \alpha \in \mathbb R^m:\boldsymbol \alpha \ge \mathbf 0} \sum_{i=1}^m \alpha_i (1-y_i \mathbf w^{\top}\mathbf x_i)=\begin{cases} 0 & \forall i, \ y_i(\mathbf w^{\top}\mathbf x_i) \ge 1 \\ \infty & \text{otherwise} \end{cases}$$

上式中 $\alpha_i$ 全部非负，显然在 $y_i(\mathbf w^{\top}\mathbf x_i) \ge 1$ 条件下，$\forall i , \ \alpha=0$ 可使得 $g(\mathbf w)$ 最大，为 $0$，否则，$\forall i, \ \alpha=\infty$ 可使得 $g(\mathbf w)$ 最大，为 $\infty$。

对于线性可分训练集，考虑齐次型，即 $\eqref{3}$ 式，问题可等价为

$$\min_{\mathbf w}\ (\|\mathbf w\|^2+ g(\mathbf w))$$

综合起来就是

$$\min_{\mathbf w} \max_{\boldsymbol \alpha \in \mathbb R^m:\boldsymbol \alpha \ge \mathbf 0} \left(\frac 1 2 \|\mathbf w\|^2+\sum_{i=1}^m \alpha_i (1-y_i \mathbf w^{\top}\mathbf x_i)\right)$$

增加 $\frac 1 2$ 因子是为了后面计算方便。现在将最小最大位置对调，那么目标值只可能变小（弱对偶），

$$\min_{\mathbf w} \max_{\boldsymbol \alpha \in \mathbb R^m:\boldsymbol \alpha \ge \mathbf 0} \left(\frac 1 2 \|\mathbf w\|^2+\sum_{i=1}^m \alpha_i (1-y_i \mathbf w^{\top}\mathbf x_i)\right)
\\ \ge
 \max_{\boldsymbol \alpha \in \mathbb R^m:\boldsymbol \alpha \ge \mathbf 0} \min_{\mathbf w} \left(\frac 1 2 \|\mathbf w\|^2+\sum_{i=1}^m \alpha_i (1-y_i \mathbf w^{\top}\mathbf x_i)\right)$$

实际上在这里，强对偶也成立，即上式中等式成立，于是问题转化为对偶问题

$$ \max_{\boldsymbol \alpha \in \mathbb R^m:\boldsymbol \alpha \ge \mathbf 0} \min_{\mathbf w} \left(\frac 1 2 \|\mathbf w\|^2+\sum_{i=1}^m \alpha_i (1-y_i \mathbf w^{\top}\mathbf x_i)\right)$$

当固定 $\boldsymbol \alpha$ 时，优化问题转换为无约束条件且目标可微，根据梯度为 0 求解，得

$$\mathbf w-\sum_{i=1}^m \alpha_i y_i \mathbf x_i=\mathbf 0 \Rightarrow \mathbf w=\sum_{i=1}^m \alpha_i y_i \mathbf x_i$$

这表示解 $\mathbf w$ 处于样本向量所张空间中。于是对偶问题变成

$$\max_{\boldsymbol \alpha \in \mathbb R^m:\boldsymbol \alpha \ge \mathbf 0} \left(\frac 1 2 \|\sum_{i=1}^m \alpha_i y_i \mathbf x_i\|^2+\sum_{i=1}^m \alpha_i \left(1-y_i \sum_{j=1}^m \alpha_j y_j \mathbf x_j^{\top}\mathbf x_i \right)\right)$$

简化上式，其中第一项，

$$\frac 1 2 \|\sum_{i=1}^m \alpha_i y_i \mathbf x_i\|^2=\frac 1 2 \left(\sum_i \alpha_i y_i \mathbf x_i^{\top}\right)\left(\sum_j \alpha_j y_j \mathbf x_j \right)$$

第二项为

$$\sum_{i=1}^m \alpha_i \left(1-y_i \sum_{j=1}^m \alpha_j y_j \mathbf x_j^{\top}\mathbf x_i \right)=\sum_i \alpha_i-\left(\sum_{j=1}^m \alpha_j y_j \mathbf x_j^{\top}\right)\left(\sum_{i=1}^m \alpha_i y_i \mathbf x_i\right)$$

于是对偶问题简化为

$$\max_{\boldsymbol \alpha \in \mathbb R^m:\boldsymbol \alpha \ge \mathbf 0} \left(\sum_{i=1}^m \alpha_i-\frac 1 2 \sum_{1=1}^m \sum_{j=1}^m \alpha_i  \alpha_j y_iy_j \mathbf x_i^{\top}\mathbf x_j \right)$$

在 [核方法](2021/09/26/ml/kernel) 这篇文章中，也有类似的思想，两个地方联系起来看看，会更有心得。

## SGD 求解 Soft-SVM

使用 hinge 损失，那么 Soft-SVM 可写为

$$\min_{\mathbf w} \left(\frac {\lambda} 2 \|\mathbf w\|^2 + \frac 1 m \sum_{i=1}^m \max \{0, 1-y_i \mathbf w^{\top} \mathbf x_i\}\right) \tag{4} \label{4}$$

将上式写成 $f(\mathbf w)=\frac {\lambda} 2 \|\mathbf w\|^2+L_S(\mathbf w)$ 的形式，这是带正则项的经验损失，然而我们使用随机梯度下降算法，需要求真实损失的梯度，即 $t$ 时刻更新的梯度向量 $\mathbf v_t \in \partial l_{\mathcal D}(\mathbf w^{(t)})$，其中 $\partial l_{\mathcal D}(\mathbf w^{(t)})$ 表示损失在真实样本分布 $\mathcal D$ 下的 $\mathbf w^{(t)}$ 处的次梯度集，$l$ 这里表示 hinge 损失函数，由于真实分布 $\mathcal D$ 未知，我们构造其无偏估计，即 从训练集中 $S$ 均匀随机抽取一个样本 $z$，然后计算 $\partial l(\mathbf w^{(t)}, z)$，于是 $\mathbb E[\lambda \mathbf w^{(t)}+\mathbf v_t]$ 就是 $f=\frac {\lambda} 2 \|\mathbf w\|^2+L_{\mathcal D}(\mathbf w)$ 在 $\mathbf w^{(t)}$ 处的一个次梯度，选择学习率 $\eta=\frac 1 {\lambda t}$，于是更新公式为

$$\begin{aligned}\mathbf w^{(t+1)} &=\mathbf w^{(t)}-\frac 1 {\lambda t}(\lambda \mathbf w^{(t)}+\mathbf v_t)
\\\\ &=\left(1-\frac 1 t\right)\mathbf w^{(t)}-\frac 1 {\lambda t} \mathbf v_t
\\\\ &=\frac {t-1} t \mathbf w^{(t)}-\frac 1 {\lambda t} \mathbf v_t
\\\\ &=\frac {t-1} t \left(\frac {t-2}{t-1}\mathbf w^{(t-1)}-\frac 1 {\lambda (t-1)}\mathbf v_{t-1}\right)-\frac 1 {\lambda t} \mathbf v_t
\\\\ &=\frac {t-2} t \mathbf w^{(t-1)}-\frac 1 {\lambda t} \mathbf v_{t-1}-\frac 1 {\lambda t} \mathbf v_t\end{aligned}$$

根据上述迭代公示，可知

$$\mathbf w^{(t+1)}=-\frac 1 {\lambda t}\sum_{i=1}^t \mathbf v_i$$

$\mathbf v_i$ 是损失（不包括正则损失）即 hinge 损失函数在 $\mathbf w^{(i)}$ 处的次梯度，当 $y\mathbf w^{(i)\top} \mathbf x \ge 1$ 时，次梯度为 $0$，当 $y\mathbf w^{(i)\top} \mathbf x < 1$ 时，次梯度为 $-y\mathbf x$，记 $\boldsymbol {\theta}^{(t)}=-\sum_{i=1}^t \mathbf v_i$，那么 SGD 学习过程具体步骤为

---
<center>SGD 求解 Soft-SVM</center>

**目标：** 求解式 $\eqref{4}$

**参数：** $T$ （总迭代次数）

**初始化：** $\boldsymbol {\theta}^{(1)}=\mathbf 0$

**for** $\ t=1,\cdots, T$

&emsp; $\mathbf w^{(t)}=\frac 1 {\lambda t} \boldsymbol {\theta}^{(t)}$

&emsp; 从 $[m]$ 中均匀随机选择一个值 $i$

&emsp; 若 $\ y_i\mathbf w^{(t)\top} \mathbf x_i < 1$

&emsp; &emsp; $\boldsymbol {\theta}^{(t+1)}=\boldsymbol {\theta}^{(t)}-\mathbf v_t=\boldsymbol {\theta}^{(t)}+y_i \mathbf x_i$

&emsp; 否则

&emsp; &emsp; $\boldsymbol {\theta}^{(t+1)}=\boldsymbol {\theta}^{(t)}$

**输出：** $\overline {\mathbf w}=\frac 1 T \sum_{t=1}^T \mathbf w^{(t)}$

---

当然也可以使用 $\mathbf w^{T}$ 或者 $\overline {\mathbf w}=\frac 1 {k} \sum_{t=T-k+1}^T \mathbf w^{(t)}$ （latest k 个 $\mathbf w{(t)}$ 的平均） 作为最终的输出。