---
title: 多分类的学习算法
date: 2021-09-29 17:58:26
tags: machine learning
mathjax: true
p: ml/multiclass_algo
---

多分类问题，对于线性可分的训练集，可以使用 Perception 算法学习，此时损失相当于使用了 $0-1$ 损失，而对于非线性可分的情况，$0-1$ 损失显然由于其不可导，不再适用，需要寻找一种“凸替代”（convex surrogate）损失函数。
<!--more-->

# 凸替代
凸替代需要满足两个条件：

1. 是凸函数
2. 凸替代损失必须是 upper bound 原损失（是原损失的上边界）。

例如，使用 hinge 损失作为 $0-1$ 的凸替代，以二分类问题为例说明，

$$l^{0-1}(\mathbf w, (\mathbf x, y))=\mathbb I_{y\neq \text{sign}(\langle \mathbf w, \mathbf x\rangle)}=\mathbb I_{y \langle \mathbf w, \mathbf x\rangle \le 0}$$

$$l^{hinge}(\mathbf w, (\mathbf x, y))=\max \{0, 1-y \langle \mathbf w, \mathbf x\rangle\}$$

显然有 $\forall \ y \langle \mathbf w, \mathbf x\rangle \in \mathbb R, \ l^{hinge}(\mathbf w, (\mathbf x, y)) \ge l^{0-1}(\mathbf w, (\mathbf x, y))$。

# 泛 Hinge 损失

上一节是二分类情况下的凸替代，现在需要确定多分类问题下的凸替代，即，泛 Hinge 损失。根据 [多分类的基本介绍](/2021/09/22/ml/multiclass)，给出预测函数的一般形式：

$$h_{\mathbf w}(\mathbf x)=\mathop{\arg}\max_{y' \in \mathcal Y} \langle \mathbf w, \Psi(\mathbf x, y')\rangle \tag{1}$$

根据上式，即，$h_{\mathbf w}(\mathbf x)$ 是使 得分 最高的那个分类，在当前参数 $\mathbf w$ 下，得分最高的那个分类不一定是样本真实分类，

$$\langle \mathbf w, \Psi(\mathbf x, y)\rangle \le \langle \mathbf w, \Psi(\mathbf x, h_{\mathbf w}(\mathbf x))\rangle$$

其中 $y$ 为样本真实分类，在当前参数 $\mathbf w$ 下，如果得分最高的那个分类恰好是样本真实分类（或者说，当前参数对这个样本分类预测正确），那么等式成立。

记原损失函数为 $\Delta(h_{\mathbf w}(\mathbf x), y)$，其中 $y$ 表示样本真实分类，$h_{\mathbf w}(\mathbf x)$ 表示样本预测分类。原损失函数常见的可以是 $0-1$ 损失，我们需要构造原损失函数的凸替代，根据上式不等式关系以及上面凸替代的第二个要求，不难想到写出如下关系

$$\Delta(h_{\mathbf w}(\mathbf x), y) \le \Delta(h_{\mathbf w}(\mathbf x), y)+\langle \mathbf w, \Psi(\mathbf x, h_{\mathbf w}(\mathbf x))-\Psi(\mathbf x, y)\rangle$$

由于 $h_{\mathbf w}(\mathbf x) \in \mathcal Y$，可以将上面这个不等式的右端 upper bound 为

$$\max_{y' \in \mathcal Y} \ (\Delta(y', y)+\langle \mathbf w, \Psi(\mathbf x, y')-\Psi(\mathbf x, y)\rangle) \stackrel{def}=  l(\mathbf w, (\mathbf x, y)) \tag{100} \label{100}$$

于是 $l(\mathbf w, (\mathbf x, y)) \ge \Delta(h_{\mathbf w}(\mathbf x), y)+\langle \mathbf w, \Psi(\mathbf x, h_{\mathbf w}(\mathbf x))-\Psi(\mathbf x, y)\rangle\ge \Delta(h_{\mathbf w}(\mathbf x), y)$，第一个非严格不等关系中，当

$$h_{\mathbf w}(\mathbf x)=\mathop{\arg} \max_{y' \in \mathcal Y} \ (\Delta(y', y)+\langle \mathbf w, \Psi(\mathbf x, y')-\Psi(\mathbf x, y)\rangle) \tag{2}$$

时，等号成立。第二个非严格不等关系中，当 

$$h_{\mathbf w}(\mathbf x)=y \tag{3}$$

时, 等号成立，将 (3) 式代入 (2) 式 有

$$\Delta(y, y)+\langle \mathbf w, \Psi(\mathbf x, y)-\Psi(\mathbf x, y)\rangle \ge \Delta(y', y)+\langle \mathbf w, \Psi(\mathbf x, y')-\Psi(\mathbf x, y)\rangle, \ \forall y' \in \mathcal Y \setminus \{y\}$$

由于 $\Delta(y,y)=0$，化简上式得

$$\langle \mathbf w,\Psi(\mathbf x, y)\rangle \ge \langle \mathbf w, \Psi(\mathbf x, y')\rangle+\Delta(y', y) \tag{4}$$

即，满足 (4) 式关系时，下面不等式关系中等号成立。

$$l(\mathbf w, (\mathbf x, y)) \ge \Delta(h_{\mathbf w}(\mathbf x), y) \tag{5}$$

注：(5) 式恒成立，只是在满足 (4) 条件时，(5) 式中等号成立。

$l(\mathbf w, (\mathbf x, y))$ 是若干个有关 $\mathbf w$ 的线性函数的最大值函数，根据本文附录的 **定理 1**， **$l(\mathbf w, (\mathbf x, y))$ 是 $\mathbf w$ 的凸函数。并且 $l(\mathbf w, (\mathbf x, y))$ 是 $\rho-$Lipschitz 函数。**（证明见下方附录）


$\eqref{100}$ 式定义的 $l(\mathbf w, (\mathbf x, y))$ 就是泛 hinge 损失。对于二分类情况，令原损失 $\Delta$ 为 $0-1$ 损失，当 $\mathcal Y = \{\pm 1\}$ 时，设置 $\Psi(\mathbf x, y)=y\mathbf x/2$，那么泛 hinge 损失将退化为普通 hinge 损失，

$$l(\mathbf w, (\mathbf x, y))=\max_{y'\in \{y, -y\}} \Delta(y', y)+\langle \mathbf w, (y'-y)\mathbf x/2 \rangle$$

当 $y'=y$ 时， $\Delta(y', y)=0$，$\langle \mathbf w, (y'-y)\mathbf x/2 \rangle=0$

当 $y'=-y$ 时，$\Delta(y', y)=1$，$\langle \mathbf w, (y'-y)\mathbf x/2 \rangle=-y \langle \mathbf w, \mathbf x\rangle$

于是 $l(\mathbf w, (\mathbf x, y))=\max \{0, 1-y \langle \mathbf w, \mathbf x\rangle\}$，这与前文 [多分类的基本介绍](/2021/09/22/ml/multiclass) 中所讨论的完全一致。

再以 $\mathcal Y=\{1, 2\}$ 的二分类为特例，介绍这个 泛 hinge 损失。前文 [多分类的基本介绍](/2021/09/22/ml/multiclass) 中已经给出了 $\Psi(\mathbf x, y)$ 和 $\mathbf w$ 的具体形式，这里直接引用过来，

$$\mathbf w'=\begin{bmatrix} \mathbf w \\\\ -\mathbf w \end{bmatrix}

\\ \Psi(\mathbf x, y=1)=[x_1,\cdots, x_n, 0,\cdots 0]^{\top}=\begin{bmatrix} \mathbf x \\\\ \mathbf 0 \end{bmatrix}
\\ \Psi(\mathbf x, y=2)=[0,\cdots 0, x_1,\cdots, x_n]^{\top}=\begin{bmatrix} \mathbf 0 \\\\ \mathbf x \end{bmatrix}
\\h(\mathbf x)=arg \max_{y \in \mathcal Y} \mathbf w'^{\top} \Psi(\mathbf x, y)$$

根据 $\eqref{100}$ 式，有

$$l(\mathbf w, (\mathbf x, y))=\max_{y' \in \{1,2\}} \Delta(y', y)+\mathbf w'^{\top}(\Psi(\mathbf x, y')-\Psi(\mathbf x, y))$$

1. 当 $y'=y$

显然有 $\Delta(y', y)+\mathbf w'^{\top}(\Psi(\mathbf x, y')-\Psi(\mathbf x, y))=0$

2. 当 $y' \neq y$，那么 $\Delta(y', y)=1$

如果是 $y'=1, y=2$，那么 $\mathbf w^{\top}(\Psi(\mathbf x, y')-\Psi(\mathbf x, y))=2\mathbf w^{\top}\mathbf x$

如果是 $y'=2, y=1$，那么 $\mathbf w^{\top}(\Psi(\mathbf x, y')-\Psi(\mathbf x, y))=-2\mathbf w^{\top}\mathbf x$

综合以上，有 $l(\mathbf w, (\mathbf x, y))=\max \{0, 1-(-1)^{y-1}2 \mathbf w^{\top}\mathbf x\}$ 。 不难看出，这个损失函数的形式与上面的是一致的。

# 学习算法

解决多分类问题，其核心是将原来二分类中的 $\mathbf w, \ \mathbf x \in \mathbb R^n$ 映射到更高维的空间中 $\mathbf w, \ \Psi(\mathbf x, y) \in \mathbb R^{nk}$ 中去。现在我们使用 SGD 学习算法，并给损失函数增加一个正则项，

$$L=\mathbb E_{(\mathbf x, y)\sim \mathcal D}[l(\mathbf w, (\mathbf x, y))]+ \frac {\lambda} 2 \Vert \mathbf w \Vert^2$$

$t$ 时刻更新的梯度向量 $\mathbf v_t \in \partial l(\mathbf w^{(t)})$，其中 $\partial l(\mathbf w^{(t)})$ 表示真实分布 $\mathcal D$ 下，泛 hinge 损失函数在 $\mathbf w^{(t)}$ 的次梯度集。从训练集中随机抽取一个样本 $(\mathbf x, y)$，计算 $\partial l(\mathbf w^{(t)},(\mathbf x, y))$，选择学习率 $\eta$，那么更新公式为

$$\begin{aligned}
\mathbf w^{(t+1)}&=\mathbf w^{(t)}-\eta(\lambda \mathbf w^{(t)}+\mathbf v_t)
\\ &= \frac {t-2} t \mathbf w{(t-1)}-\eta \mathbf v_{t-1}-\eta \mathbf v_t
\\ &= \cdots
\\&=\frac 0 t \mathbf w^{(1)}-\eta\sum_{i=1}^t \mathbf v_i
\\&=-\eta\sum_{i=1}^t \mathbf v_i
\end{aligned}$$

其中初始化 $\mathbf w^{(t)}=\mathbf 0$ 。 具体推导过程与 [SVM](/2021/09/22/ml/svm) 中完全一样。

由于 $\mathbf v_i$ 是泛 hinge 损失在 $\mathbf w{(i)}$ 处的次梯度，根据 $\eqref{100}$ 式，易知次梯度为 

$$\mathbf v_i=\Psi(\mathbf x, \hat y)-\Psi(\mathbf x, y)$$

其中 $\hat y= \arg_{y'}  l(\mathbf w, (\mathbf x, y))$，显然当 $\hat y=y$ 时，次梯度 $\mathbf v_i=\mathbf 0$。

总结算法步骤如下

---
<center> 多分类问题的 SGD 算法</center>

**参数：**

&emsp; 学习率 $\eta$，迭代次数 $T$。

&emsp; 原损失函数 $\Delta: \mathcal Y \times \mathcal Y \rightarrow \mathbb R_+$

&emsp; 映射到特征空间的函数 $\Psi: \mathcal X \times \mathcal Y \rightarrow \mathbb R^d$

**初始化：** $\ \mathbf w^{(1)}=\mathbf 0 \in \mathbb R^d$

**for** $t=1,2,\cdots, T$

&emsp; 随机取样 $(\mathbf x, y) \in \mathcal D$

&emsp; $\hat y=\arg \max_{y' \in \mathcal Y} (\Delta(y', y)+\langle \mathbf w^{(t)}, \Psi(\mathbf x, y')-\Psi(\mathbf x, y)\rangle)$

&emsp; $\mathbf v_t=\Psi(\mathbf x, \hat y)-\Psi(\mathbf x, y)$

&emsp; $\mathbf w^{(t+1)}=\mathbf w^{(t)}-\eta \mathbf v_t$

**输出：** $\ \overline {\mathbf w}=\frac 1 T \sum_{t=1}^T \mathbf w^{(t)}$

---

# Appendix
**1. 定理1**

对 $i=1,\cdots, r$，令 $f_i: \mathbb R^d \rightarrow \mathbb R$ 为凸函数，那么 $g(x)=\max_{i \in [r]} f_i(x)$ 也是凸函数。

证：

$$\begin{aligned} g(\alpha u +(1-\alpha)v) &= \max_i f_i(\alpha u + (1-\alpha)v)
\\\\ & \le \max_i [\alpha f_i(u)+(1-\alpha)f_i(v)]
\\\\ &=\alpha \max_i f_i(u) + (1-\alpha) \max_i f_i(v)
\\\\ &=\alpha g(u) + (1-\alpha) g(v)
\end{aligned}$$

上面推导中，$\alpha \in (0,1)$。证毕。

**2. $l(\mathbf w, (\mathbf x, y))$ 是 $\rho-$Lipschitz 函数。**

对任意 $\mathbf w_1, \mathbf w_2$，记 $y_i=\arg_{y'} \ l(\mathbf w_i, (\mathbf x, y))$，注意，这里的下标 $i$ 不对应分类下标 $1,2,\cdots , k$，是为了方便，仅对应 $\mathbf w_i$ 的下标。

对 $\mathbf w_1$ 而言，$y_1$ 是使得 $(\Delta(y', y)+\langle \mathbf w, \Psi(\mathbf x, y')-\Psi(\mathbf x, y)\rangle)$ 最大的值，这意味着 $(\Delta(y_1, y)+\langle \mathbf w, \Psi(\mathbf x, y_1)-\Psi(\mathbf x, y)\rangle)>(\Delta(y_2, y)+\langle \mathbf w, \Psi(\mathbf x, y_2)-\Psi(\mathbf x, y)\rangle)$

如果 $l(\mathbf w_1, (\mathbf x, y)) \ge l(\mathbf w_2, (\mathbf x, y))\ge 0$，那么有

$$\begin{aligned}\|l(\mathbf w_1, (\mathbf x, y)) - l(\mathbf w_2, (\mathbf x, y))\|
&=\|(\Delta(y_1, y)+\langle \mathbf w_1, \Psi(\mathbf x, y_1)-\Psi(\mathbf x, y)\rangle)-(\Delta(y_2, y)+\langle \mathbf w_2, \Psi(\mathbf x, y_2)-\Psi(\mathbf x, y)\rangle)\|
\\\\& \le \|(\Delta(y_1, y)+\langle \mathbf w_1, \Psi(\mathbf x, y_1)-\Psi(\mathbf x, y)\rangle)-(\Delta(y_1, y)+\langle \mathbf w_2, \Psi(\mathbf x, y_1)-\Psi(\mathbf x, y)\rangle)\|
\\\\ & \le \|\mathbf w_1-\mathbf w_2\|\cdot \|\Psi(\mathbf x, y_1)-\Psi(\mathbf x, y)\|
\end{aligned}$$

同理，如果 $l(\mathbf w_2, (\mathbf x, y)) \ge l(\mathbf w_1, (\mathbf x, y)) \ge 0$，那么有

$$\begin{aligned}\|l(\mathbf w_1, (\mathbf x, y)) - l(\mathbf w_2, (\mathbf x, y))\|
&=\|(\Delta(y_2, y)+\langle \mathbf w_2, \Psi(\mathbf x, y_2)-\Psi(\mathbf x, y)\rangle)-(\Delta(y_1, y)+\langle \mathbf w_1, \Psi(\mathbf x, y_1)-\Psi(\mathbf x, y)\rangle)\|
\\\\& \le \|(\Delta(y_2, y)+\langle \mathbf w_2, \Psi(\mathbf x, y_2)-\Psi(\mathbf x, y)\rangle)-(\Delta(y_2, y)+\langle \mathbf w_1, \Psi(\mathbf x, y_2)-\Psi(\mathbf x, y)\rangle)\|
\\\\ & \le \|\mathbf w_1-\mathbf w_2\|\cdot \|\Psi(\mathbf x, y_2)-\Psi(\mathbf x, y)\|
\end{aligned}$$

综上，$\|l(\mathbf w_1, (\mathbf x, y)) - l(\mathbf w_2, (\mathbf x, y))\|\le \rho \|\mathbf w_1 - \mathbf w_2\|$，其中 

$$\rho=\max_{y' \in \mathcal Y} \|\Psi(\mathbf x, y')-\Psi(\mathbf x, y)\|$$
