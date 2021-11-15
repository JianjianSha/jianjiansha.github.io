---
title: 多分类
date: 2021-09-28 13:52:35
tags: machine learning
mathjax: true
p: ml/multiclass
---

前面讲过二分类模型，例如 SVM，本文讨论更一般的情况，即多分类问题。

<!--more-->
# Reduction

首先我们考虑一些降解思路即，将多分类问题转化为二分类。要学习的假设函数为 $h: \mathcal X \rightarrow \mathcal Y$。对于多分类，记 $\mathcal Y=\{1,\cdots,k\}$，训练集为 $S=\{(\mathbf x_1, y_1), \cdots, (\mathbf x_m, y_m)\}, \ y_i \in \mathcal Y$。

## One-V.S.-All

其本质是 One-V.S.-Rest，也就是说，训练 k 个二分类器 $h_i: \mathcal X \rightarrow \{1,-1\}, \ \forall i \in [k]$，每个二分类器用于判别某个分类和其他所有分类。首先创建 k 个二分类训练集 $S_1, \cdots, S_k$，其中

$$S_i=\{(\mathbf x_1, (-1)^{\mathbb I(y_1\neq i)}), \cdots, (\mathbf x_m, (-1)^{\mathbb I(y_m\neq i)})\}$$

上式表面，第 $j$ 个数据 $(\mathbf x_j, y_j)$，如果 $y_j=i$，那么这个数据为 $S_i$ 中的正例，否则为负例。二分类器 $h_i$ 的训练集为 $S_i$，所有二分类器训练完成后，构建多分类器如下

$$h(\mathbf x) \in arg \max_{i \in [k]} \ h_i(\mathbf x)$$

理解为什么这样构建这个多分类器：对于训练集中的某样本 $(\mathbf x_j, y_j)$，记其分类 $y_j=c$，那么理想情况下，只有 $h_c(\mathbf x_j)=1$，其他 $h_i(\mathbf x_j)=-1, \ \forall i \neq c$，所以 $h_c(\mathbf x_j)$ 可取得最大值，即，样本所属的那个分类取得最大值 $-1$。

但是，对于一个新的样本点 $\mathbf x$，可能有多个 $h_i(\mathbf x)=1$，即，多个最大值，如下图所示，$k=3$，训练集位于三个圆内，那么如果新样本位于阴影区域，那么 $h_1(\mathbf x)=h_2(\mathbf x)=1$，这个时候可以采用 $h_i(\mathbf x)=\mathbf w_i^{\top} \mathbf x$，这表示（非归一化）置信度，通过置信度来做分类决策。

![](/images/ml/multiclass_fig1.png)

<center>图1： h1:蓝色为正例；h2:橙色为正例；h3:绿色为正例</center>

## All Pairs
思想与上面类似，仍然是训练 k 个二分类器，不同的是训练集，对任意 $1 \le i < j \le k$，创建一个二分类训练集 

$$S_{ij}=\{(\mathbf x_l, (-1)^{I(y_l=j)}): y_l=i \lor y_l=j \}$$

上式说明在训练集 $S_{ij}$ 中，样本 $(\mathbf x_l, y_l)$ 的分类必须是 $i$ 或者 $j$，否则样本不在 $S_{ij}$ 中。这样的训练集一共有 $C_k^2$ 个，这是一个组合数，对应的则有 $C_k^2$ 个判别函数 $h_{ij}: \mathcal X \rightarrow \{\pm 1\}$。这样，每一个样本分类 $c$ 均有 $k-1$ 个分类器 $h_{ij}$，其中满足 $i=c$ 或者 $j=c$。

**构建最终多分类器的准则是：对每个分类 $c$，均执行 $k-1$ 次预测，预测正确的次数之和作为这个分类的得分，那么得分最大的分类就是最终的预测分类。**

注：这里的“预测正确”指的是：对 $\forall i < c, \ h_{ic}(\mathbf x)=-1$，以及对 $\forall i>c, \ h_{ci}(\mathbf x)=1$。

伪代码如下：

---
<center>All-Pairs 学习算法</center>

**input:** 训练集 $S=\{(\mathbf x_1, y_1), \cdots, (\mathbf x_m, y_m)\}$，二分类学习算法 $A$.

**for** $\ i = 1, \cdots, k-1$

&emsp; **for** $\ j=i+1, \cdots, k$

&emsp;&emsp; **for** $\ t=1,\cdots, m$

&emsp;&emsp;&emsp; 如果 $y_t=i$，$S_{ij} += \{(\mathbf x_t, 1)\}$

&emsp;&emsp;&emsp; 如果 $y_t=j$，$S_{ij} += \{(\mathbf x_t, -1)\}$

&emsp;&emsp; $h_{ij}= A(S_{ij})$

**output:**
$$h(\mathbf x) \in arg\max_{i \in [k]}\left(\sum_{j=1}^k \text{sign}(j-i) \cdot h_{ij}(\mathbf x)\right)$$
---

同样的，可能出现得分相同的情况，与上面分析相同，可以使用 $h_{ij}(\mathbf x)=\mathbf w_{ij}^{\top}\mathbf x$。

## reduction 缺点

上面这两种 reduction 方法的缺点是，对于二分类器 $h_i$ 或者 $h_{ij}$，它们自己并不知道是被用来做多分类预测的，这会导致一些问题。以 One-V.S.-All 方法为例说明，如图 2 所示，
![](/images/ml/multiclass_fig2.png)

<center>图2：k=3，样本位于各自圆内</center>

假设分类 $1,2,3$ 的样本数量比例分别为 $40\%, 20\%, 40\%$，那么在训练 $h_2$ 的时候，由于线性不可分，$h_2$ 会将所有样本判为负，于是所有分类为 $2$ 的样本均被预测错误。然而，如果选择 $\mathbf w_1=(-1/\sqrt 2, 1/\sqrt 2), \ \mathbf w_2=(0,1), \ \mathbf w_3=(1/\sqrt 2, 1/\sqrt 2)$，那么 $h(\mathbf x)=arg\max_i h_i(\mathbf x)=arg\max_i \mathbf w_i^{\top}\mathbf x$ 则会全部样本分类正确。

上面示例表示，尽管假设空间 $\mathcal H=\{\mathbf x \rightarrow  arg\max_i \mathbf w_i^{\top}\mathbf x: \mathbf w \in \mathbb R^d\}$ 的 approximation error 为 0，但是 One-V.S.-All 学习方法却无法从中找到最好的分类器。

> approximation error: 假设空间中某判别函数在真实分布上的最小损失。

鉴于 reduction 方法存在不足之处，研究适用于多分类预测的更直接的方法。

# 线性多分类预测器

$$h(\mathbf x)=arg\max_{i \in [k]} W^{\top} \mathbf x$$

其中，参数矩阵 $W \in \mathbb R^{d \times k}$，向量 $\mathbf x \in \mathbb R^n$，$W^{\top} \mathbf x$ 是一个向量，其中每个元素表示对应分类的得分，这个向量中最大值的下标就是预测分类。但是这里 $W$ 是一个矩阵，参数矩阵可写为 $W=[\mathbf w_1, \cdots, \mathbf w_k]$，相当于每个分类有一个独立的参数向量，而线性分类中常用的是参数向量 $\mathbf x$，所以把它改写为向量形式，

$$h(\mathbf x)=arg \max_{y \in \mathcal Y} \mathbf w^{\top} \Psi(\mathbf x, y) \tag{1}\label{1} $$

其中，$\Psi(\mathbf x, y)$ 将样本 $(\mathbf x, y)$ 映射为相应的特征，且有

$$\Psi(\mathbf x, y)=[\underbrace {0,\cdots, 0}_{(y-1)n}, \underbrace {x_1,\cdots, x_n}_{n}, \underbrace { 0,\cdots 0}_{(k-y)n}]^{\top}$$

上式中，$x_1$ 前面有 $(y-1)n$ 个 0，$x_n$ 后面有 $(k-y)n$ 个 0。$\Psi(\mathbf x, y), \mathbf w \in \mathbb R^{nk}$。$\mathbf w$ 可以写成 

$$\mathbf w=\begin{bmatrix} \mathbf w_1 \\\\ \vdots \\\\ \mathbf w_k \end{bmatrix}$$

此时，预测函数变成

$$h(\mathbf x)=arg \max_{y \in \mathcal Y} \mathbf w_y^{\top} \mathbf x \tag{2}\label{2}$$

于是，上面两种 $h(\mathbf x)$ 的形式是等价的，称后一种为一般形式。

二分类中只有一个分类无关的参数向量 $\mathbf w$，事实上，二分类也可以有两个权值向量 $\mathbf w_1, \mathbf w_2$，只不过我们令 $\mathbf w_2=-\mathbf x_1$，从而简化成一个参数向量，如果写成上面的两种形式，则分别为：

1. 矩阵形式
$$h(\mathbf x)=arg \max_{i\in \{1, 2\}} \begin{bmatrix} \mathbf w_1, & -\mathbf w_1 \end{bmatrix}^{\top} \mathbf x$$

2. 向量形式

$$\mathbf w=\begin{bmatrix} \mathbf w_1 \\\\ -\mathbf w_1 \end{bmatrix}$$
$$ \Psi(\mathbf x, y=1)=[x_1,\cdots, x_n, 0,\cdots 0]^{\top}$$
$$\Psi(\mathbf x, y=2)=[0,\cdots 0, x_1,\cdots, x_n]^{\top}$$
$$h(\mathbf x)=arg \max_{y \in \mathcal Y} \mathbf w^{\top} \Psi(\mathbf x, y)$$

前面文章讨论的二分类器函数为 $h(\mathbf x)=\text{sign}(\mathbf w_1^{\top} \mathbf x)$，它们都是一致的，例如当 $\mathbf w_1^{\top} \mathbf x>0$ 时，$\text{sign}(\mathbf w_1^{\top} \mathbf x)=1$，而矩阵形式中 $\begin{bmatrix} \mathbf w_1 , & -\mathbf w_1 \end{bmatrix}^{\top} \mathbf x=[y_1, \ -y_1]^{\top}$，显然 $y_1>-y_1$，故判断结果为分类 $1$。

另一方面，二分类的一般形式中， $\mathbf w$ 的通常可写为

$$\mathbf w=\begin{bmatrix} \mathbf w_1 \\\\ \mathbf w_2 \end{bmatrix}$$
而且不一定需要有 $\mathbf w_2=-\mathbf w_1$ 成立，下一篇文章讨论多分类的学习算法之后，就能理解了。

## 一个例子
这个例子如下图所示，$\mathcal X = \mathbb R^2, \ \mathcal Y =\{1,2,3,4\}, \ k=4$，
![](/images/ml/multiclass_fig3.png)

样本分布在一个圆内，每个类别用不同的颜色标注，每个类别的样本均匀分布在某个扇形区域内，那么有

$$\mathbf w=\begin{bmatrix} \mathbf w_1 \\\\ \vdots \\\\ \mathbf w_4 \end{bmatrix}$$

且有 $\forall i \in [4], \ \mathbf w_i \in \mathbb R^2$。

根据前面的分析，并利用式 $\eqref{2}$，对于任意某分类 $i \in [k]$，需要这个分类中的所有样本 $\mathbf x$ 与对应的参数分量 $\mathbf w_i$ 的内积最大，等价于求

$$\max_{\mathbf w_i} \sum_{j \in [m]:\ y_j = i} \mathbf w_i^{\top} \mathbf x_j \tag{3} \label{3}$$

我们知道，$(\alpha \mathbf w^{\top})\mathbf x=\alpha (\mathbf w^{\top} \mathbf x)$，当 $\alpha$ 任意大时，表达式的值是没有上边界的，所以不妨增加限制 $\|\mathbf w\|=1$，否则求最大值没有意义。

我们先固定 $\|\mathbf x\|$ 为某个值，然后求得 $\eqref{3}$ 式的最优解，记为 $\mathbf w_i^{\star}$，那么当改变 $\|\mathbf x\|$ 为另一个固定值值时，由于样本均匀分布在扇形内，$\eqref{3}$ 式的最优解仍是 $\mathbf w_i^{\star}$，故我们只需要考虑  $\|\mathbf x\|$ 为某个固定值的情况，不妨令  $\|\mathbf x\|=1$。

记分类 $i$ 的样本 $\mathbf x$ 与 x 轴正向的夹角为 $\theta$，扇形的起始边与终止边与 x 轴正向的夹角分别为 $\theta_1, \ \theta_2$，参数向量 $\mathbf w_i$ 与 x 轴正向夹角为 $\theta_0$，则有

$$\mathbf w_i=(\cos \theta_0, \sin \theta_0), \quad \mathbf x=(\cos \theta, \sin \theta)$$

目标函数转化为如下优化问题

$$\begin{aligned}\max_{\theta_0}& \int_{\theta_1}^{\theta_2} \cos \theta_0 \cos \theta+\sin \theta_0 \sin \theta d\theta
\\\\ &=\cos \theta_0 \sin \theta - \sin \theta_0 \cos \theta |_{\theta_1}^{\theta_2}
\\\\&=-(\cos \theta_2-\cos \theta_1) \sin \theta_0+(\sin \theta_2-\sin \theta_1)\cos \theta_0
\\\\ &=\sqrt{a^2+b^2} \sin(\theta_0+\phi)
\end{aligned}$$

上式最后一个等式使用了三角函数的辅助角公式，其中 
$$a=-(\cos \theta_2-\cos \theta_1), \quad b = \sin \theta_2-\sin \theta_1$$

$$\sin \phi=\frac b {\sqrt{a^2+b^2}}, \quad \cos \phi = \frac a {\sqrt{a^2+b^2}}$$

根据上面推导，易知最大值满足 $\theta_0+\phi=\pi / 2 + 2k\pi, \ k \in \mathbb Z$，即

$$\theta_0=\pi/2- \phi+2k \pi, \ k \in Z \tag{4}\label{4}$$

下一步求 $\phi$  。

$$\begin{aligned}\sin \phi&=\frac {\sin \theta_2-\sin \theta_1}{[(\cos \theta_2-\cos \theta_1)^2+(\sin \theta_2-\sin \theta_1)^2]^{1/2}}
\\\\ &=\frac {2 \cos \alpha \sin \beta} {[2-2(\cos \theta_2\cos \theta_1+\sin \theta_2 \sin \theta_1)]^{1/2}}
\\\\ &=\frac {2 \cos \alpha \sin \beta} {[2-2\cos(\theta_2-\theta_!)]^{1/2}}
\\\\ &=\frac {2 \cos \alpha \sin \beta} {(4 \sin^2 \beta)^{1/2}}
\\\\ &= \frac {\cos \alpha \sin \beta} {(\sin^2 \beta)^{1/2}}
\end{aligned}$$

其中 $\alpha=(\theta_2+\theta_1)/2, \ \beta=(\theta_2-\theta_1)/2$，上式推导中，第二个等号用了“和差化积”公式，第三个等号用了“两角和差”公式，第四个等号用了“二倍角”公式。同理可得

$$\cos \phi=\frac {\sin \alpha \sin \beta} {(\sin^2 \beta)^{1/2}}$$

回顾一下本例示意图，不难理解有 $\theta_2-\theta_1 < 2\pi$，于是 $\beta < \pi \Rightarrow \sin \beta > 0$，
此时上述两个等式变成 $\sin \phi = \cos \alpha, \ \cos \phi=\sin \alpha$ 。
我们现在需要将 $\cos \alpha$ 变成 $\sin (\pm \alpha+\gamma)$ 的形式，并且将 $\sin \alpha$ 变成 $\cos (\pm \alpha+\gamma)$，这样就容易得到 $\phi=\pm \alpha+\gamma$，其中 $\gamma$ 是某个待定角，且 $\alpha$ 前面是 $+$ 还是 $-$ 还是 $+,-$ 均可，这一点也需要确定。根据三角函数的诱导公式，列出如下关系：

$$\begin{array}{c|c}
 & \pi/ 2 + \alpha &&  \pi/ 2 - \alpha && 3\pi/ 2 + \alpha && 3\pi/ 2 - \alpha \\\\
\hline 
\alpha \in (0, \pi/2) \quad \sin(\cdot)  & + && + && - && -\\\\
\hline
=&\cos \alpha && \cos \alpha && - \cos \alpha && - \cos \alpha
\\\\
\hline
\\\\
\hline
\alpha \in (0, \pi/2) \quad \cos(\cdot)  & - && + && + && -\\\\
\hline
=&-\sin \alpha && \sin \alpha &&  \sin \alpha && - \sin \alpha
\end{array}$$

（表中，第一项列举了几个范式，第二行表示当 $\alpha \in (0,\pi)$ 时，$\sin(\cdot)$ 的符号，其中 $\cdot$ 表示第一行各范式。第三行表示 $\sin(\cdot)$ 等价关系，这里的等价关系对 $\forall \alpha \in \mathbb R$ 均成立。注意范式中忽略了周期项 $2m\pi$）

根据上表，只有当 $\phi=\pi/2-\alpha$ 时，满足 $\sin \phi=\cos \alpha, \ \cos \phi=\sin \alpha$。故结合 $\eqref{4}$ 式有

$$\phi=\pi/2-\alpha+2m \pi, \ m \in \mathbb Z$$

$$\theta_0=\alpha+2k \pi=(\theta_2+\theta_1)/2+2k \pi, \ k \in \mathbb Z$$

由于周期为 $2\pi$，所以不需要考虑周期项，得 $\theta_0=(\theta_2+\theta_1)/2$，这正如图中所示的 $\mathbf w_i$ 的方向，也就是位于扇形的中间方向。

