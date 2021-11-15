---
title: 二分排序
date: 2021-10-12 11:42:08
tags: machine learning
mathjax: true
p: ml/bi_rank
---

本文讨论二分排序的问题。所谓“二分排序”，指的是 $\mathbf y \in \{\pm 1\}^r$ ，表示样例 $\mathbf x_i$ 要么是相关的，要么是不相关的。

<!--more-->

乍一看，二分排序跟二分类很像，但是用二分类的常用方法来解决二分排序问题，往往效果不好，这是因为排序问题中，相关的实例（或直接称作 正例）的占比很小，例如在缺陷检测中，绝大多少样本（例如 99.9%）都是良好的，那么二分类的学习方法得到的判决器对所有样本都判为负例，那么错误率已经非常小（0.1%），这显然对我们的缺陷检测这一目标毫无意义。二分类学习方法不奏效的根本原因是使用了 0-1 损失（或者它的凸替代），这一损失不足以解决排序问题。我们介绍另外的对二分排序问题适用的损失函数。

# 损失
实例序列 $\overline {\mathbf x}=(\mathbf x_1,\cdots, \mathbf x_r)$，预测值 $\mathbf y'=\mathbf w^{\top} \mathbf x \in \mathbb R^r$，这里为简单起见，直接使用 $\mathbf x$ 而非 $\Psi(\mathbf x, \mathbf y)$ 作为特征， target 值为 $\mathbf y \in \{\pm 1\}^r$，所以还需要一个参数 $\theta$，以便将预测向量转化为二值向量，

$$b(\mathbf y')=(\text{sign}(y_1'-\theta),\cdots, \text{sign}(y_r'-\theta))$$

通常设置 $\theta=0$，但也有时候在某些具体问题中需要满足某些约束，从而调整 $\theta$ 值。

列出 `真阳，假阳，假阴，真阴` 四个概念如下：

True positives： $\ a=|\{i:y_i=+1 \land \text{sign}(y_i'-\theta)=+1\}|$

False positives： $\ b=|\{i:y_i=-1 \land \text{sign}(y_i'-\theta)=+1\}|$

False negatives： $\ c=|\{i:y_i=+1 \land \text{sign}(y_i'-\theta)=-1\}|$

True negatives： $\ d=|\{i:y_i=-1 \land \text{sign}(y_i'-\theta)=-1\}|$

recall（真阳在阳例中的占比）： $\ \frac a {a+c}$

precision（真阳在预测阳例中的占比）： $\ \frac a {a+b}$

specificity（特异度：真阴在阴例中的占比）： $\ \frac d {d+b}$


如果 $\theta \uparrow$，那么 $recall \downarrow$，但是 $precision \uparrow$


以下损失结合了 recall 以及 precision，也称 “多变量性能测度” （$0-1$ 损失在这里为 $\frac {b+d}{a+b+c+d}$）。

1. 利用 recall 和 specificity 的均值作为测度

    $$\Delta(\mathbf y', \mathbf y)=1-\frac 1 2 (\frac a {a+c}+\frac d {d+b})$$

    当 $b=c=0$ 时，即没有假阳和假阴，全部预测正确，此时 $\Delta=0$

2. F1-score

    $F_1$ -score 计算如下

    $$F_1=\frac 1 {\frac 1 {\text{precision}}+\frac 1 {\text{recall}}}=\frac {2a}{2a+b+c}$$

    相关损失

    $$\Delta(\mathbf y', \mathbf y)=1-F_1$$

3. $F_{\beta}$-score

    $$F_{\beta}=\frac {1+\beta^2} {\frac 1 {\text{precision}}+\beta^2\frac 1 {\text{recall}}}=\frac {(1+\beta^2)a} {(1+\beta^2)a+b+\beta^2 c}$$

    相关损失

    $$\Delta(\mathbf y', \mathbf y)=1-F_{\beta}$$

以下介绍两个概念：

1. Recall at $k$ ：$a+b \le k$

2. precision at $k$：$a+b \ge k$

# 线性预测器

记参数为 $\mathbf w$，线性预测器为

$$\mathbf y'=h_{\mathbf w}(\overline {\mathbf x})=(\mathbf w^{\top}\mathbf x_1, \cdots, \mathbf w^{\top}\mathbf x_r)$$

转到二值向量，

$$\begin{aligned}b(\mathbf y')&=(\text{sign}(y_1'-\theta),\cdots, \text{sign}(y_r'-\theta))
\\&=\arg \max_{\mathbf v \in V} \ \sum_{i=1}^r v_i y_i'
\end{aligned}$$

其中 $V=\{\pm 1\}^r$ 。

## 凸替代

记损失函数为 $\Delta$，可以是上述三个损失之一。那么有

$$\begin{aligned}\Delta(h_{\mathbf w}(\overline {\mathbf x}), \mathbf y)&=\Delta(b(h_{\mathbf w}(\overline {\mathbf x})), \mathbf y)
\\ & \le \Delta(b(h_{\mathbf w}(\overline {\mathbf x})), \mathbf y)+\sum_{i=1}^r (b(h_{\mathbf w}(\overline {\mathbf x}))_i-y_i)\mathbf w^{\top}\mathbf x_i
\\ & \le \max_{\mathbf v \in V} \left[\Delta(\mathbf v, \mathbf y)+\sum_{i=1}^r(v_i-y_i) \mathbf w^{\top}\mathbf x \right]
\end{aligned} \tag{1} \label{1}$$

由于 $|V|=2^r$，当 $r$ 较大时，$|V|$ 较大，这就导致 $\eqref{1}$ 式右侧最大值计算较为耗时，需要寻求一种高效的计算方法。

对 $\forall a,b \in [r]$，令

$$\mathcal Y_{a,b}=\{\mathbf v: |\{i: v_i=1 \land y_i=1\}|=a \land |\{i:v_i=1 \land y_i=-1\}|=b\}$$

$\mathcal Y_{a,b}$ 表示在给定 target $\mathbf y$ 的情况下，满足真阳数量为 $a$，且假阳数量为 $b$ 的所有预测序列集合。

固定 $a,b$ 值，在 $\mathcal Y_{a,b}$ 集合中求目标局部最大值，然后比较各个局部最大值，进而求得全局最大值。

在 $\mathcal Y_{a,b}$ 集合中求局部最大值的时候，观察  $\eqref{1}$ 式优化目标，发现 $y_i \mathbf w^{\top}\mathbf x$ 是固定不变的，由于固定了 $a,b$，所以 $c,d$ 也确定下来，所以对任意 $\mathbf v \in \mathcal Y_{a,b}$，$\Delta(\mathbf v, \mathbf y)$ 也固定不变（这对上述三个损失均成立），所以只要求以下最大值即可，

$$\max_{\mathbf v \in \mathcal Y_{a,b}} \ \sum_{i=1}^r v_i \mathbf w^{\top} \mathbf x_i \tag{2} \label{2}$$

要求 $\eqref{2}$ 这个最大值，由于 $v_i \in \{\pm 1\}$，所以理论上只需要令 $v_i=\text{sign}(\mathbf w^{\top}\mathbf x_i)$。

然而由于 $\mathcal Y_{a,b}$ 的存在，使得真阳数量必须为 $a$，假阳数量必须为 $b$，也就是说，预测 $v_i=1$ 的数量固定为 $a+b$，因为 $v_i=1$，所以 $v_i \mathbf w^{\top} \mathbf x_i=\mathbf w^{\top} \mathbf x_i$，故需要 $\mathbf w^{\top} \mathbf x_i$ 尽可能大，才符合 $\eqref{2}$ 式的求最大值这一目标，即，尽可能地将较大的 $\mathbf w^{\top} \mathbf x_i$ 赋予 $v_i=1$ 。于是，根据真阳数量 $a$ 和假阳数量 $b$，分别在正例中将 top-a 的 $\mathbf w^{\top} \mathbf x_i$ 以及负例中 top-b 的 $\mathbf w^{\top} \mathbf x_i$ 所对应的样本预判为正 $v_i=1$，具体操作如下：

对样例进行排序，使得 

$$\mathbf w^{\top} \mathbf x_1 \ge \cdots \ge \mathbf w^{\top} \mathbf x_r$$

其下标 $1,2,\cdots, r$ 是排序后的下标。此时根据 target $\mathbf y$ 值，将样例分为正例和负例两部分，每个部分内部是保持 $\mathbf w^{\top} \mathbf x_i$ 的降序排列，记这两部分的下标集合为

$$S_+=\{i_1,\cdots, i_P\}
\\S_-=\{j_1,\cdots, j_N\}$$

其中 $P=|\{i:y_i=1\}|, \ N=|\{i:y_i=-1\}|$，且有

$$\mathbf w^{\top} \mathbf x_{i_1} \ge \cdots \ge \mathbf w^{\top} \mathbf x_{i_P}
\\\mathbf w^{\top} \mathbf x_{j_1} \ge \cdots \ge \mathbf w^{\top} \mathbf x_{j_N}$$

那么，只要令 $i_1,\cdots, i_a$ 以及 $j_1,\cdots, j_b$ 对应的样本预测为正，

$$v_{i_1}=\cdots = v_{i_a}=v_{j_1}=\cdots = v_{j_b}=1$$

另外，剩余的样例中，由于其 $\mathbf w^{\top} \mathbf x_i$ 较小，所以 $v_i \mathbf w^{\top} \mathbf x_i=-\mathbf w^{\top} \mathbf x_i$ 也是尽可能地大，符合 $\eqref{2}$ 求最大值这一目标。

求 $\eqref{2}$ 式的解 $\mathbf v^{\star}$ 后，根据 $\eqref{1}$ 式计算相应地目标局部最大值

$$L_{a,b}=\left[\Delta(\mathbf v^{\star}, \mathbf y)+\sum_{i=1}^r(v_i^{\star}-y_i) \mathbf w^{\top}\mathbf x \right]$$

对每一组 $(a,b)$ 值，均计算其局部最大值 $L_{a,b}$ 以及对应的解 $\mathbf v^{\star}$，最后进行比较，可以求得全局最大值以及相应的解。

当然，由于 $y_i\mathbf w^{\top} \mathbf x_i$ 与 $\mathbf v$ （以及 $a,b$）无关，所以也可以更简单的求 

$$L_{a,b}=\left[\Delta(\mathbf v^{\star}, \mathbf y)+\sum_{i=1}^r v_i^{\star} \mathbf w^{\top}\mathbf x \right]$$

然后比较求全局最大值和相应的解。

以上这种算法高效的本质是，将 $V$ 按 $(a,b)$ 值划分成若干组，每组求组内最大值，而每组内部由于 $\Delta(\mathbf v, \mathbf y)$ 以及 $y_i\mathbf w^{\top}\mathbf x$ 均相等，所以在组内演变成求 $\eqref{2}$ 式，而 $\eqref{2}$ 式可以快速求最优值和最优解。

注：上述计算过程中，$\theta$ 采用默认值 0 。

计算 $\eqref{1}$ 式步骤总结如下：

---
**输入：** $(\mathbf x_1, \cdots, \mathbf x_r)$，$(y_1,\cdots, y_r)$，$\mathbf w, \ V, \Delta$

**初始化：**

&emsp; $P=|\{i:y_i=1\}|, \ N=|\{i:y_i=-1\}|$

&emsp; $\boldsymbol \mu=($$\mathbf w^{\top} \mathbf x_1 , \cdots , \mathbf w^{\top} \mathbf x_r$$)$，$\alpha^{\star}=-\infty$

&emsp; 排序使得 $\mu_1 \ge \cdots \ge \mu_r$

&emsp; 排序后正、负例下标集合分别为： $S_+=\{i_1,\cdots, i_P\}, \ S_-=\{j_1,\cdots, j_N\}$

**for** $\ a=0,1,\cdots, P$

&emsp; $c=P-a$

&emsp; **for** $\ b=0,1,\cdots,N$

&emsp; &emsp; $d=N-b$

&emsp;&emsp; 计算 $\Delta$

&emsp;&emsp; 设置 $\mathbf v$ 使得 $v_{i_1}=\cdots = v_{i_a}=v_{j_1}=\cdots = v_{j_b}=1$，其余的 $v_i$ 全部设置为 $v_i=-1$

&emsp;&emsp; $\alpha=\Delta+\sum_{i=1}^r v_i \mu_i$

&emsp;&emsp; **if** $\alpha > \alpha^{\star}$

&emsp;&emsp;&emsp; $\alpha = \alpha^{\star}, \ \mathbf v^{\star}=\mathbf v$

**输出：** $\ \mathbf v^{\star}$

---

然后可以使用 [多分类算法](2021/09/29/ml/multiclass_algo) 一文中所述算法学习模型参数。