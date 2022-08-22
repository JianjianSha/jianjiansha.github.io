---
title: 排序问题
date: 2021-10-09 15:04:39
tags: machine learning
p: ml/rank
mathjax: true
---

排序问题的应用场景例如：搜索引擎根据搜索相关度对结果进行排序。

<!--more-->

# 描述
记序列集合 $\mathcal X^{\star}=\bigcup_{n=1}^{\infty} \mathcal X^n$，其中每个序列长度可能不等，记序列 $\overline {\mathbf x}=(\mathbf x_1,\cdots,\mathbf x_r) \in \mathcal X^{\star}$，长度为 $r \in \mathbb N_+$，序列中每个实例 $\mathbf x \in \mathbb R^d$。假设函数 $h(\overline {\mathbf x}) \in \mathbb R^r$，记这个输出为向量 $\mathbf y$，对 $\mathbf y$ 排序（本文如不特别指定，则均为升序排序），记排序为 $\pi(\mathbf y)$，例如 $\mathbf y=(2,1,6,-1,0.5)$，那么 $\pi(\mathbf y)=(4,3,5,1,2)$ 表示 $\mathbf y$ 个元素在排序后的位置下标，例如 $2$ 升序排列为第 $4$ 位。

# 损失
对于这个有标记的样本，记样本域为 $Z=\bigcup_{r=1}^{\infty}(\mathcal X^r \times \mathbb R^r)$。定义损失

$$l(h,(\overline {\mathbf x}), \mathbf y))=\Delta(h(\overline {\mathbf x}), \mathbf y)$$

损失可以取以下三种形式

1. 0-1 损失

    $\Delta(\mathbf y', \mathbf y)=\mathbb I[\pi(\mathbf y')=\pi(\mathbf y)]$，这个损失缺点是无法区分 $\pi(\mathbf y')$ 与 $\pi(\mathbf y)$ 差异程度。实际应用中这个损失几乎不使用。

2. Kendall-Tau 损失

    $$\Delta(\mathbf y', \mathbf y)=\frac 2 {r(r-1)} \sum_{i=1}^{r-1}\sum_{j=i+1}^r \mathbb I[\text{sign}(y_i'-y_j')\neq \text{sign}(y_i-y_j)] \tag{1}$$

    求和项数为一个等差数列的和 $1+2+\cdots+(r-1)=r(r-1)/ 2$ 。

3. Normalized Discounted Cumulative Gain (NDCG)

    借助一个非递减折扣函数 $D: \mathbb N \rightarrow \mathbb R_+$，累积增益函数为

    $$G(\mathbf y', \mathbf y)=\sum_{i=1}^r D(\pi(\mathbf y')_i) y_i$$

    由于 $D$ 是非递减的，所以 $D(\pi(\mathbf y'))$ 可以看作是 $\mathbf y'$ 的非负权重，$y_i'$ 越大，其权值越大，故有 $0 \le G(\mathbf y', \mathbf y) \le G(\mathbf y, \mathbf y)$ 。定义损失为

    $$\Delta(\mathbf y', \mathbf y)=1-\frac {G(\mathbf y', \mathbf y)} {G(\mathbf y, \mathbf y)}=\frac 1 {G(\mathbf y, \mathbf y)} \sum_{i=1}^r [D(\pi(\mathbf y)_i)-D(\pi(\mathbf y')_i)] y_i \tag{2}$$

    一个常用的折扣函数为

    $$D(i)=\begin{cases} \frac 1 {\log_2 (r-i+2)} & i \in \{r-k+1,\cdots, r\} \\ 0 & \text{o.w.} \end{cases}$$

    上式，$k$ 用于调节序列中 top-k 的元素对损失起作用，或者说对排名有影响，top-k 以下的元素即使排名错误，我们也不关心。

# 线性预测器

线性函数为

$$h_{\mathbf w}(\mathbf x_1, \cdots, \mathbf x_r)=(\mathbf w^{\top}\mathbf x_1, \cdots, \mathbf w^{\top}\mathbf x_r)$$

其中 $\mathbf w \in \mathbb R^d$ 。

样本集 $S=(\overline {\mathbf x}_1, \mathbf y_1), \cdots ,(\overline {\mathbf x}_m, \mathbf y_m)$，其中 $(\overline {\mathbf x}_i, \mathbf y_i) \in (\mathcal X, \mathbb R)^{r_i}$ 。 寻找 $\mathbf w$ 使得经验损失 $\sum_{i=1}^m \Delta(h_{\mathbf w}(\overline {\mathbf x}_i), \mathbf y_i)$ 最小。由于上面介绍的第二、三种损失无法求导，所以使用凸替代方法，与前面分类问题相似，下面介绍这两种损失的 Hinge Loss 形式。

## 损失凸替代

首先是 Kendall-Tau 损失的凸替代。

注意到 $\mathbb I[\text{sign}(y_i'-y_j')\neq \text{sign}(y_i-y_j)]=\mathbb I[\text{sign}(y_i-y_j)(y_i'-y_j')]$，这里可以将 sign 函数写成

$$\text{sign}(x)=\begin{cases} 1 & x \ge 0 \\ -1 & x <0\end{cases}$$

然后根据 $y_i'-y_j'=\mathbf w^{\top}(\mathbf x_i-\mathbf x_j)$，所以

$$\mathbb I[\text{sign}(y_i-y_j)(y_i'-y_j')]\le \max \{0, 1- \text{sign}(y_i-y_j)\mathbf w^{\top}(\mathbf x_i-\mathbf x_j)\}$$

代入 Kendall Tau 损失，得

$$\Delta(h_{\mathbf w}(\overline {\mathbf x}), \mathbf y)\le\frac 2 {r(r-1)}\sum_{i=1}^{r-1}\sum_{j=i+1}^r \max \{0,  1- \text{sign}(y_i-y_j)\mathbf w^{\top}(\mathbf x_i-\mathbf x_j)\} \tag{3}$$

接着是 NDCG 的凸替代。

记 $[r]$ 的所有排列为 $V$，注意到有关系

$$\pi(\mathbf y')=\arg \max_{\mathbf v \in V} \ \sum_{i=1}^r v_i y_i' \tag{4}$$

根据 $\pi(\mathbf y')$ 的定义， $y_i'$ 越大，其对应在 $\pi(\mathbf y')$ 中的值就越大，将 $v_i$ 看作 $y_i'$ 的权值，由于权值非负，那么较大的数分配较大的权值，加权和自然是最大。

定义 $\Psi(\overline {\mathbf x}, \mathbf v)=\sum_{i=1}^r v_i \mathbf x_i$，那么根据 (4) 式有

$$\begin{aligned} \pi(h_{\mathbf w}(\overline {\mathbf x}))&=\arg \max_{\mathbf v \in V} \ \sum_{i=1}^r v_i \mathbf w^{\top} \mathbf x_i
\\&=\arg \max_{\mathbf v \in V} \  \mathbf w^{\top} \left(\sum_{i=1}^r v_i \mathbf x_i \right)
\\&=\arg \max_{\mathbf v \in V} \ \mathbf w^{\top} \Psi(\overline {\mathbf x}, \mathbf v)
\end{aligned}$$

上式意味着 $\mathbf w^{\top} \Psi(\overline {\mathbf x}, \pi(h_{\mathbf w}(\overline {\mathbf x}))) \ge \mathbf w^{\top} \Psi(\overline {\mathbf x}, \pi(\mathbf y))$，当且仅当 $ \pi(h_{\mathbf w}(\overline {\mathbf x})) = \mathbf y$ 时等号成立，于是

$$\begin{aligned}\Delta(h_{\mathbf w}(\overline {\mathbf x}), \mathbf y)&\le \Delta(h_{\mathbf w}(\overline {\mathbf x}), \mathbf y)+ \mathbf w^{\top}[\Psi(\overline {\mathbf x}, \pi(h_{\mathbf w}(\overline {\mathbf x})))-\Psi(\overline {\mathbf x}, \pi(\mathbf y))]
\\& \le \max_{\mathbf v \in V} \ \{\Delta(\mathbf v, \mathbf y)+\mathbf w^{\top}[\Psi(\overline {\mathbf x}, \mathbf v)-\Psi(\overline {\mathbf x}, \pi(\mathbf y))]\}
\\&=\max_{\mathbf v \in V} \ \left[\Delta(\mathbf v, \mathbf y)+\sum_{i=1}^r(v_i-\pi(\mathbf y)_i) \mathbf w^{\top}\mathbf x_i\right]
\end{aligned}$$

上面的推导过程中，$\Delta(\mathbf v, \mathbf y)$ 的参数 $\mathbf v$ 实际应为对应的某个 $\mathbf y'$ 满足 $\pi(\mathbf y')=\mathbf v$，但是为了突出这个凸替代函数与 $\mathbf v$ 有关，所以直接使用 $\mathbf v$ 作为参数。代入 NDCG 损失，则优化目标为

$$L= \frac 1 {G(\mathbf y, \mathbf y)}\sum_{i=1}^r [D(\pi(\mathbf y)_i)-D(v_i)]y_i+\sum_{i=1}^r (v_i-\pi(\mathbf y)_i)\mathbf w^{\top}\mathbf x_i$$

优化问题（即，NDCG 损失的凸替代）可以进一步改写为

$$\arg \min_{\mathbf v \in V} \ \sum_{i=1}^r (\alpha_i v_i+\beta_i D(v_i))$$

其中 $\alpha_i=-\mathbf w^{\top}\mathbf x_i, \ \beta_i=y_i/G(\mathbf y, \mathbf y)$ 。定义矩阵 $A \in \mathbb R^{r,r}$ 如下

$$A_{ij}=j \alpha_i + D(j) \beta_i$$

可以将 $A_{ij}$ 理解为将任务 $i$ 分配给工人 $j$ 完成所需的成本，总共 $r$ 个工人， $r$ 个任务，目标则变成求如何分配，使得总成本最小。这个分配问题可以采用“KM算法”或者“线性规划算法”解决。

## 线性规划算法

将分配问题重新改写如下

$$\arg \min_{B \in \mathbb R_+^{r,r}} \sum_{i,j=1}^r A_{ij} B_{ij}$$

$$\begin{aligned} s.t. \quad & \forall i \in [r], \ \sum_{j=1}^r B_{ij}=1
\\& \forall j \in [r], \ \sum_{i=1}^r B_{ij}=1
\\& \forall i,j, \ B_{ij} \in \{0,1\}
\end{aligned}$$

上式中，$B$ 矩阵相当于将 $r$ 个 $1$ 放在不同行不同列，其余矩阵元素均为 $0$。这样的 $B$ 一共有 $r!$ 种，这种矩阵称作置换矩阵。为方面起见，记 $\langle A, B\rangle=\sum_{i,j} A_{ij}B_{ij}$ 。

上面最后一个约束条件 $\forall i,j, \quad B_{ij} \in \{0,1\}$ 可以直接去掉，即 $B_{ij} \in \mathbb R$ 就可以，最终解不变。下面对此进行证明。

分配问题：
$$\arg \min_{B \in \mathbb R_+^{r,r}} \sum_{i,j=1}^r A_{ij} B_{ij}$$

$$\begin{aligned} s.t. \quad & \forall i \in [r], \ \sum_{j=1}^r B_{ij}=1
\\& \forall j \in [r], \ \sum_{i=1}^r B_{ij}=1
\end{aligned} \tag{5}$$


令 $B$ 为 (5) 式的最优解。$B$ 可以写为

$$\begin{aligned} B=\sum_i \gamma_i C_i
\\ \gamma_i > 0, \ \sum_i \gamma_i = 1 \end{aligned}\tag{6}$$

其中 $C_i$ 为置换矩阵，即 $r$ 个 $1$ 分别位于不同行不同列，其余元素为 $0$。这个等到后面再予以证明。

由于 B 是最优解，那么必然有 $\forall i, \ \langle A, B\rangle \le \langle A,C_i\rangle$，且必然 $\exists \ i$ 使得等号成立，这是因为如果 $\forall i, \ \langle A, B\rangle < \langle A,C_i\rangle$，那么必然有 

$$\langle A, B\rangle=\langle A, \sum_i \gamma_i C_i\rangle = \sum_i \gamma_i \langle A, C_i\rangle > \sum_i \gamma_i \langle A, B\rangle = \langle A, B\rangle$$

显然矛盾。所以必然 $\exists \ i$，使得 $\langle A, B\rangle = \langle A,C_i\rangle$，由于 $B$ 是最优解，那么自然 $C_i$ 也是最优解，即 (5) 式最优解为一个置换矩阵。证毕。

下面证明 (6) 式成立。


根据 $B=\sum_i \gamma_i C_i$，前面讲到置换 $C_i$ 一共有 $r!$ 个，于是有关系

$$B=\sum_{i=1}^{r!} \gamma_i C_i$$

所以，如果上式存在解，那么就表明上式关系成立。代入矩阵各元素，那么上式实际上是 $r^2$ 个等式，$r!$ 个变量，即由 $r^2$ 组成的 $r!$ 元一次方程组。现在我们将这 $r^2$ 个 $r!$ 元一次方程按矩阵行分组进行列举，

第 1 组

$$\sum_{i \in S_{11}} \gamma_i=B_{11}
\\ \vdots
\\\sum_{i \in S_{1r}} \gamma_i=B_{1r}$$

其中 $S_{mn}$ 表示满足关系 $(C_i)_{mn}=1$ 的那些置换矩阵 $C_i, \ i \in [r!]$ 的下标集合，且 $\bigcup_{n=1}^r S_{mn}=[r!]$，于是

$$\sum_{k=1}^r B_{1k}=\sum_{k=1}^r \sum_{i \in S_{1k}} \gamma_i =\sum_{i=1}^{r!} \gamma_i=1$$

这证明了 (6) 式中第二个等式关系。

第 k 组

$$\sum_{i \in S_{k1}} \gamma_i=B_{k1}
\\ \vdots
\\\sum_{i \in S_{kr}} \gamma_i=B_{kr}$$

$\cdots$


根据 $B$ 的行和和列和均为 1，那么 $B$ 中元素取值自由度实际上 $r^2-2r+1=(r-1)^2$，令 $f(r)=(r-1)^2$，这表示只要确定了 $B$ 中任意 $f(r)$ 个元素的值，那么 $B$ 中所有元素的值均被确定。这是因为，原本 $B$ 的元素取值自由度为 $r^2$，根据行和为 1，自由度变成 $r^2-r$，即每行有一个值由本行其他元素值确定，不妨令最后一个元素值不自由；根据列和为 1，那么每列有一个值由本列其他元素值确定，显然最后一列不用考虑了，因为最后一列的值已经不自由了，所以此时是少了 $r-1$ 个自由度，于是最终元素取值自由度为 $r^2-r-(r-1)=(r-1)^2$。

例如，$r=2$，此时 $f(2)=1$，$B$ 的形式为 $\begin{bmatrix}B_{11} & 1-B_{11} \\ 1-B_{11} & B_{11}\end{bmatrix}$；或者 $r=3$，此时 $f(3)=4$，此时矩阵形式为

$$\begin{bmatrix}B_{11} &  B_{12} & 1-B_{11}-B_{12} \\  B_{21} & B_{22} & 1-B_{21} - B_{22}\\ 1-B_{11}-B_{21} &  1-B_{12}-B_{22} & B_{11}+B_{12}+B_{21}+B_{22}-1\end{bmatrix}$$

所以，我们只需要 $(r-1)^2$ 个等式，例如 $B$ 的左上角的 $r-1$ 阶子方阵的元素，即

$$\sum_{i \in S_{1,1}} \gamma_i=B_{1,1}
\\ \vdots
\\\sum_{i \in S_{1,r-1}} \gamma_i=B_{1,r-1}
\\\vdots 
\\\sum_{i \in S_{r-1,1}} \gamma_i=B_{r-1,1}
\\ \vdots
\\\sum_{i \in S_{r-1,r-1}} \gamma_i=B_{r-1,r-1}
$$

剩余的 $2r-1$ 个等式则全部与 $\sum_{i=1}^{r!} \gamma_i=1$ 这一个等式等价，所以最终变成 **具有 $(r-1)^2+1$ 个等式的 $r!$ 元一次方程组** 。


令 $d=r!-[(r-1)^2+1]$ ，即 $d$ 个 $\gamma_i$ 值是自由的，可以令它们均为 0，从而确定剩余的 $(r-1)^2+1$ 个 $\gamma_i$ 的值，再根据 $\sum_{i=1}^{r!} \gamma_i=1$，可知剩余的 $(r-1)^2+1$ 个 $\gamma_i$ 的和依然为 1，

于是 

$$B=\sum_i^{(r-1)^2+1} \gamma_i C_i \ , \quad \sum_i^{(r-1)^2+1} \gamma_i=1$$

当 $r=2$ 时，$r!=(r-1)^2+1$，方程组具有唯一解。

当 $r\ge3$ 时，$r!>(r-1)^2+1$，方程组具有无穷多解。