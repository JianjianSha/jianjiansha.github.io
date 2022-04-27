---
title: 线性代数基本概念总结
date: 2022-02-10 09:40:14
tags: 
    - math
    - machine learning
p: math/linear_algebra_concepts
---

总结线性代数的常用知识点。
<!--more-->

# 1. 解线性方程组

形式如
$$A\mathbf x = \mathbf b, \quad A \in \mathbb R^{m \times n}$$

解线性方程组之前先来了解几个概念。

**基本变换**

1. 交换两个等式位置（即，矩阵或增广矩阵的行）
2. 对某个等式两边乘以一个非 0 常数
3. 两个等式相加

> 基本变换不会改变线性方程组的解。

**行阶梯矩阵**

满足：

1. 全零行在矩阵下方，非全零行（row 中存在 nonzero 元素）在矩阵上方
2. 对非全零行而言，左侧起第一个 nonzero 元素称为 `pivot`，下方行 pivot 严格在上方行 pivot 的右侧。这里的“严格”表明不能在同一列。


**基本/自由 变量**

行阶梯型矩阵中，pivot 对应的变量称为基本变量，其他变量称为自由变量。

**规约行阶梯矩阵**

满足：

1. 是行阶梯型矩阵
2. 每个 pivot 值为 `1`
3. pivot 是所在列唯一的 nonzero 元素

举例说明，下方是一个规约行阶梯矩阵，粗体的 `1` 均为 pivot，没有其他 pivot 了，pivot 所在列的其他元素均为 `0`。

$$A=\begin{bmatrix}\mathbf 1 & 3 & 0 & 0 & 3
\\ 0 & 0 & \mathbf1 & 0 & 9
\\ 0 & 0 & 0 & \mathbf 1 & -4\end{bmatrix}$$

规约行阶梯矩阵有利于解方程组。

**一个简单例子：**

例如齐次方程组 $A\mathbf x = \mathbf 0$，其中 $A$ 就是上面这个，那么展开得

$$A_{:,1} x_1 +A_{:,2}x_2 + A_{:,3}x_3 +A_{:,4}x_4 + A_{:,5}x_5 = \mathbf 0 \tag{1}$$ 

由于 pivot 所在列已经是基本向量了，所以可用 pivot 列来表示非 pivot 列，

$$\begin{aligned} A_{:,2}&=3A_{:,1}
\\A_{:,5}&=3A_{:,1}+9A_{:,3}-4A_{:,4}\end{aligned}$$

移项得，

$$\begin{aligned} 3A_{:,1}- A_{:,2}&=\mathbf 0
\\3A_{:,1}+9A_{:,3}-4A_{:,4} -A_{:,5}&=\mathbf 0 \end{aligned} \tag{2}$$

上面方程组（2）的第一个等式的一个特殊解为 $\mathbf x=[3,-1,0,0,0]^{\top}$，一般解为 $\mathbf x=\lambda_1 [3,-1,0,0,0]^{\top}, \quad \lambda_1 \in \mathbb R$ 。

第二个等式的一个特殊解为 $\mathbf x=[3, 0, 9,-4,-1]^{\top}$，一般解为 $\mathbf x=\lambda_2 [3, 0, 9,-4,-1]^{\top}, \quad \lambda_2 \in \mathbb R$ 。


由于方程组 (2) 的两个等式均满足 (1) 式，故两个一般解均是 (1) 的解，于是最终的一般解为 

$$\mathbf x = \lambda_1 \begin{bmatrix} 3 \\ 1 \\ 0 \\ 0 \\ 0 \end{bmatrix} + \lambda_2 \begin{bmatrix} 3 \\ 0 \\ 9 \\ -4 \\ -1 \end{bmatrix}, \quad \lambda_1, \lambda_2 \in \mathbb R$$

**高斯消除：**
使用 **基本变换** 将线性方程组转变为规约行阶梯型。

# 2. 几种算法
1. 高斯消除法
2. 求逆法

    $$A \mathbf x = \mathbf b \Rightarrow \mathbf x=A^{-1} \mathbf b$$

    这种方法要求 $A$ 可逆，否则，可以求 Moore-Penrose 逆，

    $$A \mathbf x = \mathbf b \Leftrightarrow A^{\top}A \mathbf x = A^{\top} \mathbf b \Leftrightarrow \mathbf x=(A^{\top}A)^{-1}A^{\top}\mathbf b$$

3. 数值计算法

    实际应用中（例如未知数量达百万级别），上面 `1,2` 方法计算量都会特别大，此时可以采用数值计算法，通常是迭代计算，例如 Richardson 法，Jacobi 法，$Gau\beta-Seidel$ 法等。参见 《Stoer, Josef, and Burlirsch, Roland. 2002. Introduction to Numerical Analysis. Springer.》，《Strang, Gilbert. 2003. Introduction to Linear Algebra. Wellesley-Cambridge Press》。

# 3. 向量空间

向量空间 $V$ 的相关概念这里不再列出。

## 3.1 rank
矩阵的秩的一些重要性质

假设 $A \in \mathbb R^{m \times n}, \ \mathbf b \in \mathbb R^m$，
1.  $A \mathbf x=\mathbf b$ 有解 $\Leftrightarrow rk(A)=rk(A|\mathbf b)$

    $A$ 的行阶梯型与 $A|\mathbf b$ 的行阶梯型，下方具有相同数量的全零行（ `0` 个全零行或更多），才能有解。这很好理解，否则会出现 $0 \neq 0$，矛盾

2. 满足 $A\mathbf x=\mathbf 0$ 的解构成 $A$ 的 kernel，记为 $K$ ；所有满足 $\mathbf b=A\mathbf x, \ \forall \mathbf x \in \mathbb R^n$ 的 $\mathbf b$ 向量 构成 $A$ 的 image，记为 $U$。有以下关系，
    $$dim(U) = rk(A), \ dim(K) + rk(A)= n$$

## 3.2 线性映射的矩阵表示

两个向量空间 $V,W$，基向量分别为 $B=(\mathbf b_1,\ldots, \mathbf b_n), C=(\mathbf c_1, \ldots, \mathbf c_m)$，考虑一个线性变换 $\Phi:V\rightarrow W$，那么对 $j \in \{1,\ldots, n\}$，
$$\Phi(\mathbf b_j)=\sum_{i=1}^m a_{ij} \mathbf c_i$$

那么 $\Phi$ 对应的转换矩阵为 $A_{\Phi}$，满足 $A_{\Phi}(i,j)=a_{ij}$。

$\Phi(\mathbf b_j)$ 在 $W$ 空间中，其关于向量基 $C$ 的坐标为 $A_{\Phi}$ 的第 $j$ 列，也就是说，线性变换 $\Phi$ 将 $\mathbf b_j$ 变换到 $A_{\Phi}$ 第 `j` 列坐标（所表示的向量）。

对于 $\mathbf x \in V$ ，其坐标记作 $\mathbf {\hat x}$（除特别说明，$V$ 中向量坐标均基于 $B$，$W$ 中向量坐标均基于 $C$），其线性变换结果为 $\mathbf y=\Phi(\mathbf x)$，位于 $W$ 中，坐标记为 $\mathbf {\hat y}$，那么根据定义，

$$\begin{aligned}\mathbf y=\Phi(\mathbf x)&=\Phi(B\mathbf {\hat x})
\\&=\Phi\left(\sum_{j=1}^n \mathbf b_j \hat x_j\right)
\\&=\sum_{j=1}^n \Phi(\mathbf b_j \hat x_j)
\\&=\sum_{j=1}^n \Phi(\mathbf b_j) \hat x_j
\\&=\sum_{j=1}^n \sum_{i=1}^m a_{ij} \mathbf c_i \hat x_j
\\&=\sum_{i=1}^m [A_{\Phi}(i,:) \mathbf {\hat x}] \mathbf c_i\end{aligned}$$

其中 $A_{\Phi}(i,:)$ 表示矩阵 $A_{\Phi}$ 的第 `i` 行。用坐标表示则为，

$$\mathbf {\hat y}=A_{\Phi} \mathbf {\hat x}$$


**Rank-Nullity 定理**

$$dim(ker(\Phi))+dim(Im(\Phi))=dim(V)$$

## 3.3 解析几何

**对称/正定 矩阵**

**对称：** $A=A^{\top}$

**正定：** $\mathbf x^{\top} A \mathbf x > 0, \ \forall \mathbf x \in V \setminus {\mathbf 0}$

对称正定矩阵 $A$ 的性质：
1. $A$ 的 kernel 为 $\{\mathbf 0\}$
2. $A$ 的主对角线元素 $a_{ii} > 0$，因为 $a_{ii}=\mathbf e_i^{\top} A \mathbf e_i > 0$

**正交矩阵**

$A \in \mathbf R^{n \times n}$ 是正交矩阵，当且仅当其列向量是正交归一向量。

正交矩阵性质：
1. $AA^{\top}=I=A^{\top}A$
2. $A^{-1}=A^{\top}$

3. 正交矩阵所表示的映射 $\Phi$ 不会改变 $\mathbf x$ 的长度
4. $\mathbf x, \mathbf y$ 的夹角等于 $\Phi(\mathbf x), \Phi(\mathbf y)$ 的夹角

**投影**

$U \subseteq V$ 是两个空间，映射 $\pi: V \rightarrow U$ 称为投影，如果 $\pi^2 = \pi \circ \pi = \pi$

令矩阵 $P_{\pi}$ 表示投影 $\pi$，那么 $P_{\pi}^2=P_{\pi}$

## 3.4 矩阵

**行列式**

性质：
1. $det(AB)=det(A) \cdot det(B)$
2. $det(A)=det(A^{\top})$
3. $A$ 可逆，那么 $det(A^{-1})=1/ det(A)$
4. 将某行（列）乘上一个常数，再加到零一行（列）上，不改变行列式值
5. 某行（列）乘上一个常数 $\lambda$，行列式值也变成原来的 $\lambda$ 倍
6. 交行两行（列）改变行列式符号（正负性）


根据最后三条性质，可以将 $A$ 通过高斯消除变成行阶梯型，从而求 $det(A)$

**迹（trace）**

矩阵 $A \in \mathbb R^{n \times n}$ 的 trace 定义为
$$tr(A):=\sum_{i=1}^n a_{ii}$$

性质：
1. $tr(A+B)=tr(A)+tr(B)$，其中 $A, B \in \mathbb R^{n \times n}$
2. $tr(\alpha A)=\alpha \cdot tr(A)$，其中 $\alpha \in \mathbb R$
3. $tr(AB)=tr(BA)$，其中 $A \in \mathbb R^{n \times k}, \ B \in \mathbb R^{k \times n}$

    根据第 `3` 条性质，可知 $tr(ABC)=tr(CAB)=tr(BCA)$，推广到更一般的情况，有性质 `4`。
4. m 个矩阵相乘，循环轮转矩阵位置，其乘积矩阵的 trace 保持不变

5. 线性映射 $\Phi$ 对于不同的 $V$ 空间的基向量组，其有不同的矩阵表示，但是这些矩阵的 trace 均相同

    例如 $\Phi$ 在 $V$ 的某一个基向量组下的矩阵表示记为 $A$，在 $V$ 的另一个基向量组下的矩阵表示记为 $B$，那么一定可以找到 $S$，使得 $B=S^{-1}A S$（<font color='magenta'>这样的 $A,B$ 称为相似矩阵</font>），于是
        $$tr(B)=tr(S^{-1}AS)=tr(ASS^{-1})=tr(A)$$

**特征多项式**

$$\begin{aligned}p_{A}(\lambda) &:= det(A-\lambda I)
\\ &=c_0+c_1 \lambda + \cdots + c_{n-1} \lambda ^{n-1}+(-1)^n \lambda ^n\end{aligned}$$

第二个等式是使用 Laplace 展开，其中

$$c_0=det(A)
\\c_{n-1}=(-1)^{n-1} tr(A)$$

**特征值与特征向量**

$$A\mathbf x = \lambda \mathbf x$$

其中 $\lambda \in \mathbb R$，且 $\mathbf x \neq \mathbf 0$。

性质：
1. $rk(A-\lambda I_n) < n$

    $$A\mathbf x = \lambda \mathbf x \Leftrightarrow (A-\lambda I_n) \mathbf x=\mathbf 0 \Leftrightarrow ker(A-\lambda I_n) \neq \{\mathbf 0\}$$
2. $det(A-\lambda I_n)=0$
3. $A$ 的特征值 $\lambda$ 也是其特征多项式 $p_{A}(\lambda)$ 的根（root）
4. $p_{A}(\lambda)$ 的 root 出现的次数就是对应特征值的**代数重数**

    例如多项式有一个 二重根 $\lambda_i$，那么特征值 $\lambda_i$ 的代数重数就是 `2`。

5. $A$ 与 $A^{\top}$ 的特征值均相同，但是对应的特征向量不一定相同
6. 特征空间 $E_{\lambda}$ 是 $A-\lambda I$ 的 kernel 空间
7. 相似矩阵有相同的特征值
8. <font color="red">对阵正定矩阵的特征值是正实数</font>

    $$\mathbf x^{\top}A\mathbf x>0 \Leftrightarrow \mathbf x^{\top} \lambda \mathbf x>0 \Leftrightarrow \lambda > 0$$

9. $A$ 的一个特征值 $\lambda_i$ 所关联的特征空间的维度就是其**几何重数**

    几何重数不能大于代数重数，但是可能会小于代数重数。例如，
    $$A=\begin{bmatrix}2 & 1 \\ 0 & 2 \end{bmatrix}$$
    特征值 $\lambda_1=\lambda_2=2$，代数重数为 `2`，但是只有一个特征向量（线性相关的其他向量不算）$\mathbf x_1=[1, 0]^{\top}$，故几何重数为 `1`。
10. $det(A)=\prod_{i=1}^n \lambda_i$，其中 $\lambda_i \in \mathbb C$ 是 $A$ 特征值（可能有重复）。
11. $tr(A)=\sum_{i=1}^n \lambda_i$，其中 $\lambda_i \in \mathbb C$ 是 $A$ 特征值（可能有重复）。



<font color="red">**重要定理：**</font>

给定 $A \in \mathbf R^{m \times n}$，那么 $S:=A^{\top}A$ 是对称半正定矩阵。如果 $rk(A)=n$，那么 $S$ 是对称正定矩阵，即 <font color="red">满秩的且对角线元素值为正的对称矩阵是正定的</font>。

对称是显而易见的。
1. 证明半正定性，

    $$\mathbf x^{\top}S\mathbf x=\mathbf x^{\top}A^{\top}A\mathbf x\Leftrightarrow \mathbf y^{\top}\mathbf y \ge 0$$
    其中，$\mathbf y= A \mathbf x \in \mathbb R^{m\times 1}$

2. 证明正定性，

    $rk(A)=n \Rightarrow m\ge n$，$A$ 所有 $n$ 个列均线性无关，那么对 $\forall \mathbf x \in \mathbb R^n \setminus \mathbf 0$，均有 $\mathbf y = [A_1,\ldots, A_n]\mathbf x \neq \mathbf 0$，其中 $A_j$ 表示第 `j` 列，故 $\mathbf y^{\top}\mathbf y> 0$，证毕。

    另外，如果 $rk(A)<n$，即 $A$ 的所有列线性相关，那么存在 $j \in [0,n]$，使得 $A_j=\sum_{k\neq j} \lambda_k A_k$，故存在 $\mathbf x=[\lambda_1,\ldots, \lambda_{j-1}, -1, \lambda_{j+1}, \ldots, \lambda_n]^{\top} \neq \mathbf 0$，使得 $\mathbf y=[A_1,\ldots, A_n]\mathbf x=\mathbf 0$，故 $S$ 非正定。

<font color="red">**Spectral Theorem：**</font>

若 $A \in \mathbb R^{n \times n}$ 对称，则 $A$ 的特征值均为实数，且 $A$ 的特征向量为空间 $V$ 的正交归一基（ONB）。

这表明对称方阵 $A$ 可以做特征分解。

## 3.5 矩阵分解

**Cholesky 分解**

对称正定矩阵 $A$ 可以被唯一地分解为 $A=LL^{\top}$，其中 $L$ 是下三角矩阵，且对角线元素值均为正。


Cholesky 分解可以类比正实数的求平方根。


**可对角化：** 若方阵 $A$ 可写成 $D=P^{-1}AP$，那么 $A$ 是可对角化的。对称方阵 $S$ 总是可对角化（结合 Spectral theorem 和特征分解的定义可证）。

**特征分解**

方阵 $A$ 可被写成 $A=PDP^{-1}$，其中，$D$ 是 $A$ 的特征值组成的对角阵，这个过程称为特征分解。

$A$ 可被特征分解的充要条件：$A$ 的特征向量可构成 $V$ 空间的一组基（注意不能得出 $det(A) \neq 0$ 的结论），这表明，**对称方阵可以特征分解**。

**SVD**

任意矩阵 $A^{m \times n}$ 均可被奇异值分解为

$$A=U \Sigma V^{\top}$$
其中，$U \in \mathbb R^{m \times m}$ 和 $V \in \mathbb R^{n \times n}$ 是正交矩阵（正交归一列向量），$\Sigma$ 是 $m \times n$ 的矩阵，且 $\Sigma_{ii} = \sigma_i \ge 0$，$\Sigma_{ij}=0, \ i\neq j$。

注：正交矩阵 $M$ 满足 $M^{\top}=M^{-1}$

**矩阵近似**

对于矩阵 $A \in \mathbb R^{m \times n}$，构建 rank=1 的 $A_i \in \mathbb R^{m\times n}$ 如下：

$$A_i := \mathbf u_i \mathbf v_i^{\top}$$

其中 $\mathbf u_i, \mathbf v_i$ 分别来自 $U, V$ 中的列向量。根据 SVD 为 $A=U\Sigma V^{\top}$，可知

$$A=\sum_{i=1}^r \sigma_i \mathbf u_i \mathbf v_i^{\top}=\sum_{i=1}^r \sigma_i A_i$$

其中，$rk(A)=r$，$\sigma_i$ 为第 `i` 个奇异值。

如果我们将奇异值排序，并取 top `k` （$k < r$）的这样的 rank=1 的 $A_1,\ldots, A_k$，那么有 $A$ 的近似，

$$\hat A(k)=\sum_{i=1}^k \sigma_i A_i$$

且 $rk(\hat A(k))=k$ 。

**应用**

一幅（灰度）图片（彩色图片可以看作是 r b g 三个灰度图片），height width 为 `1432, 1910`，那么需要存储 $1432 \times 1910=2735120$ 个数，如果使用上述 SVD 的近似，例如取 `k=5`，那么仅需存储 $5$ 个特征值，以及 5 个左奇异向量和 5 个右奇异向量，共 $5(1432+1910+1)=16715$ 个数，存储量大大降低。[参见代码](https://gist.github.com/JianjianSha/9e76411bd4a5570c1363c7c3bcc3900c)

为什么使用 top `k` 最大绝对值的奇异值呢？

**光谱范数**

Spectral Norm

对于 $\mathbf x \neq \mathbf 0$，矩阵 $A \in \mathbb R^{m \times n}$ 的光谱范数定义为：

$$||A||_2:=\max_{\mathbf x} \frac {||A\mathbf x||_2}{||\mathbf x||_2}$$

即，向量经过 $A$ 变换后，长度 scale 比例最大值，这个最大值是 $A$ 的最大奇异值 $\sigma_1$ 。
