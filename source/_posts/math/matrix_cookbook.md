---
title: Matrix Cookbook
date: 2022-05-21 11:33:53
tags: math
mathjax: true
---
# 1 Basics

1.  $A_{n \times n}, \ B_{n \times n} \cdots$ 逆均存在，

    $$(AB\cdots) ^ {-1}=\cdots B^ {-1}A^ {-1} \tag{1}$$

    > 将矩阵看作变换，那么 $(AB)$ 表示对向量先做 B 变换，然后做 A 变换，那么 $(AB)^ {-1}$ 表示做逆变换，此时先做 A 的逆变换，然后再做 B 的逆变换

2.  $A_{m \times n}, \ B_{n \times p} \cdots$，且可以做矩阵乘法，

    $$(AB\cdots)^ {\top}=\cdots B^ {\top}A^ {\top} \tag{2}$$

    证： $(AB)^ {\top}_{ji}=A_{i,:} \ B_{:,j}=B^ {\top}_{j,:} A^ {\top}_{:,i}$
    >

3. $A_{n \times n}$，且 $A^ {-1}$ 存在，
    
    $$(A^ {\top})^ {-1}=(A^ {-1})^ {\top} \tag{3}$$

4. $A_{m \times n}, \ B_{m \times n}$，

    $$(A+B)^ {\top}=A^ {\top}+B^ {\top} \tag{4}$$

## 1.1 迹和行列式

1. $A_{n \times n}$，

    $$Tr(A) \stackrel{\Delta}=\sum_i A_{ii}=\sum_i \lambda_i, \quad \lambda_i=\text{eig}(A)
    \\\\ Tr(A)=Tr(A^ {\top})$$

2. $A_{n \times n}$，
    $$|A|=\prod_i \lambda_i, \quad \lambda_i=\text{eig}(A)$$

3. $A_{n \times n}, \ B_{n \times n}$，
    $$Tr(A+B)=Tr(A)+Tr(B)$$

4. $A_{m \times n}, \ B_{n \times m}$，
    $$Tr(AB)=Tr(BA)$$

    由此可得 $Tr(ABC)=Tr(A(BC))=Tr((BC)A)=Tr(BCA)=Tr(CAB)$
    >

5. $A_{n \times n}, \ B_{n \times n}$，

    $$|A| \cdot |B| = |A B|$$

    根据第 `2` 点可证。由此可得 $|A^ {-1}|=1/|A|$

# 2 导数

1. $\mathbf x \in \mathbb R^ {n \times 1}$，$\mathbf a$ 是行向量，$A$ 是常矩阵，

    $$\frac {\partial (\mathbf a^ {\top} \mathbf x)} {\partial \mathbf x}=\frac {\partial (\mathbf x^ {\top} \mathbf a)}{\partial \mathbf x}=\mathbf a
    \\\\ \frac {\partial (\mathbf x^ {\top} \mathbf x)} {\partial \mathbf x}=2\mathbf x
    \\\\ \frac {\partial (\mathbf x^ {\top} A \mathbf x)}{\partial \mathbf x}=A \mathbf x + A^ {\top} \mathbf x
    \\\\ \frac {\partial (\mathbf a^ {\top} \mathbf x \mathbf x^ {\top} \mathbf b)}{\partial \mathbf x}= \mathbf a \mathbf b^ {\top} \mathbf x + \mathbf b \mathbf a^ {\top}\mathbf x$$

    记忆：
    
    $$\frac {d (ax)}{dx}=a \stackrel{标量转向量}\longrightarrow \frac {\partial (\mathbf a^ {\top} \mathbf x)} {\partial \mathbf x}=\mathbf a$$

    $\mathbf a, \ \mathbf x$ 均为列向量，如果偏导数写为 $\mathbf a^ {\top}$ 那就成了行向量。

    $$\frac {\partial (\mathbf x^ {\top} \mathbf x)} {\partial \mathbf x}$$

    先将 $\mathbf x^ {\top}$ 看作 $\mathbf a^ {\top}$，对后一个 $\mathbf x$ 求偏导，然后将 $\mathbf x^ {\top} \mathbf x$ 转置后，对 $\mathbf x$ 求偏导，就是对原来的 $\mathbf x^ {\top}$ 求偏导。第三个和第四个均类似处理。

## 2.1 矩阵求导

1. $X_{m \times n}$，$c$ 是常数

    $$\frac {\partial c} {\partial X}=\mathbf 0_{m \times n}$$

2. 线性法则

    $$\frac {\partial [c_1 f(X)+c_2g(X)]}{\partial X}=c_1 \frac {\partial f(X)}{\partial X}+c_2 \frac {\partial g(X)}{\partial X}$$

3. 乘法法则

    $$\frac {\partial [f(X)g(X)]}{\partial X}=\frac {\partial f(X)}{\partial X} g(X) + f(X)\frac {\partial g(X)}{\partial X}$$

4. 除法法则

    $$\frac {\partial \frac {f(X)}{g(X)}}{\partial X}=\frac 1 {g^ 2(X)}\left [\frac {\partial f(X)}{\partial X} g(X)-f(X)\frac {\partial g(X)}{\partial X} \right]$$

5. $X_{m \times n}, \ \mathbf a_{m \times 1}, \ \mathbf b_{n \times 1}$，

    $$\frac {\partial (\mathbf a^ {\top} X \mathbf b)}{\partial X}=\mathbf {ab}^ {\top}$$

    证：$\mathbf a^ {\top} X \mathbf b=sum[(\mathbf a \mathbf b^ {\top}) \circ X]$，得证。

    >

6. $X_{m \times n}, \ \mathbf a_{n \times 1}, \ \mathbf b_{m \times 1}$，

    $$\frac {\partial (\mathbf a^ {\top} X^ {\top} \mathbf b)}{\partial X}=\mathbf {ba}^ {\top}$$

    证：$\frac {\partial (\mathbf a^ {\top} X^ {\top} \mathbf b)}{\partial X}=\frac {\partial (\mathbf a^ {\top} X^ {\top} \mathbf b)^ {\top}}{\partial X}=\frac {\partial (\mathbf b^ {\top} X \mathbf a)}{\partial X}=\mathbf {ba}^ {\top}$
    >

7. $X_{m \times n}, \ \mathbf a_{m \times 1}, \ \mathbf b_{m \times 1}$，

    $$\frac {\partial (\mathbf a^ {\top} XX^ {\top} \mathbf b)}{\partial X}=\mathbf {ab}^ {\top} X + \mathbf {ba}^ {\top}X$$

    证：先将 $X^ {\top} \mathbf b$ 看作一个整体，根据第 `5` 点结论，求导为 $\mathbf a(X^ {\top}\mathbf b)^ {\top}=\mathbf {ab}^ {\top} X$，然后将 $\mathbf a^ {\top} X$ 看作一个整体，根据第 `6` 点结论，求导为 $\mathbf b \mathbf a^ {\top} X$，得证。
    >

8. $X_{m \times n}, \ \mathbf a_{n \times 1}, \ \mathbf b_{n \times 1}$，

    $$\frac {\partial (\mathbf a^ {\top} X^ {\top}X \mathbf b)}{\partial X}=X\mathbf {ab}^ {\top}  + X\mathbf {ba}^ {\top}$$

    证：分别将 $\mathbf a^ {\top} X^ {\top}$ 和 $X \mathbf b$ 看作一个整体，然后求导。
    >

9. W, X 均为矩阵，

    $$\frac {\partial tr(W^ {\top}XX^ {\top}W)}{\partial W}=2XX^ {\top}W$$

## 2.2 矩阵微分

1. 矩阵变元的实矩阵函数 $F(X), \ F_{p \times q}, \ X_{m \times n}$，矩阵函数的每一个元素是矩阵变元的实值标量函数 $f_{ij}(X)$，

    $$dF_{p \times q}(X)= \begin{bmatrix} df_{11}(X) & df_{12}(X) & \cdots & df_{1q}(X) \\\\ \vdots  & \vdots & \vdots & \vdots \\\\ df_{p1}(X) & df_{p2}(X) & \cdots & df_{pq}(X) \end{bmatrix}$$

2. 常数矩阵 $A$ 的矩阵微分，

    $$dA_{m \times n} = \mathbf 0_{m \times n}$$
    >
3. 线性法则

    $$d(c_1 F(X)+C_2 G(X)) = c_1 dF(X) + c_2 dG(X)$$
    >
4. 乘法法则，$F_{p \times q}, \ G_{q \times s}$

    $$d[F(X)G(X)]=d[F(X)]G(X)+F(X)d[G(X)]$$
    >

5. 转置法则，$F_{p \times q}$

    $$dF^ {\top}(X)=(dF(X))^ {\top}$$

## 2.3 根据微分求导

$X_{m \times n}$ 就是矩阵变元 $X_{m \times n}$ 的实矩阵函数，每个元素的微分为 $dx_{ij}$，故矩阵微分为

$$dX=\begin{bmatrix} dx_{11} & \cdots & dx_{1n} \\\\ \vdots & \vdots & \vdots \\\\ dx_{m1} & \cdots & dx_{mn} \end{bmatrix}$$

矩阵变元的实标量函数的全微分如下，

$$\begin{aligned} df(X)&=\frac {\partial f} {\partial x_{11}} dx_{11} + \frac  {\partial f} {\partial x_{12}} dx_{12} + \cdots + \frac  {\partial f} {\partial x_{mn}} dx_{mn}
\\\\&= tr \left(\begin{bmatrix}\frac  {\partial f} {\partial x_{11}} & \frac  {\partial f} {\partial x_{21}} & \cdots & \frac  {\partial f} {\partial x_{m1}} \\\\ \vdots & \vdots & \vdots & \vdots \\\\ \frac  {\partial f} {\partial x _ {1n}} & \frac  {\partial f} {\partial x_{2n}} & \cdots & \frac  {\partial f} {\partial x_{mn}}\end{bmatrix} _ {n \times m}  \begin{bmatrix}dx _ {11} & dx _ {12} & \cdots & dx_{1n} \\\\ \vdots & \vdots & \vdots & \vdots \\\\ dx _ {m1} & dx_{m2} & \cdots & dx _ {mn} \end{bmatrix} _ {m \times n} \right)
\\\\ &=tr(\frac {\partial f(X)}{\partial X^ {\top}} dX)
\end{aligned} \tag{2-3-1}$$

其中 $\frac {\partial f(X)}{\partial X}$ 就是实标量函数关于矩阵变元的导数。如果能写出形如(2-3-1) 式的全微分，那么导数就很容易得到了。

下面给出几个微分公式，

1. $A_{p \times m}, \ B_{n \times q}$ 是常数矩阵，矩阵变元 $X_{m \times n}$，

    $$d(AXB)=Ad(X)B$$

    证：$d(AXB)=d(A)XB+Ad(X)B + AXd(B)=Ad(X)B$。

    一般形式，$d(AF(X)B)=Ad(F(X))B$
    >
2. $X_{n \times n}$，且可逆，

    $d|X|=|X|tr(X^ {-1}dX)=tr(|X|X^ {-1}dX)$

    根据 (2-3-1) 式，易知

    $$\frac {\partial |X|}{\partial X}=|X|X^ {-\top}$$

    >
3. $X_{n \times n}$ 可逆，

    $$d(X^ {-1})=-X^ {-1}d(X)X^ {-1}$$

    证：$XX^ {-1}=I \Rightarrow d(XX^ {-1})=\mathbf 0 \Rightarrow d(X)X^ {-1}+Xd(X^ {-1})=\mathbf 0 \Rightarrow d(X^ {-1})=-X^ {-1}d(X)X^ {-1}$

__如何利用矩阵微分求导？__

对于实值标量函数 $f(X)$，有 $tr(f(X))=f(X)$，$df(X)=tr(df(X)$（注 $df(X)$ 是标量）。根据线性法则，

$$d(tr F_{p \times p}(X))=d(\sum_{i=1}^ p f_{ii}(X))=\sum_{i=1}^ p d(f_{ii}(X)) = tr(dF_{p\times p}(X))$$

从上面的例子可以看出，我们这里只考虑实标量函数关于矩阵变元求导，导数也是一个矩阵，而对应的全微分则是一个标量，且全微分中 $d(X)$ 位于最后位置（即 $d(X)$ 之后不再有项）。

常用的矩阵求导公式：

### 2.3.1

$$\frac {\partial \log |X|} {\partial X}=X^ {-\top}$$

证明：$\frac {\partial \log |X|} {\partial X}=\frac 1 {|X|} \frac {\partial |X|}{\partial X}=X^ {-\top}$

### 2.3.2

$$\frac {\partial |X^ {-1}|}{\partial X}=-|X^ {-1}|X^ {-\top}$$

证明： 

$$d|X^ {-1}|=tr(|X^ {-1}|Xd(X^ {-1}))=tr[|X^ {-1}|X(-X^ {-1})d(X)X^ {-1}]=-|X^ {-1}|tr(d(X)X^ {-1})$$

根据 $tr$ 操作的旋转不变性，上式等于 $-|X^ {-1}|tr(X^ {-1}dX)$，对比 (2-3-1) 式，得证。

### 2.3.3

$$\frac {\partial tr(X+A)^ {-1}}{\partial X}=-((X+A)^ {-2})^ {\top}$$

证明：

$d(tr(X+A)^ {-1})=tr(d(X+A)^ {-1})=-tr[(X+A)^ {-1}d(X)(X+A)^ {-1}]$

根据 $tr$ 操作的旋转不变性，上式等于 $-tr[(X+A)^ {-2})d(X)]$，对比 (2-3-1) 式，得证。

### 2.3.4

$$\frac {\partial |X^ 3|}{\partial X}=\frac {\partial |X|^ 3}{\partial X}=3|X|^ 3X^ {-\top}=3|X^ 3|X^ {-\top}$$

证明：

因为 $|A|\cdot|B|=|AB|$，故上式第一个等号和最后一个等号可证。

$$\frac {\partial |X|^ 3}{\partial X}=3|X|^ 2 \frac {\partial |X|}{\partial X}=3|X|^ 3X^ {-\top}$$

证毕。

# ref
1^. The Matrix Cookbook, Kaare Brandt Petersen.
2^. [矩阵求导公式的数学推导（矩阵求导——进阶篇）](https://zhuanlan.zhihu.com/p/288541909)