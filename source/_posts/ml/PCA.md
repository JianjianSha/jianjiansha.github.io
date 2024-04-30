---
title: 主成分分析
date: 2022-02-14 18:31:07
tags: machine learning
mathjax: true
---


主成分分析用于数据降维

<!--more-->

（本文有的大写符号表示矩阵，有的大写符号表示 scalar，可根据上下文辨别。）

# 1. 问题描述

考虑 i.i.d 数据集 $\mathcal X=\{\mathbf x _ 1,\ldots, \mathbf x _ N\}$，$\mathbf x _ n \in \mathbb R ^ D$，且经验期望为 $\hat {\mathbb E}[\mathbf x]=\mathbf 0$，那么经验方差（协方差矩阵）为

$$S=\frac 1 N \sum _ {n=1} ^ N \mathbf x _ n \mathbf x _ n ^ {\top} \tag{1}$$

为了降维，将数据映射到低维 

$$\mathbf z = B ^ {\top} \mathbf x \in \mathbb R ^ M \tag{2}$$

其中 
$$B:=[\mathbf b _ 1,\ldots, \mathbf b _ M] \in \mathbb R ^ {D \times M}$$

且 $B$ 的列相互正交且归一，$M < D$


从 $\mathbf z$ 再恢复到原空间的数据 $\tilde {\mathbf x}$，
$$\begin{aligned}\tilde {\mathbf x}=B ^ {-\top}\mathbf z =B \mathbf z&=BB ^ {\top}\mathbf x
\\\\&=[\mathbf b  _  1,\ldots, \mathbf b  _  M]\begin{bmatrix}\mathbf b  _  1 ^ {\top} \\\\ \vdots \\\\ \mathbf b _ M ^ {\top}\end{bmatrix}\mathbf x
\\\\&=\sum _ {i=1} ^ M \mathbf b  _  i \mathbf b  _  i ^ {\top} \mathbf x \end{aligned}\tag{3}$$

其中 $B$ 是正交矩阵 $\Rightarrow B ^ {\top}B=I$。

那么如何求这样的转换矩阵 $B$ ？显然需要根据某种指标优化来求解，因为满足降维的正交矩阵有无数个。

# 2. 最大化方差

最大化方差来求解 $B$。


上面我们假设数据集是 centered data，即均值为 $\mathbf 0$，这种假设是合理的。因为如果数据均值为 $\boldsymbol \mu$，那么变换为
$$\mathbf z =B ^ {\top}(\mathbf x - \boldsymbol \mu)$$

于是方差为

$$\mathbb V _ {\mathbf z}[\mathbf z]=\mathbb V _ {\mathbf x}[B ^ {\top}(\mathbf x-\boldsymbol \mu)]=\mathbb V _ {\mathbf x}[B ^ {\top}\mathbf x]$$

即，映射到低维空间的数据分布的方差与原数据是否 centered 无关。于是可以假设 $\mathbb E _ {\mathbf x}[\mathbf x]=\mathbf 0$，此时

$$\mathbb E _ {\mathbf z}[\mathbf z]=B ^ {\top} \mathbb E _ {\mathbf x}[\mathbf x]=\mathbf 0$$

根据 (2) 式有

$$z _ 1=\mathbf b _ 1 ^ {\top} \mathbf x$$

考虑 $z _ 1$ 方向（或者称，变换后这个 axis）上的方差，

$$\begin{aligned}V _ 1:=\mathbb V[z _ 1]&=\frac 1 N \sum _ {n=1} ^ N z _ {1n} ^ 2
\\\\&=\frac 1 N \sum _ {n=1} ^ N (\mathbf b _ 1 ^ {\top} \mathbf x _ n) ^ 2
\\\\&=\frac 1 N \sum _ {n=1} ^ N \mathbf b _ 1 ^ {\top} \mathbf x _ n\mathbf x _ n ^ {\top}\mathbf b _ 1
\\\\&=\mathbf b _ 1 ^ {\top} \left(\frac 1 N \sum _ {n=1} ^ N \mathbf x _ n \mathbf x _ n ^ {\top} \right)\mathbf b _ 1
\\\\&=\mathbf b _ 1 ^ {\top} S \mathbf b _ 1
\end{aligned}$$


在求 $\max V _ 1$ 之前，还需要增加限制条件 $||\mathbf b _ 1|| ^ 2=1$，否则，将 $\mathbf b _ 1$ 进行 scale，方差 $V$ 则会二次幂 scale，故得到优化问题

$$\max _ {\mathbf b _ 1} \mathbf b _ 1 ^ {\top} S \mathbf b _ 1
\\\\ s.t. \ ||\mathbf b _ 1|| ^ 2=1$$


对于受限优化问题，可以使用拉格朗日乘子法，
$$\mathcal L(\mathbf b _ 1, \lambda _ 1)=\mathbf b _ 1 ^ {\top} S \mathbf b _ 1 + \lambda _ 1(1-\mathbf b _ 1 ^ {\top} \mathbf b _ 1)$$

求偏导
$$\frac {\partial \mathcal L}{\partial \mathbf b _ 1}=2\mathbf b _ 1 ^ {\top} S-2\lambda _ 1 \mathbf b _ 1 ^ {\top}, \quad \frac {\partial \mathcal L}{\partial \lambda _ 1}=1-\mathbf b _ 1 ^ {\top}\mathbf b _ 1$$

令偏导为 $\mathbf 0$，解得
$$S\mathbf b _ 1=\lambda _ 1 \mathbf b _ 1, \quad \mathbf b _ 1 ^ {\top}\mathbf b _ 1=1$$

即 $\lambda _ 1, \mathbf b _ 1$ 分别是矩阵 $S$ 的特征根和（归一化）特征向量。于是

$$V _ 1=\mathbf b _ 1 ^ {\top} S \mathbf b _ 1=\lambda _ 1 \mathbf b _ 1 ^ {\top} \mathbf b _ 1=\lambda _ 1 \ge 0$$

<font color="magenta">$S$ 是对称矩阵，故是半正定的，其特征值非负。</font>

**故当 $\lambda _ 1$ 是 $S$ 的最大特征值时，方差 $V _ 1$ 最大，此时 $\mathbf b _ 1$ 为最大特征值对应的归一化特征向量。**


映射到低维空间的 $z _ 1$ 分量为

$$z _ 1=\mathbf b _ 1 ^ {\top} \mathbf x$$

根据 $z _ 1$ 恢复到原空间数据

$$\tilde {\mathbf x} ^ {(1)}=\mathbf b _ 1 z _ 1=\mathbf b _ 1 \mathbf b _ 1 ^ {\top} \mathbf x$$

## 2.1 最大方差的 M 维子空间

我们考虑一般情况，假设已经找到 $m-1$ 个主成分对应的特征向量，由于协方差矩阵是对称矩阵，根据矩阵光谱定理，其特征向量可以组成 $m-1$ 维子空间的一组正交归一基，即 
$$\mathbf b _ i ^ {\top} \mathbf b _ j=\begin{cases} 1 & i=j \\\\ 0 & i \ne j \end{cases}, \quad 1 \le i,j \le m-1$$

对数据做如下处理

$$\hat X:=X-\sum _ {i=1} ^ {m-1} \mathbf b _ i \mathbf b _ i ^ {\top} X=X-B _ {m-1}X$$

其中 $B _ {m-1}=\sum _ {i=1} ^ {m-1} \mathbf b _ i \mathbf b _ i ^ {\top}$，$X =[\mathbf x _ 1, \ldots,\mathbf x _ N]$ 表示数据集，$\mathbf b _ i \mathbf b _ i ^ {\top} X$ 表示根据主成分分量恢复到原空间的数据集，故 $\hat X$ 表示去掉原数据中已确定的 $m-1$ 个主成分，这样 $\hat X$ 在这 $m-1$ 子空间的方差为 $\mathbf 0 _ {m-1}$，于是对 $\hat X$ 求第一个主成分就是对 $X$ 求第 $m$ 个主成分，

$$V _ m =\mathbb V[z _ m]=\frac 1 N \sum _ {n=1} ^ N z _ {mn} ^ 2=\frac 1 N \sum _ {n=1} ^ N (\mathbf b _ m ^ {\top} \hat {\mathbf x} _ n) ^ 2=\mathbf b _ m ^ {\top}\hat S \mathbf b _ m$$

满足 $||\mathbf b _ m|| ^ 2=1$。

即 **$\hat S$ 的第一个特征值和对应特征向量就是 $S$ 的第 $m$ 个特征值和特征向量（特征值降序排列）**。

于是，

$$V _ m=\mathbf b _ m ^ {\top}S \mathbf b _ m=\lambda _ m$$



事实上有：
1. $\hat S$ 的特征向量与 $S$ 的特征向量相同
2. $\hat S$ 的后 $m-1$ 特征值均为 0，其特征向量对应 $S$ 前 $m-1$ 个特征向量
3. $\hat S$ 的前 $D-(m-1)$ 特征值和特征向量对应 $S$ 后 $D-(m-1)$ 个特征值和特征向量。

**<font color="magenta">结论</font>**
转换矩阵 $B \in \mathbb R ^ {D \times M}$ 由数据协方差矩阵 $S$ 的前 $m$ 个特征向量（列向量）组成。

**数据恢复**

使用前 $m$ 个主成分进行数据恢复，根据 (3) 式，
$$\tilde {\mathbf x}=\sum _ {i=1} ^ m \mathbf b _ i \mathbf b _ i ^ {\top}\mathbf x=[\mathbf b _ 1,\ldots,\mathbf b _ m]\begin{bmatrix}\mathbf b _ 1 ^ {\top} \\\\ \vdots \\\\ \mathbf b _ m ^ {\top}\end{bmatrix}\mathbf x=B _ {:,1:m}B _ {:,1:m} ^ {\top}\mathbf x$$


## 2.2 映射

最大化方差是尽可能的保证数据分布的信息，而映射则是根据最小化原数据与恢复数据之间的误差进行求解。

选取空间 $\mathbb R ^ D$ 的一组正交归一基 （ONB）$B=(\mathbf b _ 1, \ldots, \mathbf b _ D)$，那么任意数据点

$$\mathbf x=\sum _ {d=1} ^ D \zeta _ d \mathbf b _ d=\sum _ {m=1} ^ M \zeta _ m \mathbf b _ m+\sum _ {j=M+1} ^ D \zeta _ j \mathbf b _ j$$

注意 $\zeta _ j, \ j=1,\ldots, D$ 表示数据相对于 $B$ 的坐标，而 $x _ j, \ j=1,\ldots, D$ 是数据在标准基 $E=[\mathbf e _ 1, \ldots, \mathbf e _ D]$（one-hot 向量组）下的坐标。

我们目的是想将数据维度从 $D$ 降到 $M$，这里 $M$ 是超参数，事先设定好的一个固定值。使用前 $M$ 个坐标来表示数据（的一个近似）：

$$\tilde {\mathbf x}=\sum _ {m=1} ^ M z _ m \mathbf b _ m \tag{4}$$

事实上，由于 $\{\mathbf b _ m| m=1,\ldots, D|$ 是正交归一基，故 $z _ m=\zeta _ m, \ m=1,\ldots, M$，注意 $m$ 的取值范围。下文统一使用 $z _ m$ 表示 $\mathbf x$ 在 $B$ 下的坐标。

求解的依据是对整个数据集而言，恢复的数据近似 $\tilde {\mathbf x}$ 与原数据 $\mathbf x$ 误差最小，误差（loss）使用欧氏距离，于是总误差为

$$L=\frac 1 N \sum _ {n=1} ^ N ||\tilde {\mathbf x} _ n-\mathbf x _ n|| ^ 2 \tag{5}$$

其中 $N$ 是数据集大小。



**总结**

要求解两个部分：
1. 正交归一基组 $\{\mathbf b _ m, \ m=1,\ldots, M\}$
2. $\mathbf x$ 在基组 $\{\mathbf b _ m, \ m=1,\ldots, M\}$ 下的坐标

以使得 $L=\frac 1 N \sum _ {n=1} ^ N ||\tilde {\mathbf x} _ n-\mathbf x _ n|| ^ 2$ 最小。




### 2.2.1 求解坐标


根据 (4),(5) 式，误差对坐标求偏导，
$$\begin{aligned}\frac {\partial L}{\partial z _ {mn}}&=\frac {\partial L}{\partial \tilde {\mathbf x} _ n} \frac {\partial \tilde {\mathbf x} _ n}{\partial z _ {mn}} 
\\\\&=\frac 2 N(\tilde {\mathbf x} _ n-\mathbf x _ n) ^ {\top} \mathbf b _ m
\\\\&=\frac 2 N \left(\sum _ {i=1} ^ M z _ {in} \mathbf b _ i-\mathbf x _ n\right) ^ {\top} \mathbf b _ m
\\\\& \stackrel{ONB}= \frac 2 N (z _ {mn} \mathbf b _ m ^ {\top} \mathbf b _ m - \mathbf x _ n ^ {\top} \mathbf b _ m)
\\\\&=\frac 2 N (z _ {mn} - \mathbf x _ n ^ {\top} \mathbf b _ m)
\\\\&= 0
\end{aligned}$$

于是 

$$z _ {mn}=\mathbf x _ n ^ {\top} \mathbf b _ m=\mathbf b _ m ^ {\top}\mathbf x _ n \tag{6}$$

其中 $m=1,\ldots, M; \ n=1,\ldots,N$

这表明 $\mathbf x$ 在 $\mathbf b _ m$ 上的投影值就是对应的坐标 $z _ m$，于是数据恢复为

$$\tilde {\mathbf x}=\sum _ {m=1} ^ M  \mathbf b _ m z _ m=\sum _ {m=1} ^ M \mathbf b _ m \mathbf b _ m ^ {\top} \mathbf x=B _ m B _ m ^ {\top} \mathbf x \tag{7}$$
其中 $B _ m:=[\mathbf b _ 1, \ldots, \mathbf b _ M] \in \mathbb R ^ {D \times M}$，表示主子空间的变换矩阵。(7) 式表示将数据 $\mathbf x$ 投影到主子空间，所得的投影 $\tilde {\mathbf x}$ 与原数据 $\mathbf x$ 的距离需要尽可能小。









### 2.2.2 求解基向量

根据 (6)(7) 式， 

$$\tilde {\mathbf x}=\sum _ {m=1} ^ M z _ m \mathbf b _ m =\sum _ {m=1} ^ M (\mathbf x ^ {\top} \mathbf b _ m)\mathbf b _ m$$

注意右侧小括号不能省略，表示一个 scalar 与矩阵相乘。

当 $M=D$ 时，$\tilde {\mathbf x}=\mathbf x$，所以，$\mathbf x=\sum _ {m=1} ^ D (\mathbf x ^ {\top} \mathbf b _ m)\mathbf b _ m$，于是向量差

$$\mathbf x - \tilde {\mathbf x}=\sum _ {j=M+1} ^ D (\mathbf x ^ {\top} \mathbf b _ j)\mathbf b _ j$$

代入 (5) 式得损失为

$$\begin{aligned}L &=\frac 1 N \sum _ {n=1} ^ N \begin{Vmatrix}\sum _ {j=M+1} ^ D (\mathbf x _ n ^ {\top} \mathbf b _ j)\mathbf b _ j \end{Vmatrix} ^ 2
\\\\&=\frac 1 N \sum _ {n=1} ^ N \left(\sum _ {j=M+1} ^ D (\mathbf x _ n ^ {\top} \mathbf b _ j)\mathbf b _ j\right) ^ {\top} \left(\sum _ {j=M+1} ^ D (\mathbf x _ n ^ {\top} \mathbf b _ j)\mathbf b _ j\right)
\\\\&\stackrel{ONB}=\frac 1 N \sum _ {n=1} ^ N \sum _ {j=M+1} ^ D (\mathbf x _ n ^ {\top} \mathbf b _ j) ^ 2\mathbf b _ j ^ {\top} \mathbf b _ j
\\\\&=\frac 1 N \sum _ {n=1} ^ N \sum _ {j=M+1} ^ D (\mathbf x _ n ^ {\top} \mathbf b _ j) ^ {\top}(\mathbf x _ n ^ {\top} \mathbf b _ j)
\\\\&=\frac 1 N \sum _ {n=1} ^ N \sum _ {j=M+1} ^ D \mathbf b _ j ^ {\top}\mathbf x _ n\mathbf x _ n ^ {\top} \mathbf b _ j
\\\\&=\sum _ {j=M+1} ^ D \mathbf b _ j ^ {\top} \underbrace{\left(\frac 1 N \sum _ {n=1} ^ N \mathbf x _ n\mathbf x _ n ^ {\top} \right)} _ {:=S}\mathbf b _ j
\\\\&= \sum _ {j=M+1} ^ D \mathbf b _ j ^ {\top} S \mathbf b _ j
\end{aligned}$$

由于 $S$ 是对阵半正定，故 $\mathbf b _ j ^ {\top} S \mathbf b _ j\ge 0$，且
$\mathbf b _ j ^ {\top} S \mathbf b _ j$ 最小为 $S$ 的最小特征根，此时 $\mathbf b _ j$ 就是对应的特征向量。下面简单证明。

**证明：**

做特征分解 $S=P ^ {\top} \Lambda P$，其中 $P$ 是正交矩阵，$P ^ {\top}$ 的列为 $S$ 的特征向量。要求下式的最小值

$$\mathbf b ^ {\top} S \mathbf b=\mathbf b ^ {\top} P ^ {\top} \Lambda P\mathbf b \stackrel{\mathbf c:=P\mathbf b}=\mathbf c ^ {\top} \Lambda \mathbf c=\sum _ {i=1} ^ D \lambda _ i c _ i ^ 2$$

由于 $\mathbf c ^ {\top}\mathbf c=\mathbf b ^ {\top}P ^ {\top}P\mathbf b=1$，特征值 $\lambda _ i \ge 0 , \ i=1,\ldots, D$ 且降序排列，那么最值问题

$$\min _ {\mathbf c} \ \sum _ {i=1} ^ D \lambda _ i c _ i ^ 2$$

的解为 $\mathbf c=[0, \ldots, 0, 1] ^ {\top}$，最小值为 
$$\min \mathbf b ^ {\top} S \mathbf b=\lambda _ D$$

此时 $\mathbf b=P ^ {\top} \mathbf c=P _ {: , D} ^ {\top}$，此即 $S$ 对应于 $\lambda _ D$ 的一个特征向量（$P ^ {\top}$ 的最后一列）。<p align="right">$\square$</p>

故损失 $L$ 最小为

$$L=\sum _ {j=M+1} ^ D \lambda _ j$$

其中 $\lambda _ j$ 可重复（可重复是指特征值几何重数 $n>1$）。

<font color="magenta">结论：</font>

矩阵 $B$ 由协方差矩阵 $S$ 的特征向量（列向量）组成，**变换矩阵 $B _ M$ 由其 top `M` 特征值（可重复）对应的特征向量组成**。

可见，这与最大化方差的结果是相同的。

## 2.3 why centered data

我们假设数据集经验期望为 $\boldsymbol \mu=\mathbf 0$，这样方便下文的讨论，如果不为 $\mathbf 0$，那么最终的求解结果（指转换矩阵）仍然不变，只是求解过程的展示更繁冗些。事实上，当经验期望为 $\boldsymbol \mu$ 时，做变换

$$\mathbf z=B _ m ^ {\top} (\mathbf x-\boldsymbol \mu) \in \mathbb R ^ M$$
那么 $\tilde {\mathbf x}=B _ m \mathbf z + \boldsymbol \mu$，损失对坐标 $z _ {mn}$ 的偏导为

$$\begin{aligned} \frac {\partial L}{\partial z _ {mn}}&=\frac 2 N(\tilde {\mathbf x} _ n-\mathbf x _ n) ^ {\top} \mathbf b _ m
\\\\&=\frac 2 N (z _ {mn} + \boldsymbol \mu ^ {\top} \mathbf b _ m- \mathbf x _ n ^ {\top} \mathbf b _ m)=0
\end{aligned}\Rightarrow z _ {mn}=(\mathbf x _ n-\boldsymbol \mu) ^ {\top} \mathbf b _ m$$

可见基向量 $\mathbf b _ m$ 是不变的，于是对数据 $\mathbf x$ 转换的矩阵 $B _ m$ 也是不变的，改变的只是相对于这些基向量的坐标值（平移 $-\boldsymbol \mu ^ {\top} \mathbf b _ m$）。

**non-centered data** 处理步骤：
1. 将数据进行 centering 处理 $\mathbf x:=\mathbf x - \boldsymbol \mu$，其中 $\boldsymbol \mu= \frac 1 N \sum _ n \mathbf x _ n$
2. 将数据按 centered data 处理：求协方差矩阵的 top `m` 对应的特征向量，得到转换矩阵 $B _ m$
3. 恢复数据 $\tilde {\mathbf x}=B _ m \mathbf z + \boldsymbol \mu=B _ mB _ m ^ {\top} \mathbf x + \boldsymbol \mu$

# 3. 低 rank 近似

## 3.1 特征向量求解
数据集的协方差矩阵

$$S=\frac 1 N \sum _ {n=1} ^ N \mathbf x _ n \mathbf x _ n ^ {\top}=\frac XX ^ {\top}$$

其中 $X=[\mathbf x _ 1, \ldots, \mathbf x _ N] \in \mathbb R ^ {D \times N}$ 表示整个数据集。

获取协方差矩阵的特征向量的方法：
1. 直接对 $S$ 进行特征分解
2. 对 $X$ 进行 SVD，

    $$X=U \Sigma V ^ {\top}$$
    
    $$S=\frac 1 N XX ^ {\top}=U (\frac 1 N \Sigma \Sigma ^ {\top}) U ^ {\top}$$

    $U$ 的列就是特征向量。


## 3.2 数据集低 rank 近似

数据集 $X \in \mathbb R ^ {D \times N}$ 的近似 

$$\tilde X _ M=\underbrace{U _ M} _ {D \times M} \underbrace{\Sigma _ M} _ {M\times M} \underbrace{V _ M ^ {\top}} _ {M \times N} \in \mathbb R ^ {D \times N} \tag{8}$$

其中：
1. $U _ M=[\mathbf u _ 1,\ldots, \mathbf u _ M]$ 为前 $M$ 个左特征向量
2. $\Sigma _ M$ 为前 $M$ 个特征值所构成的对角矩阵
3. $V _ M=[\mathbf v _ 1,\ldots, \mathbf v _ M]$ 为前 $M$ 个右特征向量

    注意，代入 (8) 式中时，$V _ M$ 需要转置为 $V _ M ^ {\top}$，而 `numpy.linalg.svd` 返回的三元组 `u,d,v` 中 `v` 已经是转置过的，所以直接取 `v[:M,:]`，根据 (8) 式恢复的数据集近似则为 
    ```python
    np.dot(np.dot(u[:,:M],np.diag(d[:M])), v[:M,:])
    ```

这跟这里的[图像压缩存储（图像近似）](/2022/02/10/math/linear_algebra_concepts)本质是一样的,只要存储 $DM+M+MN$ 个数，而原数据集需要存储 $DN$ 个数，从而达到压缩目的。