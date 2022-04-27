---
title: 主成分分析
date: 2022-02-14 18:31:07
tags: machine learning
mathjax: true
---


主成分分析用于数据降维

<!--more-->

（本文有的大小符号表示矩阵，有的大写符号表示 scalar，可根据上下文辨别。）

# 1. 问题描述

考虑 i.i.d 数据集 $\mathcal X=\{\mathbf x_1,\ldots, \mathbf x_N\}$，$\mathbf x_n \in \mathbb R^D$，且经验期望为 $\hat {\mathbb E}[\mathbf x]=\mathbf 0$，那么经验方差（协方差矩阵）为

$$S=\frac 1 N \sum_{n=1}^N \mathbf x_n \mathbf x_n^{\top} \tag{1}$$

为了降维，将数据映射到低维 

$$\mathbf z = B^{\top} \mathbf x \in \mathbb R^M \tag{2}$$

其中 
$$B:=[\mathbf b_1,\ldots, \mathbf b_M] \in \mathbb R^{D \times M}$$

且 $B$ 的列相互正交且归一，$M < D$


从 $\mathbf z$ 再恢复到原空间的数据 $\tilde {\mathbf x}$，
$$\begin{aligned}\tilde {\mathbf x}=B^{-\top}\mathbf z =B \mathbf z&=BB^{\top}\mathbf x
\\&=[\mathbf b_1,\ldots, \mathbf b_M]\begin{bmatrix}\mathbf b_1^{\top} \\ \vdots \\ \mathbf b_M^{\top}\end{bmatrix}\mathbf x
\\&=\sum_{i=1}^M \mathbf b_i \mathbf b_i^{\top} \mathbf x \end{aligned}\tag{3}$$

其中 $B$ 是正交矩阵 $\Rightarrow B^{\top}B=I$。

那么如何求这样的转换矩阵 $B$？显然需要根据某种指标优化来求解，因为满足降维的正交矩阵有无数个。

# 2. 最大化方差

最大化方差来求解 $B$。


上面我们假设数据集是 centered data，即均值为 $\mathbf 0$，这种假设是合理的。因为如果数据均值为 $\boldsymbol \mu$，那么变换为
$$\mathbf z =B^{\top}(\mathbf x - \boldsymbol \mu)$$

于是方差为

$$\mathbb V_{\mathbf z}[\mathbf z]=\mathbb V_{\mathbf x}[B^{\top}(\mathbf x-\boldsymbol \mu)]=\mathbb V_{\mathbf x}[B^{\top}\mathbf x]$$

即，映射到低维空间的数据分布的方差与原数据是否 centered 无关。于是可以假设 $\mathbb E_{\mathbf x}[\mathbf x]=\mathbf 0$，此时

$$\mathbb E_{\mathbf z}[\mathbf z]=B^{\top} \mathbb E_{\mathbf x}[\mathbf x]=\mathbf 0$$

根据 (2) 式有

$$z_1=\mathbf b_1^{\top} \mathbf x$$

考虑 $z_1$ 方向（或者称，变换后这个 axis）上的方差，

$$\begin{aligned}V_1:=\mathbb V[z_1]&=\frac 1 N \sum_{n=1}^N z_{1n}^2
\\&=\frac 1 N \sum_{n=1}^N (\mathbf b_1^{\top} \mathbf x_n)^2
\\&=\frac 1 N \sum_{n=1}^N \mathbf b_1^{\top} \mathbf x_n\mathbf x_n^{\top}\mathbf b_1
\\&=\mathbf b_1^{\top} \left(\frac 1 N \sum_{n=1}^N \mathbf x_n \mathbf x_n^{\top} \right)\mathbf b_1
\\&=\mathbf b_1^{\top} S \mathbf b_1
\end{aligned}$$


在求 $\max V_1$ 之前，还需要增加限制条件 $||\mathbf b_1||^2=1$，否则，将 $\mathbf b_1$ 进行 scale，方差 $V$ 则会二次幂 scale，故得到优化问题

$$\max_{\mathbf b_1} \mathbf b_1^{\top} S \mathbf b_1
\\ s.t. \ ||\mathbf b_1||^2=1$$


对于受限优化问题，可以使用拉格朗日乘子法，
$$\mathcal L(\mathbf b_1, \lambda_1)=\mathbf b_1^{\top} S \mathbf b_1 + \lambda_1(1-\mathbf b_1^{\top} \mathbf b_1)$$

求偏导
$$\frac {\partial \mathcal L}{\partial \mathbf b_1}=2\mathbf b_1^{\top} S-2\lambda_1 \mathbf b_1^{\top}, \quad \frac {\partial \mathcal L}{\partial \lambda_1}=1-\mathbf b_1^{\top}\mathbf b_1$$

令偏导为 $\mathbf 0$，解得
$$S\mathbf b_1=\lambda_1 \mathbf b_1, \quad \mathbf b_1^{\top}\mathbf b_1=1$$

即 $\lambda_1, \mathbf b_1$ 分别是矩阵 $S$ 的特征根和（归一化）特征向量。于是

$$V_1=\mathbf b_1^{\top} S \mathbf b_1=\lambda_1 \mathbf b_1^{\top} \mathbf b_1=\lambda_1 \ge 0$$

<font color="magenta">$S$ 是对称矩阵，故是半正定的，其特征值非负。</font>

**故当 $\lambda_1$ 是 $S$ 的最大特征值时，方差 $V_1$ 最大，此时 $\mathbf b_1$ 为最大特征值对应的归一化特征向量。**


映射到低维空间的 $z_1$ 分量为

$$z_1=\mathbf b_1^{\top} \mathbf x$$

根据 $z_1$ 恢复到原空间数据

$$\tilde {\mathbf x}^{(1)}=\mathbf b_1 z_1=\mathbf b_1 \mathbf b_1^{\top} \mathbf x$$

## 2.1 最大方差的 M 维子空间

我们考虑一般情况，假设已经找到 $m-1$ 个主成分对应的特征向量，由于协方差矩阵是对称矩阵，根据矩阵光谱定理，其特征向量可以组成 $m-1$ 维子空间的一组正交归一基，即 
$$\mathbf b_i^{\top} \mathbf b_j=\begin{cases} 1 & i=j \\ 0 & i \ne j \end{cases}, \quad 1 \le i,j \le m-1$$

对数据做如下处理

$$\hat X:=X-\sum_{i=1}^{m-1} \mathbf b_i \mathbf b_i^{\top} X=X-B_{m-1}X$$

其中 $B_{m-1}=\sum_{i=1}^{m-1} \mathbf b_i \mathbf b_i^{\top}$，$X =[\mathbf x_1, \ldots,\mathbf x_N]$ 表示数据集，$\mathbf b_i \mathbf b_i^{\top} X$ 表示根据主成分分量恢复到原空间的数据集，故 $\hat X$ 表示去掉原数据中已确定的 $m-1$ 个主成分，这样 $\hat X$ 在这 $m-1$ 子空间的方差为 $\mathbf 0_{m-1}$，于是对 $\hat X$ 求第一个主成分就是对 $X$ 求第 $m$ 个主成分，

$$V_m =\mathbb V[z_m]=\frac 1 N \sum_{n=1}^N z_{mn}^2=\frac 1 N \sum_{n=1}^N (\mathbf b_m^{\top} \hat {\mathbf x}_n)^2=\mathbf b_m^{\top}\hat S \mathbf b_m$$

满足 $||\mathbf b_m||^2=1$。

即 **$\hat S$ 的第一个特征值和对应特征向量就是 $S$ 的第 $m$ 个特征值和特征向量（特征值降序排列）**。

于是，

$$V_m=\mathbf b_m^{\top}S \mathbf b_m=\lambda_m$$



事实上有：
1. $\hat S$ 的特征向量与 $S$ 的特征向量相同
2. $\hat S$ 的后 $m-1$ 特征值均为 0，其特征向量对应 $S$ 前 $m-1$ 个特征向量
3. $\hat S$ 的前 $D-(m-1)$ 特征值和特征向量对应 $S$ 后 $D-(m-1)$ 个特征值和特征向量。

**<font color="magenta">结论</font>**
转换矩阵 $B \in \mathbb R^{D \times M}$ 由数据协方差矩阵 $S$ 的前 $m$ 个特征向量（列向量）组成。

**数据恢复**

使用前 $m$ 个主成分进行数据恢复，根据 (3) 式，
$$\tilde {\mathbf x}=\sum_{i=1}^m \mathbf b_i \mathbf b_i^{\top}\mathbf x=[\mathbf b_1,\ldots,\mathbf b_m]\begin{bmatrix}\mathbf b_1^{\top} \\ \vdots \\ \mathbf b_m^{\top}\end{bmatrix}\mathbf x=B_{:,1:m}B_{:,1:m}^{\top}\mathbf x$$


## 2.2 映射

最大化方差是尽可能的保证数据分布的信息，而映射则是根据最小化原数据与恢复数据之间的误差进行求解。

选取空间 $\mathbb R^D$ 的一组正交归一基 （ONB）$B=(\mathbf b_1, \ldots, \mathbf b_D)$，那么任意数据点

$$\mathbf x=\sum_{d=1}^D \zeta_d \mathbf b_d=\sum_{m=1}^M \zeta_m \mathbf b_m+\sum_{j=M+1}^D \zeta_j \mathbf b_j$$

注意 $\zeta_j, \ j=1,\ldots, D$ 表示数据相对于 $B$ 的坐标，而 $x_j, \ j=1,\ldots, D$ 是数据在标准基 $E=[\mathbf e_1, \ldots, \mathbf e_D]$（one-hot 向量组）下的坐标。

我们目的是想将数据维度从 $D$ 降到 $M$，这里 $M$ 是超参数，事先设定好的一个固定值。使用前 $M$ 个坐标来表示数据（的一个近似）：

$$\tilde {\mathbf x}=\sum_{m=1}^M z_m \mathbf b_m \tag{4}$$

事实上，由于 $\{\mathbf b_m| m=1,\ldots, D|$ 是正交归一基，故 $z_m=\zeta_m, \ m=1,\ldots, M$，注意 $m$ 的取值范围。下文统一使用 $z_m$ 表示 $\mathbf x$ 在 $B$ 下的坐标。

求解的依据是对整个数据集而言，恢复的数据近似 $\tilde {\mathbf x}$ 与原数据 $\mathbf x$ 误差最小，误差（loss）使用欧氏距离，于是总误差为

$$L=\frac 1 N \sum_{n=1}^N ||\tilde {\mathbf x}_n-\mathbf x_n||^2 \tag{5}$$

其中 $N$ 是数据集大小。



**总结**

要求解两个部分：
1. 正交归一基组 $\{\mathbf b_m, \ m=1,\ldots, M\}$
2. $\mathbf x$ 在基组 $\{\mathbf b_m, \ m=1,\ldots, M\}$ 下的坐标

以使得 $L=\frac 1 N \sum_{n=1}^N ||\tilde {\mathbf x}_n-\mathbf x_n||^2$ 最小。




### 2.2.1 求解坐标


根据 (4),(5) 式，误差对坐标求偏导，
$$\begin{aligned}\frac {\partial L}{\partial z_{mn}}&=\frac {\partial L}{\partial \tilde {\mathbf x}_n} \frac {\partial \tilde {\mathbf x}_n}{\partial z_{mn}} 
\\&=\frac 2 N(\tilde {\mathbf x}_n-\mathbf x_n)^{\top} \mathbf b_m
\\&=\frac 2 N \left(\sum_{i=1}^M z_{in} \mathbf b_i-\mathbf x_n\right)^{\top} \mathbf b_m
\\& \stackrel{ONB}= \frac 2 N (z_{mn} \mathbf b_m^{\top} \mathbf b_m - \mathbf x_n^{\top} \mathbf b_m)
\\&=\frac 2 N (z_{mn} - \mathbf x_n^{\top} \mathbf b_m)
\\&= 0
\end{aligned}$$

于是 

$$z_{mn}=\mathbf x_n^{\top} \mathbf b_m=\mathbf b_m^{\top}\mathbf x_n \tag{6}$$

其中 $m=1,\ldots, M; \ n=1,\ldots,N$

这表明 $\mathbf x$ 在 $\mathbf b_m$ 上的投影值就是对应的坐标 $z_m$，于是数据恢复为

$$\tilde {\mathbf x}=\sum_{m=1}^M  \mathbf b_m z_m=\sum_{m=1}^M \mathbf b_m \mathbf b_m^{\top} \mathbf x=B_m B_m^{\top} \mathbf x \tag{7}$$
其中 $B_m:=[\mathbf b_1, \ldots, \mathbf b_M] \in \mathbb R^{D \times M}$，表示主子空间的变换矩阵。(7) 式表示将数据 $\mathbf x$ 投影到主子空间，所得的投影 $\tilde {\mathbf x}$ 与原数据 $\mathbf x$ 的距离需要尽可能小。









### 2.2.2 求解基向量

根据 (6)(7) 式， 

$$\tilde {\mathbf x}=\sum_{m=1}^M z_m \mathbf b_m =\sum_{m=1}^M (\mathbf x^{\top} \mathbf b_m)\mathbf b_m$$

注意右侧小括号不能省略，表示一个 scalar 与矩阵相乘。

当 $M=D$ 时，$\tilde {\mathbf x}=\mathbf x$，所以，$\mathbf x=\sum_{m=1}^D (\mathbf x^{\top} \mathbf b_m)\mathbf b_m$，于是向量差

$$\mathbf x - \tilde {\mathbf x}=\sum_{j=M+1}^D (\mathbf x^{\top} \mathbf b_j)\mathbf b_j$$

代入 (5) 式得损失为

$$\begin{aligned}L &=\frac 1 N \sum_{n=1}^N \begin{Vmatrix}\sum_{j=M+1}^D (\mathbf x_n^{\top} \mathbf b_j)\mathbf b_j \end{Vmatrix}^2
\\&=\frac 1 N \sum_{n=1}^N \left(\sum_{j=M+1}^D (\mathbf x_n^{\top} \mathbf b_j)\mathbf b_j\right)^{\top} \left(\sum_{j=M+1}^D (\mathbf x_n^{\top} \mathbf b_j)\mathbf b_j\right)
\\&\stackrel{ONB}=\frac 1 N \sum_{n=1}^N \sum_{j=M+1}^D (\mathbf x_n^{\top} \mathbf b_j)^2\mathbf b_j^{\top} \mathbf b_j
\\&=\frac 1 N \sum_{n=1}^N \sum_{j=M+1}^D (\mathbf x_n^{\top} \mathbf b_j)^{\top}(\mathbf x_n^{\top} \mathbf b_j)
\\&=\frac 1 N \sum_{n=1}^N \sum_{j=M+1}^D \mathbf b_j^{\top}\mathbf x_n\mathbf x_n^{\top} \mathbf b_j
\\&=\sum_{j=M+1}^D \mathbf b_j^{\top} \underbrace{\left(\frac 1 N \sum_{n=1}^N \mathbf x_n\mathbf x_n^{\top} \right)}_{:=S}\mathbf b_j
\\&= \sum_{j=M+1}^D \mathbf b_j^{\top} S \mathbf b_j
\end{aligned}$$

由于 $S$ 是对阵半正定，故 $\mathbf b_j^{\top} S \mathbf b_j\ge 0$，且
$\mathbf b_j^{\top} S \mathbf b_j$ 最小为 $S$ 的最小特征根，此时 $\mathbf b_j$ 就是对应的特征向量。下面简单证明。

**证明：**

做特征分解 $S=P^{\top} \Lambda P$，其中 $P$ 是正交矩阵，$P^{\top}$ 的列为 $S$ 的特征向量。要求下式的最小值

$$\mathbf b^{\top} S \mathbf b=\mathbf b^{\top} P^{\top} \Lambda P\mathbf b \stackrel{\mathbf c:=P\mathbf b}=\mathbf c^{\top} \Lambda \mathbf c=\sum_{i=1}^D \lambda_i c_i^2$$

由于 $\mathbf c^{\top}\mathbf c=\mathbf b^{\top}P^{\top}P\mathbf b=1$，特征值 $\lambda_i \ge 0 , \ i=1,\ldots, D$ 且降序排列，那么最值问题

$$\min_{\mathbf c} \ \sum_{i=1}^D \lambda_i c_i^2$$

的解为 $\mathbf c=[0, \ldots, 0, 1]^{\top}$，最小值为 
$$\min \mathbf b^{\top} S \mathbf b=\lambda_D$$

此时 $\mathbf b=P^{\top} \mathbf c=P_{:,D}^{\top}$，此即 $S$ 对应于 $\lambda_D$ 的一个特征向量（$P^{\top}$ 的最后一列）。<p align="right">$\square$</p>

故损失 $L$ 最小为

$$L=\sum_{j=M+1}^D \lambda_j$$

其中 $\lambda_j$ 可重复（可重复是指特征值几何重数 $n>1$）。

<font color="magenta">结论：</font>

矩阵 $B$ 由协方差矩阵 $S$ 的特征向量（列向量）组成，**变换矩阵 $B_M$ 由其 top `M` 特征值（可重复）对应的特征向量组成**。

可见，这与最大化方差的结果是相同的。

## 2.3 why centered data

我们假设数据集经验期望为 $\boldsymbol \mu=\mathbf 0$，这样方便下文的讨论，如果不为 $\mathbf 0$，那么最终的求解结果（指转换矩阵）仍然不变，只是求解过程的展示更繁冗些。事实上，当经验期望为 $\boldsymbol \mu$ 时，做变换

$$\mathbf z=B_m^{\top} (\mathbf x-\boldsymbol \mu) \in \mathbb R^M$$
那么 $\tilde {\mathbf x}=B_m \mathbf z + \boldsymbol \mu$，损失对坐标 $z_{mn}$ 的偏导为

$$\begin{aligned} \frac {\partial L}{\partial z_{mn}}&=\frac 2 N(\tilde {\mathbf x}_n-\mathbf x_n)^{\top} \mathbf b_m
\\&=\frac 2 N (z_{mn} + \boldsymbol \mu^{\top} \mathbf b_m- \mathbf x_n^{\top} \mathbf b_m)=0
\end{aligned}\Rightarrow z_{mn}=(\mathbf x_n-\boldsymbol \mu)^{\top} \mathbf b_m$$

可见基向量 $\mathbf b_m$ 是不变的，于是对数据 $\mathbf x$ 转换的矩阵 $B_m$ 也是不变的，改变的只是相对于这些基向量的坐标值（平移 $-\boldsymbol \mu^{\top} \mathbf b_m$）。

**non-centered data** 处理步骤：
1. 将数据进行 centering 处理 $\mathbf x:=\mathbf x - \boldsymbol \mu$，其中 $\boldsymbol \mu= \frac 1 N \sum_n \mathbf x_n$
2. 将数据按 centered data 处理：求协方差矩阵的 top `m` 对应的特征向量，得到转换矩阵 $B_m$
3. 恢复数据 $\tilde {\mathbf x}=B_m \mathbf z + \boldsymbol \mu=B_mB_m^{\top} \mathbf x + \boldsymbol \mu$

# 3. 低 rank 近似

## 3.1 特征向量求解
数据集的协方差矩阵

$$S=\frac 1 N \sum_{n=1}^N \mathbf x_n \mathbf x_n^{\top}=\frac XX^{\top}$$

其中 $X=[\mathbf x_1, \ldots, \mathbf x_N] \in \mathbb R^{D \times N}$ 表示整个数据集。

获取协方差矩阵的特征向量的方法：
1. 直接对 $S$ 进行特征分解
2. 对 $X$ 进行 SVD，

    $$X=U \Sigma V^{\top}$$
    
    $$S=\frac 1 N XX^{\top}=U (\frac 1 N \Sigma \Sigma^{\top}) U^{\top}$$

    $U$ 的列就是特征向量。


## 3.2 数据集低 rank 近似

数据集 $X \in \mathbb R^{D \times N}$ 的近似 

$$\tilde X_M=\underbrace{U_M}_{D \times M} \underbrace{\Sigma_M}_{M\times M} \underbrace{V_M^{\top}}_{M \times N} \in \mathbb R^{D \times N} \tag{8}$$

其中：
1. $U_M=[\mathbf u_1,\ldots, \mathbf u_M]$ 为前 $M$ 个左特征向量
2. $\Sigma_M$ 为前 $M$ 个特征值所构成的对角矩阵
3. $V_M=[\mathbf v_1,\ldots, \mathbf v_M]$ 为前 $M$ 个右特征向量

    注意，代入 (8) 式中时，$V_M$ 需要转置为 $V_M^{\top}$，而 `numpy.linalg.svd` 返回的三元组 `u,d,v` 中 `v` 已经是转置过的，所以直接取 `v[:M,:]`，根据 (8) 式恢复的数据集近似则为 
    ```python
    np.dot(np.dot(u[:,:M],np.diag(d[:M])), v[:M,:])
    ```

这跟这里的[图像压缩存储（图像近似）](/2022/02/10/math/linear_algebra_concepts)本质是一样的,只要存储 $DM+M+MN$ 个数，而原数据集需要存储 $DN$ 个数，从而达到压缩目的。