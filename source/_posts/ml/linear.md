---
title: 线性回归预测
date: 2021-09-17 14:01:41
tags: machine learning
p: ml/linear
mathjax: true
---

讨论线性回归问题
<!--more-->

# 线性回归
## 模型
线性回归问题的假设（hypothesis）为

$$\mathcal H = L_d=\{\langle \mathbf w, \mathbf x \rangle+b: \mathbf w \in \mathbb R^d, b \in \mathbb R\}$$

损失使用平方差，即

$$l(h,(\mathbf x, y))=(h(\mathbf x) - y)^2$$

对于一个大小为 $m$ 的样本集，经验损失为

$$L_S(h)=\frac 1 m \sum_{i=1}^m (h(\mathbf x_i) - y_i)^2$$

## ERM

求 ERM 解，即经验损失最小化，对上式求梯度，令其为0，得

$$\sum_{i=1}^m (\mathbf w^{\top} \mathbf x_i - y_i)\mathbf x_i = \mathbf 0 \Rightarrow \sum_{i=1}^m(\mathbf x_i^{\top} \mathbf w)\mathbf x_i=\sum_{i=0}^m y_i \mathbf x_i$$

写成 $A \mathbf w=\mathbf b$ 的形式，其中

$$A=\left(\sum_{i=1}^m\mathbf x_i \mathbf x_i^{\top} \right), \quad \mathbf b = \sum_{i=0}^m y_i \mathbf x_i$$

上面的简写中用到了 $(\mathbf x^{\top} \mathbf w)\mathbf x=\mathbf x(\mathbf x^{\top} \mathbf w)=(\mathbf {x x}^{\top}) \mathbf w$ 这样一个事实。

### A 可逆
如果 $A$ 是可逆矩阵，那么 ERM 的解为

$$\mathbf w = A^{-1} \mathbf b$$

根据 $A$ 的定义，

---
**如果训练集中的向量能张开（span）整个 $\mathbb R^d$，那么 $A$ 就是可逆的。**


下面简单的证明一下这个结论。

__证：__

对于 $\forall \ \mathbf v \in \mathbb R^d$，

$$A \mathbf v=\left(\sum_{i=1}^m\mathbf x_i \mathbf x_i^{\top} \right)\mathbf v=\sum_{i=1}^m\mathbf x_i \mathbf x_i^{\top} \mathbf v=\sum_{i=1}^m\mathbf x_i (\mathbf x_i^{\top} \mathbf v)=\sum_{i=1}^m(\mathbf x_i^{\top} \mathbf v)\mathbf x_i$$


因为训练集中的样本向量可以张开 $\mathbb R^d$ 空间，这表示有 $d$ 个非线性相关的向量，不妨假设它们的编号是前 $d$，即 $\mathbf x_1, \cdots, \mathbf x_d$，任意一个样本向量可表示为这 $d$ 个样本向量的线性组合，即 $\mathbf x_j=\sum_{i=1}^d c_{ji} \mathbf x_i$，记 $\mathbf x_i^{\top} \mathbf v=p_i, \ \forall i \in [m]$，于是

$$\begin{aligned}A \mathbf v&=\sum_{i=1}^d p_i\mathbf x_i + \sum_{j=d+1}^m \sum_{k=1}^d p_j c_{jk} \mathbf x_k
\\&=\sum_{i=1}^d p_i\mathbf x_i + \sum_{k=1}^d\sum_{j=d+1}^m  p_j c_{jk} \mathbf x_k
\\&=\sum_{i=1}^d \left(p_i + \sum_{j=d+1}^m p_j c_{ji}\right)\mathbf x_i
\\&=\sum_{i=1}^d q_i \mathbf x_i\end{aligned}$$

可见，$A$ 的映射空间为 $\mathbf x_1, \cdots, \mathbf x_d$ 这一组非线性相关向量线性组合而成，且 $\exists \ \mathbf v \in \mathbb R^d$，使得 $\forall i \in [d], \ q_i\neq 0$。显然如果 $A$ 不可逆，那么至少会有一个 $q_i$ 值为 0，矛盾，所以 $A$ 可逆。

### A 不可逆

---
从 $A$ 的定义可以发现它是一个对称矩阵，并且巧合地是，$b$ 恰好处在 $A$ 的映射空间中。

__证：__

$A$ 是对称矩阵，可以进行特征值分解 $A=VDV^{\top}$，其中 $D$ 是由特征值组成的对角矩阵，$V$ 是由特征向量组成的矩阵，且 $V$ 是一个正交矩阵即 $V^{\top}V=I_{d\times d}$，定义对角矩阵 $D^+$

$$D_{ii}^+=\begin{cases}  1 /D_{ii} & D_{ii} \neq 0 \\ 0 & D_{ii}=0\end{cases}$$

$$A^+=VD^+V^{\top}, \quad \hat {\mathbf w}=A^+b$$

将 $V$ 写成

$$V=[\mathbf v_1, \cdots, \mathbf v_d]$$

于是

$$\begin{aligned}A\hat {\mathbf w}&=AA^+ \mathbf b
\\&=VDV^{\top}VD^+V^{\top}\mathbf b
\\&=VDD^+V^{\top}\mathbf b
\\&=\sum_{i:D_{ii}\neq 0} \mathbf v_i \mathbf v_i^{\top} \mathbf b
\\&= \overline A \ \mathbf b

\\&=\sum_{i:D_{ii}\neq 0}( \mathbf v_i^{\top} \mathbf b) \mathbf v_i
\\&= \sum_{i=1}^d q_i \mathbf v_i
\end{aligned}$$

其中 

$$q_i=\begin{cases} \mathbf v_i^{\top} \mathbf b & D_{ii}\neq 0 \\ 0 & D_{ii} = 0 \end{cases}$$

借鉴上面 $A$ 是可逆矩阵的证明可知（注意这里 $A$ 不假设为可逆矩阵），上式最后一项表示将 $\mathbf b$ 投影到 $\overline A$ 的投影空间，这个投影空间是由所有 $D_{ii} \neq 0$ 所对应的向量 $\mathbf v_i$ 张开（span）。而

$$A=\left(\sum_{i=1}^m\mathbf x_i \mathbf x_i^{\top} \right)=[\mathbf v_1, \cdots, \mathbf v_d]D[\mathbf v_1^{\top}, \cdots, \mathbf v_d^{\top}]^{\top}=\sum_{i:D_{ii} \neq 0} \mathbf v_i\mathbf v_i^{\top}$$

这表示 $\mathbf x_1, \cdots , \mathbf x_m$ 所张空间与 $\{\mathbf v_i: D_{ii} \neq 0\}$ 所张空间相同，且 $\mathbf b = \sum_{i=1}^m y_i \mathbf x_i$ 处在 $\mathbf x_1, \cdots , \mathbf x_m$ 所张空间中，也就是处在 $\{\mathbf v_i: D_{ii} \neq 0\}$ 所张空间中， 由于 $\{\mathbf v_i: D_{ii} \neq 0\}$ 是正交规范向量，可看到这个空间中的一组正交基，**它对处在这个空间中的向量的变换保持不变**，即 $(\sum_{i:D_{ii}\neq 0} \mathbf v_i \mathbf v_i^{\top})\mathbf b=\mathbf b$，于是有

$$A \hat{\mathbf w}=\mathbf b$$

这表明 $\hat {\mathbf w}=A^+b$ 是符合条件的解，证毕。



注：实际应用中 $A$ 的特征分解可使用 `scipy` 包。