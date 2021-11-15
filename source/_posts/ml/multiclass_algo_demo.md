---
title: 结构化预测——一个多分类例子
date: 2021-10-08 15:23:18
tags: machine learning
mathjax: true
p: ml/multiclass_algo_demo
---

前面介绍了 [多分类问题的算法实现](2021/09/29/ml/multiclass_algo)，现在讨论一个具体的多分类问题的例子。

<!-- more -->

# 描述
考虑一个 OCR 问题：预测一个手写单词的图片中的单词。为简单起见，假设所有的单词长度均为 $r$，字母表中字母数量为 $q$，且知道如何将一个图片切割成 $r$ 个子图，每个子图 size 相同，且有一个字母，定义 class-sensitive 特征映射函数为 $\Psi(\mathbf x, \mathbf y)$，其中 $\mathbf x$ 可以看成是 $n \times r$ 的一个矩阵，$n$ 为每个图片的像素数量，$\mathbf y$ 中每个元素对应一个字母在字母表中的索引，索引范围 $[1, q]$。$\Psi$ 的值域维度为 $nq+q^2$，其中前 $nq$ 部分记作 “类型1” 的特征，形式为

$$\Psi_{i,j,1}(\mathbf x, \mathbf y)=\frac 1 r \sum_{t=1}^r x_{i,t} \mathbb I_{[y_t=j]}$$

上式表明，$\Psi_{i,j,1}(\mathbf x, \mathbf y)$ 表示在像素位置 $i$ 处表征字母 $j$ 的强度，这是 $r$ 个子图的在 $i$ 处的像素均值，当然也可以考虑跟子图相关的特征，但是那样的话，缺点是特征向量的维度非常大，优点是子图的 size 不必相同，然而我们这里简单起见，所以如此处理。“类型2” 的特征形式为

$$\Psi_{i,j,2}(\mathbf x, \mathbf y)=\frac 1 r \sum_{t=2}^r \mathbb I_{[y_t=i]} \mathbb I_{[y_{t-1}=j]}$$

这是单词中 字母 $i$ 和 $j$ 相邻的强度均值。例如，`qu` 在单词中常见，而 `rz` 在单词中不常见。

从上面的分析可见，特征向量中不仅包含了单个字母的强度，还包含了相邻字母之间的关系强度。

# 预测

最后可通过如下方法预测

$$h_{\mathbf w}(\mathbf x)=\arg \max_{\mathbf y \in \mathcal Y} \ \langle \mathbf w, \Psi(\mathbf x,\mathbf y) \rangle \tag{1} \label{1}$$



由于 $\mathcal Y$ 的大小随 $r$ 指数增涨，于是采用动态规划来有效的计算 $\eqref{1}$ 式。首先将特征写成如下形式

$$\Psi(\mathbf x, \mathbf y)=\frac 1 r \sum_{t=1}^r \phi(\mathbf x, y_t, y_{t-1})$$

于是有

$$\Psi_{i,j,1}(\mathbf x, \mathbf y)=\frac 1 r \sum_{t=1}^r \phi_{i,j,1}(\mathbf x, y_t, y_{t-1})$$

$$\Psi_{i,j,2}(\mathbf x, \mathbf y)=\frac 1 r \sum_{t=1}^r \phi_{i,j,2}(\mathbf x, y_t, y_{t-1})$$

其中 $y_0=0$，且

$$\phi_{i,j,1}(\mathbf x, y_t, y_{t-1})=x_{i,t} \mathbb I_{[y_t=j]}$$

$$\phi_{i,j,2}(\mathbf x, y_t, y_{t-1})=\mathbb I_{[y_t=i]} \mathbb I_{[y_{t-1}=j]}$$

于是 $\eqref{1}$ 可以改写为

$$h_{\mathbf w}(\mathbf x)=\arg \max_{\mathbf y \in \mathcal Y} \sum_{t=1}^r \langle \mathbf w, \phi(\mathbf w, y_t, y_{t-1}) \rangle \tag{2} \label{2}$$

由于 $r$ 是常量，故省略了 $1/ r$ 这一因子。采用动态规划计算时，我们使用一个矩阵 $M \in \mathbb R^{q,r}$，其中元素

$$M_{s, \tau}=\max_{(y_1,\cdots y_{\tau}):y_{\tau}=s} \ \sum_{t=1}^{\tau} \langle \mathbf w, \phi(\mathbf w, y_t, y_{t-1}) \rangle$$

于是 $\eqref{2}$ 式变成求 $\max_s M_{s,\tau}$。而 $M$ 矩阵元素的计算遵循以下迭代关系：

$$M_{s,\tau}=\max_{s'} \ (M_{s',\tau-1}+\langle \mathbf w, \phi(\mathbf x, s, s')\rangle) \tag{3} \label{3}$$

计算 $\eqref{2}$ 式的整个过程为

---
算法 1

**输入：** 原始数据 $\mathbf x \in \mathbb R^{n,r}$，参数 $\mathbf w$

**初始化：**

&emsp; **foreach** $\ s \in [q]$

&emsp;&emsp; $M_{s,1}=\langle w, \phi(\mathbf x, s, -1)\rangle$

**for** $\ \tau=2,\cdots, r$

&emsp; **foreach** $\ s \in [q]$

&emsp;&emsp; 按 $\eqref{3}$ 式计算 $M_{s,\tau}$

&emsp;&emsp; 设置 $I_{s,\tau}=\arg \eqref{3}$ # 使 $\eqref{3}$ 式最大的那个 $s'$

**设置** $y_t=\arg \max_s M_{s,r}$

**for** $\ \tau=r,r-1,\cdots, 2$

&emsp; **set** $y_{\tau-1}=I_{y_{\tau}, \tau}$

**输出：** $\mathbf y = (y_1,\cdots, y_r)$

---

上面的步骤中，$M_{s,1}=\langle w, \phi(\mathbf x, s, -1)\rangle$ ，这里由于字母索引取值范围为 $[0,q-1]$，所以设置 $y_0=-1$。$I_{s,\tau}$ 记录了第 $\tau$ 个字母为 $s$ 时的前一个字母 $s'$，所以 $I$ 矩阵的第 $s$ 行记录了第 $r$ 个字母为 $s$ 时，最大化目标值 $M_{s,r}$ 的所有字母路径。


# 参数学习

定义损失函数为 

$$\Delta(\mathbf y', \mathbf y)=\frac 1 r \sum_{i=1}^r \mathbb I_{[y_i \neq y_i']} \tag{4}$$

学习过程采用 [多分类问题的算法实现](2021/09/29/ml/multiclass_algo_demo) 一文中的 SGD 学习算法，但是由于 $\mathcal Y$ 较大，计算 $h_{\mathbf w}(\mathbf x)$ 时采用了动态规划算法，所以还不能直接套用 SGD 的计算步骤。下面对几个关键点进行讨论，其中求

$$\hat y=\arg \max_{y' \in \mathcal Y} (\Delta(y', y)+\langle \mathbf w^{(t)}, \Psi(\mathbf x, y')-\Psi(\mathbf x, y)\rangle)$$

考虑到 $1/r$ 是常量因子，故统一省略，上式在本例中变成

$$\hat {\mathbf y}=\arg \max_{\mathbf y' \in \mathcal Y} \ \left(\sum_{i=1}^r \mathbb I_{[y_i \neq y_i']}+\langle \mathbf w^{(t)}, \phi(\mathbf x, y_i', y_{i-1}')-\phi(\mathbf x, y_i, y_{i-1})\rangle \right) \tag{5} \label{5}$$

将上面的 $M$ 矩阵改写为 $M'$，

$$M_{s, \tau}'=\max_{(y_1',\cdots y_{\tau}'):y_{\tau}'=s}  \ \left(\sum_{i=1}^{\tau} \mathbb I_{[y_i \neq y_i']}+\langle \mathbf w^{(t)}, \phi(\mathbf x, y_i', y_{i-1}')-\phi(\mathbf x, y_i, y_{i-1})\rangle \right)$$

于是优化目标变成了求 $\max_s M_{s,r}'$，迭代关系为

$$M_{s,\tau}'=\max_{s'} \ (M_{s', \tau-1}'+\mathbb I_{[y_{\tau} \neq s]}+\langle \mathbf w^{(t)}, \phi(\mathbf x, s, s')-\phi(\mathbf x, y_{\tau}, y_{\tau-1})\rangle) \tag{6} \label{6}$$

其中  $M_{s,1}'=\mathbb I_{[y_1 \neq s]}+\langle \mathbf w^{(1)}, \phi(\mathbf x,s,0)-\phi(\mathbf x, y_1,0)\rangle$

注意上式中 $\mathbf y$ 向量元素下标从 $1$ 开始，且字母表中字母索引也从 $1$ 开始，故 $\phi$ 函数中最后一个参数为 $0$，而上面的算法步骤中，遵循程序的约定，下标和索引均从 $0$ 开始，$\phi$ 函数中最后一个参数需要改为 $-1$，这一点需要搞清楚。

现在，就可以通过动态规划计算出 $\hat {\mathbf y}$。梯度 $\mathbf v_t=\Psi(\mathbf x, \hat y)-\Psi(\mathbf x, y)$，各参数均已确定，可以直接计算出梯度 $\mathbf v_t$ 。于是，整个 SGD 方法求解参数 $\mathbf w$ 的学习算法步骤均已确定，总结如下：

---
算法 2

**参数：**

&emsp; 学习率 $\eta$，迭代次数 $T$。

&emsp; 原损失函数 $\Delta: \mathcal Y \times \mathcal Y \rightarrow \mathbb R_+$

&emsp; 映射到特征空间的函数 $\Psi: \mathcal X \times \mathcal Y \rightarrow \mathbb R^d$

**初始化：** $\ \mathbf w^{(1)}=\mathbf 0 \in \mathbb R^d$

**for** $t=1,2,\cdots, T$

&emsp; 随机取样 $(\mathbf x, \mathbf y) \in \mathcal D$

&emsp; 根据 $\eqref{5}, \eqref{6}$ 和“算法 1” 计算 $\hat {\mathbf y}$

&emsp; $\mathbf v_t=\Psi(\mathbf x, \hat {\mathbf y})-\Psi(\mathbf x, \mathbf y)$

&emsp; $\mathbf w^{(t+1)}=\mathbf w^{(t)}-\eta \mathbf v_t$

**输出：** $\ \overline {\mathbf w}=\frac 1 T \sum_{t=1}^T \mathbf w^{(t)}$

---