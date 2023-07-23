---
title: 基于 Real NVP 的密度估计
date: 2022-08-19 14:41:43
tags: flow
mathjax: true
---

论文：[Density Estimation Using Real NVP](https://arxiv.org/abs/1605.08803)

数据分布 $x \sim p_X$，隐变量分布 $z \sim p_Z$，生成过程为根据 $z$ 得到样本 $x$，可表示为 $x=g(z)$，如果 $g$ 是一个双射，那么存在反函数 $f=g^{-1}$，即inference 过程 $z=f(x)$，我们也可以将 $g$ 视为解码，而 $h$ 视为编码。根据随机变量变换公式，概率密度存在关系

$$p_X(x)=p_Z(z)|\det (\frac {\partial g(z)}{\partial z^{\top}})|^{-1} \tag{1}$$

证明：

根据公式 $\mathbf y = \mathbf g(\mathbf x), \ \mathbf h = \mathbf g^{-1}$，概率密度关系为

$$f_Y(\mathbf y)=f_X(\mathbf h(\mathbf y))|J_{\mathbf h}(\mathbf y)| \tag{2}$$

其中 $J_{\mathbf h}(\mathbf y)=\det (\frac {\partial \mathbf h(\mathbf y)}{\partial \mathbf y})$。仿照上面公式写出这里 $x=g(z)$ 的关系式

$$p_Z(z)= p_X(g(z))|J_g(z)|=p_X(x)|J_g(z)|$$

其中

$$J_g(z)=\det \left (\frac {\partial g(z)}{\partial z^{\top}}\right)$$

这里求偏导中为什么是对 $z^{\top}$ 求偏导，这其实无所谓，也可以写成是对 $z$ 求导，无外乎求导后是行向量还是列向量，书写习惯问题。

变换得 

$$p_X(x)=p_Z(z)|J_g(z)|^{-1}=p_Z(x) \left|\det \left (\frac {\partial g(z)}{\partial z^{\top}}\right) \right|^{-1}$$

证毕。

如果能得到一个可逆函数 $g$ ，且我们认为 $x$ 经过变换后 $z=f(x)$ 为一个简单的分布，例如各项同性高斯分布，那么根据 (1) 式，就能得到数据 $x$ 的概率密度。

# 1. 模型

给定一个可观察变量 $x \in X$，隐变量 $z \in Z$ 的一个简单的先验概率 $p_Z$，以及一个可逆变量 $f:X \rightarrow Z$ （逆变换为 $g=f^{-1}$），根据变量变换公式，概率密度关系有

$$p_X(x)=f_Z(f(x)) \left|\det \frac {\partial f(x)}{\partial x}\right| \tag{3}$$

$$\log p_X(x)=\log f_Z(f(x)) + \log \left|\det \frac {\partial f(x)}{\partial x}\right| \tag{4}$$

这里为了简介，求导写为对 $x$ 而非 $x^{\top}$ 求导。

## 1.1 耦合层

为了使 $\det \frac {\partial f(x)}{\partial x}$ 便于计算，可以构造 Jacobian 矩阵为三角矩阵，变量变换关系为

$$\begin{aligned}y_{1:d}&=x_{1:d}
\\y_{d+1:D} &= x_{d+1:D} \odot \exp(s(x_{1:d})) + t(x_{1:d})
\end{aligned} \tag{5}$$

其中 $s, t$ 均为函数 $\mathbb R^d \rightarrow \mathbb R^{D-d}$，分别表示 scale 和 translation（尺度变换和平移变换）。

那么 (5) 式变换的 Jacobian 矩阵为

$$\frac {\partial y}{\partial x}=\begin{bmatrix} \mathbb I_d & 0 \\\\ \frac {\partial y_{d+1:D}}{\partial x_{1:d}} & \text{diag} (\exp [s(x_{1:d})]) \end{bmatrix} \tag{6}$$

(6) 式的行列式为 $\exp \sum_{j=1}^{D-d} s(x_{1:d})_j$ ，可见计算非常简单，于是 $s, t$ 可以使用任意复杂的深度神经网络。

**逆变换**

(5) 式的逆变换为

$$\begin{aligned}x_{1:d} &= y_{1:d}
\\\\ x_{d+1:D} &= (y_{d+1:D}-t(y_{1:d})) \odot \exp(-s(y_{1:d}))
\end{aligned} \tag{7}$$

所以，生成样本时，用的依然是模型的前向传播过程。

## 1.2 掩码卷积

将 $y$ 分为 $y_{1:d}$ 和 $y_{d+1:D}$，这个过程可以使用一个二值掩码 $b$ 实现，然后使用如下函数计算 $y$，

$$y=b \odot x + (1-b) \odot (x \odot \exp (s(b\odot x)) + t(b \odot x)) \tag{8}$$

当 $b_{1:d}=1$，$b_{d+1:D}=0$ 时，根据 (8) 式推导出 (5) 式。

这意味着，我们设置好掩码 $b$ 的值，可以一次性计算 $y$ 的值，而不用分别计算 $y_{1:d}$ 和 $y_{d+1:D}$ 的值。

根据 [NICE](generative_model/2022/08/06/nice) 中分析，$x$ 各维度顺序可以先随机打乱，然后在分为两部分，我们这里使用了掩码 $b$，所以 $b$ 可以任意取值，只要满足 $|\{i|b_i=1\}|=d$ 即可，即掩码 $b$ 中值为 1 的元素数量为 d，当然通常我们选择 $d=D/2$ 。

作者使用了两种分区方案用于挖掘图像的局部相关信息：空间棋盘样式和通道掩码，如图 1，

![](/images/generative_model/real_nvp_1.png)

图 1. 掩码方案。左：空间棋盘样式；右：通道掩码

空间棋盘样式掩码在坐标之和为奇数的位置有值 1。通道掩码在前半通道有值 1 。

## 1.3 组合耦合层

根据 (5) 式中第一式，单个耦合层存在部分元素保持不变的情况，所以需要组合多个耦合层，并交替 $x_{1:d}$ 和 $x_{d+1:D}$ ，试想，如果不交替，那么组合多个耦合层，$x_{1:d}$ 依然一直保持不变，交替之后，上一耦合层保持不变的部分再下一部分则被改变。

组合多个耦合层，Jacobian 行列式依然可以简单计算，因为矩阵乘积的行列式等于矩阵行列式的乘积 $\det (A B)=\det A \det B$ 。逆操作满足 $(f_b \circ f_a)^{-1}=f_a^{-1} \circ f_b^{-1}$ 。

交替变换如图 2 所示，

![](/images/generative_model/real_nvp_2.png)

图 2.

## 1.4 多尺度框架

作者利用一种压缩操作实现多尺度框架。

压缩操作：对每个通道，将图像划分为若干个 shape 为 $2 \times 2 \times c$ 的子方块，然后将它们 reshape 为 $1 \times 1 \times 4c$ 的子方块。压缩操作将 $s \times s \times c$ 的 tensor 转换为 $\frac s 2 \times \frac s 2 \times 4c$ 的 tensor，如图 1。

对于每个尺度，将若干个操作组合为一个操作序列：1. 使用三个耦合层，交替地进行棋盘掩码；2. 执行压缩操作；3. 再使用三个耦合层，交替地进行通道掩码。对于最后一个尺度，则仅仅使用 4 个交替棋盘掩码的耦合层。

D 维向量在所有耦合层中传播将是非常复杂冗余的，使得计算和内存消耗巨大，且训练参数量和很大。为此，作者将一半的维度分解，例如 $L$ 个网络层，那么传播过程为

$$\begin{aligned} h^{(0)} &= x
\\\\ (z^{(i+1)}, h^{(i+1)}) &= f^{(i+1)}(h^{(i)}), \ i=0,1,\ldots, L-2
\\\\ z^{(L)} &= f^{(L)}(h^{(L-1)})
\\\\ z &=(z^{(1)}, \ldots, z^{(L)})
\end{aligned} \tag{9}$$

过程如图 3 所示

![](/images/generative_model/real_nvp_3.png)

图 3. 分解一半的维度。图中，$f^{(1)}(x_{1:D})$ 分为两部分： $z^{(1)}$ 和 $h^{(1)}$，其中 $z^{(1)}$ 保持不变，$h^{(1)}$ 继续传播到 $f^{(2)}$，然后继续分解一半的维度，一直到最后，最后得到的 $z$ 则由每个 $f^{(i)}$ 的输出中保持不变的那一半 concat 组成，即 (9) 式的最后一式。

网络每一层 $f^{(i)}, \ i=1,\ldots, L-1$ 均为上述的 耦合-压缩-耦合组成的操作序列，即 $f^{(i)}$ 为 三耦合-压缩-三耦合。最后一层网络 $f^{(L)}$ 为四个交替棋盘掩码的耦合层。

显然每一层 $f^{(i)}$ 的输入尺度均不同，整个网络就是前面所说的多尺度框架。

由于 $z$ 的分布是一个简单的分布，我们选择使用高斯分布，那么在早期的的网络层（低层，高分辨率）的输出中，有一半的输出不经过后续的网络传播而是直接作为最后的输出，那么需要将这部分输出高斯化。

## 1.5 BN

作者使用了深度残差网络，s 和 t 网络中包含了批归一化和权重归一化操作。

关于权重归一化，可参考 [norm](dl/2021/03/08/norm)。


