---
title: 边缘检测
date: 2022-04-08 10:31:37
tags: DIP
categories: 数字图像处理
summary: 图像中边界点通常是图像像素值不连续的点
mathjax: true
---

对灰度值求导，导数不为 0 的点即边界点。图像的像素点是离散点，故求导数实际上就是求相邻像素点的差值。

# 1. 边缘检测算子

## 1.1 卷积核

### 1.1.1 简单的差分滤波
1. x 方向偏导

$$\begin{array}{|c|c|}
\hline
-1 & 1
\\
\hline
\end{array}$$

2. y 方向偏导

$$\begin{array}{|c|}
\hline
-1
\\
\hline
1
\\
\hline
\end{array}
$$

对于卷积后的像素值，如果超过 `0-255` 范围，直接取绝对值就行。


### 1.1.2 Roberts 算子

Roberts 算子考虑对角方向相邻像素差。

$$dx = \begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix}$$
$$dy = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$$

### 1.1.3 Prewitt 算子

间隔一个像素点的两个相近点的像素差，可以去掉部分伪边缘。

$$dx = \begin{bmatrix} 1 & 0 & -1 \\ 1 & 0 & -1 \\ 1 & 0 & -1 \end{bmatrix}$$
$$dy = \begin{bmatrix} -1 & -1 & -1 \\ 0 & 0 & 0 \\ 1 & 1 & 1 \end{bmatrix}$$

### 1.1.4 Sobel 算子

在 Prewitt 算子的基础上考虑权重：越靠近中心的权重越大。

$$dx = \begin{bmatrix} 1 & 0 & -1 \\ 2 & 0 & -2 \\ 1 & 0 & -1 \end{bmatrix}$$
$$dy = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$

## 1.2 图像梯度

### 1.2.1 梯度定义

$$\nabla f = \begin{bmatrix} \frac {\partial f} {\partial x}, & \frac {\partial f} {\partial y} \end{bmatrix}$$

### 1.2.2 梯度角度

$$\theta = \tan^{-1} \left(\frac {\partial f} {\partial x} / \frac {\partial f} {\partial y} \right)$$

### 1.2.3 梯度幅值

$$\|\nabla f\| = \sqrt{\left(\frac {\partial f} {\partial x} \right)^2+\left( \frac {\partial f} {\partial y}\right)^2}$$


计算出 x，y 两个方向的梯度值之后，再计算梯度的幅值从而融合成一幅图像，最后二值化处理，可以得到边缘图。但是这种方法得到的边缘图存在很多问题，例如噪声污染未排除，边缘线过于粗宽等。

解决思想：
1. 平滑处理：使用高斯滤波器降噪，
2. 对图像信号求导

## 1.2.4 高斯滤波器
使用高斯平滑降噪。二维高斯公式，

$$G(x,y)=\frac 1 {2 \pi \sigma^2} e^{-\frac {x^2+y^2} {2\sigma^2}}$$

例如：

$$\frac 1 {16} \times \begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 &2 \\ 1 & 2 & 1\end{bmatrix}$$

$$\frac 1 {273} \times \begin{bmatrix} 1 & 4 & 7 & 4& 1 \\ 4 & 16 & 26 & 16 & 4 \\ 7 & 26 & 41 & 26 & 7 \\ 4 & 16 & 26 & 16 & 4 \\ 1 & 4 & 7 & 4& 1 \end{bmatrix}$$

### 1.2.5 Candy 算子

根据卷积性质，$(f \star g)'=f \star g'$，故可以 __先对平滑核求导，然后再卷积__。

> 计算 x, y 方向的梯度后，可以进行归一化， $\nabla f / \max \nabla f$

### 1.2.6 非极大抑制

根据梯度幅值得到的图像仍然存在边缘粗宽，弱边缘干扰等问题，可以采样非极大抑制。

对于某一点 $C$，其灰度值在 8连通邻域内是否最大，如果是则继续检查其梯度方向（即上面的梯度角度）与 8 连通邻域连线的交点 $c1, c2$，如 $C$ 点值大于 $c_1, c_2$ 的值，那么 $C$ 处值不变，否则为 0。这里 $c_1, c_2$ 如果不在 8 连通邻域集合内，那么就是8 连通邻域集合里最近的两个点的线性插值得到。

### 1.2.7 双阈值边缘连接

经过非极大抑制后，仍然存在部分伪边缘，Candy 算法中采取双阈值 $t_1, t_2, \ t_1 < t_2$，将小于 $t_1$ 的点置为 0（伪边缘），大于 $t_2$ 的置 1。在 $t_1, t_2$ 之间的则等待进一步处理。

将像素值为 1 的点连接（8连通邻域）起来，形成轮廓，当到达轮廓端点时，在端点的 8 邻域内寻找满足介于 $t_1, t_2$ 之间的点，再根据此点收集新的端点，知道轮廓闭合。 

这里需要使用递归，即当一个点找不到下一个合适的 8 邻域内的点做新的端点时，那么回溯到这个点的上一个点，并重新找 8 邻域内的新端点。