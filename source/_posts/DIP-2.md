---
title: 数字图像处理（二）
date: 2019-12-07 11:08:24
tags: DIP
mathjax: true
---

> 参考教材《数字图像处理》(Gonzalez)

# 1. 空间滤波
使用空间滤波器（也称空间掩模，核，窗口）直接作用于图像，得到当前位置的像素值，通过平移得到其他位置的像素值。熟悉深度学习中的卷积操作的话，不难理解这个概念。
<!-- more -->
## 1.1 平滑空间滤波
用于模糊和降噪（通常是模糊后再阈值过滤）。例如以下线性滤波器，
1. 均值滤波
2. 加权均值滤波

又或者统计排序等非线性滤波器，
1. 中值滤波
   
   中值就是统计里面的排序后位于中间的值。中值滤波提供降噪的同时，对图像的模糊程度要低

## 1.2 锐化空间滤波
前面平滑处理使用求和平均，求和可看作积分，锐化操作则相反，通过空间微分实现，目的是突出灰度过渡部分。
对于一维函数 $f(x)$，一阶微分为
$$\frac {\partial f} {\partial x} = f(x+1) - f(x)$$
二维函数 $f(x,y)$ 类似，分别沿两个轴微分。二阶微分为，
$$\frac {\partial^2 f} {\partial x^2} = f'(x) - f'(x-1) = f(x+1) + f(x-1) - 2f(x)$$

以下为一些图像锐化增强的方法。
### 1.2.1 拉普拉斯算子
$$\nabla^2 f = \frac {\partial^2 f} {\partial x^2} + \frac {\partial^2 f} {\partial y^2} $$

又
$$\frac {\partial^2 f} {\partial x^2} = f(x+1,y)+f(x-1,y) - 2f(x,y)
\\\\ \frac {\partial^2 f} {\partial y^2} = f(x,y+1)+f(x,y-1) - 2f(x,y)$$
故
$$\nabla^2 f = f(x+1,y)+f(x-1,y)+f(x,y+1)+f(x,y-1)-4f(x,y)$$

还可以增加对角线方向的微分项，$f(x \pm 1,y \pm 1)$，以及 4 个 $-f(x,y)$，

当然，以上我们还可以将微分乘以 -1，这表示微分的方向反过来，但是其增强效果是跟上面等效的。

通常拉普拉斯算子增强得到将边线和突变点叠加到暗色背景中的图像，所以在叠加原图像，可以恢复背景并保持拉普拉斯锐化结果，如下：
$$g(x,y)=f(x,y)+c\left[ \nabla^2 f(x,y) \right]$$

使用拉普拉斯算子后的图像可能存在负的像素值，此时可将负值转换为 0，超过 255 的值转换为 255（假设为 8 bit 灰度），但是这种处理方法显然过于草率，一种更好的方法是，记拉普拉斯图像为 `f`，然后
$$f_m = f-\min(f)
\\\\ f_s=(L-1)[f_m/\max(f_m)]$$
如此就能保证像素值位于 $[0,L-1]$ 之间。如果叠加原图像，则可能不需要做如此标定。


### 1.2.2 非锐化掩蔽和高提升滤波
操作步骤：
1. 模糊原图像
2. 原图像减模糊图像（差为模板）
3. 将模板加到原图像上
   
$$g_{mask}(x,y) = f(x,y) - \overline f(x,y)
\\\\ g(x,y)=f(x,y) + k \cdot g_{mask}(x,y)$$

### 1.2.3 梯度
二维图像 $f(x,y)$ 的梯度为
$$\nabla f =\begin{bmatrix} g_x \\\\ g_y \end{bmatrix}= \begin{bmatrix} \frac {\partial f} {\partial x} \\\\ \frac {\partial f} {\partial x} \end{bmatrix}$$
这是一个二维列向量，幅值为
$$M(x,y) = \sqrt {g_x^2 + g_y^2}$$

此为梯度图像，与原图像大小相同。有时候使用绝对值来近似，
$$M(x,y)=|g_x|+|g_y|$$

将此滤波写成 $3 \times 3$ 的滤波模板，记一个 $3 \times 3$ 邻域像素值为，
$$\mathbf z=\begin{bmatrix} z_1 & z_2 & z_3 \\ z_4 & z_5 & z_6 \\z_7 & z_8 & z_9 \end{bmatrix}$$
中心为 $z_5$，一阶微分为
$$g_x=z_8-z_5, \quad g_y = z_6-z_5$$

早期的数字图像处理中， Roberts 提出使用交叉差分，
$$g_x=z_9- z_5, \quad g_y = z_8-z_6$$

以上 `x,y` 方向哪个水平哪个垂直，在计算梯度幅值时其实是无所谓的，因为滤波模板在旋转 90° 整数倍时是各向同性的。

__sobel 算子__

$\mathbf w_x=\begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$,  $\mathbf w_y=\begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}$

于是，

~~$$g_x = \mathbf w_x \ast \mathbf z, \qquad g_x = \mathbf w_x \ast \mathbf z$$~~
$$g_x = \mathbf w_x \odot \mathbf z, \qquad g_x = \mathbf w_x \odot \mathbf z$$

sobel 算子常用于边缘检测。

## 1.3 混合空间增强
使用前述多种增加方法

## 1.4 基于模糊技术的灰度变换

模糊集合是一个由 `z` 值和相应隶属度函数组成的序对，
$$A = \{z, \mu_A(z)|z \in Z, \ \mu_A(z) \in (0,1]\}$$
其中 $Z$ 为元素 `z` 的取值空间，隶属度函数的值域为 $[0,1]$。

__空集：__ $\mu_A(z) = 0$

__相等：__ $\mu_A(z) = \mu_B(z), \ \forall z$

__补集：__ $\mu_{\overline A}(z) = 1- \mu_A(z)$

__子集：__ $\mu_A(z) \le \mu_B(z) \Rightarrow A \subseteq B$

__并集：__ $\mu_U(z)=\max [\mu_A(z), \mu_B(z)]$

__交集：__ $\mu_I(z) = \min [\mu_A(z), \mu_B(z)]$

