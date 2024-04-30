---
title: 基于特征同步的抗屏摄鲁棒水印
date: 2024-04-08 17:07:48
tags: stego
---

论文：[Screen-Cam Robust Image Watermarking with Feature-Based Synchronization]()

# 1. 简介

本文提出一种抗屏摄鲁棒水印，基于特征同步方法，这里特征同步是为了获取屏摄之后图片上的嵌入水印区域，找到区域后从而可以提取水印。

# 2. 屏摄过程分析

假设图片在显示器上显示，然后使用照相机拍照，此过程中产生的畸变包含五类：

## 2.1 线性畸变

显示器显示图片时，会产生亮度，对比度以及颜色畸变，这由显示器的质量以及设置决定，这些畸变近似为线性畸变。

照相机中有图像信号处理器（ISP），拍照时 ISP 会对图像进行校正，这个校正过程也存在线性畸变。

## 2.2 伽马校正

为了匹配人眼视觉，显示器在显示时对图像进行了伽马校正，这是一种非线性畸变。

$$I = I ^ {\gamma}$$

## 2.3 几何畸变

由于拍照的距离和角度，引起透视变换。

## 2.4 噪声攻击

噪声攻击会引起图像质量的锐减。噪声攻击分为四类：

1. moire 噪声
2. 外部环境引起的噪声（光照等）
3. 硬件引起的噪声
4. 软件引起的噪声

## 2.5 低通滤波攻击

照相机的分辨率通常比原始图片分辨率高，在捕获光学信号时，原始图片的每一个像素点的信号并非独立的被捕获，相邻像素的光之间发生干扰从而引起模糊，可近似为一个低通滤波攻击。

细节对应高频，而拍照后图片模糊，细节丢失，也就是低频通过，高频被过滤。

# 3. 方法

水印分为三步：1. 计算 LSFR 作为嵌入区域；2. 在每个 LSFR 区域重复地嵌入水印；3. 拍照后图片经过透视变换后，从每个候选区域中提取水印，然后确定最终地水印。

## 3.1 LSFR

Local square feature region(LSFR)

由于拍照和用户的操作（例如 crop）均会引起不同步攻击，所为不同步，就是原始图片的嵌入区域坐标在经过不同步攻击后，坐标改变了。所以我们需要找到一个合适的同步方法，以便能定位水印嵌入区域。

作者研究了三种方法：Harris-Laplace，SIFT 和 SURF。如图 1，在不同的拍照距离下，各种特征点坐标，特征尺度和特征方向的复现率。

![]()

<center>图 1</center>

由于低通滤波攻击和透镜畸变，图片边界变模糊，所以拍照后的图片与原图存在错位，不可能完全一致，所以只要特征点在拍照前后的坐标误差小于 5 像素，特征尺度的变化还需要在 10% 以内，我们就认为这个特征点被复现出来。另外，靠近图片边界的特征点不在考虑之内。

图 1 (a) 就是在不同拍照距离下的特征点复现率（复现出来的特征数量除以总特征数量，靠近图片边界的特征不在考虑范围），图 1 (b) 是特征角度在 5 ° 以内的复现率。

我们在计算特征点之前进行了高斯滤波，即，对原图和拍照后图片先进行高斯滤波，以降低低通滤波攻击的影响。

根据图 1 (a)，作者发现对于 中高尺度的特征（特征尺度 > 15），Harris-Laplace 和 SIFT 特征的复现率较高，但是在拍照距离较小时，SIFT 特征复现率下降，说明 SIFT 对 moire 噪声更加敏感，经过比较，Harris-Laplace 特征更加稳定，适合用于同步。

根据图 1 (b)，SURF 特征角度更稳定。

因此，本文使用 Harris-Laplace 检测器和 SURF 方向描述子的结合，获取 RST（旋转缩放平移） 不变的局部特征区域（LFR）。为了增大拍照后的特征检测率，使用了高斯滤波。

图 2 是获取 LSFR 的过程。

![]()

<center>图 2. (a) 高斯函数之后的图片亮度通道；(b) Harris-Laplace 提取的特征点；(c) 特征相关的尺度和朝向；(d) 相应的 LSFR</center>

在 [Fang H. Screen-shooting resilient watermarking]() 中，使用 SIFT 特征点作为同步方法，并且以特征点作为 axes-aligned 区域的中心，但是本文中，嵌入区域不一定是 axes-aligned，如图 2 (d)，与特征点的朝向有关。

### 3.1.1

在嵌入和提取水印的过程中，检测特征点需要在经过高斯滤波的图片上进行，注意，高斯滤波的图片仅仅用于检测特征点，而不是用于嵌入水印和提取水印。

一个二维高斯函数可以使用两个一维高斯函数相乘表示（这里我们只考虑两个维度的随机变量独立），

$$G(x, y, \sigma) = \frac 1 {2 \pi \sigma ^ 2} \exp \left(-\frac {x^2 + y^2}{2 \sigma ^ 2}\right) \tag{1}$$

高斯核 $H _ G$ ，设置 $\sigma = 2$，$\sigma$ 越小衰减越快，设置窗口大小为 7，也就是 kernel size 为 7，对图像使用 $H _ G$ 卷积，

$$I'(x,y, \sigma)=H _ G * I(x,y) \tag{2}$$

步长为 1，padding size 为 3 。$*$ 表示卷积操作。

高斯卷积代码示例，

```python
import numpy as np
import cv2
import torch
import torch.nn as nn

normalize = True
sigma = 2
k = 7
x = np.linspace(-k//2, k//2, k)
x, y = np.meshgrid(x, x)
s = (x ** 2 + y ** 2) / 2 / sigma ** 2
g = np.exp(s)
if normalize:   # do normalization instead of /(2pi)/sigma/sigma
    g = g / np.sum(g)
gaussian = nn.Conv2d(1, 1, k, padding=k//2, bias=False)
gaussian.weight.requires_grad = False
gaussian.weight[:] = torch.from_numpy(g)
I = cv2.imread(path)
I = I[:,:,::-1].transpose(2, 0, 1)
I = torch.from_numpy(I).unsqueeze(0).float()
I2 = gaussian(I)    # (1, 3, h, w)
```

### 3.1.2 Harris-Laplace 检测器

本文使用 Harris-Laplace 检测器。

Harris 矩阵为

$$H = \sum w(i, j) \begin{bmatrix}I _ x ^ 2 & I _ x I _ y \\ I _ x I _ y & I _ y ^ 2 \end{bmatrix} \tag{3}$$

其中 $w(i, j)$ 是高斯核，$I _ x, \ I _ y$ 分别是水平方向导数和竖直方向导数，可以使用 Sobel 梯度算子（可以通过除以 2 对 Sobel 算子归一化），

$$S _ x = \begin{bmatrix}1 & 0 & -1 \\ 2 & 0 & -2 \\ 1 & 0 & -1\end{bmatrix}$$

$$S _ y = S _ x ^ {\top}$$


Harris 特征点具有平移和旋转不变性，但是不具有尺度不变性，为了获取尺度不变性，采用 Harris-Laplace 检测，通过不同的尺度参数的高斯核与原始图像卷积生成尺度空间，

$$L(x,y,\sigma) = G(x,y,\sigma) * I(x,y)=I'(x,y,\sigma) \tag{4}$$

$*$ 表示卷积操作。高斯核为 (1) 式表示。此时改进后的 Harris 矩阵为

$$H = \sigma _ D ^ 2 G(\sigma _ I) * \begin{bmatrix} L _ x ^ 2 (x,y,\sigma _ D) & L _ x (x,y,\sigma _ D)L _ x (x,y,\sigma _ D) \\ L _ x (x,y,\sigma _ D)L _ x (x,y,\sigma _ D) & L _ y ^ 2 (x,y,\sigma _ D)\end{bmatrix} \tag{5}$$

其中 $\sigma _ I, \ \sigma _ D$ 分别表示积分尺度和微分尺度。$\sigma _ D$ 对应 (4) 式中的 $\sigma$，用于生成尺度空间 $L(x,y,\sigma _ D)$，在尺度空间 $L(x,y,\sigma _ D)$ 上根据 Sobel 算子求导（故 $\sigma _ D$ 为微分尺度），$\sigma _ D$ 越大，导数越小（因为越平滑），为了补偿，乘以 $\sigma _ D ^ 2$（注意这里是平方，因为矩阵里面导数也是平方）。最后再使用 $G(\sigma _ I)$ 进行积分，积分尺度影响高斯核中元素的值（从高斯核中心向周围下降的速率）。

Harris 特征点得分为

$$C = \det(H) - k \cdot \text{trace} ^ 2 (H) \tag{6}$$

每一个像素点位置均有一个得分值，这样就是得到一个 score map，使用这个 score map 中的最大值，再乘以一个比例（例如 $0.1$）作为阈值，所有小于阈值的全部置 0。

这里 $k$ 根据经验值设置为 $0.04$。还有一种不依赖于经验值的得分计算方法：

$$C=\det(H) / (\text{trace}(H) + \epsilon) \tag{7}$$

然后进行非极大抑制，通过 opencv 膨胀操作实现，

```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 3)   # 3x3 kernel
dst = cv2.dilate(score, kernel)
mask = (score == dst) and (score != 0)
score[~mask] = 0
```

膨胀相当于 size 为 3x3、 stride 为 1 的 maxpool。膨胀前后，元素值相等且不为 0 则为 harris 特征位置。

现在，我们只是不同尺度空间（$\sigma _ D$）上各自独立地计算了特征点，然后根据 DoG(Difference of Gaussian) 或者 LoG(Laplacian of Gaussian) 选择特征点。

**# DoG**

根据一组不同尺度（高斯平滑尺度 $\sigma _ D$）平面（这里其实是 $L(x,y,\sigma _ D)$ 平面），求 difference，得到 DoG，例如 6 个不同尺度高斯平滑图像，可以得到 5 个 DoG 特征平面，然后遍历中间 3 个 DoG 的某位置点，是否具有极大值（即 DoG 值是否极大），如是，那么此位置是关键点，其值为此位置处的 $C$ 值。

通常，我们会有多组 DoG，组间的特征 spatial size 不同，第一组 spatial size 与原始图像 size 相同，从第二组 DoG 开始，依次进行 x2 下采样。那么每组 DoG 得到的关键点还需要去重。

去重步骤：

1. 所有关键点根据 $C$ 值降序排列，得到一个排序后的关键点列表
2. 如果两个下标相邻关键点 $C$ 值相等，且位置之间距离不超过 `max_diff`，那么设置新的关键点为这两个关键点位置的中心，并删除旧的两个关键点。

    ```python
    max_diff = 2 ** (octave + 0.5)  # octave 为组 index，从 0 开始
    ```

**# LoG**

我们需要找到局部结构的最显著的尺度，即最显著的 $\sigma _ n$，在这个尺度下，LoG 相应最大。那么匹配时，只需要求最显著尺度下的特征点。

1. 图像 size 变换前后，分别找到最显著尺度
2. 在各自最显著尺度下，响应函数的值均为最大，那么我们就得到图像缩放前后的匹配点

LoG 的响应为

$$LoG(x, \sigma _ n) = \sigma _ n ^ 2 |L _ {xx}(x, y, \sigma _ n) + L _ {yy}(x, y, \sigma _ n)| \tag{8}$$

上式中是求二阶导，$L(x,y,\sigma _ n)$ 是原始图像使用 $\sigma _ n$ 进行高斯平滑。

根据 (6) 式得到各个 scale 下的 harris 特征点，然后使用 LoG 选择不同 scale 之间的极大值，也就是说使用 LoG 选择 scale 和 location 。

寻找最显著尺度的迭代步骤为：

1. 使用 $\sigma _ n = \xi ^ n \sigma _ 0, \ n = 1,2,\ldots, N$ 作为 $N$ 个尺度空间中的积分尺度 $\sigma _ I$ 去计算 (6) 式，然后每个 scale 单独提取 8 连通域中的局部极大值，并使用一个阈值丢弃太小的局部极大值。这里 $\sigma _ D = s \cdot \sigma _ n$，其中 $s=0.7, \ \xi = 1.4$ ，得到初始的 harris 特征点 $X$ 。

2. 对于一个初始点 $\mathbf x \in X$，其对应的尺度记为 $\sigma _ I$（实际上对应的是某个 $\sigma _ n$）。第 $k$ 次迭代时，记这个点为 $\mathbf x ^ {(t)}$， 寻找不同迭代尺度的 LoG 值的极值，这里的迭代尺度为 $\sigma _ I ^ {(k+1)} = t \sigma _ I ^ {(k)}, \ t = [0.7, \ldots, 1.4]$，如果 $\mathbf x$ 不是 LoG 的极值，则丢弃该店，继续评估 $X$ 中的下一个点；否则，在 $\sigma _ n ^ {(k+1)}$ 对应的尺度层中按 8 连通域寻找最大 Harris 的 $C$ 值 $\mathbf x ^ {(k+1)}$，这里可能需要根据 $\sigma _ I ^ {(k+1)}$ 重新计算对应的 $C$ 得分平面，然后求 $\mathbf x ^ {(k)}$ 的 8 连通域内极大值 $\mathbf x ^ {(k+1)}$。

3. 如果该点的尺度和空间位置均不再变化，即 $\mathbf x ^ {(t+1)}=\mathbf x ^ {(t)}$，$\sigma _ I ^ {(k+1)} = \sigma _ I ^ {(k)}$（即此时 $t=1$），停止搜索，否则继续上一步迭代，即根据 $\sigma _ I ^ {(k+2)} = t \sigma _ I ^ {(k+1)}, \ t = [0.7, \ldots, 1.4]$ 尺度空间，寻找 $\mathbf x ^ {(k+1)}$ 的 LoG 极值，并在 $\sigma _ I ^ {(k+2)}$ 尺度层按 Harris 的 $C$ map 搜索 $\mathbf x ^ {(k+2)}$

Harris-Laplace 在特定尺度下进行 Harris 检测时不再使用 sobel 梯度算子，而是使用当前尺度（微分尺度 $\sigma _ D = 0.7 \sigma _ n$ 作为梯度算子高斯核的标准差，该高斯核算子半径为 $3\sigma _ D$。我们先对原始图片做高斯平滑，

$$I' = G(\sigma _ D) * I \tag{9}$$

这里 $I'$ 就是这个尺度下的图像，然后在 $I'$ 计算灰度变化，

$$E(u, v) = \sum _ {x,y} w (x,y) [I'(u+x,y+v) - I'(u,v)] ^ 2 \tag{10}$$

根据泰勒展开 $f(x+u,y+v) \approx f(u,v) + x f _ x(u, v) + y f _ y(u, v)$，所以

$$E(u,v) = \sum _ {x,y} w(x,y) \left[\frac {\partial I'(u,v)}{\partial x} x + \frac {\partial I'(u,v)}{\partial y} y \right] ^ 2=\sum _ {x,y} w(x,y) \cdot [x \quad y] \cdot H \cdot \begin{bmatrix} x \\ y \end{bmatrix} \tag{11}$$

其中 

$$H=\begin{bmatrix} \left(\frac {\partial I'(u,v)}{\partial x}\right) ^ 2 & \frac {\partial I'(u,v)}{\partial x}\frac {\partial I'(u,v)}{\partial y} \\ \frac {\partial I'(u,v)}{\partial x}\frac {\partial I'(u,v)}{\partial y} & \left(\frac {\partial I'(u,v)}{\partial y}\right) ^ 2\end{bmatrix} \tag{12}$$

且有

$$\begin{aligned}\frac {\partial I'(u,v)}{\partial x} &= \frac {\partial [G(\sigma _ D) * I(u,v)]}{\partial x} 
\\ &= \frac {\partial} {\partial x} [\sum _ x\frac 1 {\sqrt {2\pi}\sigma _ D}\exp (-x^2/(2\sigma _ D ^ 2)) I(u+x,v+y)]
\\ &=-\sum _ x \frac {x} {\sqrt {2\pi}\sigma _ D ^ 3}\exp (-x^2/(2\sigma _ D ^ 2)) I(u+x,v+y)
\end{aligned} \tag{13}$$

高斯核半径为 $3 \sigma _ D$，那么上式求和中 $x \in [-3\sigma _ D, 3\sigma _ D]$。

计算 LoG 特征时，使用的是 $\sigma _ n$，于是

$$\begin{aligned} L _ {xx} (x,y,\sigma _ n)&=\frac {\partial}{\partial x ^ 2}(G(\sigma _ n) * I)
\\ &= \frac {\partial} {\partial x^2} \left(\sum _ {x,y} \frac 1 {2\pi \sigma _ n ^ 2}\exp [-(x^2+y ^ 2)/(2\sigma _ D ^ 2)] \cdot I(u+x,v+y)\right)
\\ &= \frac 1 {2\pi \sigma _ n ^ 6}\sum _ {x,y} \exp  [-(x^2+y ^ 2)/(2\sigma _ D ^ 2)] (x ^2 - \sigma _ n ^ 2)\cdot I(u+x,v+y)
\end{aligned} \tag{14}$$

根据前面迭代步骤 2，由于是在同一个尺度空间 $\sigma _ n$ 寻找 8 连通域的 LoG 极值，所以这里可以将常量 $1/(2\pi \sigma _ n ^ 6)$ 去掉。

于是

$$LoG = |L _ {xx} (x,y,\sigma _ n) + L _ {yy} (x,y,\sigma _ n)|=\sum _ {x,y} \exp  [-(x^2+y ^ 2)/(2\sigma _ D ^ 2)] (x ^2 + y ^ 2- 2\sigma _ n ^ 2)\cdot I(u+x,v+y) \tag{15}$$

最后特征半径为 $\pi \sigma _ n$，其中 $\sigma _ n$ 为特征点所在的尺度空间。

迭代法比较复杂且繁琐，简化方法为：

1. 计算 harris 候选角点（与迭代法相同）
2. 计算 LoG 尺度空间的响应值，判断每个候选角点是不是 LoG 相邻尺度空间的响应极值，如是则保留，否则抛弃
3. 最后所有保留的候选角点为最终的特征点。

`opencv_contrib` 项目中的 `HarrisLaplaceFeatureDetector_Impl::detect` 使用的是 Harris + DoG 组合计算 Harris-Laplace 特征点。

### 3.1.3 SURF 方向描述子

通过 Harris-Laplace，确定特征点位置和半径，如果所选区域是一个圆，那么特征选择工作已经结果，但是这里我们嵌入水印的区域是一个方形，所以还需要确定方向，确定方向之后，即使拍照后图像经过旋转，也能提取水印。

本文使用 SURF 方向描述子。

记特征半径为 $s$，以特征点为中心，$6s$ 为半径画圆，计算这个圆形区域的 Haar 小波变换，Haar 小波模板 size 为 $4s$，对 Haar 小波变换后的图像做高斯平滑，标准差 $2s$。

Haar 小波水平方向模板和竖直方向模板分别为

$$\begin{bmatrix}-1 & -1 & \ldots & 1 & 1 \\-1 & -1 & \ldots & 1 & 1 \\ \vdots \\-1 & -1 & \ldots & 1 & 1\end{bmatrix}, \quad\begin{bmatrix}-1 & -1 & \ldots & -1 & -1 \\-1 & -1 & \ldots & -1 & -1 \\ \vdots \\1 & 1 & \ldots & 1 & 1\end{bmatrix}$$

Haar 小波模板的 size 为 $4s$，对 $6s$ 半径的圆进行加权求和，扫描步长为 $s$，这样得到 $13 \times 13$ 的 Haar 小波响应值，然后使用 $2s$ 的高斯平滑，高斯核中心为特征点位置，只有这一个高斯核，高斯核窗口半径就是 $6$，不用滑动高斯核。当然这样计算出来的是一个方形区域，我们只考虑 $x ^ 2 + y ^ 2 \le 6 ^ 2$ 的圆形区域内的 Haar 相应值。

然后在圆形区域内使用一个张角为 60° 的扇形，统计扇形区域内 Haar 小波特征总和，转动扇形，再统计 Haar 小波特征和，此特征和最大的方向为 SURF 主方向。

OpenSURF 中计算 Haar 特征值的代码如下，

```c++
for (int i = -6; i <= 6; ++i) { // output size 为 12x12
    for (int j = -6; j <= 6; ++j) {
        if (i * i + j * j < 36) {
            // 高斯平滑核的权重 gauss
            gauss = static_cast<float>(gauss25[id[i+6]][id[j+6]]);
            resX[idx] = gauss * haarX(r+j*s, c+i*s, 4*s);
            resY[idx] = gauss * haarY(r+j*s, c+i*s, 4*s);
            Ang[idx] = getAngle(resX[idx], resY[idx]);
            ++idx;
        }
    }
}
```

高斯核窗口半径为 6，其 size 为 13，由于高斯核是一个对称的，我们只考虑 $1/4$ 的高斯核，例如右下的 $1/4$ 部分，即

```c++
const double gauss25[7][7] = { ... };   // 事先计算好
```

于是高斯核中心位于 `gauss25[0][0]`，最边缘位于 `gauss25[6][6]`，那么兴趣区域内点的位置映射到高斯核元素位置使用

```c++
const int id[] = {6,5,4,3,2,1,0,1,2,3,4,5,6};
```

`r, c` 为特征点的坐标，也就是圆形中心的坐标，`haarX` 使用水平模板计算 haar 响应值，`getAngle` 根据水平和竖直两个方向的 haar 响应值计算角度 `atan(y/x)` 。

接下来是将 haar 响应值之和最大的那个扇形作为特征主方向，寻找最大值的代码为，

```c++
// ang1 扇形其实角度
// ang2 扇形结束角度
float max=0.0f;
float orientation = 0.0f;
for (ang1 = 0; ang1 < 2 * pi; ang1 += 0.15f) {
    ang2 = (ang1 + pi/3.0f > 2 * pi ? ang1 - 5.0f*pi/3.0f : ang1 + pi/3.0f);
    sumX = sumY = 0.0f;
    for (unsinged int k = 0; k < Ang.size(); ++k) {
        const float &ang = Ang[k];  // haar 响应角度
        // haar 响应角度方向位于扇形内
        if (ang1 < ang2 && ang1 < ang && ang < ang2) {
            sumX += resX[k];
            sumY += resY[k];
        }
        else if (ang2 < ang1 && (
            (ang > 0 && ang < ang2) || (ang > ang1 && ang < 2*pi)
        )) {
            sumX += resX[k];
            sumY += resY[k];
        }
    }
    if (sumX * sumX + sumY * sumY > max) {  // 考虑的幅度，不考虑符号
        max = sumX * sumX + sumY * sumY;
        orientation = getAngle(sumX, sumY);
    }
}
```

### 3.1.4 嵌入水印区域 LSFR

考虑到屏摄过程中的严重畸变，LSFR 需要有足够的大小范围以保证经过屏摄后信息仍能存活，LSFR 的边长 size 为

$$L _ 0 = 2 \cdot \text{floor} (k _ 1 \cdot s) + 1 \tag{16}$$

其中 $k _ 1$ 是一个常量系数，$s$ 是特征点的特征尺度。

如图 3，是 8 个来自测试集中的图片。当一个 LSFR 的小部分超出图片边缘，如图 3 (f)，或者两个 LSFR 重叠了一小部分，如图 3 (g)，即使这样的 LSFR 也可以用作嵌入区域。

![]()

<center>图 3</center>

## 3.2 水印嵌入

### 3.2.1 选择嵌入方法

由于屏摄后图片边界模糊，即使经过透视变换的校正，校正后图像与原图也存在一些位置偏移。幸运地是，由于离散傅里叶变换 DFT 系数的属性，屏摄后的图片的系数可以纠正到与原图相当，只要谨慎地四个角点，使得透视变换校正后的图片其旋转和尺度缩放失真不大，故本文选择 DFT 域用于嵌入水印。

为了使用 DFT 系数作为水印载体，我们需要分析在屏摄过程中 DFT 系数的变化规律。在不同屏摄中 DFT 系数幅值的变化进行了详细的分析。由于中频系数常用于水印载体，作者使用 512x512 大小的 Lena 图片的中频频谱在屏摄后的变化作为研究案例，结果如图 4 所示，

![]()

<center>图 4. (a) 原图中频带的 DFT 系数幅值的对数；(b,c) 经过 30 和 50 cm 距离的拍照后，DFT 系数幅值对数的变化值</center>

从图 4 中可以观察出，对于中频带，大多数幅值大的值经过屏摄后变化较小，而幅值小的经过屏摄后变化较大，前者例子 `(301, 299), (301, 300), (302, 304)`，后者例子 `(297, 305), (300, 296), (302, 303)` 。

### 3.2.2 消息嵌入

图 5 为消息嵌入过程示意图，

![]()

<center>图 5. 消息嵌入</center>

每个所选的 LSFR 作为独立的区域，向其中嵌入水印消息。相较于基于 DCT 的嵌入方法，消息比特嵌入区域中的每一个 8x8 或 4x4 子块中，本文提出的基于 DFT 方法将每个 LSFR 区域作为一个整体，故对于 crop 攻击具有更好的鲁棒性。

为了避免 LSFR 在朝向规范化过程中进一步畸变，作者设计了基于 DFT 系数属性的非旋转嵌入方法，另外为了提高提取水印的准确性，对 DFT 系数幅值进行了预处理。

1. 如图 5，在原图中提取外接 LSFR 的正方形，然后计算亮度，然后变换到 DFT 域。

2. 根据密钥，生成一个伪随机序列 $W=\lbrace w(i)|w(i)\in \{-1,1\}, i=0,\ldots, l-1\rbrace$，为了实现盲提取水印，$W$ 的嵌入半径 $R$ 需要设置为一个固定的值，于是 $W _ {RS}$ 的嵌入半径为

    $$R _ 1 = \text{round} \left(\frac {L _ 1} {L _ 0} \cdot R \right)$$

    其中 $L _ 1$ 是外接正方形 size，$L _ 0$ 是 (16) 式 即 LSFR  size 。

    