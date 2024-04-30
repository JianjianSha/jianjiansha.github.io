---
title: 根据 landmark 计算对齐人脸的变换矩阵
date: 2024-04-16 14:13:10
tags: face alignment
mathjax: true
---

我们检测出人脸 bbox，然后再得到人脸 landmarks 之后，需要对齐人脸之后才能进行识别，那么如何根据 landmarks 对齐人脸呢？

# 1. 代码

设定对齐之后的人脸图像尺寸，通常有两种 `112x96` 和 `112x112`，对应的目标 landmarks 也预先设定好（这里以 5 地标点为例）

```python
imgSize1 = [112, 96]
imgSize2 = [112, 112]

coord5point1 = [[30.2946, 51.6963],  # 112x96的目标点
               [65.5318, 51.6963],
               [48.0252, 71.7366],
               [33.5493, 92.3655],
               [62.7299, 92.3655]]
coord5point2 = [[30.2946+8.0000, 51.6963], # 112x112的目标点
               [65.5318+8.0000, 51.6963],
               [48.0252+8.0000, 71.7366],
               [33.5493+8.0000, 92.3655],
               [62.7299+8.0000, 92.3655]]
```

根据 5 个点的对应关系，计算变换矩阵的代码为，

```python
def transformation_from_points(points1, points2):
    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)
    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1   # normalization，消除平移和缩放
    points2 /= s2   # normalization，消除平移和缩放
    U, S, Vt = numpy.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return numpy.vstack([numpy.hstack(((s2 / s1) * R,c2.T - (s2 / s1) * R * c1.T)),numpy.matrix([0., 0., 1.])])
 
def warp_im(img_im, orgi_landmarks,tar_landmarks):
    pts1 = numpy.float64(numpy.matrix([[point[0], point[1]] for point in orgi_landmarks]))
    pts2 = numpy.float64(numpy.matrix([[point[0], point[1]] for point in tar_landmarks]))
    M = transformation_from_points(pts1, pts2)
    dst = cv2.warpAffine(img_im, M[:2], (img_im.shape[1], img_im.shape[0]))
    return dst
```

# 2. 原理

平面上的变换为 平移 T、旋转 R 和缩放 s 三种变换的组合，所以目标是求解 T, R, s，使得源 landmarks 经过变换后非常接近目标 landmarks，目标函数为，

$$L = \sum _ {i=1} ^ N ||sR p _ i ^ {\top} + T - q _ i ^ {\top}|| ^ 2 \tag{1}$$

其中 $p _ i ^ {\top}$ 是源 landmark $A \in \mathbb R ^ {2 \times N}$ 的第 i 列，$q _ i ^ {\top}$ 是目标 landmarks $B \in \mathbb R ^ {2 \times N}$ 的第 i 列，$T \in \mathbb R ^ {2 \times 1}$，$R \in \mathbb R ^ {2 \times 2}$ 是一个正交矩阵即 $R ^ {\top} R = I$，$s$ 为标量。

对两边的点消除平移和缩放之后 (1) 式变成

$$L = ||RA- B|| _F \tag{2}$$

$$s.t. \ R ^ {\top} R = I$$

其中 $F$ 表示 Frobenius 范数。

使用 Ordinary Procrustes Analysis 解 (2) 式，得

$$\begin{aligned} & M =  BA ^ {\top}
\\ & \text{svd}(M) = USV
\\ & R = UV
\end{aligned} \tag{3}$$

缩放因子为 

$$s = \sigma(B) / \sigma(A) \tag{4}$$

其中 $\sigma(X)$ 表示计算 $X$ 的标准差，将矩阵 $X$ 中所有元素看作样本，即 $\sigma(X)$ 是一个标量。

计算 N 个 landmark 的中心 $\mu (A) =\frac 1 N \sum _ i ^ N p _ i ^ {\top}$，$\mu (B) =\frac 1 N \sum _ i ^ N q _ i ^ {\top}$，这两个量均为 $2 \times 1$ 的列向量。

将中心 $\mu (A)$ 经过旋转和缩放后，再平移得到中心 $\mu(B)$，于是平移向量为

$$T = \mu(B) - sR \mu (A) \tag{5}$$

综合以上各式，人脸对齐的变换矩阵（大小为 $3 \times 3$）为

$$\begin{bmatrix} s R & T \\ \mathbf 0 & 1
\end{bmatrix}$$

参考：https://matthewearl.github.io/2015/07/28/switching-eds-with-python