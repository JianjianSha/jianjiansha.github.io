---
title: A Morphable Model For The Synthesis Of 3D Faces
date: 2024-04-25 17:13:19
tags: 3D reconstruction
---

论文：A Morphable Model For The Synthesis Of 3D Faces

本文提出一种人脸 3D 建模的方法。

# 1. 数据集

通过激光扫描 200 个年轻人（100 男性和 100 女性）的头部，得到头部结构数据的柱面表示，得到人脸表面点的径向距离 $r(h, \phi)$，均匀采样得到 512 个角度 $\phi$ 和 512 个高度 $h$，另外同时记录了每个点的颜色 $R(h, \phi), \ G(h, \phi), \ B(h, \phi)$ 。所有人脸均无化妆、挂饰和面部毛发，戴上浴帽进行扫描，后面再通过程序去掉浴帽。 预处理包括一个竖直切割去掉耳朵后面的部分，以及一个水平切割去掉肩部以下的部分，然后进行空间归一化，使得脸部朝向标准方向以及位于标准位置，经过处理之后，一个人脸大约由 70000 个点表示，每个点包含形状和纹理（即颜色）两部分。

# 2. 可变形 3D 人脸模型

使用一个形状向量表示人脸的几何特征，

$$S=(X _ 1, Y _ 1, Z _ 1, \ldots, X _ n, Y _ n, Z _ n) ^ {\top}$$

使用一个纹理向量表示人脸的纹理特征，

$$T=(R _ 1, G _ 1, B _ 1, \ldots, R _ n, G _ n, B _ n) ^ {\top}$$

使用大小为 $m$ 的数据集来建立可变性人脸模型。新人脸的形状和纹理可表示为

$$S _ {mod} = \sum _ {i=1} ^ m a _ i S _ i, \quad T _ {mod} = \sum _ {i=1} ^ m b _ i T _ i$$

其中 $\sum _ {i=1} ^ m a _ i = \sum _ {i=1} ^ m b _ i = 1$ 。

这意味我们使用一个线性模型建模人脸。

但是 $a _ i, \ b _ i$ 显然不能随便取值，否则生成的人脸 $S _ {mod}$ 可能不合理，所以要根据已有的数据集来估计 $a _ i, \ b _ i$ 的分布。实际上这里并不是估计 $a _ i$ 的分布，因为 $S _ i, \ i=1,\ldots, 3n$ 之间不是相互正交的，可以使用 PCA，各个主成分之间是相互正交的。

根据数据集计算人脸形状和纹理的均值向量，$\overline S$ 和 $\overline T$，然后计算 $\Delta S _ i = S _ i - \overline S, \ \Delta T _ i = T _ i - \overline T$，再计算协方差 $C _ S, \ C _ T$，

$$C _ S = \frac 1 m \sum _ {i = 1} ^ m \Delta S _ i \Delta S _ i ^ {\top}
\\C _ T = \frac 1 m \sum _ {i = 1} ^ m \Delta T _ i \Delta T _ i ^ {\top}$$

$\Delta S _ i, \ \Delta T _ i$ 维度均为 $3n$，故 $C _ S, \ C _ T$ 维度为 $3n \times 3n$，$m=200$，可以认为 $3n > m$，故矩阵的秩 $\text{rank}(C _ S) \le m-1, \ \text{rank}(C _ T) \le m-1$。

于是

$$\Delta S _ {model} = \sum _ {i = 1} ^ {m-1} \alpha _ i s _ i$$

而 $\Delta S _ {model} = S _ {model} - \overline S$，于是

$$S _ {model} =\overline S + \sum _ {i = 1} ^ {m-1} \alpha _ i s _ i, \quad T _ {model} =\overline T + \sum _ {i = 1} ^ {m-1} \beta _ i t _ i \tag{1}$$

其中 $s _ i, \ t _ i$ 为特征向量，按对应的特征值降序排列。

特征值是降序排列的，所以相当于使用了前面 $m-1$ 个主成分。

我们使用 **多维高斯分布模型** 作为参数 $\alpha _ i, \ \beta _ i$ 的分布，拟合这 200 个人脸数据集，概率分布为

$$p (\vec {\alpha}) \sim \exp \left[-\frac 1 2 \sum _ {i=1} ^ {m-1} (\alpha _ i / \sigma _ i) ^ 2\right] \tag{2}$$

其中 $\sigma _ i ^ 2$ 为 $C _ S$ 的特征值。$\vec \beta$ 的分布类似处理。

这样人脸形状和纹理均有 $m-1$ 个自由度，修改每个参数就相当于对每个特征向量子空间独立的修改，一个子空间可能对应眼睛、鼻子或者其他周围区域，从而修改人脸。

## 2.1 面部属性

人脸可变模型的参数 $\alpha _ i, \ \beta _ i$ 并不对应人类语言所描述的面部属性，例如面部的女性气质或者胖瘦程度。

这里我们为样本手动设计标签，这些标签描述了面部属性，然后找到一个方法与人脸可变模型的参数联系起来。在人脸空间中，定义形状和纹理向量使得当增加到人脸或者从人脸中减去，其实就是对某个特别的属性进行操作而其他属性保持不变。

使用面部表情，那么形状和纹理的表情模型为 

$$\Delta S = S _ {expression} - S _ {neutral}, \ \Delta T = T _ {expression} - T - {neutral}$$

其中 expression 下标表示有表情，neutral 下标表示无表情。

这样就得到数据集中每个人脸样本的 $\Delta S$ 和 $\Delta T$ 值，使用与上面相同的方法（PCA）建模，得到人脸表情的三维模型。

面部表情是统一的，即每个个体的面部表情没有什么差异，面部属性则不同，每个个体的差异很明显，下面介绍如何建模来表示性别、面部丰满程度、眉毛的黑度、是否双下巴以及是钩鼻还是凹鼻。

数据集 $(S _ i, T _ i)$，手动打标签 $\mu _ i$ 表示属性的显著程度，然后计算加权和，

$$\Delta S = \sum _ {i=1} ^ m \mu _ i (S _ i - \overline S), \quad \Delta T = \sum _ {i=1} ^ m \mu _ i(T _ i - \overline T)$$

然后对每个个体的人脸，加上或者减去 $(\Delta S, \Delta T)$ 的数倍，可以对人脸进行面部属性的修改。对于二进制属性，例如性别，即属性类别为 A 和 B，对数量为 $m _ A$ 的样本类别 A 赋值为常量 $\mu _ A$，对数量为 $m _ B$ 的样本类别 B 赋值为常量 $\mu _ B$。

但是面部属性很多，对每一种属性，均人工打 $m$ 个标签，这是很困难的。我们使用一个函数 $\mu(S, T)$ 表示人脸 $(S, T)$ 的属性显著程度，我们假设 $\mu (S, T)$ 是一个线性函数，这样最优解在以下条件达到最小时满足，

$$||\Delta S|| _ M ^ 2 = \langle \Delta S, C _ S ^ {-1} \Delta S \rangle, \quad ||\Delta T|| _ M ^ 2 = \langle \Delta T, C _ T ^ {-1} \Delta T \rangle$$

# 3. 可变形模型与图像的匹配

本文提出的是一种自动将可变形人脸模型与一个或多个图像匹配的算法。3D 模型的系数根据一组渲染参数进行优化，使得生成的图像与输入图像的差距越来越小。

模型参数为形状与纹理的系数 $\alpha _ j, \ \beta _ j, j=1,\ldots, m-1$。渲染参数 $\vec \rho$ 表示相机参数，包括方位角，仰角，目标尺度，图像平面的旋转角度和偏移，背景光强度 $i _ {r,amb}, i _ {g,amb}, i _ {b,amb}$，直射光强度 $i _ {r,dir}, i _ {g,dir}, i _ {b,dir}$。为了处理在不同条件下拍摄的照片，$\vec \rho$ 还包含了颜色对比度，红绿蓝三通道的偏移和增益，其他参数如相机距离等由用户估计后确定。

根据参数 $\alpha, \beta, \vec \rho$，生成图像为

$$$$

# 4. 代码讲解

如果之前不了解人脸 3D 重建，那么看完论文还是一头雾水，其实本文的主要思想就是根据 200 个人脸数据（形状和纹理），建立人脸模型 (1) 式，这是平均人脸的模型。然后对于给定一张图片，将人脸模型与这张图片进行匹配，这样就得到输入图片对应的人脸 3D 模型，最后调整参数 $\alpha _ i, \ \beta _ i$，相当于对这个人脸的性别、年龄等面部特征进行调整。

为了弄清楚，这里大概讲解一下[代码](https://github.com/icygurkirat/3DMM-Matlab)。

## 4.1 数据集

在讲源码之前，首先认识一下数据集。

[**Basel Face Model**](https://faces.dmi.unibas.ch/bfm/index.php?nav=1-2&id=downloads)

- 3D 可变形人脸模型（形状与纹理）
- 性别、身高、体重和年龄的属性向量
- ...

此数据集中人脸的顶点数量为 53490 个，也就是说，人脸形状和纹理向量维度均为 `53490*3=160470`。

加载数据

```python
from scipy.io import loadmat

original_BFM = loadmat('01_MorphableModel.mat')
shapePC = original_BFM['shapePC']   # 形状主成分，(199, 160470) (1) 式中的 s_i
shapeEV = original_BFM['shapeEV']   # 形状特征值，(199,) (2) 式中的 \sigma_i^2
shapeMU = original_BFM['shapeMU']   # 平均人脸， (160470,) (1) 式中的 \overline S
... # 纹理与形状类似
```

BFM 中已经计算好了平均人脸，以及 PCA 特征值和特征向量，这样就已经得到平均人脸的 3D 模型。

`EditorApp.m` 是基于平均人脸的 3D 模型进行微调，得到微调后的人脸。GUI 操作界面如图 1，

![]()

