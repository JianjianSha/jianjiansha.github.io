---
title: 基于attention双流卷积网络的人脸欺骗检测
date: 2024-04-23 10:59:28
tags: face anti-spoofing
---

论文：[Attention-Based Two-Stream Convolutional Networks for Face Spoofing Detection](https://pureadmin.qub.ac.uk/ws/portalfiles/portal/174233508/Attention_Based_Two_Stream_Convolutional_Networks_for_Face_Spoofing_Detection.pdf)

源码：[Attention-Based-Two-Stream-Convolutional-Networks-for-Face-Spoofing-Detection](https://github.com/Vincent9797/Attention-Based-Two-Stream-Convolutional-Networks-for-Face-Spoofing-Detection)

# 1. 简介

本文提出 two stream 卷积网络（TSCNN），用于解决亮度变化引起人脸欺骗检测性能下降的问题。two stream 是指 RGB 颜色空间和多尺度 retinex 空间（MSR），前者包含了人脸细节纹理特征，但是对亮度敏感，后者是一个亮度不变性空间，但是包含较少了人脸细节信息，所以这两者正好互补。使用两个空间的图像喂给 TSCNN，从而学习到人脸欺骗检测的判别性特征。

本文方法本质上属于对单张图片的二分类。

为了应对人脸欺骗攻击，研究路线主要有 4 条：

1. 基于微纹理

    当攻击来自照片和视频时，局部微纹理是一个很好的线索

2. 基于图像质量

    fake 图片的质量通常较低

3. 基于动作（随时间）

    基于生理反应和基于物理动作

    由于视频回放攻击中也能呈现人脸部动作，所以基于动作的方法对视频回放攻击效果不好

4. 基于反射

    光照反射是一个可以利用的线索，真实人脸是 3D，而攻击通常是 2D，那么两者的光照反射肯定不同


本文提出的方法是基于微纹理（MTB）方法 。

# 2. 方法

## 2.1 MSR

multi scale retinex 算法

要了解多尺度 retinex 算法，首先了解单尺度 retinex（SSR）算法。对源图像使用高斯模糊，然后求对数得到对数图像，对源图像求对数图像，两者相减得到增强后图像。单尺度指的是高斯模糊核大小，显然 SSR 依赖于核大小，为了克服这种限制，提出 MSR，一个高斯模糊核 size 列表，分别进行 SSR，然后将增强图像求平均。

记源图像为 $S(x,y)$，Retinex 理论认为可以将 $S(x,y)$ 分成两个部分：反射部分 $R(x,y)$ 和亮度部分 $L(x,y)$，$R$ 和 $L$ 分别包含不同的频率成分，$R$ 主要包含高频，$L$ 主要包含低频，使用 (1) 式表示 Retinex，

$$S(x,y) = R(x,y) \cdot L(x,y) \tag{1}$$

$L(x,y)$ 表示亮度，由光源决定，$R(x,y)$ 表示人脸皮肤的反射成分，由人脸的性质决定。亮度与分类任务无关，人脸欺骗检测是一种特殊的二分类任务。所以 (1) 式的分离是有用的，分离出的反射成分可用于亮度无关的分类任务。

为了计算方便，将 (1) 式变为

$$\log S(x,y) = \log R(x,y) + \log L(x,y) \tag{2}$$

分别使用 $s, r, l$ 表示 (2) 式中的各项，那么 

$$r(x,y) = s(x,y) - l(x,y) \tag{3}$$

估算出 $l(x,y)$，就可以算出 Retinex 图像 $r(x,y)$ 。令 $l(x,y) = \log [S(x,y) * F(x,y)]$，其中 $*$ 表示卷积，那么就得到 SSR 的公式，

$$r(x,y) = s(x,y) - \log [S(x,y) * F(x,y)] \tag{4}$$

一种简单的 $F(x,y)$ 是使用高斯滤波，如下

$$G(x,y) = K e ^ {-(x ^ 2 + y ^ 2) / c} \tag{5}$$


选择 $K$ 使得

$$\int\int G(x,y) dx dy = 1 \tag{6}$$

这表示以 $(x,y)$ 为中心，半径为 $c$ 的正方形作为定义域，$G(x,y)$ 是一个概率分布。

于是 MSR 为

$$r _ {MSR} (x,y) = \sum _ {i=1} ^ k w _ i \{\log S(x,y) - \log [S(x,y)*G _ i (x,y)] \} \tag{7}$$

MSR 的特点：

1. 将图像分成亮度成分、反射成分和去除亮度后的反射成分
2. MSR 可视作一个优化的高通滤波器，保留图像的高频成分，这有利于活体检测

## 2.2 TSCNN

如图 1，

![]()
<center>图 1</center>

TSCNN 包含两个一样的网络（ResNet 18 或 MobileNet），输入分别对应 RGB 和 MSR 图像，输出的两个特征基于 Attention 进行融合，最后再使用一个线性层调整后，得到非归一化分类得分。

给定一个图像或一帧图像，首先使用 MTCNN 进行人脸检测和人脸对齐，得到的 RGB 图像喂给 RGB 的网络分支，另外，RGB 转为灰度图之后再转为 MSR 图像，如图 1 (B)，图 1 (C) 是基于 attention 的特征融合模块。

两个并列的特征提取分支结构也可以不同，但是输出的特征维度需要相同。

记特征为 $f _ {RGB}$ 和 $f _ {MSR}$，融合函数为 $F$，分类器为 $C$，那么优化目标为

$$\min _ w \frac 1 N \sum _ {i=1} ^ N l[C(F(f _ {RGB}, f _ {MSR})), y] \tag{8}$$

其中 $l$ 是损失函数，本文使用交叉熵损失函数。

**# Attention based fusion**

融合方法包含：得分平均、特征 concatenation，特征平均，特征最大池化和最小池化等。如图 1 (C)，本文使用基于 attention 的融合方法。

给定一个特征集 $\{f _ i | i = 1,\ldots, N\}$，学习特征的权重 $\{w _ i | i=1,\ldots, N\}$ ，那么融合为

$$v = \sum _ {i=1} ^ N w _ i f _ i \tag{9}$$

在人脸欺骗检测任务中，$N=2$。

本文作者没有直接学习权重 $w _ i$，而是学习了一个 kernel $q$，与特征维度相同，$q$ 用于提取特征中重要的成分，通过向量点乘实现，得到特征对应的重要程度 $d _ i$，

$$d _ i = q ^ {\top} f _ i \tag{10}$$

然后可以计算出权重为

$$w _ i = \frac {e ^ {d _ i}} {\sum _ j e ^ {d _ j}} \tag{11}$$

# 3. 实验

## 3.1 数据集

**# CASIA Face Anti-Spoofing**

训练集包含 20 个主题，测试集包含 30 个人。三种不同的照相机收集三种不同质量的图片和视频。每个人进行了眨眼，身体不保持静止。

三种类型的攻击：

1. 打印照片
2. cut 照片。攻击者将照片的眼部区域挖洞，然后攻击者眼睛透过小洞进行眨眼

    还有，将完整的照片放在 cutted 照片之后，然后上下移动完整的照片，以模拟眨眼动作

3. 视频攻击。在 ipad 上播放高清视频

另外两个数据集的介绍这里省略。

# 4. 代码讲解

**# 数据集加载**

训练数据集就是两个文件夹，一个包含 real 人脸数据，另一个包含 fake 人脸数据，标签分别为 `0` 和 `1` 。

训练过程比较简单，这里主要看看 test 代码。

```python
# test.py
from cv2 import CascadeClassifier

# opencv 的级联分类器，用于人脸检测
classifier = CascadeClassifier('haarcascade_frontface_default.xml')
model = attention_model(1, backbone='MobileNetV3', shape=(299, 299, 3))
model.load_weights('...')       # 加载模型

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'MPEG')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while True:
    ret, frame = cap.read()
    bboxes = classifier.detectMultiScale(frame) # 检测人脸位置
    for box in bboxes:
        x, y, width, height = box
        x2, y2 = int(x + 1.0*width), int(y + 1.2*height)
        x, y = int(x), int(y - 0.2*height)
        # height 维度上下 pad 0.2
        img = frame[y:y2, x:x2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (299, 299))

        # 构造 MSR
        new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        new_img = np.expand_dims(new_img, -1)
        new_img = automatedMSRCR(new_img, [10, 20, 30])
        new_img = cv2.cvtColor(new_img[:, :, 0], cv2.COLOR_GRAY2RGB)

        preds = model.predict([
            np.expand_dims(img/255., 0),
            np.expand_dims(new_img/255., 0)
        ])

        if preds[0][0] > 0.9:
            # real
            rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 1)
        else:
            # fake
            rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 1)
    out.write(frame)
```