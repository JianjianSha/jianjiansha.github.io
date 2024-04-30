---
title: Face Anti Spoofing Using Patch and Depth Based CNNs
date: 2024-04-24 15:12:57
tags: face anti-spoofing
---

论文：Face Anti-Spoofing Using Patch and Depth-Based CNNs

# 1. 简介

本文提出用于人脸反欺骗任务的双流 CNN 模型，此模型提取 local 特征以及整体的 depth map，其中 local 特征使得模型能够辨别图像中人脸区域中的 spoof 小块，将人脸图片划分为很多小块 patches，模型输出小块的得分，这是一个二分类得分，即判别每个小块是 spoofing 的概率得分。模型同时还输出整体的 depth map，表示每个像素点的 depth，如图 1，

![]()

local 特征是从人脸区域中的随机 patches 中提取，而 depth map 则使用整个人脸区域，depth 信息表明活体人脸是 3D 的，而 spoof 是 2D（平面的），这里考虑的攻击是照片打印或者电子屏显示。

使用两个 CNN（双流 CNN）独立地学习 local 特征和整体 depth 特征。第一个 CNN 是端到端训练，对人脸图片中随机选择的 patches 进行预测得分，将所有 patches 的得分进行平均，得到人脸图片的得分。第二个 CNN 则输出整个人脸的 depth map，并计算得到一个预测得分，最后融合两个得分，判断是活体还是 spoof 。

**# 图像深度估计**

估计一个 RGB 图像的深度是 CV 的一个基础任务。最近这几年使用深度学习，采用 RGB-D 数据集进行训练，也有弱监督。另外，人脸重建也可以看作是一种深度估计方法。

本文首次将深度估计引入人脸反欺骗任务。注意，这里的深度并非是人脸距离照相机的距离，而是人脸作为 3D 目标的高度，因为前者在不同的人脸像素点的相对变化很小，所以采用后者。

# 2. 方法

双路 CNN 包括：patch-based CNN 和 depth-based CNN，如图 2，

![]()

对于 depth-based CNN，需要使用全卷积（FCN）网络，即输出 spatial size 与输入 spatial size 相同。

两个分支的输出均可独立用于判断 live 或者 spoof，但是为了获得更好的性能，将两个预测得分进行融合，本文将融合输出作为 spoof 得分，即 spoof gt label 为 1，而 live gt label 为 0。

## 2.1 Patch-based CNN

使用 patch 而非整个人脸区域作为 CNN 输入的原因：

1. 增加训练样本

    例如 CASIA-FASD 仅包含 20 个主题，每个主题有 12 个 videos，由于视频帧之间的相似度较高，使用人脸进行训练必然会导致过拟合。

2. 使用整个人脸区域，需要进行 resize，这会破坏图片的判别性信息

3. 使用整个人脸区域，就要求 spoof 覆盖整个人脸区域，而实际上 spoof 可能仅覆盖局部区域

### 2.1.1 输入特征

CNN 网络可以从原始 RGB 图像中很好地学习到特征，但是本文作者发现，手动设计的特征也能让 CNN 网络受益，这是领域知识与 CNN 学习两者相结合，这在人脸反欺骗中很重要，如果没有领域知识，CNN 很可能学到非泛化信息，而无法学到真正的可以判别的信息。

本文使用 HSV 和 Ycbcr 两种颜色空间，因为考虑到 RGB 三个通道之间亮度和色度分离的不够好。另外，作者使用了其他输入特征用于 CNN 网络，例如 LBP map 和高频 patches。

对于 LBP map，作者使用 $LBP _ {8, 1}$（即半径为 1，采样点数量为 8）算子从人脸图片上提取纹理特特征，然后随机选择 patches。

对于高频 patches，从人脸图片中去掉低频信息，即，人脸图片 $I$ 减去低通滤波得到高频图像 $I - f_{lp}(I)$ ，低通滤波例如均值滤波、高斯滤波等。

不同输入特征的对比图如图 3 所示，

![]()

根据作者实验，所有提出的输入特征均是学习 CNN 的有用表征。

### 2.1.2 CNN 框架

patch-based CNN 框架如表 1 所示，

![]()

<center>表 1</center>

网络包含 5 个 conv 层和 3 个 fc 层，最终输出 2D 向量，表示二分类的非归一化预测得分。

## 2.2 Depth-based CNN

人脸图片中的高频信息对反欺骗检测非常关键，而 resize 则可能会破坏高频信息，因为为了处理不同 size 的人脸图片，使用 FCN，每一层输出 spatial size 均与原始 spatial size 相同。

那么输入 batch size 为 1？还是说使用 batch 内最大的 spatial size，然后靠左上角进行拷贝，其余位置使用 0 填充？

### 2.2.1 生成 depth label

记活体人脸的 3D shape 为 

$$A = \begin{pmatrix} x _ 1 & x _ 2 & \ldots & x _ Q
\\ y _ 1 & y _ 2 & \ldots & y _ Q
\\ z _ 1 & z _ 2 & \ldots & z _ Q
\end{pmatrix}$$

其中 $Q$ 是顶点数量，$z$ 表示深度值，$x,y$ 表示坐标值。

给定一个人脸图片（2D），3D 人脸模型匹配算法估计出 shape 参数 $\mathbf p \in \mathbb R ^ {1 \times 228}$ 和投影矩阵 $\mathbf m \in \mathbb R ^ {3 \times 4}$，然后使用 3DMM 模型计算出密集 3D 人脸 shape $A$，

$$A = \mathbf m \cdot \begin{bmatrix} \overline S + \sum _ {i=1} ^ {228} p ^ i  S ^ i \\\\ \mathbf 1 ^ {\top}\end{bmatrix} \tag{1}$$

其中 $\overline S$ 是形状均值，$S ^ i$ 是 PCA 形状，表示身份基和表情基。

空间中一点 $(x,y,z)$ 可由三个线性无关的基向量 $(1, 0, 0), (0, 1, 0), (0, 0, 1)$ 通过线性组合来表示，人脸形状类似地可由形状基通过线性组合来表示，所以模型估计值 $\mathbf p$ 实际上就是这些形状基的权重值。

因为原始的 2D 人脸图片不一定是正面，所以得到 3D 人脸形状 $S$ 之后还需要使用投影矩阵 $\mathbf m$ 将 $S$ 变换正面。

得到人脸 3D 形状矩阵 $A$ 之后，其中的 $z$ 值就是 depth，为了从离散的 z 值中获得平滑且一致的 depth map，使用 z-buffering 算法，计算出目标的“纹理”，作为深度信息插入到离散 z 值中。

不同 size 的人脸得到的 z 值范围也不同，所以 depth map $M$ 需要归一化，然后作为 CNN 训练的 gt label，本文使用 max-min 归一化。

图 4 是人脸 depth map 的样例，spoof 和 live 的背景区域的 depth 值均为 0 。对于打印的图片纸张攻击，纸张也可能存在弯曲，由于难以估计纸张弯曲程度，作者也当成是平面进行处理。

![]()

### 2.2.2 FCN 框架

使用 FCN 学习一个非线性变换 $\hat M=f(I;\Theta)$，其中 $I$ 是输入图像，$\hat M$ 是预测的 depth map。使用 HSV + Ycrcb 作为 FCN 的输入特征。

优化目标为

$$\arg \min _ {\Theta} J = ||f(I;\Theta) - M|| _ F ^ 2 \tag{2}$$

其中 $M$ 是 gt depth map。

### 2.2.3 用于分类的 depth map

为了利用 depth map 进行分类 live vs spoof，作者训练了一个 SVM 分类器，训练数据为训练集图片的 depth map 。

各个人脸图片的 depth map 其 size 可能不同，作为 SVM 的输入，输入向量维度需要相同，所以，将 depth map 均值池化为 $N \times N$ 的特征，然后转为向量作为 SVM 的输入。由于 resize depth map 为 $N \times N$ 大小会丢失一些信息，所以作者提议训练多个不同 $N$ 值的 SVM 分类器。

作者表示，给定一个人脸视频，还可以利用时间信息。对于活体视频，depth 随时间变化较小，而 spoof 视频的 depth 随时间变化较大，这是因为手持电子屏时手会无意识地动。所以先计算每一帧的 $N ^ 2$ 向量，然后计算标准差，每个维度独立计算，所以标准差也是一个 $N ^ 2$ 向量，然后每一帧为 $N ^ 2$ 的 depth 向量与标准差构成 $2N ^ 2$ 向量喂给 SVM，得到每一帧的预测得分，最后所有帧得分取平均，得到最终预测得分。

# 3. 代码分析

[源码](https://github.com/shicaiwei123/patch_based_cnn)

patch based CNN 与 depth based CNN 分开训练，前者比较简单，就是一个二分类模型的训练，这里我们看下后者。

## 3.1 depth cnn

### 3.1.1 数据加载

图像数据转换

```python
depth_test_transform = ts.Compose([
    ts.Resize((128, 128)),
    rgb2ycrcb(),
    ts.ToTensor(),
    ts.Normalize(mean=(0.56, 0.45, 0.58), std=(0.18, 0.04, 0.04))
])
```

获取一个样本数据

```python
# ImgPixelDataset
def __getitem__(self, idx):
    img_path = self.img_path_list[idx]
    face_img = cv2.imread(img_path)
    face_img_pil = Image.open(img_path).convert('RGB')

    if self.data_transform is not None: # 就是上面的 depth_test_transform
        face_img_trans = self.data_transform(face_img_pil)
        face_img = face_img_trans
    img_path_split = img_path.split('/')
    if img_path_split[-3] == 'spoofing' or img_path_split[-2] == 'spoofing':
        label = np.float32(np.zeros((self.pixel_size, self.pixel_size)))
    else:
        label = np.float32(np.ones((self.pixel_size, self.pixel_size)))
```

由于这个源码不是官方实现，所以这里没有使用论文中的 label generating 方法，而是直接对 spoof 全 0，对 live 全 1，有点简单粗暴。

代码比较简单，只有两个网络的训练，没有 SVM 以及融合，当然论文里面也没讲怎么融合，可能是两个得分的加权和？