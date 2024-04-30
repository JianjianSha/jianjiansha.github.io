---
title: PSENet
date: 2024-04-02 10:28:48
tags: OCR
---

源码：[whai362/PSENet](https://github.com/whai362/PSENet.git)

# 1. 简介

场景文本检测的难点：

1. 基于 bbox 的检测器难以定位任意形状的文本，因为这些文本很难用一个矩形框包住
2. 像素级别的分割检测器难以将靠在一起的文本实例分离

本文提出 PSENet（渐进尺度放大网络），作为一个基于分割的检测器，每个文本实例有多个预测，通过缩小原始文本实例到不同的尺度，从而生成不同的预测，这就是渐进尺度方法算法，此算法可以分离靠在一起的文本实例，如图 1 (d) 所示，

![](/images/ocr/psenet_1.png)

具体而言，我们为每个文本实例分配多个预测分割区域，每个分割区域称作 “kernel”。 kernels 具有与原始文本实例相似的形状，并且均位于文本实例的中心，仅仅是尺度不同。使用渐进尺度放大算法获得最终的检测。此算法基于宽度优先搜索算法：

1. 从最小尺度的 kernel 开始，文本实例可以在这个尺度上被分离
2. 逐步检查更大的 kernel，直到找到可被区分的最大 kernel

渐进尺度放大有四层含义：

1. 最小 kernel 尺度上，文本实例可被区分，从而克服基于分割方法的缺点
2. 最大尺度的 kernel 也不可缺少，这样才能获得准确的预测（最小尺度上的预测的文本边界并不准确）
3. kernel 尺度从小到大逐步增大，网络容易学习
4. 尺度从小到大逐步增大，可以很准确地获取文本实例的边界

# 2. 方法

## 2.1 整体流程

如图 2 所示，

![](/images/ocr/psenet_2.png)

FPN 输出 4 个不同 scale 的特征，融合进 $F$（需要先进行适当上采样），$F$ 再输出到 n 个分支以得到不同的分割 $S _ 1, S _ 2, \ldots, S _ n$ 。$S _ i$ 是某个尺度上的所有文本实例的分割掩模。分割掩模的尺度由超参数决定，我们后面再介绍。$S _ 1$ 对应最小的尺度，$S _ n$ 对应最大的尺度。

## 2.2 渐进尺度放大算法

渐进尺度放大算法基于 BFS，如图 3 是一个例子，

![](/images/ocr/psenet_3.png)

这里假设有 3 个分割结果，$S=\lbrace S _ 1, S _ 2, S _ 3 \rbrace$ ，见图 3 中 (a) (e) (f)。开始时，根据最小 kernel $S _ 1$，得到 4 个不同的部分 $C = \lbrace c _ 1, c _ 2, c _ 3, c _ 4 \rbrace$，见图 3 (b) 中 4 个不同的颜色标注，现在我们检测到所有的文本实例的中心，然后我们逐步扩大 kernel，将 $S _ 2$ 中的像素合并进来，然后再合并 $S _ 3$ 的像素，两次尺度方法的结果在图 3 (c) 和 (d) 中显示，最后提取连接体，每个连接体使用不同的颜色标注。

尺度放大的过程在图 3 (g) 中说明。

![](/images/ocr/psenet_a1.png)

算法 1 中，$T$ 是所有表示文本前景的像素和像素 label 的集合，$P$ 是所有表示文本前景的像素集合，$T$ 比 $P$ 多了一个 label 维度的数据。$Q$ 是待遍历的文本前景像素集合。$S _ i [q]$ 表示像素点 $q$ 在尺度掩模 $S _ i$ 中是否是某个文本前景。

图 3 (g) 中红色方框表示一个冲突的点，冲突点仅被合并到第一个遇到这个点的 kernel 中。

## 2.3 标签生成

PSENet 生成分割结果 $S _ 1, \ldots, S _ n$，由于掩模尺度不同，那么 ground truths 也要具有不同的尺度。我们将原始文本实例进行不同尺度的缩小，得到不同尺度的 gt label。

![](/images/ocr/psenet_4.png)

图 4 (b) 中蓝色多边形表示原始文本实例，对应图 4 (c) 中最右的 gt label mask。使用 Vatti clipping 算法得到缩小的掩模：将原始多边形 $p _ n$ 缩小 $d _ i$ 个像素，得到缩小多边形 $p _ i$，如图 4 (a) 所示，然后根据每个缩小多边形 $p _ i$ 得到 0/1 掩模作为训练 gt label，记为 $G _ 1, \ldots, G _ n$ 。

考虑缩放比例为 $r _ i$，那么 $p _ n$ 与 $p _ i$ 的距离 $d _ i$ （图 4 (a) ）为

$$d _ i = \frac {\text{Area}(p _ n) \times (1 - r _ i ^ 2)}{\text{Perimeter}(p _ n)} \tag{1}$$

gt map $G _ i$ 的尺度比例 $r _ i$ 为

$$r _ i = 1 - \frac {(1-m)\times (n-i)}{n-1} \tag{2}$$

其中 $m$ 是最小 scale ratio，范围是 $m \in (0, 1]$ ，$n$ 是多尺度数量。

## 2.4 损失函数

$$L = \lambda L _ c + (1 - \lambda) L _ s \tag{3}$$

$L _ c$ 为完整（未缩小）文本实例的损失，$L _ s$ 为缩小文本实例的损失。

由于文本实例仅占图片的非常小的区域，如果使用 binary 交叉熵损失，这使得训练正负例不均衡，模型预测偏向非文本区域。为解决此问题，使用 dice 系数，

$$D(S _ i, G _ i)=\frac {2 \sum _ {x,y}(S _ {i,x,y} * G _ {i,x,y})}{\sum _ {x,y} S _ {i,x,y} ^ 2 + \sum _ {x,y} G _ {i,x,y} ^ 2} \tag{4}$$

dice 损失函数为

$$L _ i ^ {dice} = 1 - D(S _ i, G _ i)$$

采用在线难例挖掘 OHEM。记 OHEM 得到训练 mask 为 $M$，那么 

$$L _ c = 1 - D(S _ n \cdot M, G _ n \cdot M) \tag{5}$$

由于缩小的文本实例被原始的文本实例包围，我们忽略 $S _ n$ 中的非文本区域的像素，避免了重复，于是

$$L _ s = 1 - \frac {\sum _ {i=1} ^ {n-1} D(S _ i \cdot W, G _ i \cdot W)} {n-1}$$

$$W _ {x,y} = \begin {cases} 1 & S _ {n,x,y} \le 0.5 \\ 0 & \text{o.w.} \end{cases}$$

## 2.5 实现细节

backbone 中有 4 个特征，每个特征的 channel 均为 256，特征记为 $(P _ 2, P _ 3, P _ 4, P _ 5)$，融合 4 个特征

$$F = C(P _ 2, P _ 3, P _ 4, P _ 5)=P _ 2 || Up _ {\times 2} (P _ 3) || Up _ {\times 4} (P _ 4) || Up _ {\times 8} (P _ 5)$$

其中 $||$ 表示 concatenate 操作。$F$ 喂给 `Conv(3,3)-BN-ReLU`，channel 从 1024 降到 256，然后再通过 n 个 `Conv(1,1)-Up-Sigmoid` 生成 n 个分割预测 $S _ 1, \ldots, S _ n$ 。

设置 $n=6, m=0.5$，多尺度为 $(0.5, 0.6, \ldots, 1.0)$，损失均衡因子 $\lambda = 0.7$ 。OHEM 正负例比例为 `1:3` 。

数据增强方法：

1. rescale 图片，随机选择缩放比例 $(0.5, 1.0, 2.0, 3.0)$
2. 随机水平翻转图片和旋转 $[-10 ^ {\degree}, 10^{\degree}]$
3. 随机 crop 一个 $640\times 640$ 的区域
4. 使用均值和标准差对图片标准化。均值为 $(0.485, 0.456, 0.406)$，标准差为 $(0.229, 0.224, 0.225)$，通道顺序为 rgb

对四边形文本数据集，计算最小的举行 bounding box 作为最终预测，对弯曲文本数据集，使用 Ramer-Douglas-Peucker 算法生成任意形状的 bounding box 。

# 3. 实验

## 3.1 数据集

ICDAR 2015 包含 1500 个图片，其中 1000 用于训练，500 用于测试。文本区域使用四边形的 4 个顶点标注

ICDAR 2017 MLT 是一个大的多语言文本数据集，包含 7200 训练图片，1800 验证图片和 9000 测试图片。数据集由完整场景图片构成，涉及 9 个语言。使用四边形的 4 个顶点标注文本区域。

SCUT-CTW 1500 用于弯曲文本检测。包含 1000 训练图片和 500 测试图片。文本使用 14 顶点的多边形标注。


## 3.2 OHEM

在线难例挖掘

根据前向传播，得到分割预测 $S _ 1, S _ 2, \ldots, S _ n$，所选样本点满足条件：

1. 正例，gt=1
2. 负例，gt=0，但是对应的预测值为 top neg_num，这里 neg_num 为负例数量

    gt=0 而预测值大表示难例，因为预测值应该小才对


使用未经缩小的预测 $S _ n$ 和 gt label $G _ n$ 进行 OHEM 。

## 3.3 渐进尺度放大算法

此算法实际上用于 test 阶段，因为此阶段没有 gt label，我们需要根据预测的分割 mask，计算出所有的连接体，一个连接体就表示一个文本实例。

相关代码为

```python
# psenet.py
# forward 方法
# 将预测分割 map 插值上采样到输入图片 size
det_out = self._upsample(det_out, img.size(), 1)
# det_out 就是 Sn...S1，非归一化
# img_metas 保存了输入图片size
det_res = self.det_head.get_results(det_out, img_metas, cfg)
```

重点位于 `get_results` 这个方法，

```python
# psenet_head.py

def get_results(self, out, img_meta, cfg):
    '''
    out: (b, 7, h, w) Sn...S1, 非归一化
    '''
    outputs = dict()
    # 归一化 Sn，作为预测得分
    score = torch.sigmoid(out[:, 0, :, :])

    # 预测 kernel，由于非归一化，故 >0 就表示预测得分 >0.5
    kernels = out[:, :cfg.test_cfg.kernel_num, :, :] > 0
    text_mask = kernels[:, :1, :, :]

    # Sn-1...S1 的正例应该在 Sn 之中，故
    # 在 Sn 之外的，将其纠正为负例
    kernels[:, 1:, :, :] = kernels[:, 1:, :, :] * text_mask

    # Sn 分割得分 map 转为 numpy 数据
    score = score.data.cpu().numpy()[0].astype(np.float32)
    # 注意这里 batch size 为 0
    kernels = kernels.data.cpu()[0].astype(np.float32)

    # pse 就是算法 1 的方法实现
    label = pse(kernels, cfg.test_cfg.min_area) # min_area 配置为 16
    # label: (h, w) 是算法 1 的输出
    # label: 元素值 0 1 2 表示这是第 i 个连通体
    # 原始图片大小
    org_img_size = img_meta['org_img_size'][0]
    # 短边对齐到 736
    img_size = img_meta['img_size'][0]
    # 连通体数量，也就是预测文本实例的数量
    label_num = np.max(label) + 1

    scale = (float(org_img_size[1]) / float(img_size[1]),
             float(org_img_size[0]) / float(img_size[0]))

    bboxes = []
    scores = []
    for i in range(1, label_num):
        ind = label == i
        # 第 i 个连通体的位置坐标 (2, n)，转置后 (n, 2) 其中 n 表示此连通体内像素量
        points = np.array(np.where(ind)).transpose((1, 0))

        if points.shape[0] < cfg.test_cfg.min_area: # 如果小于 16，
            label[ind] = 0  # 0 表示背景这个连通体
            continue

        score_i = np.mean(score[ind])
        if score_i < cfg.test_cfg.min_score:# 小于 0.85
            label[ind] = 0  # 置信度不够高，也认为是背景
            continue

        if cfg.test_cfg.bbox_type == 'rect':
            # 计算包含连通体内所有像素的最小矩形框
            rect = cv2.minAreaRect(points[:,::-1])
            # 计算这个矩形的 4 个角点，注意矩形不一定 axes aligned
            bbox = cv2.boxPoints(rect) * scale
        elif cfg.test_cfg.bbox_type == 'poly':
            binary = np.zeros(label.shape, dtype=np.uint8)
            binary[ind] = 1
            # 绘制等高线（的点）
            _, contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bbox = contours[0] * scale
        bbox = bbox.astype(np.int32)
        bboxes.append(bbox.reshape(-1))
        scores.append(score_i)  # 这个预测文本实例的预测得分

    outputs.update(dict(
        bboxes=bboxes,
        scores=scores
    ))
    return outputs
```

训练阶段，对图片进行数据增强之后，再 crop 出一个预设 size 的区域，论文中说明是 $640 \times 640$，代码中有的是 $760 \times 736$，如果图片 size 小于这个值，可以进行边缘 padding（镜像，反射等 opencv 常用的 padding 方法）

测试阶段，batch size 为 1，并且没有训练阶段那个将图片 crop 出一个预设的固定 size 的操作，但是有一个将图片短边对齐到 736 的操作，长边则是根据相同的缩放比例进行缩放，同时保证是 32 的整数倍，因为 backbone 下采样率为 32，如长边不满足 32 的整数倍，则

```python
if h % 32 != 0:
    h = h + (32 - h % 32)
if w % 32 != 0:
    w = w + (32 - w % 32)
img = cv2.resize(img, dsize=(w, h))
```

也就是说，测试阶段，将短边缩放到 736，长边按类似的比例缩放到一个靠近 32 的倍数的值，然后根据预测的文本实例 bbox，再缩放回原始图片 scale 即可。