---
title: yolov5-face 论文解读
date: 2024-04-15 17:05:04
tags: face detection
---

论文：[YOLO5Face: Why Reinventing a Face Detector](https://arxiv.org/abs/2105.12931)

源码：[deepcam-cn/yolov5-face](https://www.github.com/deepcam-cn/yolov5-face)

# 1. 简介

本文采用 yolov5 作为人脸检测器，基于 yolov5，设计了一系列不同 size 的模型，从 large、medium 到 small。另外还采用了 ShuffleNetV2 作为 backbone，这在手机上取得了 SOTA 结果。

# 2. 方法

## 2.1 网络结构

使用 YOLOv5 作为 baseline，并引入一些修改，用于检测较小和较大的人脸。

YOLOv5-face 如图 1，包含了 backbone，neck 和 head。YOLOv5 中使用 CSPNet 作为 backbone，neck 部分使用了 SPP 和 PAN 聚合特征，head 包含了回归和分类两个分支。

![](/images/face/yolov5_face_1.png)
![](/images/face/yolov5_face_2.png)

<center>图 1 </center>

本文仅考虑 VGA 分辨率图像，即图像长边 scale 到 640，短边则根据相同比例进行缩放，并调整到 32 的整数倍（如果未使用 P6 特征）或者 64 的整数倍（如果使用 P6 特征）。

## 2.2 关键修改

**# 地标回归头**

增加人脸地标回归头到 YOLOv5 网络，对这个回归头使用 Wing Loss。
增加这个回归头使得人脸检测更加有用，因为地标有很多应用。使用地标监督让检测结果更准确。

**# Stem block 替换 Focus layer**

这种替换增加网络泛化能力，降低计算复杂度，同时不会损失性能。

**# 修改 SPP**

SPP block 使用更小的 kernel，从而使得 YOLOv5 更适用人脸检测。

原生 YOLOv5 的 SPPF block 中 kernel 为 `13x13, 9x9, 5x5` ，本文修改为 `7x7, 5x5, 3x3` 。



**# 增加 P6 特征**

P6 特征的 stride 为 64，更适用于大人脸。


**# 数据增强**

目标检测中的数据增强方法如 up-down flipping 和 Mosaic 。随机裁剪有助于性能提升。

**# 使用 ShuffleNetV2**

ShuffleNetV2 模型更小，在手机等设备上有 SOTA 性能。

ShuffleNet block 使用了两个操作：pointwise 分组卷积（分组数量为 channel 大小）；channel shuffle 。

## 2.3 地标回归

地标是人脸的重要特征，用于人脸对齐，人脸识别，人脸表达分析以及年龄分析等。传统地标包含 68 个点，在 MTCNN 中简化为 5 个点，自此 5 个点地标在人脸识别中广泛应用。

添加地标回归头到 YOLO5Face 中，输出的地标预测用于人脸对齐，然后下一步可以进行人脸识别。

地标回归的通用损失函数是（预测坐标与 gt 坐标） L2，L1 或者 smooth-L1。 MTCNN 使用 L2 损失函数，但是 L2 对小错误不敏感（在接近 0 时曲线平缓），使用 Wing-loss 解决这个问题，

$$\text{wing}(x) = \begin{cases} w \cdot \log(1+|x|/e) & x < w \\ |x|-C & \text{o.w.}\end{cases} \tag{1}$$

非负值 $w$ 指定非线性部分的范围 $(-w, w)$ 。$e$ 限制了曲率。

$$C = w - w \cdot \log (1 + w / e)$$

记地标点位 $s=\lbrace s _ i \rbrace$，gt label 为 $s'=\lbrace s _ i \rbrace$，其中 $i=1,2,\ldots,10$ ，那么 Wing loss 为

$$\text{loss} _ L (s) = \sum _ i \text{wing}(s _ i - s _ i ') \tag{2}$$

记 YOLOv5 的检测损失为 $\text{loss} _ O$，包含 bbox，class 和 probability 三种，那么总的损失为

$$\text{loss}(s) = \text{loss} _ O + \lambda _ L \cdot \text{loss} _ L \tag{3}$$

# 3. 实验

## 3.1 数据集

**# WiderFace**

[下载地址](http://shuoyang1213.me/WIDERFACE/)

数据集的图片来源是WIDER数据集，从中挑选出了 32,203 图片并进行了人脸标注，总共标注了 393,703 个人脸数据。对于每张人脸都附带有更加详细的信息，包扩 blur（模糊程度）, expression（表情）, illumination（光照）, occlusion（遮挡）, pose（姿态）。

标注文件解析：

- 第一行File name为图片的路径名称
- 第二行Number of bounding box为该图片中标注人脸的个数
- 接下来的Number of bounding box行信息为每个人脸的详细信息x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose

    - 其中x1, y1, w, h代表人脸边界框的左上角x、y坐标，以及宽、高信息，注意这里是绝对坐标。
    - blur代表人脸的模糊程度，0代表清晰，1代表有点模糊，2代表很模糊。
    - expression代表表情，0代表正常的表情，1代表夸张的表情。
    - illumination代表光照条件，0代表正常光照，1代表极端的光照条件。
    - invalid基本都是很小，很难分辨的人脸（不仔细看，看不出来的那种），个人觉得在使用时可以忽略掉invalid的人脸即为1的情况。
    - occlusion代表人脸的遮挡程度，0代表没有遮挡，1代表部分遮挡（1%-30%），2代表严重遮挡（30%以上）。
    - pose代表人脸的姿态，0代表典型姿态，1代表非典型姿态。论文中给出的解释Face is annotated as atypical under two conditions: either the roll or pitch degree is larger than 30-degree; or the yaw is larger than 90-degree.。

关键点标签下载地址

链接: https://pan.baidu.com/s/1otGaQyCCbVi3w6EIyyRFXg 提取码: wede

例如：

```sh
# 0--Parade/0_Parade_marchingband_1_849.jpg
449 330 122 149 488.906 373.643 0.0 542.089 376.442 0.0 515.031 412.83 0.0 485.174 425.893 0.0 538.357 431.491 0.0 0.82
# 0--Parade/0_Parade_Parade_0_904.jpg
361 98 263 339 424.143 251.656 0.0 547.134 232.571 0.0 494.121 325.875 0.0 453.83 368.286 0.0 561.978 342.839 0.0 0.89
```
以第一个例子为例
```sh
449 330 122 149 #表示box（x1, y1, w, h）
```

接着是5个关键点信息，分别用0.0隔开 或者1.0分开
```sh
488.906 373.643 0.0
542.089 376.442 0.0
515.031 412.83 0.0
485.174 425.893 0.0
538.357 431.491 0.0
```

最后是 1个置信度值。

# 4. 代码分析

我们直接看检测头的代码，

```python
# yolo.py

class Detect(nn.Module):
    stride = None       # 动态计算
    export_cat = False  # onnx export cat output

    def __init__(self, nc=1, anchors=(), ch=()):
        super(Detect, self).__init__()
        self.nc = nc    # 分类数量，这里只检测是否有人脸，所以分类数量为 1
        self.no = nc + 5 + 10   # 输出 channel，包含分类数量 1，x1y1x2y2+人脸置信度，以及 5 个 landmark 点坐标
        self.na = len(anchors[0]) // 2 # 每个 location 处 anchor 数量
        # ch: 各个 level 的输入特征的 channel 列表
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
```

其余部分可参考 [yolov5](/2023/08/09/obj_det/yolov5) 的解读。

再看一下 test 部分如何处理输出的，

```python
# yolo.py/Detect

def forward(self, x):
    z = []  # inference output
    for i in range(self.nl):    # 特征层数
        x[i] = self.m[i](x[i])  # 使用 conv1x1，转换 channel 数量
        bs, _, ny, nx = x[i].shape
        x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        # (bs, na, ny, nx, no)
        if not self.training:
            self.grid[i] = self._make_grid(nx, ny).to(x[i].device)  # 构造 grid
            # 由于 anchor 仅仅是 3 个 box，加上 grid，才得到所有 location 的 3 个 anchor
            y = torch.full_like(x[i], 0)
            # no 分别表示 xywh，conf，landmark，cls
            class_range = list(range(5)) + list(range(15, 15+self.nc))
            # xywh conf cls 归一化
            y[..., class_range] = x[i][..., class_range].sigmoid()
            y[..., 5:15] = x[i][..., 5:15]  # landmark coords
            # 参考 yolov5 文章的公式 (2)，计算出 xy wh
            y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]

            for j in range(5):
                # grid 中心坐标 + offset 得到 landmark point 坐标
                y[..., 5+j*2:7+j*2] = y[..., 5+j*2:7+j*2] * self.anchor_grid[i] + self.grid[i] * self.stride[i]
            z.append(y.view(bs, -1, self.no))   # reshape as (bs, na*ny*nx, no)
    return x if self.training else (torch.cat(z, 1), x)
```

接下来使用非极大抑制，然后再根据 conf 阈值进行过滤，

```python
# detect.py/detect

pred = model(img)[0]
pred = non_max_suppression_face(pred, conf_thres, iou_thres)
for i, det in enumerate(pred):  # 遍历批次
    # 文件路径，原始图像（注意与模型输入图像数据 img 不同，img 经过 size 调整）
    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
    if len(det):
        # 调整 xywh 尺度
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        # 调整 landmark 点坐标尺度
        det[:, 5:15] = scale_coords_landmark(img.shape[2:], det[:, 5:15], im0.shape).round()
        for j in range(det.size()[0]):  # 遍历每一个人脸框
            xyxy = det[j, :4].view(-1).tolist()
            conf = det[j, 4].cpu().numpy()
            landmarks = det[j, 5:15].view(-1).tolist()
            class_num = det[j, 15].cpu().numpy()

            im0 = show_result(im0, xyxy, conf, landmarks, class_num)
```

注意模型输出的坐标是基于模型输入 size，但是模型输入 size 不等于原始图像 size，所以上面代码中还进行了坐标尺度调整。

