---
title: 使用可微二值化的实时场景文本检测
date: 2024-04-03 08:36:42
tags: ocr
---

源码：[MhLiao/DB](https://github.com/MhLiao/DB.git)

# 1. 简介

基于分割的场景文本检测，在得到预测分割结果（分割得分 map）后需要进行 post-processing 以得到一个个预测文本实例的边界框，例如 [psenet](2024/04/02/ocr/psenet.md) 中的算法 1 。

本文提出可微二值化（Differentiable Binarization, DB），将这个二值化处理放在分割网络中，实现自适应设置用于二值化的阈值，简化了 post-processing，并增强检测性能。

场景文本检测面临的问题：文本可能是任意 scale，任意 shape，任意 oriention。基于分割的方法可以很好的解决这些问题，但是需要比较复杂的 post-processing：将像素级预测结果进行分组得到文本实例，通常 post-processing 是比较耗时的，降低了检测速度。

大多数场景文本检测方法使用如图 1 中蓝色线的方法即，设置一个阈值，然后将分割结果二值化，再根据二值化 map 使用 post-processing 得到检测结果。本文则是图 1 中红色线的方法，根据网络输出得到阈值 map，然后再进行二值化，由于阈值自适应，模型可以更好地区分前景背景。由于标准的二值化函数不可求导，所以本文使用一个近似的可导函数代替，称为 differentiable binarization (DB) 。

![](/images/ocr/db_1.png)

<center>图 1</center>

# 2. 方法

网络结构如图 2，

![](/images/ocr/db_2.png)
<center>图 2</center>

输入图像首先经过特征金字塔的 backbone 网络，类似于 FPN 的结构，然后，金字塔特征经过适当的上采样得到相同的 spatial size，然后 concat 之后经过 conv 将 channel 降低，得到特征 F，利用特征 F 得到预测概率 map P 和阈值 map T ，然后根据 P 和 T 得到近似的二值化 map $\hat B$ 。

训练阶段对概率 map ，阈值 map 和近似二值 map 进行监督。测试阶段，根据近似二值 map 获得边界框，或者对概率 map 使用 box formulation 模块得到边界框。

## 2.1 二值化

**# 标准二值化**

概率 map $P \in R ^ {H \times W}$，二值化过程为

$$B _ {ij} = \begin{cases} 1 & P _ {ij} \le t \\ 0 & \text{o.w.} \end{cases} \tag{1}$$

其中 $t$ 是一个阈值。值为 1 表示是一个文本像素点。

**# 可微二值化**

$$\hat B _ {ij} = \frac 1 {1 + \exp(-k(P _ {ij} - T _ {ij}))} \tag{2}$$

其中 $T$ 是自适应阈值 map，有网络学习得到，$k$ 为放大因子，根据经验设置 $k=50$ 。可微二值化与标准二值化类似，图 3 (a) 是两者的对比，

![](/images/ocr/db_3.png)
<center>图 3. (a) 标准二值化与可微二值化对比；(b) l+ 的导数；(c) l- 的导数</center>

可微二值化与自适应阈值 map 不但有助于区分文本区域和背景区域，还可以区分两个靠近的文本区域。

DB 可以改善性能，这可以通过梯度的反向传播解释。以二值交叉熵损失为例，定义 DB 函数

$$f(x)=\frac 1 {1+ \exp(-kx)}, \quad x = P _ {ij} - T _ {ij}$$

正负例的损失分别记为 $l _ +, \ l _ -$，于是

$$l _ + = - \log \frac 1 {1 + \exp(-kx)}$$

$$l _ - = -\log \left( 1 - \frac 1 {1 + \exp(-kx)} \right)$$

对应导数为

$$\frac {\partial l _ +}{\partial x} = -kf(x) e ^ {-kx}$$

$$\frac {\partial l _ -}{\partial x} = kf(x)$$

导数如图 3 (b) 和 (c) 所示，可以观察到：

1. 梯度可以通过 $k$ 得到增强
2. 梯度的幅值在预测错误时明显大，即 $x < 0$ 时的$l _ +$，以及 $x > 0$ 时的 $l _ -$ 。

## 2.2 自适应阈值

阈值 map 与 文本边界 map[<sup>1</sup>](#refer-anchor-1) 看起来有点类似，但是两者的出发点以及用途不同。是否对阈值 map 进行监督对结果的影响如图 4 所示，即使不对阈值 map 监督，阈值 map 也能强调文本边界区域，所以我们对阈值 map 使用边界监督。文本边界 map 用于分离文本实例，本文的阈值 map 用做二值化的阈值。

![](/images/ocr/db_4.png)
<center>图 4. (a) 原图；(b) 概率图；(c) 阈值 map，没有监督；(d) 阈值 map，有监督</center>

## 2.3 可变形卷积

可变形卷积有灵活的感受野，这对场景文本检测非常有利。在 ResNet-18 或 ResNet-50 backbone 中的 stage `conv3,4,5` 中的所有 3x3 卷积替换为可变形卷积。

## 2.4 标签生成

借鉴 [psenet](2024/04/02/ocr/psenet.md) 中的标签生成方法。

一个文本图片中文本区域使用多边形表示，

$$G = \lbrace S _ k \rbrace _ {k=1} ^ n$$

其中 $n$ 是多边形顶点数，$S _ k$ 是多边形顶点。正例区域则通过对多边形缩小得到，记为 $G _ s$，$G _ s$ 与 $G$ 的边距 $D$ 满足

$$D = \frac {A(1 - r ^ 2)} L$$

其中 $A$ 为 $G$ 的面积，$r$ 为缩小比例，$L$ 为 $G$ 的周长。

类似地，对 $G$ 进行膨胀，碰撞边距依然为 $D$，记膨胀后多边形为 $G_d$，那么 $G _ d$ 与 $G _ s$ 之间的空间（gap）就是文本区域的 border。

阈值 map 的 gt label：（没理解，看源码）

## 2.5 优化

因为有 3 个监督，故对应有 3 个损失：概率 map 的损失 $L _ s$，二值 map 损失 $L _ b$ 和阈值 map 的损失 $L _ t$，总损失为三者加权和，

$$L = L _ s + \alpha L _ b + \beta L _ t$$

平衡因子设置为 $\alpha = 1.0, \ \beta = 10$ 。

使用二值交叉熵 BCE 计算 $L _ s$ 和 $L _ b$ 。为了平衡正负例样本，使用了难负例挖掘（即，选择 gt label 为 0，但是预测分割得分较高的 topn 个负例，n 为预设的负例数量），BCE 损失为

$$L _ s = L _ b = \sum _ {i \in S _ l} y _ i \log x _ i + (1 - y _ i) \log (1 - x _ i)$$

其中 $x _ i$ 为概率 map 或二值 map 上某点。$S _ l$ 为样本集，其中正负例比例为 $1:3$ 。

$L _ t$ 为膨胀多边形 $G _ d$ 中阈值 map 的预测和 label 之间的 L1 距离和，

$$L _ t = \sum _ {i \in R _ d} |y _ i ^ * - x _ i ^ *|$$

其中 $R _ d$ 是 $G _ d$ 中像素点的索引集合，$y ^ *$ 是阈值 map 的 label，$x ^ *$ 是阈值 map 。

**# 测试阶段**

可以使用概率 map 或者二值 map 生成文本边框。为了更高的效率，我们使用概率 map，故网络中的阈值 map 的分支可以去掉。生成文本边框的步骤：

1. 概率 map（或者近似二值 map）使用一个固定的阈值（0.2）进行二值化
2. 根据二值 map 得到连通区域，此即缩小的文本区域
3. 膨胀这个缩小的区域，边距为 $D'$，根据 Vatti clipping 算法，

    $$D' = \frac {A' \times r'}{L'}$$

    其中 $A'$ 为缩小多边形的面积，$L'$ 则为其周长，$r'$ 为膨胀比例，根据经验设置为 $r'=1.5$

# 3. 代码分析

以 `ic15_resnet50_deform_thre.yaml` 配置文件为例进行说明，配置使用继承方式，此配置文件继承了 `base_ic15.yaml` 文件。

## 3.1 数据集加载

数据集加载类为 `ImageDataset`，此类加载数据乏善可陈，无非就是读取图像数据，以及文本框的多边形顶点坐标，但是加载之后的数据处理类比较多，

```yaml
# base_ic15.yaml
class: AugmentDetectionData # 水平翻转，平移旋转，resize
class: RandomCropData   # 随机 crop 出一个 640x640
class: MakeICDARData    # 生成ICDAR格式数据
class: MakeSegDetectionData
class: NormalizeImage
class: FilterKeys
```

**# MakeSegDetectionData** 

实现了缩小多边形，从 $G$ 缩小到 $G _ s$，

```python
# make_seg_detection_data.py

# 根据多边形顶点创建多边形对象
polygon_shape = Polygon(polygon)
# 根据缩小比例 r=0.4 计算边距 D
distance = polygon_shape.area * \
    (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
# 提取多边形的顶点
subject = [tuple(l) for l in polygons[i]]
padding = pyclipper.PyclipperOffset()
padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEPOLYGON)
# 根据指定边距缩小
shrinked = padding.Execute(-distance)
# 缩小多边形的顶点
shrinked = np.array(shrinked[0]).reshape(-1, 2)
cv2.fillPoly(gt[0], [shrinked.astype(np.int32)], 1)
data.update(image=image, polygons=polygons,
            gt=gt, mask=mask, filename=filename)
```

`MakeSegDetectionData` 生成了两个数据 `gt` 和 `mask`，前者是缩小多边形的 label（也是二值 map），后者是像素有效性掩模（也是二值 map），无效的像素表示既非正样本，也非负样本（即，ignore 这些样本）。

注意 `polygons` 是原始（未缩小）的多边形的顶点。

**# MakeBorderMap**

获取文本的边框

```python
# make_seg_detection_data.py

def process(self, data, *args, **kwargs):
    image = data['image']
    polygons = data['polygon']
    ignore_tags = data['ignore_tags']
    # 以边界线为中心，margin=d的边界带，值为
    # 1 - 边界带内点与中心的归一化距离
    canvas = np.zeros(image.shape[:2], dtype=np.float32)
    # 膨胀多边形 Gd 的掩模
    mask = np.zeros(image.shape[:2], dtype=np.float32)
    for i in range(len(polygons)):
        if ignore_tags[i]: continue
        self.draw_border_map(polygons[i], canvas, mask=mask)
    # 归一化到 0.3~0.7 之间
    canvas = canvas * (self.thresh_max - self.thresh_min) + self.thresh_min
    data['thresh_map'] = canvas
    data['thresh_mask'] = mask
```

其中计算边界带的过程示意图如下所示，

![](/images/ocr/db_5.png)

上图中，黑绿蓝三种颜色的多边形分别表示 原始多边形 $G$、膨胀多边形 $G_d$ 和缩小多边形 $G _ s$ 。膨胀和缩小的边距均为 $d$，根据缩小系数 $r=0.4$ 计算可得 $d$ 的大小。

$G _ d$ 和 $G _ s$ 的 gap 形成一个边界带，此边界带的中心线为 $G$ 的边界线，如图 5 中阴影部分就是这个边界带，边界带外部的像素值均为 0，边界带内部的像素值，从中心线（图 5 黑线多边形）值为 1，往两边均匀的下降到图 5 的蓝线多边形和绿线多边形，值为 0。

`data['thresh_map']` 存储了膨胀多边形的掩模，即 $G _ d$ 内部的像素值为 1，其余为 0。 

## 3.2 损失计算

```python
# model.py 
self.criterion = SegDetectorLossBuilder(
    args['loss_class'], ...
)
```

`loss_class` 配置为

```yaml
# ic15_resnet50_deform_thre.yaml
loss_class: L1BalanceCELoss
```

损失类位于

```python
# seg_detector_loss.py

class L1BalanceCELoss(nn.Module):
    def __init__(self, eps=1e-6, l1_scale=10, bce_scale=5):
        super().__init__()
        self.dice_loss = DiceLoss(eps)
        self.l1_loss = MaskL1Loss()
        self.bce_loss = BalanceCrossEntropyLoss()
        self.l1_scale = l1_scale
        self.bce_scale = bce_scale

    def forward(self, pred, batch):
        bce_loss = self.bce_loss(pred['binary'], batch['gt'], batch['mask'])
        # 训练阶段模型输出自适应阈值，测试阶段不输出
        if 'thresh' in pred:
            l1_loss, l1_metric = self.l1_loss(pred['thresh'], batch['thresh_map'], batch['thresh_mask'])
            dice_loss = self.dice_loss(pred['thresh_binary'], batch['gt'], batch['mask'])
            loss = dice_loss + self.l1_scale * l1_loss + bce_loss * self.bce_scale
        # 省略了测试阶段的代码
        return loss
```

上面代码中，使用二值交叉熵计算损失 $L _ s$，`gt` 是缩小多边形的掩模，`mask` 是有效像素的掩模（无效多边形内的像素掩模值为 0），`pred['binary']` 是概率得分预测 map $P$，`BalanceCrossEntropyLoss` 内部进行了难负例挖掘。

使用 L1 损失计算阈值 map 的损失 $L _ t$，`batch['thresh_map']` 为上面所说的边界带的值，边界带外的值均为 0，`batch['thresh_mask']` 为膨胀多边形 $G _ d$ 内部点的掩模，也就是说 $G _ d$ 多边形外部的点不考虑（既非正样本，也非负样本），边界带内部的点为正样本，位于 $G _ d$ 内部但是在边界带外部的点为负样本。

使用 DiceLoss 计算 $L _ t$ ，`pred['thresh_binary']` 是近似二值预测 map $\hat B$，根据 (2) 式计算可得。

这里根据这个配置文件， $L _ b$ 使用的是 DiceLoss，且没有进行难负例挖掘，只有 $L _ s$ 使用的是 BCE，并进行了难负例挖掘。当然也可以使用其他配置。

概率预测得分 map 和近似二值 map 共享相同的监督，即上面代码中的 `batch['gt']` 。



## 3.3 模型

配置文件中使用的是 `SegDetectorModel` 模型，

```yaml
# ic15_resnet50_deform_thre.yaml
model: SegDetectorModel
model_args:
    backbone: deformable_resnet50
    decoder: SegDetector
    decoder_args: 
        adaptive: True
        in_channels: [256, 512, 1024, 2048]
        k: 50
    loss_class: L1BalanceCELoss
```

这是一个封装类，封装了模型和 Loss 计算类等，其内部实际使用的模型代码为，

```python
# model.py

class BasicModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = getattr(backbones, args['backbone'])(**args.get('backbone_args', {}))
        self.decoder = getattr(decoders, args['decoder'])(**args.get('decoder_args', {}))

    def forward(self, data, *args, **kwargs):
        return self.decoder(self.backbone(data), *args, **kwargs)
```

**# backbone**

backbone 使用 `deformable_resnet50`，

```python
# resnet.py
def deformable_resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model with deformable conv.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   dcn=dict(modulated=True,
                            deformable_groups=1,
                            fallback_on_stride=False),
                   stage_with_dcn=[False, True, True, True],
                   **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(
            model_urls['resnet50']), strict=False)
    return model
```

backbone 提取 4 个特征，

```python
# resnet.py/ResNet
def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)     # 下采样率 x2

    x2 = self.layer1(x)     # x4
    x3 = self.layer2(x2)    # x8
    x4 = self.layer3(x3)    # x16
    x5 = self.layer4(x4)    # x32

    return x2, x3, x4, x5
```

**# decoder**

decoder 使用 `SegDetector`，

```python
# seg_detector.py
class SegDetector(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 inner_channels=256, k=10,
                 bias=False, adaptive=False, smooth=False, serial=False,
                 *args, **kwargs):
        '''
        bias: Whether conv layers have bias or not.
        adaptive: Whether to use adaptive threshold training or not.
        smooth: If true, use bilinear instead of deconv.
        serial: If true, thresh prediction will combine segmentation result as input.
        '''
        super(SegDetector, self).__init__()
        self.k = k
        self.serial = serial
        # 自顶向下，上采样后与下一层特征 spatial size 相同，融合
        # up(x5) + x4 -> x4
        # up(x4) + x3 -> x3
        # up(x3) + x2 -> x2
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        # FPN 中的横向连接
        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, bias=bias)
        # 然后再分别进行 upsample，所有 level 的特征 spatial size 相同后，concat 为 F
        # F spatial size 为输入图片 size 的 1/4
        self.out5 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=8, mode='nearest'))
        self.out4 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=4, mode='nearest'))
        self.out3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out2 = nn.Conv2d(
            inner_channels, inner_channels//4, 3, padding=1, bias=bias)

        self.binarize = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //     # 降低 F 的 channel
                      4, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels//4, inner_channels//4, 2, 2), # 上采样 x2
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels//4, 1, 2, 2), # 上采样 x2，同时通道数降为 1
            nn.Sigmoid())
        # 经过降低 channel，两次上采样，得到最终的概率预测 map
        self.binarize.apply(self.weights_init)

        self.adaptive = adaptive    # 是否自适应，训练阶段为 True
        if adaptive:
            # 输出自适应阈值的分支
            # 输入同样是特征 F
            self.thresh = self._init_thresh(
                    inner_channels, serial=serial, smooth=smooth, bias=bias)
            self.thresh.apply(self.weights_init)

    def _init_thresh(self, inner_channels,
                     serial=False, smooth=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels //    # 降低 F 的 channel
                      4, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            # 上采样 x2
            self._init_upsample(inner_channels // 4, inner_channels//4, smooth=smooth, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            # 上采样 x2，同时通道数降为 1
            self._init_upsample(inner_channels // 4, 1, smooth=smooth, bias=bias),
            nn.Sigmoid())
        return self.thresh

    def _init_upsample(self,
                       in_channels, out_channels,
                       smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=1, padding=1, bias=True))

            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def forward(self, features, gt=None, masks=None, training=False):
        c2, c3, c4, c5 = features
        # FPN 中的横向连接输出
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)
        # 自顶向下融合特征
        out4 = self.up5(in5) + in4  # 1/16
        out3 = self.up4(out4) + in3  # 1/8
        out2 = self.up3(out3) + in2  # 1/4
        # 经过 conv 调整以及上采样，使得所有level 的特征 spatial size 均为 1/4
        p5 = self.out5(in5)
        p4 = self.out4(out4)
        p3 = self.out3(out3)
        p2 = self.out2(out2)
        # concat 为 F
        fuse = torch.cat((p5, p4, p3, p2), 1)
        # this is the pred module, not binarization module; 
        # We do not correct the name due to the trained model.
        binary = self.binarize(fuse)        # 输出概率得分 map
        if self.training:
            result = OrderedDict(binary=binary)
        else:       # test 阶段直接返回概率得分 map
            return binary
        if self.adaptive and self.training:
            if self.serial:
                fuse = torch.cat(
                        (fuse, nn.functional.interpolate(
                            binary, fuse.shape[2:])), 1)
            thresh = self.thresh(fuse)      # 自适应阈值 map
            thresh_binary = self.step_function(binary, thresh)  # 近似二值 map
            result.update(thresh=thresh, thresh_binary=thresh_binary)
        return result

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))
```

## 3.4 test 阶段代码

以 `demo.py` 文件为例进行说明。

```python
def inference(self, image_path, visualize=False):
    self.init_torch_tensor()    # 设置默认 tensor dtype，device 等
    model = self.init_model()   # 加载模型
    model.eval()
    batch = dict()
    batch['filename'] = [image_path]
    # 加载图片，保证短边缩放到指定长度，长边则是类似比例缩放到 32 的整数倍
    img, original_shape = self.load_image(image_path)
    batch['shape'] = [original_shape]
    with torch.no_grad():
        batch['image'] = img
        # pred: 概率预测得分 map
        pred = model.forward(batch, training=False)
        output = self.structure.representer.represent(batch, pred, is_output_polygon=self.args['polygon'])
        ...
```

上述代码中，`representer` 为

```yaml
# ic15_resnet50_deform_thre.yaml
representer:
    class: SegDetectorRepresenter
    max_candidates: 1000
```

`represent` 方法定义为

```python
# seg_detector_representer
def representer(self, batch, _pred, is_output_polygon=False):
    images = batch['image'] # (1, 3, h, w)
    if isinstance(_pred, dict):
        pred = _pred[self.dest]
    else:
        pred = _pred
    # 这里使用标准二值化，即 pred > thresh，test 阶段移除了自适应阈值分支
    segmentation = self.binarize(pred)
    boxes_batch = []
    scores_batch = []
    for batch_index in range(images.size(0)):
        height, width = batch['shape'][batch_index]
        if is_output_polygon:   # 输出多边形
            boxes, scores = self.polygons_from_bitmap(
                pred[batch_index],
                segmentation[batch_index], width, height)
        else:   # 输出矩形框
            boxes, scores = self.boxes_from_bitmap(
                pred[batch_index],
                segmentation[batch_index], width, height)
        boxes_batch.append(boxes)
        scores_batch.append(scores)
    return boxes_batch, scores_batch
```
# REF

<div id="refer-anchor-1"></div>

- [1] Xue, C.; Lu, S.; and Zhan, F. 2018. Accurate scene text detection
through border semantics awareness and bootstrapping. In Proc.
ECCV, 355–372.