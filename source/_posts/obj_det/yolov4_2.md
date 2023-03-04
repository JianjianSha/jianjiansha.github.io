---
title: YOLOv4 代码实现
date: 2022-12-01 16:17:23
tags: object detection
mathjax: true
---

列出项目用到的相关配置
```python
# cfg
cfg.width = 608
cfg.height = 608

cfg.jitter = 0.2
cfg.hue = 0.1
cfg.saturation = 1.5
cfg.exposure = 1.5

cfg.flip = 1
cfg.blur = 0

cfg.boxes = 60      # box 数量。如果一个图像中 gt box 数量超过 60，那么截断数量
cfg.classes = 80    # COCO 分类数量
```

# 1. dataset

## 1.1 data augmentation

数据增强有 mosaic，cutmix 等。

### 1.1.1 mosaic

mosaic 使用 4 个图像进行 mix，如图 1，

![](/images/obj_det/yolov4_2_1.png)
<center>图 1. 图源：YOLOv4 论文</center>





# 2. 模型

网络结构如图 2，

![](/images/obj_det/yolov4_2_2.png)
<center>图 2. </center>

block 具有 `x2` 的下采样率，总共有 5 个 block，故总的下采样率为 `x32`，其中 2 个 block 用到了左侧的输出，用于径向连接。`block, xn` 表示 block 内部用了 n 个连续的 residual 结构。观察网络结构我们可知，

1. 输入 size 为 $608 \times 608$ （根据配置可知）
2. 网络右侧 encoder 的输出特征 size 为 $19 \times 19$ （下采样率为 32）
3. 网络左侧有两个 `x2` 的 upsample，与径向连接融合（concatenation），输出特征 size 为 $76 \times 76$，在这个 size 的特征上进行一个 scale 的预测
4. 然后再使用两个 `x2` 下采样率的 conv，得到 size 为 $38\times 38$ 和 $19 \times 19$ 的特征，分别在这两个 scale 的特征上进行预测

## 2.1 yolo

重点看 yolo 这个 layer。

使用了 3 个 scale 的特征进行预测，每个 scale 的特征使用专门的 yolo layer 进行预测，如图 2， 从下到上特征的 stride 分别为 `8, 16, 32`。

每个位置使用 9 个 anchor 进行预测，由于之前使用 backbone 的最后一个特征进行预测，所以每个位置使用 3 个不同 aspect ratio 和 3 个不同 scale 的 anchor 进行预测（$3 \times 3 = 9$），而现在是使用三个不同 scale 的特征，所以每个特征使用 3 个不同 aspect ratio 的 anchor 进行预测，不同 scale 的特征，即对应的 anchor 的 scale 不同，具体配置如下：

```
anchors = 12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401
```

以上数值每两个表示一个 anchor 的 w 和 h，一共表示 9 个 anchor 的 w 和 h。

以第一个 yolo layer 为例，其对应的特征 stride=8，使用前三个 anchor，由于 anchor 的 w 和 h 是基于原始 network input size，所以在 feature maps 上，需要将 anchor 的 w 和 h 除以相应的 stride，

```python
stride = 8
anchors = [float(i) for i in block['anchors'].split(',')]
mask = [int(i) for i in block['mask'].split(',')]
masked_anchors = []

for m in mask:
    masked_anchors += anchors[m * 2 : (m+1) * 2]
# 12.0/8, 16.0/8, 19.0/8, 36.0/8, 40.0/8, 28.0/8
masked_anchors = [anchor / stride for anchor in masked_anchors]
masked_anchors = np.array(mask_anchors).reshape(3, 2)
```

COCO 分类数量为 80，坐标使用 `x, y, w, h` 4 个值，以及一个 conf 值，所以每个 anchor 预测 `80+4+1=85` 个值，每个位置有 3 个 anchor，一共 `85*3=255` 个值，所以 yolo layer 的输入特征 channel 为 255 。对于这个 255 channel 的特征，每 85 个值用于预测一个 anchor ，这 85 个值依次表示 `x, y, w, h, conf` 以及 80 个 `cls`。

yolo layer 的输入是前一 conv 的输出，这个输出是不经过任何激活的，所以取值范围为 $\mathbb R$。坐标预测如下图，

![](/images/obj_det/yolov4_2_3.png)
<center>图 3. 虚线框：anchor；实线框：根据 anchor 以及输出特征值得到预测 box</center>

yolo layer 的输出为 tuple `(boxes, confs)`：

1. `boxes`， shape 为 `(batch_size, 3 * H * W,  1, 4)`，这里 `H, W` 指的是这个 yolo layer 的输入特征平面的 size。最后一维 `4` 表示归一化的 `x1, y1, x2, y2`（值范围为 `[0, 1]`）
2. `confs`，conf 与分类得分的乘积，shape 为 `(batch_size, 3 * H * W, num_classes)`，其中 `num_classes=80`，在使用 COCO 数据集时。


## 2.2 loss

### 2.2.1 gt boxes

gt boxes 的 shape 为 `(batch_size, cfg.boxes, 5)`，这里 `cfg.boxes` 配置为 60，表示一个图像（可能是 mosaic 之后的图像）中 gt boxes 数量最多为 60，最后一维 `5` 表示 `x1, y1, x2, y2, id` ，其中 `id` 为分类 id，从 0 开始取值。

直接看 loss 的计算代码，以第一个 yolo layer 为例说明， 这个 yolo layer 对应的 stride=8，anchor mask = [0, 1, 2]，

```python
anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]]
batch_size = 4
n_anchors = 3       # 单个 yolo layer 使用的 anchor 数量
image_size = 608
strides = [8, 16, 32]
masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]



def loss(xin, labels):
    loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = 0, 0, 0, 0, 0, 0
    for idx, output in enumerate(xin):    # train 模式下，yolo layer 的输出就是其输入
        stride = strides[idx]
        batch_size = output.shape[0]    # output: (B, 255, H, W)
        fsize = output.shape[2]         # 当前 yolo layer 对应的输入特征 size
        n_ch = 5 + n_classes            # 5+80, x y w h conf, 80 cls scores
        all_anchors_grid = [(w / stride, h / stride) for w, h in anchors]
        masked_anchors = np.array([all_anchors_grid[j] for j in masks[idx]], dtype=np.float32)
        ref_anchors = torch.zeros((len(anchors), 4), dtype=torch.float32)
        ref_anchors[:,2:] = all_anchors_grid

        grid_x = torch.arange(fsize, dtype=torch.float).repeat(batch_size, 3, fsize, 1).to(device)  # (B, 3, H, W)
        grid_y = torch.arange(fsize, dtype=torch.float).repeat(batch_size, 3, fsize, 1).permute(0, 1, 3, 2).to(device)
        # (B, 1, H, W)
        anchor_w = torch.from_numpy(masked_anchors[:, 0]).repeat(batch_size, fsize, fsize, 1).permute(0, 3, 1, 2).to(device)
        anchor_h = torch.from_numpy(masked_anchors[:, 1]).repeat(batch_size, fsize, fsize, 1).permute(0, 3, 1, 2).to(device)

        output = output.view(batch_size, n_anchors, n_ch, fsize, fsize)     # 一个 yolo 使用 3 scales 的 anchor
        output = output.permute(0, 1, 3, 4, 2)      # (B, 3, H, W, 85)

        # x, y, w, h, conf, 80 cls
        # 除了 w 和 h，其他列均需要归一化（sigmoid）
        output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(output[..., np.r_[:2, 4:n_ch]])

        pred = output[..., :4].clone()      # 预测 box 的 x y w h
        pred[..., 0] += grid_x              # x, (B, 3, H, W)
        pred[..., 1] += grid_y              # y, (B, 3, h, W)
        pred[..., 2] = torch.exp(pred[..., 2]) * anchor_w   # w, (B, 3, H, W)
        pred[..., 3] = torch.exp(pred[..., 3]) * anchor_h   # h, (B, 3, H, W)

        obj_mask, tgt_mask, tgt_scale, target = build_target(pred, labels, batch_size, fsize, 
                                                             n_ch, ref_anchors, masked_anchors, idx)
        # loss calculation
        output[..., 4] *= obj_mask      # (B, n_anchors, H, W)
        output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
        output[..., 2:4] *= tgt_scale

        target[..., 4] *= obj_mask
        target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
        target[..., 2:4] *= tgt_scale

        loss_xy += F.binary_cross_entropy(input=[..., :2], target=target[..., :2], 
                                          weight=tgt_scale * tgt_scale, reduction='sum')
        loss_wh += F.mse_loss(input=output[..., 2:4], target=target[..., 2:4], reduction='sum') / 2
        loss_obj += F.binary_cross_entropy(input=output[..., 4], target=target[..., 4], reduction='sum')
        loss_cls += F.binary_cross_entropy(input=output[..., 5:], target=target[..., 5:], reduction='sum')
    loss = loss_xy + loss_wh + loss_obj + loss_cls
    return loss


def build_target(pred, labels, batch_size, fsize, n_ch, ref_anchors, masked_anchors, idx):
    tgt_mask = torch.zeros(batch_size, n_anchors, fsize, fsize, 4 + 80).to(device)
    obj_mask = torch.ones(batch_size, n_anchors, fsize, fsize).to(device)
    tgt_scale = torch.zeros(batch_size, n_anchors, fsize, fsize, 2).to(device)
    target = torch.zeros(batch_size, n_anchors, fsize, fsize, n_ch).to(device)

    # labels: (B, 60, 5)，每个 image 中最多 60 个 gt box，不足 60 个，则使用 0 填充
    nlabel = (labels.sum(dim=2) > 0).sum(dim=1) # batch 中所有 gt box 总数，(B,)
    gt_x = (labels[:,:,0] + labels[:,:,2]) / (strides[idx] * 2) # gt box 中心 x 坐标，(B, 60)
    gt_y = (labels[:,:,1] + labels[:,:,3]) / (strides[idx] * 2) # gt box 中心 y 坐标，(B, 60)
    gt_w = (labels[:,:,2] - labels[:,:,0]) / (strides[idx])
    gt_h = (labels[:,:,3] - labels[:,:,1]) / (strides[idx])

    gt_i = gt_x.to(torch.int16).cpu().numpy()
    gt_j = gt_y.to(torch.int16).cpu().numpy()

    for b in range(batch_size):
        n = int(nlabel[b])
        if n == 0: continue

        gt_box = torch.zeros(n, 4).to(device)
        gt_box[:, 2] = gt_w[b,:n]
        gt_box[:, 3] = gt_h[b,:n]
        gt_i_ = gt_i[b, :n]     # gt box 中心投影到 feature map 上的坐标， (n,)
        gt_j_ = gt_j[b, :n]
        
        ious = bboxes_iou(gt_box.cpu(), ref_anchors, CIoU=True) # (n, n_anchors * 3)
        best_n_all = anchor_ious_all.argmax(dim=1)              # (n,)，每个值位于 [0, 8]
        best_n = best_n_all % 3                                 # 每个 gt box 最佳匹配的 anchor
        best_n_mask = ((best_n_all == masks[idx][0]) ||
                       (best_n_all == masks[idx][1]) ||
                       (best_n_all == masks[idx][2]))
        if sum(best_n_mask) == 0: continue

        gt_box[:, 0] = gt_x[b, :n]
        gt_box[:, 1] = gt_y[b, :n]

        pred_ious = bboxes_iou(pred[b].view(-1, 4), gt_box, xyxy=False) # (n_anchors*H*W, n)
        pred_best_iou, _ = pred_ious.max(dim=1)         # (n_anchors*H*W)
        pred_best_iou = (pred_best_iou > ignore_thre)
        pred_best_iou = pred_best_iou.view(pred[b].shape[:3])   # (n_anchors, H, W)
        obj_mask[b] = ~pred_best_iou    # 指定 (n_anchors, H, W) 哪些位置最大匹配到某个 gt box

        for ti in range(best_n.shape[0]):
            if best_n_mask[ti] == 1:    # 与 gt box 匹配的 anchor 属于当前 yolo (idx)
                i, j = gt_i_[ti], gt_j_[ti]     # 当前 gt box 投影到 feature map 上的像素坐标
                a = best_n[ti]          # 与 gt box 匹配的 anchor 索引（取值范围为 {0, 1, 2}
                obj_mask[b, a, j, i] = 1    # 与 gt box 最佳匹配 anchor 也进行标记
                tgt_mask[b, a, j, i, :] = 1
                target[b, a, j, i, 0] = gt_x[b, ti] - gt_x[b, ti].to(torch.int16).to(torch.float)
                target[b, a, j, i, 1] = gt_y[b, ti] - gt_y[b, ti].to(torch.int16).to(torch.float)
                target[b, a, j, i, 2] = torch.log(gt_w[b, ti] / torch.Tensor(masked_anchors[best_n[ti], 0] + 1e-16))
                target[b, a, j, i, 3] = torch.log(gt_h[b, ti] / torch.Tensor(masked_anchors[best_n[ti], 1] + 1e-16))
                target[b, a, j, i, 4] = 1
                target[b, a, j, i, 5 + labels[b, ti, 4].to(torch.int16).cpu().numpy()] = 1
                tgt_scale[b, a, j, i, :] = torch.sqrt(2 - gt_w[b, ti] * gt_h[b, ti] / fsize / fsize)
    return obj_mask, tgt_mask, tgt_scale, target
```