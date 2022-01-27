---
title: Faster RCNN 回顾
date: 2021-12-23 16:38:06
tags: object detection
mathjax: true
---
论文：[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
<!--more-->
本文对 Faster R-CNN 进行梳理，主要用于复习 Faster R-CNN，重温一些实现细节。

整个网络的结构示意图如图 1，
![](/images/obj_det/faster-rcnn_1.png)
图 1. Faster R-CNN 网络示意图

说明：
1. 任意 size 的 image 经过 一个 backbone （全卷积）得到 feature maps
2. 一方面，feature maps 经过 RPN 网络 得到 proposals，以及每个 proposal 的 objectness score，这是一个**二分类**，表示 proposal 是否是正例。
3. 另一方面，利用上一步得到的 proposal 在 feature maps 进行 crop，crop 之后的 部分 feature maps 经过 ROI pooling，得到固定 size 的 feature （例如 `7x7`)，作为 Fast R-CNN 的输入，Fast R-CNN 输出分类得分，以及坐标（$t_x, t_y, t_w, t_h$）。
4. 两个子网络 RPN （用于生成 proposals）和 Fast R-CNN（目标检测网络）共享 baseline。

下面分别对 backbone 和后续的两个子网络予以讨论。

# backbone
 
论文中采用 VGG-16，即 `5` 个 stage， stage 内的 `3x3 conv` 数量分别为 `2, 2, 3, 3, 3`，相邻 stage 之间使用 `mp` 进行降采样，故 baseline 输出的 feature maps 的 size 相较于 input size ，总的降采样率为 `16`。

## image 预处理
**image 读取**

```python
im = cv2.imread(img_path)   # BGR order
# use cv2.cvtColor(im, cv2.COLOR_BGR2RGB) ot convert the channel order
```

**随机flip**
```python
if random.random() > 0.5:
    im = im[:,::-1,:]       # left to right flipped
```

gt box 的坐标 $(x_1,y_1,x_2,y_2)$ 变成 $(W-x_2,y_1,W-x_1,y_2)$


**减去均值**
```python
# pixel mean follows the order BGR
im -= np.array([102.9801, 115.9465, 122.7717])[np.newaxis, np.newaxis,:]
```

**resize**

resize image 使得短边为 `600`，同时 resize 后长边不超过 `1000`，
```python
target_size = 600.0
max_size = 1000.0
h, w = im.shape[0:2]    # im is an object returned from cv2.imread()
size_min, size_max = np.min(h, w), np.max(h, w)
im_scale = target_size / size_min

if np.round(im_scale * size_max) > max_size:
    # decrease the im_scale
    im_scale = max_size / size_max
im = cv2.resize(im, fx=im_scale, fy=im_scale)

# update the gt_box coordinates
gt_box = np.array([x1, y1, x2, y2])
gt_box = (gt_box * im_scale).astype(np.int16)
```

**组建 blob/tensor**
经过上面的 resize 后，各个 image size 相差不大了，取这组 images 中最大的 height，和最大的 width，创建一个可以容纳 batch 内所有 images 的 blob/tensor，
```python
# im.shape -> (h, w) -> (batch_size, 2) -> (2,)
max_shape = np.array([im.shape for im in ims]).max(axis=0)
batch_size = len(imgs)
blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                dtype=np.float32)
for i in range(batch_size):
    im = ims[i]
    # the image is aligned with the top-left cornor
    blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
blob = blob.transpose((0, 3, 1, 2)) # from (B,H,W,C) to (B,C,H,W)
```

network 的 input size 以 `(600, 1000)` 为例，下采样率为 `16`，于是输出 feature size 大约是 `(38, 63)`。


# RPN
在 baseline 得到的 feature maps，使用 `3x3 conv`，得到 `512-d`（或者如图 2 的 `256-d`） 的中间 feature，然后分别经过两个 full-connected layer，得到 `2k` 的 objectness scores，和 `4k` 个坐标，即 feature maps 上每个 location 预测 `k` 个 proposals，如图 2，
![](/images/obj_det/faster-rcnn_2.png)
图 2. RPN

说明：
1. 每个 location 的中间 feature 均为 `512-d`，对于 cls layer，全连接层参数 $W^{512\times 2k}$，reg layer 的全连接层参数 $W^{512 \times 4k}$，所有 location 处的这两个全连接层共享参数，即，分别使用 `1x1 Conv 2k` 和 `1x1 Conv 4k` 的两个卷积层，输出 shape 分别为 `(B, H, W, 2k)` 和 `(B, H, W, 4k)`， `(H, W)` 为 feature maps 上 size，`B` 为 `batch_size`。

2. 训练 RPN 时，`batch_size=1`，即每个 mini-batch 内仅有 `1` 个 image。在训练 Fast RCNN 子网络时，保持 RPN 不变，此时取 `batch_size=2`。

## Anchors
RPN （在每个 location 处）使用 `k` 个 Anchors 辅助预测 proposals，`k` 个 Anchors 具有不同的 scale 和 aspect ratio。通常 `k=9`：3 个 scales 和 3 个 aspect ratios。对于 $H \times W$ 的 feature maps，一共有 $H \times W \times k$ 个 anchors。

**如何确定 anchor 的大小和位置**

anchor 的 scale 取 `8, 16, 32`（人为确定，可以根据实际任务进行调整。由于 backbone 的下采样率为 `16`，映射到原 input image 就是 `128,256,512`，而 input size 大概是 `600 x 1000`，故这个 scale 较为合理）。aspect ratio （记为 $r$）取 `0.5, 1, 2` 三个值，正好覆盖 矮胖，方正，高瘦 三种情况，anchor 的 size 记为 $(h, w)$，那么有
$$r=\frac h w$$


对于标准 scale，即 $r=1, h \times w = 16 \times 16$。改变 $r$ 值，但是面积保持相同，故
$$s = 16 \times 16 = h  w = r w^2=\frac {h^2} r$$
于是
$$w= \sqrt {s/r} , \quad h = \sqrt{sr}$$

考虑到 scale 可取不同值，那么最终 anchor size 为

$$w = a \sqrt{s/r}, \quad h = a \sqrt{sr}, \quad a = 0.5, 1, 2, \quad r=0.5, 1, 2, \quad s = 16^2=256$$

上式可以确定 `9` 个 anchors 的 size，其中心点坐标相同，均为 $x_c=0+0.5(16-1), \ y_c=0+0.5(16-1)$（考虑到 C/Python 语言习惯以 `0` 开始表示第一个位置），于是左上右下坐标为
$$x_1=x_c-\frac 1 2(w-1), \quad y_1=y_c-\frac 1 2(h-1)$$
$$x_2=x_c+\frac 1 2(w-1), \quad y_2=y_c+\frac 1 2(h-1)$$
```python
s = 16**2
r = torch.tensor([[0.5, 1., 2.]])           # (1, 3)
a = torch.tensor([[0.5, 1., 2.]]).t()       # (3, 1)
w = torch.sqrt(s / r) * a                   # (3, 3)
h = r * w                                   # (3, 3)

h = h.reshape(1, 9)        # (1, 9)
w = w.reshape(1, 9)        # (1, 9)
xc = 0 + 0.5 * (16-1)
yc = 0 + 0.5 * (16-1)
x1 = xc - 0.5 * (w-1)
y1 = yc - 0.5 * (h-1)
x2 = xc + 0.5 * (w-1)
y2 = yc + 0.5 * (h-1)

# >>>>>> each location has its own 9 anchors <<<<<<
H = 40  # Here I simply assign 40 and 60 to H and W respectively,
W = 60  # but in practice, H and W may be other values.
grid_x = torch.arange(W).repeat(H, 1).view(H*W, 1)        # (HW, 1)
grid_y = torch.arange(H).t().repeat(1, W).view(H*W, 1)    # (HW, 1)

# all anchors locations are
k = 9
x1 = (x1 + grid_x).view(H*W*k)
y1 = (y1 + grid_y).view(H*W*k)
x2 = (x2 + grid_x).view(H*W*k)
y2 = (y2 + grid_y).view(H*W*k)
anchors = torch.hstack((x1, y1, x2, y2))

# filter out those out-of-scope anchors
# inside_indices = torch.where(x1 >= 0 & y1 >= 0 & x2 < W & y2 < H)
```

## Loss
RPN 这个网络的 Loss，包含分类（proposal 是否含有 object，二分类）损失，以及坐标回归损失。

**如何确定正例**： 记 image 中 gt boxes 数量为 $m$，计算 $m$ 个 gt boxes 与 $HW k$ 个 anchors 之间的 IOU ， IOU 矩阵记为 $M_{m \times HWk}$，
1. 与某个 gt box 具有最大 IOU 的 anchor 为正例，负责预测这个 gt box，这种正例 anchor 有 $m$ 个
    ```python
    # gt: (m, 4)
    M = bbox_iou(gt/16, anchors)   # anchors: (H*W*k, 4)
    # positive anchors(have a max iou with some gt box)
    max_ious, positive_anchor_indices = torch.max(M, dim=1)

    labels = torch.ones(anchors.shape[0], dtype=torch.int8) * (-1)
    # set the positive anchors with label `1`
    labels[positive_anchor_indices] = 1
    ```
2. 与某个 gt box 的 IOU 大于一个阈值（论文中使用 `0.7`），则认为这样的 anchor 是正例，$m$ 个 gt box 中，与这个 anchor 有最大 IOU 的 gt box，将被这个 anchor 预测，
    ```python
    # the max iou(with some one gt box) for each anchor
    # both the two tensor have the same shape of (HWk,)
    max_ious, gt_indices = torch.max(M, dim=0)     

    # indices of positive anchors(with a iou > 0.7)
    # all indices are in [0, HWk)
    positive_anchor_indices = torch.where(max_ious >= 0.7)[0]
    # extract indices of gt boxes ( in [0, m) ) for positive anchors
    positive_gt_indices = gt_indices[positive_anchor_indices]

    # set the positive anchors with label `1`
    labels[positive_anchor_indices] = 1
    ```
满足以上两个条件中的一个，认为是 positive。

**如何确定负例**：对于非正例的 anchor，没有全部作为负例，否则正负例样本严重不均衡。事实上，对于 IOU 在 0.5 附近的 anchor，属于 hard example，不予采用。与 gt box 的最大 IOU 小于阈值 （论文中取 `0.3`）的认为是负例：
```python
max_ious, gt_indices = torch.max(M, dim=0) 
# indices( in [0, HWk) ） of negative anchors
negative_anchor_indices = torch.where(max_ious < 0.3)[0]
# extract indices of gt boxes ( in [0, m) ) for negative anchors
negative_gt_indices = gt_indices[negative_anchor_indices]

# set the negative anchors with label `0`
labels[negative_anchor_indices] = 0
```

单个 image 的损失如下，
$$L=\frac 1 {N_cls} \sum_i L_{cls}(p_i, p_i^{\star})+\lambda \frac 1 {N_{reg}}\sum_i p_i^{\star} L_{reg}(t_i, t_i^{\star})$$

其中：

1. $N_{cls}=256$ 表示一个 mini-batch 内，所选取的用于分类任务的正负 anchors 的数量。按照 `1:1` 比例分配正负 anchors，如果正例 anchors 不足 `128`，那么使用负例 anchors 进行补充。前面说到 `batch_size=1`，这表明，**在每个 image 上随机选择 `128` 个 positive anchors，如果不够，使用 negative anchors 补充**。
2. $N_{reg}=HWk \approx 10 N_{cls}$，$N_{reg}$ 为一个 image 上所有的 anchors 数量， 故取 $\lambda=10$ 。注意：**回归损失使用全部 anchors，而非第 `1` 点中的取样 `256` 个**。
3. 第 `i` 个 anchor 如果是 positive，那么 $p_i^{\star}=1$，否则 $p_i^{\star}=0$
4. $p_i$ 表示第 `i` 个 anchor 被预测为正的得分。
5. $L_{cls}$ 可以使用负对数似然损失，也就是交叉熵损失
    ```python
    rpn_batch = 256
    # imitate the sampling process
    scores = torch.randn((rpn_batch, 2))            # (256, 2)
    gt_conf = torch.empty(rpn_batch).random_(2)     # (256,)
    loss = nn.CrossEntropyLoss()    # do not need to normalize the input
    cls_loss = loss(scores, gt_conf)
    ```
6. $L_{reg}$ 为 smooth L1 函数

    $$L_{reg}(t_i, t_i^{\star})=smooth_{L_1}(t_i-t_i^{\star})$$
    $$smooth_{L_1}=\begin{cases}0.5 x^2 & |x|<1 \\ |x|-0.5 & |x|\ge 1\end{cases}$$

    这是为了防止梯度太大，导致训练不稳定

    $t_i$ 是一个向量，表示第 `i` 个 anchor 的预测坐标偏差 $(t_x, t_y, t_w, t_h)$，$t_i$ 就是 reg layer （即 `1x1 conv 4k`，k=9）的输出。坐标偏差有如下关系：
    $$t_x=(x_p-x_a)/w_a, \quad t_y=(y_p-y_a)/h_a$$
    $$t_w=\log(w_p/w_a), \quad t_h=\log(h_p/h_a)$$
    根据上面 4 个等式，以及 anchor 的坐标 $(x_a, y_a, w_a, h_a)$ 可以很容易得到预测 proposal 的坐标 $(x_p, y_p, w_p, h_p)$。

    $t_i^{\star}$ 为 gt box 对 anchor 的坐标偏差，可看作是 gt offset，其计算如下：
    $$t_x^{\star}=(x_{gt}-x_a)/w_a, \quad t_y^{\star}=(y_{gt}-y_a)/h_a \tag{1}$$
    $$t_w^{\star}=\log(w_{gt}/w_a), \quad t_h^{\star}=\log(h_{gt}/h_a) \tag{2}$$

    目标就是使得正例 anchor 的预测偏差 $t_i$ 尽量逼近真实偏差 $t_i^{\star}$。
    ```python
    assert B == 1
    loss = torch.nn.SmoothL1Loss()
    # t_i: in practice, t_i is gotten from reg_layer(`1x1 conv 4k`)
    t_i = torch.randn((B, 4*k, H, W)).permute(0, 2, 3, 1).view(B*H*W*k, 4)
    t_i = t_i[labels==1]

    # anchors: (HWk, 4)
    # gt_indices: index of gt(with max iou) for each anchor
    t_i_gt = bbox_target(anchors, gt[gt_indices])
    t_i_gt = t_i_gt[labels==1]
    reg_loss = loss(t_i, t_i_gt)
    ```

根据 backbone 和 RPN 以及 RPN 对应的 loss，就可以训练 RPN ，训练好 RPN 之后，利用 RPN 来生成 proposals，见下一节内容。

# ROI Proposal
RPN 中，我们说到有两个分支：cls 分支和 reg 分支，输出分别表示 anchor 的分类 objectness scores（未归一化）以及坐标偏差，shape 分别为 $(B, H, W, 2k), \ (B, H, W, 4k)$，按以下步骤得到 proposals：
1. 二分类预测得分（cls分支输出），取预测为正例的得分
2. 根据 (1,2) 式和 anchors 坐标，得到 proposals 坐标，并 clip 使得在 input image 范围内
3. 过滤掉太小的 proposals
4. 得分降序排列，proposals 顺序保持与 scores 的一致，取 top 6000 的 proposals，此举是为了加快 nms 速度
5. 执行 nms
6. 对 nms 之后的 proposals 继续取 top 300

（代码中，TRAIN 和 TEST 阶段，上面 nms 前后的两组 top n 中 `n` 值各不相同）

**注：`batch_size=1`**


```python
assert B == 1
scores = torch.rand((B, 2*k, H, W))
t_i    = torch.randn((B, 4*k, H, W))

# normalize scores, and extract the scores of objectness(positive)
scores = torch.softmax(scores, dim=1)[:, k:, :, :]
scores = scores.permute(0, 2, 3, 1).view(-1, 1)
t_i = t_i.permute(0, 2, 3, 1).view(-1, 4)
# according to eq(1) and eq(2), recover the (x1y1x2y2) of proposals
# multiply by 16 to recover the size relatived to image input
proposals = bbox_transform(anchors, t_i) * 16
# clip the x1, y1 to 0, and x2, y2 to max_shape[1] and max_shape[0], repectively
proposals = clip(proposals)
# filter out those very little proposals
min_scale = 16 * im_scale
pw = proposals[:, 2] - proposals[:,0] + 1
ph = proposals[:, 3] - proposals[:,1] + 1
keep = pw >= min_scale & ph >= min_scale

proposals = proposals[keep]
scores = scores[keep]

order = scores.sort(0, descending=True)
pre_nms_topn = 6000
order = order[:pre_nms_topn]
proposals = proposals[order]
scores = scores[order]

keep = nms(proposals, scores, nms_thre)
post_nms_topn = 300
keep = keep[:post_nms_topn]

proposals = proposals[keep]
scores = scores[keep]
```

以上便是使用 RPN 生成 proposals 的过程，当然因为 input image 是经过 resize 的，故，proposals 还需要除以 `im_scale`，以恢复原先的 size。

在 `alt_opt` 这种训练方式中，就是使用 `rpn_generate` 专门为数据集中每个 image 生成 proposals，得到 proposals 的列表（双重列表），然后 dump 到一个 `.pkl` 文件中。


---
以下内容为 `end2end` 的近似联合训练方式，可以跳过

上面得到的 top 300 proposals 与这个 image 中的所有 gt boxes 一起，得到总的候选 proposals，然后：
1. 总的候选 proposals 与 gt boxes 计算 IOU 矩阵，并求出每个候选 proposal 对应最大 IOU 的那个 gt box，以及最大 IOU
2. 最大 IOU 大于某阈值（`0.5`）的 proposal 被认为是 正例
3. 单个 image 中取 `128` 个 proposals，作为 RCNN 检测网络的输入，其中正负 proposals 的比例为 `1:3`，故正例 proposals 数量为 `32`，如果第 `2` 步中筛选出的正例数量大于 `32`，那么随机取 `32` 个正例 proposals。
4. 最大 IOU 位于 `[0.1, 0.5)` 之间的 proposals 为负例（难负例挖掘）。负例数量为 `96`，如果第 `3` 步中正例数量小于 `32`，那么缺少的使用负例补充。最后正负 proposals 总数量可能小于 `128`。
5. 与 RPN 中类似，RCNN 中以 proposal 为 anchor，预测 proposal 的分类（PASCAL VOC 为例，为 20+1 种分类，包含 bg 分类），以及 proposal 的坐标偏差，故需要计算 gt box 相对 proposal 的坐标偏差（固定某个 proposal，取与其有最大 IOU 的那个 gt box）
```python
# gt_labels: class labels of all gt boxes, shape: (m,)

all_rois = torch.vstack((proposals, gt))    # (n, 4)
M = bbox_iou(all_rois, gt)      #(n, m), m is number of all gt boxes
max_ious, gt_indices = torch.max(M, dim=1)
# fix one proposal, the class label of the gt box (with the max iou) is used
labels = gt_labels[gt_indices]  

fg_thre = 0.5
fg_indices = torch.where(max_ious >= fg_thre)[0]
bg_thre_low = 0.1
bg_thre_up = 0.5
bg_indices = torch.where(max_ious >= bg_thre_low & max_ious < bg_thre_up)[0]

rois_num = 128
fg_ratio = 0.25
fg_num = int(rois_num * fg_ratio)
if len(fg_indices) > fg_num:
    fg_indices = fg_indices[torch.randint(len(fg_indices), (fg_num,))]
bg_num = rois_num - fg_num
if len(bg_indices) > bg_num:
    bg_indices = bg_indices[torch.randint(len(bg_indices), (bg_num,))]

keep = torch.cat((fg_indices, bg_indices), axis=0)
labels = labels[keep]
# class label for any bg proposal is 0
labels[len(fg_indices):] = 0
rois = all_rois[keep]
# gt_indices: index of gt(with max iou) for each anchor
gt_indices_keep = gt_indices[keep]
# according to eq(1) and eq(2), get the gt coordinate offsets for all candidated proposals
bbox_target = bbox_target(rois, gt[gt_indices_keep])
# 1. for regression task, only positive proposals are used in reg-loss calculation
# 2. gt coordinate offsets are class related
t_i_gt = torch.zeros((rois.shape[0], 4 * num_classes))
for i in range(len(fg_indices)):
    cls_id = labels[i]
    start = 4 * cls_id
    end   = start + 4
    t_i_gt[i, start:end] = bbox_target[i,:]
```
上面代码，最后的 `t_i_gt` 保存了正例 proposal 所对应的 gt box 的坐标偏差。

以上内容为 `end2end` 的近似联合训练方式，可以跳过

---

# RCNN

训练 Fast R-CNN，需要加载数据集的 gt boxes，以及上一节所说的 proposals，既然这些 proposals 用于训练，就需要确定其分类，以及正负例（注：gt boxes 全部看作正例且已经有分类的 proposals），
```python
# gt_boxes of one image in the dataset
# gt_classes of gt_boxes of that image
# proposals of that image
M = bbox_iou(proposals, gt_boxes)

max_ious, gt_indices = torch.max(M, dim=1)
fg_indices = torch.where(max_ious > 0)[0]

# classes of proposals
classes = torch.zeros((len(proposals),))
classes[fg_indices] = gt_classes[gt_indices[fg_indices]]

# combine gt boxes and proposals
# rois: coordinates of all rois
rois = torch.vstack(gt_boxes, proposals)        # concatenate two 2-d vectors
roi_classes = torch.hstack(gt_classes, classes) # concatenate two 1-d vectors
overlaps = torch.hstack(torch.ones(len(gt_classes)), max_ious)
```

`batch_size=2`

Fast R-CNN 的输入为 image，以及（包含 gt boxes）的 proposals。对 image 的预处理与训练 RPN 相同（各 channel 减去 mean value，resize，然后拷贝到一个 batch 中）。重点看 rois 的处理，
```python
fg_indices = torch.where(overlaps >= 0.5)[0]
rois_per_img = 64       # 128 // batch_size
fg_rois_per_img = int(0.25 * roi_per_img)   # 16
if fg_indices.size > fg_rois_per_img:
    fg_indices = fg_indices[torch.randint(len(fg_indices), (fg_rois_per_img,))]

bg_indices = torch.where(overlaps < 0.5 and overlaps >= 0.1)[0]
bg_rois_per_img = rois_num - fg_rois_per_img
if len(bg_indices) > bg_num:
    bg_indices = bg_indices[torch.randint(len(bg_indices), (bg_rois_per_img,))]

keep = torch.cat((fg_indices, bg_indices), axis=0)
roi_classes = roi_classes[keep]
# class label for any bg proposal is 0
roi_classes[len(fg_indices):] = 0
rois = rois[keep]
overlaps = overlaps[keep]

# calcucate the gt coordinate offsets (based on positive proposals)
gt_indices = torch.where(overlaps == 1)[0]
fg_indices = torch.where(overlaps >= 0.5)[0]
M = bbox_iou(rois[fg_indices,:], rois[gt_indices,:])

# find each positive proposal and its matched gt box
max_ious, gt_assignments = torch.max(M, dim=1)
fg_rois = rois[fg_indices,:]
gt_rois = rois[gt_indices[gt_assignments],:]
# according to eq(1) and eq(2), get the gt coordinate offsets for all candidated proposals
bbox_target = bbox_target(fg_rois, gt_rois)
# predicted bbox is class related, so provides the positive proposals' classes
bbox_classes = roi_classes[fg_indices]  
```
当然，`rois` 还需要乘以 `im_scale` 以与 resized input image 匹配。然后
将每个 image 的 `rois` 数据进行打包，
```python
batched_rois = torch.zeros((0, 5))
for i, rois in enumerate(rois_list):
    batch_ind = torch.ones((rois.shape[0], 1)) * i
    ind_rois = torch.hstack((batch_ind, rois))
    batched_rois = torch.vstack((batched_rois, ind_rois))
```
其他数据如 `bbox_target` 等类似进行打包。



准备好 image 数据以及 target 数据之后，就可以进行前向传播了。 batched image 经过 VGG-16 ，在 `conv5_3` 这个 layer 输出 feature maps，size 大约是 `40x60`（backbone 降采样率为 `16`），然后执行步骤：
1. 在 feature maps 上根据 `rois` crop 出 feature patches，并执行 ROI pooling 得到 `7x7` 的特征
    ```python
    scale = 1/16
    x = conv5_3(x)      # (B, C, H, W)
    B = x.shape[0]
    pool = nn.AdaptiveMaxPool2d((7, 7))
    pooled_feats = []
    for roi in batched_rois:
        ltrb = roi[1:] / scale      # left, top, right, bottom
        # x1, y1, x2, y2
        feat = x[int(roi[0])].unsqueeze(0)
        feat = feat[:, :, ltrb[1]: ltrb[3], ltrb[0]:ltrb[2]]
        pooled_feat = pool(feat)
        pooled_feats.append(pooled_feat)
    x = torch.vstack(pooled_feats)

    # x.shape (N, 512, 7, 7)，N 为 batch image 中所选的 rois 数量：128
    ```

2. 上一步的输出，连续经过两个全连接层 `fc+relu+drop`，输出特征的 shape 为 `(N, 4096)`，这个特征分别经 cls layer（`out_channel=1+C` 的全连接层，其中 `C` 为分类数量，`1` 为背景数量），以及 reg layer （`out_channel=4x(1+C)` 的全连接层）这两个并列输出分支，得到（未归一化）分类得分，以及 bbox 的坐标偏差（与分类有关）。
3. 分类损失和坐标回归损失与 RPN 中的相同，使用 `CrossEntropyLoss` 作为分类损失，`SmoothL1Loss` 作为坐标回归损失。

以上分析过程参照 `alt_opt` 训练方法，即交替训练方法。

# 近似联合训练
源码还提供了 `end2end` 训练方法，即**近似联合训练**方法（真正的端到端训练方法是一个 non trivial 问题，比较复杂，而采用近似端到端训练方法，已经可以取得较好的结果）。近似联合训练思路：
1. backbone , RPN 和 Fast R-CNN 均合并到一个网络中，前向传播时，RPN 生成 proposals，然后使用 proposals 得到 rois，与 backbone 的 feature maps 一起作为 Fast R-CNN 的输入，继续进行前向传播。反向传播时，RPN 和 Fast R-CNN 的损失一起反向传播，用于更新网络参数。