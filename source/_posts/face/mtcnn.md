---
title: MTCNN 论文解读
date: 2024-04-17 09:22:37
tags: face detection
---

论文：[Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/abs/1604.02878)


# 1. 简介

本文提出了一种级联的多任务框架，实现人脸检测和对齐，此方法可以挖掘这两个任务之间的联系，从而提高性能。

级联的 CNN 网络包含三个阶段：

1. 通过一个较浅 CNN 快速生成候选窗口
2. 通过一个较复杂 CNN 去掉大部分不包含人脸的窗口
3. 使用一个更强的 CNN 精调窗口并输出人脸 landmarks

本工作的主要贡献：

1. 提出级联 CNN 用于人脸检测和对齐，设计轻量 CNN 框架达到 real time 性能
2. 使用在线难例挖掘，提高性能
3. 执行大量实验，在 benchmark 上展示了明显的性能提升

# 2. 方法

## 2.1 框架

整个方法管线如图 1 所示，

![](/images/face/mtcnn_1.png)


对于一个图片，首先 resize 到不同尺度，建立一个图像金字塔，这就是 3-stage 级联网络的输入。

**stage 1** 

一个全卷积网络，称作 P-Net (Proposal Network) 用于获取候选窗口，以及它们的 bbox 回归向量，然后使用这些 bbox 回归向量校正候选窗口，再使用 NMS 筛选候选窗口。

**stage 2**

将候选窗口喂给另一个 CNN，称作 R-Net (Refine Network)，进一步过滤掉了大量的错误候选窗口，再进行 bbox 回归校正和 NMS 筛选。

**stage 3**

与 stage 2 类似，但是这个阶段得到更详细的人脸信息。

整个网络结构如图 2 所示，

![](/images/face/mtcnn_2.png)

P-Net 输入 size `12x12x3`

R-Net 输入 size `24x24x3`

O-Net 输入 size `48x48x3`

## 2.2 训练

根据三个任务：人脸/非人脸 分类、bbox 回归和人脸 landmark 定位来训练网络。

**# 人脸/非人脸 分类**

记样本为 $x _ i$，使用交叉熵作为损失函数

$$L _ i ^{det} = -(y _ i ^ {det} \log p _ i + (1- y _ i ^ {det})(1 - \log p _ i)) \tag{1}$$

其中 $p _ i$ 为预测得分，$y _ i ^ {det} \in \{0, 1\}$ 为 gt label。

**# bbox 回归**

对于每个候选窗，预测 box 与最近 gt box 对应，损失函数为，

$$L _ i ^ {box} = ||\hat y _ i ^ {box} - y _ i ^ {box}|| _ 2 ^ 2 \tag{2}$$

其中 $\hat y _ i ^ {box}$ 为 bbox 回归预测，$y _ i ^ {box}$ 为 gt box，包括 left top，height 和 width 。

**# 5 landmarks**

损失函数为

$$L _ i ^ {landmark} = ||\hat y _ i ^ {landmark} - y _ i ^ {landmark}|| _ 2 ^ 2 \tag{3}$$

其中 $\hat y _ i ^ {landmark}$ 为 landmark 回归预测，$y _ i ^ {landmark} \in \mathbb R ^ {10}$ 为 gt landmark。

**# 多源训练**

训练过程中有不同类型的训练图片：人脸、非人脸和部分对齐的人脸，所以损失函数 (1)~(3) 中有的损失函数用不到。例如对于背景窗口，仅仅计算分类损失 $L _ i ^ {det}$，而两个回归损失则不需要，使用样本类型指示器，那么总的训练目标为

$$\min \ \sum _ {i=1} ^ N \sum _ {j \in \{det, box, landmark\}} \alpha _ j \beta _ i ^ j L _ i ^ j \tag{4}$$

其中 $N$ 是训练样本数量，$\alpha _ j$ 为损失类型的权重因子。

本文设置 P-Net 和 R-Net 中使用 $\alpha _ {det} = 1, \alpha _ {box} = 0.5, \alpha _ {landmark} = 0.5$，而 O-Net 中设置 $\alpha _ {det} = 1, \alpha _ {box} = 0.5, \alpha _ {landmark} = 1$ 以便获得更精确的 landmarks 坐标。

$\beta _ i ^ j \in \{0, 1\}$ 是样本类型指示器。

**# 在线难例挖掘**

在人脸分类任务中进行在线难例挖掘。

每个 minibatch 中，根据分类损失排序，选择 top 70% 作为难例，然后使用这 70% 的样本的梯度进行反向传播。实验显示这个策略效果很好。

# 3. 实验

## 3.1 训练数据

1. 负样本

    与所有 gt box 的 IOU 均小于 0.3 的 box

2. 正样本

    与某个 gt box 的 IOU 大于 0.65 的 box

3. part face

    0.4 <= IOU <= 0.65

4. landmark face

    有 5 个 landmark 的人脸

负样本和正样本用于人脸分类任务。正样本和 part face 用于 bbox 回归任务。landmark face 用于 landmark 定位任务。

1. P-Net 

    从 WIDER FACE 数据集的图片中随机 crop 若干 patches，得到正负样本和 part face。从 CelebA 数据集中 crop faces 作为 landmark face。

2. R-Net
    
    使用 first stage 对 WIDER FACE 数据集进行检测，收集正负样本和 part face，对 CelebA 数据集进行检测，收集 landmark face。

3. O-Net

    与 R-Net 类似，但是使用前两个 stage 对数据集检测从而收集 O-Net 的训练数据

# 4. 代码解读

由于官方代码使用 matlab 实现，这里我选择 [MTCNN-Tensorflow](https://github.com/AITTSMD/MTCNN-Tensorflow) 进行讲解。

## 4.1 训练数据

准备训练数据的步骤见项目的说明文档。

对于负样本，gt label 为 `img_path 0`

对于正样本，gt label 为

```sh
img_path 1 offset_x1 offset_y1 offset_x2 offset_y2
```

对于 part face，gt label 为

```sh
img_path 1 offset_x1 offset_y1 offset_x2 offset_y2
```

准备 P-Net 分类任务和 bbox 回归任务的训练数据关键代码

```python
# prepare_data/gen_12net_data.py
for annotation in annotations:  # 遍历每一个图片的标注信息
    annotation = annotation.strip().split(' ')
    im_path = annotation[0] # 文件名
    bbox = list(map(float, annotation[1:])) # bbox 坐标
    boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4) # (n, 4)，n 是此图片中人脸数量
    img = cv2.imread(os.path.join(im_dir, im_path + '.jpg'))
    neg_num = 0
    while neg_num < 50:     # 每个图片上 crop 50 个负样本
        crop_box = np.array([nx, ny, nx+size, ny+size]) # 随机生成的 crop 坐标
        Iou = IoU(crop_box, boxes)

        cropped_im = img[ny:ny+size, nx:nx+size, :] # crop 的图片 patch
        resized_im = cv2.resize(cropped_im, (12, 12))

        if np.max(Iou) < 0.3:   # crop 了一个负样本
            f2.write('../../DATA/12/negative/%s.jpg'%n_idx + ' 0\n')    # 负样本的标注
            cv2.imwrite(save_file, resized_im)  # 保存负样本图片
            ...
    for box in boxes:
        x1, y1, x2, y2 = box
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        # gt box 附近再 crop 5 个 patch，如果 IOU < 0.3，那么保存为负样本
        for i in range(5):
            ...
            crop_box = np.array([nx1, ny1, nx1+size, ny1+size]) # 随机生成的 crop 坐标
            Iou = IoU(crop_box, boxes)
            cropped_im = img[ny1:ny1+size, nx1:nx1+size, :]
            resized_im = cv2.resize(cropped_im, (12, 12))
            if np.max(Iou) < 0.3:
                f2.write('../../DATA/12/negative/%s.jpg'%n_idx + ' 0\n')    # 负样本的标注
                cv2.imwrite(save_file, resized_im)  # 保存负样本图片
                ...
        # 生成正样本和 part face
        for i in range(20):
            ...
            nx2 = nx1 + size
            ny2 = ny1 + size
            crop_box = np.array([nx1, ny1, nx2, ny2])   # 随机，但是 crop 的中心位于 gt box 中心附近 delta_x 和 delta_y 距离处
            offset_x1 = (x1 - nx1) / float(size)
            offset_y1 = (y1 - ny1) / float(size)
            offset_x2 = (x2 - nx2) / float(size)
            offset_y2 = (y2 - ny2) / float(size)
            cropped_im = img[ny1:ny2, nx1:nx2, :] # crop 的图片 patch
            resized_im = cv2.resize(cropped_im, (12, 12))
            box_ = box.reshape(1, -1)
            iou = IoU(crop_box, box_)
            if iou >= 0.65:     # 正样本
                f1.write(...)
                cv2.imwrite(...)
            elif iou >= 0.4:    # part face
                f3.write(...)
                cv2.imwrite(...)
```

注意上面计算正样本和 part face 的 bbox target: `(x1 - nx1) / float(size)`，即两个坐标之差然后归一化。

准备 P-Net landmark 回归任务的训练数据关键代码

landmark 的 gt label 格式为

```sh
img_path -2 x1 y1 x2 y2 ... x5 y2
```

1. 先根据 gt bbox crop 出人脸 patch
2. 根据 gt bbox 重新计算 landmark 点坐标，并归一化
3. 在 patch 内根据 patch 中心随机微扰得到新的中心，以这个新中心 crop 出新 region，并计算相应的 landmark 点坐标


```python
# gen_landmark_aug_12.py
size = 12
for (imgPath, bbox, landmarkGt) in data:
    F_imgs = []
    F_landmarks = []
    img = cv2.imread(imgPath)   # 加载图片数据
    # 此数据集中，一个图片只有一个 gt box，以及 5 个 landmark points
    f_face = img[bbox.top:bbox.bottom+1, bbox.left:bbox.right+1]
    f_face = cv2.resize(f_face, (size, size))
    landmark = np.zeros((5, 2))

    for index, one in enumerate(landmarkGt):    # 遍历 5 个 landmark 点坐标
        rv = ((one[0] - bbox.left)/(bbox.right-bbox.left),
              (one[1] - bbox.top) /(bbox.bottom-bbox.top))  # 归一化 landmark 坐标
        landmark[index] = rv
    F_imgs.append(f_face)
    F_landmarks.append(landmark.reshape(10))
    landmark = np.zeros((5, 2))
    if argument:    # True，实现数据增强
        x1, y1, x2, y2 = bbox
        gt_w = x2 - x1 + 1
        gt_h = y2 - y1 + 1

        for i in range(10): # 对应上面的第 3 点
            bbox_size = npr.randint(int(min(gt_w, gt_h) * 0.8), np.ceil(1.25 * max(gt_w, gt_h)))    # 随机选择 size
            delta_x = npr.randint(-gt_w*0.2, gt_w*0.2)  # 微扰 offset
            delta_y = npr.randint(-gt_h*0.2, gt_h*0.2)
            # nx1 + bbox_size/2 是 crop 的中心 x 坐标，等于 patch 中心 + delta_x
            nx1 = int(max(x1+gt_w/2-bbox_size/2+delta_x, 0))
            ny1 = int(max(y1+gt_h/2-bbox_size/2+delta_y, 0))
            nx2 = nx1 + bbox_size
            ny2 = ny1 + bbox_size

            crop_box = np.array([nx1, ny1, nx2, ny2])
            cropped_im = img[ny1:ny2+1, nx1:nx2+1, :]
            resized_im = cv2.resize(cropped_im, (size, size))

            iou = IoU(crop_box, np.expand_dims(gt_box, 0))
            if iou > 0.65:
                F_imgs.append(cropped_im)
                # 基于 crop region 归一化 landmark 坐标
                ...
                # 数据增强
                if random.choice([0, 1]) > 0:   # mirror
                    # 将 resized_im flip
                    ...
                if random.choice([0, 1]) > 0:   # rotate
                    ...
```

这里有两个数据集，第一个收集了正负样本和 part face，第二个收集了 landmark face。

## 4.2 训练

## 4.2.1 P-Net

训练入口为 `train_PNet.py` 。

获取 batch 数据，

```python
# read_tfrecord_v2.py/read_single_tfrecord
...
label = tf.reshape(label, [batch_size]) # (B,) 0/1/-1/-2
roi = tf.reshape(roi, [batch_size, 4])  # (B, 4)
landmark = tf.reshape(landmark, [batch_size, 10])   # (B, 10)
```

如果某类型样本缺乏某个数据，那么对应的 target 值 为 0，例如正样本没有 landmark 点坐标，或者 landmark face 样本缺乏 bbox offset（即 `roi`） 数据。

P-Net 网络的结构比较简单，这里不再解释，仅说明计算 loss 的代码，如下

```python
# mtcnn_model.py/P_Net
cls_prob = tf.squeeze(conv4_1, [1, 2], name='cls_prob') # (B,1,1,2)->(B,2) 参见图 2
# label: 分类 gt ，(B,)
cls_loss = cls_ohem(cls_prob, label)
```

分类任务的在线难例挖掘代码，与上文 2.2 一节中关于 OHEM 的说明一致。

```python
# mtcnn_model.py
def cls_ohem(cls_prob, label):
    zeros = tf.zeros_like(label)
    # pos->1, neg->0, others(part, landmark)->0
    label_filter_invalid = tf.where(tf.less(label, 0), zeros, label)
    ...
    label_prob = tf.squeeze(tf.gather(cls_prob_reshape, indices_))  # 根据 gt 获取对应的预测得分，part 和 landmark 与 neg 一样均使用 p_i(0)，pos 使用 p_i(1)
    loss = -tf.log(label_prob + 1e-10)  # -\log p_i
    ...
    valid_inds = tf.where(label < zeros, zeros, ones)   # 只考虑 pos 和 neg
    num_valid = tf.reduce_sum(valid_inds)   # pos neg 的数量
    keep_num = tf.cast(num_valid * num_keep_radio, dtype=tf.int32)  # top 70%
    loss = loss * valid_inds    # 提取 pos 和 neg 的 loss
    loss, _ = tf.nn.top_k(loss, k=keep_num)
    return tf.reduce_sum(loss)
```

bbox 回归任务的 OHEM 代码如下，实际上是使用了全部的 pos 和 neg，丢弃了 part face 和 landmark face。

```python
# mtcnn_model.py
def bbox_ohem(bbox_pred, bbox_target, label):
    # 只考虑 pos(1) 和 part(-1)
    valid_inds = tf.where(tf.equal(tf.abs(label), 1), ones_index, zeros_index)
    square_error = tf.square(bbox_pred - bbox_target)   # L2 范数, (B, 4)
    square_error = tf.reduce_sum(square_error, axis=1)  # (B,)
    num_valid = tf.reduce_sum(valid_inds)   # pos neg 的数量
    keep_num = tf.cast(num_valid, dtype=tf.int32)  # 保留全部 pos 和 neg
    square_error = square_error * valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)
```

landmark 回归任务的 OHEM 则类似的，使用了全部的 landmark face，丢弃了正负样本和 part face。


### 4.2.2 R-Net

训练数据准备

使用 P-Net 检测 Wider face 数据集，从而收集训练 R-Net 的数据。具体步骤：

1. 加载 Wider face 所有图片路径
2. 加载 P-Net 模型，为 Wider face 每个图片进行预测，根据预测值生成 box。注意这里是逐步缩小原始图片尺寸，然后喂给 P-Net（此即图片金字塔）。 

    ```python
    # mtcnn_model.py/P_Net
    # test 模式下，batch size 为 1
    cls_pro_test = tf.squeeze(conv4_1, axis=0)  # (h, w, 2) 
    bbox_pred_test = tf.squeeze(bbox_pred, axis=0)  # (h, w, 4)
    landmark_pred_test = tf.squeeze(landmark_pred, axis=0)  # (h, w, 10)
    ```

    注意 train 模式下，由于 network input size 为 crop 的 `12x12`，输出 size 刚好为 `1x1`，但是 test 模式下，输入 size 是变化的，是将整个原始图片通过尺度变换生成图片金字塔，所以输入 size 大于等于 `12x12`，那么输出 size 则大于等于 `1x1`。图片金字塔作为输入的代码为，

    ```python
    # MtcnnDetector.py/detect_pnet 方法
    net_size = 12   # network 输入 size 为 12
    current_scale = float(net_size) / self.min_face_size    # 最小人脸尺寸，手动设置为 20
    im_resized = self.processed_image(im, current_scale)    # 将原始图片缩放 0.6
    current_height, current_width, _ = im_resized.shape
    all_boxes = list()

    while min(current_height, current_width) > net_size:    # 当前level的输入图片尺寸不小于 12，否则输出 size 为 0
        cls_cls_map, reg = self.pnet_detector.predict(im_resized)
        # 根据预测值生成 box（需要先对预测得分进行阈值筛选）
        boxes = self.generate_bbox(cls_cls_map[:,:,1], reg, current_scale, self.thresh[0])
        current_scale *= self.scale_factor  # 进一步缩小图片（从而得到图片金字塔）
        im_resized = self.processed_image(im, current_scale)
        current_height, current_width, _ = im_resized.shape
        ...
        keep = py_nms(boxes[:, :5], 0.5, 'Union')   # NMS 筛选
        boxes = boxes[keep]
        all_boxes.append(boxes)
    # 所有金字塔结构的图片的预测结果再进行 NMS 筛选
    keep = py_nms(all_boxes[:, :5], 0.5, 'Union')
    all_boxes = all_boxes[keep]
    boxes = all_boxes[:, :5]
    ```

3. 经过 P-Net 预测出来的 box 即， P-Net 认为是正样本，再作为 R-Net 的输入，经过 R-Net 的预测，其他部分可能就预测为负样本，从而达到 refine 的目的。

    P-Net 预测出来的 box，再根据与 gt box 的 IoU 判断出是正样本还是负样本，还是 part face，判断出来之后保存到文件，作为 R-Net 的训练数据。代码见 `gen_hard_example.py/save_hard_example` 方法。

4. 使用 `gen_landmark_aug_24.py` 生成 R-Net 的 landmark 训练数据。

5. 最后所有的数据一起，作为 R-Net 的训练数据。训练代码与 P-Net 的类似。

### 4.2.3 O-Net

生成 O-Net 训练数据的过程与 R-Net 的类似，代码位于 `gen_hard_example.py`，下面我们列出其中的几个关键变量的值，

```python
# gen_hard_example.py
net = 'RNet'    # 生成 R-Net 训练数据时设置
net = 'ONet'    # 生成 O-Net 训练数据时设置
test_mode = 'PNet'  # 生成 R-Net 训练数据时设置
test_mode = 'RNet'  # 生成 O-Net 训练数据时设置
slide_window = False    # 设置为 False 就好（for R-Net and O-Net）
```

上一节讲到生成 R-Net 的训练数据时，使用 P-Net 对图片进行预测，那么生成 O-Net 训练数据时，除了第一步使用 P-Net 对图片进行预测之外，还需要第二步，使用 R-Net 对 P-Net 的预测再次进行预测，

```python
# MtcnnDetector.py/detect_face
if self.pnet_detector:
    _, boxes_c, landmark = self.detect_pnet(im)

if self.rnet_detector:
    _, boxes_c, landmark = self.detect_rnet(im, boxes_c)
```

其中 `detect_rnet` 方法所作的事情为：

1. 根据 P-Net 的预测 box，对原始图片 `im` 进行 crop，并调整 crop 后的图片 size 为 `24x24`
2. 使用 R-Net 再进行预测，根据一个预测得分阈值进行筛选，然后使用 NMS 筛选，然后使用 R-Net 的预测 bbox offset 对 P-Net 的预测 box 进行校正


## 4.3 检测人脸

```python
# one_image_test.py

all_boxes, landmarks = mtcnn_detector.detect_face(test_data)
```

与上一节生成 O-Net 的训练数据类似，在此基础上，即 R-Net 对 P-Net 的预测进行 refine 之后，再喂给 O-Net，

```python
# MtcnnDetector.py
def detect_onet(self, im, dets):
    ... # 代码不再具体分析了，比较简单
```