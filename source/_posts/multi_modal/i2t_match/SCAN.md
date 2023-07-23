---
title: SCAN：用于 Image-Text 匹配的堆叠交叉注意力
date: 2023-07-17 16:21:12
tags: image-text retrieve
mathjax: true
---

论文：[Stacked Cross Attention for Image-Text Matching](https://arxiv.org/abs/1803.08024)

源码：[kuanghuei/SCAN](https://github.com/kuanghuei/SCAN)

# 1. 简介

当描述一个图时，通常描述图中目标和其他显著物体区域，以及目标的属性和动作，如图 1，

![](/images/multi_modal/SCAN_1.png)
<center>图 1. 图和描述图的句子</center>

从某种意义上讲，句子描述是弱标注，因为句子中单词对应图中某个特殊区域，但是对应哪个区域并不知道。推断图像区域与单词之间的潜在关联是 image-text 匹配任务的关键。

作者提出 Stacked Cross Attention，在两个阶段（two stages）中从图像和句子的上下文中获取注意力。

作者讨论了两条研究线路，引入注意力机制，在图像区域和单词这一层级上计算视觉语言的潜在关联度，这两条线分别如下，

1. Image-Text 匹配，实验 bottom-up 注意力

    bottom-up 注意力是一个术语，首次提出用于给图像取标题以及视觉问答，指的是视觉反馈注意力机制，类似于人类视觉系统自发的 bottom-up 注意力（例如，人类注意力总是先注意到显著的东西例如目标而非背景）。

2. 基于传统注意力的方法。


# 2. SCAN

本节描述 Stacked Cross Attention Network（SCAN）。我们的目标是将单词和图像区域映射到一个共同的 embedding space，然后推断整个图像与句子之间的相似度。

先使用 bottom-up 注意力检测图像区域并将图像区域编码为特征向量，然后将句子单词以及上下文信息映射成特征向量。

## 2.1 Stacked Cross Attention

Stacked Cross Attention 有两个输入：1. 图像特征集 $V=\lbrace v _ 1, \ldots, v _ k \rbrace, v _ i \in \mathbb R ^ D$ ，每个特征表示一个图像区域；2. 单词特征集 $E=\lbrace e _ 1, \dots, e _ n \rbrace, e _ i \in \mathbb R ^ D$ ，其中每个特征表示句子中的一个单词。Stacked Cross Attention 输出为相似度得分，这个得分测量 image-text pair 的相似度。

以下是 Stacked Cross Attention 两个很赞的构思。

### 2.1.1 Image-Text Stacked Cross Attention

这种方式如图 2 所示，涉及到两个注意阶段。

![](/images/multi_modal/SCAN_2.png)

<center>图 2. Image-Text Stacked Cross Attention</center>

在阶段 1，根据图像区域特征 $v _ i$，注意句子中的单词，生成注意后的句子特征 $a _ i ^ t$，如图 2 中的 stage 1，$v _ 1$ 是坐在洗池中的猫，那么句子中 `cat`、`sitting` 和 `sink` 被注意到，三个单词的权重不同，颜色越深表示权重越大。

在阶段 2，比较图像区域 $v _ i$ 与相应的注意后句子向量 $a _ i ^ t$，计算这两者的相似度 $R(v _ i, a _ i ^ t)$，决定图像区域的重要性，每个图像区域有一个对应的相似度 $R(v _ i, a _ i ^ t)$，所有的相似度经过 Pooling 得到整个图像与文本之间的相似度 $S(I, T)$ 。

具体而言，给定一个图像 $I$，其中检测出 $k$ 个区域，一个句子 $T$，其中有 $n$ 个单词，计算所有的区域单词 pair 的余弦相似度矩阵，

$$s _ {ij} = \frac {v _ i ^ {\top} e _ j}{||v _ i|| \ ||e _ j||}, \ i \in [1,k], j \in [1,n] \tag{1}$$

作者根据经验发现，最好是对相似度进行整流和归一化处理，即

$$\overline s _ {ij} = [s _ {ij}]_ + \ / \sqrt {\sum _ {i=1} ^ k [s _ {ij}] _ + ^ 2}$$

其中 $[x] _ + = \max (x, 0)$ 。

注意，这里是从图像角度出发（命名时 Image 在前，Text 在后），所以求归一化时，对图像区域 index $i$ 求和。

对于某个图像区域，注意句子中与其关联的单词，将单词加权求和，得到注意后的句子的表征向量，

$$a _ i ^ t = \sum _ {j=1} ^ n \alpha _ {ij} e _ j \tag{2}$$

其中权重为

$$\alpha _ {ij} = \frac {\exp (\lambda _ 1 \overline s _ {ij})} {\sum _ {j=1} ^ n \exp (\lambda _ 1 \overline s _ {ij})} \tag{3}$$

$\lambda _ 1$ 是 softmax 中的 scaling 因子。

**注意后句子向量 $a _ i ^ t$ 中 $t$ 表示文本（text），$i$ 表示与图像区域 `i` 关联** 。

然后计算图像区域 $v _ i$ 与注意后的句子向量之间的相关度，使用余弦相关度，即

$$R(v _ i, a _ i ^ t) = \frac {v _ i ^ {\top} a _ i ^ t}{||v _ i|| \ ||a _ i ^ t||} \tag{4}$$

根据语音识别中最小分类误差的思路，图像 $I$ 和句子 $T$ 之间的相关度使用 Log-SumExp pooling (LSE) 计算，

$$S_{LSE}(I, T) = \log \left[ \left(\sum _ {i=1} ^ k \exp (\lambda _ 2 R(v _ i, a _ i ^ t)) \right) ^ {1/\lambda _ 2} \right] \tag{5}$$

其中因子  $\lambda _ 2$ 决定最大的 $R (v _ i, a _ i ^ t)$ 值在 $S_{LSE} (I, T)$ 中的占比。当 $\lambda _ 2 \rightarrow \infty$ 时， $S(I, T) = \max _ {i=1} ^ k \ R(v _ i, a _ i ^ t)$，占比最大为 100%；当 $\lambda _ 2 \rightarrow -\infty$ 时，占比最小，为 0% 。

我们也可以使用 均值 pooling，即

$$S _ {AVG} (I, T) = \frac {\sum _ {i=1} ^ k R(v _ i, a _ i ^ t)} k \tag{6}$$

### 2.1.2 Text-Image Stacked Cross Attention

与 Image-Text Stacked Cross Attention 类似，对某个 word `j`，注意图像中的区域，加权求和得到关于 word `j` 的注意后图像特征，求 word 与注意后的图像之间的相关度，比较所有的相关度（一共 $n$ 个），得到最终的整个句子与图像之间的相关度（ Text 在前，Image 在后）。整个过程如图 3 所示，

![](/images/multi_modal/SCAN_3.png)

<center>图 3. Text-Image Stacked Cross Attention</center>

具体而言，计算区域 `i` 与单词 `j` 之间的余弦相似度，

$$\overline s' _ {ij} = [s _ {ij}] _ + \ / \sqrt {\sum _ {j=1} ^ n [s _ {ij}] _ + ^ 2}$$

注意，这里是从句子角度出发（命名时 Text 在前，Image 在后），所以求归一化时，对单词 index $i$ 求和。

某个单词 `j`，对其关联的图像区域计算加权求和，得到注意后的图像特征，

$$\begin{aligned} a _ j ^ v &= \sum _ {i=1} ^ k \alpha _ {ij} ' v _ i
\\\\ \alpha _ {ij}' &= \frac  {\exp (\lambda _ 1 \overline s _ {ij}')} {\sum _ {i=1} ^ k \exp (\lambda _ 1 \overline s _ {ij} ')}
\end{aligned}$$

**注意后图像向量 $a _ j ^ v$ 中 $v$ 表示视觉（visual），$j$ 表示与单词 `i` 关联** 。

那么单词 `j` 与对应的注意后图像之间的相关度为

$$R'(e _ j, a _ j ^ v) =\frac {e _ j ^ {\top} a _ j ^ v}{||e _ j|| \ ||a _ j ^ v||}$$

使用 LSE 计算图像与句子之间的相似度得分，

$$S _ {LSE}' (I, T)=\log \left[ \left(\sum _ {j=1} ^ n \exp (\lambda _ 2 R'(e _ j, a _ j ^ v))\right) ^ {1/ \lambda _ 2}\right] \tag{7}$$

或者使用 AVG 池化，

$$S _ {AVG}' (I, T) = \frac {\sum _ {j=1} ^ n R'(e _ j , a _ j ^ v)} n \tag{8}$$

**# 对比前人的相似度计算**

前人的研究工作中，曾使用区域向量 $v _ i$ 与单词向量 $e _ j$ 之间的点积作为区域-单词相似度，即 $s _ {ij} = v _ i ^ {\top} e _ j$ ，然后整个句子与图像之间的相似度计算为

$$S _ {SM}'(I, T)=\sum _ {j=1} ^ n \max _ i (s _ {ij}) \tag{9}$$

称 (9) 式为 Sum-Max Text-Image 。类似地有 Sum-Max Image-Text，计算式为

$$S _ {SM} (I, T) = \sum _ {i=1} ^ k \max _ j (s _ {ij}) \tag{10}$$

作者在消融实验中使用了 (9) 和 (10) 式作对比。

## 2.2 匹配的目标函数

Image-Text 匹配任务中，目标函数常使用三元组损失（triplet loss）。前人的研究工作中，使用如下目标函数，

$$l(I,T)=\sum _ {\hat T} [\alpha - S(I, T)- S(I, \hat T)] _ + \ + \sum _ {\hat T}[\alpha - S(I, T)+ S(\hat I, T)]_ + \tag{11}$$

上式中，$\hat T$ 和 $\hat I$ 分别是 minibatch 中不等于 $T$ 的其他文本和不等于 $I$ 的其他图像。

(11) 式中使用 hinge loss，这表明，给定一个 matched pair $(I, T)$，如果其相似度比负例 pair $(I, \hat T)$ 的相似度大超过 $\alpha$，那么 hinge loss 为 0，也就是说，这种负例我们不关心，不使用这些负例来学习；而相似度比正例相似度相差不超过 $\alpha$ 的负例 pairs（也就是说，正负 pairs 有些靠近），这种负例需要用来学习，使得模型能够对其加以区分。

本文作者仅关注 minibatch 中最难负例。例如一个正 pair $(I, T)$，最难负 pair 为

$$\hat I _ h = \argmax _ {m \neq I} S(m, T), \quad \hat T _ h = \argmax _ {d \neq T} S (I, d)$$

上式具体实现思路，给定一个 minibatch，大小为 $B$，计算图像文本相似度矩阵，大小为 $B \times B$，显然矩阵对角线元素为正 pair 的相似度 $S(I, T)$，其他元素均为负 pair 的相似度，将矩阵对角线元素置零，然后按行求最大值，得到 $\hat T _ h$，按列求最大值得到 $\hat I _ h$ 。

本文使用的三元组损失定义为，

$$l _ {hard} (I, T) = [\alpha - S(I, T)+S(I, \hat T _ h)]_ + \ + [\alpha - S(I, T) + S(\hat I _ h, T)] _ + \tag{12}$$

## 2.3 使用 Bottom-up 注意力表示图像

给定一个图像 $I$，我们目标是使用一个特征集合 $V=\lbrace v _ 1, \ldots, v _ k \rbrace, v _ i \in \mathbb R ^ D$ 来表示这个图像，其中每个特征向量表示图像一个区域。目前我们仅关注图像中目标以及其他物体所对应的区域，所以想到使用 Faster R-CNN 检测图中显著目标区域，并称之为 bottom-up 注意力。

采用 Faster R-CNN 和 ResNet-101 结合的模型。模型预测属性类和实例类，实例类包括目标以及其他例如 “天空”，“草地”，“建筑物”，属性类例如 “皮毛的”，这么做是为了得到具有丰富语义的特征表示。

每个选择的区域 `i`，其经过均值池化后的卷积特征记为 $f _ i$，然后经过一个全连接层得到最终的图像区域的特征向量，

$$v _ i = W _ v f _ i + b _ v \tag{13}$$

## 2.4 句子表征

我们需要将句子也映射到相同维度的特征空间，以便可以连接视觉和语言特征。给定一个句子 $T$，一种间的方法是将每个 word 分别映射为向量，但是这种方法没有考虑句子中上下文的语义，作者使用 RNN 解决这个问题。

句子中第 `i` 个单词，先使用 one-hot 向量表示（词典 vocab 已知），然后使用一个嵌入矩阵将 one-hot 向量转为 300-dim 的嵌入向量，嵌入矩阵 $W _ e$，大小为 $|V| \times 300$，其中 $|V|$ 是词典大小。嵌入向量为 $x _ i = W _ e w _ i, \ i \in [1,n]$ 。然后使用 BiGRU 将嵌入向量转为最终的词特征向量，BiGRPU 中前向和后向的隐层节点输出求平均，就是这个最终的词特征向量，

1. 前向

    $$\overrightarrow  h _ i = \overrightarrow {GRU} (x _ i), \ i \in [1,n] \tag{14}$$

2. 反向

    $$\overleftarrow h _ i = \overleftarrow {GRU} (x _ i), \ i \in [1, n] \tag{15}$$

最终的词特征为

$$e _ i = \frac {\overrightarrow h _ i + \overleftarrow h _ i} 2, \ in \in [1,n] \tag{16}$$

# 3. 代码

## 3.1 bottom-up attention 得到图像特征。

使用项目的子模块 `bottom-up-attention` 中的 `tools/generate_tsv.py` 获取图像特征。这里给出关键部分的代码，

**# 生成图像特征**

对于数据集，使用 n 个 GPU，n 等分数据集，每个 GPU 负责一个子集的图像特征生成。

```python
def generate_tsv(gpu_id, prototxt, weights, images_ids, outfile):
    '''
    gpu_id: GPU 编号
    prototxt: 使用的模型原型。模型为：Faster RCNN，其中 backbone 使用 ResNet-101
                参见 vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt
                RCNN 分支也经过修改（也使用了 ResBlock 结构）
    weights: 训练好的模型参数文件
    images_ids: 当前 GPU 负责的子集中的图像
    outfile: 输出文件，保存图像特征
    '''
    ...
    net = caffe.Net(prototxt, caffe.TEST, weights=weights)

    for im_file, image_id in image_ids:
        # 调用这个方法得到图像特征
        d = get_detection_from_im(net, im_file, image_id)
    ...

def get_detection_from_im(net, im_file, image_id, conf_thresh=0.2):
    im = cv2.imread(im_file)
    # Faster-RCNN 与原版略有改动，增加了几个检测 heads
    # scores: 分类得分
    # boxes: region 坐标（proposals 经过 RCNN box head 精调之后的坐标
    # attr_scores: 属性得分（比原版增加的head）参见 Visual Genome 数据集
    # rel_scores: 关系得分（比原版增加的head）参见 Visual Genome 数据集
    scores, boxes, attr_scores, rel_scores = im_detect(net, im)

    # rois: (NUM, 5)
    rois = net.blobs['rois'].data.copy()    # proposals 的坐标
    # 主要是为了得到 im_scales，即，原始图像，到网络输入大小的比例
    # 网络输入 size 为 224x224
    blobs, im_scales = _get_blobs(im, None)

    # 第一列数据表示 object conf。 cls_boxes 的 shape (NUM, 4)
    cls_boxes = rois[:, 1:5] / im_scales[0] # 将 proposals 坐标恢复到基于原始图像大小
    # cls_prob 的 shape (batch_size, num_classes)
    cls_prob = net.blobs['cls_prob'].data   # 这个跟上面的 scores 一样，都是 softmax 后的分类概率得分
    # RCNN 子网络的输出特征（最后经过 GAP）
    pool5 = net.blobs['pool5_flat'].data    # (NUM, 2048)，见下方解释

    max_conf = np.zeros((rois.shape[0]))    # (NUM, )
    for cls_ind in range(1, cls_prob.shape[1]): # 0 为背景分类
        cls_scores = scores[:, cls_ind] # (batch_size,)
        # dets: (batch_size, 5)
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        # 最后 box head 预测输出的坐标是 class 相关的，即输出 num_classes * 4
        # 但是这里使用的是 ROIs，坐标是 class 无关的
        # 每个分类使用相同的 ROI 坐标，进行 NMS（非极大抑制）
        keep = np.array(nms(dets, cfg.TEST.NMS))
        # 保留 keep 位置的 ROI，但是其对应的分类得分，应该使用最大的那个分类得分
        max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

    keep_boxes = np.where(max_conf >= conf_thresh)[0]
    if len(keep_boxes) < MIN_BOXES: # 数量不足 10
        # 检测出来的 region 数量太少，那么根据分类得分倒序排列，取 top 10 的位置
        keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
    elif len(keep_boxes) > MAX_BOXES:   # 100，数量太多
        keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]
    
    return {
        'image_id': image_id,
        'image_h': np.size(im, 0),
        'image_w': np.size(im, 1),
        'num_boxes': len(keep_boxes),
        'boxes': base64.b64encode(cls_boxes[keep_boxes]),
        'features': base64.b64encode(pool5[keep_boxes])
    }
```

rois（即 proposals）对应的图像区域的特征（由 backbone 输出），经过 roipool，得到固定的 `14x14` size 的特征，然后经过一个 conv，使得输出 channel 为 `2048`，即 shape 为 `(NUM, 2048, 14, 14)`，其中 `NUM` 为选择的 ROIs 数量，这是预先确定好的，例如 128，然后根据某种策略筛选 ROIs，然后再经过三个 ResBlock（conv512+conv512+conv2048 三个 conv 组成），输出 shape 保持不变，然后使用一个 global average pooling，使得输出 shape 为 `(NUM, 2048, 1, 1)`，利用 Flatten layer，使得 shape 变成 `(NUM, 2048)`，这就是 `pool5_flat` 。

上述 GAP 的输出特征，经过一个 FC 层，输出为 `(NUM, num_classes)`，这就是未归一化的分类得分。需要注意，`num_classes` 包含了背景分类，对于 Visual Genome 数据集，`num_classes=1601` 。

上述代码中的 `max_conf`，记录被保留的的 ROI 的最大分类得分。例如有 `NUM` 个 ROIs，每个 ROI 有 4 个坐标值，每个 ROI 有 `num_classes` 个分类得分，去掉第一个（第一个表示背景分类）。在每个分类下，分别执行一次 NMS，被保留的 ROIs 记录其最大的分类得分，最后根据被保留的 ROIs 的最终分类得分，筛选最终得分大于阈值 `conf_thresh` 的 ROIs 。