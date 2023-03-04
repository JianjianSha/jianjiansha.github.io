---
title: PointRCNN 论文解读（一）
date: 2022-10-24 14:35:41
tags: 3d object detection
mathjax: true
---

论文：[PointRCNN:3D Object Proposal Generation and Detection from Point Cloud](https://arxiv.org/abs/1812.04244)
源码：[sshaoshuai/PointRCNN](https://github.com/sshaoshuai/PointRCNN)

# 1. 简介

自动驾驶中，最常用的 3d 传感器是 LiDAR，可以生成 3D 点云。传统的处理点云的方法是：投影到鸟瞰图（bird eye view）或者正视图（frontal view），或者是生成规则的 3D voxels，但是这些处理方法都会丢失点云的部分信息。

PointRCNN 直接在点云上操作而省去了预处理步骤。PointRCNN 是二阶段的 3D 检测框架：

1. stage 1 用于生成 3D bbox proposal，自底向上
2. stage 2 操作正规的 3D box 精调

# 2. PointRCNN

## 2.1 自底向上生成 3D proposal

由于巨大的 3D 搜索空间以及点云数据的不规则性，二阶段目标检测从 2D 转换到 3D 是非凡的。

stage 1 以自底向上的方式生成 3D proposal，那么具体是如何生成 3D proposal 的呢？

1. 框架学习并获得点特征（Encoder+Decoder）

    使用 [pointnet++](/2022/10/20/obj_det/3d/pointnet++) 学习点特征，使用 multi-scale grouping (MSG) 的 set abstraction （下文使用简写 SA） 作为 encoder，上采样和特征传播作为 decoder。

2. 根据点特征，生成 bin-based 3D box 和对前景点的分割（point-wise 二分类）

    3D 场景下，各个目标相互之间没有重叠，根据目标的 3D 标注可以获得目标的分割掩码（segmentation mask）(二分类的 ground truth 的分割掩码)，也就是说将位于 3D 标注框之内的点均视作前景点，从而得到 gt 分割掩码。


如何 1 (a) 所示，

![](/images/obj_det/3d/point_rcnn_1.png)

<center>图 1</center>




### 2.1.1 前景分割

前景点提供了丰富的信息用于预测目标位置和目标偏向角。

经过 backbone 网络的 encode+decode，可以得到 point-wise feature（每个点的特征），在点特征上应用一个 segmentation head，从而得到前景点（point-wise 的二分类），在点特征上另外再应用一个 regression head 用于生成 3D proposal。

这里点分割的 ground-truth 可以通过 3D 标注 box 获取，即位于 3D gt box 内的点均为前景点。对于户外场景而言，背景点数量远大于前景点，为了均衡样本，采用 focal loss，

$$\mathcal L_{focal}(p_t)= -\alpha_t (1-p_t)^{\gamma} \log p_t \tag{1}$$

其中

$$p_t=\begin{cases} p & y=1 \\\\ 1- p & y=0 \end{cases}, \quad\quad \alpha_t = \begin{cases} 0.75 &y=1 \\\\ 0.25 & y=0\end{cases}$$

分别是预测得分和权重因子，$\gamma=2$ 是超参数——幂因子。

### 2.1.2 生成 bin-based 3D bbox

与 segmentation head 并列应用到 point-wise 特征之上的，还有 regression head，用于生成 3D proposals。

回归分支仅根据前景点的特征回归 3D box 的位置。注意到虽然不从背景点中回归，但是背景点也提供了支撑信息，毕竟获取 point-wise 特征的过程中网络的感受野也包含了背景点。

在 LiDAR 下，3D box 可表示为 $(x,y,z,h,w,l,\theta)$，其中 $(x,y,z)$ 表示为目标中心位置，$(h,w,l)$ 表示目标的高宽长，以及 $\theta$ 表示鸟瞰图下目标的偏向角。

如图 2，注意这个图是俯视图，即 X 和 Z 轴构成实际的水平面，Y 轴是实际的竖直方向。

![](/images/obj_det/3d/point_rcnn_2.png)
<center>图 2</center>

下面我们来说明如何估计目标的中心位置。

将前景点的周围区域沿着 X 和 Z 轴切分为一系列的离散 bins。对每个前景点的 X 轴和 Z 轴分别设置一个搜索范围 $\mathcal S$，每个 1D 搜索范围被分为长度固定为 $\delta$ 的若干 bins，这样对 bin 做分类二分类预测：目标中心是否位于 bin 内，使用交叉熵损失。也就是说，对坐标进行离散化，然后转化为分类问题。

但是离散化坐标为 bin-based 分类，预测的坐标不够准确，所以还需要对分类出的 bin 内做残差回归。 对 y 坐标，直接使用坐标残差回归，损失使用 smooth L1，这是因为大部分目标的 y 坐标的范围较小。

目标定位的数学表达式如下，

$$\text{bin}_x^{p}=\lfloor \frac {x^c - x^{p}+\mathcal S} \delta \rfloor, \quad \text{bin}_z^{p}=\lfloor \frac {z^c - z^{p}+\mathcal S} \delta \rfloor \tag{2}$$

$$\text{res}_u^{p} = \frac 1 {\mathcal C} \left( u^c -u^{p} + \mathcal S - \left( \text{bin}_u^{p} \cdot \delta +\frac \delta 2 \right) \right), \quad u \in \{x, z\} \tag{3}$$

$$\text{res}_y^{p} = y^c - y^{p} \tag{4}$$

其中，$(x^{p}, y^{p}, z^{p})$ 是某前景点的坐标，$(x^c, y^c, z^c)$ 是相关目标的中心点坐标，$\text{bin}_x^{p}$ 和 $\text{bin}_y^{p}$ 分别是 X 和 Z 轴的 ground-truth bin。 $\text{res}_x^{p}$ 和 $\text{res}_z^{p}$ 分别是关联 bin 内的 ground-truth residual 值。

对某个前景点，我们以其 x 坐标为中心，沿 X 轴搜索一个范围 $\mathcal S$，那么固定 bin 的长度为 $\delta$，就可以确定搜索范围沿 X 轴有几个 bin，例如图 2 中，X 轴有 6 个 bin，对 Z 轴同样处理，图 2 中 Z 轴也有 6 个 bin，所以一共预测 $6+6=12$ 的分类得分，而 target 则是两个长度均为 6 的 one-hot 向量，确定了 $\text{bin}_x^{p}$ 和 $\text{bin}_y^{p}$ 就可以知道这两个 one-hot 向量中哪个位置元素为 1。

如何理解 (2) 和 (3) 式中的 $\mathcal S$ 呢？ 。仍以图 2 为例，固定 bin 长度 $\delta$ 后，设置搜索范围为 $S=3\delta$ 即，沿 X 轴搜索范围为 $[-S,S)$，一共 6 个 bin，X 轴移动步长 为 $\delta$，那么当 $x \in [-3\delta, -2\delta)$ 时，对应 0-index 的 bin（约定 bin index 从 0 开始），移动 $\delta$ 步长后，$x \in [-2\delta, -1 \delta)$，此时对应 1-index 的 bin，以此类推。

另外一方面，根据 (2) 式易知 

$$0 \le x^c - x^{p}+\mathcal S < \delta \Rightarrow \text{bin}_x^{p}=0$$

$$\delta \le x^c - x^{p}+\mathcal S < 2\delta \Rightarrow \text{bin}_x^{p}=1$$


图 2 中最左边的汽车为例，gt bin 的 x 下标为 $\text{bin}_x^{p}=1$ (下标从 0 开始，从左往右)，而 $-2 \delta \le x^c - x^p < -\delta$ （即汽车中心X坐标 - 当前点X坐标，以当前点为原心，在 local region 内进行搜索），由于设置了搜索范围 $S=3\delta$，那么 $x^c - x^{p} + \mathcal S \in [\delta, 2\delta)$，与 $\text{bin}\_x^{p}=1$ 相符。

(3) 式中 $\text{res}\_u^{p}$ 的范围是 $[-\frac {\delta} {2\mathcal C}, \frac {\delta}{2\mathcal C})$ ，取 $\mathcal C=\delta$ 可归一化残差值到范围 $[-\frac 1 2, \frac 1 2)$。

俯视图偏向角 $\theta$ 和大小 $(h,w,l)$ 的估计则与 Frustum PointNets 类似。将偏向角范围 $2\pi$ 分成 n 个 bins，然后计算 target bin $\text{bin}\_{\theta}^{p}$ 和残差 $\text{res}\_{\theta}^{p}$（与 x, z 坐标预测相似）。目标 size $(h,w,l)$ 则直接计算出与训练集中相同类的平均目标 size 之间的残差值 $(\text{res}\_h^p, \text{res}\_w^p, \text{res}\_l^p)$，然后对这些残差值回归。

作者实验中使用了 bin size $\delta=0.5m$ 以及搜索范围 $S=3m$（m 指距离单位米，$\mathcal S$ 是半径大小），于是 X,Z 轴的 的 bin 均为 12 个。偏向角的 bin 数量配置为 $n=12$，那么对每个前景点，回归 head 一共需要预测的数据量为

$$4\times 12 + 1 + 2 \times 12 + 3=76 \tag{a}$$

表示 12 个 bin 的 X,Z 的 bin 分类预测得分和 bin 内残差值回归，然后是 1 个 y 坐标的残差值，12 个偏向角 bin 的分类预测和 bin 内残差值回归，最后是 3 个 h,w,l 的残差值。

**inference** 阶段，对那些 bin-based 预测参数即，$x,z,\theta$，首先选择最高 confidence 的 bin，然后在加上预测的残差值，以便得到更精确的值。对其他直接回归的参数即 $y,h,w,l$，将预测的残差值加上初始参照值即可。

整个 3D box 回归 head 的损失计算如下：

$$\begin{aligned}\mathcal L_{bin}^{p} &= \sum_{u \in \{x,z,\theta\}} [\mathcal F_{cls}(\hat {bin}\_u^{p}, bin_u^{p})+\mathcal F_{reg}(\hat {res}\_u^{p}, res_u^{p})]
\\\\ \mathcal L_{reg}^{p} &=\sum_{v \in \{y,h,w,l\}} \mathcal F_{reg}(\hat {reg}\_v^{p}, res_v^{p})
\\\\ \mathcal L_{reg} &= \frac 1 {N_{pos}} \sum_{p \in pos} (\mathcal L_{bin}^{p}+\mathcal L_{reg}^{p})
\end{aligned} \tag{5}$$

其中 $N_{pos}$ 是前景点数量。带 ^ 的符号的表示相应的预测值。$\mathcal F_{cls}$ 是交叉熵损失，$\mathcal F_{reg}$ 是 smooth L1 损失。$\hat {bin}\_u^{p}$ 的在前景点 p 关于维度 u 的预测 bin，可以看作是一个预测得分向量，作者实验中每个维度（$X,Z,\theta$）的 bin 均为 12 个，即 $\hat {bin}\_u^{p}$ 是长度为 12 的向量，而 target $bin_u^{p}$ 则是长度为 12 的 one-hot 向量。

假设预测得分向量是归一化的（经过了 softmax），那么交叉熵和残差值回归损失计算过程为

1. 获取 one-hot target 中元素 1 的下标，记为 $i$
2. 取预测得分向量中对应位置的预测概率记为 $P_i$，然后求 $-\log P_i$ 就是交叉熵损失
3. 取 $i$ 位置的预测残差值 $\hat {reg}\_u^{p}$，以及实际的残差值 target $reg_u^{p}$，计算 smooth L1 就是残差值回归损失

每个前景点均进行预测，显然会产生很多 bbox，所以还需要使用基于鸟瞰图的 oriented IoU 的 NMS 去除冗余 bbox。作者实验中，IoU 阈值取 0.85，NMS 之后仅保留 top 300 的 proposals 用于 stage 2。inference 阶段，oriented NMS 的 IoU 阈值取 0.8，然后取 top 100 proposal 用于 stage 2。

## 2.2 点云区域池化

得到 3D proposal 之后（**这之后就由 RCNN 处理**），进一步对这些 3D proposal 的 location 和偏向精细调整（refine）。为了学习到每个 proposal 更具体的 local 特征，作者提出，根据每个 3D proposal 池化其内部的 3D 点数据以及相应的来自 stage 1 的点特征。

使用 $\mathbf b_i = (x_i,y_i,z_i,h_i,w_i,l_i,\theta_i)$ 表示一个 3D proposal，稍微放大它得到一个新的 3D box $\mathbf b_i^e=(x_i,y_i,z_i, h_i+\eta, w_i+\eta, l_i+\eta, \theta_i)$，放大 3D box 是为了更好地将上下文也编码进去。

对于每个点 $p=(x^p, y^p, z^p)$，使用 inside/outside 测试其是否在 $\mathbf b\_i^e$ 内部。如在其内部，那么这个点以及其特征用于 refine $\mathbf b_i$ 。对于一个内部点 p，其特征包含:

1. 3D 点坐标 $(x^p, y^p, z^p) \in \mathbb R^3$
2. 激光反射强度 $r^{p} \in \mathbb R$
3. 预测的分割掩码 $m^p \in \{0,1\}$（来自 stage 1 的 segmentation head 输出）
4. C-dim 的点特征 $\mathbf f^p \in \mathbb R^C$ （图 1 (a)， stage 1 的 Point-wise feature vector）。

## 2.3 正规 3D bbox 精调

如图 1 (b)，每个 3D proposal 放大的 bbox 内部点坐标，特征，mask，经过 Point Cloud Region Pooling 池化后，喂给 stage 2 子网络，然后精调 3D box 位置和前景 object confidence。

### 2.3.1 正规变换

将每个 proposal 池化后的点转换为正规坐标系，如图 3，

![](/images/obj_det/3d/point_rcnn_3.png)
<center>图 3. 每个 proposal 池化后额点被转换到相应的正规坐标系，以便更好的学习 local 空间特征</center>

一个 3D proposal 所对应的正规坐标系（CCS）是指：

1. 原点位于 3D proposal 的中心
2. X'和 Z' 轴几乎平行于地平面，其中 X' 指向 proposal 的朝向，Z' 与 X' 垂直
3. Y' 轴与 LiDAR 坐标系统中相同

对 proposal 中的点 p，进行适当的旋转和平移，将 p 变换到 CCS 中，记为 $\tilde p$ 。使用 CCS 可以使得 stage 2 网络学习每个 proposal 更好局部空间特征。

### 2.3.2 学习特征精调 proposal

尽管上述正规变换让 local 空间特征学习更加鲁棒，但是却也丢失了 depth 信息，这是因为平移 proposal 使得 CCS 的原点位于 proposal 的中心位置。在 LiDAR 坐标系中，距离远的目标其点数较少，距离近的目标其点数较多，但是经过正规转换，所有 proposal 内部的点均在坐标系原点附近，所以丢失了 depth 信息，为了补偿丢失的 depth 信息，作者引入了 proposal 与 sensor 之间的距离，如下，将距离 $d^{p}$ 加入到点 p 的特征中。

$$d^{p} = \sqrt {(x^p )^2 + (y^p )^2 + (z^p )^2 } \tag{6}$$


每个 proposal 的关联点的 local 空间特征 $\tilde p$ 以及额外特征 $[r^{p}, m^{p}, d^{p}]$ （反射强度，点分割掩码，距离）concatenate 然后喂给 MLP（如图 1 (b) 中的红色方框），encode 得到与 global 特征 $\mathbf f^{p}$ 相同维度的特征，然后 local 特征和 global 特征继续 concatenate （如图 1 (b) 中的黑色方框）并喂给一个 Encoder 网络（图 1 (b) 中的梯形框），这个 Encoder 网络与 PointNet++ 中的相同（多个 SA 组成的网络），最后得到具有判别能力的特征向量，用于 confidence 分类和 box 精调。

### 2.3.3 proposal 精调的损失

对 proposal 精调采用类似的 bin-based 回归损失。如果 3D proposal 与某个 gt box 的 3D IoU 大于 0.55，那么这一对 box 可用于训练。3D proposal 与其对应的 3D gt box 均被转换到 CCS，这说明 3D proposal $\mathbf b_i=(x_i, y_i, z_i, h_i, w_i, l_i, \theta_i)$ 和 3D gt box $\mathbf b_i^{gt} = (x_i^{gt}, y_i^{gt}, z_i^{gt}, h_i^{gt}, w_i^{gt}, l_i^{gt}, \theta_i^{gt})$ 均被转换为

$$\begin{aligned}\tilde {\mathbf b}\_i & = (0,0,0, h_i, w_i, l_i, 0)
\\\\ \tilde {\mathbf b}\_i^{gt} & = (x_i^{gt} - x_i, y_i^{gt} - y_i, z_i^{gt} - z_i, h_i^{gt}, w_i^{gt}, l_i^{gt}, \theta_i^{gt} - \theta_i)
\end{aligned} \tag{7}$$

CCS 中，proposal 中心点为原点，绕 Y 轴的偏向角也是 0，但是长宽高不变。

XZ 两个坐标和偏向角 $\theta$ 采用 bin-based 回归，剩下的 Y 坐标和 h，w，l 则直接回归残差值。

记第 `i` 个 proposal 的中心位置 target 为 $(bin_{\Delta x}^i, bin_{\Delta z}^i, res_{\Delta x}^i, res_{\Delta z}^i, res_{\Delta y}^i)$，计算方式与 (2) (3) (4) 式类似，只是搜索范围 $\mathcal S$ 变小。

计算 gt box 的size $h_i^{gt}, w_i^{gt}, l_i^{gt}$ 与训练集中同类的目标 size 的均值之间的残差 $(res_{\Delta h}^i, res_{\Delta w}^i, res_{\Delta}^i)$，并对其回归。

精调偏向角，我们假设 $\theta_i^{gt} - \theta_i$ 在范围 $[-\frac \pi 4, \frac \pi 4]$ 之内，因为 proposal 与 gt box 之间的 3D IoU 至少为 0.55，所以这个角度残差范围的假设是合理的。我们将这 $\pi /2$ 的范围划分为离散的 bin，bin 宽度为 $\omega$，bin-based 朝向的 target 计算如下，

$$\begin{aligned} bin_{\Delta \theta}^i &= \lfloor \frac {\theta_i^{gt} - \theta_i + \frac \pi 4} \omega \rfloor
\\ res_{\Delta \theta}^i &= \frac 2 \omega \left(\theta_i^{gt} - \theta_i  + \frac \pi 4 - (bin_{\Delta \theta}^i \cdot \omega + \frac \omega 2 \right)
\end{aligned} \tag{8}$$

上式中 $\theta_i^{gt} - \theta_i$ 是正规变换后的 gt 朝向角，$+\frac \pi 4$ 是因为搜索范围起始值为 $-\frac \pi 4$，需要减去这个起始值 $-(-\frac \pi 4))=+\frac \pi 4$ 从而得到相对于搜索范围起始值，bin index 从 0 开始（否则 bin index 从负数开始）。

$\theta_i^{gt} - \theta_i  + \frac \pi 4 -bin_{\Delta \theta}^i \cdot \omega$ 是 bin 内的残差值，范围为 $[0, \omega)$，然后再 $-\frac \omega 2$ 使得残差值范围调整为 $[-\frac \omega 2, \frac \omega 2)$ 。

stage 2 的总损失为

$$\mathcal L_{refine} = \frac 1 {\| \mathcal B \|} \sum_{i \in \mathcal B} \mathcal F_{cls} (prob_i, label_i)+ \frac 1 {\|\mathcal B_{pos}\|} \sum_{i \in \mathcal B_{pos}} (\tilde {\mathcal L}\_{bin}^{(i)} + \tilde {\mathcal L}\_{res}^{(i)}) \tag{9}$$

其中 $\mathcal B$ 是来自 stage 1 NMS 之后 3D proposal 集合，$\mathcal B_{pos}$ 是 $\mathcal B$ 中被 stage 2 预测为正例的 proposal 集合。$prob_i$ 是 $\tilde {\mathbf b}_i$ 的 confidence 预测，$label_i$ 是相应的 label。

# 3. 源码解析

## 3.1 训练

### 3.1.1 生成 gt 数据库

```sh
python generate_gt_database.py --class_name 'Car' --split train
```

1. 获取所有训练数据。每个训练样本有唯一的 id，根据这个 id 可以获取点云，label 等。
2. 每个样本 label 中有多个 objects，筛选 `class_name` 所指定分类的 objects。
3. 每个样本中，M 个 objects 的 box3d，对每个 object 的 box3d，筛选出 inside box3d 的点云
4. 对每个 object，构造如下一个字典

    ```python
    sample_dict = {
        'sample_id': sample_id,             # 样本 id，6 位数字
        'cls_type': obj_list[k].cls_type,   # 样本中当前 object 的分类
        'gt_box3d': gt_boxes3d[k],          # gt box，包括中心点坐标 xyz，h,w,l,ry
        'points': cur_pts,                  # 当前 box3d 内的点集坐标，(., 3)
        'intensity': cur_pts_intensity,     # 当前 box3d 内的点反射强度，(.,)
        'obj': obj_list[k]                  # 当前 object 对应的Object3d 类实例，记录了对应的 label 值
    }
    ```
5. 所有样本的所有 object 的 `sample_dict` 存储到一个 list 中，然后 dump 到文件中

**点是否在 3d box 内**

```c++
// x,y,z 点的坐标
// cx, bottom_y, cz，底部中心点坐标
// h,w,l 高宽长（沿 Y Z X 上的 size）
// angle: 物体的朝向角
int pt_in_box3d_cpu(float x, float y, float z, float cx, float bottom_y, float cz, float h, float w, float l, float angle) {
    float max_dis = 10, x_rot, z_rot, cosa, sina, cy;
    // 假设最大距离是 10 米(in X-Z plane)。超过 10 mi 距离的，通通考虑是3d box 之外
    int in_flat;
    // 物体中心的 y 坐标。由于 Y 正向向下，参见 dataset_kitti 一文的图 6。所以 h=bottom_y - top_y =>
    // cy=(bottom_y+top_y)/2 = (bottom_y+bottom_y-h)/2 =bottom_y - h/2
    cy = bottom_y - h / 2.0;    
    // 物体绕 y 轴旋转，所以点与物体中心的 Y轴投影的距离超过 h/2 ，点必然不在 3d box 内
    if ((fabsf(x - cx) > max_dis) || (fabsf(y - cy) > h / 2.0) || (fabsf(z - cz) > max_dis)) {
        return 0;   // 不在 3d box 内
    }

    // 投影到 X-Z 平面，由于 box 可能是斜的（物体朝向不是camera坐标系 Z轴方向），所以需要先进行旋转
    // 参见图 4.
    cosa = cos(angle); sina = sin(angle);
    x_rot = (x - cx) * cosa + (z - cz) * (-sina);
    z_rot = (x - cx) * sina + (z - cz) * cosa;
    in_flag = (x_rot >= -l/2.0) & (x_rot <= l/2.0) & (z_rot >= -w/2.0) & (z_rot <= w/2.0);
    return in_flag;
}
```

![](/images/obj_det/3d/point_rcnn_4.png)
<center>图 4. 逆时针旋转 rotation_y 角度（具有正负性）后，p 到 p'，box 实线框变成虚线框，此时很容易判断 p' 是否在虚线框内</center>

### 3.1.2 训练 RPN

```sh
python train_rcnn.py --cfg_file cfgs/default.yaml --batch_size 16 --train_mode rpn --epochs 200
```

#### 3.1.2.1 数据集

数据集类为 `KittiRCNNDataset`，load 上一步 dump 的 gt_database 文件，并按难易两种模式，将 sample object 的 gt 分成两个 list。将符合要求（即存在相关类的 object）的 sample id 存储至 `sample_id_list` 中。

我们看一下 `__get_item__(self, index)` 函数如何组装训练数据，

```python
def __get_item__(self, index):
    if cfg.RPN.ENABLED:
        return self.get_rpn_sample(index)
    elif cfg.RCNN.ENABLED:
        ...
    else:
        raise NotImplementedError
```

这里先看 RPN 的训练过程，故接着看 `get_rpn_sample` 的函数，

```python
def get_rpn_sample(self, index):
    sample_id = int(self.sample_id_list[index]) # 获取样本 id
    if sample_id < 10000:
        calib = self.get_calib(sample_id)   # 此样本对应得校正
        img_shape = self.get_image_shape(sample_id)
        pts_lidar = self.get_lidar(sample_id)
        # 雷达坐标系转 rectified 相机坐标系，参见 dataset_kitti 文章关于 kitti 的介绍
        pts_rect = calib.lidar_to_rect(pts_lidar[:,:3])
        pts_indensity = pts_lidar[:,3]
    else:
        # 辅助数据，这里略
        ...
    # rectified 相机坐标系转 2号相机图像像素坐标系；0 号机修正坐标系的深度（z坐标）
    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    # 对于点云数据 pts_rect，筛选：1. y,x 坐标在 img_shape 内；2. z 坐标大于 0；3. x,y,z
    # 位于 cfg.PC_AREA_RANGE 内。筛选出同时满足以上 3 个条件的点
    pts_valid_flag = self.get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape)
    pts_rect = pts_rect[pts_valid_flag][:,:3]       # 筛选出符合条件的点 0号相机rect坐标系
    pts_intensity = pts_indensity[pts_valid_flag]
    if cfg.GT_AUG_ENABLED and self.mode == 'TRAIN':
        # 获取当前样本 label 中的有效 Object3d（其中存储了物体的 xyz，h,w,l,r,ry,alpha 等信息)
        all_gt_obj_list = self.filtrate_dc_objects(self.get_label(sample_id))
        # 从 Object3d 中提取 x,y,z,h,w,l,ry
        all_gt_boxes3d = kitti_utils.objs_to_boxes3d(all_gt_obj_list)   # (n, 7)

        # 随机从训练集中选择某个 sample 的某个 gt box3d，如果判断其可以插入到当前 sample，那么执行插入，进行数据扩增
        # gt_aug_flag: True 扩增成功；False 扩增失败
        # pts_rect: 扩增之后的点云坐标
        # pts_intensity: 扩增之后的点云反射强度
        # extra_gt_boxes3d: 增加的 gt box3d (M, 7)，M 为增加的数量，7 表示 xyzhwl,ry
        # extra_gt_obj_list: 增加的 Object3d 对象集合，List<Object3d> 类型
        # apply_gt_aug_to_one_scene 函数中部分代码逻辑可参考图
        gt_aug_flag, pts_rect, pts_intensity, extra_gt_boxes3d, extra_gt_obj_list = \
            self.apply_gt_aug_to_one_scene(sample_id, pts_rect, pts_intensity, all_gt_boxes3d)
    if self.mode == 'TRAIN' or self.random_select:
        if self.npoints < len(pts_rect):     # npoints: 16384。如果云数量太多，那么随机去掉一部分近的点
            pts_depth = pts_rect[:, 2]      # 点云 z 坐标。
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]   #  远点较少，近点较多，
            near_idxs = np.where(pts_near_flag == 1)[0]     # 所以可以随机去掉一部分而没有太大影响
            near_idxs_choice = np.random.choice(near_idxs, self.npoints - len(far_idxs_choice))
            choice = np.concatenate(near_idxs_choice, far_idxs_choice, axis=0) \
                if len(far_idxs_choice) > 0 else near_idxs_choice
            np.random.shuffle(choice)
        else:
            # 点云数量不足，那么有放回抽样所缺数量的点，凑齐 npoints 个点
            ...
        ret_pts_rect = pts_rect[choice,:]
        ret_pts_intensity = pts_intensity[choice] - 0.5     # translate intensity to [-0.5, 0.5]
    else:
        ret_pts_rect = pts_rect
        ret_pts_intensity = pts_intensity

    ret_pts_features = ret_pts_intensity.reshape(-1, 1)     # (N, 1)
    sample_info = {'sample_id': sample_id, 'random_select': self.random_select}
    
    gt_obj_list = self.filtrate_object(self.get_label(sample_id))   # 过滤出样本中符合要求的 Object3d
    if cfg.GT_AUG_ENABLED and self.mode == 'TRAIN' and gt_aug_flag:
        gt_obj_list.extend(extra_gt_obj_list)
    gt_boxes3d = kitti_utils.objs_to_box3d(gt_obj_list)       # extract xyzhwl,ry from Object3d
    gt_alpha = np.zeros(len(gt_obj_list), dtype=np.float32)
    for k, obj in enumerate(gt_obj_list):
        gt_alpha[k] = obj.alpha
    
    aug_pts_rect = ret_pts_rect.copy()
    aug_gt_boxes3d = gt_boxes3d.copy()
    if cfg.AUG_DATA and self.mode == 'TRAIN':
        # 数据增强：rotation，scaling，flip
        aug_pts_rect, aug_gt_boxes3d, aug_method = self.data_augmentation(aug_pts_rect, 
            aug_gt_boxes3d, gt_alpha, sample_id)
        sample_info['aug_method'] = aug_method      # ['ratation', 'scaling', 'flip']
    if cfg.RPN.USE_INTENSITY:   # False
        pts_input = np.concatenate((aug_pts_rect, ret_pts_features), axis=1) # (N, 4)
    else:
        pts_input = aug_pts_rect    # (N, 3)

    sample_info['pts_input'] = pts_inputs   # 点云输入，(npoints, 3)
    sample_info['pts_rect'] = aug_pts_rect  # 样本中的点云，(npoints, 3)
    sample_info['pts_feature'] = ret_pts_feature # 点云特征/反射强度，(npoint, 1)
    sample_info['gt_boxes3d'] = aug_gt_boxes3d  # 样本中物体数量为 M，(M, 7)，xyzhwl,ry
    if not cfg.RPN.FIXED:   # True    
        # 计算点云的分类标签，1 表示位于 gt box3d 内，0 表示位于 gt box3d 外。-1 表示靠近 gt box3d 边缘面。
        # rpn_cls_label: (N,) N 是点云数量。1 表示点是正例，0 表示点是负例。-1 的点忽略
        # rpn_reg_label: (N, 7)，点相对于某个 gt box3d 是正例，那么记录这个点的 dx,dy,dz（gt box3d 到这个点的坐标差）
        #               以及这个 gt box3d 的 h,w,l,ry
        rpn_cls_label, rpn_reg_label = self.generate_rpn_training_label(aug_pts_rect, aug_gt_boxes3d)
        sample_info['rpn_cls_label'] = rpn_cls_label
        sample_info['rpn_reg_label'] = rpn_reg_label
    return sample_info
```

![](/images/obj_det/3d/point_rcnn_6.png)
<center>图 6.</center>

接着我们看 DataLoader 中的 `collate_fn=train_set.collate_batch` 是如何组装批数据的，

```python
# class KittiRCNNDataset
def collate_fn(self, batch):
    '''
    batch: [sample_info1, sample_info2, ...]
    '''
    batch_size = len(batch)
    ans_dict = {}
    for key in batch[0].keys():
        if cfg.RPN.ENABLED and key == 'gt_boxes3d' or \
            (cfg.RCNN.ENABLED and cfg.RCNN.ROI_SAMPLE_JIT and key in ['gt_boxes3d', 'roi_boxes3d']):
            max_gt = 0  # 这批样本中，具有最大 gt boxes3d 的数量
            for k in range(batch_size):
                max_gt = max(max_gt, len(batch[k][key]))
            batch_gt_boxes3d = np.zeros((batch_size, max_gt, 7), dtype=np.float32)
            for i in range(batch_size):
                batch_gt_boxes3d[i, :len(batch[i][key]),:] = batch[i][key]
            ans_dict[key] = batch_gt_boxes3d
            continue
        if isinstance(batch[0][key], np.ndarray):
            if batch_size == 1:
                ans_dict[key] = batch[0][key][np.newaxis,...]   # 增加一个维度
            else:   # 例如点云，(batch_size, npoints, 3)
                ans_dict[key] = np.concatenate([batch[k][key] for k in range(batch_size)], axis=0)
        else:
            ans_dict[key] = [batch[k][key] for k in range(batch_size)]
            if isinstance(batch[0][key], int):
                ans_dict[key] = np.array(ans_dict[key], dtype=np.int32)
            elif isinstance(batch[0][key], float):
                ans_dict[key] = np.array(ans_dict[key], dtype=np.float32)
    return ans_dict
```

#### 3.1.2.2 模型

如图 1，模型分 RPN 和 RCNN 两部分。本节仅讨论 RPN 部分，即图 1 (a) 部分。

```python
if args.train_mode = 'rpn':
    cfg.RPN.ENABLED = True
    cfg.RCNN.ENABLED = False
```

**RPN**

```python
class RPN(nn.Module):
    def __init__(self, use_xyz=True, mode='TRAIN'):
        super().__init__()
        self.training_mode = (mode == 'TRAIN')

        MODEL = importlib.import_module(cfg.RPN.BACKBONE)   # pointnet2_msg
        # 创建 带 MSG 的 pointnet2 模型作为 backbone
        self.backbone_net = MODEL.get_model(input_channel=int(cfg.RPN.USE_INTENSITY), use_xyz=use_xyz)

        cls_layers = [] # backbone 获取每个 点的特征。backbone 后接两个分支：classification 和 regression
```

**pointnet2**

我们这里先暂停，回顾一下 [pointnet2 模型](/2022/10/20/obj_det/3d/pointnet++)，pointnet 先通过连续的几个 SA 模块进行下采样，然后使用 skip link concatenation 上采样，整个过程可以看作是一个 Encoder+Decoder 结构，最终得到每个点的特征。

上采样过程中，点特征传播（FP）过程：上一层的点，上一次点特征，下一层点（上一层点的 grouping 之后各 group 的中心点），下一层点特征。将上一层每个点在下一层点中取 k 个最近邻，然后按距离的倒数为权重计算 k 最近邻特征的加权和，这就是下一层点特征传播到上一层点的过程。

点特征传播之后再与上一层点的原来的特征进行 concatenate，然后经过几个 MLP（一维 conv），输出本模块的点特征，即本次融合后的点特征。

依次进行如上所述的 FP+concatenation+MLP 过程，直到输出所有点融合后的点特征。

再来看下 pointnet2 的模型配置，

```python
# tools/cfgs/default.yaml 中配置
NPOINTS = [4096, 1024, 256, 64]
RADIUS = [[0.1, 0.5], [0.5, 1.0], [1.0, 2.0], [2.0, 4.0]]
NSAMPLE = [[16, 32], [16, 32], [16, 32], [16, 32]]
MLPS = [[[16, 16, 32], [32, 32, 64]], [[64, 64, 128], [64, 96, 128]],
        [[128, 196, 256], [128, 196, 256]], [[256, 256, 512], [256, 384, 512]]]
FP_MLPS = [[128, 128], [256, 256], [512, 512], [512, 512]]
CLS_FC = [128]
DP_RATIO = 0.5
```

NPOINTS: 每个 SA 的输出点数（上一层采样的 group 中心点数量）。根据配置可知，采用了 4 个 SA

RADIUS: 每个 SA 使用 MSG，即多 scale 分组。本参数指定每个 SA 中的 scale。例如第一个 SA 使用 `[0.1, 0.5]` 两个半径值的圆。

NSAMPLE: 每个 group 中采样点数量。例如第一个 SA，`NPOINTS[0]=4096, NSAMPLE=[16, 32]`，根据第一个 scale， grouping 后数据 shape 为 `(B, 4096, 16, 3+D)`，review shape 为 `(B, 3+D, 16, 4096)`，然后经过 pointnet（第一个版本，若干个 conv2d 得到特征 `(B, D1', 16, 4096)`，然后是 maxpool 得到特征 `(B D1', 4096)`），然后根据第二个 scale， grouping 后数据 shape 为 `(B, 4096, 32, 3+D)`，经过同样的变换（若干个 conv2d+maxpool）得到特征 `(B, D2', 4096)`，最后根据 MSG 进行 concatenate，即得到 `(B, D1'+D2', 4096)` 的特征。

MLPS: 上述 conv2d 中输出 channel。例如第一个 SA 对应 `[[16, 16, 32], [32, 32, 64]]`，第一个 scale 对应 `[16, 16, 32]`。四个 SA 的输出 concatenated 的特征 channel 分别是 `(32+64, 128+128, 256+256, 512+512)` 。

FP_MLPS: 上述 FP+concat 之后的 MLP （即 conv1d） 的输出 channel。参考如下代码片段，对于第一个 skip link connection 模块而言，其输入是第三个 SA 的输出点和点特征（上一层），以及第四个 SA 的输出点和点特征（下一层），根据下一层输出点特征通过 FP 得到上一层点特征，特征维度为 `channel_out`，并于上一层原来的点特征 concat，上一层原点特征维度为 `skip_channel_list[k=3]`（因为 `k=4` 时 `k+1 >= len(FP_MLPS)`），所以 concat 之后点特征为 `channel_out+skip_channel_list[k=3]`，然后通过 FP 模块的 MLP，得到 FP 模块最终的输出特征，其维度为 `FP_MLPS[k=3][-1]`。同理，对于第二个 skip link connection （`k=2`），输入是第二个 SA 的输出点和点特征，以及上一个 FP 模块的输出点和点特征，那么根据同样的分析，第二个 SA 输出点的 FP 特征维度为 `FP_MLPS[k+1=3][-1]`，第二个 SA 输出点的特征为 `skip_channel_list[2]`，两个特征 concatenate 之后再经过 MLP，输出特征维度为 `FP_MLPS[k=2][-1]`。依次进行 FP，直到原始点的两路特征 concatenate 之后，再经 MLP，得到最终所有点的特征维度为 `FP_MLPS[0][-1]=128`，即特征 size 为 `(B, npoints, 128)`。

```python
# file: lib/net/pointnet2_msg.py
# skip_channel_list: 四个 SA 输出 concat 之后的 channel。
# channal_out: 512+512，最后一个 SA 的输出 concat 之后的 channel
self.FP_modules = nn.ModuleList()

for k in range(len(cfg.RPN.FP_MLPS)):
    pre_channel = FP_MLPS[k+1][-1] if k+1 < len(FP_MLPS) else channel_out
    self.FP_modules.append(
        # mlp 参数：第一个 item 是 FP 模块的输入 channel，后续的 items 分别是 MLP 的输出 channel
        PointnetFPModule(mlp=[pre_channel + skip_channel_list[k]] + cfg.RPN.
        FP_MLPS[k])
    )
```

backbone 的整个过程，如图 5 所示，

![](/images/obj_det/3d/point_rcnn_5.png)
<center>图 5 </center>

backbone 输出为输入点的坐标 shape 为 `(B, npoints, 3)`，以及输入点的特征 `(B, 128, npoints)`（卷积输出，第二个维度表示 out channel）。

**RPN 分类**

我们接着看 RPN 类的定义部分：1. 分类分支（将 backbone 的特征经 MLP ，最后是一个 channel=1 的 Conv1d，每个点处进行二分类。）

```python
# 分类分支
cls_layers = []
pre_channel = cfg.RPN.FP_MLPS[0][-1]
# backbone 得到各点的特征 (B, 128, N)，然后再经 MLP 
for k in range(0, cfg.RPN.CLS_FC.__len__()):
    cls_layers.append(pt_utils.Conv1d(pre_channel, cfg.RPN.CLS_FC[k], bn=cfg.RPN.USE_BN))
    pre_channel = cfg.RPN.CLS_FC[k]
cls_layers.append(pt_utils.Conv1d(pre_channel, 1, activation=None))     # 最后得到二分类，(B, 1, N)
if cfg.RPN.DP_RATIO >= 0:
    cls_layers.insert(1, nn.Dropout(cfg.RPN.DP_RATIO))
self.rpn_cls_layer = nn.Sequential(*cls_layers)
```

**RPN 回归**

然后是 2. 回归分支。先看相关的配置，

```python
# file: default.yaml
LOC_SCOPE: 3.0
LOC_BIN_SIZE: 0.5
NUM_HEAD_BIN: 12
REG_FC: [128]
```

根据图 2，在 X-Z 平面（俯视图）上，

LOC_SCOPE: 每个前景点的 local region 的搜索范围（半径），3.0 米，$\mathcal S=3$，搜索范围为 $[-\mathcal S, \mathcal S]$ 。

LOC_BIN_SIZE: 将前景点的 local region 的搜索范围划分为若干个 bin，每个 bin 的大小为 0.5 米，即 $\delta = 0.5$ 故沿着 X 和 Z 轴分别切割为 $2 \mathcal S /\delta = 12$

NUM_HEAD_BIN: 朝向角 bin 的数量， 12 个 bin。

REG_FC: 回归分支的 MLP 的输出 channel。

继续看 RPN 类的定义（回归分支部分），回归分支是将 backbone 的特征经 MLP，最后是一个无 activation 的 Conv1d 调整维度。

```python
# 每个前景点有一个 local，其中 X，Z 轴方向各 12 个 bin
per_loc_bin_num = int(cfg.RPN.LOC_SCOPE / cfg.RPN.LOC_BIN_SIZE) * 2
if cfg.RPN.LOC_XZ_FINE: # 精细位置，除了 bin index，还需要 residual
    reg_channel = per_loc_bin_num * 4 + cfg.RPN.NUM_HEAD_BIN * 2 + 3
else: # 非精细位置，只需要 bin index
    reg_channel = per_loc_bin_num * 2 + cfg.RPN.NUM_HEAD_BIN * 2 + 3
reg_channel += 1    # 参考上面 (a) 式

reg_layers = []
pre_channel = cfg.RPN.FP_MLPS[0][-1]    # backbone 的输出特征维度 (B, D, N)
for k in range(len(cfg.RPN.REG_FC)):
    reg_layers.append(pt_utils.Conv1d(pre_channel, cfg.RPN.REG_FC[k], bn=cfg.RPN.USE_BN))
    pre_channel = cfg.RPN.REG_FC[k]

# MLP 后再跟一个 conv1d，调整 channel 数
reg_layers.append(pt_utils.Conv1d(pre_channel, reg_channel, activation=None))
if cfg.RPN.DP_RATIO >= 0:
    reg_layers.insert(1, nn.Dropout(cfg.RPN.DP_RATIO))
self.rpn_reg_layer = nn.Sequential(*reg_layers)
```

**RPN 分类损失**

RPN 的二分类损失使用 `SigmoidFocalLoss`，参见上面公式 (1) 。

RPN 的前向传播过程，代码如下，

```python
def forward(self, input_data):
    '''
    input_data: dict(point_cloud)
    '''
    pts_input = input_data['pts_input'] # (B, N, 3)
    # 原始输入点坐标，原始输入点特征
    # (B, N, 3), (B, C, N)
    backbone_xyz, backbone_features = self.backbone_net(pts_input)

    rpn_cls = self.rpn_cls_layer(backbone_features).transpose(1, 2).contiguous()    # (B, N, 1)
    rpn_reg = self.rpn_reg_layer(backbone_features).transpose(1, 2).contiguous()    # (B, N, D=76)

    ret_dict = {'rpn_cls': rpn_cls, 'rpn_reg': rpn_reg,
                'backbone_xyz': backbone_xyz, 'backbone_features': backbone_features}
    return ret_dict
```

调用模型前向传播并计算损失的过程代码如下，

```python
# lib/net/train_functions.py
# def model_joint_fn_decorator()
def model_fn(model, data):
    '''
    model: PointRCNN
    data: DataLoader 的一批数据（字典形式）
    '''
    # inputs: (B, npoints, 3) 点云数据坐标
    # gt_boxes3d: (B, max_gt, 7) 批样本中的 gt box3d，包含 x,y,z,h,w,l.ry。
    #             max_gt 是这批中单个样本中最大gt 数量
    # rpn_cls_label: (B, npoints) 点云的二分类。1: pos；0: neg；-1: 忽略
    # rpn_reg_label: (B, npoints, 7) 相关gt box3d 中心点与点云的坐标差，dx,dy,dz，以及 hwl,ry
    input_data = {'pts_input': inputs, 'gt_boxes3d': gt_boxes3d}

    ret_dict = model(input_data)    # 返回的就是上一段代码中的 ret_dict

    tb_dict = {}
    disp_dict = {}
    loss = 0
    if cfg.RPN.ENABLED and not cfg.RPN.FIXED:   # 使用 RPN 网络，且不固定 RPN 网络参数
        rpn_cls, rpn_reg = ret_dict['rpn_cls'], ret_dict['rpn_reg']
        rpn_loss = get_rpn_loss(model, rpn_cls, rpn_reg, rpn_cls_label, rpn_reg_label, tb_dict)
        loss += rpn_loss
        disp_dict['rpn_loss'] = rpn_loss.item()
    if cfg.RCNN.ENABLED:    # 使用 RCNN 网络
        rcnn_loss = get_rcnn_loss(model, ret_dict, tb_dict)
        disp_loss['reg_fg_sum'] = tb_dict['rcnn_reg_fg']
        loss += rcnn_loss

    disp_dict['loss'] = loss.item()
    return ModelReturn(loss, tb_dict, disp_dict)
```

现在我们可以不考虑 RCNN 分支，只看 RPN 的损失计算，相关函数为，

```python
def get_rpn_loss(model, rpn_cls, rpn_reg, rpn_cls_label, rpn_reg_label, tb_dict):
    rpn_cls_loss_func = model.rpn.rpn_cls_loss_func     # 不考虑使用 DataParallel 模型
    rpn_cls_label_flat = rpn_cls_label.view(-1)         # (B*npoints,)
    rpn_cls_flat = rpn_cls.view(-1)                     # (B*npoints,)
    fg_mask = (rpn_cls_label_flat > 0)                  # foreground mask

    if cfg.RPN.LOSS_CLS == 'SigmoidFocalLoss':
        rpn_cls_target = fg_mask.float()
        pos = fg_mask.float()
        neg = (rpn_cls_label_flat == 0).float()

        cls_weights = pos + neg     # -1 的点，其分类损失权重为 0，等于忽略，其他点的权重为 1
        pos_normalizer = pos.sum()  # pos 点数
        cls_weights = cls_weights / torch.clamp(pos_normalizer, min=1.0)
        # 计算 focal loss，参考 (1) 式。另外给每个点的损失增加了权重
        rpn_loss_cls = rpn_cls_loss_func(rpn_cls_flat, rpn_cls_target, cls_weights)
        rpn_loss_cls_pos = (rpn_loss_cls * pos).sum()   # 正例损失均值
        rpn_loss_cls_neg = (rpn_loss_cls * neg).sum()   # 负例损失和除以正例数量
        rpn_loss_cls = rpn_loss_cls.sum()               # 所有损失和处理正例数量
        ...
    point_num = rpn_reg.size(0) * rpn_reg.size(0)       # rpn_reg 是模型回归分支输出，(B,npoint,76)
    fg_sum = fg_mask.long().sum().item()    # 正例的点数
    if fg_sum != 0:
        loss_loc, loss_angle, loss_size, reg_loss_dict = \
            loss_utils.get_reg_loss(rpn_reg.view(point_num, -1)[fg_mask],
                                    rpn_reg_label.view(point_num, 7)[fg_mask],
                                    loc_scope=cfg.RPN.LOC_SCOPE,                # 3.0 半径
                                    loc_bin_size=cfg.RPN.LOC_BIN_SIZE,          # 0.5
                                    num_head_bin=cfg.RPN.NUM_HEAD_BIN,          # 12
                                    anchor_size=MEAN_SIZE,  # 训练集中同类 box3d 的均值 h,w,l
                                    get_xz_fine=cfg.RPN.LOC_XZ_FINE,            # True
                                    get_y_by_bin=False,
                                    get_ry_fine=False)
        loss_size = 3 * loss_size
        rpn_loss_reg = loss_loc + loss_angle + loss_size
        ...

    rpn_loss = rpn_loss_cls * cfg.RPN.LOSS_WEIGHT[0] + rpn_loss_reg * cfg.RPN.LOSS_WEIGHT[1]
    ...
    return rpn_loss
```

上述代码中 `rpn_cls_loss_func` 函数计算 focal loss，参考 (1) 式，但是经过我阅读此函数代码，发现实际上是将 (1) 式中的 $-\log p_t$ 改成了

$$-\log p_t + \max(\frac p {1-p}, 0) - y \cdot \frac p {1-p} \tag{10}$$

也就是说，在 $-\log p_t$ 之外增加了一个损失，$x=\frac p {1-p}$  为 RPN 分类分支输出的 logit，范围为 $\mathbb R$，也就是 sigmoid 函数的自变量记为 x（p相当于 sigmoid 函数的因变量）。根据一下分析可知这个新增损失项是合理的：
1. $x > 0, y=1$，新增损失项变为 $0$，即预测正确，损失为 0
2. $x < 0, y=1$，预测错误，损失为 $-x$
3. $x > 0, y=0$，预测错误，损失为 $x$
4. $x < 0, y=0$，预测正确，损失为 0


**RPN 回归损失**

```python
# lib/utils/loss_utils.py
def get_reg_loss(pred_reg, reg_label, loc_scope, loc_bin_size, num_head_bin, anchor_size,
    get_xz_fine=True, get_y_by_bin=False, loc_y_scope=0.5, loc_y_bin_size=0.25, get_ry_fine=False):
    '''
    pred_reg: (B*npoints, 76)
    reg_label: (B*npoints, 7)
    '''
    per_loc_bin_num = int(loc_scope / loc_bin_size) * 2     # 每个 local，有 3.0/0.5 * 2 = 12 个 bin
    loc_y_bin_num = int(loc_y_scope / loc_y_bin_size) * 2   # Y 轴的 bin 数量。实际上 Y 轴不使用 bin-based

    reg_loss_dict = {}
    loc_loss = 0

    # dx, dy, dz  (gt)，shape 均为 (B*npoints,)
    # 体中心点与点云的坐标差
    x_offset_label, y_offset_label, z_offset_label = reg_label[:,0], reg_label[:,1], reg_label[:,2]

    # 我们仅考虑 loc_scope 半径范围内的点，即 dx \in (-loc_scope, loc_scope) 
    # 超出范围的，将 dx 取范围边界；然后将坐标差范围移动到 (0, 2*loc_scope)，参考 (2) 式
    x_shift = torch.clamp(x_offset_label + loc_scope, 0, loc_scope * 2 - 1e-3)
    z_shift = torch.clamp(z_offset_label + loc_scope, 0, loc_scope * 2 - 1e-3)
    # 根据坐标差确定点云中各点所属的 bin index。例如 X 轴，从正方向到负方向，index 为 0,1,...,11。共12个 bin
    # 同时理解上面的 clamp 操作：dx 超出 -loc_scope 的点归到 index 0，超出 +loc_scope 的点归到 index 11
    x_bin_label = (x_shift / loc_bin_size).floor().long()   # (B*npoints,) 中心点X轴在local中的bin index
    z_bin_label = (z_shift / loc_bin_size).floor().long()   # (B*npoints,) 中心点Z轴在local中的bin index

    # 预测是一个长度为76的向量，参见式 (a) 。
    x_bin_l, x_bin_r = 0, per_loc_num         # 预测长向量中，预测 x bin index 的子向量的起始截止下标
    z_bin_l, z_bin_r = per_loc_num, per_loc_num * 2
    start_offset = z_bin_r

    # bin index 的预测是分类预测，例如某个 local 的 X轴 bin index 预测，
    # 预测是长度为12的向量 p[x_bin_l:x_bin_r]，gt label 为一个标量
    loss_x_bin = F.cross_entropy(pred_reg[:,x_bin_l:x_bin_r], x_bin_label)  # 损失均值
    loss_z_bin = F.cross_entropy(pred_reg[:,z_bin_l:z_bin_r], z_bin_label)  # 损失均值
    reg_loss_dict['loss_x_bin'] = loss_x_bin.item()
    reg_loss_dict['loss_z_bin'] = loss_z_bin.item()
    loc_loss += loss_x_bin + loss_z_bin

    if get_xz_fine:     # 使用 xz 精细预测（即，除了bin index，还预测 bin 内 residual）
        x_res_l, x_res_r = per_loc_bin_num * 2, per_loc_bin_num * 3
        z_res_l, z_res_r = per_loc_bin_num * 3, per_loc_bin_num * 4
        start_offset = z_res_r

        # 计算 gt residual，参考 (3) 式
        x_res_label = x_shift - (x_bin_label.float() * loc_bin_size + loc_bin_size / 2)
        z_res_label = z_shift - (z_bin_label.float() * loc_bin_size + loc_bin_size / 2)
        x_res_norm_label = x_res_label / loc_bin_size   # 归一化 residual 到 [-0.5, 0.5)
        z_res_norm_label = z_res_label / loc_bin_size   # 归一化 residual 到 [-0.5, 0.5)

        # (B*npoints, 12). 原本每个 local 的 gt bin index 是一个标量，现在将其改为 one-hot vector
        x_bin_onehot = torch.cuda.FloatTensor(x_bin_label.size(0), per_loc_bin_num).zero_()
        x_bin_onehot.scatter_(1, x_bin_label.view(-1, 1).long(), 1) # bin index 下标的元素为1
        z_bin_onehot = torch.cuda.FloatTensor(z_bin_label.size(0), per_loc_bin_num).zero_()
        z_bin_onehot.scatter_(1, z_bin_label.view(-1, 1).long(), 1)

        # 计算 residual 的 smooth L1 损失，显然只用 gt bin index 的 residual 预测与gt residual
        loss_x_res = F.smooth_l1_loss((pred_reg[:,x_res_l:x_res_r] * x_bin_onehot).sum(dim=1), x_res_norm_label)
        loss_z_res = F.smooth_l1_loss((pred_reg[:,z_res_l:z_res_r] * z_bin_onehot).sum(dim=1), z_res_norm_label)
        reg_loss_dict['loss_x_res'] = loss_x_res.item()
        reg_loss_dict['loss_z_res'] = loss_z_res.item()
        loc_loss += loss_x_res + loss_z_res

    if get_y_by_bin:    # Y 轴坐标不使用 bin-based
        ...
    else:               # Y 轴坐标直接使用回归预测，smooth L1 损失
        y_offset_l, y_offset_r = start_offset, start_offset+1
        start_offset = y_offset_r

        loss_y_offset = F.smooth_l1_loss(pred_reg[:, y_offset_l:y_offset_r].sum(dim=1), y_offset_label)
        reg_loss_dict['loss_y_offset'] = loss_y_offset.item()
        loc_loss += loss_y_offset

    # rotation_y 损失计算。rotation_y 预测也是 bin-based
    ry_bin_l, ry_bin_r = start_offset, start_offset + num_head_bin
    ry_res_l, ry_res_r = ry_bin_r, ry_bin_r + num_head_bin
    ry_label = reg_label[:, 6]  # (B*npoints,) 来自 label 文件的每行第15个数据，rotation_y，范围为 [-pi,pi]

    if get_ry_fine:     # rotation_y 是否精调？RCNN 中精调，RPN 中不精调
        ...             # RCNN 中精调 ry，将 ry 范围 [-pi/4,pi/4] 划分为 num_head_bin 个 bin
    else:
        angle_per_class = (2 * np.pi) / num_head_bin    # 360°，等分成 12 个bin，每个 bin 跨越角度 30°
        heading_angle = ry_label % (2 * np.pi)  # 将原来的 -pi~pi 转换到 0~2pi 范围
        # bin index 和 residual 的计算参考下方的 (11) 和 (12) 式
        shift_angle = (heading_angle + angle_per_class / 2) % (2 * np.pi)
        ry_bin_label = (shift_angle / angle_per_class).floor().long()
        ry_res_label = shift_angle - (ry_bin_label.float() * angle_per_class + angle_per_class / 2)
        ry_res_norm_label = ry_res_label / (angle_per_class / 2)    # (B*npoints, ) 归一化 residual 到 [-1, 1)
    
    ry_bin_onehot = torch.cuda.FloatTensor(ry_bin_label.size(0), num_head_bin).zeros_() # (B*npoints, 12)
    ry_bin_onehot.scatter_(1, ry_bin_label.view(-1, 1).long(), 1)
    loss_ry_bin = F.cross_entropy(pred_reg[:, ry_bin_l:ry_bin_r], ry_bin_label) # 分类损失
    loss_ry_res = F.smooth_l1_loss((pred_reg[:, ry_res_l:ry_res_r] * ry_bin_onehot).sum(dim=1), ry_res_norm_label)

    reg_loss_dict['loss_ry_bin'] = loss_ry_bin.item()
    reg_loss_dict['loss_ry_res'] = loss_ry_res.item()
    angle_loss = loss_ry_bin + loss_ry_res

    size_res_l, size_res_r = ry_res_r, ry_res_r + 3     # 预测 h,w,l 的损失

    # t_h = (h_gt - h_m)/h_m
    size_res_norm_label = (reg_label[:,3:6] - anchor_size) / anchor_size
    size_res_norm = pred_reg[:, size_res_l:size_res_r]      # (B*npoints, 3)
    size_loss = F.smooth_l1_loss(size_res_norm, size_res_norm_label)
    reg_loss_dict['loss_loc'] = loc_loss
    reg_loss_dict['loss_angle'] = angle_loss
    reg_loss_dict['loss_size'] = size_loss
    return loc_loss_, angle_loss, size_loss, reg_loss_dict
```

上述代码中，RPN 模型对 rotation_y 的预测也是 bin-based，共 12 个 bin，首先将 rotation_y 的范围从 $[\pi, \pi)$ 调整到 $[0, 2\pi)$。记某个点云中点所关联的 gt box （调整后）朝向角为 $\theta^{gt} \in [0, 2\pi)$，每个 bin 的角度范围为 $w$， 其所属 bin index 为

$$\text{bin}_{\theta}^{gt} = \lfloor \frac {\theta^{gt} + w/2} w \rfloor \tag{11}$$

注意 index 为 0 的 bin 的 $\theta^{gt}$ 角度范围不是 $[0, w)$，而是顺时针旋转了 $w/2$ 角度，即第一个 bin 的 $\theta^{gt}$ 的范围是 $[2\pi - w/2, 2\pi) \cup [0, w/2)$，注意 $0^{\circ}$ 就是这个 bin 角度范围的中间点，然后逆时针每旋转 $w$ 范围， bin index 增加 1  。

记 $\theta^{gt}$ 逆时针旋转了 $w/2$ 角度后为 $\theta^{gt'}=(\theta^{gt} + w/2) \ \text{mod} \ 2\pi$，

bin residual 计算如下，

$$\text{res}_{\theta}^{gt} = \theta^{gt'} - (\text{bin}_{\theta}^{gt} \cdot w + w/2) \in [-w/2, w/2) \tag{12}$$

实际上从二维平面的圆来看，每个bin 对应的 $\theta^{gt}$ 都是一段连续的角度范围，且前一半范围对应 $\text{res}_{\theta}^{gt} \in [-w/2, 0)$，后一半范围对应 $\text{res}_{\theta}^{gt} \in [0, w/2)$ 。(12) 式就是角度与所在 bin 范围中心角度的差。那么我们根据 RPN 的预测值，可以计算出最终的角度预测值为

$$[(\text{bin}_{\theta} \cdot w + w/2) + \text{res}_{\theta} - w/2] \text{mod} \ 2 \pi=(\text{bin}_{\theta} \cdot w + \text{res}_{\theta}) \text{mod} \ 2 \pi \tag{13}$$

(13) 式中，括号中为预测 bin 的中心角，然后加上 res 的预测角，得到的是对  $\theta^{gt'}$  角度的预测，所以还需要顺时针旋转 $w/2$。
