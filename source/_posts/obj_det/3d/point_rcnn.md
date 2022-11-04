---
title: PointRCNN 论文解读
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

    使用 [pointnet++](/2022/10/20/obj_det/3d/pointnet++) 学习点特征，使用 multi-scale grouping (MSG) 的 set abstraction 作为 encoder，上采样和特征传播作为 decoder。

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

$$p_t=\begin{cases} p & y=1 \\ 1- p & y=0 \end{cases}, \quad\quad \alpha_t = \begin{cases} 0.75 &y=1 \\ 0.25 & y=0\end{cases}$$

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


图 2 中最左边的汽车为例，gt bin 的 x 下标为 $\text{bin}_x^{p}=1$ (下标从 0 开始)，而 $ -2 \delta \le x^c - x^{p} < -\delta$，由于设置了搜索范围 $S=3\delta$，那么 $x^c-x^{p}+\mathcal S \in [\delta, 2\delta)$，与 $\text{bin}_x^{p}=1$ 相符。

(3) 式中 $\text{res}_u^{p}$ 的范围是 $[-\frac {\delta} {2\mathcal C}, \frac {\delta}{2\mathcal C})$ ，取 $\mathcal C=\delta /2$ 可归一化残差值。

俯视图偏向角 $\theta$ 和大小 $(h,w,l)$ 的估计则与 Frustum PointNets 类似。将偏向角范围 $2\pi$ 分成 n 个 bins，然后计算 target bin $\text{bin}_{\theta}^{p}$ 和残差 $\text{res}_{\theta}^{p}$（与 x, z 坐标预测相似）。目标 size $(h,w,l)$ 则直接计算出与训练集中相同类的平均目标 size 之间的残差值 $(\text{res}_h^{p},\text{res}_w^{p},\text{res}_l^{p})$，然后对这些残差值回归。

作者实验中使用了 bin size $\delta=0.5m$ 以及搜索范围 $S=3m$（m 指距离单位米），于是 X,Y 轴的 的 bin 均为 12 个。偏向角的 bin 数量配置为 $n=12$，那么对每个前景点，回归 head 一共需要预测的数据量为

$$4\times 12 + 1 + 2 \times 12 + 3=76$$

表示 12 个 bin 的 X,Z 的 bin 分类预测得分和 bin 内残差值回归，然后是 1 个 y 坐标的残差值，12 个偏向角 bin 的分类预测和 bin 内残差值回归，最后是 3 个 h,w,l 的残差值。

**inference** 阶段，对那些 bin-based 预测参数即，$x,z,\theta$，首先选择最高 confidence 的 bin，然后在加上预测的残差值，以便得到更精确的值。对其他直接回归的参数即 $y,h,w,l$，将预测的残差值加上初始参照值即可。

整个 3D box 回归 head 的损失计算如下：

$$\begin{aligned}\mathcal L_{bin}^{p} &= \sum_{u \in \{x,z,\theta\}} [\mathcal F_{cls}(\hat {bin}_u^{p}, bin_u^{p})+\mathcal F_{reg}(\hat {res}_u^{p}, res_u^{p})]
\\ \mathcal L_{reg}^{p} &=\sum_{v \in \{y,h,w,l\}} \mathcal F_{reg}(\hat {reg}_v^{p}, res_v^{p})
\\ \mathcal L_{reg} &= \frac 1 {N_{pos}} \sum_{p \in pos} (\mathcal L_{bin}^{p}+\mathcal L_{reg}^{p})
\end{aligned} \tag{5}$$

其中 $N_{pos}$ 是前景点数量。带 ^ 的符号的表示相应的预测值。$\mathcal F_{cls}$ 是交叉熵损失，$\mathcal F_{reg}$ 是 smooth L1 损失。$\hat {bin}_u^{p}$ 的在前景点 p 关于维度 u 的预测 bin，可以看作是一个预测得分向量，作者实验中每个维度（$X,Y,\theta$）的 bin 均为 12 个，即 $\hat {bin}_u^{p}$ 是长度为 12 的向量，而 target $bin_u^{p}$ 则是长度为 12 的 one-hot 向量。

假设预测得分向量是归一化的（经过了 softmax），那么交叉熵和残差值回归损失计算过程为

1. 获取 one-hot target 中元素 1 的下标，记为 $i$
2. 取预测得分向量中对应位置的预测概率记为 $P_i$，然后求 $-\log P_i$ 就是交叉熵损失
3. 取 $i$ 位置的预测残差值 $\hat {reg}_u^{p}$，以及实际的残差值 target $reg_u^{p}$，计算 smooth L1 就是残差值回归损失

每个前景点均进行预测，显然会产生很多 bbox，所以还需要使用基于鸟瞰图的 oriented IoU 的 NMS 去除冗余 bbox。作者实验中，IoU 阈值取 0.85，NMS 之后仅保留 top 300 的 proposals 用于 stage 2。inference 阶段，oriented NMS 的 IoU 阈值取 0.8，然后取 top 100 proposal 用于 stage 2。

## 2.2 点云区域池化

得到 3D proposal 之后，进一步对这些 3D proposal 的 location 和偏向精细调整（refine）。为了学习到每个 proposal 更具体的 local 特征，作者提出，根据每个 3D proposal 池化其内部的 3D 点数据以及相应的来自 stage 1 的点特征。

使用 $\mathbf b_i = (x_i,y_i,z_i,h_i,w_i,l_i,\theta_i)$ 表示一个 3D proposal，稍微放大它得到一个新的 3D box $\mathbf b_i^e=(x_i,y_i,z_i, h_i+\eta, w_i+\eta, l_i+\eta, \theta_i)$，放大 3D box 是为了更好地将上下文也编码进去。

对于每个点 $p=(x^{p},y^{p},z^{p})$，使用 inside/outside 测试其是否在 $\mathbf b_i^e$ 内部。如在其内部，那么这个点以及其特征用于 refine $\mathbf b_i$。对于一个内部点 p，其特征包含 3D 点坐标 $(x^{p}, y^{p}, z^{p}) \in \mathbb R^3$，激光反射强度 $r^{p} \in \mathbb R$，预测的分割掩码 $m{p} \in \{0,1\}$（来自 stage 1 的 segmentation head 输出），以及 C-dim 的点特征 $\mathbf f^{p} \in \mathbb R^C$ （图 1 (a)， stage 1 的 Point-wise feature vector）。

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

$$d^{p}=\sqrt {(x^{p})^2+(y^{p})^2+(z^{p})^2} \tag{6}$$


每个 proposal 的关联点的 local 空间特征 $\tilde p$ 以及额外特征 $[r^{p}, m^{p}, d^{p}]$ （反射强度，点分割掩码，距离）concatenate 然后喂给 MLP（如图 1 (b) 中的红色方框），encode 得到与 global 特征 $\mathbf f^{p}$ 相同维度的特征，然后 local 特征和 global 特征继续 concatenate （如图 1 (b) 中的黑色方框）并喂给一个 Encoder 网络（图 1 (b) 中的梯形框），这个 Encoder 网络与 PointNet++ 中的相同（多个 set abstraction 组成的网络），最后得到具有判别能力的特征向量，用于 confidence 分类和 box 精调。

### 2.3.3 proposal 精调的损失

对 proposal 精调采用类似的 bin-based 回归损失。如果 3D proposal 与某个 gt box 的 3D IoU 大于 0.55，那么这一对 box 可用于训练。3D proposal 与其对应的 3D gt box 均被转换到 CCS，这说明 3D proposal $\mathbf b_i=(x_i, y_i, z_i, h_i, w_i, l_i, \theta_i)$ 和 3D gt box $\mathbf b_i^{gt}=(x_i^{gt}, y_i^{gt}, z_i^{gt}, h_i^{gt}, w_i^{gt}, l_i^{gt}, \theta_i^{gt})$ 均被转换为

$$\begin{aligned}\tilde {\mathbf b}_i&=(0,0,0, h_i, w_i, l_i, 0)
\\ \tilde {\mathbf b}_i^{gt}&=(x_i^{gt} - x_i, y_i^{gt} - y_i, z_i^{gt} - z_i, h_i^{gt}, w_i^{gt}, l_i^{gt}, \theta_i^{gt} - \theta_i)
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

$$\mathcal L_{refine} = \frac 1 {\|\mathcal B\|} \sum_{i \in \mathcal B} \mathcal F_{cls} (prob_i, label_i)+ \frac 1 {\|\mathcal B_{pos}\|} \sum_{i \in \mathcal B_{pos}} (\tilde {\mathcal L}_{bin}^{(i)} + \tilde {\mathcal L}_{res}^{(i)}) \tag{9}$$

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

**数据集**

数据集类为 `KittiRCNNDataset`，load 上一步 dump 的 gt_database 文件，并按难易两种模式，将 sample object 的 gt 分成两个 list。将符合要求的 sample id 存储至 `sample_id_list` 中。

**模型**

