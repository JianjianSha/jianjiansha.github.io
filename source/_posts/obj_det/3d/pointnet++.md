---
title: PointNet++ 论文解读
date: 2022-10-20 09:53:58
tags: 3d object detection
mathjax: true
---

论文：[PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413)

源码：[yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git)

# 1. 简介 

**TL;DR**

PointNet 提出了直接使用点云进行深度学习，而不是对点云预处理为 voxel grids，鸟瞰图或正视图，这避免了预处理的时间消耗，以及预处理过程中点云数据的部分信息丢失。但是 PointNet 使用了 MLP（也就是 kernel_size=1 的 conv1d），导致无法捕获局部结构特征。


CNN 可以捕获特征，且可以在多层 multi 分辨率上逐步扩大捕获的特征尺寸（特征分辨率降低，CNN 捕获的特征 scale 更大），这使得 CNN 可以提取出局部 pattern 以便能泛化到未见过的情况。

本文作者提出了 PointNet++，其采用了 CNN 结构，从邻域捕获局部特征，这个邻域是对输入的点集进行划分（partition）得到一组相互之间有 overlap 的 regions，对输入点集下采样（其实是每个邻域取一个中心点），这样特征分辨率降低，多层 conv layer 可以逐步捕获更高 level 的特征，直到最后捕获全局特征。

PointNet++ 主要解决两个问题：

1. 如何生成点集的分区
2. 如何学习得到局部特征

对于问题 2， 作者沿用了 PointNet，因为 PointNet 很好地解决了输入点集无序性的问题。PointNet 作为 PointNet++ 网络中一个 basic block，用于生成 high level feature，所以 PointNet++ 实际上是递归地调用 PointNet。

对于问题 1，每个分区定义为欧氏空间中的一个邻域球，相关参数为中心位置和半径。为了使分区均匀地覆盖整个点集，在点集上使用 farthest point sampling （FPS）算法选取中心点，采用这个算法确定邻域，从而 CNN 在这个邻域上进行扫描并提取特征，而非固定 stride 扫描整个点集空间。一个邻域空间依赖于输入数据以及空间距离的测度方法。

如何选择邻域球的半径？由于输入点集额不均匀性，以及特征之间的纠缠，使得较难确定半径。我们假定输入点集在不同的区域具有不同的密度，小的邻域包含太少的点，不足以让 PointNet 鲁棒地捕获局部 pattern，但是大的邻域会导致计算量太大，这其中包括邻划分邻域的计算量，以及在邻域内进行 CNN 操作的计算量等等。 PointNet++ 利用了多个不同尺寸的邻域，以便兼具鲁棒性和捕获细节的能力。

# 2. 方法

## 2.1 PointNet 回顾

一个无序点集 $\{x_1,\ldots, x_n\}$ 其中 $x_i \in \mathbb R^d$，定义一个函数 $f:\mathcal X \rightarrow \mathbb R$，函数参数为一个集合，$f$ 分解为

$$f(x_1,\ldots, x_n) = \gamma \left(\max_{i=1,\ldots,n} \{h(x_i)\} \right) \tag{1}$$

其中 $\gamma$ 是一个标量，$h$ 是一个 MLP。

## 2.2 层级 Point Set 特征学习

PointNet 使用一个 max pooling 聚合全部的点集，本文作者则构造一个层级结构，沿着层级方向逐步对越来越大的局部区域进行提取。

层级结构如图 1 所示，由多个 set abstraction（集合提取）组成，

![](/images/obj_det/3d/pointnet++_1.png)

<center>图 1</center>

每个 set abstraction 均有三个关键 layer：

1. Sampling layer

    从点集中选择一系列的点，这些点都是 local region 的中心点。

2. Grouping layer

    通过寻找邻域点，为每个中心点构造 local region

3. PointNet layer

    使用 mini-PointNet 来 encoder local region 生成一个特征向量

图 1 中展示了两个 set abstraction。

sampling + grouping 就是前面所说的对输入点集的分区（partition）。多个 set abstraction 就对应 hierarchical 。一个 set abstraction 就对应一个 level。


**set abstraction**

set abstraction 的输入是一个 $N \times (d+C)$ 的矩阵，其中 $N$ 是点数，d-dim 的点坐标，以及 C-dim 的点特征；输出是一个 $N' \times (d+C')$ 的矩阵，其中 $N'$ 是下采样的点数，也是 Sampling layer 中寻找的所有 local region 的中心点的数量，$d$ 是点坐标数量，$C'$ 是点特征向量长度，由 PointNet 中的 conv 输出 channel 确定。点特征是 PointNet 根据 local context 进行提取而得。

**Sampling layer**

给定输入点集 $\{x_1,\ldots, x_n\}$，重复使用 FPS 算法选择一个点子集 $\{x_{i_1}, \ldots, x_{i_m}\}$，使得 $x_{i_j}$ 与 $\{x_{i_1}, \ldots, x_{i_{j-1}}\}$ 都距离最远。

**Grouping layer**

此 layer 的输入是一个点集，size 为 $N \times (d+C)$，其中所采样的中心点坐标是一个 $N' \times d$ 的矩阵。此 layer 的输出是点集 groups，其 size 为 $N' \times K \times (d+C)$，每个 group 对应一个 local region，$K$ 是中心点的邻域中点数（包括中心点）。

注意不同的 group 中 $K$ 是不同的，幸运地是 PointNet 可以将不同数量的点集转变为一个固定长度的 local region feature（通过沿着点集数量维度进行 max 操作）。实际操作中，配置一个 $K$ 值，region 邻域半径内点数超过 $K$ 则被截断，不足 $K$ 则重复使用半径内点，使得数量达到 $K$。

CNN 中，一个像素的 local region 是根据 Manhattan 距离（又称城市街道距离，L1 范数）确定的一个邻域内的点。这里我们也类似地根据某个测度距离来确定一个点的邻域。

Ball query 寻找距离 query point 的某个半径以内的所有点（实际实现中会设置一个点数上限 $K$）。一种替代 range query 方法是 kNN （k 近邻），搜索最近的 k 个点。相较于 kNN，ball query 搜索得到的邻域保证了一个固定的 region 大小，这使得 local region 特征更加具有概括性的，这更有利于局部 pattern 的识别。


**PointNet layer**

此 layer 的输入是 $N'$ 个 local region 中的点，输入 size 为 $N' \times K \times (d+C)$。每个 local region 经 PointNet 输出相应的 feature vector，vector 长度为 $C'$，整个输出 size 为 $N' \times (d+C')$。

local region 中的点坐标被转换为相对于这个 region 中心点的坐标：$x_i^{(j)}=x_i^{(j)} - \hat x^{(j)}$，其中 $j=1,\ldots, d$ 且 $i=1,\ldots, K$，$\hat x$ 是中心点。这起到归一化的作用，且转换后不同 region 的点坐标同等对待，不存在某个region 的点坐标完全大于另一个 region 的点坐标。

## 2.3 非均匀采样密度下的健壮特征学习

不难理解不同的区域，点数密度也不同，非均匀性给 point set 特征学习带来一定的困难。密集数据的特征学习的泛化能力，可能不如稀疏数据的特性学习的泛化能力，另外，稀疏数据训练出来的模型可能对具有细粒度的局部结构无法较好的识别。


理想情况下，我们希望尽可能仔细的去检视高密度区域的细节，但是这会使得低密度区域的点数太少而无法提取 pattern，故低密度区域我们应该在更大的范围内提取更大尺寸的 pattern。为此，作者提出密度可适应的 PointNet layer，如图 2，根据不同的区域密度使用不同的 region 尺寸。下面给出两种不同类型的密度适应 layer。

![](/images/obj_det/3d/pointnet++_2.png)

<center>图 2. (a) MSG; (b) MRG</center>

**Multi-scale grouping (MSG)**

如图 2 (a)，使用不同尺寸（半径）的 grouping layer，每个尺寸有自己的一个 PointNet，用于提取这个尺寸下的 feature，不同尺寸下的特征最终被 concatenate 形成一个多尺寸 feature。

为了让网络能够学习到一个最佳的 combine 多尺寸 feature 的策略，作者使用了 random input dropout。具体而言，对每个训练点集，从 $[0, p]$ 中均匀地随机选择一个概率 $\theta$，以此作为 dropout ratio，然后对点集中的点进行 dropout，这样网络的输入点集的稀疏性就是变化的。注意 test 阶段，需要保留全部点。

**Multi-resolution grouping (MRG)**

MSG 在大尺寸 region 上计算较为耗时，尤其是 low level 的 set abstraction 中，大尺寸 region 的点数更多，计算特别耗时。

作者介绍了另一种方法，如图 2 (b)，region 的某 level 的特征 $L_i$ 由两个 vector 组成：左边的 vector 是对较低 level 的特征 $L_{i-1}$ 的总结，而 $L_{i-1}$ 是使用 set abstraction 应用到 local region 中的点，即 set abstraction 的输出（注意是使用单一 region 尺寸）；右边的 vector 是使用 PointNet 应用到 local region 中的点的输出。

当 local region 中点数较少时，经过 set abstraction 进一步下采样，左边 vector 没有右边 vector 可靠，此时右边 vector 需要更高的权重；反之，左边 vector 具有更精细细节的信息，因为其在 set abstraction 输出 $L_{i-1}$ 的过程中采用了更小的 region 尺寸，参见图 2 (b) 下层的三个更小虚线圆圈。

由于使用了更小 region 尺寸，所以 MRG 比 MSG 计算效率高。

## 2.4 用于 Set 分割的点特征传播

set abstraction 中使用了下采样，然而在 set 分割任务（例如语义点分类标注）中，需要对全部原始点标注。一种方法是以所有点作为中心点，即上面的 $N'=N$，但是这显然会导致计算量增加。另一种方法是将 feature 从下采样点传播到原始点。

如图 1 所示，作者采用一个层级传播策略，即特征传播也有多个 level，是一个层级结构。在某个特征传播 level，将 $N_l \times (d+C)$ 的点特征传播到 $N_{l-1}$ 个点，其中 $N_l < N_{l-1}$，这里 $N_{l-1}$ 和 $N_l$ 分别是第 l 个 set abstraction 的输入和输出点数。

如何进行特征传播？通过对 $N_l$ 个点的特征插值，得到 $N_{l-1}$ 个点特征。

使用哪种插值方法？对于某个不在 $N_l$ 中的点，使用 k 近邻进行插值，权重为距离的倒数，如 (2) 式计算，令 $f$ 表示 $N_l$ 个点特征，$f$ 的 size 为 $N_l \times C$（未考虑坐标，因为各点坐标使用自己原来的坐标，不需要传播），

$$f^{(j)}(x) = \frac {\sum_{i=1}^k w_i(x) f_i^{(j)}}{\sum_{i=1}^k w_i(x)}, \quad w_i(x) = \frac 1 {d(x, x_i)^p} \tag{2}$$

其中 $j=1,\ldots, C$ 是特征向量中元素 index，$i=1,\ldots, N_l$ 是 region 的中心点 index 。默认取 $p=2, k=3$ 。

# 3. 一些算法

## 3.1 FPS

记点集为 $P$，要对其分区，求各个 local region 的中心点，步骤如下：

1. 构造两个集合 $A=\{\}, \ B=P$
2. 从 $B$ 中随机移出一个点 $x$，并将 $x$ 加入 $A$
3. $\forall z \in B$，计算 $z$ 与 $A$ 中所有点的距离 $d_1, d_2, \ldots d_{|A|}$，取最小距离 $d(z)=\min_i d_i$，表示 $z$ 到集合 $A$ 的距离
4. 选择 $B$ 中距离 $A$ 最远的点 $\hat z=\max_z d(z)$，从 $B$ 中移出 $\hat z$ ，并将其加入到 $A$
5. 重复 `3,4` 步骤直到 $A$ 中点数到达设定的值。

代码如下：

```python
def farthest_point_sample(xyz, npoint):
    '''
    xyz: pointcloud data, (B, N, 3), 即上面所说的点集 P
    npoint: 中心点的数量
    '''
    device = xyz.device
    B, N , C = xyz.shape
    # 中心点在点集中的 index
    centroids = torch.zero(B, npoint, dtype=torch.long).to(device)
    # B 中各点到 A 的距离
    distance = torch.ones(B, N).to(device) * 1e10
    # B 中距离 A 最远的点的 index，初始时随机选择一个点（参考上面步骤2）
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_ind = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        # 将最远点加入到 A
        centroids[:, i] = farthest
        # B 中各点到 A 中各点的距离已经计算过，只要计算 B 中点到 A 中新加入的点的距离
        # 如果与 A 中新点的距离更小，那么需要更新 B 中点与 A 的距离（参见下方的 mask）
        centroid = xyz[batch_ind,farthest,:]    # 新加入 A 的中心点坐标
        # 这里没有将 farthest 从 B 中移出，然而这并不影响
        # 要求的是距离 A 中新加入点的距离最远的 B 中点，显然这些点
        # 肯定不是自己，所以可以不用从 B 中移出
        dist = torch.sum((xyz - centroid)**2, -1)
        # 标记 B 中各点与 A 的距离需要调整的 index
        # 这是因为 A 中新加入了点，所以如果 B 中各点与 A 中新点的距离更小，
        mask = dist < distance  
        distance[mask] = dist[mask] # 调小 B 中部分点与 A 的距离
        # 注意 B 中没有按上面步骤中说的那样移出加入 A 中的点，因为加入 A 中的点
        # 与 A 的距离始终为 0，所以这些点肯定不是距离 A 最远点，所以不会被再次选中为 farthest
        farthest = torch.max(distance, -1)[1]   # 获取 B 中下一个距离 A 最远点
    return centroids
```

## 3.2 grouping

采样了中心点后，如何对点集分组？直接看代码，

```python
def query_ball_point(radius, nsample, xyz, new_xyz):
    '''
    radius: ball radius
    nsample: 上文说的 K
    xyz: 被划分前的点集 (B, N, 3)
    new_xyz: 采样出的中心点 (B, S, 3)

    @return: (B, S, nsample)    S 个邻域，每个邻域 nsample 个点
    '''
    device = xyz.device
    B, N, C = xyz.shape
    S = new_xyz.shape[1]
    # B x S 个 0~N
    # 对于每个中心点，求其半径内的点 index
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)    # 计算中心点与点集中各点的距离平方，(B, S, N)
    group_idx[sqrdists > radius ** 2] = N       # 半径以外的 index 全部置为 N（无效index）
    group_idx = group_idx.sort(dim=-1)[0][:,:,:nsample]
    # 取第一个 半径内的点 index，后面将所有位于半径外的点替换为这第一个半径内点
    group_first = group_idx[:,:,0].view(B,S,1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx
```

## 3.3 MSG

数据集仍使用 [pointnet](/2022/10/18/obj_det/3d/pointnet) 中介绍的数据。

使用多个 region scale，看下相关类的构造参数。

```python
class PointNetSetAbastractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        '''
        npoint: 配置的中心点数量
        radius_list: 邻域半径列表，表示使用多个 region scale
        nsample_list: 单个邻域内采样点数量，K
        in_channel: Set Abstraction 输入 channel。
        mlp_list: 使用 MLP，设置 conv2d 输出 channel。 
            MLP 的数量与邻域半径的数量一致，每个半径使用独立的 MLP（其实是 conv_block）进行特征提取
        '''
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()  # 每个半径对应一个独立的 conv_block
        self.bn_blocks = nn.ModuleList()    # 每个 conv_block 由若干个 conv+bn+relu 组成
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3   # +3 是因为特征+坐标，其中坐标=3，参见上面的 set abstraction 一节的分析
            for out_channel in mlp_list[i]:
                # 输入 shape 为 (B, 3, K, S)，S 是邻域数量，K 是一个邻域内点数
                convs.append(nn.Conv2d(last_channel, out_channel, 1))  
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)
```

看下 MSG set abstraction 的前向传播，

```python
def forward(self, xyz, points):
    '''
    xyz: coords of point cloud. (B, C=3, N)
    points: other infos of point cloud, e.g. 法向数据或前一个 set abstraction 的输出特征, (B, D, N)
    '''
    xyz = xyz.permute(0, 2, 1)      # (B, N, C)
    if points:
        points = points.permute(0, 2, 1)
    
    B, N, C = xyz.shape
    S = self.npoint                 # 中心点数量
    # (B, S) -> (B, S, 3): 得到中心点 index，然后获取中心点坐标
    new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
    new_points_list = []

    for i, radius in enumerate(self.radius_list):
        K = self.nsample_list[i]    # number of points in a neighbourhood
        group_idx = query_ball_point(radius, K, xyz, new_xyz)   # 各邻域点的 index, (B, S, K)
        grouped_xyz = index_point(xyz, group_idx)   # 各邻域点的坐标，  (B, S, K, 3)
        grouped_xyz -= new_xyz.view(B, S, 1, C)     # 各邻域点的相对坐标
        if points is not None:  
            grouped_points = index_points(points, group_idx)
            grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
        else:
            grouped_points = grouped_xyz
        
        grouped_points = grouped_points.permute(0, 3, 2, 1)     # (B, 3, K, S)
        for j in range(len(self.conv_blocks)):
            conv = self.conv_blocks[i][j]       # conv2d
            bn = self.bn_blocks[i][j]           # batchnorm2d
            grouped_lists = F.relu(bn(conv(grouped_points)))    # (B, D', K, S)
        new_points = torch.max(grouped_points, 2)[0]            # (B, D', S)，unordered points’ feature
        new_points_list.append(new_points)      # 每个 block 输出的 unordered points' feature
    new_xyz = new_xyz.permute(0, 2, 1)          # (B, 3, S)
    new_points_concat = torch.cat(new_points_list, dim=1)   # (B, D1'+D2'+D3', S)
    return new_xyz, new_points_concat
```

## 3.4 classification

classification 任务使用的模型，

```python
class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        '''
        num_class: 分类数量
        normal_channel: 点云数据中是否包含法向数据
        '''
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1,0.2,0.4], [16,32,128], in_channel, 
                                             [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2,0.4,0.8], [32,64,128], 320, 
                                             [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640+3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)
```

从上面模型定义的代码大概可知，有三个 set abstraction，前两个是 MSG 类型的，最后一个根据图 1 可知其实是用于执行 PointNet 的功能，也就是没有 sample 和 group 这两个功能，但是为了保持 PointNet 中的 MLP（conv_block）结构一致，将点坐标从 `(B, N, 3)` 转为 `(B, 1, N, 3)`，这里 $K=N, \ S=1$，也就是全部点集看作一个邻域，邻域内点数为点集大小，见如下代码

```python
def sample_and_group_all(xyz, points):
    '''
    xyz: (B, N, 3)，点集坐标
    points: (B, N, D)，点特征
    @return: new_xyz, (B, 1, C)，唯一的一个中心点坐标，
                        这是最后一个 set abstraction，中心点坐标用不到了
             new_points, (B, 1, N, C+D), 唯一的邻域点的坐标+特征 concatenation
    '''
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)   # 中心点坐标，只有一个中心点
    grouped_xyz = xyz.view(B, 1, N, C)          # 整个点集看作一个邻域
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)  # 坐标与特征 concatenate
    else:
        new_points = grouped_xyz        # 仅使用坐标，没有特征
    return new_xyz, new_points
```

`sample_and_group_all` 将上一个 set abstraction 输出的点集的坐标和特征进行 concatenate，然后输入到 PointNet，如图 1 的 classification 分支，这个 PointNet 就是一个 conv_block 和 max pooling 的组合，其中每个 conv layer 由 Conv+BN+ReLU 构成，max pooling 用于输出 unordered point set 的 feature。`PointNetSetAbstraction` 的前向传播比较简单，这里不贴出代码了。

每个 set abstraction 输出下采样的点的特征，然后经过 MLP 输出长度为 `num_class` 的向量，作为分类得分预测，具体过程可通过模型的前向传播搞清楚：

```python
def forward(self, xyz):
    '''
    xyz: (B, D, N)  点云数据，包含法向数量则 D=6，否则 D=3
    '''
    B, _, _ = xyz.shape
    if self.normal_channel:
        norm = xyz[:,3:,:]
        xyz = xyz[:,:3,:]
    else:
        norm = None
    # l1_xyz: 下采样的点，各邻域的中心点，(B, 3, S)
    # l1_points: 下采样的点的特征 (B, D1'+D2'+D3', S)，
    # Di' 表示第 i 个邻域半径对应的 conv_block 输出 channel
    # 每个 conv_block 输出特征沿着 channel concatenate，参见图 2(a)
    l1_xyz, l1_points = self.sa1(xyz, norm)
    l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
    # sa3 是一个 fake set abstraction，其内部只有 PointNet，没有 sample 和 group
    l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # (B, C, 1), (B, D', 1)
    x = l3_points.view(B, 1024)
    x = self.drop1(F.relu(self.bn1(self.fc1(x))))
    x = self.drop2(F.relu(self.bn2(self.fc2(x))))
    x = self.fc3(x)     # (B, num_class)
    x = F.log_softmax(x, -1)
    return x, l3_points
```

由于是分类任务，损失采用了负对数似然损失，由于没有对齐特征的 T-Net，故损失中没有 PointNet 中那个正则约束项。

## 3.5 Part-Seg

分割任务中用到了特征传播模块。先看下 part seg 的模型结构代码，

```python
class get_model(nn.Module):
    def __init__(self, num_classes, normal_channel=False):
        super(get_model, self).__init__()
        additional_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel

        # 3+additional_channel，这里比 classification 任务中多了个 3，这是因为分割任务中
        # 第一个 set abstraction 的额外特征信息除了用了 normal，也用到了坐标数据，参考下方的 forward 方法代码
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1,0.2,0.4], [32,64,128], 3+additional_channel,
                                             [[32,32,64], [64,64,128],[64,96,128])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64,128], 128*2+64,
                                             [[128,128,256], [128,196,256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512+3, 
                                          mlp=[256,512,1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])      # 1536=1024+(256*2)
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])       # 579=256+(128*2+64)
        # 150+additional_channel=128+16+3+C
        self.fp1 = PointNetFeaturePropagation(in_channel=150+additional_channel, mlp=[128, 128])    
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
```

forward 方法代码，三个 Set Abstraction 与 classification 中类似，

```python
def forward(self, xyz, cls_label):
    '''
    xyz: (B, C, N)，如果有法向数据，C=6，否则 C=3，点云数据
    cls_label: (B, k)，点云的 one-hot 分类，对于 ShapeNet 数据集，k=16，即 16 种目标分类

    @return:
        1. x：(B, N, num_classes)   B 个点云，每个点云 N 个点，每个点各 part 分类的预测得分
        2. l3_points: (B, 1024, 1) 最后一个 set abstraction 的输出点特征，即点云的全局特征
    '''
    B, C, N = xyz.shape
    if self.normal_channel:
        l0_points = xyz     # 第一个 set abstraction，包含了 coord 和 normal
        l0_xyz = xyz[:,:3,:]# coord 和 normal 一起作为 feature
    else:
        l0_points = xyz     # 第一个 set abstraction 使用 coord 作为 feature，
        l0_xyz = xyz        # 而非 classification 任务中那样没有 feature

    # 前两个 set abstraction 的前向传播与 classification 任务中类似
    l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)     # (B, 3, S=512), (B, 64+128*2, S=512)
    l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)     # (B, 3, S=128), (B, 256*2, S=128)
    # sa3 没有 sample 和 group
    l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)     # (B, 3, 1), (B, 1024, 1)

    # 特征传播
    l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # (B, 256, 128)
    l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # (B, 128,512)
    cls_label_one_hot = cls_label.view(B, 16, 1).repeat(1, 1, N)# (B, 16, N)

    # 分类one-hot，l0 点坐标，l0 点特征，concatenate，第 2 维为 16+3+C，输出 l0_points (B, 128, N)
    l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot, l0_xyz, l0_points], 1), l1_points)
    feat = F.relu(self.bn1(self.conv1(l0_points)))  # (B, 128, N)
    x = self.drop1(feat)
    x = self.conv2(x)       # (B, num_classes, N)
    x = F.log_softmax(x, dim=1)
    x = x.permute(0, 2, 1)  # (B, N, num_classes)
    return x, l3_points     # (B, N, num_classes), (B, 1024, 1)
```


**特征传播**

set abstraction 得到下采样的点的特征，而分割任务则需要所有点的特征，即特征的分辨率需要再次增大，作者采用的方法是，将下采样的点特征进行插值，然后与自底向上路径的特征 concatenate，类似 FPN 结构（FPN 采用 conv 增大特征 spatial size，然后是 elementwise sum 融合）。

两个进行 concatenate 的特征其 pattern 肯定不同，故 concatenate 之后还需要经过 conv_block 进一步融合特征。


代码如下，

```python
class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

def forward(self, xyz1, xyz2, points1, points2):
    '''
    xyz1: 上一层的点坐标，  (B, C, N)
    xyz2: 这一层的点坐标（即上一层点的下采样，centroids）(B, C, S)
    points1: 上一次的点特征 (B, D1, N)
    points2: 这一层的点特征 (B, D2, S)
    @return:
            上采样后的点特征    (B, D3, N)
    '''
    xyz1 = xyz1.permute(0, 2, 1)            # (B, N, C)
    xyz2 = xyz2.permute(0, 2, 1)            # (B, S, C)

    points2 = points2.permute(0, 2, 1)      # (B, S, D2)
    B, N, C = xyz1.shape
    _, S, _ = xyz2.shape

    if S == 1:      # 只有 1 个点的特征，无法进行插值，直接复制
        interpolated_points = points2.repeat(1, N, 1)  # (B, N, D2)
    else:
        dists = square_distance(xyz1, xyz2) # (B, N, S), 距离平方
        # 对于每个 xyz1 中的点，取 k=3 最近邻，进行插值
        dists, idx = dists.sort(dim=-1)     # (B, N, S), 最后一维排序
        dists, idx = dists[:,:,:3], idx[:,:,:3]

        dist_recip = 1.0 / (dists + 1e-8)   # 距离平方的倒数作为 weight, (B, N, 3)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)   # (B, N, 1)，(2) 式分母
        weight = dist_recip / norm          # 归一化的 weight, (B, N, 3)
        # 根据 idx 获取 k=3 近邻的点特征，  (B, N, 3, D2)，加权求和 -> (B, N, D2)
        interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

    if points1 is not None:
        points1 = points1.permute(0, 2, 1)  # (B, N, D1)
        new_points = torch.cat([points1, interpolated_points], dim=-1)  # (B, N, D1+D2)
    else:
        new_points = interpolated_points
    
    new_points = new_points.permute(0, 2, 1)        # (B, D1+D2, N)
    for i, conv in enumerate(self.mlp_convs):
        bn = self.mlp_bns[i]
        new_points = F.relu(bn(conv(new_points)))
    return new_points
```

## 3.6 Sem-Seg

语义分割，与 Part-Seg 类似，不再具体分析。

