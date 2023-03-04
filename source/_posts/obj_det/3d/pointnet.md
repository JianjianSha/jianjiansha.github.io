---
title: PointNet 论文解读
date: 2022-10-18 14:13:14
tags: 3d object detection
mathjax: true
---

论文：[PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.0053)

源码：[yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git)

PointNet 直接以点云作为输入，而非像 VoxelNet 那样先将点云转为 3d voxel grids，这么做是为了避免丢失输入信息，以及节省了点云数据的预处理时间。

PointNet 输出可以是整个输入的类标签，或者某些部分的标签，或者每个点的分割标签。

# 1. PointNet 框架

如图 1 所示，

![](/images/obj_det/3d/pointnet_1.png)
<center>图 1. PointNet 框架</center>

网络架构主要包含三个模块：1. max pooling；2. 局部和整体的信息融合；3. 两个对齐子网络，分别对齐输入点和点的特征。

## 1.2 适用无序输入的对称函数

由于是直接使用点云作为网络输入，而点云的一个性质就是无序性，即改变点云的顺序，网络输出应该保持不变。一个点云图，取前 $n$ 个点作为网络输入，那么对这 $n$ 个点做 permutation，网络输出保持不变。要达到这个要求，作者考虑了三种方法：

1. 将输入按某个规则排序，也就是说序列经 permutation 后再进行排序，顺序总是相同的。
2. 类似 RNN 那样，输入是序列，对输入序列进行各种 permutation，即训练集的扩增。
3. 使用一个简单的对称函数以聚合点云中所有点的信息。

    这里的对称函数是指将点序列作为输入，无论如何改变输入点的顺序，对称函数总是输出某个固定的向量，例如 $+, \ *$ 相加和相乘对输入序列的顺序 invariant

对于方法 1，在高维空间中不存在这样一个排序规则，例如按点的坐标大小进行排序，一维情况下各点容易排序，二维以及更高维则无法比较坐标大小。

对于方法 2，通过随机 permute 输入序列，然后训练 RNN，使得 RNN 对输入序列的顺序不敏感，从而达到 invariant 的目的，然而 “OrderMatters” 作者认为 RNN 的输入顺序是对输出有影响的，这种影响无法忽略，并且 RNN 对短序列顺序相对不敏感，但是对长序列（成百上千的长度）则缺乏 robustness。

故最终考虑方法 3：对输入点进行变换，然后使用一个对称函数，输出与输入序列顺序无关。

使用数学表达式进行描述，

一个输入序列 $x_1,\ldots, x_n$，满足 $x_i \in \mathbb R^N$，这里考虑一般化情况，而不仅仅是原始的输入点云中的点，数据维度不一定是 3，故使用 $N$ 表示输入数据的维度，

$$f(\{x_1,\ldots,x_n\})\approx g(h(x_1),\ldots, h(x_n)) \tag{1}$$

其中 $f: 2^{\mathbb R^N} \rightarrow \mathbb R^K$ 是整体变换函数，可以分解为一个普遍变换函数 $h: \mathbb R^N \rightarrow \mathbb R^K$ 和一个对称函数 $g: \underbrace {\mathbb R^K \times \cdots \times \mathbb R^K}_n \rightarrow \mathbb R^K$ （这里我觉得论文中输出维度写错了）。

PointNet 中，作者使用一个 MLP（多层感知机）来近似 $h$，而 $g$ 则使用 max pooling 来近似。如图 1 中，$n \times 1024$ feature maps 的左边 mlp（对应 $h$）和右边 max pool（对应 $g$），输出为 global feature。

## 1.2 局部和整体信息的聚合

上一小节的输出应该为一个向量 $[f_1, \ldots, f_K]$，这是输入序列的整体特征，根据此特征，我们可以使用一个 SVM 或者 MLP 作为分类器，记分类总数为 $k$，那么分类器输出 unit 数量为 $k$，得到输入整体的 label 预测（非归一化得分）。

对于点的分割任务，则需要合并局部和整体信息。如图 1 中右下角的红色标注部分，$n \times 64$ 的局部信息与 $1 \times 1024$ 的整体信息进行 concatenate，得到 $n \times 1088$ 的信息，然后经过 MLP（输出 channel 依次为 512, 256, 128）得到 $n \times 128$ 的特征，然后再继续使用 MLP 得到 $n \times m$ 的输出结果。

MLP 中一层由 `fc+bn+relu` 组合而成。

## 1.3 联合对齐网络

对点云进行几何变换（刚性变换），其语义 label 应该 invariant，一个自然的解决方法是将输入对齐到某个 canonical 空间，这个对齐操作指对输入应用某个变换，显然这个变换应该跟输入相关，我们使用一个 mini 网络，mini 网络输入就是这里要被对齐的输入，mini 网络输出就是这个对齐的变换操作矩阵，如图 1 中的 T-net。

除了对齐原始的点云数据输入，还可以对齐特征空间中的特征，相比较于原点云数据输入，特征空间的维度更高，这会导致优化困难大大增加，为了解决此问题，损失函数中增加一个正则项，

$$L_{reg} = \|I-AA^{\top}\|_F^2 \tag{2}$$

(2) 式中 $A$ 就是 mini 网络输出的对齐变换矩阵，(2) 式这个正则项表示约束并使得 $A$ 趋于一个正交矩阵，因为正交矩阵不会改变输入的形状结构，仅仅是使其旋转，所以不会丢失输入的信息。

# 2. 代码分析

## 2.1 分类

如图 1 的上面部分（Classification Network），model 代码为

```python
class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=True):
        '''
        k: 分类总数
        normal_channel: 点云数据中是否包含法向信息
        '''
        super(get_model, self).__init__()
        if normal_channel:  # 点云中的点有法向数据，点 (x,y,z,nx,ny,nz)
            channel = 6
        else:
            channel = 3     # 没有法向数据，则点 (x,y,z)
        # feat 对应图 1 上面部分的 point feature 提取部分，输出 global feature
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        # 对应图 1 右上部分的 mlp(512, 256, k)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        '''
        x: 输入点云，每个点为 (x, y, z) 不包含法向信息。
            x 的 shape (B, 3, N)，B 为 batch size，N 为每个点云中前 N 个点。
        '''
        x, trans, trans_feat = self.feat(x)
        # x: global feature, (B, 1024)
        # trans: 图 1 输入变换中的 T-net 输出的 3x3 变换矩阵，(B, 3, 3)
        # trans_feat: 图 1 特征变换中的 T-net 输出的 64x64 变换矩阵，(B, 64, 64)
        x = F.relu(self.bn1(self.fc1(x)))       # (B, 512)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)                         # (B, k)
        x = F.log_softmax(x, dim=1)             # (B, k)
        # x: 各分类的概率 log
        return x, trans_feat
```

了解了 classification 网络的整体框架后，我们具体看是如何得到 global feature 的，如下

```python
class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)
    
    def forward(self, x):
        '''
        x: point clouds, (B, D=3, N)
        '''
        B, D, N = x.size()
        # 输入变换中的 T-net
        trans = self.stn(x)     # (B, 3, 3)
        x = x.transpose(2, 1)   # (B, N, 3)
        if D > 3:
            feature = x[:,:,3:]
            x = x[:,:,:3]
        x = torch.bmm(x, trans) # 批量矩阵相乘 (B, N, 3)
        # x: aligned input
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        # x 对应图 1 上面的 input transform 右边的那个 nx3 部分
        x = x.transpose(2, 1)   # (B, 3, N)
        x = F.relu(self.bn1(self.conv1(x))) # (B, 64, N)

        # 下面进行 feature transform
        if self.feature_transform:
            # trans_feat: 用于对齐特征的变换矩阵
            trans_feat = self.fstn(x)       # (B, 64, 64)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)    # (B, N, 64)
            x = x.transpose(2, 1)           # (B, 64, N)
        else:
            trans_feat = None

        pointfeat = x                       # (B, 64, N)
        x = F.relu(self.bn2(self.conv2(x))) # (B, 128, N)
        x = self.bn3(self.conv3(x))         # (B, 1024, N)
        x = torch.max(x, 2, keepdim=True)[0]# (B, 1024, 1)
        x = x.view(-1, 1024)                # (B, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            # concatenate local and global info
            x = x.view(-1, 1024, 1).repeat(1, 1, N) # (B, 1024, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat
```

下面再看 T-Net 结构代码，

```python
class STNkd(nn.Module):     # k-dim T-Net
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        '''
        x: 输入  (B, k, N)
        @return: 对齐 x 的变换矩阵 (B, k, k)
        '''
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x))) # (B, 64, N)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x))) # (B, 1024, N)
        x = torch.max(x, 2, keepdim=True)[0]# (B, 1024, 1)

        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))   # (B, 512)
        x = F.relu(self.bn5(self.fc2(x)))   # (B, 256)
        x = self.fc3(x)                     # (B, k*k)

        # identity: (B, k*k)
        identity = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k*self.k).repeat(batchsize, 1)
        if x.is_cuda:
            identity = identity.cuda()
        x = x + identity
        x = x.view(-1, self.k, self.k)
        return x    # (B, k, k)
```

实验使用 modelnet40 数据集，一个点云图像包含很多点，实验中仅取前 $N=1000$ 个点，一个点云图像对应一个分类 id，

部分训练代码如下，

```python
criterion = model.get_loss()

for batch_id, (points, target) in enumerate(trainloader):
    # points (B, N, 3), target (B,)
    optimizer.zero_grad()

    points = points.data.numpy()
    # 随机丢掉部分点，使用点云图像中第一个点进行填充，使的一个点云图像依然是 N 个点
    points = provider.random_point_dropout(points)  
    # 点坐标随机伸缩，同一个点云图中的点缩放因子相同
    points[:,:,:3] = provider.random_scale_point_cloud(points[:,:,:3])  
    # 随机平移
    points[:,:,:3] = provider.shift_point_cloud(points[:,:,:3])
    points = torch.Tensor(points).transpose(2, 1)   # (B, 3, N)

    if not args.use_cpu:
        points, target = points.cuda(), target.cuda()

    # pred: (B, k)
    # trans_feat: (B, 64, 64)
    pred, trans_feat = classifier(points)
    loss = criterion(pred, target.long(), trans_feat)
    pred_choice = pred.data.max(1)[1]                   # (B,)

    correct = pred_choice.eq(target.long().data).cpu().sum()    # scalar
    correct = correct.item() / float(points.size()[0])
    loss.backward()
    optimizer.step()
```

损失计算的代码如下，

```python
class get_loss(nn.Module):
    def __init__(self, mat_diff_loss_scale=0.01):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        # 负对数似然
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_regularizer(trans_feat)
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss

def feature_transform_regularizer(trans):
    '''
    trans: (B, 64, 64), 对齐特征的变换矩阵
    '''
    d = trans.size()[1]     # 特征维度 d=64
    I = torch.eye(d)[None, :, :]    # (1, d, d)
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1,2)))
    return loss
```

## 2.2 part 分割

Part 分割是给目标物体的 part 标注 label。 例如 椅子腿是椅子这个 object 的 part，杯子把手是杯子这个 object 的 part。

### 2.2.1 数据集

ShapeNet：包含了 16881 个 shape（或者说是 object），来自 16 个分类，总共标注了 50 种 part。大部分 object 被标注了 2~5 个 part。

16 个 object 分类：

```
airplane bag cap car  chair earphone guitar knife lamp laptop moterbike mug pistol rocket skateboard table
```

每个 object 分类有若干个 part 分类，总共 50 个 part 分类。

```python
self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
```

每个 `.txt` 文件中包含了这个 object 的点云数据： xyz 三个坐标，以及法向三个量，最后一个是所属 part id。

PartNormalDataset 类中，

```python
def __getitem__(self, index):
    '''index: int, sample index'''
    file_path, cat = self.datapath[index]
    cat_id = self.classes[cat]              # object 分类 id
    cls = np.array([cls]).astype(np.int32)
    data = np.loadtxt(file_path).astype(np.float32) # 加载 object 中的点云数据
    point_set = data[:,0:3]     # 取点云的 xyz 三个坐标数据
    seg = data[:,-1].astype(np.int32)       # 点的 part id
    point_set[:,0:3] = pc_normalize(point_set[:,0:3])
    choice = np.random.choice(len(seg), self.npoints, replace=True)     # 随机抽样选取 npoints=2048 个点
    point_set = point_set[choice, :]
    seg = seg[choice]
    return point_set, cls, seg      # (npoints, 3), (1,), (npoints,)
```

这里 `pc_normalize` 是对点云数据求均值中心点坐标，然后计算点云中每个点与中心点的相对坐标，计算欧氏距离，求最大欧氏距离，然后将相对坐标除以最大欧式距离，进行归一化。

### 2.2.2 Network

直接看前向传播函数，

```python
def forward(self, point_cloud, label):
    '''
    point_cloud: (B, D=3, N=2048), 点云的 xyz 坐标，如果使用法向数据，那么 D=6
    label: (B, C=16)，object 的 shape 分类 id
    '''
    B, D, N = point_cloud.size()
    # ① network 与 classification 任务中类似，input 经过一个 T-Net 进行 align 操作，
    trans = self.stn(point_cloud)       # 获取 T-Net 的输出：一个用于 align 输入的变换矩阵 (B, D, D)
    point_cloud = point_cloud.transpose(2, 1)
    point_cloud = torch.bmm(point_cloud, trans).tranpose(2, 1)  # (B, D, N)
    # ② 然后是使用 MLP 得到 feature，
    out1 = F.relu(self.bn1(self.conv1(point_cloud)))            # (B, 64, N)
    out2 = F.relu(self.bn2(self.conv2(out1)))                   # (B, 128, N)
    out3 = F.relu(self.bn3(self.conv3(out2)))                   # (B, 128, N)
    # ③ 然后对 feature 使用另一个 T-Net 进行 align 操作
    trans_feat = self.fstn(out3)    # 用于 align feature 的变换矩阵 (B, 128, 128)
    x = out3.transpose(2, 1)
    net_transformed = torch.bmm(x, trans_feat).transpose(2, 1)  # (B, 128, N)

    # ④ 然后再经过 MLP + max pooling 操作，得到 global feature
    out4 = F.relu(self.bn4(self.conv4(net_transformed)))        # (B, 512, N)
    out5 = self.bn5(self.conv5(out4))                           # (B, 2048, N)
    out_max = torch.max(out5, 2)[0]                             # (B, 2048)
    # ⑤ 类似于 conditional-gan 的思路，将 global feature 与 label 进行 concatenate
    # 然后扩展维度，以便与 local feature 进行 concatenate。使用了 multi-level features
    out_max = torch.cat([out_max, label], dim=1)                # (B, 2048+16)
    expand = out_max.view(-1, 2048+16, 1).repeat(1, 1, N)       # (B, 2048+16, N)
    concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)   # ch=2048+16+64+128*2+512+2048
    net = F.relu(self.bns1(self.convs1(concat)))                # (B, 256, N)
    net = F.relu(self.bns2(self.convs2(net)))                   # ch = 256
    net = F.relu(self.bns3(self.convs3(net)))                   # ch = 128
    net = self.convs4(net)                                      # ch = k = 50 (part 分类总数)
    net = net.transpose(2, 1).contiguous()                      # (B, N, k)
    net = F.log_softmax(net, dim=-1)
    return net, trans_feat          # (B, N, k), (B, 128, 128)
```

损失与 classification 中类似。


## 2.3 语义分割

### 2.3.1 数据集

使用 s3dis 数据集，物体的分类共 13 个，依次为

```
ceiling, floor, wall, beam, column, window, door, table,  chair, sofa, bookcase, board, clutter
```

每个 room 作为一个样本，room 中的所有物体全部 concatenate，得到 room 中所有物体的所有点云，每个点的 label 为这个点所属物体的 class id。

**均衡分类**

由于各分类点数不均衡，所以统计数据集中所有 room 中各 label 的所有点数，记为 `labelweights`，归一化后取倒数，然后求其 $1/3$ 的幂值（权重衰减），作为各 label 的权重。

```python
# labelweights: 数据集中各 class 对应的所有点数，shape 为 (13,)
labelweights = labelweights.astype(np.float32)
self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
```

**采样概率**

每个样本（一个 room）被选中的概率与其中所含点数成正比，
```python
sample_prob = num_point_all / np.sum(num_point_all)
```

**固定单个样本中点数**

每个 room 中含有大量点，根据配置，每个样本取 `4096` 个点，但是不能直接取前 4096 个点，否则每个 room 中每次训练都是使用相同的物体，其他物体则一直不被使用，由于物体的点云在 $x-y$ 平面的投影是一个个团簇，所以我们设置一个 1 单位的 bbox，然后随机选择一个点，以此点为中心，如果其 1 单位的 bbox 内存在超过 1024 个点（1024 是配置的超参数，我们认为超过这个值就算一个有效的物体点云），那么我们就随机选择这个 bbox 内的 4096 个点（如果 bbox 内不足 4096 个点，则可以进行有放回的随机选取）。

**归一化样本数据**

```python
current_points = np.zeros(self.num_point, 9)    # (4096, 9)
# selected_points: (4096, 6)，上文所说的 bbox 内随机选取的 4096 个点，xyzrgb
# room_coord_max: (num_rooms, 3)，每个 room 中点坐标的 xyz 最大值
# center: 上面所说的随机选择的 bbox 的中心

# x, y, z 归一化
current_points[:,6] = selected_points[:,0] / self.room_coord_max[room_idx][0]
current_points[:,7] = selected_points[:,1] / self.room_coord_max[room_idx][1]
current_points[:,8] = selected_points[:,2] / self.room_coord_max[room_idx][2]
# 所选点沿 xy 轴距中心的距离
selected_points[:,0] -= center[0]
selected_points[:,1] -= center[1]
# rgb 颜色归一化
selected_points[:,3:6] /= 255.0
current_points[:,0:6] = selected_points
```

返回的单个样本的数据 shape 为 `(4096, 9)`，包含了各点的 `(offset_x, offset_y, offset_z, r,g,b,x,y,z)`，其中 `offset_z` 就是 point 原来的未归一化的 z 值 `z_0`， label 为各点的 class id， shape 为 `(4096,)`。

### 2.3.2 Network

分割模型与 classification 任务中的模型相似，如图 2， 只是得到 global feature 后需要与 local feature 进行 concatenate，然后再经 MLP 得到各点的分类预测。

注意，`PointNetEncoder` 的参数 `channel=9`，这与上面样本数据的单个点的维度 9 是一致的。


### 2.3.3 损失

损失同 classification 任务中损失计算，需要注意的是这里还需要使用各 label 的权重，即 train dataset 中的 `self.labelweights`。


### 2.3.4 测试

训练时使用样本的一个随机选择的 bbox，测试时如果随机选择这样一个 bbox，显然不合适，代码中使用了一个新类用于获取 room 中全场景点云数据。

数据的加载与训练阶段相同，在取样函数 `__getitem__` 逻辑不同，这里陈述其过程如下：

1. 根据样本（room） index，获取 room 中所有点云数据 (N, 6)，维度 `6` 包含非归一化 xyzrgb，以及 room 中所有点的 label (N,)。
2. 获取当前 room 中 xyz 坐标的最小和最大值
3. xy 轴，从 `x_min, y_min` 开始，以 `stride=0.5`，`window_size=1` 进行滑窗，获取每个滑窗中的点
4. 一个块大小为 block_size=4096，即点云中的点按 block 划分作为最小数据结构， 这是配置的超参数，如果滑窗中的点数不是 block_size 的整数倍，那么从滑窗点中随机抽样补充，直到点数是 block_size 的整数倍。随机打乱滑窗中点的顺序

    ```python
    # 随机抽样补充滑窗中点数为 block_size 的整数倍，然后随机打乱点顺序
    num_batch = int(np.ceil(point_idxs.size / self.block_size))
    point_size = int(num_batch * self.block_size)
    replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
    point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
    point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
    np.random.shuffle(point_idxs)
    data_batch = points[point_idxs,:]   # (point_size, 6)
    ```

5. 滑窗中的点 xyz 坐标归一化，注意这里的归一化是基于整个 room 中的点坐标，而不是基于当前滑窗中的点坐标。然后获取 xy 轴上各点与滑窗中心的距离，并将 rgb 归一化

    ```python
    normalized_xyz = np.zeros((point_size, 3))
    normalized_xyz[:,0] = data_batch[:,0] / coord_max[0]    # coord_max 是对整个 room 内点的统计最大
    normalized_xyz[:,1] = data_batch[:,1] / coord_max[1]
    normalized_xyz[:,2] = data_batch[:,2] / coord_max[2]
    data_batch[:,0] = data_batch[:,0] - (s_x + self.block_size / 2.0)
    data_batch[:,1] = data_batch[:,1] - (s_y + self.block_size / 2.0)
    data_batch[:,3:6] /= 255.0
    data_batch = np.concatenate((data_batch, normalized_xyz), axis=1)   # (point_size, 6)
    ```

6. 整合所有滑窗的数据，窗口滑动顺序是从左到右，从上到下。

    ```python
    # append 当前滑窗的点云数据    (\sum point_size, 6)
    data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch
    # append 当前滑窗点的 class id， (\sum point_size,)
    label_room = np.hstack([label_room, label_batch]) if label_room.size else label_batch
    # append 当前滑窗点的分类权重（参考前面的 labelweights）， (\sum point_size,)
    sample_weight = np.hstack([sample_weight, batch_weight]) if samaple_weight.size else batch_weight
    # append 当前滑窗点的 index（基于 room 内所有点 points）， (\sum point_size,)
    index_room = np.hstack([index_room, point_idxs]) if index_room.size else point_idxs
    ```

7. 将 room 内所有滑窗的 concatenate 的数据按 block 切割，block_size=4096 是我们配置好的，适合网络处理的点数

    ```python
    # \sum point_size = b * 4096
    data_room = data_room.reshape((-1, self.block_size, data_room.shape[1]))    # (b, 4096, 6)
    label_room = label_room.reshape((-1, self.block_size))                      # (b, 4096)
    sample_weight = sample_weight.reshape((-1, self.block_size))                # (b, 4096)
    index_room = index_room.reshape((-1, self.block_size))                      # (b, 4096)
    ```

单个样本中所有滑窗的点整合起来然后按 block 切分，得到数据 shape 为 `(b, 4096, 6)`，但是这里 `b` 可能会很大，我们配置 `batch_size=32`，所以需要将 `b` 继续按 `batch_size` 分割。

**投票**

每个样本（room），预测 `num_votes` 次，统计 每个 point 的预测 class id，例如 point1，
预测为 class id `i` 的次数为 `ni`，那么 \sum ni = num_votes
取最大次数 \argmax ni 对应的 class id `i`，整个过程相当于投票
为什么要进行投票呢？训练好模型后，预测结果不是确定的吗，难道每次预测结果还能不同？
输入不变，模型预测结果肯定也不变，但是前面分析到，为了使得点数是 block_size 的整数倍，
进行了随机重采样，所以实际上，同一个样本（room），每次喂给网络的数据都有一点不同。

```python
def add_vote(vote_label_pool, point_idx, pred_label, weight):
    '''
    vote_label_pool: (num_points, 13),  num_points 是一个样本中所有点的数量
    point_idx:  (BATCH_SIZE, 4096)，分割后的小批量 block 中点相对于整个 room 的 index
    pred_label: (BATCH_SIZE, 4096), 分割后的小批量的 block 的预测 class id
    weight: (BATCH_SIZE, 4096)，分割后的小批量 block 中点的 label 对应的权重
    '''
    B, N = pred_label.shape[:2]
    for b in range(B):
        for n in range(N):
            # 这里还检测了 weight 的有效性，即这个 class id 必须要在数据集中出现过
            # 例如预测的分类 id 为 ceiling，但是测试数据集中并没有出现过 ceiling
            # 那么不对这个分类 id 进行投票
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool
```
