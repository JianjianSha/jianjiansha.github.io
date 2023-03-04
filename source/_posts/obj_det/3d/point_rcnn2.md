---
title: PointRCNN 论文解读（二）
date: 2022-11-14 09:42:55
tags: 3d object detection
---

前面我在 [PointRCNN 论文解读（一）](/obj_det/3d/2022/10/24/point_rcnn) 中介绍了 PointRCNN 的模型结构，以及 RPN 的训练代码。由于 PointRCNN 内容较多，所以分成两篇文章来介绍，这篇文章重点介绍第二部分 RCNN 网络。

# 1. RCNN

具体而言， [PointRCNN 论文解读（一）](/obj_det/3d/2022/10/24/point_rcnn)  中给出 RCNN 网络的数据准备，以及预测由 RPN 得到的 3D proposal 的位置精调和目标分类。这篇文章以代码为主，对代码进行注释并穿插讲解原理。

项目 github 代码的 readme 文档中，训练 RCNN 过程为：假设 RPN 网络训练好，模型保存在文件 `output/rpn/default/ckpt/checkpoint_epoch_200.pth`，那么训练 RCNN 有两种策略：
1. 固定 RPN，训练 RCNN。在线进行 GT 扩增。

    ```sh
    python train_rcnn.py --cfg_file cfgs/default.yaml --batch_size 4 --train_mode rcnn --epochs 70 --ckpt_save_interval 2 --rpn_ckpt ../output/rpn/default/ckpt/checkpoint_epoch_200.pth
    ```

2. 离线进行 GT 扩增。

    - 生成扩增的离线场景

        ```sh
        python generate_aug_scene.py --class_name Car --split train --aug_times 4
        ```
    
    - 保存 RPN 特征和 proposals。具体命令参见 readme 文档。这里略

    - 此时可训练 RCNN。命令参见 readme 文档。

由于第一种策略更加优雅，所以本文以第一种策略为例进行原理和代码讲解。

由于启用了 RPN ，所以整个网络的输入依然与训练 RPN 相同，数据准备与训练 RPN 过程中的数据准备相似，具体可阅读 [PointRCNN 论文解读（一）](/obj_det/3d/2022/10/24/point_rcnn) 中相关内容，这里仅列出 `__getitem__` 方法的返回结果，代码如下，注意不需要计算 `rpn_cls_label` 和 `rpn_reg_label`，因为不需要计算 RPN 的损失。

```python
def get_rpn_sample(self, index):
    ...
    if cfg.RPN.FIXED:
        sample_info['pts_input'] = pts_input
        sample_info['pts_rect'] = aug_pts_rect
        sample_info['pts_feature'] = ret_pts_features
        sample_info['gt_boxes3d'] = aug_gt_boxes3d
        return sample_info
```



## 1.1 模型

此时 RPN 和 RCNN 均启用，且 RPN 是 fixed，即网络参数固定不变。

```python
elif args.train_mode == 'rcnn':
    cfg.RCNN.ENABLED = True
    cfg.RPN.ENABLED = cfg.RPN.ENABLED = True    # 启动 RPN，且其 FIXED 属性为 True
```


然后看模型定义，

```python
# lib/net/point_rcnn.py
class PointRCNN(nn.Module):
    def __init__(self, num_classes, use_xyz=True, mode='TRAIN'):
        ...         # 定义 RPN 网络，接着定义 RCNN 网络，如下
        rcnn_input_channels = 128   # RPN 网络的 backbone 的输出特征 channel
        if cfg.RCNN.BACKBONE == 'pointnet': # RCNN 网络仅用于精调，不需要很复杂故采用 pointnet
            self.rcnn_net = RCNNNet(num_classes=num_classes, 
                                    input_channels=rcnn_input_channels, use_xyz=use_xyz)
        elif cfg.RCNN.BACKBONE == 'pointsift':
            pass
        else:
            raise NotImplementedError
```

PointNet 其实是将点云数据经 MLP 最后在进行 global pooling，得到一维特征（shape 为 `(B,1024)`）后再与网络中间层特征（shape 为 `(B, npoints, 64)`）进行 concatenate（将一维全局特征 `(B, 1024)` repeat 为 `(B, npoints)` 然后再与中间层特征 concatenate 成 `(B, npoints, 1088)` 的特征），然后再经 MLP 得到 `(B, npoints, m)` 的特征，这里 m 指分类数量，也就是代码中的 `num_classes`。更多 PointNet 细节可参考 [PointNet 论文解读](/obj_det/3d/2022/10/18/pointnet)。

RPN 是训练好的，且在训练 RCNN 过程中保持固定不变，故加载 RPN 模型参数，

```python
if args.rpn_ckpt is not None:
    total_keys = model.state_dict().keys().__len__()
    train_utils.load_part_ckpt(model, filename=args.rpn_ckpt, logger=logger, total_keys=total_keys)
```

具体加载部分网络参数的函数定义为，

```python
def load_part_ckpt(model, filename, logger=cur_logger, total_keys=total_keys):
    checkpoint = torch.load(filename)
    model_state = checkpoint['model_state']

    update_model_state = {key: val for key, val in model_state.items() if key in model.state_dict()}
    state_dict = model.state_dict()
    state_dict.update(update_model_state)
    model.load_state_dict(state_dict)
```

加载了预训练的 RPN 模型后，我们分析整个模型的前向传播过程，

```python
def model_fn(model, data):
    '''
    model: 整个模型 RPN+RCNN
    data: 一批训练据。dict 类型
    '''
    # inputs: (B, npoints, 3) 点云数据坐标
    # gt_boxes3d: (B, max_gt, 7) 批样本中的 gt box3d，包含 x,y,z,h,w,l.ry。
    #             max_gt 是这批中单个样本中最大gt 数量
    input_data = {'pts_input': inputs, 'gt_boxes3d': gt_boxes3d}

    ret_dict = model(input_data)    # 模型前向传播

    tb_dict = {}
    disp_dict = {}
    loss = 0
    if cfg.RCNN.ENABLED:
        rcnn_loss = get_rcnn_loss(model, ret_dict, tb_dict)
        loss += rcnn_loss           # 这是 RCNN 子网络的损失
```

从以上代码中可见，固定 RPN 网络参数，模型输入与原来训练 RPN 时相同，输入经过模型前向传播后，再计算 RCNN 的损失。所以我们接下来先看前向传播，然后再看如何计算损失。

## 1.2 forward

```python
# class PointRCNN

def forward(self, input_data):
    if cfg.RPN.ENABLED:
        output = {}
        with torch.set_grad_enabled((not cfg.RPN.FIXED) and self.training):
            if cfg.RPN.FIXED:
                self.rpn.eval()     # 训练 RCNN 时，固定 RPN，故 RPN 网络运行于 EVAL 模式
            rpn_output = self.rpn(input_data)
            output.update(rpn_output)
    
        if cfg.RCNN.ENABLED:
            with torch.no_grad():
                # (B, npoints, 1), (B, npoints, 76)
                rpn_cls, rpn_reg = rpn_output['rpn_cls'], rpn_output['rpn_reg']
                # 原始输入点云数据 (B, npoints, 3)
                # backbone 输出特征 (B, 128, npoints)
                backbone_xyz, backbone_features = rpn_output['backbone_xyz'], rpn_output['backbone_features']
                rpn_scores_raw = rpn_cls[:, :, 0]        # 未归一化的点二分类得分，(B, npoints)
                rpn_scores_norm = torch.sigmoid(rpn_score_raw)   # RPN 输出二分类得分归一化到 (0, 1)
                seg_mask = (rpn_scores_norm > cfg.RPN.SCORE_THRESH).float() # (B, npoints)
                # (B, npoints) 转换到 CCS 坐标，丢失了深度信息，故增加一个点到相机距离作为深度信息
                pts_depth = torch.norm(backbone_xyz, p=2, dim=2)    # (B, npoints,)

                # (B, RPN_POST_NMS_TOP_N, 7),  (B, RPN_POST_NMS_TOP_N)。top n 的点对 box3d 的预测，以及
                # top n 的点二分类得分预测。第二维度实际有效的点的数量不足 RPN_POST_NMS_TOP_N，不足的使用 0 填充
                rois, roi_scores_raw = self.rpn.proposal_layer(rpn_score_raw, rpn_reg, backbone_xyz)

            # save rois, roi_scores_raw, seg_mask to output
            rcnn_input_info = {'rpn_xyz': backbone_xyz,                             # (B, npoints, 3)
                               'rpn_features': backbone_features.permute(0, 2, 1),  # (B, npoints, 128)
                               'seg_mask': seg_mask,                                # (B, npoints)
                               'roi_boxes3d': rois,
                               'pts_depth': pts_depth}                              # (B, npoints)
            if self.training:
                rcnn_input_info['gt_boxes3d'] = input_data['gt_boxes3d']
            rcnn_output = self.rcnn_net(rcnn_input_info)
            output.update(rcnn_output)
    return output
```

这里根据 RPN 输出结果来生成 3D proposals 的过程由 `ProposalLayer` 类实现，

```python
class ProposalLayer(nn.Module):
    def __init__(self, mode='TRAIN'):
        super().__init__()
        self.mode = mode
        self.MEAN_SIZE = torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda()
    
    def forward(self, rpn_scores, rpn_reg, xyz):
        '''
        rpn_scores: RPN 点二分类 fg/bg 预测，未归一化，(B, N)
        rpn_reg: RPN 点对相关 gt box3d 的预测，(B, N, 76)
        xyz: 输入点云的坐标，(B, N, 3)
        :return bbox3d: (B, M, 7)
        '''
        batch_size = xyz.shape[0]
        # 根据点云中每个点进行 local region 的预测，得到预测 proposals，(B*npoint, 7)
        # 预测的是 bin-based，转换为相机坐标系的 x y z h w l ry
        proposals = decode_bbox_target(xyz.view(-1, 3), rpn_reg.view(-1, rpn_reg.shape[-1]),
                                       anchor_size=self.MEAN_SIZE,
                                       loc_scope=cfg.RPN.LOC_SCOPE,
                                       loc_bin_size=cfg.RPN.LOC_BIN_SIZE,
                                       num_head_bin=cfg.RPN.NUM_HEAD_BIN,
                                       get_xz_fine=cfg.RPN.LOC_XZ_FINE,
                                       get_y_by_bin=False,
                                       get_ry_fine=False)
        # 将体中心 Y 坐标变为底部面中心 Y 坐标，由于 Y 轴方向向下，所以 Y 坐标增大
        proposals[:, 1] += proposals[:, 3] / 2
        proposals = proposals.view(batch_size, -1, 7)   # (B, N, 7)

        scores = rpn_scores     # 点云的二分类得分，(B, N)
        _, sorted_idxs = torch.sort(scores, dim=1, descending=True) # 得分排序  (B, N)

        # (B, 512, 7)，NMS 之后选 top 512 个 proposals，以及相应的得分
        ret_bbox3d = scores.new(batch_size, cfg[self.mode].RPN_POST_NMS_TOP_N, 7).zero_()
        ret_scores = scores.new(batch_size, cfg[self.mode].RPN_POST_NMS_TOP_N).zero_()
        for k in range(batch_size):
            scores_single = scores[k]           # (N,)
            proposals_single = proposals[k]     # (N, 7)
            order_single = sorted_idxs[k]       # (N,)，N 个点的得分排序

            if cfg.TEST.RPN_DISTANCE_BASED_PROPOSE: # 基于距离的 NMS
                # (n1+n2,)  (n1+n2, 7) 
                # top n 的点的二分类得分，以及基于各个点 local region 对 box3d 的预测 (x,y,z,h,w,l,ry)
                scores_single, proposals_single = self.distance_based_proposal(scores_single, 
                                                                               proposals_single, order_single)
            else:
                pass
            proposals_tot = proposals_single.size(0)
            ret_bbox3d[k, :proposals_tot] = proposals_single
            ret_scores[k, :proposals_tot] = scores_single
    return ret_bbox3d, ret_scores 
```


其中 `decode_bbox_target` 函数是根据 RPN 的输出恢复出各个 point 对物体的预测（预测物体的 xyzhwl,ry 7个数值)，函数定义为，

```python
def decode_bbox_target(roi_box3d, pred_reg, loc_scope, loc_bin_size, num_head_bin, anchor_size,
                       get_xz_fine=True, get_y_by_bin=False, loc_y_scope=0.5, loc_y_bin_size=0.25,
                       get_ry_fine=False):
    '''
    roi_box3d: (B * npoints, 3) 模型的原始输入点云
    pred_reg: (B * npoints, 76)
    loc_scope: 3m， RPN 的 X Z 轴的搜索半径
    loc_bin_size: 0.5m，RPN 的 X Z 轴 bin size
    num_head_bin: 朝向角的 bin 数量
    anchor_size: 长度为 3 的列表，表示数据集中同一类的 box3d 的 h w l
    get_y_by_bin: bool 类型，表示是否对 Y 坐标使用 bin-based 预测
    '''
    anchor_size = anchor_size.to(roi_box3d.get_device())    # 训练集中同一分类的 box3d, hwl 的平均值
    per_loc_bin_num = int(loc_scope / loc_bin_size) * 2     # RPN 的 X, Z  bin-based, 12 个 bin
    loc_y_bin_num = int(loc_y_scope / loc_y_bin_size) * 2   # Y bin-based， 4 个 bin

    x_bin_l, x_bin_r = 0, per_loc_bin_num
    z_bin_l, z_bin_r = per_loc_bin_num, per_loc_bin_num * 2
    start_offset = z_bin_r

    x_bin = torch.argmax(pred_reg[:, x_bin_l:x_bin_r], dim=-1)      # (B*npoints,) 预测最可能的 x bin index
    z_bin = torch.argmax(pred_reg[:, z_bin_l:x_bin_r], dim=-1)      # (B*npoints,) 预测最可能的 z bin index

    pos_x = x_bin.float() * loc_bin_size + loc_bin_size / 2 - loc_scope # 这个最可能的 bin 中心X坐标 - 当前点X坐标
    pos_y = z_bin.float() * loc_bin_size + loc_bin_size / 2 - loc_scope # 这个最可能的 bin 中心X坐标 - 当前点X坐标

    if get_xz_fine:     # 使用 residual
        x_res_l, x_res_r = per_loc_bin_num * 2, per_loc_bin_num * 3
        y_res_l, y_res_r = per_loc_bin_num * 3, per_loc_bin_num * 4
        start_offset = z_res_r

        # 根据 bin index，获取相应的 bin residual
        x_res_norm = torch.gather(pred_reg[:, x_res_l:x_res_r], dim=1, index=x_bin.unsqueeze(dim=1)).squeeze(dim=1)
        z_res_norm = torch.gather(pred_reg[:, z_res_l:z_res_r], dim=1, index=x_bin.unsqueeze(dim=1)).squeeze(dim=1)
        x_res = x_res_norm * loc_bin_size   # 预测的是归一化值 [-1/2，1/2）， x bin size，得到实际的 residual
        z_res = z_res_norm * loc_bin_size

        pos_x += x_res      # 预测物体中心点X坐标 - 当前点X坐标。
        pos_z += z_res      # 预测物体中心点Y坐标 - 当前点Y坐标。参见 PointRCNN 论文解读（一） 一文中的图2
    
    if get_y_by_bin:    # 不使用 Y bin-based
        ...
    else:               # 直接使用 Y residual，即 center_y - point_y
        y_offset_l, y_offset_r = start_offset, start_offset+1
        start_offset = y_offset_r

        pos_y = roi_box3d[:,1] + pred_reg[:,y_offset_l]     # center_y，中心点Y坐标预测
    
    # rotation_y  bin-based
    ry_bin_l, ry_bin_r = start_offset, start_offset + num_head_bin
    ry_res_l, ry_res_r = ry_bin_r, ry_bin_r + num_head_bin

    ry_bin = torch.argmax(pred_reg[:, ry_bin_l:ry_bin_r], dim=1)    # 对 ry bin index 的预测
    # 
    ry_res_norm = torch.gather(pred_reg[:, ry_res_l:ry_res_r], dim=1, index=ry_bin.unsqueeze(dim=1)).squeeze(dim=1)
    if get_ry_fine:     # False, RPN 角度预测不精调，将 [-pi, pi] 划分为 n 个 bin
        ...             # RCNN 精调角度预测，将 [-pi/4, pi/4] 范围划分为 n 个 bin
    else:
        angle_per_class = (2 * np.pi) / num_head_bin    # 每个 bin 张开角度
        ry_res = ry_res_norm * (angle_per_class / 2)    # theta 角度预测是归一化到 [-1, 1]，所以需要先scale 到[-0.5, 0.5]
                                                        # 然后再 scale 到实际角度，* delta
        # 参见 PointRCNN 论文解读（一）一文的 (13) 式
        ry = (ry_bin.float() * angle_per_class + ry_res) % (2 * np.pi)
        ry[ry > np.pi] -= 2*np.pi   # 将 [pi, 2pi] 范围内的角转为 [-pi, 0]
    size_res_l, size_res_r = ry_res_r, ry_res_r + 3     # hwl 的回归预测
    assert size_res_r == pred_reg.shape[1]
    size_res_norm = pred_reg[:size_res_l:size_res:r]    # 是针对同类size均值的归一化
    hwl = size_res_norm * anchor_size + anchor_size

    roi_center = roi_box3d[:, 0:3]  # 每个点均独立进行一个 box3d 的预测
    # (B*npoints, 7)
    shift_ret_box3d = torch.cat((pos_x.view(-1, 1), pos_y.view(-1, 1), pos_z.view(-1, 1), hwl, ry.view(-1, 1)), dim=1)
    ret_box3d = shift_ret_box3d
    if roi_box3d.shape[1] == 7:
        ...     # 第二维度为 3
    ret_box3d[:, [0, 2]] += roi_center[:, [0, 2]]       # 加上当前点坐标，得到预测物体中心点 X,Z 坐标
    return ret_box3d    # (B*npoint, 7)，预测的物体中心点坐标，预测物体 hwl 和 ry
```

点云中每个点均进行物体 box3d 的预测，显然这会生成大量 proposals，类似 2d 目标检测中那样，使用 NMS 方法降低 proposals 数量，

```python
def distance_based_proposal(self, scores, proposals, order):
    '''
    scores: (N,) 当前样本中各点的二分类得分，未排序
    proposals: (N, 7) 各点对物体的 box3d 预测，未排序
    order: (N,) 各点二分类得分降序排列 index
    '''
    nms_range_list = [0, 40.0, 80.0]
    pre_tot_top_n = cfg[self.mode].RPN_PRE_NMS_TOP_N    # 9000
    pre_top_n_list = [0, int(pre_tot_top_n * 0.7), pre_tot_top_n - int(pre_tot_top_n * 0.7)]
    post_tot_top_n = cfg[self.mode].RPN_POST_NMS_TOP_N  # 512
    post_top_n_list = [0, int(post_tot_top_n * 0.7), post_tot_top_n - int(post_tot_top_n * 0.7)]

    # 分两个（Z坐标）范围段 0~40， 40~80，分别称为 range1 和 range2
    # range1 内取 top 6300 个点，range2 内取 top 2700 个点（由于range2较远，如果range2内无任何点，那么到range1内取top2700个点）
    # 将 range 内所取点的预测 box3d 转为 bev
    scores_single_list, proposals_single_list = [], []

    scores_ordered = scores[order]
    proposals_ordered = proposals[order]

    dist = proposals_order[:, 2]    # Z 坐标，(N,)
    first_mark = (dist > nms_range_list[0]) & (dist <= nms_range_list[1])   # Z坐标40米之内的点 index
    for i in range(1, len(nms_range_list)): # [1, 2]
        dist_mark = ((dist > nms_range_list[i-1]) & (dist <= nms_range_list[i]))
        if dist_mark.sum() != 0:
            cur_scores = scores_ordered[dist_mask]      # 满足 Z 坐标范围的点
            cur_proposals = proposals_ordered[dist_mask]

            cur_scores = cur_scores[:pre_top_n_list[i]]
            cur_proposals = cur_proposals[:pre_top_n_List[i]]
        else:
            assert i == 2, '%d' % i
            cur_scores = scores_ordered[first_mask]
            cur_proposals = proposals_ordered[first_mask]

            cur_scores = cur_scores[pre_top_n_list[i-1]: pre_top_n_list[i]]
            cur_proposals = cur_proposals[pre_top_n_list[i-1]: pre_top_n_list[i]]
        
        # box3d 转鸟瞰图，得到鸟瞰图的 (x1y1x2y2, ry)，注意这个bev坐标值是相机坐标系逆时针旋转ry所得
        boxes_bev = kitti_utils.boxes3d_to_bev_torch(cur_proposals)
        if cfg.RPN.NMS_TYPE == 'rotate':
            ...
        elif cfg.RPN.NMS_TYPE == 'normal':  # RPN_NMS_THRESH: 0.85。执行 NMS，得到需要保留的点 index
            keep_idx = iou3d_utils.nms_normal_gpu(boxes_bev, cur_scores, cfg[self.mode].RPN_NMS_THRESH)
        else:
            raise NotImplementedError
        keep_idx = keep_idx[:post_top_n_list[i]]    # NMS 之后，再取 top n 的点 index

        scores_single_list.append(cur_scores[keep_idx]) # NMS 之后 top n 的点的得分
        proposals_single_list.append(cur_proposals[keep_idx])   # NMS 之后 top n 的点得分

    scores_single = torch.cat(scores_single_list, dim=0)    # [n1, n2] -> (n1+n2,)
    proposals_single = torch.cat(proposals_single_list, dim=0)  # (n1+n2, 7)， n1+n2 <= RPN_POST_NMS_TOP_N
    return scores_single, proposals_single
```

上述代码中，根据 Z 坐标的远近分成两个范围，这两个范围内分别执行：

1. 根据得分取 top pre_n 的点
2. 执行 NMS
3. 取 top post_n 的点

两个范围内的点的 top post_n 的点再 concatenate，得到 pre_n + post_n 个点。第 2 步中 NMS 方法为 `nms_normal_gpu` ，其函数内部又调用了 `iou3d_cuda.nms_normal_gpu` 方法，

```c++
int nms_normal_gpu(at::Tensor boxes, at::Tensor keep, float nms_overlap_thresh) {
    // 按得分降序排列的 bev boxes: (N, 5) -> (x1,y1,x2,y2,ry)
    // keep: NMS 之后，需要保留的 index
    // nms_overlap_thresh: train->0.85  test->0.8
    int boxes_num = boxes.size(0)   
    const float *boxes_data = boxes.data<float>();  
    long *keep_data = keep.data<long>(); // 记录被保留的 box 在原 boxes 数组中的下标

    // ceil(N / 64) 。blocks 数量
    const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);

    unsigned long long *mask_data = NULL;   // mask_data: 长度为 N x M 的向量，参见下方原理说明
    CHECK_ERROR(cudaMalloc((void**)&mask_data, boxes_num * col_blocks * sizeof(unsigned long long)));
    nmsNormalLaucher(boxes_data, mask_data, boxes_num, nms_overlap_thresh); // 使用 GPU 并行执行 NMS
                                                                            // 代码和原理见下文
    std::vector<unsigned long long> mask_cpu(boxes_num * col_blocks);
    CHECK_ERROR(cudaMemcpy(&mask_cpu[0], mask_data, boxes_num * col_blocks * sizeof(unsigned long long),
                           cudaMemcpyDeviceToHost));
    cudaFree(mask_data);

    unsigned long long remv_cpu[col_blocks];    
    // M 长度的 vector，第 i 个元素的前 THREADS_PER_BLOCK_NMS 个 bit 表示第 i 组 boxes 是否被抑制
    memset(remv_cpu, 0, col_blocks * sizeof(unsigned long long));

    int num_to_keep = 0; // 记录被保留的 box 数量

    for (int i = 0; i < boxes_num; i++) {
        int nblock = i / THREADS_PER_BLOCK_NMS; // 这个 box 所在分组 index
        int inblock = i % THREADS_PER_BLOCK_NMS;// 组内下标

        // 求分组 index 所在位置的向量元素的第 inblock bit 值(0 or 1)
        if (!(remv_cpu[nblock] & （1ULL << inblock)) {// 如果为 0，即未被抑制
            keep_data[num_to_keep++] = i;
            // 这个未被抑制的 box，记为 box a，获取被 box a 所抑制的 box index 信息
            // mask 中对每个 box 均计算了 col_blocks 个 group，指针移到这 col_blocks 个组
            //      的掩码信息起始处 (p 处)，考虑 p 处开始的 col_blocks 个组被 box a 抑制情况
            unsigned long long *p = &mask_cpu[0] + i * col_blocks;
            // box a 所在分组为 nblocks，那么 col_blocks 个组中前 (nblocks-1) 个分组的掩码信息
            //      没必要再考虑，因为对称性。相当于只考虑掩码矩阵的上三角信息
            for (int j = nblocks; j < col_blocks; j++) {
                remv_cpu[j] |= p[j]     // 或运算，只要被某个box抑制，那么这组就被抑制
            }
        }
    }
    if ( cudaSuccess != cudaGetLastError() ) printf( "Error!\n" );
    return num_to_keep;
}

void nmsNormalLauncher(const float *boxes, unsigned long long *mask, int boxes_num, float nms_overlap_thresh) {
    dim3 blocks(DIVUP(boxes_num, THREADS_PER_BLOCK_NMS),
                DIVUP(boxes_num, THREADS_PER_BLOCK_NMS));   // 分配 ceil(N/64) * ceil(N/64) 个 blocks
    dim3 threads(THREADS_PER_BLOCK_NMS);    // 一个 block 有 64 个线程
    nms_normal_kernel<<blocks, threads>>(boxes_num, nms_overlap_thread, boxes, mask);
}

// GPU 并行计算 nms 的代码，原理见下文说明。
__global__ void nms_normal_kernel(const int boxes_num, const float nms_overlap_thread,
                                  const float *boxes, unsigned long long *mask) {
    // NMS 的输入 boxes: (N, 5) (x1,y1,x2,y2,ry)
    // mask: (N, ceil(N/64) )  
    // THREADS_PER_BLOCK_NMS = 64
    const int row_start = blockIdx.y    // block 的 x, y 坐标。block 是二维 grid
    const int col_start = blockIdx.x

    // 行方向上 block 有效大小。最后一个 block 可能不足 64
    const int row_size = fminf(boxes_num - row_start * THREADS_PER_BLOCK_NMS, THREADS_PER_BLOCK_NMS);
    // 列方向上 block 有效大小。最后一个 block 可能不足 64
    const int col_size = fminf(boxes_num - col_start * THREADS_PER_BLOCK_NMS, THREADS_PER_BLOCK_NMS);

    __shared__ float block_boxes[THREADS_PER_BLOCK_NMS * 5];    // 一个 block 内各线程共享的内存块

    if (threadIdx.x < col_size) {   // 如果条件不满足，那么实际上此线性不需要干活
        // 当前线性填充一个 box
        // 每一列对应一个 box，所有列对应所有box
        // 每一行都进行相同的填充
        // 水平方向看，某个 box 对应某列 blocks 内的第几个 thread
        block_boxes[threadIdx.x * 5 + 0] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 5 + 0];
        block_boxes[threadIdx.x * 5 + 1] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 5 + 1];
        block_boxes[threadIdx.x * 5 + 2] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 5 + 2];
        block_boxes[threadIdx.x * 5 + 3] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 5 + 3];
        block_boxes[threadIdx.x * 5 + 4] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 5 + 4];
    }
    __syncthreads();        // 等待 block 内所有线程填充完毕

    if (threadIdx.x < row_size) { // 如果条件不满足，那么此线性不需要干活
        // 计算 子 IOU 的某一行，行的 global index 为 cur_box_idx，
        const int cur_box_idx = THREADS_PER_BLOCK_NMS * row_start + threadIdx.x;
        const float *cur_box = boxes + cur_box_idx * 5;

        int i = 0;
        unsigned long long t = 0;
        int start = 0;
        if (row_start == col_start) {   // 子矩阵的 行号和列号相等，那么计算上三角矩阵元素（去掉主对角线）
            start = threadIdx.x + 1;    // 去掉主对角线，因为主对角线元素为最大 IOU 值 1
        }

        for (i = start; i < col_size; i++) {    // 计算子矩阵中，上三角中的某行，行号为 threadIdx.x
            // 列号为 i，如果 IOU > thresh，那么列号所表示的 box 应当被抑制
            if (iou_normal(cur_box, block_boxes + i * 5) > nms_overlap_thresh) {
                t |= 1ULL << i; // 标记子矩阵中，第 i+1 位 为 1（表示被抑制）
            }
        }
        const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);
        mask[cur_box_idx * col_blocks + col_start] = t;
    }
}
```

要完全理解以上代码，我们还需要搞清楚并行计算 IOU 的原理。

通常而言，NMS 一组 boxes 的步骤为：

1. boxes 按得分降序排列。记 boxes 为 N。
2. 计算两两 box 之间的 IOU，得到一个 $N \times N$ 的 IOU 矩阵。
3. IOU > thread 的那个较大 index 的 box 被抑制

记 boxes 数量为 N，使用 GPU 计算，将 boxes 进行分组，每组 p 个 boxes，那么一共有 $M=\lceil N / p \rceil$ 组，一个组称为一个 block，于是一个 block 内最多 p 个 boxes，因为最后一个 block 可能不足 p 个 boxes。以 block 为基本单位（类比上面计算 NMS 三个步骤 中的 box），也就是说，两两 block 之间计算 IOU，最终就能得到所有 boxes 的 IOU。创建一个 $M \times M$ 的 block 矩阵，这个矩阵内的某个 block 的 index 使用行列坐标 $(r, c)$ 表示，其中 $1 \le r,c \le M$，从行和列角度看，分别对应 boxes 的 index 范围为

$$\mathbb R_r=[p_r * (r-1)+1, p_r * r], \quad \mathbb R_c=[p_c * (c-1)+1, p_c * c]$$

其中 $p_c$ 定义如下（$p_r$ 定义类似，只要将条件中的 c 改为 r 即可），

$$p_c=\begin{cases} p & c < M \ \text{or} \ N \ \text{mod} \ p=0 \\ N \ \text{mod} \ p & \text{o.w.} \end{cases}$$

即 $|\mathbb R_r| = p_r, \ |\mathbb R_c| = p_c$ 。


于是 block $(r,c)$ 的任务就是计算 $\mathbb R_r$ 与 $\mathbb R_c$ boxes 之间的 IOU。这个思想本质是将最终的 $N \times N$ IOU 矩阵划分为 $M \times M$ 个子矩阵，每个 IOU 子矩阵由一个 block 负责计算。


进一步看单个 block 内部如何计算。对于 block $(r, c)$，我们使用 p 个线程，每个线程计算 $p_r \times p_c$ IOU 子矩阵中的一行。

特殊地，如果 $r=c$，那么 $\mathbb R_r = \mathbb R_c$，即同一小组 boxes 内的两两 box 计算 IOU，即这个子 IOU 矩阵是对称的，此时我们可以计算这个子 IOU 矩阵的上三角元素，甚至不用计算主对角线元素（因为主对角线元素为最大值 1）。

对每个计算出的 IOU，判断其是否大于阈值 `nms_overlap_thresh`，如是，那么对应位置标记为 1，表示被抑制。

block $(r, c)$ 中某个线程其线程 id 记为 $i \in [1, p_r]$，这个线程需要计算子 IOU 中第 $i$ 行共 $p_c$ 个 IOU，当然如果 $r=c$，那么只需要计算 $p_c-i$ 个 IOU（上三角除主对角线的第 i 行元素数量）。

使用 `mask` 作为全局变量记录所有将被抑制的 box index，`mask` 是长度为 $N \times M$ 的向量，每个向量元素为 `unsigned long long` 类型的变量，记为 `t`，记录某个子矩阵中某行中被抑制的 box index，`t` 的从低到高的前 $p_c$ 个 bit 记录了掩码信息，所以实际上 $N \times M$ 大小的 `mask` 记录了 $N \times N$ 的掩码信息。

代码中 NMS 类型有 `normal` 和 `rotate` 两种，我个人觉得使用 `rotate` 才合理，因为这种计算方式考虑了 BEV box 的偏向角，但是代码配置中默认使用 `normal` ，可能是因为计算量大大减少，且准确率下降并不大，综合效果更好的原因吧，没有实际比较过，这里存疑。


以上，得到 rois 之后，就打包 RCNN 网络的输入，参见以上 model forward 方法代码中的变量 `rcnn_input_info` 。

## 1.3 RCNNNet

```python
class RCNNNet(nn.Module):
    def __init__(self, num_classes, input_channels=0, use_xyz=True):
        '''
        num_classes: 分类数量。这里以 Car 分类为例说明。虽然指定了 Car 分类，但
                    实际上还包括 Background。具体参见 KittiRCNNDataset 类
        input_channels: 128。 RPN 中 backbone 的输出特征 channel
        '''
        # 由于代码示例使用了 Car 分类，故 RCNN 中只有 bg 和 Car 二分类
        # RCNN 子网络中，输入首先经过整合，然后是 Pointnet Set abstraction 下采样和 MLP
        # 最后分别喂给 cls head 和 reg head
        # cls head 是三个 conv1d，输出 channel 为 [512, 256, 256, 1]
        cls_channel = 1 if num_classes == 2 else num_classes    # 二分类使用一个分类得分

        # reg 预测也是 bin-based，但是 bin 搜索范围比 RPN 的小
        per_loc_bin_num = len(cfg.RCNN.LOC_SCOPE / cfg.RCNN.LOC_BIN_SIZE) * 2
        loc_y_bin_num = len(cfg.RCNN.LOC_Y_SCOPE / cfg.RCNN.LOC_Y_BIN_SIZE) * 2
        # 1. x,z 的 bin index 和 bin residual
        # 2. ry 角度的 bin index 和 bin residual
        # 3. h w l ，直接回归预测
        reg_channel = per_loc_bin_num * 4 + cfg.RCNN.NUM_HEAD_BIN * 2 + 3
        # 4. y 的 bin index 和 bin residual（若 y 不是 bin-based，那么直接回归预测 y）
        reg_channel += (1 if not cfg.RCNN.LOC_Y_BY_BIN else loc_y_bin_num * 2)
        # 其他 layer 的构造代码略、
        ...

    def forward(self, input_data):
        if cfg.RCNN.ROI_SAMPLE_JIT:     # True 
            if self.training:
                with torch.no_grad():
                    target_dict = self.proposal_target_layer(input_data)
                pts_input = torch.cat((target_dict['sampled_pts'], target_dict['pts_feature']), dim=2)
                target_dict['pts_input'] = pts_input
            else:
                pass
        else:
            pass
        
        # 划分为点云坐标，以及反射强度等组成的特征（配置中不使用反射强度，故 features 为 None）
        xyz, features = self._break_up_pc(pts_input)    
        if cfg.RCNN.USE_RPN_FEATURES:   # True
            xyz_input = pts_input[..., 0:self.rcnn_input_channel]
```

根据 RPN 输出得到 rois 之后，使用 `proposal_target_layer` 得到 target proposals，逻辑如下，

```python
# class ProposalTargetLayer

def forward(self, input_data):
    '''input_data: 输入数据字典，参见 1.2 一节代码中的 rcnn_input_info 变量'''
    # NMS 后 top n 的点，每个点根据 local region 预测的 box3d   (B, n, 7)，n<= POST_NMS_TOP_N
    # gt boxes3d    (B, mn, 7)  mn 是某样本中具有最多 gt boxes3d 的数量
    roi_boxes3d, gt_boxes3d = input_dict['roi_boxes3d'], input_dict['gt_boxes3d']
    # (B, ROI_PER_IMAGE, 7)
    # (B, ROI_PER_IMAGE, 7)
    # (B, ROI_PER_IMAGE)
    batch_rois, batch_gt_of_rois, batch_roi_iou = self.sample_rois_for_rcnn(roi_boxes3d, gt_boxes3d)
    # RPN 的输入点云坐标 (B, npoints, 3)
    # RPN backbone 输出点云特征 (B, npoints, 128)
    rpn_xyz, rpn_features = input_dict['rpn_xyz'], input_dict['rpn_features']
    
    pts_extra_input_list = [input_dict['seg_mask'].unsqueeze(dim=2)]    # (B, npoints, 1)
    pts_depth = input_dict['pts_depth'] / 70.0 - 0.5    # sqrt(x2+y2+z2)
    pts_extra_input_list.append(pts_depth.unsqueeze(dim=2)) # (B, npoints, 1)
    pts_extra_input = torch.cat(pts_extra_input_list, dim=2)# (B, npoints, 2)
    pts_feature = torch.cat((pts_extra_input, rpn_features), dim=2) # (B, npoints, 2+128)
    # 在 box3d 内的所有点中取样 sampled_pt_num 个点，
    #   每个点包含 3 个坐标，预测二分类 seg_mask，depth 以及 backbone 输出特征 128
    #   故输出 pooled 特征 pooled_features: (B, boxes_num, sampled_pt_num, 3+2+128)
    #   pooled_empty_flag: (B, boxes_num)， 值为 1 表示这个预测 roi 内部没有点。默认为 0
    pooled_features, pooled_empty_flag = roipool3d_utils.roipool3d_gpu(rpn_xyz, pts_feature, batch_rois,
        cfg.RCNN.POOL_EXTRA_WIDTH, sampled_pt_num=cfg.RCNN.NUM_POINTS)  # 1.0,  512
    # (B, boxes_num, sampled_pt_num, 3)
    # (B, boxes_num, sampled_pt_num, 2+128)
    sampled_pts, sampled_features = pooled_features[:,:,:,0:3], pooled_features[:,:,:,3:]
    if cfg.AUG_DATA:
        sampled_pts, batch_rois, batch_gt_of_rois = self.data_augmentation(sampled_pts, batch_rois, batch_gt_of_rois)
    batch_size = batch_rois.shape[0]
    roi_ry = batch_rois[:,:,6] % (2 * np.pi)
    roi_center = batch_rois[:,:,0:3]
    sampled_pts = sampled_pts - roi_center.unsqueeze(dim=2) # 各点与中心点的坐标差.相当于坐标系原点平移到物体中心，新坐标系下的点坐标
    batch_gt_of_rois[:,:,0:3] = batch_gt_of_rois[:,:,0:3] - roi_center
    batch_gt_of_rois[:,:,6] = batch_gt_of_rois[:,:,6] - roi_ry
    for k in range(batch_size):
        # 以物体朝向为 Z 轴，旋转坐标系，得到新坐标系下的点坐标
        sampled_pts[k] = kitti_utils.rotate_pc_along_y_torch(sampled_pts[k], batch_rois[k,:,6])
        # gt 的中心点坐标也旋转为新坐标系下的坐标
        batch_gt_of_rois[k] = kitti_utils.rotate_pc_along_y_torch(batch_gt_of_rois[k].unsqueeze(dim=1), roi_ry[k].squeeze(dim=1))
    
    valid_mask = (pooled_empty_flag == 0)   # roi 内部有 points
    reg_valid_mask = ((batch_roi_iou > cfg.RCNN.REG_FG_THRESH) & valid_mask).long() # IoU > 回归前景阈值
    batch_cls_label = (batch_roi_iou > cfg.RCNN.CLS_FG_THRESH).long()               # IoU > 分类前景阈值
    invalid_mask = (batch_roi_iou > cfg.RCNN.CLS_BG_THRESH) & (batch_roi_iou < cfg.RCNN.CLS_FG_THRESH)
    batch_cls_label[valid_mask == 0] = -1
    batch_cls_label[invalid_mask > 0] = -1
    output_dict = {'sampled_pts': sampled_pts.view(-1, cfg.RCNN.NUM_POINTS, 3),
                   'pts_feature': sampled_features.view(-1, cfg.RCNN.NUM_POINTS, sampled_features.shape[3]),
                   'cls_label': batch_cls_label.view(-1),
                   'reg_valid_mask': reg_valid_mask.view(-1),
                   'gt_of_rois': batch_gt_of_rois.view(-1, 7),
                   'gt_iou': batch_roi_iou.view(-1),
                   'roi_boxes3d': batch_rois.view(-1, 7)}
    return output_dict


def sample_rois_for_rcnn(self, roi_boxes3d, gt_boxes3d):
    '''
    按一定的比例，对每个样本，采样一定的 fg rois 和 bg rois
    每个 roi，均有对应的一个 IoU，以及相关的 gt box
    '''
    batch_size = roi_boxes3d.size(0)
    # RPN 中，每个样本中取 top n 个 预测 rois。然后我们在其中取 0.5 比例的正例 rois
    # 单个样本取 64（配置）个 rois，其中包含 32 （一半） 的正例 rois
    fg_rois_per_image = int(np.round(cfg.RCNN.FG_RATIO * cfg.RCNN.ROI_PER_IMAGE))

    batch_rois = gt_boxes3d.new(batch_size, cfg.RCNN.ROI_PER_IMAGE, 7).zero_()
    batch_gt_of_rois = gt_boxes3d.new(batch_size, cfg.RCNN.ROI_PER_IMAGE, 7).zero_()
    batch_roi_iou = gt_boxes3d.new(batch_size, cfg.RCNN.ROI_PER_IMAGE).zero_()

    for idx in range(batch_size):
        cur_roi, cur_gt = roi_boxes3d[idx], gt_boxes3d[idx]
        k = len(cur_gt) - 1
        while cur_gt[k].sum() == 0:
            k -= 1      # 定位当前样本的最后一个 gt boxes 的 index，因为不足 mn 的，进行了 tail padding
        
        cur_gt = cur_gt[:k+1]

        iou3d = iou3d_utils.boxes_iou3d_gpu(cur_roi, cur_gt[:, 0:7])    # (n, m)
        max_overlaps, gt_assignment = torch.max(iou3d, dim=1)   # (n,)

        fg_thresh = min(cfg.RCNN.REG_FG_THRESH, cfg.RCNN.CLS_FG_THRESH) # min(0.55, 0.6)
        fg_inds = torch.nonzero((max_overlaps >= fg_thresh)).view(-1)   # (n1,) fg rois 的 index

        easy_bg_inds = torch.nonzero((max_overlap < cfg.RCNN.CLS_BG_THRESH_LO)).view(-1)    # (n2,)
        hard_bg_inds = torch.nonzero((max_overlap < cfg.RCNN.CLS_BG_THRESH) &
                                     (max_overlap >= cfg.RCNN.CLS_BG_THRESH_LO)).view(-1)   # (n3,)
        fg_num_rois = fg_inds.numel()       # fg rois 数量
        bg_num_rois = hard_bg_inds.numel() + easy_bg_inds.numel()

        if fg_num_rois > 0 and bg_num_rois > 0:
            fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)
            rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(gt_boxes3d).long()
            fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]    # 随机取一定数量的 fg rois

            bg_rois_per_this_image = cfg.RCNN.ROI_PER_IMAGE - fg_rois_per_this_image
            # 选择一定比例的 hard bg，剩余的选择 easy bg
            bg_inds = self.sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image)
        elif fg_num_rois > 0 and bg_num_rois == 0:
            rand_num = np.floor(np.random.rand(cfg.RCNN.ROI_PER_IMAGE) * fg_num_rois)
            # [0, fg_num_rois) 均匀随机抽取 ROI_PER_IMAGE 个 index
            rand_num = torch.from_numpy(rand_num).type_as(gt_boxes3d).long()
            fg_inds = fg_inds[rand_num]     # 随机抽取 ROI_PER_IMAGE 个 fg
            fg_rois_per_this_image = cfg.RCNN.ROI_PER_IMAGE
            bg_rois_per_this_image = 0
        elif bg_num_rois > 0 and fg_num_rois == 0:
            bg_rois_per_this_image = cfg.RCNN.ROI_PER_IMAGE
            bg_inds = self.sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image)
            fg_rois_per_this_image = 0
        else:
            raise NotImplementedError

        roi_list, roi_iou_list, roi_gt_list = [], [], []
        if fg_rois_per_this_image > 0:
            fg_rois_src = cur_roi[fg_inds]          # 正例 rois
            gt_of_fg_rois = cur_gt[gt_assignment[fg_inds]]  # 正例 rois 对应的 gt box
            iou3d_src = max_overlaps[fg_inds]       # 正例 rois 的 IoU
            # 对正例做随机增强
            fg_rois, fg_iou3d = self.aug_roi_by_noise_torch(fg_rois_src, gt_of_fg_rois, iou3d_src,
                                                            aug_times=cfg.RCNN.ROI_FG_AUG_TIMES)# 10 
            roi_list.append(fg_rois)
            roi_iou_list.append(fg_iou3d)
            roi_gt_list.append(gt_of_fg_rois)
        if bg_rois_per_this_image > 0:
            bg_rois_src = cur_roi[bg_inds]
            gt_of_bg_rois = cur_gt[gt_assignment[bg_inds]]
            iou3d_src = max_overlaps[bg_inds]
            aug_times = 1 if cfg.RCNN.ROI_FG_AUG_TIMES > 0 else 0
            bg_rois, bg_iou3d = self.aug_roi_by_noise_torch(bg_rois_src, gt_of_bg_rois, iou3d_src,
                                                            aug_times=aug_times)
            roi_list.append(bg_rois)
            roi_iou_list.append(bg_iou3d)
            roi_gt_list.append(gt_of_bg_rois)
        rois = torch.cat(roi_list, dim=0)   # (ROI_PER_IMAGE, 7)
        iou_of_rois = torch.cat(roi_iou_list, dim=0)    # (ROI_PER_IMAGE,)
        gt_of_rois = torch.cat(roi_gt_list, dim=0)      # (ROI_PER_IMAGE, 7)

        batch_rois[idx] = rois
        batch_gt_of_rois[idx] = gt_of_rois
        batch_roi_iou[idx] = iou_of_rois
    return batch_rois, batch_gt_of_rois, batch_roi_iou
```

```python
# utils/roipool3d/roipool3d_utils.py
def roipool3d_gpu(pts, pts_feature, boxes3d, pool_extra_width, sampled_pt_num=512):
    '''
    pts: 点云坐标 (B, npoints, 3)
    pts_feature: 点云特征 (B, npoints, 2+128)，点云经 Backbone 的输出特征+预测二分类seg_mask+depth
    boxes3d: (B, ROI_PER_IMAGE, 7)，预测 rois
    pool_extra_width: 1.0 m，将 rois 稍微扩大，以包含 box 边缘的点
    '''
    # boxes_num: ROI_PER_IMAGE
    batch_size, boxes_num, feature_len = pts.shape[0], boxes3d.shape[1], pts_feature.shape[2]
    pooled_boxes3d = kitti_utils.enlarge_boxes3d(boxes3d.view(-1, 7), pool_extra_width).view(batch_size, -1, 7)

    # 批样本数量，每个样本中 roi 数量，每个 roi 中点数，每个点的特征维度
    pooled_features = torch.cuda.FloatTensor((batch_size, boxes_num, sampled_pt_num, 3 + feature_len)).zero_()
    pooled_empty_flag = torch.cuda.FloatTensor((batch_size, boxes_num)).zero_()
    roipool3d_cuda.forward(pts.contiguous(), pooled_boxes3d.contiguous(), pts_feature.contiguous(), 
                           pool_features, pooled_empty_flag)
    return pooled_features, pooled_empty_flag

# roipool3d_cuda.forward 绑定了 c++ 的 roipool3d_gpu 这个方法，其内部调用了 roipool3dLauncher 
```

```c++
void roipool3dLauncher(int batch_size, int pts_num, int boxes_num, int feature_in_len, int sampled_pts_num,
    const float *xyz, const float *boxes3d, const float *pts_feature, float *pooled_features, int *pooled_empty_flat) {
    int *pts_assign = NULL;
    // 存储每个 point，对所有 boxes 是否匹配(point 在 box 内则匹配)。与某个 box 匹配，那么 flag=1，否则 flag=0
    cudaMalloc(&pts_assign, batch_size * pts_num * boxes_num * sizeof(int));

    dim3 blocks(DIVUP(pts_num, THREADS_PER_BLOCK), boxes_num, batch_size);
    dim3 threads(THREADS_PER_BLOCK);

    assign_pts_to_box3d<<blocks, threads>>(batch_size, pts_num, boxes_num, xyz, boxes3d, pts_assign);

    int *pts_idx = NULL;
    // 每个 box 取 sampled_pts_num 个点
    cudaMalloc(&pts_idx, batch_size * boxes_num * sampled_pts_num * sizeof(int));

    dim3 blocks2(DIVUP(boxes_num, THREADS_PER_BLOCK), batch_size);
    get_pooled_idx<<<blocks2, threads>>>(batch_size, pts_num, boxes_num, sampled_pts_num, pts_assign, pts_idx, pooled_empty_flag);

    dim3 blocks_pool(DIVUP(sampled_pts_num, THREADS_PER_BLOCK), boxes_num, batch_size);
    roipool3d_forward<<<blocks_pool, threads>>>(batch_size, pts_num, boxes_num, feature_in_len, sampled_pts_num,
        xyz, pts_idx, pts_feature, pooled_features, pooled_empty_flag);
}

__global__ void assign_pts_to_box3d(int batch_size, int pts_num, int boxes_num, const float *xyz, const float *boxes3d, int *pts_assign) {
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;     // point index
    int box_idx = blockIdx.y;                               // box index
    int bs_idx = blockIdx.z;                                // inner-batch index

    // 每个线程负责一个点，与某个 box 是否匹配。若干个线程一组，判断是否与某个 box 匹配（这就是 block(i,j,k)）
    if (pt_idx >= pts_num || box_idx >= boxes_num || bs_idx >= batch_size) {    // 当前线程是多余的，没有分配到执行任务
        return;
    }

    // (batch_size, pts_num, boxes_num)
    int assign_idx = bs_idx * pts_num * boxes_num + pt_idx * boxes_num + box_idx;
    pts_assign[assign_idx] = 0;     // 初始化：当前 point 与当前 box 不匹配

    int box_offset = bs_idx * boxes_num * 7 + box_idx * 7;  // 当前 box 数据起始点。boxes 数据 shape (batch_size, boxes_num, 7)
    int pt_offset = bs_idx * pts_num * 3 + pt_idx * 3;      // 当前 point 数据起始点，points 数据 shape (batch_size, pts_num, 3)

    // 判断 point 是否在 box 中
    int cur_in_flag = pt_in_box3d(xyz[pt_offset], xyz[pt_offset+1], xyz[pt_offset+2], boxes3d[box_offset],
        boxes3d[box_offset+1], boxes3d[box_offset+2], boxes3d[box_offset+3], boxes3d[box_offset+4],
        boxes3d[box_offset+5], boxes3d[box_offset+6], 10.0);
    
    pts_assign[assign_idx] = cur_in_flag;
}

__device__ inline int pt_in_box3d(float x, float y, float z, float cx, float bottom_y, float cz, float h, float w,
    float l, float angle, float max_dis) {
    float x_rot, z_rot, cosa, sina, cy;
    int in_flag;

    cy = bottom_y - h / 2.0;
    if ((fabsf(x - cx) > max_dis) || (fabsf(y - cy) > h / 2.0) || (fabsf(z - cz) > max_dis)) {
        return in_flag;
    }

    cosa = cos(angle);
    sina = sin(angle);

    // 向量 PO 顺时针旋转 angle 角度。相当于将 P 点坐标系从相机坐标系转换为 CCS 坐标系，CCS 原点位于物体中心，
    //  坐标系 Z 轴沿物体朝向，X 轴垂直于 Z 轴
    x_rot = (x - cx) * cosa + (z - cz) * (-sina);
    z_rot = (x - cx) * sina + (z - cz) * cosa;

    // 切换坐标系后，判断 P 点是否在 box 之内非常简单。
    in_flag = (x_rot >= -l / 2.0) & (x_rot <= l / 2.0) & (z_rot >= -w / 2.0) & (z_rot <= w / 2.0)
    return in_flag;
}

__global__ void get_pooled_idx(int batch_size, int pts_num, int boxes_num, int sampled_pts_num,
    const int *pts_assign, int *pts_idx, int *pooled_empty_flag) {
    // 每个 box 从所有的匹配点中（所谓匹配即，点位于 box 内）采样出 sampled_pts_num 个匹配点。
    // block+thread (batch_size, boxes_num/num_threads_per_block, num_threads_per_block)
    int boxes_idx = blockIdx.x * blockDim.x + threadIdx.x;  
    if (boxes_idx >= boxes_num) {
        return;
    }

    int bs_index = blockIdx.y;
    int cnt = 0;

    for (int k = 0; k < pts_num; k++) {
        if (pts_assign[bs_idx * pts_num * boxes_num + k * boxes_num + boxes_idx]) {
            if (cnt < sampled_pts_num) {    // 匹配，且为达到采样点数上限
                pts_idx[bs_idx * boxes_num * sampled_pts_num + boxes_idx * sampled_pts_num + cnt] = k;
                cnt++;
            }
            else break;
        }
    }
    if (cnt == 0) {
        pooled_empty_flag[bs_idx * boxes_num + boxes_idx] = 1;
    }
    else if (cnt < sampled_pts_num) {
        for (int k = cnt; k < sampled_pts_num; k++) {
            // 重复采样某些点，使得当前 box 匹配的点数达到 sampled_pts_num
            int duplicate_idx = k % cnt;
            int base_offset = bs_idx * boxes_num * sampled_pts_num + boxes_idx * sampled_pts_num;
            pts_idx[base_offset + k] = pts_idx[base_offset + duplicate_idx];
        }
    }
}
```
    