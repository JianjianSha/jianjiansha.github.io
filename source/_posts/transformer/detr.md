---
title: DETR
date: 2022-01-21 13:39:50
tags: 
    - transformer
    - object detection
mathjax: true
---
论文：[End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

代码：[facebookresearch/detr](https://github.com/facebookresearch/detr)

首次将 transformer 应用于目标检测任务中。模型简称 `DETR`。

<!--more-->

# 1. 简介
特点：
1. 端到端训练
2. 可以基于任意的 CNN baseline
3. 采用非自回归并行解码（non-autoregressive parallel decoding）

    自回归模型例如机器翻译任务，每一 step，Decoder 的输出依赖之前的输出，即，每次 Decoder 输出一个新的 token 后，都附加到输入 sequence 之后，作为下一次 Decoder 的输入。

    非自回归模型，则是并行生成所有的 tokens，这样解码速度更快。

    在 DETR 中，对于一个 gt box，仅将一个预测 box 与它对应起来，得到一个 pair，并计算匹配 loss，这样预测 boxes 的顺序可以是任意，即 预测 boxes 之间是独立的，从而使得可以并行计算得到所有预测 boxes。
4. 对大目标的检测上，DETR 比之前的目标检测模型效果更好，这是因为 transformer 的 attention 是 global 的，而 CNN 则是 local 的。相对的，在小目标检测上，则效果差些。

5. set prediction

    DETR 一次性预测所有的 box 集合，需要一个 matching loss 函数，用于 预测 box 与 gt box 之间的匹配，采用基于匈牙利算法（Hungarian algorithm）的 loss 计算方式，一个 gt box 最多只有一个预测 box 与之匹配，从而省去了 NMS 等 postprocessing。

    匈牙利算法可以参考这个 [代码实现](https://gist.github.com/JianjianSha/ed5ea9022a8aa1217113dc7d30b52044)


# 2. DETR

DETR 的结构示意图如下，

![](/images/transformer/DETR1.png)
<center>图 1. DETR 直接一次性（并行）预测所有的 box 集合。</center>

1. 输入 image ，shape `(batch_size, 3, H, W)` 
2. 经过一个 CNN 网络输出 features 。shape `(batch_size, c, h, w)`

    例如 ResNet-50，下采样率为 32，输出 channel 为 2048，即 `c=2048, h=H/32, w=W/32`
    
3. backbone 的输出 features 作为 transformer 的输入，另外还使用了位置嵌入向量 `PE` 作为 transformer 的输入，具体参见下文分析

4. transformer decoder 输出经前馈网络 FFN 输出（feature map）上各 location 的预测分类得分和预测 box。（注：整个过程是并行的）



## 2.1 DETR 结构

![](/images/transformer/DETR2.png)
<center>图 2. DETR 包含：1. CNN backbone，输出 feature maps；2. encoder-decoder transformer；3. 前馈网络 FFN。</center>

**Backbone**

CNN backbone 的输入 image $x \in \mathbb R^{3 \times H_0 \times W_0}$，输出 features 为 $f \in \mathbb R^{C \times H \times W}$。通常取，$C=2048$，$H,W=H_0/32, W_0/32$。

代码中 backbone 默认使用 `ResNet50`，（代码 1）
```python
parser.add_argument('--backbone', default='resnet50', type=str)
backbone = getattr(torchvision.models, 'resnet50')(...) # create resnet50
# 标记 layer4 为 backbone 的输出层，其编号为 0
# 对于 segmentation task，则有多个输出层
return_layers = {'layer4': '0'}
# self.body 输出 layer4 的 output features
self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
```

机器翻译任务中，对短句子的末尾进行填充 `<pad_tok>`，然后再创建 `src_mask`，其中 `<pad_tok>` 对应 `mask=1`，这里对 image 采取类似的预处理，（代码 2）
```python
# code snippet 2
# 将 list 中每个图像数据紧靠左上角填充（左上角为图像坐标系 O 点）
def nested_tensor_from_tensor_list(tensor_list):
    # tensor_list: a list of tensors. each tensor represents an image data
    # each tensor has a shape of (C, H, W)
    # max_size: [Cmax,Hmax,Wmax]。 tensor list中，最大 channel，最大 H，最大 W
    max_size = _max_by_axis([list(img.shape) for img in tensor_list])
    # 得到一个最大的 size，可以容纳 mini-batch 中所有的 images
    # [batch_size, Cmax, Hmax, Wmax]，其实 Cmax=3，因为所有图像通道均为 3
    batch_shape = [len(tensor_list)] + max_size
    b, c, h, w = batch_shape
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device
    tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
    # 创建 mask，指示哪些 spatial pixeles 是填充数据
    mask = torch.ones((b, h, w), dtype=dtype, device=device)
    for img, pad_img, m in zip(tensor_list, tensor, mask):
        pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
        m[:img.shape[1], :img.shape[2]] = 0
    return NestedTensor(tensor, mask)   # 打包 image 数据和 mask
```

（_为什么不将 image  resize 到相同 size 并保存各 image 的 scale 比例，而是使用 padding 和 mask？这是因为后者处理方法应该效果更好_）

经过 backbone 之后，image 转变成 features，其 spatial size 缩小了 $32$ 倍，故 mask 也需要等比例缩小 $32$ 倍，（代码 3）
```python
# code snippet 3
# class BackboneBase(nn.Module)
def forward(self, tensor_list: NestedTensor):
    xs = self.body(tensor_list.tensors) # (batch_size, C=2048, H, W)
    # xs: backbone 的输出 features
    out = {}
    # 检测任务 xs -> {'0':res}
    # 分割任务 xs -> {'0':res0, '1':res1, '2':res2, '3':res3}
    for name, x in xs.items():  # 对应上面 return_layers 的输出，name 为 编号
        m = tensor_list.mask
        # mask 先从 3-D，转为 4-D，然后对最低的两个维度（spatial dimension）进行
        # 插值，rescale 之后，再转为 3-D
        # 这里使用最近邻插值，将 mask resize 到原来的 1/32
        mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        out[name] = NestedTensor(x, mask)
    # 根据 return_layers 的输出，继续打包 features 与 masks
    return out
```

代码 3 中关于检测任务和分割任务的 backbone 的输出不同，检测只需要最后的特征，而分割需要中间层特征和最后的特征，如下所示

```
+-------------+    +---------+    +---------+    +---------+    +---------+
|conv1+bn    -|--->| stage2 -|-+->| stage3 -|-+->| stage4 -|-+->| stage5 -|-+->
|+relu+maxpool|    |         | |  |         | |  |         | |  |         | |
+-------------+    +---------+ |  +---------+ |  +---------+ |  +---------+ |
                               v              v              v              v
(object detection)                                                         res
(segmentation)                res0           res1           res2           res3

res0: (B, 256, H0//4, W0//4)
res1: (B, 512, H0//8, W0//8)
res2: (B,1024, H0//16, W0//16)
res3: (B, 2048, H0//32, W0//32)
(H0, W0) 是网络的 input size
```


**Transformer encoder**

1. 使用一个 `1x1 conv` 对 CNN backbone 的输出 features 进行降维，从维度 $C$ 降到 $d=256$，得到 features 为 $z_0 \in \mathbb R^{d \times H \times W}$
2. 将 spatial 特征压缩至一维，即 $(d,H,W)\rightarrow (d,HW)$，这里 $d$ 就是(transformer)特征维度，$HW$ 则作为输入 sequence 的 `seq_len`。
3. Encoder 为标准结构，包含一个 multi-head self-attention 和 一个 FFN
4. 对特征  $z_0 \in \mathbb R^{d \times H \times W}$ 进行 positional encoding，然后加到 $z_0$ 上

**position encoding**

$$PE(pos_x, 2i)=\sin(pos_x / 10000^{2i/128})
\\\\ PE(pos_x, 2i+1)=\cos(pos_x/10000^{2i/128})
\\\\ PE(pos_y, 2i)=\sin(pos_y/10000^{2i/128})
\\\\ PE(pos_y, 2i+1)=\cos(pos_y/10000^{2i/128})$$

考虑了二维 spatial 位置上 x 轴 与 y 轴的位置编码，$i \in [0, d//4)$，每个空间位置 `pos` 处，位置 encoding 向量维度为 $d=256$，前 `128` 维表示 `pos_y` 位置编码，sin 和 cos 间隔，后 `128` 维表示 `pos_x` 位置编码，sin 和 cos 间隔，记 `pos` 坐标为 $(x, y)$，那么此处位置 encoding 为 
```
[PE(y,0), PE(y,1), PE(y,2),PE(y,3), ..., PE(y,126),PE(y,127),PE(x,0),PE(x,1), ... ,PE(x,126),PE(x,127)]
```
pixel 像素坐标为 (x, y)，其中 $x \in [1,W], \ y \in [1,H]$，如果需要归一化到 $[0, 2\pi]$ 之间，使用 $2\pi x / W, \ 2\pi y / H$ 。

```python
# code snippet 4
# class PositionEmbeddingSine(nn.Module):
# 获取 position embedding
def forward(self, tensor_list: NestedTensor):
    # tensor_list: the data pack of one return_layer(refer to return_layers)
    x = tensor_list.tensors     # feartures of batch-images
    mask = tensor_list.mask     # corresponding masks of features
    # x: (B, C, H, W)
    # mask: (B, H, W), where `1` elements represent padding pixels
    not_mask = ~mask

    # position of y-axis, (B, H, W)
    # for one image features: [[1,1,...], 
    #                          [2,2,...],...]
    # 对于i-th图像特征而言，y_embed[i] 沿y轴增 1
    y_embed = not_mask.cumsum(1, dtype=torch.float32)
    # position of x_axis
    # for one image feature: [[1,2,3,...],
    #                         [1,2,3,...],...]
    x_embed = not_mask.cumsum(2, dtype=torch.float32)
    if self.normalize:   # True, normalize position to [0, 1]*2*pi
        eps = 1e-6
        # normalize the y-position。y_embed[:,-1,:] 是 H axis 最大值，相除使用 H axis 范围(0,1]
        y_embed = y_embed / (y_embed[:,-1:,:] + eps) * self.scale   # scale: 2*math.pi
        # x_embed 与 y_embed 类似处理，使得值范围 (0, 2*pi]
        x_embed = x_embed / (x_embed[:,:,-1:] + eps) * self.scale
    # self.temperature: 10000
    # self.num_pos_feats = d//2 = 256/2=128
    dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
    # dim_t // 2: 0, 0, 1, 1, 2, 2, ... , 63, 63
    # 2 * (dim_t // 2): 0, 0, 2, 2, 4, 4, ... , 126, 126
    # dim_t: 一维向量，size=128,  10000^{2i/128} ,  i = 0,1,...,63
    dim_t = self.temperature ** (2 * (dim_t //2) / self.num_pos_features)
    # pos_x / 10000^{2i/128}
    # PE(pos_x, (2i, 2i+1)), PE(pos_y, (2i, 2i+1))
    pos_x = x_embed[:,:,:,None] / dim_t     # (B, H, W, 128)
    pos_y = y_embed[:,:,:,None] / dim_t     # (B, H, W, 128)
    # cross: sin和cos间隔，[(B,H,W,64),(B,H,W,64)] => (B,H,W,64,2) => (B,H,W,128)
    #   [sin, cos, sin, cos, sin, ...]
    pos_x = torch.stack((pos_x[:,:,:,0::2].sin(), pos_x[:,:,:,1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:,:,:,0::2].sin(), pos_y[:,:,:,1::2].cos()),
    dim=4).flatten(3)
    # (B, H, W, 256) => (B, 256, H, W)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    return pos # (B, 256, H, W)
```

backbone 输出包含两部分，
1. image 数据经过 ResNet 输出的 features
2. position embedding，与第 `1` 步中的 features 具有相同的 spatial size

    分割任务有 4 个不同 spatial size 的特征，每个特征均执行一次 position embedding（除最后一个特征的 PE，其他 PE 并没有用到）

代码如下：
```python
class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
    
    def forward(self, tensor_list: NestedTensor):
        # self[0] 指的 ResNet
        xs = self[0](tensor_list)   # 检测任务 -> {'0': NestTensor0}
                                # 分割任务 -> {'0': NT0, '1': NT1, '2': NT2, '3': NT3}
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            pos.append(self[1](x).to(x.tensors.dtype))  # 对应的 PE
    return out, pos
```

整个 features 的位置编码 shape 为 $(d, H, W)$ （未考虑 batch_size 这一维度），而 features 的 shape 为 `(512,H,W)` 或者 `(2048,H,W)` (参考各ResNet的输出 channel)，故Backbone 的输出 features 经过 `1x1 Conv` 降维后特征 shape 为 $(d, H, W)$，两者执行 element-wise 相加，然后 flatten spatial，得到 $d \times HW$ 的特征，作为 encoder 的输入。
```python
# code snippet 5
# class DETR(nn.Module):
# decrease channels from 2048 to 256
input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)


# def forwrad(sef, samples: NestedTensor):
# features, pos is just `out, pos` in last code snippet.
features, pos = self.backbone(samples)
src, mask = features[-1].decompose()
# param: src is the output of backbone, its shape (B, 2048, H, W)
# return: src (B, 256, H, W)
src = input_proj(src)
hs = self.transformer(src, mask, self.query_embed.weight, pos[-1])[0]
```



![](/images/transformer/DETR3.png)

<center>图 3. DETR 中 Transformer 的具体结构</center>

从图 3 中可见，backbone 输出特征经过 `1x1 conv` 降维后，直接作为 Encoder 的一个 input，记为 `f`，position encoding 作为另一个 input，记为 `PE`，这两个 tensor 的 shape 均为 $(d, HW)$（实际实现中，习惯按 `(seq_len, batch_size, feature_dim)` 的顺序 reshape，于是一个 mini-batch 中这两个 tensor 的 shape 为 $(HW,B,d)$ ），然后：
1. `f+PE` 作为 query, key；`f` 作为 value。value 中不需要 position encoding，可能是因为最终是一次性解码得到所有 object 列表，这个列表是无序的，例如原 image 上编号 `1` 的 object，其可以解码输出的列表中任意位置（index，下标），但是计算 attention 需要位置信息，故 `query` 和 `key` 上叠加了 `PE`。
2. multi-head self-attention 的输出与输入 `f` 做 Add&Norm 操作，得到输出记为 `f1`，然后 `f1` 经过一个 FFN 得到的输出特征记为 `f2`，`f1` 与 `f2` 再次做 Add&Norm 操作，得到单个 block 的输出。
3. Encoder 除了 `PE` 接入的位置不同，其他均与原生 transfromer 相同。

**Encoder 总结：**

`batch_size` 记为 $B$，考虑维度顺序 `(seq_len, batch_size, feature_dim)`。 $d=256$。

1. 输入 image backbone 的特征经过一个 `1x1 Conv` 降维，输出为 $z_0 \in \mathbb R^{HW \times B \times d}$，位置编码 $PE \in \mathbb R^{HW \times B \times d}$
2. PE 叠加到 `Q, K` 上
3. Block 输出 tensor 的 shape 为 $(HW, B, d)$ ，保持不变
4. 第 `3` 步的输出继续作为下一个 block 的输入（仍使用一开始的那个 PE），重复步骤 `2~3` $N=6$ 次，最后得到整个 Encoder 的输出 shape 依然是 $(HW, B, d)$。

Encoder 的代码：（代码 6）
```python
# code snippet 6
# class TransformerEncoder(nn.Module):
def forward(self, src, mask=None, src_key_padding_mask=None, pos=None):
    # src: `1x1 Conv` 输出特征 （reshape 之后） （HW, B, d)
    # src_key_padding_mask: (HW, B, d)，backbone 输出的 mask，由于batch 内各 
    #      image size 大小不一，使用了 zero-padding，故需要使用 mask
    # pos: position embedding (HW, B, d)
    # mask: 对 attention 是否需要做 mask。在 Encoder 对 attention 不需要做
    #       mask，故这里 mask=None
    output = src    # `1x1 Conv` 输出特征 （reshape 之后）
    for layer in self.layers:
        output = layer(output, src_mask=mask, \
            src_key_padding_mask=src_key_padding_mask, pos=pos)
    if self.norm is not None:   # normalize_before is false, so self.norm is None
        # normalize_after, so do not need norm here.
        output = self.norm(output)
    return output
```

**Encoder layer (block) 小结：**

1. features 与 position embedding 相加，作为 `query` 和 `key`，features 作为 `value`
2. 计算 multi-head self-attn 的输出，然后与输入相加（残差结构，identity connection），然后计算 layer_norm
3. FFN 的输出再与 FFN 的输入相加，然后计算 layer_norm（沿 C,H,W 归一化，与 BatchNorm 不同）。
4. Encoder layer 的输入输出 shape 均为 $(HW, B, d)$。

Encoder layer 代码：（代码 7）
```python
# code snippet 7
def forward_post(self, src, src_mask, src_key_padding_mask, pos):
    # src: input embedding      (HW, B, d)
    # pos: position embedding   (HW, B, d)
    # src_mask：对 attention 参数矩阵做 mask
    # src_key_padding_mask: 对 `key` 做 mask
    q = k = self.with_pos_embed(src, pos)   # 叠加 position embedding
    src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                          key_padding_mask=src_key_padding_mask)[0]
    src = src + self.dropout(src2)      # residual 结构
    src = self.norm1(src)
    # linear -> act -> dropout -> linear
    src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
    src = src + src2
    src = self.norm2(src)
    return src  # (HW, B, d)
```


**Transformer decoder**

1. decoder 采用标准结构，但是是并行解码得到 $N$ 个预测 objects，非自回归。

    $N$ 是手动给出的，且需要大于单个 image 中的 object 数量，通常 $N \neq HW$（预先设置例如 `N=100`）。注意需要与图 3 Encoder 中的 block 数量 N 区分开来，这是两个不同的变量。

2. decoder 结构如图 3 所示，输入称为 object queries，这是 N 个 positional embedding（向量，维度为 $d$），是可学习的 positional embedding。

    ```python
    # code snippet 8
    N  =  100       # 默认为 100，大于单个 image 中可能的 object 数量
    # hidden_dim = 256，就是前面 Encoder 中的参数 `d` 
    # Embedding.weight 根据标准正态分布进行初始化
    query_embed = nn.Embedding(N, hidden_dim)

    # src: output of `backbone + 1x1 Conv`  (B,d, H, W)
    # mask: set mask=1 for all padding pixels in mini-batch  (B, H, W)
    # query_emb: N x d, object queries，N=100 是预设的目标数据
    # pos: position embeddings of all return_layers
    #   pos[-1] -> PE of the last return_layer, (B, d, H, W)
    transformer(src, mask, query_emb.weights, pos[-1])
    ```

    图 3 关于 Decoder 的输入标注会有些误导，其实 Decoder 还有一个输入，是与 object queries 相同 shape 的全 0 tensor，如下代码中的 `tgt`，因为还没见到 encoder 的输出（或者说没有见到 transformer 输入），故初值为全 0 tensor，

    Transformer （Encoder+Decoder）代码实现：（代码 9）
    ```python
    # class Transformer(nn.Module):
    def forward(self, src, mask, query_embed, pos_embed):
        # src: output of input_proj(...), the input of encoder, (B, 256, H, W)
        # mask: mask of backbone output, (B, H, W)
        # query_embed: object query embedding of decoder, (N, 256)
        # pos_embed: positional embedding, (B, 256, H, W)
        b, c, h, w = src
        # (B, 256, H, W) -> (B, 256, HW) -> (HW, B, 256)
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2),permute(2, 0, 1)   # (HW, B, 256)
        # (N, 256) -> (N, B, 256)
        # query_embed 是手动设置的 N 个目标的 object_queries
        query_embed = query_embed.unsqueeze(1).repeat(1, b, 1)
        # (B, H, W) -> (B, HW)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)     # (N, B, 256)
        # memory: output of encoder, (HW, B, 256)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        # hs：(M, N, B, d)，其中 M 为 decoder layer iteration number。参见代码 11 的返回结果
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                        pos=pos_embed, query_pos=query_embed)
        # hs: (M, N, B, 256) -> (M, B, N, 256)
        # memory: (HW, B, 256) -> (B, 256, HW) -> (B, C, H, W)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(b, c, h, w)
    ```
3. Decoder 的第一个 mh self-attn 的 `query` 和 `key` 均为 `tgt` 与 object queries 相加，query embedding 相当于 pos embedding，

    Decoder layer 的代码：（代码 10）
    ```python
    def forward_post(self, tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, 
        memory_key_padding_mask, pos, query_pos):
        # query_pos: 就是前面说的 object queries `query_emb`， shape 为 (N, B, d)
        # tgt: 初始时为全零 tensor，shape 为 (N, B, d)
        # tgt_mask: 对 tgt 做 mask，这里不需要，为 None
        # memory: encoder 的最终输出 (HW, B, d), 
        # memory_mask： 第二个 mh self-attn 中与 memory 计算 attention 之后的的 mask，这里为 None
        # tgt_key_padding_mask: 第一个 mh self-attn 中 `key` 的 mask， 为 None
        # memory_key_padding_mask: 第二个 mh self-attn 中 `key` 的 mask
        #   由于 batch 中 image 大小各不相同，左上角对齐，右下防 padding，padding pixels 的 mask=1
        #   memory_mask 缩放到 (H, W) 空间大小，(B,H,W) -> (B, HW)，参见代码 3 中的 mask
        # pos: encoder 中的 position embedding，(HW, B, d)
        q = k = self.with_pos_embed(tgt, query_pos)     # target 输入，Q, K 需要叠加 query embedding
        # 调用第一个 mh self-attn，参数 tgt_mask, tgt_key_padding_mask 均为 None，即不做 mask
        # 第一个 multi-head self-attn 因为没有用到 encoder 输出的特征，自然不需要 mask
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)     # redisual 结构
        tgt = self.norm1(tgt)

        # 调用第二个 mh self-attn，first mh self-attn 的输出作为 query，encoder 的最终输出作为 key 和 value，
        # query 和 key 分别使用 query_embedding 和 position embedding 叠加，value 保持不变
        # memory_mask：为 None，计算出 attention 之后不需要做 mask；这跟 encoder 不同，因为 
        #               encoder 的输入是经过填充的，所以对于填充的位置的 attention，需要置 0
        #               decoder 是固定好 N 个位置的目标，N 个位置的 attention 都需要。
        # memory_key_padding_mask：与 encoder 中 src mask 相同，(B, HW)，由于 images 大小各不相同，存在 padding，故需要 mask
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos), key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)       # (N, B, d)
        return tgt
    ```

4. 整个 Decoder 的前向过程：（代码 11）
    ```python
    def forward(self, tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask,
                pos, query_pos):
        # 参数与 Decoder layer 的前向传播方法参数相同，略过解释
        output = tgt        # 全零 tensor，(N, B, d)，N=100
        intermediate = []

        for layer in self.layers:   # 循环执行 Decoder layer 若干次
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            # refer to the paper's section "Auxiliary decoding losses":
            #   add prediction FFNs and Hungarian loss after each decoder layer...
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.norm is not None:   # True
            output = self.norm(output)  # the last (final) decoder layer's output
        if self.return_intermediate:    # 默认为 True，即，使用辅助 decoding loss
            # (M, N, B, d)，M is the total iteration number for decoder layer
            return torch.stack(intermediate)
        return output.unsqueeze(0)      # (1, N, B, d)
    ```

    将 Decoder 中每个 block（总共 $M=6$ 个 block）的输出均存储起来 `intermediate`，并 stack 后返回，每个 block 的输出 `(N, B, d)`，那么 stack 后 Decoder 的输出为 `(M, N, B, d)`，其中 $M=6, \ N=100$。

5. Transformer 的输出包含两部分

    - Decoder 的输出 `(M, B, N, d)` （经过了 shape 转置）
    - Encoder 的输出 `(B, d, H, W)` （shape permute+view）

**prediction heads**

decoder 的输出 shape 为 $(M, B, N, d)$，其中 $M$ 为 decoder layer 循环次数，$B$ 为 `batch_size`，`d=256` 表示模型维度，$N$ 表示单个 image 中预测的 object 数量。

最后将 decoder 的输出分别经过 cls head 和 box head，预测分类得分和坐标，

1. decoder 的输出经一个线性变换，使得维度从 `d` 变为 `C+1`，这里 `C` 表示 fg 分类数量，`1` 表示 bg 。（代码 12）
    ```python
    class_embed = nn.Linear(hidden_dim, num_classes+1)
    outputs_class = class_embed(hs)         # hs 为 decoder layers 的输出，shape 为 (M, B, N, d)
    # 得到分类（非归一化）得分，(M, B, N, C+1)
    ```
2. decoder 的输出经另一路分支即，由三层全连接层组成的分支，中间层的输出单元保持不变，输出层的输出单元数量为 4，表示坐标，坐标数据 shape 为 `(M, B, N, 4)`，（代码 13）
    ```python
    bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
    outputs_coord = bbox_embed(hs).sigmoid()        # 归一化坐标
    # tensor 经 MLP，shape 变化为
    # MLP 输入 (M, B, N, d) -> (M, B, N, d) -> (M, B, N, d) -> (M, B, N, 4)
    # 每一个 "->" 表示一个全连接层
    ```
3. DETR 的输出

    ```python
    # 使用 Decoder 最后一个 block 进行预测
    # pred_logits: (B, N, C+1)
    # pred_boxes: (B, N, 4)
    out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
    if self.aux_loss:   # 为 True，其他 Decoder block 输出用于辅助 loss 计算
        out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
    # aux_outputs: [{}]
    ```

## 2.2 Loss



> 预测集损失用于反向传播，优化网络。Hungarian 匹配损失用于寻找匹配的预测 box，不用于反向传播。

记 $y$ 为 gt box 集合，$\hat y = \lbrace \hat y_i \rbrace _ {i=1}^N$ 为预测 box 集合，                      

1. $N$ 为某固定不变的值，表示对单个 image，预测 box 的数量。设置 $N$ 的值使得较大于一般意义上单个 image 中 object 数量。论文中设置 $N=100$
2. 如果 gt box 数量不足 $N$，用 no-object进行填充，使得数量为 $N$。
3. 填充的表示 no-object 的 gt boxes，其分类 index 为 `0`，表示背景 bg，坐标无所谓，因为坐标回归 loss 中只对正例预测 box 的坐标计算损失
4. __`2~3` 是论文中原话，代码中并没有使用 no-object 进行填充使得 gt box 数量为 `N=100`__。参考 下方关于 `HungarianMatcher` 的代码

### 2.2.1 Hungarian 损失

在预测 box 集合和 gt box 集合上的二分匹配（bipartite matching）loss 为，

$$\hat {\sigma} = \arg \min_{\sigma \in \mathcal G _ N } \Sigma _ i ^ N \mathcal L _ {match}(y _ i, \hat y _ {\sigma(i)})$$

其中 $\sigma$ 表示 `1~N` 个自然数集合 $[N]$ 的一个 permutation（排列），$\sigma(i)$ 表示这个排列中第 $i$ 个数。$\mathcal G _ N$ 表示 $[N]$ 的所有排列的集合。$\mathcal L _ {match}(y _ i, \hat y _ {\sigma(i)})$ 表示 $y _ i$ 和 $\hat y _ {\sigma(i)}$ 的匹配 loss，这个 loss 包含了分类预测 loss 和 box 位置大小预测

记 gt box 为 $y _ i=(c _ i, b _ i)$，其中 $c _ i$ 表示分类 label index（约定 `0` 为 bg index），$b_i \in [0,1] ^ 4$ 表示 box 的 center 坐标和 height，width（相对于 image size 进行了归一化）。单个 matching pair 的损失包含两部分：分类损失和坐标损失

$$\mathcal L _ {match}(y _ i, \hat y _ {\sigma(i)})=-\hat p _ {\sigma(i)}(c _ i)+ \mathbb I _ {c _ i \neq 0} \cdot \mathcal L _ {box}(b _ i, \hat b _ {\sigma(i)})$$

**注：这里没有使用 NLL 损失，而是直接使用概率的负数作为损失**

对于单个 image，输出的预测分类概率应该类似于一个矩阵 $P \in [0, 1] ^ {N \times (C+1)}$，其中 $N=100$ 为单个 image 中预测 box 的数量，$C$ 为分类数量，$C+1$ 则包含了 bg。

第 `i` 个 gt box $y_i=(c_i, b_i)$ 与之匹配的预测 box 下标为 $\sigma(i)$，那么其对应到 $c_i$ 这个分类的预测概率为 $\hat p _ {\sigma(i)}(c _ i)=P _ {\sigma(i),c _ i}$


定义 Hungarian loss 表示单个 image 中所有 matching pairs 的损失，

$$\mathcal L _ {Hungarian}(y, \hat y)=\sum _ {i=1} ^ N \left[-\log \hat p _ {\hat \sigma(i)}(c _ i) + \mathbb I _ {c _ i \neq 0} \cdot \mathcal L _ {box}(b _ i, \hat b _ {\hat \sigma(i)})\right] \tag{1}$$

说明：

1. <font color="magenta">使用概率而非对数概率，即，去掉 （1）式中的 log，这样分类损失与坐标损失就比较相称。</font>（在下方的 Hungarian 代码中，没有对概率取对数操作）

**Bound box loss**

DETR 直接预测 box，而非 box 相对于 anchor 的坐标偏差，故直接使用 $L_1$ 损失不合适，没有考虑到 scale 带来的影响，故 __结合 $L_1$ 和 GIOU 作为坐标损失__。

$$L_1(b, \hat b) = |b-\hat b|$$

GIOU 损失参考 [这篇文章](/2019/06/13/GIoU)。

于是 

$$\mathcal L _ {box}(b _ i, \hat b _ i)=\lambda_{iou} \mathcal L _ {iou}(b _ i, \hat b _ {\sigma(i)})+\lambda_{L _ 1}||b _ i - \hat b _ {\sigma(i)}|| _ 1$$

上式中使用了两个平衡因子 $\lambda _ {iou}, \ \lambda _ {L_i}$，代码中 $\lambda _ {iou}=2, \ \lambda _ {L _ 1}=5$，实际上分类损失也有平衡因子，只不过 $\lambda _ {cls}=1$。

### 2.2.2 Hungarian 代码


**HungarianMatcher**

预测集与 target 集 的匹配采用匈牙利算法匹配，Hungarian 匹配算法仅仅是用于获取与 gt boxes 匹配的预测 boxes，这个匹配过程，也用到了一些损失计算，目标是求使得损失最小的二分图匹配，这个损失与上面求网络的优化目标损失不同，后者用于反向传播更新梯度，而前者（即 Hungarian 匹配损失）不是。（代码 17）


```python
# class HungarianMatcher(nn.Module):
def forward(self, outputs, targets):
    # outputs 是 DETR 的输出，这是一个 dict 类型，key 可以是：
    #   pred_logits: 最后一个 decoder layer 的输出分类未归一化得分  (B, N, C+1)
    #   pred_boxes: 最后一个 decoder layer 的输出坐标       (B, N, 4)
    # targets: dict list，每个 dict 表示一个 image 的 target，包含 key：
    #   boxes: 某个 image 中 objects 的 (x, y, w, h)， shape 为 (M, 4)，
    #           M 表示 object 数量，每个 image 的 M 均不同
    #   labels：某个 image 中 objects 的分类 index（0 表示 bg），shape 为 (M, )
    #   image_id：image 的 id（coco 数据集中每个 image 有一个数值编号）
    #   ...：其他 keys 省略
    bs, num_queries = outputs['pred_logits'].shape[:2]  # B, N
    out_prob = outputs['pred_logits'].flatten(0, 1).softmax(-1) # (B*N, C+1)
    out_bbox = outputs['pred_boxes'].flatten(0, 1)  # (B*N, 4)

    tgt_ids = torch.cat([v['labels'] for v in targets]) # (BM,), BM = M_1+M_2+...+M_B
    tgt_bbox = torch.cat([v['boxes'] for v in targets]) # (BM, 4)

    # =================================================
    # 注意：以下三个损失计算理论上应分别在单个 image 内计算
    # 但是为了计算效率提升，故将 mini-batch 内所有预测和 
    # target 各自混合然后再计算这三种损失，最后取各 image
    # 内的预测与 target 之间的匹配损失，参见下方 c[i] 变量
    # =================================================
    # 计算预测分类与 gt 分类 两两之间的损失
    cost_class = -out_prob[:, tgt_ids]      # (B*N, BM)
    # 计算预测 boxes 与 gt boxes，两两 之间的 p1 范数 => 差的绝对值
    cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)     # (B*N, BM)

    #（B*N, BM)
    cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

    # 计算总的损失，加权求和，损失 tensor shape: (B*N, BM)
    # cost_bbox=5, cost_class=1, cost_giou=2
    C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
    C = C.view(bs, num_queries, -1).cpu()       # (B, N, BM)

    sizes = [len(v["boxes"]) for v in targets]  # (B,)  gt number of all images in batch
    # C.split(sizes, -1) -> ((B, N, M_1), (B, N, M_2), ... , (B, N, M_B))
    # c[i] -> (N, M_i)  assignment the i-th image
    # 注意：这里预测数量 N，target 数量 M_i，所以并没有将 target 数量通过
    #   no-object 填充到 N
    # linear_sum_assignment: 计算二分图匹配中最小损失的匹配对，返回结果：(row_ind, col_ind)
    # row_ind 和 col_ind 均为长度为 M_i 的数量（这里假设了 N >= M_i）
    # row_ind 表示匹配的 pairs 中预测 box 的索引
    # col_ind 表示对应的 target 的索引
    indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
    # tuple list，每个 tuple 表示对应 image 中，最佳匹配（loss 最小）的 预测 box ind 和 gt box ind
    #       每个 tuple 的 shape ((M_i,), (M_i,))
    return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i,j in indices]
```

### 2.2.3 目标函数 

几种种损失（检测任务只有前三种，分割任务包含以下所有损失）：
1. 分类损失，平衡系数 $\lambda_{cls}=1$
2. 坐标损失 $L_1$，平衡系数 $\lambda_{L_1}=5$
3. GIOU 损失，平衡系数 $\lambda_{iou}=2$
4. mask 损失，平衡系数 $1$ （分割任务中使用）
5. dice 损失，平衡系数 $1$ （分割任务中使用）
6. 实际应用中，对于 bg 的分类损失，相较于 fg 的分类损失，我们使用一个权重因子 $\lambda_{no-object}=0.1$，以便缓和分类不平衡的问题。

__分类损失：__

单个 image 的分类预测损失：

$$L _ {cls}=-\frac 1 N \sum_{i=1} ^ N w_i \log \hat p _ {\hat \sigma(i)}(c _ i)$$

其中权重

$$w _ i=\begin{cases} 1 & 0\le c _ i <C(\text{fg}) \\\\ 0.1 & c _ i=C(\text{bg})\end{cases}$$

__L1 坐标损失：__

mini-batch 的 L1 坐标损失，

$$L _ {L _ 1}=\frac 1 {\sum _ i M _ i}\sum _ i \sum _ {j=1} ^ {M _ i} \sum _ {c \in \{x,y,w,h\}} ||\hat b _ {j,c}-b _ {j,c}||$$

其中 $i$ 表示 mini-batch 中第 `i` 个 image， $M_i$ 是 `i-th` image 中 targets 数量。$b_{j,c}$ 表示第 `j` 个 target 的某个坐标 (`cx,cy,w,h`)，$\hat b_{j,c}$ 表示与第 `j` 个 target 匹配的预测 box 的某个坐标。

__GIoU:__

$$L _ {iou}=\frac 1 {\sum _ i M _ i} \sum _ {j=1} ^ {M_i} GIoU(\hat b _ j, b _ j)$$


目标损失：

$$L=\lambda _ {cls}L _ {cls}+\lambda _ {L _ 1}L _ {L _ 1}+\lambda _ {iou}L _ {iou}$$



### 2.2.4 目标函数的代码

**SetCriterion**

集合匹配损失，用于反向传播优化网络（下面的 Hungarian 匹配损失则不参加反向传播，仅用于寻找匹配的 预测 box，注意区别）

（代码 14）

```python
def forward(self, outputs, targets):
    # 只保留最后一个 decoder layer 的输出，辅助输出（非最后 decoder layer 的输出）的损失后面再计算
    outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}  
    indices = self.matcher(output_without_aux, targets)     # 计算 Hungarian Loss，见下文代码 17

    num_boxes = sum(len(t['labels']) for t in targets)      # 统计 minibatch 中所有 gt box 数量
    # 
    num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
    num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()# 不考虑分布式训练，get_world_size()=1

    losses = {}
    for loss in self.losses:    ['labels', 'boxes', 'cardinality']
        # get_loss: 根据指定的 loss 类型，获取相应的 loss 值；
        # labels -> loss_labels(); boxes -> loss_boxes; cardinality -> loss_cardinality
        losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
    if 'aux_outputs' in outputs:
        for i, aux_outputs in enumerate(outputs['aux_outputs']):    # 计算辅助 loss
            indices = self.matcher(aux_outputs, targets)            # 获取 匹配 indices
            for loss in self.losses:
                if loss == 'masks': continue    # 分割任务，不计算辅助 mask loss
                l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, log=loss!='labels')
                l_dict = {k+f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
    return losses
```

最终返回一个 loss dict，以 `loss_ce` 开头的 key 表示交叉熵分类损失，`loss_bbox` 开头的 key 表示 l1 坐标（xywh）损失，以 `loss_giou` 开头的 key 表示 giou 损失。对于非最后一个 decoder layer 的损失，使用 `_<i>` 结尾，其中 `i` 为从 0 开始的编号。

1. 分类损失。注意，这里使用 NLL，用于反向传播更新梯度。（代码 15）
    ```python
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        # outputs: {'pred_logits': (B, N, C+1), 'pred_boxes': (B, N, 4)}
        # targets: dict list, 每个 dict 表示一个 image 的 target
        # indices: tuple list，每个 tuple 表示一个 image 的预测 box ind 和 gt box ind
        # num_boxes: minibatch 中所有 gt box 的数量
        src_logits = outputs['pred_logits']     # (B, N, C+1)

        # batch_idx: (BM,)  where BM=M_1+M_2+...+M_B, first M_1 is `0`, and
        #   then are M_2 `1`, and so on...
        # src_idx: (BM,)   first M_1 are row ind(pred box ind) of first image, and so on...
        # idx: (batch_idx, src_idx)
        idx = self._get_src_permutation_idx(indices)

        # t: i-th target, this is a dict. t['labels'] has a shape of (M_i,)
        # J: gt box ind of matched pairs in i-th img, its shape is (M_i,)
        # target_boxes_o: (BM,) ，minibatch 中所有匹配的 gt boxes 的 分类 id
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        # target_classes: (B, N), 预测 box 对应的 gt 分类 id，
        #       默认为 bg id，即 `num_classes`，不是 `0`，`0` 是第一个 fg 分类id
        #       表示 预测 box 是 no-object（负例）
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, 
                                    dtype=torch.int64, device=src_logits.device)
        # idx: mini-batch 中匹配对中的预测 box ind（范围 0~N-1）
        # target_classes_o: mini-batch 中匹配对中的 target 分类 id
        # 设置 target_classes 中被匹配中的预测 box 的分类，其分类为对应的 target 分类 id
        target_classes[idx] = target_classes_o

        # 计算交叉熵，即 NLL 损失
        # empty_weight: torch.ones(num_classes+1), 且 empty_weight[-1] = 0.1
        #       正例损失系数 1.0， 负例损失系数为 0.1
        # 交叉熵的 input shape：(B, C+1, N), target shape：(B, N)
        # 交叉熵的各分类权重 shape：(C+1)
        # loss_ce: (B, N) -> (reduction: mean) -> scalar
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce':loss_ce}
        return losses
    ```

2. 坐标损失，包括 l1 损失和 GIOU 损失，（代码 16）
    ```python
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        # idx: 获取 batch 中所有匹配对中的预测 box ind
        idx = self._get_src_permutation_idx(indices)
        # predicted boxes，获取 batch 匹配对中的预测 box（归一化坐标，cx,cy,w,h）
        src_boxes = outputs['pred_boxes'][idx]  # (BM, 4)
        # t: j-th target, this is a dict. t['boxes'] has a shape of (M_j, 4)
        # i: j-th gt box ind, its shape is (M_j,)
        # target_boxes: (BM, 4)，batch 中所有 target box 坐标（归一化，cx,cy,w,h)
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        # 计算 L1 损失
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')    # (BM, 4)

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes   # num_boxes 应该等于 src_boxes.shape[0]?

        # GIOU loss = 1 - GIOU
        loss_giou = 1-torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)
        ))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses
    ```
3. cardinality loss：预测 fg box 数量与 gt box 数量（在一个 mini-batch 内）的平均差。

    预测 box 为非 bg 的数量 `card_pred`，其 shape 为 $(B,)$，gt box 的数量 `tgt_lengths`，其 shape 为 $(B,)$，表示 mini-batch 中各个 image 中的预测为 fg 的数量和 gt box 数量，计算这两个 tensor 的 L1 损失，并求均值，这个损失 __不用于反向传播__。
    ```python
    card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
    ```




**反向传播**

```python
loss_dict = criterion(outputs, targets)     # 分类损失，l1 损失，giou 损失
weight_dict = criterion.weight_dict     # 各损失的权重，分类损失为基准（其权值为 1），l1 权值为 5，giou 权值为 2

# 计算所有损失的加权和，包括所有 decoder layer 的各项损失
losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

optimizer.zero_grad()
losses.backward()
if max_norm > 0:    # 默认 0.1
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
optimizer.step()
```

## 2.3 测试

对一个新的 input image 进行预测时，根据前面分析，知道两个预测分支的输出为：1.分类得分$(1, N, C+1)$，2.预测坐标（xywh） $(1, N, 4)$。

首先根据分类得分得到 fg boxes 以及对应的分类 id
```python
# pred_logits, pred_boxes
# batch_size  B=1
pred_ind = pred_logits.argmax(-1)   # (B, N)
# (B, N, C+1)，the last class id `C` represents bg
fg_ind = pred_ind != pred_logits.shape[-1] - 1  # shape: (B, N)
# (B,)  each element is the number of pred_fg boxes for some one image
fg_num = torch.sum(fg_ind.int(), dim=-1)   

# (n, 4)  n 为 mini-batch 中所有预测为 fg 的数量
fg_boxes = pred_boxes[fg_ind]       
x, y, w, h = fg_boxes.unbind(-1)    # 4 个变量 shape 均为 (n,)
# (cx, cy, w, h) -> (x1, y1, x2, y2)
xyxy = torch.stack([x-0.5*w, y-0.5*h, x+0.5*w, y+0.5*h], dim=-1)    # (n, 4)
cls_id = pred_ind[fg_ind]           # (n,)   class id of predicted box
# normalized (x1, y1, x2, y2)
# split x1y1x2y2 into a tuple, each element represents predicted coords of one image in mini-batch
xyxy_tuple = torch.split(xyxy, fg_num)  # (tensor_1,...,tensor_B), each tensor has a shape (n_i, 4), s.t. sum_i n_i = n
# fg predicted class id
cls_id_tuple = torch.split(cls_id, fg_num)  # (tensor1,...,tensor_B), each tensor's shape (n_i,), s.t. sum_i n_i = n
```

DETR 没有 post processing，故获取预测结果非常简单。

# 3. 总结
DETR 首次（不是？）将 Transformer 用于目标检测，后面有很多研究均基于 DETR 进行改进，故对 DETR 研究透彻非常有必要，为了以后能快速恢复对 DETR 的了解，对 DETR 各关键点进行总结。


## 3.1 结构
DETR 结构图如图 1 和图 2。

### 3.1.1 Network Input

对于一个 mini batch 中图像数据，左上对齐，右下 zero-padding，得到一个 batch 数据，以及一个相同 spatial size 的 mask。

对于 target bbox 坐标，从 COCO anno 文件加载的是 `(x1, y1, w, h)`，经过 `ConvertCocoPolysToMask` 转换后为 `(x1, y1, x2, y2)`，然后再经 `transforms.Normalize` 转换为 __归一化后的__ `(cx, cy, w, h)`。（归一化指 将 `x, y` 除以 `w, h`）。

### 3.1.2 Backbone

ResNet50/ResNet101（这两个常用），下采样率 `32`。

预处理的数据经过 backbone，输出 feature，然后将 Network Input 中的 mask 数据 rescale 到原来的 `1/32`。两个输出：
1. features
2. PE

### 3.1.3 Encoder

上一步的 features 和 PE 的 channel 数不同，对 features 采用 `1x1 conv` 降维。

Encoder 输入：
1. 降维后的 features
2. PE
3. src_key_padding_mask，这是由于 batch 内各 image size 不同而进行 zero padding，由此需要引入 mask，对部分位置上的 attention 进行 mask。

Encoder 结构如图 3 所示，为了方便查看，这里在下方再贴出来。features 为 query, key, value，其中 query 和 key 还需要另外 element-wise 加上 PE，value 不需要叠加 PE。

单个 block 的输出 `output` shape 为 `(HW, B, d)`，保持不变，这个 `output` 继续作为下一个 block 的 query, key, value，且同时 query 和 key 需要叠加 PE（与最开始的 PE 相同，value 不需要叠加 PE。

重复 $N=6$ 次 Encoder block 后输出 `output`，其 shape 为 `(HW, B, d)` 。


### 3.1.4 Decoder

Decoder 结构如图 3 所示，输入包含：

1. object_query：用于表征 $N=100$ 个预测目标，这是一个可学习的 Embedding（weight使用 $\mathcal N(0,1)$ 进行初始化）。shape 为 `(N=100, d)`，其中 $d$ 为模型维度，维度调整后 shape 为 `(N, B, d)`。
2. tgt：与 object_query 相同 shape，初始化为全零 tensor。
3. Encoder 的最后一个 block 的输出，记为 `memory`，shape 为 `(HW, B, d)`。
4. memory_key_padding_mask：这是对 `memory` 进行 mask。仍然是因为 network input image 大小不一，导致 zero-padding，从而需要对 padded position 进行 mask。
5. PE：position embedding，与 Encoder 中所用 PE 相同。

注意 Encoder block 中只有一个 attn layer，而 Decoder 中有两个 attn layer，这两个 attn layer 的输入 __不同__。

__第一个 attn layer：__

query 和 key 均为 `tgt` 与 `object_query` 的叠加。value 为 `tgt`，attn layer 的输出，记为 `tgt2`，其 shape 仍然是 `(N, B, d)`。

这个输出 `tgt2` 依次经过 dropout，residual connect 和 norm 处理之后，记为 `tgt`，准备进入 第二个 attn layer。

__第二个 attn layer：__

注意看图 3， q, k, v 均与第一个 attn layer 不同。

1. 将 `tgt`（上一个步的输出） 和 `object_query` （与上一步的相同）叠加，作为 query，shape 为 `(N, B, d)`
2. 将 Encoder 的最终输出 `memory` 与 PE 叠加，作为 key，shape 为 `(HW,B,d)`。
3. 将 Encoder 的最终输出 `memory` 作为 value，shape 为 `(HW, B, d)`。

Q 与 K 的 attention weight，`(B, N, HW)`，与  value 作用后结果 shape 为 `(N, B, d)`，然后再依次经过 dropout，residual connect 和 norm 操作，记这一步结果为 `tgt`。

__FFN：__

上一步结果 `tgt` 经一个双 fc layer 组成的前馈网络，输出的 shape 保持不变为 `(N, B, d)`，然后再依次经过 dropout，residual connect 和 norm 操作，得到 Decoder 中单个 block 的输出，记结果为 `tgt`。

将第一个 block 的输出 `tgt` 代替原始的全零 tensor 的 `tgt`，其他参数保持不变，送入第二个 block。重复 $M=6$ 次 block。

每个 block 的输出仅保存起来，最后再 stack，得到 Decoder 的输出 `(M, N, B, d)`。

### 3.1.6 检测 heads

分类和位置两个 heads。

1. 分类 head：一个 fc 层，输出 channel 为 `num_classes+1`，因为 $N=100$ 的预测目标通常有 bg。输出 shape 为 `(M, B, N, num_classes+1)`
2. 坐标 head：三个 fc 层组成的 MLP。输出 shape `(M, B, N, 4)`

    MLP 输入 channel 和 hidden channel 均为 d，输出 channel 为 4 （bbox 坐标）
    MLP 中前两个 fc 有 relu，最后一个 fc 没有 relu


## 3.2 LOSS

见上面 `2.2` 节。

## 3.3 Prediction

见上面 `2.3` 节。

# 4. Segmentation

在 decoder 输出之后增加一个 mask head，实现分割功能。对于全景分割，则需要将 thing 和 stuff（除目标之外的东西，如天空、草地、建筑等）统一看待，对它们所在的区域（region）均需要进行检测。COCO 全景数据包含 80 个目标分类和 53 个 stuff 分类。

![](/images/transformer/DETR4.png)
<center>图 4. 全景分割的 mask head</center>

分割任务的实现思路：

1. 使用前面 DETR 在目标检测集上训练

2. 冻结 DETR 的参数，然后增加一个 mask head，在分割数据集上训练这个 mask head

具体而言，

1. 在 decoder 之后继续使用 multi-head self-attention，，如图 4 左起第一个 attention，其 Q 为 decoder 的输出，K 为 encoder 的输出（参考图 3 decoder 的第二个 attention），计算 $QK^{\top}$ 得到 attention 作为输出（没有 value），shape 为 `(B, N, 8, H, W)`，其中 N=100，8 是 multi-heads 数量。

2. 使用 FPN 网络，输入是 backbone 经过一个 conv 适配后的输出以及第 `1` 步得到的 attention：

    - 前者 shape 为 `(B, d, H, W)`，增加一个维度后为 `(B, 1, d, H, W)`，repeat 为 `(B, N, d, H, W)`，最后 flatten 为 `(BN, d, H, W)`
    - 后者 shape 为 `(B, N, 8, H, W)`，flatten 为 `(BN, 8, H, W)`，
    - 两者 concatenate，得到 `(BN, d+8, H, W)`
    - 然后经过 FPN ，得到输出 `pred_masks`，其 shape 为 `(BN, 1, 8H, 8W)`，review 为 `(B, N, 8H, 8W)`

    整个过程如图 5 所示，

    ![](/images/transformer/DETR5.png)
    <center>图 5. FPN-style 的 mask head</center>

## 4.1 代码

```python
class DETRsegm(nn.Module):
    def __init__(self, detr, freeze_detr=False):
        super().__init__()
        self.detr = detr    # 目标检测任务的 DETR
        if freeze_detr:
            for p in self.parameters():
                p.requires_grad_(False)
        
        # transformer 模型 dim，例如 256
        # multi-head self-attn 的 head 数量
        hidden_dim, nheads = detr.transformer.d_model, detr.transformer.nhead
        self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0)
        self.mask_head = MaskHeadSmallConv(hidden_dim + nheads, [1024, 512, 256], hidden_dim)
    
    def forward(self, samples):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)   # size 不同的图像填充
        # 对于分割任务，backbone 返回 4 个 layer 的输出特征
        # 4 个 PE，PE 与 feature 一一对应
        features, pos = self.detr.backbone(samples)
        bs = features[-1].shape[0]  # B, batch_size
        # src: (B, 2048, H, W), mask: (B, H, W)
        src, mask = features[-1].decompose()    # 最后一个特征 H0/32,W0/32
        src_proj = self.detr.input_proj(src)    # 维度适配，(B, d, H, W)
        # 最小 size 的特征输入到 transformer
        # hs: decoder 输出（经过 review）(M, B, N, d)，其中 N=100，query_emb 数量
        # memory: encoder 输出（经过 reshape）(B, d, H, W)
        hs, memory = self.detr.transformer(src_proj, mask, self.detr.query_embed.weight, pos[-1])

        # M 是 decoder attention block 的数量，所有 attention block 的输出
        outputs_class = self.detr.class_embed(hs)   # cls head 输出 (M, B, N, C+1)
        outputs_coord = self.detr.bbox_embed(hs).sigmoid()  # (M, B, N, 4)
        # 记录最后一个 decoder atten block 对应的 cls 和 box 输出
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.detr.aux_loss:
            # 保存其他 decoder atten block 对应的 cls 和 box 输出
            output['aux_outputs'] = self.detr._set_aux_loss(outputs_class, outputs_coord)
        
        # 在这之前，执行逻辑与目标检测的 DETR 完全一样
        # 在这之后，开始 mask head 前向传播

        # 输入为最后一个 decoder atten block 的输出，encoder 输出，输入图像 mask
        #                           (B, N, d),   (B, d, H, W),  (B, H, W)
        # bbox_mask: (B, 1, 1, H, W)
        bbox_mask = self.bbox_attention(hs[-1], memory, mask=mask)

        # features 中各 feature 的 shape 为 
        # (B, 256, H0//4, W0//4)
        # (B, 512, H0//8, W0//8)
        # (B,1024, H0//16, W0//16)
        # (B, 2048, H0//32, W0//32)
        # (H0, W0) 是网络的 input size
        # seg_masks: (BN, 1, H0//4, W0//4)。二分类，非归一化得分
        seg_masks = self.mask_head(src_proj, bbox_mask, [features[2].tensors, features[1].tensors, features[0].tensors])
        # seg_masks: (B, N, H0//4, W0//4)
        outputs_seg_masks = seg_masks.view(bs, self.detr.num_queries, seg_masks.shape[-2], seg_masks.shape[-1])
        out['pred_masks'] = outputs_seg_masks
        return out
```


**# Dataset**

数据集代码，这里我们主要看 target 的构造，因为 image 输入与检测任务一样，target 中包含 mask 的数据。

anno 文件中，记录了每个目标/stuff 的多边形区域，每个目标/stuff 可能会有多个多边形区域，这是因为目标/stuff 被其他东西遮挡。

```python
# 
class ConvertCocoPolysToMask(object):
    def __call__(self, image, target):
        '''
        一个图像中可能有多个 目标/stuff，每个 目标/stuff 可能有多个多边形
        image: 单个图像
        target: dict 类型

        返回：dict 类型
        '''
        w, h = image.size

        image_id = target['image_id']   # 图像 id
        image_id = torch.tensor([image_id]) # 转为 tensor

        # anno 是一个 list[dict] 对象，每个 dict 记录一个目标/stuff 的标注数据
        anno = target['annotations']

        # 筛选那些不是 拥挤的目标/stuff（太难，使用了反而影响学习效果）
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        # 所有 目标/stuff 的 bbox。一个目标/stuff 仅有一个 bbox
        boxes = [obj['bbox'] for obj in anno]

        # (N, 4)    其中 N 为当前图像中的 目标/stuff 数量
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]    # (x,y,w,h) -> (x1,y1,x2,y2)
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj['category_id'] for obj in anno]  # 所有 目标/stuff 的分类
        classes = torch.tensor(classes, dtype=torch.int64)  # (N, )

        if self.return_masks:   # 这里是分割任务，所以为 True
            # list[list[float]]，内层的 list[float] 表示一个多边形
            segmentations = [obj['segmentation'] for obj in anno]
            # masks: (N, h, w) ，这里 N 表示当前图像中 目标/stuff 数量
            # 每个 目标/stuff 均使用与图像 size 相等的 binary map 表示 mask
            masks = convert_coco_poly_to_mask(segmentations, h, w)
        
        # x2 > x1, y2 > y1
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]         # (n, 4)
        classes = classes[keep]     # (n, )

        target = {'boxes': boxes, 'labels': classes}
        if self.return_masks:
            masks = masks[keep]     # (n, h, w)
            target['masks'] = masks
        ...
        return image, target
```

分析了 model 和 dataset 的代码，然后我们再看训练代码（其中关键的部分），

```python
def train_one_epoch(model, criterion, data_loader, optimizer, device
    epoch, max_norm):
    ...

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        ...
        # targets: list[dict]，每个 dict 表示一个图像 target，参考 ConvertCocoPolysToMask 输出

        # outputs 是一个 dict，包含：
        # pred_logits: (B, N, C+1)  # 最后一个 attn block 经 cls head 的输出
        # pred_boxes: (B, N, 4)     # 最后一个 attn block 经 box head 的输出
        # pred_masks: (B, N, H0//4, W0//4)
        # aux_outputs: 其余 attn blocks 经 cls 和 box head 输出的loss
        #               [{'pred_logits': xxx, 'pred_boxes': xxx}]
        outputs = model(samples)
        loss_dict = criterion(outputs, targets) # 计算各种损失
        weight_dict = criterion.weight_dict     # 每种损失的权重不同
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        ...
```

**# 损失**


mask loss 使用两个损失：1. Focal loss；2. DICE loss

1. flocal loss

    $$L = -\alpha _ t (1-x _ t) ^ {\gamma} \log x _ t$$

    其中 $x _ t= \begin{cases} x & y=1 \\\\ 1-x & y=0 \end{cases}$

2. DICE loss

    $$L = 1-  \frac {2 \sum _ i ^ N p _ i g _ i}{\sum _ i ^ N p _ i + \sum _ i ^ N g _ i}$$


其他损失包括分类和坐标损失，与检测任务一样，这里不再细说。

计算损失的代码如下，

```python
class SetCriterion(nn.Module):
    ...
    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        # indices: [(tensor, tensor)]，匈牙利匹配结果
        # 一共 B 个二元素 tuple，tuple[0] 表示匹配的预测box 下标，tuple[1] 表示gt box 下标
        indices = self.matcher(outputs_without_aux, targets)

        # 这个 batch 内所有图像中的 box 数量之和
        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, ...)
        nux_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:    # labels, boxes, cardinality, masks
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        
        # 使用 aux_outputs ，decoder 其他  attn blocks 的输出经过 cls head
        # 和 box head 的分类预测和坐标预测，计算相关损失
        ...

        return losses
    
    def loss_masks(self, outputs, targets, indices, num_boxes):
        # 根据匈牙利算法匹配的结果中提取匹配的预测 box id
        # src_idx: (batch_idx, src_box_idx)，两个元素均为 1-D tensor
        src_idx = self._get_src_permutation_idx(indices)
        # tgt_idx: (batch_idx, tgt_box_idx)，匹配的 gt box id
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs['pred_masks']   # (B, N=100, H0//4, W0//4)
        # 筛选出匹配的预测 box。这里 m 表示 batch 中所有匹配的 box pair 的数量
        src_masks = src_masks[src_idx]  # (m, H0//4, W0//4)
        masks = [t['masks'] for t in targets]   # [(n, H0, W0)]
        # masks 是 batch 中所有图像的 box mask，由于每个图像中 box 数量不等
        # 所以使用具有最多 box 数量 n1，即 target_masks: (B, n1, H0, W0)
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        # 筛选出匹配的 gt box
        target_masks = target_masks[tgt_idx]    # (m, H0, W0)

        # 插值。二维 spatial size (H0//4, W0//4) 放大到 (H0, W0)
        # 输入必须是 4-d tensor，所以 src_mask 从 (m, H0//4, W0//4) 
        # 转为 (m, 1, H0//4, W0//4)，然后插值上采样，得到 (m, 1, H0, W0)
        src_masks = interpolate(src_masks[:, None], 
                                size=target_masks.shape[-2:],
                                mode='bilinear', 
                                align_corners=False) # (m, 1, H0, W0)
        src_masks = src_masks[:, 0].flatten(1)  # (m, H0*W0)
        target_masks = target_masks.flatten(1)  # (m, H0*W0)
        target_masks = target_masks.view(src_masks.shape)
        # src_masks 表示非归一化预测得分
        # target_masks，binary map： 0/1
        losses = {
            'loss_mask': sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            'loss_dice': dice_loss(src_masks, target_masks, num_boxes)
        }
```