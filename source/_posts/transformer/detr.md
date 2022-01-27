---
title: DETR
date: 2022-01-21 13:39:50
tags: transformer, object detection
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
图 1. DETR 直接一次性（并行）预测所有的 box 集合。输入 image 经过一个 CNN 网络输出 features，然后作为 transformer 的输入，（并行）输出预测 box 集合。



## 2.1 DETR 结构

![](/images/transformer/DETR.png)
图 2. DETR 包含：1. CNN backbone，输出 feature maps；2. encoder-decoder transformer；3. 前馈网络 FFN。

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
def nested_tensor_from_tensor_list(tensor_list):
    # tensor_list: a list of tensors. each tensor represents an image data
    # each tensor has a shape of (C, H, W)
    max_size = _max_by_axis([list(img.shape) for img in tensor_list])
    # 得到一个最大的 size，可以容纳 mini-batch 中所有的 images
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

经过 backbone 之后，image 转变成 features，其 spatial size 缩小了 $32$ 倍，故 mask 也需要等比例缩小 $32$ 倍，（代码 3）
```python
def forward(self, tensor_list: NestedTensor):
    xs = self.body(tensor_list.tensors) # (batch_size, C=2048, H, W)
    out = {}
    for name, x in xs.items():  # 对应上面 return_layers 的输出，name 为 编号
        m = tensor_list.mask
        # mask 先从 3-D，转为 4-D，然后对最低的两个维度（spatial dimension）进行
        # 插值，rescale 之后，再转为 3-D
        mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        out[name] = NestedTensor(x, mask)
    # 根据 return_layers 的输出，继续打包 features 与 masks
    return out
```

**Transformer encoder**

1. 使用一个 `1x1 conv` 对 CNN backbone 的输出 features 进行降维，从维度 $C$ 降到 $d=256$，得到 features 为 $z_0 \in \mathbb R^{d \times H \times W}$
2. 将 spatial 特征压缩至一维，即 $d \times HW$，这里 $d$ 就是特征维度，$HW$ 则作为输入 sequence 的 `seq_len`。
3. Encoder 为标准结构，包含一个 multi-head self-attention 和 一个 FFN
4. 对特征  $z_0 \in \mathbb R^{d \times H \times W}$ 进行 positional encoding，然后加到 $z_0$ 上

**position encoding**

$$PE(pos_x, 2i)=\sin(pos_x / 10000^{2i/128})
\\PE(pos_x, 2i+1)=\cos(pos_x/10000^{2i/128})
\\PE(pos_y, 2i)=\sin(pos_y/10000^{2i/128})
\\PE(pos_y, 2i+1)=\cos(pos_y/10000^{2i/128})$$

考虑了二维 spatial 位置上 x 轴 与 y 轴的位置编码，$i \in [0, d//4)$，每个空间位置 `pos` 处，位置 encoding 向量维度为 $d=256$，前 `128` 维表示 `pos_y` 位置编码，后 `128` 维表示 `pos_x` 位置编码。（代码 4）

```python
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
    y_embed = not_mask.cumsum(1, dtype=torch.float32)
    # position of x_axis
    # for one image feature: [[1,2,3,...],
    #                         [1,2,3,...],...]
    x_embed = not_mask.cumsum(2, dtype=torch.float32)
    if self.normalize:   # True, normalize position to [0, 1]
        eps = 1e-6
        # normalize the y-position
        y_embed = y_embed / (y_embed[:,-1:,:] + eps) * self.scale   # scale: 2*math.pi
        x_embed = x_embed / (x_embed[:,:,-1:] + eps) * self.scale
    # self.temperature: 10000
    # self.num_pos_feats = d//2 = 256/2=128
    dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
    # dim_t // 2: 0, 0, 1, 1, 2, 2, ... , 63, 63
    # 2 * (dim_t // 2): 0, 0, 2, 2, 4, 4, ... , 126, 126
    # dim_ := ...,  10000^{2i/128}
    dim_t = self.temperature ** (2 * (dim_t //2) / self.num_pos_features)
    # dim_t: (128,)
    # pos_x / 10000^{2i/128}
    # PE(pos_x, (2i, 2i+1)), PE(pos_y, (2i, 2i+1))
    pos_x = x_embed[:,:,:,None] / dim_t     # (B, H, W, 128)
    pos_y = y_embed[:,:,:,None] / dim_t     # (B, H, W, 128)
    # cross: [(B,H,W,64),(B,H,W,64)] => (B,H,W,64,2) => (B,H,W,128)
    #   [sin, cos, sin, cos, sin, ...]
    pos_x = torch.stack((pos_x[:,:,:,0::2].sin(), pos_x[:,:,:,1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:,:,:,0::2].sin(), pos_y[:,:,:,1::2].cos()),
    dim=4).flatten(3)
    # (B, H, W, 256) => (B, 256, H, W)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    return pos
```

整个 features 的位置编码 shape 为 $(d, H, W)$ （未考虑 batch_size 这一维度）。Backbone 的输出 features 经过 `1x1 Conv` 降维后特征 shape 为 $(d, H, W)$，两者执行 element-wise 相加，然后 flatten spatial，得到 $d \times HW$ 的特征，作为 encoder 的输入。（代码 5）
```python
# decrease channels from 2048 to 256
input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
# param: src is xs, where xs is the output of backbone
#   xs = self.body(tensor_list.tensors) # (batch_size, C=2048, H, W)
src = input_proj(src)
```



![](/images/transformer/DETR3.png)
图 3. DETR 中 Transformer 的具体结构

从图 3 中可见，backbone 输出特征经过 `1x1 conv` 降维后，直接作为 Encoder 的一个 input，记为 `f`，position encoding 作为另一个 input，记为 `PE`，这两个 tensor 的 shape 均为 $(d, HW)$（实际实现中，习惯按 `(seq_len, batch_size, feature_dim)` 的顺序 reshape，于是一个 mini-batch 中这两个 tensor 的 shape 为 $(HW,B,d)$ ），然后：
1. `f+PE` 作为 query, key；`f` 作为 value。value 中不需要 position encoding，可能是因为最终是一次性解码得到所有 object 列表，这个列表是无序的，例如原 image 上编号 `1` 的 object，其可以解码输出的列表中任意位置（index，下标），但是计算 attention 需要位置信息，故 `query` 和 `key` 上叠加了 `PE`。
2. multi-head self-attention 的输出与输入 `f` 做 Add&Norm 操作，得到输出记为 `f1`，然后 `f1` 经过一个 FFN 得到的输出特征记为 `f2`，`f1` 与 `f2` 再次做 Add&Norm 操作，得到 block 的输出。
3. Encoder 除了 `PE` 接入的位置不同，其他均与原生 transfromer 相同。

**Encoder 总结：**

`batch_size` 记为 $B$，考虑维度顺序 `(seq_len, batch_size, feature_dim)`。 $d=256$。

1. 输入 image backbone 的特征经过一个 `1x1 Conv` 降维，输出为 $z_0 \in \mathbb R^{HW \times B \times d}$，位置编码 $PE \in \mathbb R^{HW \times B \times d}$
2. PE 叠加到 `Q, K` 上
3. Block 输出 tensor 的 shape 为 $(HW, B, d)$ ，保持不变
4. 上一步的输出继续作为输入 input embedding，重复 `2~3` 若干次，最后得到整个 Encoder 的输出 shape 依然是 $(HW, B, d)$。

Encoder 的代码：（代码 6）
```python
def forward(self, src, mask=None, src_key_padding_mask=None, pos=None):
    # 由于 position embedding 直接作用到 attention 模块上的，所以直接调用
    #   attention 模块
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
2. 计算 mh self-attn 的输出，然后与输入相加，然后计算 layer_norm
3. FFN 的输出再与 FFN 的输入相加，然后计算 layer_norm。
4. Encoder layer 的输入输出 shape 均为 $(HW, B, d)$。

Encoder layer 代码：（代码 7）
```python
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
    return src
```


**Transformer decoder**

1. decoder 采用标准结构，但是是并行解码得到 $N$ 个预测 objects，非自回归。

    $N$ 是手动给出的，且需要大于单个 image 中的 object 数量，通常 $N \neq HW$

2. decoder 结构如图 3 所示，输入称为 object queries，这是 N 个 positional embedding（向量，维度为 $d$），是可学习的 positional embedding。

    （代码 8）
    ```python
    N  =  100       # 默认为 100，大于单个 image 中可能的 object 数量
    # hidden_dim = 256，就是前面 Encoder 中的参数 `d` 
    query_embed = nn.Embedding(N, hidden_dim)

    # src: output of `backbone + 1x1 Conv`
    # mask: set mask=1 for all padding pixels in mini-batch
    # query_emb: N x d, object queries
    # pos: position embeddings of all return_layers
    #   pos[-1] -> PE of the last return_layer, (B, d, H, W)
    transformer(src, mask, query_emb, pos[-1])
    ```

    图 3 关于 Decoder 的输入标注会有些误导，其实 Decoder 还有一个输入，是与 object queries 相同 shape 的全 0 tensor，如下代码中的 `tgt`。

    Transformer （Encoder+Decoder）代码实现：（代码 9）
    ```python
    def forward(self, src, mask, query_embed, pos_embed):
        # src: output of input_proj(...), the input of encoder, (B, 256, H, W)
        # mask: mask of backbone output, (B, H, W)
        # query_embed: object query embedding of decoder, (N, 256)
        # pos_embed: positional embedding, (B, 256, H, W)
        b, c, h, w = src
        # (B, 256, H, W) -> (B, 256, HW) -> (HW, B, 256)
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2),permute(2, 0, 1)
        # (N, 256) -> (N, B, 256)
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
3. Decoder 的第一个 mh self-attn 的 `query` 和 `key` 均为 `tgt` 与 object queries 相加，

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
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)     # redisual 结构
        tgt = self.norm1(tgt)

        # 调用第二个 mh self-attn，first mh self-attn 的输出作为 query，encoder 的最终输出作为 key 和 value，
        # query 和 key 分别使用 query_embedding 和 position embedding 叠加，value 保持不变
        # memory_mask：为 None，计算出 attention 之后不需要做 mask；
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

**prediction heads**

decoder 的输出 shape 为 $(M, B, N, d)$，其中 $M$ 为 decoder layer 循环次数，$B$ 为 `batch_size`，`d=256` 表示模型维度，$N$ 表示单个 image 中预测的 object 数量。

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


## 2.2 set prediction loss

### 2.2.1 原理

记 $y$ 为 gt box 集合，$\hat y = \{\hat y_i \}_{i=1}^N$ 为预测 box 集合，                      

1. $N$ 为某固定不变的值，表示对单个 image，预测 box 的数量。设置 $N$ 的值使得较大于一般意义上单个 image 中 object 数量。论文中设置 $N=100$
2. 如果 gt box 数量不足 $N$，用 no-object进行填充，使得数量为 $N$。
3. 填充的表示 no-object 的 gt boxes，其分类 index 为 `0`，表示背景 bg，坐标无所谓，因为坐标回归 loss 中只对正例预测 box 的坐标计算损失
4. 官方代码实现中，没有执行 `2~3` 两个步骤即，没有填充 gt boxes 使得数量达到 $N$。

那么在预测 box 集合和 gt box 集合上的二分匹配（bipartite matching）loss 为，

$$\hat {\sigma} = \arg \min_{\sigma \in \mathcal G_N} \Sigma_i^N \mathcal L_{match}(y_i, \hat y_{\sigma(i)})$$

其中 $\sigma$ 表示 `1~N` 个自然数集合 $[N]$ 的一个 permutation（排列），$\sigma(i)$ 表示这个排列中第 $i$ 个数。$\mathcal G_N$ 表示 $[N]$ 的所有排列的集合。$\mathcal L_{match}(y_i, \hat y_{\sigma(i)})$ 表示 $y_i$ 和 $\hat y_{\sigma(i)}$ 的匹配 loss，这个 loss 包含了分类预测 loss 和 box 位置大小预测

记 gt box 为 $y_i=(c_i, b_i)$，其中 $c_i$ 表示分类 label index（约定 `0` 为 bg index），$b_i \in [0,4]^4$ 表示 box 的 center 坐标和 height，width（相对于 image size 进行了归一化）。单个 matching pair 的损失

$$\mathcal L_{match}(y_i, \hat y_{\sigma(i)})=-\hat p_{\sigma(i)}(c_i)+ \mathbb I_{c_i \neq 0} \cdot \mathcal L_{box}(b_i, \hat b_{\sigma(i)})$$

**注：这里没有使用 NLL 损失，而是直接使用概率的负数作为损失**

对于单个 image，输出的预测分类概率应该类似于一个矩阵 $P \in [0, 1]^{N \times C}$，其中 $N$ 为单个 image 中预测 box 的数量，$C$ 为分类数量（包含了 bg）。第 `i` 个 gt box $y_i=(c_i, b_i)$ 与之匹配的预测 box 下标为 $\sigma(i)$，那么其对应到 $c_i$ 这个分类的预测概率为 $\hat p_{\sigma(i)}(c_i)=P_{\sigma(i),c_i}$


定义 Hungarian loss 表示单个 image 中所有 matching pairs 的损失，

$$\mathcal L_{Hungarian}(y, \hat y)=\sum_{i=1}^N \left[-\log \hat p_{\hat \sigma(i)}(c_i) + \mathbb I_{c_i \neq 0} \cdot \mathcal L_{box}(b_i, \hat b_{\hat \sigma(i)})\right] \tag{1}$$

说明：

1. 实际应用中，对于 $c_i=0$ 的分类损失，相较于 $c_i \neq 0$ 的分类损失，我们使用一个权重因子 $\lambda_{no-object}=0.1$，以便缓和分类不平衡的问题。

2. <font color="magenta">使用概率而非对数概率，即，去掉 （1）式中的 log，这样分类损失与坐标损失就比较相称</font>

**Bound box loss**

DETR 直接预测 box，而非 box 相对于 anchor 的坐标偏差，故直接使用 $L_1$ 损失不合适，没有考虑到 scale 带来的影响，故结合 $L_1$ 和 GIOU 作为坐标损失。

$$L_1(b, \hat b) = |b-\hat b|$$

GIOU 损失参考 [这篇文章](/2019/06/13/GIoU)。

于是 

$$\mathcal L_{box}(b_i, \hat b_i)=\lambda_{iou} \mathcal L_{iou}(b_i, \hat b_{\sigma(i)})+\lambda_{L_1}||b_i - \hat b_{\sigma(i)}||_1$$

上式中使用了两个平衡因子 $\lambda_{iou}, \ \lambda_{L_i}$，实际上分类损失也有平衡因子，只不过 $\lambda_{cls}=1$。

### 2.2.2 代码

**SetCriterion**

集合匹配损失，（代码 14）

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
        # src_idx: (BM,)   first M_1 are row ind of first image, and so on...
        # idx: (batch_idx, src_idx)
        idx = self._get_src_permutation_idx(indices)

        # t: i-th target, this is a dict. t['labels'] has a shape of (M_i,)
        # J: i-th gt box ind, its shape is (M_i,)
        # target_boxes_o: (BM,) ，minibatch 中所有匹配的 gt boxes 的 分类 id
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        # target_classes: (B, N), 预测 box 对应的 gt 分类 id，
        #       默认为 bg id，即 `num_classes`，不是 `0`，表示 预测 box 是 no-object（负例）
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, 
                                    dtype=torch.int64, device=src_logits.device)
        # 设置预测正例的 gt 分类 id
        # target_classes[idx]: (BM,)，因为 len(batch_idx)=len(src_idx)=BM
        #   且 batch_idx 取值范围 [0, B), src_idx 取值范围 [0, max(M_1,M_2,...) )
        target_classes[idx] = target_classes_o

        # 计算交叉熵，即 NLL 损失
        # empty_weight: torch.ones(num_classes+1), 且 empty_weight[-1] = 0.1
        #       正例损失系数 1.0， 负例损失系数为 0.1
        # 交叉熵的 input shape：(B, C+1, d1,d2,...), target shape：(B, d1, d2,...)
        # 交叉熵的各分类权重 shape：(C+1)
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce':loss_ce}
        return losses
    ```

2. 坐标损失，包括 l1 损失和 GIOU 损失，（代码 16）
    ```python
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        # predicted boxes
        src_boxes = outputs['pred_boxes'][idx]  # (BM, 4)
        # t: j-th target, this is a dict. t['boxes'] has a shape of (M_j, 4)
        # i: j-th gt box ind, its shape is (M_j,)
        # target_boxes: (BM, 4)
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
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

    预测 box 为非 bg 的数量 `card_pred`，其 shape 为 $(B,)$，gt box 的数量 `tgt_lengths`，其 shape 为 $(B,)$，表示 mini-batch 中各个 image 中的预测为 fg 的数量和 gt box 数量，计算这两个 tensor 的 l1 损失，并求均值，这个损失不用于反向传播。
    ```python
    card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
    ```



**HungarianMatcher**

预测集与 target 集 的匹配采用匈牙利算法匹配，Hungarian 匹配算法仅仅是用于获取与 gt boxes 匹配的预测 boxes，这个匹配过程，也用到了一些损失计算，目标是求使得损失最小的二分图匹配，这个损失与上面求网络的优化目标损失不同，后者用于反向传播更新梯度，而前者不是。（代码 17）


```python
# HungarianMatcher 是一个 nn.Module 子类
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

    cost_class = -out_prob[:, tgt_ids]      # (B*N, BM)
    cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)     # (B*N, BM)

    #（B*N, BM)
    cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

    # 计算总的损失，加权求和，损失 tensor (B*N, BM)
    C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
    C = C.view(bs, num_queries, -1).cpu()       # (B, N, BM)

    sizes = [len(v["boxes"]) for v in targets]  # (B,)  gt number of all images in batch
    # C.split(sizes, -1) -> ((B, N, M_1), (B, N, M_2), ... , (B, N, M_B))
    # c[i] -> (N, M_i)  assignment the i-th image
    # linear_sum_assignment: 计算二分图匹配中最小损失的匹配对，返回结果：(row_ind, col_ind)
    indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
    # tuple list，每个 tuple 表示对应 image 中，最佳匹配（loss 最小）的 预测 box ind 和 gt box ind
    #       每个 tuple 的 shape ((M_i,), (M_i,))
    return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i,j in indices]
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
fg_ind = pred_ind != pred_logits.shape[-1] - 1  # (B, N)
fg_num = torch.sum(fg_ind.int(), dim=-1)        # (B,)  each element is the number of pred_fg boxes for some one image
fg_boxes = pred_boxes[fg_ind]       # (n, 4)  n 为 mini-batch 中所有预测为 fg 的数量
x, y, w, h = fg_boxes.unbind(-1)    # 4 个变量 shape 均为 (n,)
xyxy = torch.stack([x-0.5*w, y-0.5*h, x+0.5*w, y+0.5*h], dim=-1)    # (n, 4)
cls_id = pred_ind[fg_ind]           # (n,)
# normalized (x1, y1, x2, y2)
xyxy_tuple = torch.split(xyxy, fg_num)  # (tensor_1,...,tensor_B), each tensor has a shape (n_i, 4), s.t. sum_i n_i = n
# fg predicted class id
cls_id_tuple = torch.split(cls_id, fg_num)  # (tensor1,...,tensor_B), each tensor's shape (n_i,), s.t. sum_i n_i = n
```

DETR 没有 post processing，故获取预测结果非常简单。