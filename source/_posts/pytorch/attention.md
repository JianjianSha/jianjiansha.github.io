---
title: Attention
date: 2022-01-25 09:25:37
tags: PyTorch
mathjax: true
---


相关论文：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

[论文解读](/transformer/2022/01/17/transformer/self_attention)

# 1. 流程简介
为了方便理解，这里我简洁地进行总结。以机器翻译任务为例说明。

将 multi-head self-attention 简记为 mh self-attn

## 1.1 Encoder
输入预处理：

1. 输入 shape 按 `(batch_size, seq_len)` 的顺序，记为 `(B, L)`，input 中各元素表示 word 的 index，index 范围为 `[0, V]`，其中 `0` 表示 `<pad_tok>`，`V` 表示词汇表大小。
2. heads 数量为 `n`，模型维度为 `d`，每个 head 的维度为 $d/n$ 
3. 输入 tensor 经 embedding 转化为 input embedding，其 shape 为 `(B, L, d)`，embedding 参数矩阵维度为 `(V+1, d)`
4. 叠加 position embedding，故叠加后 shape 相同，仍为 `(B, L, d)`。



mh self-attn 过程：

1. input embedding 使用参数矩阵 $W^Q \in \mathbb R^{d \times d}$，得到 `query` 数据，记为 $Q$，其 shape 为 $(B, L, d)$。
2. input embedding 分别使用参数矩阵 $W^K \in \mathbb R^{d \times d_k}$，$W^V \in \mathbb R^{d \times d_v}$，线性转换为 `key` 和 `value`，记为 $K, \ V$，其 shape 分别为 $(B, L, d_k), \ (B, L, d_v)$。

1. $Q, K ,V$ 的最后一维，实际上是 `n` 个 heads concatenate 的结果，可以 `Q1=Q.view(B, L, n, d/n)`，`K1=K.view(B, L, n, d_k/n)`， `V1=V.view(B, L, n, d_v/n)`。
2. 执行 attention 操作，$A_i=Q_iK_i^{\top}, \ i=1,2,...,n$，注意这里是每个 head 内部进行 atention 操作，由于 attention 涉及到 `L` 和 最后一维，为了方便矩阵计算，进行以下维度调整， `Q1.permute(0,2,1,3)`，使得 shape 变为 `(B, n, L, d/n)`，$K$ 进行同样的维度调整，为 `(B,n,L,d_k/n)`，通常 $d_k=d$，然后计算 attention tensor，其 shape 为 $(B,n,L,L)$
3. 对 attention 做 softmax 进行归一化，结果记为 $\hat A$
4. 使用 attention 作为权重，加权求和输出，这个步骤依然是每个 heads 内部进行，`V1.permute(0, 2, 1, 3)`，调整维度顺序变为 `(B,n,L,d_v/n)`，然后就可以执行矩阵乘法 $\hat A_i V_i$，得到的 n 个 heads 的输出 tensor，记为 $O$，其 shape 为 $(B,n,L,d_v/n)$
5. 然后再调整维度顺序，使得 shape 变为 $(B, L, d_v)$
    ```python
    O.permute(0, 2, 1, 3).contiguous().view(B,L, d_v)
    ```
8. 最后使用一个线性变换，参数矩阵 $W^O \in \mathbb R{d_v \times d}$，将上一步结果映射为 $OW^O$，其 shape 为 $(B, L, d)$

9. FFN 就不详细介绍了。 将第 `8` 步中的输出作为 input embedding，循环执行 `1~8` 步 `N-1` 次，最终得到输出的 shape 依然是 $(B, L, d)$

注：
1. $d_k \equiv d$，这样才能执行向量内积 $\mathbf q_i \mathbf k_i$，或者矩阵相乘 $Q_iK_i^{\top}$，但是 `MultiheadAttention` 构造函数中用到了表示 $d_k$ 的参数 `kdim`，难道还能 $d_k \neq d$？不理解。

**Image 相关任务如目标检测，分割等**

_Image 经过 backbone 得到特征 `features`，其 shape 为 $(B, C, H, W)$，通过一个 `1x1 Conv`，将维度 `C` 调整为模型维度 `d`，然后再 `features.view(B, d, HW).permute(0, 2, 1)`，使得顺序为 `(batch_size, seq_len, feature_dim)`。但是图像任务中，Transformer 结构稍有不同，具体参考论文 [DETR]()，以及我的文章 [detr 解读]()_

## 2.2 Decoder
Encoder 中的 `N` 个循环的 Block 中，每个 Block 均由一个 mh self-attn 和一个 FFN 构成，而 Decoder 中对应的 Block 则是两个 mh self-attn 和一个 FFN 构成。

Decoder 输入的预处理部分：
1. 输入 shape 为 `(B, L)`，这里 `seq_len` 为 $L$，与 Encoder 中的 `L` 可能不相等，输入 tensor 中各元素表示 word index，通过线性变换转为 embedding，其 shape 为 `(B, L, d)`。
2. 叠加 position embedding，shape 不变。
3. 将 input embedding 分别线性变换为 $Q, K, V$，shape 均为 $(B, L, d)$，作为第一个 mh self-attn 的 `query`，`key`，`value`。第一个 mh self-attn 的输出 shape 保持不变，为 $(B,L,d)$
4. 第一个 mh self-attn 输出作为第二个 mh self-attn 的 `query`，而 Encoder 最终（`N` 次循环之后）的输出 `src_enc`，作为第二个 mh self-attn 的 `key` 和 `value`，这两个变量的 shape 均为 $(B, S, d)$，注意与 `query` 具有不同的 shape，且将 Encoder 输出 shape 中的 `seq_len` 记为 $S$ 。
5. 根据 $A=QK^{\top}$ 可知，attention 矩阵维度为 $L \times S$，故第二个 mh self-attn 的输出 $\hat A V$ 的维度为 $(B, L, d)$。
6. FFN 结构略。Decoder 中 block（两个 mh self-attn 和一个 FFN）的输出 shape 为 $(B, L, d)$。
7. 第 `6` 步的输出作为 input embedding，循环步骤 `3~6` 若干次，得到最终的输出。
8. 使用线性变换将上一步的输出 $(B, L, d)$ 变为 $(B, L, V+1)$，然后执行 Softmax 分类。

构造函数参数：

1. `embed_dim`： 模型维度，通常指 `query` 变量的最后一维的大小。
2. `num_heads`：multi-head 中 heads 数量
3. `dropout`：防止过拟合，丢弃率。
4. `bias`：是否对线性变换 (nn.Linear) 增加偏置。
5. `kdim`： `key` 的最后一维大小，默认为 `embed_dim`
6. `vdim`： `value` 的最后一维大小，默认为 `embed_dim`
7. `batch_first`：True，那么维度顺序为 `(batch_size, seq_len, feat_dim)`，否则为 `(seq_len, batch_size, feat_dim)`。

前向传播参数：

1. `query`：$(B, L, d)$
2. `key`：  $(B, S, d)$
3. `value`：$(B, S, d)$

4. `key_padding_mask`：$(B, S)$，对 `key` 做 mask。

    attention 中，对 `query` 中的每个部分（mini-batch 中单个 instance，query 一共有 $L$ 个部分）对 $S$ 个 key 做 attention，但是有时候由于 sequence 长度不够而进行 padding，或者 autoregression 中预测是 one-by-one 的，所以无法对后面的 `key` 的部分做 attention，这两种情况下，都需要对 `key` 做 mask。

5. `need_weights`：True，返回 attention 权重矩阵 $\hat A$，否则不返回。
6. `attn_mask`：指定 对 attention 做 mask。shape 为 $(L, S)$ 或者 $(B \cdot n, L, S)$

    由于 $A_i =Q_i K_i^{\top} \in \mathbb R^{L \times S}$，所以 mask shape 为 $(L, S)$ ，表示 mini-batch 中 $B \cdot n$ 个 heads 均使用相同的 mask；如果为 $(B \cdot n, L, S)$，那么为  mini-batch 中 $B \cdot n$ 个 heads 分别指定 mask。

前向传播输出参数：

1. `attn_output`：attention 的输出，shape 为 $(B, L, d)$，其中 $d$ 为模型维度。
2. `attn_output_weights`：attention 权重参数 $\hat A$，shape 为 $(B,L, S)$。
    由于 multi-head，本来 weights shape 应该为 $(B, n,L, S)$，沿着 `dim=1` 求均值。

# 2. 代码

Attention 模块 PyTorch 源码解读。
<!--more-->
```python
torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True,
add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False,
device=None, dtype=None)
```

参数说明：
1. `embed_dim`： model dimension，即上面的 $d$。
2. `num_heads`：多头 attention 中的 head 的数量
3. `drop_out`：`attn_output_weights` 上的丢弃率。

    `attn_output_weights` shape 为 `(tgt_seq_len, src_seq_len)` 表示各 element 之间的 weight。通常情况，`tgt_seq_len=src_seq_len=seq_len`，参考 [attention](https://jianjiansha.github.io/2022/01/17/transformer/self_attention/) 一文中的矩阵 $A$。 

4. `bias`： 默认为 `True`，表示在输入输出的 `linear` layer（全连接层）上使用 bias。参考 [attention](https://jianjiansha.github.io/2022/01/17/transformer/self_attention/) 一文中的图 2 中右图。
5. `batch_first`：默认 `False`，表示 `(seq_len, batch_size, embed_dim)`，否则输入 shape 应为 `(batch_size, seq_len, embed_dim)`。

```python
forward(query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True)
```

参数说明（以默认 `batch_first=False` 为例说明）：
1. `query`：`(L,N,E)`，其中 `L=seq_len, N=batch_size, E=embed_dim`。`L` 是 target sequence length。
2. `key`: `(S,N,E)`，`S` 表示 source sequence length。
3. `value`：`(S,N,E)`。考虑单个样本，query 和 key 做 attention，得到 attention weights，这是一个矩阵 `(L, S)`，然后与 value（数据矩阵为 `(S, E)`）相乘得到结果 `(L, E)`。

4. `key_padding_mask`：`(N, S)`，指示 key 中哪些元素是需要忽略的，即被看作是 padding。`key_padding_mask` 中 `True` 值指示相应的 key 元素值将被忽略。

    例如，某个 sequence 中，序列长度为 `S`，第 $i$ 个 `key_padding_mask` 元素值为 `1`，$i < S$，那么得到的 attention 矩阵中第 $i$ 列全为 `0`。

5. `attn_mask`：阻止某些位置上的 attention。shape 为 `(L,S)` 或者 `(N*num_heads, L, S)`。2D mask 会广播到 3D。这个 mask 直接对 query 和 key 的 attention weight 矩阵进行 mask。