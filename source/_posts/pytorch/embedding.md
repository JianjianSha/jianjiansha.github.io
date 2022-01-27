---
title: 词嵌入向量
date: 2022-01-19 10:05:19
tags: PyTorch
p: pytorch/embedding
---
embedding 就是将一个 word 转换为 一个 vector，最简单的方法就是使用 `one-hot vector`，其长度为词汇表大小，但是 `one-hot vector` 无法表示词的语义，比如两个词是相近还是相反含义，或者无关，embedding vector 则可以很好的表征词的语义。

<!--more-->

# 1. 原理

给词汇表中每个 word 分配一个唯一的编号 index，例如词汇表大小为 $V$，那么 index 为 $1,\ldots, V$，记 embedding vector 维度为 $d$，那么可以使用一个矩阵 $E \in \mathbb R^{V \times d}$，矩阵每一行表示一个 word 对应的 embedding。

这个 embedding 过程其实就是一层全连接层（fc），fc 的参数就是上面说的矩阵 $E$，fc 的输入是 `one-hot vector`，记为 $\mathbf x_i \in \mathbb R^{1 \times V}$，第 `i` 个元素值为 `1`，那么 fc 的输出 $\mathbf e_i = \mathbf x E \in \mathbb R^{1 \times d}$。

将 word 转换为 embedding vector 后，可以作为网络的输入，既然如此，可以直接像训练网络那样训练这个 fc。

# 2. Embedding 类型
在 PyTorch 中，使用 `Embedding` 类来实现 embedding 相关操作，

```python
torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None,
    max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False,
    _weight=None, device=None, dtype=None)
```

参数说明：
1. `num_embeddings` ： embeddings 数量，即词汇表大小。
2. `embedding_dim`： embedding vector 维度
3. `padding_idx`：可选，指定填充 index 值。通常选取 `batch_size` 数量的句子（word sequence）作为一个 batch，但是各个句子长度（word 数量，即 sequence length）可能不同，所以固定一个值作为 sequence length，然后对于那些较短的句子，使用 `padding_idx` 进行填充。一种经典的做法是：
    - 取 `batch_size` 的 sentences，使用分词器分词，得到 `batch_size` 个 sequences
    - 取 sequences 中长度最大值，记为 `L`
    - 对每个 sequence， prepend `<init_token>`，然后再 append `<end_token>`，最后，对长度不足 `L` 的 sequence，填充 `<pad_token>`，使得长度为 `L`。这里的长度 `L` 不包括 `<init_token>` 和 `<end_token>`。

    **`padding_idx` 如果指定，那么对应的 embedding vector 在训练过程中将保持不变，这个 embedding vector 初始为全 0 向量，但是可以手动更改。**
    ```python
    # 默认 padding embedding 为 0-vector
    embedding = nn.Embedding(10, 3, padding_idx=0)
    input = torch.LongTensor([[0, 2, 0, 5]])    # (B, seq_len) = (1, 4)
    embedding(input)

    # 手动修改 padding embedding
    padding_idx = 0
    embedding.weight    # print embeddings before updating
    with torch.no_grad():
        embedding.weight[padding_idx] = torch.ones(3)
    embedding.weight    # print embeddings after updating
    ```
    `padding_idx` 如果指定，那么 `num_embeddings` 为 `V+1` 。

4. `max_norm` ： 可选。如果指定，那么任何 embedding vector，如果其 norm 超过这个值，会被收缩使得 norm 等于 `max_norm`。
5. `norm_type`：norm 类型，与 `max_norm` 配合使用。
6. `scale_grad_by_freq`：如果为 True，将根据 word 在 mini-batch 中的频率倒数，rescale （weight 关联的）gradient，类似于样本均衡，否则如果某个 word 出现太多，导致其对应的 gradient 太大，网络注意力全在这些高频 word 上了。
7. `sparse`：如果为 True，那么 `gradient` w.r.t. `Embedding.weight` 矩阵为稀疏矩阵，由于这个矩阵维度较大，采用稀疏矩阵可以节约内存，但是只有几个优化器支持稀疏矩阵：`optim.SGD(CUDA 和 CPU)`，`optim.SparseAdam(CUDA 和 CPU)`，以及 `optim.Adagrad(CPU)`。

注意：`max_norm` 不为 none 时，Embedding 类的 `forward` 方法会 in-place 修改其 `weight` tensor，然而在梯度计算时，不能 in-place 修改 tensor，否则梯度计算不准确，所以如果在损失计算式中包含了 `weight`，那么必须要使用 `weight` 的 clone，示例代码：
```python
n, d, m = 10, 3, 5
embedding = nn.Embedding(n, d, max_norm=1)
W = torch.randn((m, d), requires_grad=True)
input = torch.tensor([1, 2])

# 这里，a 必须要使用 weight 的 clone，如果不使用 clone，那么 a 的计算
#   必须要放在 b 的计算之后，此时表示另一种含义的 out
a = embedding.weight.clone() @ W.t()
b = embedding(input) @ W.t()    # 这一步，修改了 weight

out = (a.unsqueeze(0) + b.unsqueeze(1))
loss = out.sigmoid().prod()
loss.backward()
```

`forward` 参数：
1. `input`：为 IntTensor 或 LongTensor 类型，为任意 shape，记为 $(\star)$，元素值为 index，注意不是 `one-hot` vector，第 `1` 节中使用 `one-hot` vector 仅仅是为了说明，PyTorch 中使用 `index` （例如 $idx \in [0, V]$，其中 `0` 表示 `padding_idx` ）。

2. `forward` 方法的输出：shape 为 $(\star, d)$， 其中 embedding 维度为 `d`，相当于为最后增加一个维度，维度大小为 `d`。


# 3. from_pretrained 方法
```python
CLASSMETHOD from_pretrained(embeddings, freeze=True, padding_idx=None,
    max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False)
```
根据给定的一个 embeddings 矩阵，创建一个 `Embedding` 实例
参数说明
1. `embeddings`：指定 embeddings 矩阵，shape 为 `(num_embeddings, embedding_dim)`。
2. 其他参数含义与 `Embedding` 构造函数的参数意义相同。

```python
weight = torch.FloatTensor([[1,2,3], [4,5,6]])  # (2, 3)
embedding = nn.Embedding.from_pretrained(weight)

input = torch.LongTensor([1])   # input shape: (1,)
embedding(input)                # output shape: (1, 3)
# tensor([[4.0000, 5.0000, 6.0000]])
```