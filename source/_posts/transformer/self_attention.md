---
title: attention is all you need
date: 2022-01-17 17:22:56
tags: transformer
mathjax: true
---
论文：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)
<!--more-->

# 1. Sequence
处理 Sequence ，自然想到 RNN，如图 1，
![](/images/ml/lstm2.png)
图 1. RNN

输入 Sequence 长度为 `T`，即 $\mathbf x_1, \ldots, \mathbf x_T$，输入为一个 $T \times l$ 的 tensor，其中 $l$ 为向量 $\mathbf x_i$ 的特征长度。

RNN 的缺点是无法并行化计算，因为 `t` 时刻输出依赖 `t-1` 时刻输出，即对于网络 $\mathcal L$，`t+1` 时刻的输出 

$$\begin{aligned}\mathbf h_{t+1}&= \mathcal L(\mathbf x_{t+1}, \mathbf h_t)
\\&=\mathcal L[\mathbf x_{t+1}, \mathcal L(\mathbf x_t, \mathbf h_{t-1})]
\\&=\mathcal L\{\mathbf x_{t+1}, \mathcal L[\mathbf x_t, \mathcal L(\cdots \mathcal L(\mathbf x_1, \mathbf h_0))]\}
\end{aligned}$$

很明显，网络 $\mathcal L$ 是串行计算的。

而 attention 可以解决这个问题，主要思想是：抛弃 RNN 中前一时刻 `t` 整个网络的输出 $\mathbf h_t$ 与下一时刻 `t+1` 的输入 $\mathbf x_{t+1}$ 对齐后作为网络的新的输入（用于计算 $\mathbf h_{t+1}$），而是将所有时刻的输入 $\mathbf x_1, \ldots, \mathbf x_T$ 作为一个 size 为 T 的 batch，送入网络，并行计算，并在网络内部，进行交叉计算（类似于 batch norm），实现依赖性。具体参见下文。



# 2. Self-attention
## 2.1 scaled dot-product attention
记输入序列为 $\mathbf x_1, \ldots, \mathbf x_T$，

1. 每个时刻的 input 乘以一个矩阵 $W \in \mathbb R^{n \times l}$，得到对应的 embedding 向量，$\mathbf a_i=W \mathbf x_i, \ i = 1,2,\ldots, T$。$n$ 为 embedding dimension。（这一步不属于 attention 模块，但是为了完整，这里也写出来了）
2. 使用三个矩阵分别与 embedding 向量相乘，得到 query，key，value 三个向量，这三个向量长度相同，记为 $d$，**<font color='red'>称 $d$ 为 model dimension</font>**，论文中取 $d=512$。
    $$\mathbf q_i=W_q^{\top} \mathbf a_i, \ \mathbf k_i=W_i^{\top} \mathbf a_i, \ \mathbf v_i =W_v^{\top} \mathbf a_i, \quad \mathbf q_i, \mathbf k_i, \mathbf v_i \in \mathbb R^d$$
3. $\mathbf q_i$ 与 $\mathbf k_i$ 做 attention，attention 操作可以实现 time step 之间的依赖性。$\mathbf q_i$ 与 $\mathbf k_i$ 是齐头并进地计算出来的，即，可以并行计算。**scaled dot-product attention：**
    $$\alpha_{ij}=\mathbf q_i \cdot \mathbf k_i / \sqrt d, \quad i,j=1,\ldots, T$$
    向量内积需要乘以因子 $1/\sqrt d$，因为如果 `d` 太大，内积分布的方差就很大，那么执行第 `4` 步的 softmax 之后，会位于 softmax 的梯度较小的区域，影响反向传播。
4. 对 $\alpha_{i1}, \ldots, \alpha_{iT}$ 做 softmax 进行归一化，
    $$\hat {\alpha}_{ij} = \exp(\alpha_{ij})/\sum_l \exp(\alpha_{il})$$
    记矩阵 $A \in \mathbb R^{T \times T}$，表示上述的 attention 矩阵，对每一行做 softmax，
    ```python
    # A is the attention matrix, A_{ij} means attention between
    # query_i and key_j. Both i and j are some time steps.
    torch.softmax(A, dim=1)
    ```
5. 归一化后的 attention 作为 weight，将表征信息的 $\mathbf v_i$ 向量加权求和，得到这一阶段的计算结果（$\mathbf b$ 类似于上文 RNN 中的 $\mathbf h$ ），
    $$\mathbf o_i=\sum_j \hat {\alpha}_{ij} \mathbf v_j, \quad i=1,\ldots, T$$

**上面在每一步中的计算中，所有 time step 都可以同时计算，于是可以实现并行计算。**

<font color='magenta'>$\mathbf o_i \in \mathbb R^d \ , \ i=1,\ldots, T$</font>

## 2.2 矩阵表示
1. 输入矩阵 $X \in \mathbb R^{T \times l}$，每一行表示一个 time step 的输入向量
2. embedding vector 维度为 $n$，参数矩阵 $W \in \mathbb R^{n \times l}$，得到所有 embedding 序列 $I \in \mathbb R^{T \times n}$：$I=X\cdot W^{\top}$
3. 三个参数矩阵 $W_q,W_k,W_v \in \mathbb R^{d\times n}$， 得到 query，key，value 序列 $Q, K, V \in \mathbb R^{T \times d}$，
    $$Q=I W_q^{\top}, \quad K=I W_k^{\top}, \quad V=IW_v^{\top}$$
4. attention op，得到 attention 矩阵 $A \in \mathbb R^{T \times T}$：$A=Q K^{\top}/\sqrt d$
5. 归一化 attention：$\hat A_{ij} =\exp( A_{ij})/ \sum_l \exp(A_{il})$。第 `i` 行 $\hat A_{i,:}$ 作为 time step `i` 的 weights。
6. 输出矩阵 <font color='magenta'>$O \in \mathbb R^{T \times d}$</font>，$O=\hat A V$

## 2.3 Multi-head Self-attention
`2.1` 节的内容可以看作是 single-head self-attention，重复横向堆叠多个相同的 **scaled dot-product attention** 可以得到 multi-head self-attention，具体过程如下：
1. 按 `2.1` 节中得到 $Q, K, V$ 三个矩阵（即输入的 embedding 经三个权重参数线性变换得到 $Q,K,V$ 三个矩阵，这三个权重参数分别为 $W_q,W_k,W_v \in \mathbb R^{d\times n}$）。
2. 记 heads 的数量为 <font color='red'> $h$ </font>，对于第 $i \in [h]$ 个 head，使用三个参数矩阵 $W_i^Q \in \mathbb R^{d \times d_k}, \quad W_i^K \in \mathbb R^{d \times d_k}, \quad W_i^V \in \mathbb R^{d \times d_v}$，分别将 $Q,K,V$ 映射为新的矩阵 <font color='magenta'>$Q_i \in \mathbb R^{T \times d_k}, \quad K_i \in \mathbb R^{T \times d_k}, \quad V_i \in \mathbb R^{T \times d_v}$</font>，注意这里 $\mathbf q_i, \ \mathbf k_i$ 维度相同均为 $d_k$，因为这两个向量需要做内积，

    $$Q_i = Q W_i^Q, \quad K_i = K W_i^K, \quad V_i=V W_i^V, \quad i=1,\cdots, h$$

    由于 $Q_i=QW_i^Q=IW_q^TW_i^Q \Rightarrow Q_i=IW_i^{Q'}$，其中 $I$ 为输入的 embedding。所以也可以认为直接从 输入 embedding（word embedding）直接线性转换为 `query`；对于 `key` 和 `value` 类似处理。
3. 每个 head 单独执行 scaled dot-product attention 即，对每个 head $i=1,\cdots,h$
    $$A_i=Q_i K_i^{\top} / \sqrt {d_k} \in \mathbb R^{T \times T} \\ \hat A_i=\text{softmax} (A_i) \\ O_i =\hat A_i V_i \in \mathbb R^{T \times d_v}$$
4. 将每个 head 的输出沿着 `axis=1` 方向 concatenate（类似于`torch.hstack`），再乘以个输出参数矩阵 $\color{magenta} W^O \in \mathbb R^{hd_v \times d}$，
    $$O=\text {Concat}(O_1,\cdots, O_h) \in \mathbb R^{T \times hd_v} \\ O:= O W^O \in \color{magenta} \mathbb R^{T \times d}$$

![](/images/transformer/self_attention_1.png)
图 2. 左：scaled dot-product attention; 右：multi-head self-attention

通常取 $d=512, \ h=8, \ d_k=d_v=d/h=64$。

__总结：__

$\text{Transformer}:\mathbb R^{T \times n} \rightarrow \mathbb R^{T \times d}$，其中 $n$ 是 embedding 维度，$d$ 是隐层维度。


**示例代码**

实际操作中，可以将这 `h` 个 head 中的参数 concatenate 起来，然后一起执行矩阵操作，

$$W^Q=\begin{bmatrix} W_1^Q & \cdots & W_h^Q \end{bmatrix} \ \in \mathbb R^{d \times d}$$
$$Q' = QW^Q=\begin{bmatrix}QW_1^Q & \cdots & QW_h^h \end{bmatrix} \ \in \mathbb R^{T \times d}$$

注：embedding dimension 与 model dimension 相同，即 $n=d$。
```python
# hidden_dim is `d`, fc_q is W^Q
fc_q = nn.Linear(hidden_dim, hidden_dim)
fc_k = nn.Linear(hidden_dim, hidden_dim)
fc_v = nn.Linear(hidden_dim, hidden_dim)
fc_o = nn.Linear(hidden_dim, hidden_dim)
# given `query` , `key`, `value`, which are 3 linear transformed results for embedding, respectively
# query, key, value: (batch_size, seq_len, d)
Q = fc_q(query)         # Q': (batch_size, seq_len, d)
K = fc_k(key)           # K'
V = fc_v(value)         # V'
```

然后是每个 head 单独执行 scaled dot-product attention，这里必须各个 head 分开执行，因为每个 head 的 attention 维度为 $d_k$ 而不是 $d$，如果是 single head，那么就不需要分开，但是 multi head，必须要分开，
```python
scale = torch.sqrt(torch.FloatTensor([num_heads]))
# hidden_dim = num_heads * head_dim
# after permuting dimensions, shape is (batch_size, num_heads, seq_len, head_dim)
Q = Q.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
K = K.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
V = V.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)


# to do attention, we must make the lowest two dimensions be (seq_len, attention_dim)
# A_i = Q_i K_i^{\top}
# transpose the lowest two dimensions of K
# A's shape: (batch_size, num_heads, seq_len, seq_len)
A = torch.matmul(Q, K.permute(0, 1, 3, 2))/scale      
```
对 attention tensor 归一化，然后执行 dropout 以增强泛化能力，接着与 value 相乘，所有 heads 的结果 concatenate，最后经过一个输出层的线性变换，得到 multi-head self-attention layer 的输出，
```python
A = torch.softmax(A, dim=-1)
O = torch.matmul(self.dropout(A), V)    # (batch_size, num_heads, seq_len, head_dim)
O = O.permute(0, 2, 1, 3).contiguous()  # (batch_size, seq_len, num_heads, head_dim)
O = O.view(batch_size, -1, hidden_dim)  # concatenate all heads

# O: (batch_size, seq_len, hidden_dim)
O = fc_o(O)                             # refer to fig 2.
```

注：
1. 代码注释中使用 `seq_len`，表示一个 mini-batch 中最长 sentence 中 word 数量。上文 $T$ 表示单个 sentence 中 word 数量。代码实现中，不使用单个 sentence，而是一个 mini-batch，故对于 short sentence，需要在最后进行 padding，使得长度一致，记为 `seq_len`。
2. 输出 $O$ 与 `query` ,`key`,`value` 等具有相同的 shape。

# 3. Transformer
![](/images/transformer/self_attention_2.png)
图 3. Transformer 模型结构

## 3.1 Encoder
如图 3，左边是 Encoder，$N=6$，即串联 $N$ 个相同结构的 block，
1. 每个 block 包含两个 layer：multi-head attention 和全连接前馈网络，这两个 layer 有 residual 连接，后跟一个 layer normalization（Layer Norm 对每个样本进行归一化，即在 `(C,H,W)` 上做归一化，每个样本独立进行）。
2. 全连接前馈网络由两个全连接层组成
2. 由于存在 residual 连接，channel dimension 全部取 $d=512$，即，input embedding vector 的维度，multi-head attention 的输出 vector 维度，全连接层的输出维度，全部为 $d=512$。
3. Encoder 的输出可用矩阵 $O_e \in \mathbb R^{T \times d}$ 表示。

## 3.2 Decoder
解码器的主体结构是 $N=6$ 的相同 block 的串联，
1. Decoder 无法并行计算，因为其 `t+1` 时刻的输入是其 `t` 时刻的输出
2. 每个 block 包含两个 multi-head attention，以及一个全连接前馈网络，这三个 layer 均有 residual 连接。
3. 第二个 multi-head attention 的 `key` 和 `value` 为 encoder 的输出，均位于向量空间 $\mathbb R^{T_1 \times d}$，且相同，而 `query` 为第一个 multi-head attention 的输出，位于向量空间 $\mathbb R^{T_2 \times d}$，此时 `query` 与 `key` 的 attention 矩阵为 $A = QK^{\top} \in \mathbb R^{T_2 \times T_1}$，这可能不是一个方阵，但是没关系，不影响对 `value` 的加权求和，$\hat AV \in \mathbb R^{T_2 \times d}$。

4. 虽然 Encoder 和 Decoder 中的 block 数量均为 $N=6$，但是，**<font color='red'>使用 Encoder 的最终输出（即最后一个 block 的输出）作为 Decoder 中每个 block 中第二个 multi-head attention 的 `key` 和 `value`，而不是 Encoder 中的各个 block 的输出分别作为 Decoder 中各 block 的 `key` 和 `value`。</font>**
    ```python
    # trg: (batch_size, trg_len)
    # enc_src: output of encoder. has a shape of (batch_size, src_len, hidden_dim)
    N = 6
    layers = nn.ModuleList([DecoderLayer(hidden_dim, num_heads,...) for _ in range(N)])
    for layer in layers:    # all block use the same output of encoder
        trg = layer(trg, enc_src, ...)
    ```

Decoder 在训练和测试阶段，稍有不同。
### 3.2.1 Decoder 训练
**<font color=#FF88>使用 Teacher Forcing 且是并行化训练</font>** 。

例如将“我有一只猫”翻译为 “I have a cat”，对于 src sentence 和 trg sentence，都要 prepend  `<sos_tok>` 和 append `<eos_tok>`，那么 trg sentence 变为 “`<sos_tok>` I have a cat `<eos_tok>`”，于是 Decoder 输入应为 (`<sos_tok>` I have a cat)，输出应为 （I have a cat `<eos_tok>`），这里输入 sequence 应该去掉最后一个 token，因为我们设计的 Decoder 是根据输入一个词，输出下一个词，而输出 sequence 应该去掉第一个 token。

使用 sequence 输入输出，实现 Decoder 并行化计算，
```sh
# target sequence: <sos> I have a cat <eos> <pad> <pad>
# （填充了两个 <pad> token）

        <sos>  I  have  a  cat    <eos>  <pad>
          |    |   |    |   |       |      |
    +-------------------------------------------+
    |                   Decoder                 |
    +-------------------------------------------+
          |    |   |    |   |       |      |
          I  have  a   cat <eos>   <pad> <pad>
```

但是输出是一个一个出来的，而 attention 是对整个 sequence 全局进行的。实际的顺序应该是下面 `3.2.2` 节中说明的那样，这里一次性给出 sequence 整体仅仅是为了并行化训练，所以对于第 `i` 个 token，它不能看到之后的 token，也没法跟之后的 token 做 attention，无法进行 $\mathbf q_i \mathbf k_j / \sqrt {d_k}, \ j > i$ 这样的 attention，故需要对 attention 得到的矩阵（上面的矩阵 $A$ ）进行 mask 操作，如图 2 左边部分中的 `Mask (opt.)`，在 softmax 之前执行 Mask 操作。

回顾前面内容，attention 矩阵表示一个 sentence 内各 token 之间的 attention，矩阵中第 `i` 行表示第 `i` 个 token 与所有 token 之间的 attention，那么第 `1` 个 token 仅与自身有 attention，第 `2` 个 token 与前 `2` 个 token 有 attention，即这个矩阵应该是下三角矩阵（左下方有非 0 值），即，**使用左下方全 1 的下三角矩阵实现 mask 操作**。

顺便一提，mini-batch 内部分 sentences 在末尾进行了 padding，显然前面的 token 也不应该与这些 padding token 做 attention，应该这些 padding token 本不存在，仅仅是因为 tensor 需要才进行补全。

由于 attention 是作为 $\mathbf v_i$ 的权重，不存在的 attention 其权重应该为 $0$，使得相应的 $\mathbf v_i$ 贡献为 $0$。

```python
src_pad_idx = 0
trg_pad_idx = 0
def make_src_mask(src):
    # src: (batch_size, seq_len), src contains all indices of tokens
    src_mask = (src != src_pad_idx).unsqueeze(1).unsqueeze(2)
    # src_mask: (batch_size, 1, 1, seq_len)
    # after dimension broadcasting, the right-bottom sub-matrix of src_mask is 0
    return src_mask

def make_trg_mask(trg):
    # trg: (batch_size, seq_len), where `seq_len` may not be equal to that in src
    trg_pad_mask = (trg != trg_pad_idx).unsqueeze(1).unsqueeze(2)
    # trg_pad_mask: (batch_size, 1, 1, seq_len)
    seq_len = trg.shape[1]
    trg_sub_mask = torch.tril(torch.ones((seq_len, seq_len)), diagonal=0).bool()
    trg_mask = trg_pad_mask & trg_sub_mask
    # trg_mask: (batch_size, 1, seq_len, seq_len)
    return trg_mask
```

如图 2 左侧，对 scale 之后的结果（ $Q_i K_i^{\top} / \sqrt {d_k}$）进行 mask，
```python
# A = torch.matmul(Q, K.permute(0, 1, 3, 2))/scale   # refer to the related code above in section 2.3
# A's shape: (batch_size, num_heads, seq_len, seq_len)
A = A.masked_fill(mask==0, -1e10)   # after softmax, -1e10 is enough to approach 0
# A = torch.softmax(A, dim=-1)
```

从 Decoder 的输入输出 tensor shape 来分析一波：

从 target dataset 获得的 mini-batch，记为变量 `trg`， 具有 shape： `(batch_size, trg_len+1)`，Decoder 的输入为 `trg[:,:-1]`，Decoder 网络的 gt sequence 为 `trg[:,1:]`，这两个变量的 shape 均为 `(batch_size, trg_len)`，

其中输入经过 embedding，shape 变为 `(batch_size, trg_len, hidden_dim)`，注意 `embedding_dim=hidden_dim`，然后经过 Decoder 的 `Nx` 个 block 后，shape 保持不变，最后经过一个全连接层（fc）和 Softmax 层，其中 fc 线性变换后 tensor shape 变为
`(batch_size, trg_len, out_dim)`，这里 `out_dim` 为 target 词汇表大小（包括了 pad_tok，sos_tok, eos_tok），fc 的输出就是 tokens 的非归一化得分，而 Decoder 的 gt shape 为 `(batch_size, trg_len)`，很显然，使用 PyTorch 中的 `CrossEntropyLoss` 可计算损失（这个类内部包含了 softmax，所以预测为非归一化得分）。


**目标函数（loss）**

```python
# suppose trg is the torch.tensor that stores indices of all tokens in a mini-batch
#   after padding and aligning
# trg: (batch_size, trg_len+1). trg_len is just seq_len, but in order to distinguish 
#   with that for src, use trg_len instead and use src_len to represent 
#   seq_len of src
criterion = nn.CrossEntropyLoss(ignore_idx=TRG_PAD_IDX)
model = Seq2Seq(...)
output = model(src, trg[:,:-1])     # （batch_size, trg_len, out_dim)
output = output.contiguous().view(-1, out_dim)
trg = trg[:,1:].contiguous().view(-1)
loss = criterion(output, trg)
```

注意上面代码片段中，指明 `ignore_idx=TRG_PAD_IDX`。以前面举的例子来说明，
```
target sequence: <sos> I have a cat <eos> <pad> <pad>
填充了两个 <pad> token）

gt: I have a cat <eos> <pad> <pad>
```
显然，计算交叉熵损失时，应该计算 `I have a cat <eos>` 这些 token 的损失，而后面填补的两个 `<pad>` token 则不应该包含在损失计算中。因为这两个 `<pad>` token 仅仅是为了填充使得数据对齐从而可以存储在 tensor 中，实际上它们本不应该存在，损失则无从说起，毕竟前面已经有 `<eos>` 了，后面的 token 可以随便预测，都算对，因为从 `<eos>` 往后都被截断了，对与不对都不重要，故不应该作为惩罚性添加到 loss 中。这类似于目标检测中，只对正例 region 计算坐标回归损失，而负例 region 则不需要计算坐标回归损失。


 ### 3.2.2 Decoder 测试
 根据上一小节的分析，Decoder 输入和输出（这里的输出指经过了 argmax 之后的值）具有相同的 shape：`(batch_size, trg_len)`，简单起见，令 `batch_size=1`，于是
 Decoder 实际的顺序应该是：
 1. 输入 `[[<sos>]]`，输出 `[[I]]`
 2. 输入 `[[<sos>, I]]`，输出 `[[I, have]]`
 3. 输入 `[[<sos>, I, have]]`，输出 `[[I, have, a]]`
 4. 输入 `[[<sos>, I, have, a]]`，输出 `[[I, have, a, cat]]`
 5. 输入 `[[<sos>, I, have, a, cat]]`，输出 `[[I, have, a, cat, <eos>]]`，结束。
 ```python
 max_len = 50 # 根据经验手动设置一个值，使得 sentence 长度不超过 `max_len-2`
 model = Seq2seq(...)
# enc_src: output of encoder
trg_idxs = [TRG.vocab.stoi[TRG.init_token]]
for i in range(max_len):
    trg = torch.LongTensor(trg_idxs).unsqueeze(0)
    trg_mask = make_trg_mask(trg)
    with torch.no_grad():
        output = model.decoder(trg, enc_src, trg_mask, src_mask)
    pred_idx = output.argmax(dim=2)[:,-1].item()
    trg_idxs.append(pred_idx)
    if pred_token == TRG.vocab.stoi[TRG.eos_token]:
        break
# pred_tokens contains <sos_tok> and <eos_tok >
pred_tokens = [TRG.vocab.itos[i] for i in trg_idxs]
 ```



## 3.3 前馈网络
Encoder 和 Decoder 的 block 中除了 attention 模块，还有前馈网络（FFN），这个 FFN 由两个全连接层组成，两个全连接层中间有一个 ReLU，
$$FFN(x)=\max(0, xW_1+b_1)W_2 + b_2$$


## 3.4 Position Encoding
通过前面对 attention 的介绍可知，各 time step 的输入其实是位置无关的，因为每个 time step 输入的 attention 操作都是全局进行的，即 `i` 位置的输入 $\mathbf x_i$，其 attention 记为 $\mathbf o_i$，如果换到 `j` 位置，其 attention 结果记为 $\mathbf o_j$，显然有 $\mathbf o_i = \mathbf o_j$。例如 “A打B” 和 “B打A”，前者 A 是打人，后者 A 是被打，但是 attention 输出却一样，所以不合理。考虑位置信息后，就可以解决这个问题。

使用 one-hot vector 来表示位置信息，例如第 `i` time step 输入 $\mathbf x_i$，其位置信息的 one-hot vector 为 $\mathbf p_i = [\underbrace{0,\cdots, 0}_{i-1}, 1, \underbrace {0, \cdots, 0}_{L-i}]$，其中 $L$ 是 max sequence length，即数据集（或一个 minibatch 中）所有 sentences 中最长的 sentence 长度（words 数量），或者根据具体任务和经验手动设置一个较大的数，数据集中长度大于 $L$ 的 sentence 都会被截断使得长度为 $L$，例如 $L=100$。于是叠加位置信息后的最终的 embedding 为

$$\begin{bmatrix}W^I & W^P\end{bmatrix}\begin{bmatrix}\mathbf x_i \\ \mathbf p_i\end{bmatrix}=\mathbf o_i + \mathbf e_i$$

即，输入的 embedding 与位置信息的 embedding 相加。论文中提到，对于 $\mathbf o_i$ 需要进行 scale，相当于对这两种 embedding 赋予不同的权重，$\lambda \cdot\mathbf o_i + \mathbf e_i$，通常取 $\lambda = \sqrt d$。

$W^I$ 就是上面所说的 embedding 矩阵，可以训练得到（参见 [embedding](/2022/01/19/pytorch/embedding) ）。 $W^P \in \mathbb R^{n \times L}$ 则表示位置信息的 embedding 矩阵，$n$ 为 embedding dimension。$W^P$ 可以与 $W^I$ 一样训练得到，也可以使用公式计算得到。

训练得到 embedding 矩阵
```python
hidden_dim = 512
L = 100
# hidden_dim is usually equal to model dimension `d`
tok_embedding = nn.Embedding(input_dim, hidden_dim)   # (l, d)
pos_embedding = nn.Embedding(L, hidden_dim)
# scale the token embedding, i.e., a balance factor between
#   token embedding and position embedding
scale = torch.sqrt(torch.FloatTensor([hidden_dim]))

# src: (batch_size, seq_len)
#   where seq_len is the max sequence length of sentences in this batch
batch_size = 8
seq_len = 10
V = 100     # 1-99 is indices of words, 0 is the index of padding
src = torch.empty((batch_size, seq_len))
for i in range(batch_size):
    l = random.randint(5, seq_len)
    s = torch.randint(1, 100, (random.randint(5, seq_len)))
    c = torch.zeros((seq_len-l), dtype=torch.int)
    src[i,:] = torch.hstack((s, c))

# pos: (batch_size, seq_len)
pos = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1)

tok_embedding(src)*scale + pos_embedding(pos)
```

公式计算 position embedding
$$PE(pos, 2i)=\sin (pos/10000^{2i/d})
\\PE(pos, 2i+1)=\cos(pos/10000^{2i/d})$$
其中 $pos$ 表示 sequence 中 word 的位置，范围为 `[0,seq_len-1]`，$2i$ 和 $2i+1$ 表示 position embedding vector 中的 index，由于 position embedding 维度为 $d$，故
$i \le \lfloor d/2 \rfloor$
```python
d = 512                             # model dimension
L = 100                             # seq_len <= L
a = torch.arange(0, L)              # [1~L]
a = a / 10000                       # [1/10000 ~ L/10000]
a = a.unsqueeze(-1).repeat(1, d//2) # (L, d//2)

e = torch.arange(0, d//2) * 2 / d   # (d//2,)  [0/d,2/d,...]
e = e.unsqueeze(0).repeat(L, 1)   # (L, d//2)
a = torch.pow(a, e)
s = torch.sin(a)
c = torch.cos(a)
PE = torch.empty(L, d)
PE[:,::2] = s
PE[:,1:d:2] = c

# get the position embedding for current mini-batch
pos = torch.arange(0, seq_len)
pe = PE[pos]    # (seq_len, d)
pe = pe.unsqueeze(0).repeat(batch_size, 1, 1)
tok_embedding(src)*scale + pe
```
# ref
1. [Vision Transformer 超详细解读](https://zhuanlan.zhihu.com/p/340149804)